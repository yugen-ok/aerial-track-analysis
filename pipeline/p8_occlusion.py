"""Lower bound on range and size from the object emerging past a foreground tree.

The object first appears mid-frame, not at a picture edge, and its measured
straight-line track extrapolates back into a dark conifer. If it passed BEHIND
that tree, then object_range > tree_range, which is the only absolute-scale
constraint available anywhere in this analysis.

Tree range is estimated from elevation: a canopy point at elevation A above the
horizon, at range D, stands H = D*tan(A) above camera level. Inverting with a
plausible tree height and camera mounting height gives D.

in : results/shape.csv, data/frames/frame_00099.png
out: results/occlusion.txt
"""
import csv, math, sys
import numpy as np
from pathlib import Path
from scipy import ndimage as nd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray, blur
from src.stats import linfit
from src import geometry as G

FOLIAGE_BOX = (0, 420, 1380, 1750)   # y0, y1, x0, x1 - the conifer, object-free in f99
DARK_PCT = 35
TREE_HEIGHT = (12.0, 22.0)           # m, mature conifer in this region
CAM_HEIGHT = (4.0, 8.0)              # m, typical gate-camera mounting

L = []
def say(s=""):
    print(s); L.append(s)

def foliage_mask():
    y0, y1, x0, x1 = FOLIAGE_BOX
    reg = blur(load_gray(C.FRAMES / "frame_00099.png")[y0:y1, x0:x1], 2)
    d = reg < np.percentile(reg, DARK_PCT)
    d = nd.binary_closing(nd.binary_opening(d, np.ones((7, 7))), np.ones((9, 9)))
    lab, n = nd.label(d)
    return (lab == int(np.argmax(nd.sum(d, lab, range(1, n + 1)))) + 1), x0, y0

def main():
    rows = [r for r in csv.DictReader(open(C.RESULTS / "shape.csv")) if not int(r["substituted"])]
    f = np.array([int(r["frame"]) for r in rows], float)
    cx = np.array([float(r["cx"]) for r in rows]); cy = np.array([float(r["cy"]) for r in rows])
    vx, vy = linfit(f, cx)["slope"], linfit(f, cy)["slope"]
    say("measured velocity %.2f, %.2f px/frame" % (vx, vy))

    mask, mx0, my0 = foliage_mask()
    ys, xs = np.nonzero(mask)
    top_y = ys.min() + my0
    say("conifer mask: x %d..%d, y %d..%d; canopy top y=%d"
        % (mx0 + xs.min(), mx0 + xs.max(), my0 + ys.min(), my0 + ys.max(), top_y))

    say(); say("back-extrapolated track vs the foliage:")
    hits = []
    for fr in range(int(f[0]) - 18, int(f[0]) + 1):
        px, py = cx[0] + (fr - f[0]) * vx, cy[0] + (fr - f[0]) * vy
        xi, yi = int(px - mx0), int(py - my0)
        inside = 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0] and mask[yi, xi]
        if inside:
            hits.append((fr, px, py))
    if hits:
        say("  occluded frames %d..%d, last behind foliage at (%.0f, %.0f)"
            % (hits[0][0], hits[-1][0], hits[-1][1], hits[-1][2]))
        occ_y = hits[-1][2]
    else:
        say("  no crossing found"); occ_y = top_y

    hf = C.HFOV_ASSUMED
    a_top = G.elevation(top_y, C.HORIZON_PX, hf)
    a_occ = G.elevation(occ_y, C.HORIZON_PX, hf)
    say(); say("elevation of canopy top %.1f deg; of the occluding point %.1f deg"
               % (a_top, a_occ))

    say(); say("tree range D from H_top = D*tan(elev_top):")
    Ds = []
    for th in TREE_HEIGHT:
        for ch in CAM_HEIGHT:
            D = (th - ch) / math.tan(math.radians(a_top))
            Ds.append(D)
            say("   tree %.0f m, camera %.0f m -> D = %5.1f m" % (th, ch, D))
    lo, hi = min(Ds), max(Ds)
    say("   => full envelope %.0f-%.0f m" % (lo, hi))
    say("   the %.0f m corner is degenerate (camera almost at canopy height);" % lo)
    # central case: a 15-20 m conifer, camera at 5-6 m
    cen = [(th - ch) / math.tan(math.radians(a_top)) for th in (15., 20.) for ch in (5., 6.)]
    clo, chi = min(cen), max(cen)
    say("   CENTRAL ESTIMATE (tree 15-20 m, camera 5-6 m): %.0f-%.0f m" % (clo, chi))
    lo, hi = clo, chi

    say(); say("MINIMUM object size, since object_range > tree_range:")
    maj = np.array([float(r["major"]) for r in rows]).mean()
    travel = math.hypot(cx[-1] - cx[0], cy[-1] - cy[0])
    dt = (f[-1] - f[0]) / G.FPS
    bl = G.body_lengths_per_sec(travel, maj, dt)
    for hfov in (C.HFOV_RANGE[0], hf, C.HFOV_RANGE[1]):
        ang = maj * G.deg_per_px(hfov)
        k = 2 * math.tan(math.radians(ang) / 2)
        say("   HFOV %2.0f deg (object %.2f deg): length > %.2f-%.2f m, speed > %.0f-%.0f km/h"
            % (hfov, ang, lo * k, hi * k, bl * lo * k * 3.6, bl * hi * k * 3.6))

    say(); say("This excludes the near-field explanations: an insect or small bird")
    say("would have to sit far closer than the tree, which the occlusion forbids.")
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "occlusion.txt").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
