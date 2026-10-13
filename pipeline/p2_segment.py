"""Trace the object's boundary in each frame: the colour-jump contour.

Walks inward from the sky ring; a pixel is enclosed when every path to it from
the sky had to cross a jump bigger than the threshold.

in : results/positions.csv, data/frames/*.png
out: results/masks/*.npz, results/contours[-loose].csv, results/figures/contours[-loose]/
usage: python pipeline/p2_segment.py [loose]
"""
import csv, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_rgb, blur, rgb_to_lab
from src.segment import minimax_cost, stable_threshold, noise_threshold, carve
from src.render import with_outline, with_circle, zoom

OBJ_COLOR = (0, 255, 255)

def prepare(frame, cx, cy, r):
    """Patch, Lab, the three regions and the cost field - everything geometry-only."""
    rgb = load_rgb(C.FRAMES / f"frame_{frame:05d}.png")
    H, W = rgb.shape[:2]
    R = r * C.GROW
    x0, y0 = int(max(0, cx - R - 2)), int(max(0, cy - R - 2))
    x1, y1 = int(min(W, cx + R + 2)), int(min(H, cy + R + 2))
    lab = rgb_to_lab(blur(rgb[y0:y1, x0:x1], (1.0, 1.0, 0)))
    yy, xx = np.mgrid[0:lab.shape[0], 0:lab.shape[1]]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    disc = d <= R
    seed = disc & (d >= R - 2.0)
    cost = minimax_cost(lab, disc, seed)
    sky = np.median(lab[seed].reshape(-1, 3), axis=0)
    delta = np.linalg.norm(lab - sky, axis=-1)
    ring = disc & (d >= r * 1.30) & (d < R - 2.0)
    return dict(rgb=rgb, lab=lab, d=d, disc=disc, seed=seed, ring=ring, cost=cost,
                sky=sky, delta=delta, x0=x0, y0=y0, cx=cx, cy=cy, r=r)

def segment(S, loose):
    """Threshold + carve, lifting the inner limit if the object reaches it."""
    keep = C.KEEP
    while True:
        inner = S["d"] <= S["r"] * keep
        if loose:
            T, stab = noise_threshold(S["cost"], S["ring"], C.LOOSE_SENS)
        else:
            T, stab = stable_threshold(S["cost"], inner)
        obj = carve(S["cost"], S["delta"], inner, T,
                    (S["cx"] - S["x0"], S["cy"] - S["y0"]), loose)
        rim = inner & (S["d"] >= S["r"] * keep - 1.5)
        if not (obj & rim).any() or keep >= C.GROW - 0.1:
            return obj, inner, T, stab, keep
        keep += C.KEEP_STEP

def main(loose=False):
    tag = "-loose" if loose else ""
    figs = C.FIGURES / f"contours{tag}"; figs.mkdir(parents=True, exist_ok=True)
    C.MASKS.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in csv.DictReader(open(C.RESULTS / "positions.csv")):
        f = int(rec["frame"]); cx, cy, r = float(rec["cx"]), float(rec["cy"]), float(rec["r"])
        S = prepare(f, cx, cy, r)
        obj, inner, T, stab, keep = segment(S, loose)
        if not loose:
            np.savez_compressed(C.MASKS / f"mask_{f:05d}.npz",
                                mask=obj, x0=S["x0"], y0=S["y0"])
        img = with_outline(S["rgb"].astype(np.uint8), obj, (S["x0"], S["y0"]), OBJ_COLOR)
        img = with_circle(img, cx, cy, r, width=1)
        zoom(img, cx, cy, int(r * C.ZOOM), label=f"frame {f}").save(figs / f"frame_{f:05d}.png")
        rows.append(dict(frame=f, threshold_dE=round(T, 3), stability=round(stab, 4),
                         keep=round(keep, 2), object_px=int(obj.sum()),
                         region_px=int(inner.sum()),
                         frac=round(float(obj.sum() / inner.sum()), 3),
                         mean_dE_from_sky=round(float(S["delta"][obj].mean()), 2) if obj.any() else 0.0,
                         sky_L=round(float(S["sky"][0]), 2)))
        print(rows[-1]["frame"], rows[-1]["object_px"], rows[-1]["keep"], rows[-1]["mean_dE_from_sky"])
    with open(C.RESULTS / f"contours{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main(loose="loose" in sys.argv)
