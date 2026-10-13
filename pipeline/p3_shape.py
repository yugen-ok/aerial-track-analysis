"""Shape per frame, and substitution of the frames whose carve collapsed.

Three frames are low-contrast interpolation ghosts; their contour degenerates to
a speck. They are replaced by the mean shape at the mean area, flagged, and
excluded from every statistic.

in : results/masks/*.npz, results/positions.csv
out: results/shape.csv, results/masks/mean_shape.npz, orange figures for the three
"""
import csv, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_rgb
from src.shape import mask_stats, mean_shape, place
from src.render import with_outline, with_circle, zoom

SUB_COLOR = (255, 160, 0)
UNDERSIZE = 0.5          # fraction of the median area below which a carve is rejected

def load_masks():
    out = {}
    for f in range(C.FIRST, C.LAST + 1):
        z = np.load(C.MASKS / f"mask_{f:05d}.npz")
        out[f] = (z["mask"], int(z["x0"]), int(z["y0"]))
    return out

def main():
    pos = {int(r["frame"]): r for r in csv.DictReader(open(C.RESULTS / "positions.csv"))}
    M = load_masks()
    frames = sorted(M)
    area = np.array([M[f][0].sum() for f in frames], float)
    under = area < UNDERSIZE * np.median(area)
    print("median area %.0f; undersized: %s" % (np.median(area),
          [frames[i] for i in range(len(frames)) if under[i]]))

    good = [M[f][0] for i, f in enumerate(frames) if not under[i]]
    target = int(round(np.mean([m.sum() for m in good])))
    shape = mean_shape(good, target)
    np.savez_compressed(C.MASKS / "mean_shape.npz", mask=shape, area=target)
    print("mean shape rescaled to %d px (target %d)" % (shape.sum(), target))

    rows = []
    for i, f in enumerate(frames):
        m, x0, y0 = M[f]
        if under[i]:
            s0 = mask_stats(m)
            m = place(shape, m, s0["cx"], s0["cy"])
            np.savez_compressed(C.MASKS / f"mask_{f:05d}.npz", mask=m, x0=x0, y0=y0,
                                substituted=True)
            rgb = load_rgb(C.FRAMES / f"frame_{f:05d}.png").astype(np.uint8)
            img = with_outline(rgb, m, (x0, y0), SUB_COLOR)
            cx, cy, r = float(pos[f]["cx"]), float(pos[f]["cy"]), float(pos[f]["r"])
            img = with_circle(img, cx, cy, r, width=1)
            zoom(img, cx, cy, int(r * C.ZOOM), label=f"frame {f} (size substituted)",
                 label_color=SUB_COLOR).save(C.FIGURES / "contours" / f"frame_{f:05d}.png")
        st = mask_stats(m, origin=(x0, y0))
        st.update(frame=f, substituted=int(under[i]))
        rows.append(st)
        print(f, st["area"], round(st["major"], 1), round(st["minor"], 1),
              round(st["angle"], 1), "SUB" if under[i] else "")
    cols = ["frame", "area", "major", "minor", "elong", "angle", "cx", "cy",
            "bbox_w", "bbox_h", "substituted"]
    with open(C.RESULTS / "shape.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

if __name__ == "__main__":
    main()
