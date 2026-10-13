"""Locate the object in every frame by matching the hand-drawn cutouts.

in : data/frames/*.png, data/cutouts/cutouts.png
out: results/positions.csv, results/figures/marked/
"""
import csv, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray, load_rgb
from src.templates import load_cutouts
from src.locate import best_match, refine_centroid, radius_for
from src.render import with_circle
from PIL import Image

def main():
    cuts = load_cutouts(C.CUTOUTS)
    out = C.FIGURES / "marked"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in range(C.FIRST, C.LAST + 1):
        p = C.FRAMES / f"frame_{f:05d}.png"
        gray = load_gray(p)
        prev = (rows[-1]["cx"], rows[-1]["cy"]) if rows else None
        vel = (0.0, 0.0)
        if len(rows) >= 3:
            vel = ((rows[-1]["cx"] - rows[-3]["cx"]) / 2.0,
                   (rows[-1]["cy"] - rows[-3]["cy"]) / 2.0)
        score, k, cx, cy = best_match(gray, cuts, C.SKY_BAND, prev, vel, C.GATE_RADIUS)
        r = radius_for(cuts[k])
        cx, cy = refine_centroid(gray, cx, cy, r)
        rows.append(dict(frame=f, cx=round(cx, 1), cy=round(cy, 1), r=round(r, 1),
                         cutout=k + 1, ncc=round(score, 3)))
        Image.fromarray(with_circle(load_rgb(p), cx, cy, r).astype(np.uint8)).save(
            out / f"frame_{f:05d}.png")
        print(rows[-1])
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS / "positions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
