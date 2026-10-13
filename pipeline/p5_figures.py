"""Composite track figure, de-rotated stack, and the cut clip.

in : results/shape.csv, results/masks/*.npz, data/frames/*.png

out: results/figures/composite.png, stack.png, clip.mp4
"""
import csv, math, sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_rgb, load_gray
from src.render import path_overlay, with_outline
from src.stats import linfit
from src.video import encode_clip
from scipy import ndimage as nd

def main():
    rows = list(csv.DictReader(open(C.RESULTS / "shape.csv")))
    base = load_rgb(C.FRAMES / "frame_00099.png").astype(np.uint8).copy()
    cents = []
    for i, r in enumerate(rows):
        f = int(r["frame"])
        z = np.load(C.MASKS / f"mask_{f:05d}.npz")
        m, x0, y0 = z["mask"], int(z["x0"]), int(z["y0"])
        cents.append((float(r["cx"]), float(r["cy"])))
        tcol = i / (len(rows) - 1)
        col = (int(40 + 215 * tcol), int(255 - 180 * tcol), int(255 - 120 * tcol))
        base = with_outline(base, m, (x0, y0), col)
    use = [(float(r["cx"]), float(r["cy"])) for r in rows if not int(r["substituted"])]
    X = np.array([p[0] for p in use]); Y = np.array([p[1] for p in use])
    fit = linfit(X, Y)
    xa, xb = X.max() + 40, X.min() - 40
    line = [(xa, fit["slope"] * xa + fit["intercept"]), (xb, fit["slope"] * xb + fit["intercept"])]
    im = path_overlay(base, cents, line)
    d = ImageDraw.Draw(im)
    d.line([(xa, line[0][1]), (xb, line[0][1])], fill=(120, 200, 255), width=2)
    d.text((xb + 6, line[0][1] + 6), "horizontal", fill=(120, 200, 255))
    d.text((xa - 250, line[0][1] - 34),
           "path %.0f deg above horizontal" % math.degrees(math.atan(-fit["slope"])), fill=(255, 255, 0))
    im.save(C.FIGURES / "composite.png")
    im.crop((780, 0, 1400, 250)).resize((620 * 3, 250 * 3), Image.LANCZOS).save(
        C.FIGURES / "composite_zoom.png")

    S = 160; acc = np.zeros((S, S)); k = 0
    for r in rows:
        if int(r["substituted"]):
            continue
        f = int(r["frame"])
        g = load_gray(C.FRAMES / f"frame_{f:05d}.png")
        cy, cx = float(r["cy"]), float(r["cx"])
        pat = g[int(cy) - S // 2:int(cy) + S // 2, int(cx) - S // 2:int(cx) + S // 2]
        if pat.shape != (S, S):
            continue
        pat = pat - np.median(pat)
        acc += nd.rotate(pat / (pat.max() + 1e-6), float(r["angle"]), reshape=False, order=1)
        k += 1
    a = acc / k; a = (a - a.min()) / (a.max() - a.min())
    Image.fromarray((a * 255).astype(np.uint8)).resize((S * 5, S * 5), Image.LANCZOS).save(
        C.FIGURES / "stack.png")
    print("stacked", k, "frames")

    encode_clip(C.FRAMES, C.FIRST - C.PRE_ROLL, C.LAST - C.FIRST + 1 + C.PRE_ROLL,
                C.FIGURES / "clip.mp4")

if __name__ == "__main__":
    main()
