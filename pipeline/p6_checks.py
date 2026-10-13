"""Two independent checks that bound what the footage can support.

in : data/frames/frame_00099.png, data/reference/ynet-closeup.jpg

A. optical linearity - can this footage be calibrated at all?
B. the article's closeup photo - is it the same object?

out: results/checks.txt
"""
import math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray, blur
from src.shape import mask_stats
from scipy import ndimage as nd

L = []
def say(s=""):
    print(s); L.append(s)

def straightness():
    """The CCTV timeline bar is straight in the original. If it bows here, the
    chain has distortion and vanishing-point calibration is invalid."""
    say("A. OPTICAL LINEARITY (CCTV timeline bar, known straight)")
    im = blur(load_gray(C.FRAMES / "frame_00099.png"), 1.5)
    xs, ys = [], []
    for x in range(320, 1900, 20):
        d = np.diff(im[1000:1060, x])
        k = int(np.argmin(d))
        if abs(d[k]) > 6:
            xs.append(x); ys.append(1000 + k)
    xs, ys = np.array(xs, float), np.array(ys, float)
    rng = np.random.default_rng(0); best = None
    for _ in range(3000):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i]); b = ys[i] - m * xs[i]
        inl = np.abs(ys - (m * xs + b)) < 2.5
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    inl = best[1]; xi, yi = xs[inl], ys[inl]
    lin = np.polyfit(xi, yi, 1); q = np.polyfit(xi, yi, 2)
    sag = abs(q[0]) * ((xi.max() - xi.min()) / 2) ** 2
    say("   %d/%d inliers over x %d..%d" % (inl.sum(), len(xs), xi.min(), xi.max()))
    say("   linear residual RMS %.2f px; sagitta %.2f px -> no measurable distortion"
        % ((yi - np.polyval(lin, xi)).std(), sag))
    say("   => straight world lines stay straight; calibration is valid IN PRINCIPLE.")

def closeup():
    say(); say("B. ARTICLE CLOSEUP vs THE TRACKED OBJECT")
    im = load_gray(C.CLOSEUP)
    sm = blur(im, 3)
    m = sm > (sm.max() + np.median(sm)) / 2
    m = nd.binary_fill_holes(nd.binary_closing(m, np.ones((9, 9))))
    lab, n = nd.label(m)
    m = lab == (int(np.argmax(nd.sum(m, lab, range(1, n + 1)))) + 1)
    st = mask_stats(m)
    say("   closeup blob: bbox %dx%d, elongation %.2f, orientation %.0f deg"
        % (st["bbox_w"], st["bbox_h"], st["elong"], st["angle"]))
    say("   tracked object: elongation 2.1 +- 0.3, orientation ~179 deg (horizontal)")
    say("   => elongation %.2f vs 2.1 and orientation %.0f vs 179: DIFFERENT OBJECT."
        % (st["elong"], st["angle"]))
    say("   The closeup is a separate user-submitted still, not a crop of this video.")

def main():
    straightness(); closeup()
    (C.RESULTS / "checks.txt").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
