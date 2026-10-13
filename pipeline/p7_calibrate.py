"""Attempt camera calibration from vanishing points - A NEGATIVE RESULT.

The report claims the FOV cannot be recovered from this footage. That claim is
only credible if the attempt is reproducible, so it lives here.

in : data/frames/frame_00099.png
out: results/calibration.txt
"""
import math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray, blur

L = []
def say(s=""):
    print(s); L.append(s)

# approximate seeds for thin vertical structures: (name, x, y_from, y_to)
POSTS = [("fg post A", 1012, 824, 1000), ("fg post B", 1240, 816, 950),
         ("fg post C", 1140, 900, 1008), ("sign pole", 706, 650, 760),
         ("fence post", 1003, 648, 706)]
ACCEPT_RMS = 1.6     # px; a line this straight is usable for a vanishing point

def track_post(im, x0, ya, yb, half=14):
    """Follow a thin vertical feature by its per-row intensity-weighted centre."""
    pts, x = [], float(x0)
    for y in range(ya, yb):
        a, b = max(0, int(x) - half), int(x) + half
        seg = im[y, a:b]
        if len(seg) < 5:
            break
        w = seg.max() - seg
        w = np.clip(w - 0.5 * w.max(), 0, None)
        if w.sum() < 1e-6:
            continue
        x = a + float((np.arange(len(seg)) * w).sum() / w.sum())
        pts.append((x, y))
    return np.array(pts)

def main():
    im = blur(load_gray(C.FRAMES / "frame_00099.png"), 1.0)
    say("Vertical structures, fitted sub-pixel (x = m*y + c):")
    kept = []
    for name, x, ya, yb in POSTS:
        P = track_post(im, x, ya, yb)
        if len(P) < 20:
            say("  %-11s too short" % name); continue
        m, c = np.polyfit(P[:, 1], P[:, 0], 1)
        rms = float((P[:, 0] - (m * P[:, 1] + c)).std())
        ok = rms < ACCEPT_RMS
        say("  %-11s n=%3d  rms %6.2f px  %s" % (name, len(P), rms, "USABLE" if ok else "rejected"))
        if ok:
            kept.append((m, c))
    say()
    say("usable lines: %d" % len(kept))
    if len(kept) < 2:
        say("FAILED: a vanishing point needs at least two lines.")
        say("Cause: night CCTV rephotographed off a monitor leaves man-made edges")
        say("short (60-190 px), low-contrast and motion-blurred. The usable verticals")
        say("are also near-parallel, so their intersection is badly conditioned.")
        say("CONSEQUENCE: the horizontal field of view remains an ASSUMPTION")
        say("(%.0f deg assumed; %.0f-%.0f deg bracket), and every absolute angle," %
            (C.HFOV_ASSUMED, *C.HFOV_RANGE))
        say("rate and speed in the report scales with it.")
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "calibration.txt").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
