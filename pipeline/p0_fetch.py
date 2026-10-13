"""Acquire the source video and generate data/frames - the input to every other stage.

This stage OWNS data/frames. Delete that directory, run this script, and it is
reproduced exactly; no frame in the repository is hand-staged.

All frames are decoded to a scratch directory (the full set is needed to measure
the replay loop), then the analysis window is copied into data/frames.

in : data/source/ynet-r1ptysdime.mp4 (downloaded from SOURCE_URL if absent)
out: data/frames/frame_000{87..108}.png, and the measured loop period
usage: python pipeline/p0_fetch.py [--keep] [scratch_dir]
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray
from src.video import extract_frames, fetch


def loop_period(framedir, probe=400, lo=100, hi=260):
    """The monitor replays one short clip repeatedly. Recover the period by
    self-similarity of a downsampled frame sequence."""
    fs = sorted(Path(framedir).glob("frame_*.png"))[:probe]
    V = np.stack([load_gray(f)[::8, ::8].ravel() for f in fs])
    V = (V - V.mean(0)) / (V.std(0) + 1e-6)
    best = None
    for p in range(lo, min(hi, len(fs) - 20)):
        c = float(np.mean([np.corrcoef(V[i], V[i + p])[0, 1] for i in range(len(fs) - p)]))
        if best is None or c > best[0]:
            best = (c, p)
    return best[1], best[0]


def main(scratch, keep=False):
    video = C.DATA / "source" / "ynet-r1ptysdime.mp4"
    if not video.exists():
        print("downloading", C.SOURCE_URL)
        fetch(C.SOURCE_URL, video)

    scratch = Path(scratch)
    extract_frames(video, scratch)
    n = len(list(scratch.glob("frame_*.png")))
    print("decoded %d frames to %s" % (n, scratch))

    period, corr = loop_period(scratch)
    print("replay loop period: %d frames (mean self-correlation %.3f)" % (period, corr))
    if period != C.LOOP_PERIOD:
        print("WARNING: measured period %d differs from config.LOOP_PERIOD %d"
              % (period, C.LOOP_PERIOD))

    C.FRAMES.mkdir(parents=True, exist_ok=True)
    first, last = C.FIRST - C.PRE_ROLL, C.LAST
    for f in range(first, last + 1):
        shutil.copyfile(scratch / f"frame_{f:05d}.png", C.FRAMES / f"frame_{f:05d}.png")
    print("staged frames %d..%d into %s" % (first, last, C.FRAMES))

    if not keep:
        shutil.rmtree(scratch, ignore_errors=True)
        print("removed the scratch set (pass --keep to retain it)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--keep"]
    main(Path(args[0]) if args else C.ROOT / "scratch" / "frames",
         keep="--keep" in sys.argv)
