"""Download the source video, extract all frames, and measure the replay loop.

The repository ships only the 22 frames the analysis needs; this regenerates the
full 845-frame set (~1.6 GB) into a scratch directory if you want it.

usage: python pipeline/p0_fetch.py [scratch_dir]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.video import fetch, extract_frames
from src.imaging import load_gray
import numpy as np

def loop_period(framedir, probe=400, lo=100, hi=260):
    """The monitor replays one short clip repeatedly. Find the period by
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

def main(scratch):
    video = C.DATA / "source" / "ynet-r1ptysdime.mp4"
    if not video.exists():
        fetch(C.SOURCE_URL, video)
    extract_frames(video, scratch)
    print("frames in", scratch)
    p, c = loop_period(scratch)
    print("replay loop period: %d frames (mean self-correlation %.3f)" % (p, c))

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else C.ROOT / "scratch" / "frames")
