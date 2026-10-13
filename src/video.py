"""Acquisition and encoding. Sole owner of the external yt-dlp / ffmpeg calls."""
import subprocess, sys
from pathlib import Path

def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def fetch(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "yt_dlp", "-o", str(dest), url], check=True)
    return dest

def extract_frames(video, outdir, pattern="frame_%05d.png"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-i", str(video), "-start_number", "1",
                    str(outdir / pattern)], check=True)
    return outdir

def encode_clip(framedir, first, count, dest, fps=25, crf=16):
    subprocess.run([ffmpeg(), "-y", "-framerate", str(fps), "-start_number", str(first),
                    "-i", str(Path(framedir) / "frame_%05d.png"), "-frames:v", str(count),
                    "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dest)],
                   check=True)
    return dest
