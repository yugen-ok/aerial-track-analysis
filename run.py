"""Run the whole analysis. Each stage reads the previous stage's files only.

    python run.py            strict contours (the reported result)
    python run.py loose      also produce the less-strict boundary variant
"""
import subprocess, sys
from pathlib import Path

STAGES = ["p1_locate", "p2_segment", "p3_shape", "p4_analyze", "p5_figures", "p6_checks", "p7_calibrate"]

def main(loose):
    root = Path(__file__).resolve().parent
    for s in STAGES:
        print(f"\n--- {s} ---")
        subprocess.run([sys.executable, str(root / "pipeline" / f"{s}.py")], check=True)
    if loose:
        print("\n--- p2_segment (loose) ---")
        subprocess.run([sys.executable, str(root / "pipeline" / "p2_segment.py"), "loose"], check=True)

if __name__ == "__main__":
    main("loose" in sys.argv)
