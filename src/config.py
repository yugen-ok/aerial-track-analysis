"""Paths and every tunable constant. Single source of truth for both pipelines."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"
CUTOUTS = DATA / "cutouts" / "cutouts.png"
CLOSEUP = DATA / "reference" / "ynet-closeup.jpg"
RESULTS = ROOT / "results"
MASKS = RESULTS / "masks"
FIGURES = RESULTS / "figures"

SOURCE_URL = "https://www.ynet.co.il/news/article/r1ptysdime"
LOOP_PERIOD = 169              # frames; the monitor replays a ~6.8 s clip
FIRST, LAST = 90, 108          # frames in which the object is visible (operator call)
PRE_ROLL = 3                   # extra frames kept before FIRST, for the clip

SKY_BAND = (45, 340, 600, 1750)   # y0, y1, x0, x1 - where the object crosses
GATE_RADIUS = 75                  # px, search gate around predicted position

GROW = 1.85     # sky ring radius, in units of the marking radius
KEEP = 1.25     # object may grow this far past the marking before the limit lifts
KEEP_STEP = 0.25
ZOOM = 2.4      # output crop half-width, in marking radii
LOOSE_SENS = 0.75   # loose-mode boundary sharpness, in units of the sky's own noise

HFOV_ASSUMED = 70.0            # deg; the one uncalibrated number
HFOV_RANGE = (60.0, 90.0)
HORIZON_PX = 665.0             # from the ground-plane vanishing point
HORIZON_ALT = 311.0            # the hillside lights - what was used by mistake first
