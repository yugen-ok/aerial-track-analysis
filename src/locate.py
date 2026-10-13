"""Per-frame object position. Sole owner of the search gate and centroid refinement."""
import math
import numpy as np
from scipy import ndimage as nd
from .templates import masked_ncc

def _gate(shape, cx, cy, radius):
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(gx - cx, gy - cy) <= radius

def best_match(gray, cutouts, band, prev=None, vel=(0.0, 0.0), radius=75):
    """Scan every cutout over `band` = (y0, y1, x0, x1); optionally gate the
    search to `radius` px around the velocity-predicted previous position.
    Returns (score, cutout_index, cx, cy) in full-frame coordinates."""
    y0, y1, x0, x1 = band
    sub = gray[y0:y1, x0:x1]
    best = None
    for k, o in enumerate(cutouts):
        r = masked_ncc(sub, o["img"], o["mask"])
        if prev is not None:
            px = prev[0] + vel[0] - x0 - o["w"] / 2
            py = prev[1] + vel[1] - y0 - o["h"] / 2
            r = np.where(_gate(r.shape, px, py, radius), r, -1)
        j = np.unravel_index(np.argmax(r), r.shape)
        s = float(r[j])
        if best is None or s > best[0]:
            best = (s, k, x0 + j[1] + o["w"] / 2, y0 + j[0] + o["h"] / 2)
    return best

def refine_centroid(gray, cx, cy, radius, k_mad=3.0, floor=6.0, min_px=12):
    """Snap to the intensity-weighted centroid of the brightest blob inside the
    matched window. Carries the faint late frames, where NCC alone drifts."""
    r = int(radius)
    y0, y1 = max(0, int(cy - r)), int(cy + r)
    x0, x1 = max(0, int(cx - r)), int(cx + r)
    w = gray[y0:y1, x0:x1]
    med = np.median(w)
    mad = 1.4826 * np.median(np.abs(w - med))
    hot = w > med + max(k_mad * mad, floor)
    if hot.sum() < min_px:
        return cx, cy
    lab, _ = nd.label(hot)
    big = int(np.argmax(nd.sum(hot, lab, range(1, lab.max() + 1)))) + 1
    m = lab == big
    wt = (w - med)[m]
    ys, xs = np.nonzero(m)
    return x0 + float((xs * wt).sum() / wt.sum()), y0 + float((ys * wt).sum() / wt.sum())

def radius_for(cutout, margin=1.15):
    """Marking radius = half the cutout's diagonal, plus a margin of error."""
    return 0.5 * math.hypot(cutout["w"], cutout["h"]) * margin
