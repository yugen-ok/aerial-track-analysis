"""Shape descriptors from a binary mask. Sole owner of the PCA axes."""
import math
import numpy as np
from scipy import ndimage as nd

def mask_stats(mask, origin=(0, 0)):
    """Intensity-free second-moment description. major/minor are 2-sigma full
    extents (4*sqrt(eigenvalue)); angle is axial, in [0,180)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    C = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    ev, evec = np.linalg.eigh(C)
    o = ev.argsort()[::-1]
    ev, evec = ev[o], evec[:, o]
    maj, mnr = 4 * np.sqrt(np.maximum(ev, 1e-12))
    return dict(area=int(mask.sum()),
                major=float(maj), minor=float(mnr),
                elong=float(maj / max(mnr, 1e-9)),
                angle=float(math.degrees(math.atan2(evec[1, 0], evec[0, 0])) % 180),
                cx=float(xs.mean() + origin[0]), cy=float(ys.mean() + origin[1]),
                bbox_w=int(xs.max() - xs.min() + 1), bbox_h=int(ys.max() - ys.min() + 1))

def mean_shape(masks, target_area, canvas=241):
    """Centroid-aligned median shape of the well-resolved frames, rescaled to
    `target_area`. Used to replace frames whose carve collapsed."""
    acc = np.zeros((canvas, canvas), float)
    for m in masks:
        ys, xs = np.nonzero(m)
        t = np.zeros((canvas, canvas), bool)
        yy = np.clip((ys - ys.mean() + canvas // 2).round().astype(int), 0, canvas - 1)
        xx = np.clip((xs - xs.mean() + canvas // 2).round().astype(int), 0, canvas - 1)
        t[yy, xx] = True
        acc += nd.binary_fill_holes(nd.binary_closing(t, np.ones((3, 3)))).astype(float)
    shape = (acc / len(masks)) >= 0.5
    s = math.sqrt(target_area / max(shape.sum(), 1))
    return nd.zoom(shape.astype(float), s, order=1) > 0.5

def place(shape, like_mask, cx, cy):
    """Stamp `shape` centred on (cx, cy) into an array shaped like `like_mask`."""
    out = np.zeros_like(like_mask)
    zy, zx = np.nonzero(shape)
    yy = np.clip((zy - shape.shape[0] / 2 + cy).round().astype(int), 0, out.shape[0] - 1)
    xx = np.clip((zx - shape.shape[1] / 2 + cx).round().astype(int), 0, out.shape[1] - 1)
    out[yy, xx] = True
    return nd.binary_fill_holes(nd.binary_closing(out, np.ones((3, 3))))

def outline(mask, width=3):
    b = mask & ~nd.binary_erosion(mask, np.ones((3, 3)))
    return nd.binary_dilation(b, np.ones((width, width)))
