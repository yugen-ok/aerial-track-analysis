"""Hand-drawn cutouts and masked template matching. Sole owner of NCC."""
import math
import numpy as np
from scipy import ndimage as nd
from scipy.signal import fftconvolve
from .imaging import load_rgb

def load_cutouts(path, min_px=200, row_height=200):
    """Extract the drawn shapes from a white canvas, in reading order
    (top row left->right, then lower rows)."""
    src = load_rgb(path)
    solid = nd.binary_closing(src.max(2) < 235, np.ones((5, 5)))
    lab, n = nd.label(solid)
    out = []
    for i, sl in enumerate(nd.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() < min_px:
            continue
        ys, xs = sl
        out.append(dict(x=xs.start, y=ys.start,
                        w=xs.stop - xs.start, h=ys.stop - ys.start,
                        mask=m, img=src[sl].mean(2)))
    out.sort(key=lambda o: (o["y"] // row_height, o["x"]))
    return out

def masked_ncc(image, template, mask, min_std=4.0):
    """Normalised cross-correlation over the template's own pixels only.
    Returns a 'valid'-shaped map; flat regions are set to -1 because NCC is
    meaningless (and numerically explosive) where local variance ~ 0."""
    M = mask.astype(np.float32)
    nM = M.sum()
    Mf = M[::-1, ::-1]
    T = (template - (template * M).sum() / nM) * M
    T /= math.sqrt((T * T).sum()) + 1e-9
    s1 = fftconvolve(image, Mf, "valid")
    s2 = fftconvolve(image * image, Mf, "valid")
    var = np.maximum(s2 - s1 * s1 / nM, 0.0)
    r = fftconvolve(image, T[::-1, ::-1], "valid") / np.sqrt(np.maximum(var, 1e-6))
    r = np.clip(r, -1, 1)
    r[np.sqrt(var / nM) < min_std] = -1
    return r
