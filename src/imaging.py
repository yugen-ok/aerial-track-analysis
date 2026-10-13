"""Pixel I/O and colour conversion. Sole owner of reading frames and of sRGB->Lab."""
import numpy as np
from PIL import Image
from scipy import ndimage as nd

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32)

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), np.float32)

def blur(a, sigma):
    return nd.gaussian_filter(a, sigma)

def rgb_to_lab(rgb):
    """sRGB (0-255) -> CIE L*a*b* (D65). Euclidean distance here is CIE76 dE."""
    a = rgb.astype(np.float64) / 255.0
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    t = (a @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)
