"""Every drawing operation. Sole owner of output imagery."""
import numpy as np
from PIL import Image, ImageDraw
from .shape import outline

def with_circle(rgb_arr, cx, cy, r, color=(255, 40, 40), width=4):
    im = Image.fromarray(rgb_arr.astype(np.uint8))
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)
    return np.asarray(im).copy()

def with_outline(rgb_arr, mask, origin, color):
    a = rgb_arr.copy()
    x0, y0 = origin
    sub = a[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    sub[outline(mask)] = color
    a[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = sub
    return a

def zoom(rgb_arr, cx, cy, half, scale=4, label=None, label_color=(255, 255, 0)):
    im = Image.fromarray(rgb_arr.astype(np.uint8))
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(im.width, cx + half)), int(min(im.height, cy + half)))
    z = im.crop(box).resize(((box[2] - box[0]) * scale, (box[3] - box[1]) * scale), Image.NEAREST)
    if label:
        ImageDraw.Draw(z).text((6, 4), label, fill=label_color)
    return z

def path_overlay(rgb_arr, centroids, line, color=(255, 255, 0), width=3):
    im = Image.fromarray(rgb_arr.astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.line(line, fill=color, width=width)
    for x, y in centroids:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
    return im
