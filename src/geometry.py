"""Angles, ranges and speeds. Sole owner of every world-geometry conversion.

All of it rests on one assumed number, the CCTV horizontal field of view,
because a single uncalibrated camera cannot measure absolute scale.
"""
import math
import numpy as np

PICTURE_WIDTH_PX = 1635.0   # CCTV tile width inside the photographed monitor
FPS = 25.0

def deg_per_px(hfov, picw=PICTURE_WIDTH_PX):
    return hfov / picw

def frustum(hfov, aspect_h_over_w, d):
    """Cross-section of the viewing pyramid at distance d: (width, height)."""
    w = 2 * d * math.tan(math.radians(hfov / 2))
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(hfov / 2)) * aspect_h_over_w))
    return w, 2 * d * math.tan(math.radians(vfov / 2)), vfov

def elevation(y_px, horizon_px, hfov):
    return (horizon_px - y_px) * deg_per_px(hfov)

def range_ratio_level_flight(elev1, elev2):
    """If altitude were constant, a rise in elevation forces this range ratio."""
    return math.tan(math.radians(elev1)) / math.tan(math.radians(elev2))

def _xyz(d, elev, azim):
    e, a = math.radians(elev), math.radians(azim)
    return np.array([d * math.cos(e) * math.cos(a), d * math.cos(e) * math.sin(a),
                     d * math.sin(e)])

def climb_angle(elev1, elev2, d_azim, range_ratio):
    """Climb above horizontal needed to produce the observed elevation change,
    given a range ratio over the run."""
    D = _xyz(range_ratio, elev2, d_azim) - _xyz(1.0, elev1, 0.0)
    return math.degrees(math.atan2(D[2], math.hypot(D[0], D[1])))

def velocity_axis_angle(traverse_deg, radial_fraction):
    """Angle between the flight vector and the camera axis. 90 deg = pure crossing."""
    t = math.radians(traverse_deg)
    return 90.0 if radial_fraction == 0 else math.degrees(math.atan2(t, radial_fraction))

def body_lengths_per_sec(travel_px, length_px, dt):
    """Range-INDEPENDENT speed: both travel and size scale with distance."""
    return (travel_px / length_px) / dt

def speed_for_length(bl_per_sec, length_m):
    return bl_per_sec * length_m

def range_for_length(length_m, angular_major_deg):
    return length_m / (2 * math.tan(math.radians(angular_major_deg) / 2))
