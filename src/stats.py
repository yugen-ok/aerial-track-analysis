"""Fitting and significance. Sole owner of every regression used in the report."""
import math
import numpy as np

def linfit(x, y):
    """Least-squares line with standard error, t and R^2 on the slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    s, i = np.polyfit(x, y, 1)
    res = y - (s * x + i)
    se = math.sqrt((res ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(slope=s, intercept=i, se=se, t=s / se if se else float("nan"),
                r=r, r2=r * r, n=n, rms=float(res.std()))

def expfit(x, y):
    """Fit y = A exp(k x); `pct` is the per-step percentage change."""
    f = linfit(x, np.log(np.asarray(y, float)))
    f["A"] = math.exp(f["intercept"])
    f["k"] = f["slope"]
    f["pct"] = 100 * (math.exp(f["slope"]) - 1)
    return f

def ratio_ci(x, y, span, z=1.96):
    """From a log-linear fit, the multiplicative change over `span` and its CI."""
    f = expfit(x, y)
    lo, hi = f["slope"] - z * f["se"], f["slope"] + z * f["se"]
    return dict(best=math.exp(f["slope"] * span), lo=math.exp(lo * span),
                hi=math.exp(hi * span), slope=f["slope"], se=f["se"])

def f_test(x, y, deg_lo=1, deg_hi=2):
    """Is the higher-order term justified? F = extra variance explained / residual."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r1 = y - np.polyval(np.polyfit(x, y, deg_lo), x)
    r2 = y - np.polyval(np.polyfit(x, y, deg_hi), x)
    n = len(x)
    return float(((r1 ** 2).sum() - (r2 ** 2).sum()) / ((r2 ** 2).sum() / (n - deg_hi - 1)))

def axial_stats(deg):
    """Circular statistics for undirected axes (mod 180): mean, resultant, sd."""
    a = np.radians(np.asarray(deg, float))
    z = np.exp(2j * a).mean()
    R = abs(z)
    return dict(mean=float(math.degrees(np.angle(z) / 2) % 180), R=float(R),
                sd=float(math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))) / 2)))

def unwrap_axial(deg):
    return np.degrees(np.unwrap(2 * np.radians(np.asarray(deg, float))) / 2)
