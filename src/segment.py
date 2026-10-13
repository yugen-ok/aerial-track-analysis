"""Boundary finding by minimax colour-jump paths. Sole owner of thresholds and carving."""
import heapq
import numpy as np
from scipy import ndimage as nd

NBR = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

def minimax_cost(lab, region, seed):
    """cost[p] = min over paths seed->p of the MAXIMUM single-step dE.
    Walking inward from the sky ring, cost only rises where colour spikes."""
    H, W, _ = lab.shape
    cost = np.full((H, W), np.inf)
    cost[seed] = 0.0
    heap = [(0.0, int(y), int(x)) for y, x in zip(*np.nonzero(seed))]
    heapq.heapify(heap)
    done = np.zeros((H, W), bool)
    while heap:
        c, y, x = heapq.heappop(heap)
        if done[y, x]:
            continue
        done[y, x] = True
        for dy, dx in NBR:
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= H or nx >= W:
                continue
            if not region[ny, nx] or done[ny, nx]:
                continue
            w = float(np.linalg.norm(lab[ny, nx] - lab[y, x]))
            nc = c if c > w else w
            if nc < cost[ny, nx]:
                cost[ny, nx] = nc
                heapq.heappush(heap, (nc, ny, nx))
    return cost

def otsu(v, bins=128):
    h, e = np.histogram(v, bins=bins)
    p = h / h.sum()
    m = (e[:-1] + e[1:]) / 2
    c = np.cumsum(p)
    mu = np.cumsum(p * m)
    den = c * (1 - c)
    den[den <= 0] = np.nan
    return m[np.nanargmax((mu[-1] * c - mu) ** 2 / den)]

def stable_threshold(cost, region, lo=0.004, hi=0.35):
    """STRICT rule (MSER): sweep the cut level, keep the one whose enclosed area
    is most stable - the level sitting in the 'spike'."""
    v = cost[region & np.isfinite(cost)]
    n = region.sum()
    Ts = np.unique(np.percentile(v, np.linspace(70, 99.6, 90)))
    A = np.array([(v > t).sum() for t in Ts], float)
    best = None
    for i in range(1, len(Ts) - 1):
        a = A[i]
        if a < lo * n or a > hi * n:
            continue
        stab = (A[i - 1] - A[i + 1]) / max(a, 1.0)
        if best is None or stab < best[0]:
            best = (stab, Ts[i])
    return (best[1], best[0]) if best else (float(np.percentile(v, 97.0)), float("nan"))

def noise_threshold(cost, ring, sens):
    """LOOSE rule: a step need only beat `sens` x the sky's own 95th-percentile
    jump. sens < 1 means the boundary need not be as sharp."""
    v = cost[ring & np.isfinite(cost)]
    return (sens * float(np.percentile(v, 95)) if v.size else 0.3 * sens), float("nan")

def carve(cost, delta, inner, T, centre, loose=False):
    """Pixels enclosed by the boundary, cleaned. `delta` = dE of each pixel from
    the sky reference; its cut removes halo that is still sky-coloured."""
    obj = inner & (cost > T)
    if obj.any():
        dv = delta[obj]
        if dv.max() > dv.min():
            cut = min(otsu(dv), float(np.percentile(dv, 45 if loose else 55)))
            obj = obj & (delta > cut)
    obj = nd.binary_opening(obj, np.ones((5, 5)))      # sever thin necks into the sky
    obj = nd.binary_closing(obj, np.ones((5, 5)))
    obj = nd.binary_fill_holes(obj)
    obj = nd.gaussian_filter(obj.astype(np.float32), 1.6) > 0.5
    obj &= inner
    lab, n = nd.label(obj)
    if n:
        sz = nd.sum(obj, lab, range(1, n + 1))
        cen = nd.center_of_mass(obj, lab, range(1, n + 1))
        cx, cy = centre
        score = [sz[i] / (1 + np.hypot(cen[i][0] - cy, cen[i][1] - cx)) for i in range(n)]
        obj = lab == (int(np.argmax(score)) + 1)
    return obj
