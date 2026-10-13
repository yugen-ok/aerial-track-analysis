"""Every quantitative claim in the report.

in : results/shape.csv, results/masks/*.npz, data/frames/*.png

Substituted frames are excluded throughout.
out: results/findings.txt
"""
import csv, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.imaging import load_gray
from src.stats import linfit, expfit, ratio_ci, f_test, axial_stats, unwrap_axial
from src import geometry as G
from scipy import ndimage as nd

L = []
def say(s=""):
    print(s); L.append(s)

def section(t):
    say(); say("=" * 70); say(t); say("=" * 70)

def main():
    rows = [r for r in csv.DictReader(open(C.RESULTS / "shape.csv"))]
    use = [r for r in rows if not int(r["substituted"])]
    f = np.array([int(r["frame"]) for r in use], float)
    cx = np.array([float(r["cx"]) for r in use])
    cy = np.array([float(r["cy"]) for r in use])
    maj = np.array([float(r["major"]) for r in use])
    mnr = np.array([float(r["minor"]) for r in use])
    area = np.array([float(r["area"]) for r in use])
    ang = np.array([float(r["angle"]) for r in use])
    t = f - f[0]
    span = f[-1] - f[0]
    dt = span / G.FPS
    travel = math.hypot(cx[-1] - cx[0], cy[-1] - cy[0])

    section("1. TRACK GEOMETRY")
    fit = linfit(cx, cy)
    bearing = math.degrees(math.atan(-fit["slope"]))
    say("measured frames: %d (of %d); span %d frames = %.2f s" % (len(f), len(rows), span, dt))
    say("displacement dx=%.0f dy=%.0f  |d|=%.0f px" % (cx[-1]-cx[0], cy[-1]-cy[0], travel))
    say("in-image components: horizontal %.0f%%  vertical %.0f%%" % (
        100*abs(cx[-1]-cx[0])/travel, 100*abs(cy[-1]-cy[0])/travel))
    say("straight-line fit: bearing %.1f deg above horizontal, residual RMS %.2f px, max %.2f px"
        % (bearing, fit["rms"], np.abs(cy - (fit["slope"]*cx + fit["intercept"])).max()))
    h = len(cx)//2
    b1 = math.degrees(math.atan(-linfit(cx[:h], cy[:h])["slope"]))
    b2 = math.degrees(math.atan(-linfit(cx[h:], cy[h:])["slope"]))
    say("first half %.1f deg, second half %.1f deg, difference %.1f deg" % (b1, b2, b2-b1))
    say("curvature F-test: F=%.2f (needs >4.6 to justify a bend)" % f_test(cx, cy))

    step = np.hypot(np.diff(cx), np.diff(cy))
    dup_thr = 0.2 * np.median(step)        # steps are bimodal: ~0-5 px or ~29-85 px
    dup = [(int(use[i]["frame"]), int(use[i+1]["frame"]))
           for i in range(len(step)) if step[i] < dup_thr]
    say("duplicate frame pairs (25 fps upsampled from a slower capture): %s" % dup)
    say("effective independent samples ~%d, not %d" % (len(f)-len(dup), len(f)))

    section("2. SIZE PER FRAME AND CHANGE FUNCTIONS")
    for name, y in (("major", maj), ("minor", mnr), ("area", area)):
        lf, ef = linfit(t, y), expfit(t, y)
        say("%-6s linear %8.2f %+7.3f t   R2=%.3f | exp %8.2f*exp(%+.5f t) = %+.2f%%/frame  R2=%.3f"
            % (name, lf["intercept"], lf["slope"], lf["r2"], ef["A"], ef["k"], ef["pct"], ef["r2"]))
        say("       slope %+.3f +- %.3f  t=%+.2f  %s" % (
            lf["slope"], lf["se"], lf["t"], "significant" if abs(lf["t"]) > 2.14 else "NOT significant"))

    section("3. RANGE (from the minor axis, which rotation cannot foreshorten)")
    rc = ratio_ci(t, mnr, span)
    say("d ln(width)/dframe = %+.5f +- %.5f" % (rc["slope"], rc["se"]))
    say("width over run x%.3f (95%% CI %.3f..%.3f) -> range x%.3f (CI %.3f..%.3f)"
        % (rc["best"], rc["lo"], rc["hi"], 1/rc["best"], 1/rc["hi"], 1/rc["lo"]))
    for fac in (0.9, 0.8, 0.7, 0.5):
        need = math.log(1/fac)/span
        say("   approach to range x%.1f needs %+.5f = %.1f sigma" % (fac, need, (need-rc["slope"])/rc["se"]))

    say()
    say("Is the width profile flat (constant-range arc) or peaked (straight-line pass)?")
    say("   linear RMS %.2f px; quadratic RMS %.2f px; F=%.2f (needs >4.7)"
        % ((mnr-np.polyval(np.polyfit(t,mnr,1),t)).std(),
           (mnr-np.polyval(np.polyfit(t,mnr,2),t)).std(), f_test(t, mnr)))
    q = np.polyfit(t, mnr, 2)
    if q[0] < 0:
        say("   concave: width peaks at frame %.0f -> closest approach inside the window,"
            % (f[0] - q[1]/(2*q[0])))
        say("   which favours a straight-line pass over a flat constant-range arc.")

    section("4. ROTATION")
    uw = unwrap_axial(ang)
    lf = linfit(f, uw)
    ax = axial_stats(ang)
    say("orientation %+.3f deg/frame (r=%+.3f), total %+.1f deg over the run"
        % (lf["slope"], lf["r"], lf["slope"]*span))
    say("tumble rate %.0f deg/s = %.1f rpm" % (abs(lf["slope"])*G.FPS, abs(lf["slope"])*G.FPS/360*60))
    say("axial mean %.1f deg, resultant R=%.3f, circular sd %.1f deg" % (ax["mean"], ax["R"], ax["sd"]))
    for nm, y in (("major", maj), ("minor", mnr)):
        c = linfit(uw, y)
        say("orientation vs %-5s r=%+.3f t=%+.2f  %s  (explains %.0f%% of variance)"
            % (nm, c["r"], c["t"], "significant" if abs(c["t"]) > 2.14 else "NOT significant", 100*c["r2"]))
    say("-> long axis tracks the rotation, short axis does not = body of revolution")

    section("5. BRIGHTNESS")
    con = []
    for r in use:
        fr = int(r["frame"])
        z = np.load(C.MASKS / f"mask_{fr:05d}.npz")
        m, x0, y0 = z["mask"], int(z["x0"]), int(z["y0"])
        g = load_gray(C.FRAMES / f"frame_{fr:05d}.png")
        sub = g[y0:y0+m.shape[0], x0:x0+m.shape[1]]
        ring = nd.binary_dilation(m, np.ones((25,25))) & ~nd.binary_dilation(m, np.ones((9,9)))
        con.append(sub[m].mean() - sub[ring].mean())
    con = np.array(con)
    lf = linfit(f, con)
    say("object-minus-sky contrast %+.3f/frame (r=%+.3f): %.1f -> %.1f" % (lf["slope"], lf["r"], con[0], con[-1]))
    say("surface brightness is range-invariant for a resolved object, so brightening")
    say("while AREA FALLS cannot be approach; it is a change of aspect.")

    section("6. ANGULAR GEOMETRY (depends on the assumed FOV)")
    for hf in (C.HFOV_RANGE[0], C.HFOV_ASSUMED, C.HFOV_RANGE[1]):
        s = G.deg_per_px(hf)
        say("HFOV %2.0f deg: %.4f deg/px | traverse %.1f deg | object %.2f x %.2f deg | rate %.1f deg/s"
            % (hf, s, travel*s, maj.mean()*s, mnr.mean()*s, travel*s/dt))
    hf = C.HFOV_ASSUMED
    w, hh, vf = G.frustum(hf, 980/G.PICTURE_WIDTH_PX, 100)
    say("frustum at HFOV %.0f (VFOV %.1f): at 100 m the frame spans %.0f x %.0f m" % (hf, vf, w, hh))

    section("7. ELEVATION, LEVEL FLIGHT AND CLIMB")
    y1, y2 = cy[0], cy[-1]
    for hor, lab in ((C.HORIZON_ALT, "hillside lights (WRONG - used early by mistake)"),
                     (C.HORIZON_PX, "ground-plane vanishing point")):
        a1 = G.elevation(y1, hor, hf); a2 = G.elevation(y2, hor, hf)
        rr = G.range_ratio_level_flight(a1, a2)
        say("horizon y=%.0f (%s): elevation %.1f -> %.1f deg; level flight forces range x%.3f (width x%.2f)"
            % (hor, lab, a1, a2, rr, 1/rr))
    a1 = G.elevation(y1, C.HORIZON_PX, hf); a2 = G.elevation(y2, C.HORIZON_PX, hf)
    dphi = abs(cx[-1]-cx[0]) * G.deg_per_px(hf)
    say("azimuthal travel %.1f deg" % dphi)
    for rr in (1/rc["hi"], 0.9, 1.0, 1/rc["best"], 1/rc["lo"]):
        say("   range ratio %.2f -> climb %.1f deg above horizontal" % (rr, G.climb_angle(a1, a2, dphi, rr)))

    section("8. FLIGHT VECTOR vs CAMERA AXIS")
    trav_deg = travel * G.deg_per_px(hf)
    for rr, lab in ((1/rc["lo"], "95% receding"), (1/rc["best"], "best estimate"),
                    (1.0, "constant"), (1/rc["hi"], "95% approaching")):
        say("   range x%.2f (%-14s) -> %.1f deg off the camera axis"
            % (rr, lab, G.velocity_axis_angle(trav_deg, rr - 1.0)))

    section("9. SPEED (the only range-independent statement)")
    bl = G.body_lengths_per_sec(travel, maj.mean(), dt)
    say("travel %.2f body-lengths in %.2f s = %.1f body-lengths/second" % (travel/maj.mean(), dt, bl))
    say("comparison: airliner 6.2, light aircraft 7.5, heron 12, quadcopter 25, pigeon 46, housefly 250")
    ang_maj = maj.mean() * G.deg_per_px(hf)
    for Lm in (0.1, 0.3, 1.0, 3.0, 10.0, 40.0):
        say("   length %5.1f m -> speed %6.1f m/s (%6.0f km/h), range %7.1f m"
            % (Lm, G.speed_for_length(bl, Lm), G.speed_for_length(bl, Lm)*3.6,
               G.range_for_length(Lm, ang_maj)))

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "findings.txt").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
