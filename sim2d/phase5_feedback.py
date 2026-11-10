"""
phase5_feedback.py
Phase 5 of the rocket simulation: closing the loop.

Phase 4 showed that a rocket flying a fixed plan lands anywhere in a wide
cloud, because nothing in the plan reacts to what actually happens.

Here the rocket gets one sense and one decision:

    SENSE:   at every guidance cycle (1 s, then 0.1 s near the end) it knows its own height, speed,
             flight-path angle and mass (perfect sensors for now).
    DECIDE:  it asks "if I shut the engine off right now, where would I land?"
             using its onboard model of a normal day (no wind, standard air,
             nominal drag). It shuts the engine off at the moment that answer
             equals the target.

This is called range control by engine cutoff. We compare, with the SAME
random errors as Phase 4:

    OPEN LOOP:    the engine is cut by a timer set in the plan.
    CLOSED LOOP:  the engine is cut by the prediction above.

The plan keeps a small propellant reserve (4 % of the burn). A cutoff controller
can only make the burn SHORTER, never longer, so it needs some propellant to
spare to fix a weak engine.

Put this file in the SAME folder as phase2_gravity_turn.py,
phase3_shooting.py and phase4_montecarlo.py, then run:

    python3 phase5_feedback.py                 (target 400 km, 60 flights)
    python3 phase5_feedback.py 300             (target 300 km)
    python3 phase5_feedback.py 400 --n=200     (200 flights per arc)
    python3 phase5_feedback.py 400 --no-plot   (text only)

Limitations: the sensors are perfect (real ones are noisy: next phase), the
simulation is 2D (downrange error only), and the only control is when to cut
the engine (the kick angle is still fixed in advance).
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from phase2_gravity_turn import (H_SCALE, KICK_SPEED, MU_EARTH, R_EARTH, RHO0,
                                 Rocket, integrate_fixed, integrate_until)
from phase4_montecarlo import (ERROR_SOURCES, NO_ERRORS, SEED, draw_errors,
                               make_rhs, perturbed_rocket)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_TARGET_KM = 400.0
DEFAULT_N = 60
DT_MC = 0.10            # [s] integration step of the real flight
GUIDANCE_DT = 1.0       # [s] how often the guidance computer runs
TERMINAL_DT = 0.1       # [s] faster cycle used when the cutoff is close
RESERVE = 0.04          # fraction of the burn time kept as propellant reserve


# ---------------------------------------------------------------------------
# The onboard predictor: "where would I land if I cut the engine now?"
# ---------------------------------------------------------------------------
def predict_range_km(state, base):
    """Fly the rest of the trajectory with the engine off, using the NOMINAL
    model (no wind, standard air, nominal drag), and return where it lands.

    Plain Python maths (no NumPy) so it is fast enough to call every second.
    """
    h, v, gam, x, m = state
    k_drag = 0.5 * RHO0 * base.cd * base.area / m   # drag deceleration = k * exp(-h/H) * v^2

    def deriv(h, v, gam):
        r = R_EARTH + h
        g = MU_EARTH / (r * r)
        sg, cg = math.sin(gam), math.cos(gam)
        drag = k_drag * math.exp(-max(h, 0.0) / H_SCALE) * v * v
        return v * sg, -drag - g * sg, (v / r - g / v) * cg, v * cg * R_EARTH / r

    for _ in range(5000):
        dt = 2.0 if h > 60e3 else 0.5
        a1 = deriv(h, v, gam)
        a2 = deriv(h + 0.5 * dt * a1[0], v + 0.5 * dt * a1[1], gam + 0.5 * dt * a1[2])
        a3 = deriv(h + 0.5 * dt * a2[0], v + 0.5 * dt * a2[1], gam + 0.5 * dt * a2[2])
        a4 = deriv(h + dt * a3[0], v + dt * a3[1], gam + dt * a3[2])
        h_new = h + dt / 6.0 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        v_new = v + dt / 6.0 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        g_new = gam + dt / 6.0 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        x_new = x + dt / 6.0 * (a1[3] + 2 * a2[3] + 2 * a3[3] + a4[3])
        if h_new <= 0.0:
            frac = h / (h - h_new)           # where in this step it touches the ground
            return (x + frac * (x_new - x)) / 1e3
        h, v, gam, x = h_new, v_new, g_new, x_new
    return float("nan")


# ---------------------------------------------------------------------------
# One flight, open loop or closed loop
# ---------------------------------------------------------------------------
def fly_mode(base, p, kick_deg, plan_cut_s, target_km=None, closed_loop=False,
             dt=DT_MC):
    """One complete flight.

    open loop:   the engine is cut at time plan_cut_s (a timer in the plan).
    closed loop: the engine is cut when the onboard prediction says the
                 rocket would land on target_km.
    """
    rocket = perturbed_rocket(base, p)
    rho_scale = 1.0 + p["rho"]
    f_rise = make_rhs(rocket, rho_scale, p["wind"], burning=True, hold_vertical=True)
    f_burn = make_rhs(rocket, rho_scale, p["wind"], burning=True)
    f_coast = make_rhs(rocket, rho_scale, p["wind"], burning=False)

    # Vertical rise, then the kick (same as Phase 4).
    y0 = np.array([0.0, 1e-3, np.pi / 2.0, 0.0, rocket.m0])
    ts1, ys1 = integrate_until(f_rise, y0, 0.0, dt, lambda y: y[1] >= KICK_SPEED)
    y = ys1[-1].copy()
    y[2] -= np.radians(kick_deg + p["kick"])
    t = ts1[-1]
    t_depl = rocket.burn_time               # the time the propellant runs out

    seg_y = [ys1]
    if not closed_loop:
        t_cut = min(plan_cut_s, t_depl)
        n = max(1, int(round((t_cut - t) / dt)))
        _, ys = integrate_fixed(f_burn, y, t, t_cut, n)
        seg_y.append(ys)
        y = ys[-1]
    else:
        r_prev, t_prev = None, None
        t_cut = None
        cycle = GUIDANCE_DT
        while True:
            r_now = predict_range_km(y, base)
            if r_now >= target_km:
                t_cut = t                                   # already there: cut now
                break
            if r_prev is not None and r_now > r_prev:
                rate = (r_now - r_prev) / (t - t_prev)     # km of range per second of burn
                time_to_go = (target_km - r_now) / rate
                if time_to_go <= 3.0 * GUIDANCE_DT:
                    cycle = TERMINAL_DT                     # cutoff is near: run faster
                if time_to_go <= 2.0 * cycle:
                    t_cut = min(t + time_to_go, t_depl)    # schedule the cutoff
                    n = max(1, int(round((t_cut - t) / dt)))
                    _, ys = integrate_fixed(f_burn, y, t, t_cut, n)
                    seg_y.append(ys)
                    y = ys[-1]
                    break
            if t >= t_depl - 1e-9:
                t_cut = t_depl                              # out of propellant
                break
            r_prev, t_prev = r_now, t
            t_next = min(t + cycle, t_depl)
            n = max(1, int(round((t_next - t) / dt)))
            _, ys = integrate_fixed(f_burn, y, t, t_next, n)
            seg_y.append(ys)
            y = ys[-1]
            t = t_next

    burn_ys = np.vstack(seg_y)
    crashed = bool(burn_ys[:, 0].min() < -1.0)
    state_cut = y.copy()
    ts3, ys3 = integrate_until(f_coast, y, t_cut, dt, lambda yy: yy[0] <= 0.0)
    return {
        "range_km": float("nan") if crashed else ys3[-1, 3] / 1e3,
        "apex_km": max(burn_ys[:, 0].max(), ys3[:, 0].max()) / 1e3,
        "t_cut": t_cut,
        "t_depl": t_depl,
        "saturated": bool(closed_loop and t_cut >= t_depl - 1e-6),
        "prop_left": float(state_cut[4] - rocket.m_dry),
        "state_cut": state_cut,
    }


def run_one(args):
    base, p, kick_deg, plan_cut, target_km, closed_loop = args
    return fly_mode(base, p, kick_deg, plan_cut, target_km, closed_loop)


# ---------------------------------------------------------------------------
# Planning: kick angles that hit the target when the TIMER cuts the engine
# ---------------------------------------------------------------------------
def solve_kicks(base, target_km, plan_cut_s):
    def landing(kick_deg):
        return fly_mode(base, NO_ERRORS, kick_deg, plan_cut_s)["range_km"]

    scan_k = np.arange(0.1, 2.41, 0.1)
    scan_r = np.array([landing(k) for k in scan_k])
    i = int(np.nanargmax(scan_r))
    a, b = scan_k[max(i - 1, 0)], scan_k[min(i + 1, len(scan_k) - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = landing(c), landing(d)
    for _ in range(14):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = landing(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = landing(d)
    k_peak = 0.5 * (a + b)
    r_peak = landing(k_peak)
    if target_km > r_peak:
        return None, r_peak

    def bisect(k_a, k_b):
        f_a, f_b = landing(k_a) - target_km, landing(k_b) - target_km
        if np.isnan(f_a) or np.isnan(f_b) or f_a * f_b > 0:
            return None
        for _ in range(60):
            k_mid = 0.5 * (k_a + k_b)
            f_mid = landing(k_mid) - target_km
            if abs(f_mid) < 0.005:
                return k_mid
            if f_a * f_mid < 0:
                k_b = k_mid
            else:
                k_a, f_a = k_mid, f_mid
        return 0.5 * (k_a + k_b)

    k_valid_max = scan_k[~np.isnan(scan_r)].max()
    out = {}
    k_steep = bisect(0.001, k_peak)
    k_flat = bisect(k_peak, k_valid_max)
    if k_steep is not None:
        out["steep (high arc)"] = k_steep
    if k_flat is not None:
        out["flat (low arc)"] = k_flat
    return out, r_peak


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(base, k, plan_cut, target_km):
    print("\nTests")
    results = []

    # Use a kick that hits the target when the timer cuts at plan_cut.
    print("\nTest 1: the onboard predictor must agree with the real simulation")
    print("        (predict from the state at cutoff, then compare with the actual landing)")
    r = fly_mode(base, NO_ERRORS, k, plan_cut)
    pred = predict_range_km(r["state_cut"], base)
    ok = abs(pred - r["range_km"]) < 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] predicted {pred:.3f} km, actual {r['range_km']:.3f} km, "
          f"difference {abs(pred - r['range_km']) * 1e3:.0f} m (limit 1000 m)")
    results.append(ok)

    print("\nTest 2: with no errors, closed loop must land on the target")
    c = fly_mode(base, NO_ERRORS, k, plan_cut, target_km, closed_loop=True)
    ok = abs(c["range_km"] - target_km) < 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] landed {c['range_km']:.3f} km, target {target_km:.1f} km, "
          f"miss {abs(c['range_km'] - target_km) * 1e3:.0f} m (limit 500 m); "
          f"cutoff at {c['t_cut']:.2f} s (timer plan: {plan_cut:.2f} s)")
    results.append(ok)

    print("\nTest 3: a STRONGER engine must make the controller cut EARLIER")
    p = dict(NO_ERRORS)
    p["thrust"] = +0.015
    strong = fly_mode(base, p, k, plan_cut, target_km, closed_loop=True)
    ok = strong["t_cut"] < c["t_cut"]
    print(f"  [{'PASS' if ok else 'FAIL'}] cutoff {strong['t_cut']:.2f} s "
          f"< nominal {c['t_cut']:.2f} s")
    results.append(ok)

    print("\nTest 4: a WEAKER engine must make it cut LATER (using the reserve)")
    p["thrust"] = -0.015
    weak = fly_mode(base, p, k, plan_cut, target_km, closed_loop=True)
    ok = weak["t_cut"] > c["t_cut"]
    print(f"  [{'PASS' if ok else 'FAIL'}] cutoff {weak['t_cut']:.2f} s "
          f"> nominal {c['t_cut']:.2f} s")
    results.append(ok)
    return all(results)


# ---------------------------------------------------------------------------
# Monte Carlo and error budget
# ---------------------------------------------------------------------------
def monte_carlo(base, kick_deg, plan_cut, target_km, samples, closed_loop):
    jobs = [(base, p, kick_deg, plan_cut, target_km, closed_loop) for p in samples]
    workers = max(1, (os.cpu_count() or 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_one, jobs, chunksize=4))
    except Exception:
        return [run_one(j) for j in jobs]


def stats_of(res, target_km):
    rng = np.array([r["range_km"] for r in res])
    ok = ~np.isnan(rng)
    rng = rng[ok]
    dev = rng - target_km
    t_cut = np.array([r["t_cut"] for r in res])[ok]
    return {
        "range": rng,
        "t_cut": t_cut,
        "mean": rng.mean(),
        "std": rng.std(ddof=1),
        "min": rng.min(),
        "max": rng.max(),
        "bias": rng.mean() - target_km,
        "within5": 100.0 * np.mean(np.abs(dev) <= 5.0),
        "within10": 100.0 * np.mean(np.abs(dev) <= 10.0),
        "within20": 100.0 * np.mean(np.abs(dev) <= 20.0),
        "saturated": int(sum(r["saturated"] for r in np.array(res, dtype=object)[ok])),
    }


def error_budget(base, kick_deg, plan_cut, target_km, closed_loop):
    nominal = fly_mode(base, NO_ERRORS, kick_deg, plan_cut, target_km, closed_loop)["range_km"]
    rows = {}
    for key, sigma, _, name in ERROR_SOURCES:
        effects = []
        for sign in (+1, -1):
            p = dict(NO_ERRORS)
            p[key] = sign * sigma
            r = fly_mode(base, p, kick_deg, plan_cut, target_km, closed_loop)["range_km"]
            effects.append(r - nominal)
        rows[name] = 0.5 * (abs(effects[0]) + abs(effects[1]))
    return rows


def print_histogram(arc_name, open_range, closed_range, target_km, bins=21):
    dev_o = np.asarray(open_range) - target_km
    dev_c = np.asarray(closed_range) - target_km
    span = float(np.ceil(max(np.abs(dev_o).max(), np.abs(dev_c).max()) / 5.0) * 5.0)
    span = max(span, 5.0)
    edges = np.linspace(-span, span, bins + 1)
    co = np.histogram(dev_o, edges)[0]
    cc = np.histogram(dev_c, edges)[0]
    scale = max(1, int(np.ceil(max(co.max(), cc.max()) / 28.0)))
    print(f"\n  {arc_name}: landing error histogram (each # = {scale} flight(s))")
    print(f"  {'error range [km]':>20}  {'OPEN loop':<30} {'CLOSED loop':<30}")
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        marker = "<- target" if lo <= 0.0 < hi else ""
        print(f"  {lo:+9.1f} to {hi:+8.1f}  {'#' * int(np.ceil(co[i] / scale)):<30} "
              f"{'#' * int(np.ceil(cc[i] / scale)):<30} {marker}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(results, samples, target_km):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    arc_names = list(results)

    for row, arc in enumerate(arc_names):
        data = results[arc]
        ax = axes[row, 0]
        lo = min(data["open"]["range"].min(), data["closed"]["range"].min()) - 2
        hi = max(data["open"]["range"].max(), data["closed"]["range"].max()) + 2
        bins = np.linspace(lo, hi, 35)
        ax.hist(data["open"]["range"], bins=bins, alpha=0.55, color="tab:red",
                label=f"open loop: spread {data['open']['std']:.1f} km")
        ax.hist(data["closed"]["range"], bins=bins, alpha=0.65, color="tab:green",
                label=f"closed loop: spread {data['closed']['std']:.1f} km")
        ax.axvline(target_km, color="k", ls="--")
        ax.set_title(f"{arc}: where did the rockets land?")
        ax.set_xlabel("landing distance [km downrange]")
        ax.set_ylabel("number of flights")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        thrust_err = np.array([s["thrust"] for s in samples]) * 100.0
        ok = ~np.isnan(np.array([r["range_km"] for r in data["closed_raw"]]))
        ax.scatter(thrust_err[ok], data["closed"]["t_cut"], s=16, color="tab:green")
        ax.axhline(data["plan_cut"], color="tab:red", ls="--", label="timer plan (open loop)")
        ax.set_title(f"{arc}: the controller reacts to the engine")
        ax.set_xlabel("engine thrust error [%]")
        ax.set_ylabel("time the engine was shut off [s]")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        names = list(data["budget_open"])
        xs = np.arange(len(names))
        w = 0.38
        ax.bar(xs - w / 2, [data["budget_open"][n] for n in names], w,
               color="tab:red", label="open loop")
        ax.bar(xs + w / 2, [data["budget_closed"][n] for n in names], w,
               color="tab:green", label="closed loop")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("landing change for a 1-sigma error [km]")
        ax.set_title(f"{arc}: which errors does feedback fix?")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 5: feedback (engine cutoff by prediction) vs a timer")
    fig.tight_layout()
    fig.savefig("phase5_feedback.png", dpi=150)
    print("\nSaved plot to phase5_feedback.png")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_km = float(pos[0]) if pos else DEFAULT_TARGET_KM
    n = DEFAULT_N
    for a in sys.argv[1:]:
        if a.startswith("--n="):
            n = int(a.split("=")[1])
    show_plot = "--no-plot" not in sys.argv

    base = Rocket(m0=10000.0, m_prop=7000.0, isp=280.0,
                  mass_flow=50.0, cd=0.3, area=1.767)
    plan_cut = (1.0 - RESERVE) * base.burn_time

    print("=" * 78)
    print(f"PHASE 5: CLOSING THE LOOP   target = {target_km:.0f} km")
    print("=" * 78)
    print(f"\nFull burn would last {base.burn_time:.1f} s. The plan keeps a "
          f"{RESERVE * 100:.0f}% reserve:")
    print(f"the timer cuts the engine at {plan_cut:.1f} s, leaving "
          f"{base.m_prop * RESERVE:.0f} kg of propellant unused.")
    print(f"The guidance computer runs every {GUIDANCE_DT:.0f} s with perfect sensors.")

    print("\nSolving for the planned kick angles (timer cutoff) ...")
    kicks, r_peak = solve_kicks(base, target_km, plan_cut)
    if kicks is None:
        print(f"{target_km:.0f} km is beyond the maximum range ({r_peak:.1f} km) "
              "with this plan. Pick a closer target.")
        sys.exit(2)
    for name, k in kicks.items():
        print(f"  {name:<20} kick = {k:.6f} deg")

    k_test = list(kicks.values())[0]
    if not run_tests(base, k_test, plan_cut, target_km):
        print("\nSOME TESTS FAILED. Do not trust the results below.")
        sys.exit(1)

    samples = draw_errors(n, SEED)
    print("\n" + "=" * 78)
    print(f"MONTE CARLO: the same {n} random error sets, flown open loop and closed loop")
    print("=" * 78)
    print("(This takes a few minutes: the closed-loop rocket runs its guidance")
    print(" computer every second of every flight. Several CPU cores make it faster.)")

    results = {}
    for name, k in kicks.items():
        print(f"\n  {name}: flying {n} x open loop and {n} x closed loop ...", flush=True)
        res_o = monte_carlo(base, k, plan_cut, target_km, samples, False)
        res_c = monte_carlo(base, k, plan_cut, target_km, samples, True)
        so, sc = stats_of(res_o, target_km), stats_of(res_c, target_km)
        print(f"  {name}: computing the error budget (one error at a time) ...", flush=True)
        results[name] = {
            "open": so, "closed": sc, "closed_raw": res_c, "plan_cut": plan_cut, "kick": k,
            "budget_open": error_budget(base, k, plan_cut, target_km, False),
            "budget_closed": error_budget(base, k, plan_cut, target_km, True),
        }

    for name, data in results.items():
        so, sc = data["open"], data["closed"]
        print("\n" + "-" * 78)
        print(f"{name}   (kick {data['kick']:.4f} deg)")
        print("-" * 78)
        print(f"{'':<30}{'OPEN loop (timer)':>22}{'CLOSED loop':>22}")
        rows = [
            ("Mean landing [km]", "mean", "{:.2f}"),
            ("Average miss (bias) [km]", "bias", "{:+.2f}"),
            ("Scatter, 1 sigma [km]", "std", "{:.2f}"),
            ("Closest landing [km]", "min", "{:.1f}"),
            ("Farthest landing [km]", "max", "{:.1f}"),
            ("Within 5 km of target [%]", "within5", "{:.0f}"),
            ("Within 10 km of target [%]", "within10", "{:.0f}"),
            ("Within 20 km of target [%]", "within20", "{:.0f}"),
        ]
        for label, key, fmt in rows:
            print(f"{label:<30}{fmt.format(so[key]):>22}{fmt.format(sc[key]):>22}")
        print(f"{'Engine cutoff time [s]':<30}{plan_cut:>22.1f}"
              f"{sc['t_cut'].mean():>14.1f} +/- {sc['t_cut'].std(ddof=1):.1f}")
        print(f"{'Flights that ran out of fuel':<30}{'-':>22}{sc['saturated']:>22d}")
        print(f"\n  Scatter reduced by a factor of {so['std'] / sc['std']:.1f}")

        print(f"\n  Error budget: landing change for a 1-sigma error [km]")
        print(f"  {'error source':<24}{'open loop':>12}{'closed loop':>14}")
        for src in data["budget_open"]:
            print(f"  {src:<24}{data['budget_open'][src]:>12.2f}"
                  f"{data['budget_closed'][src]:>14.2f}")

        print_histogram(name, so["range"], sc["range"], target_km)

    print("\n" + "=" * 78)
    print("WHAT TO TAKE FROM THIS")
    print("=" * 78)
    for name, data in results.items():
        left = max(data["budget_closed"].items(), key=lambda kv: kv[1])
        fixed = max(data["budget_open"].items(), key=lambda kv: kv[1])
        print(f"  * {name}: scatter {data['open']['std']:.1f} km -> {data['closed']['std']:.1f} km "
              f"(factor {data['open']['std'] / data['closed']['std']:.1f})")
        print(f"      biggest error before feedback: {fixed[0]} ({fixed[1]:.1f} km)")
        print(f"      biggest error AFTER feedback:  {left[0]} ({left[1]:.2f} km)")
        if data["closed"]["saturated"] > 0:
            print(f"      {data['closed']['saturated']} flight(s) used ALL the propellant and still fell "
                  "short: the reserve was not enough.")
    print()
    print("  IMPORTANT: these results are better than reality would be. This rocket has")
    print("  perfect sensors, an instant engine cutoff, and a predictor that matches the")
    print("  real physics except for drag and air density. Real sensors are noisy and")
    print("  real models are wrong in more ways. Handling noisy sensors is the next phase.")

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            fig = make_plot(results, samples, target_km)
            plt.show()
        except ImportError:
            print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")

    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()