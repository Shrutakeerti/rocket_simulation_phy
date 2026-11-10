"""
phase4_montecarlo.py
Phase 4 of the rocket simulation: what happens when the real world is not
exactly what we planned for?

Phase 3 found the kick angle that lands exactly on a target. That answer was
perfect only because the simulation had no errors. Here we add six realistic
errors and launch the SAME planned rocket many times with random errors:

    1. engine thrust           (the engine is a little stronger or weaker)
    2. propellant loaded       (the tanks are not filled exactly right)
    3. drag coefficient        (our drag model is a little off)
    4. air density             (the day is thicker or thinner than standard)
    5. wind                    (a head/tail wind, strongest near 11 km)
    6. kick pointing           (the kick angle is not executed exactly)

and we measure where the rockets land. We do this for BOTH answers from
Phase 3 (the steep high arc and the flat low arc) to see which one is more
robust.

Put this file in the SAME folder as phase2_gravity_turn.py and
phase3_shooting.py, then run:

    python3 phase4_montecarlo.py                 (target 400 km, 100 flights)
    python3 phase4_montecarlo.py 300             (target 300 km)
    python3 phase4_montecarlo.py 400 --n=200     (200 flights per arc)
    python3 phase4_montecarlo.py 400 --no-plot   (text only)

Limitation: the simulation is 2D (a vertical plane), so we can only measure
how far the landing spot is off in the DOWNRANGE direction. Sideways error
needs a 3D simulation, which comes later.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import phase3_shooting as p3
from phase2_gravity_turn import (KICK_SPEED, MU_EARTH, R_EARTH, Rocket,
                                 air_density, gravity, integrate_fixed,
                                 integrate_until, rk4_step)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_TARGET_KM = 400.0
DEFAULT_N = 100
SEED = 42
DT_MC = 0.10            # [s] time step for the Monte Carlo flights

# 1-sigma size of each error source: (key, sigma, unit text, description)
ERROR_SOURCES = [
    ("thrust", 0.015, "1.5 %", "engine thrust"),
    ("prop",   0.005, "0.5 %", "propellant loaded"),
    ("cd",     0.10,  "10 %",  "drag coefficient"),
    ("rho",    0.05,  "5 %",   "air density"),
    ("wind",   10.0,  "10 m/s", "wind (peak at 11 km)"),
    ("kick",   0.01,  "0.01 deg", "kick pointing"),
]
WIND_PEAK_H = 11e3      # [m] altitude of the strongest wind
WIND_WIDTH = 6e3        # [m] how wide the windy layer is


# ---------------------------------------------------------------------------
# Physics with errors
# ---------------------------------------------------------------------------
def make_rhs(rocket, rho_scale, wind_peak, burning, hold_vertical=False):
    """2D gravity-turn dynamics with wind and scaled air density.

    hold_vertical=True is used for the first seconds of flight: the rocket's
    attitude control holds it exactly vertical (a "pitch program"). This is
    needed because the turn equation contains g/v. At liftoff v is almost
    zero, so any tiny sideways push (even a 1 m/s breeze) would be amplified
    about 100,000,000,000 times and tip the rocket flat before it has any
    speed. A free gravity turn cannot be started from rest; real rockets
    hold vertical until they are fast enough, then apply the kick.

    Wind is a horizontal wind along the flight direction (positive = tailwind
    that pushes the rocket downrange). Drag acts against the velocity of the
    rocket RELATIVE TO THE AIR, so a wind makes drag point slightly sideways
    to the flight path. We split it into the part along the path (changes the
    speed) and the part across it (bends the path).

    With wind = 0 and rho_scale = 1 this is exactly the Phase 2 model.
    """

    def f(t, y):
        h, v, gam, x, m = y
        r = R_EARTH + h
        g = gravity(h)
        cg, sg = np.cos(gam), np.sin(gam)

        w = wind_peak * np.exp(-(((h - WIND_PEAK_H) / WIND_WIDTH) ** 2))
        vrx = v * cg - w                    # velocity relative to the air
        vry = v * sg
        vr = np.hypot(vrx, vry)
        k = 0.5 * rho_scale * air_density(h) * rocket.cd * rocket.area * vr
        d_x, d_y = -k * vrx, -k * vry       # drag force components
        d_along = d_x * cg + d_y * sg       # along the flight path
        d_across = -d_x * sg + d_y * cg     # across the flight path

        thrust = rocket.thrust if burning else 0.0
        dm = -rocket.mass_flow if burning else 0.0

        dh = v * sg
        dx = v * cg * R_EARTH / r
        dv = (thrust + d_along) / m - g * sg
        if hold_vertical:
            dgam = 0.0
        else:
            dgam = d_across / (m * v) + (v / r - g / v) * cg
        return np.array([dh, dv, dgam, dx, dm])

    return f


def perturbed_rocket(base, p):
    """A copy of the rocket with the thrust, propellant and drag errors applied."""
    prop = base.m_prop * (1.0 + p["prop"])
    return Rocket(
        m0=base.m_dry + prop,
        m_prop=prop,
        isp=base.isp * (1.0 + p["thrust"]),
        mass_flow=base.mass_flow,
        cd=base.cd * (1.0 + p["cd"]),
        area=base.area,
    )


def fly_with_errors(base, p, kick_deg, dt=DT_MC):
    """One complete flight with the errors in dict p. Returns a result dict.

    Same sequence as Phase 2: vertical rise (held vertical), kick, gravity
    turn, burnout, coast to the ground.
    """
    rocket = perturbed_rocket(base, p)
    rho_scale = 1.0 + p["rho"]
    f_rise = make_rhs(rocket, rho_scale, p["wind"], burning=True, hold_vertical=True)
    f_burn = make_rhs(rocket, rho_scale, p["wind"], burning=True)
    f_coast = make_rhs(rocket, rho_scale, p["wind"], burning=False)

    y0 = np.array([0.0, 1e-3, np.pi / 2.0, 0.0, rocket.m0])
    ts1, ys1 = integrate_until(f_rise, y0, 0.0, dt, lambda y: y[1] >= KICK_SPEED)

    y_kick = ys1[-1].copy()
    y_kick[2] -= np.radians(kick_deg + p["kick"])

    tb = rocket.burn_time
    t_kick = ts1[-1]
    n = max(1, int(round((tb - t_kick) / dt)))
    ts2, ys2 = integrate_fixed(f_burn, y_kick, t_kick, tb, n)

    crashed = bool(ys2[:, 0].min() < -1.0)
    ts3, ys3 = integrate_until(f_coast, ys2[-1], tb, dt, lambda y: y[0] <= 0.0)

    ts = np.concatenate([ts1, ts2, ts3[1:]])
    ys = np.vstack([ys1, ys2, ys3[1:]])
    return {
        "range_km": float("nan") if crashed else ys[-1, 3] / 1e3,
        "apex_km": ys[:, 0].max() / 1e3,
        "t_impact": ts[-1],
        "v_impact": ys[-1, 1],
    }


NO_ERRORS = {key: 0.0 for key, *_ in ERROR_SOURCES}


def run_one(args):
    """Top-level function so worker processes can run it."""
    base, p, kick_deg = args
    return fly_with_errors(base, p, kick_deg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def check(name, got, expected, tol):
    err = abs(got - expected)
    ok = err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.6f}, "
          f"expected {expected:.6f}, abs error {err:.3e}")
    return ok


def run_tests(base):
    print("\nTests (the error model must not change the physics when errors are zero)")
    results = []

    print("\nTest 1: with all errors = 0, this model must agree with the Phase 3 model")
    print("        (not exactly: Phase 3 starts at v = 0.001 m/s and its numerical")
    print("         noise is amplified about 1e11 times, worth a few hundred metres)")
    k = 0.4
    mine = fly_with_errors(base, NO_ERRORS, k, dt=0.1)["range_km"]
    ref = p3.landing_km(base, k, 0.1)
    results.append(check("landing distance [km]", mine, ref, 2.0))

    print("\nTest 2: wind only acts through drag (no air -> wind changes nothing)")
    p = dict(NO_ERRORS)
    p["rho"] = -1.0          # air density scaled to zero
    calm = fly_with_errors(base, p, k, dt=0.1)["range_km"]
    p["wind"] = 25.0
    windy = fly_with_errors(base, p, k, dt=0.1)["range_km"]
    results.append(check("landing distance [km]", windy, calm, 1e-9))

    print("\nTest 3: a tailwind must push the rocket farther, a headwind shorter")
    p_t = dict(NO_ERRORS)
    p_t["wind"] = 25.0
    p_h = dict(NO_ERRORS)
    p_h["wind"] = -25.0
    r0 = fly_with_errors(base, NO_ERRORS, k, dt=0.1)["range_km"]
    r_t = fly_with_errors(base, p_t, k, dt=0.1)["range_km"]
    r_h = fly_with_errors(base, p_h, k, dt=0.1)["range_km"]
    ok = r_h < r0 < r_t
    print(f"  [{'PASS' if ok else 'FAIL'}] headwind {r_h:.2f} km < calm {r0:.2f} km "
          f"< tailwind {r_t:.2f} km")
    results.append(ok)
    return all(results)


# ---------------------------------------------------------------------------
# Solve for the planned kick angles, using THIS model with no errors
# ---------------------------------------------------------------------------
def landing_nominal(base, kick_deg):
    r = fly_with_errors(base, NO_ERRORS, kick_deg)["range_km"]
    return r


def solve_kicks(base, target_km):
    """Scan, find the peak, then bisect on each side (same idea as Phase 3)."""
    scan_k = np.arange(0.1, 2.41, 0.1)
    scan_r = np.array([landing_nominal(base, k) for k in scan_k])
    i = int(np.nanargmax(scan_r))
    a, b = scan_k[max(i - 1, 0)], scan_k[min(i + 1, len(scan_k) - 1)]

    # golden-section search for the farthest-landing kick
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = landing_nominal(base, c), landing_nominal(base, d)
    for _ in range(14):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = landing_nominal(base, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = landing_nominal(base, d)
    k_peak = 0.5 * (a + b)
    r_peak = landing_nominal(base, k_peak)
    if target_km > r_peak:
        return None, r_peak

    def bisect(k_a, k_b):
        f_a = landing_nominal(base, k_a) - target_km
        f_b = landing_nominal(base, k_b) - target_km
        if np.isnan(f_a) or np.isnan(f_b) or f_a * f_b > 0:
            return None
        for _ in range(60):
            k_mid = 0.5 * (k_a + k_b)
            f_mid = landing_nominal(base, k_mid) - target_km
            if abs(f_mid) < 0.005:          # within 5 metres
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
# Error budget (one error at a time)
# ---------------------------------------------------------------------------
def error_budget(base, kick_deg):
    nominal = fly_with_errors(base, NO_ERRORS, kick_deg)["range_km"]
    rows = []
    for key, sigma, _, name in ERROR_SOURCES:
        p_plus = dict(NO_ERRORS)
        p_plus[key] = +sigma
        p_minus = dict(NO_ERRORS)
        p_minus[key] = -sigma
        up = fly_with_errors(base, p_plus, kick_deg)["range_km"] - nominal
        down = fly_with_errors(base, p_minus, kick_deg)["range_km"] - nominal
        rows.append((name, up, down, 0.5 * (abs(up) + abs(down))))
    rss = float(np.sqrt(sum(r[3] ** 2 for r in rows)))
    return nominal, rows, rss


def print_error_budget(name, kick, nominal, rows, rss):
    print(f"\n  {name}  (kick {kick:.4f} deg, nominal landing {nominal:.3f} km)")
    print(f"  {'error source':<24} {'+1 sigma [km]':>14} {'-1 sigma [km]':>14} "
          f"{'average [km]':>13}")
    for n, up, down, avg in sorted(rows, key=lambda r: -r[3]):
        print(f"  {n:<24} {up:+14.2f} {down:+14.2f} {avg:13.2f}")
    print(f"  Combined (square root of the sum of squares): {rss:.2f} km  "
          "<- predicted 1-sigma landing scatter")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def draw_errors(n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        samples.append({key: float(rng.normal(0.0, sigma))
                        for key, sigma, _, _ in ERROR_SOURCES})
    return samples


def monte_carlo(base, kick_deg, samples):
    jobs = [(base, p, kick_deg) for p in samples]
    workers = max(1, (os.cpu_count() or 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_one, jobs, chunksize=4))
    except Exception:
        results = [run_one(j) for j in jobs]
    return results


def text_histogram(data_by_name, target_km, bins=21):
    all_dev = np.concatenate([np.asarray(d) - target_km for d in data_by_name.values()])
    span = float(np.ceil(np.max(np.abs(all_dev)) / 5.0) * 5.0)
    span = max(span, 5.0)
    edges = np.linspace(-span, span, bins + 1)
    names = list(data_by_name)
    counts = {n: np.histogram(np.asarray(data_by_name[n]) - target_km, edges)[0]
              for n in names}
    scale = max(1, int(np.ceil(max(c.max() for c in counts.values()) / 30.0)))
    print(f"\n  Landing error histogram (each # = {scale} flight(s); "
          f"0 = exactly on target)")
    print(f"  {'error range [km]':>20}  " + "  ".join(f"{n[:5]:<32}" for n in names))
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        bars = "  ".join(f"{'#' * int(np.ceil(counts[n][i] / scale)):<32}" for n in names)
        marker = "<- target" if lo <= 0.0 < hi else ""
        print(f"  {lo:+9.1f} to {hi:+8.1f}  {bars} {marker}")


def summarize(name, kick, res, nominal_dt_km, target_km):
    rng = np.array([r["range_km"] for r in res])
    apex = np.array([r["apex_km"] for r in res])
    ok = ~np.isnan(rng)
    rng = rng[ok]
    dev = rng - target_km
    return {
        "name": name, "kick": kick, "range": rng, "apex": apex[ok],
        "mean": rng.mean(), "std": rng.std(ddof=1), "min": rng.min(), "max": rng.max(),
        "bias": rng.mean() - target_km,
        "within5": 100.0 * np.mean(np.abs(dev) <= 5.0),
        "within10": 100.0 * np.mean(np.abs(dev) <= 10.0),
        "within20": 100.0 * np.mean(np.abs(dev) <= 20.0),
        "failed": int((~ok).sum()),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(stats, budgets, target_km):
    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:orange"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    ax = axes[0]
    for s, col in zip(stats, colors):
        ax.scatter(s["range"], s["apex"], s=18, color=col, alpha=0.7, label=s["name"])
    ax.axvline(target_km, color="red", ls="--", label=f"target {target_km:.0f} km")
    ax.set_xlabel("where it landed [km downrange]")
    ax.set_ylabel("highest point of the flight [km]")
    ax.set_title("Every launch is one dot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    lo = min(s["range"].min() for s in stats) - 2
    hi = max(s["range"].max() for s in stats) + 2
    bins = np.linspace(lo, hi, 30)
    for s, col in zip(stats, colors):
        ax.hist(s["range"], bins=bins, alpha=0.55, color=col,
                label=f"{s['name']}: spread {s['std']:.1f} km")
    ax.axvline(target_km, color="red", ls="--")
    ax.set_xlabel("where it landed [km downrange]")
    ax.set_ylabel("number of flights")
    ax.set_title("How spread out are the landings?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    names = [r[0] for r in budgets[0][2]]
    width = 0.38
    xs = np.arange(len(names))
    for i, ((bname, _, rows, _), col) in enumerate(zip(budgets, colors)):
        ax.bar(xs + (i - 0.5) * width, [r[3] for r in rows], width,
               color=col, label=bname)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("landing change for a 1-sigma error [km]")
    ax.set_title("Which error hurts the most?")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 4: Monte Carlo of the planned flights")
    fig.tight_layout()
    fig.savefig("phase4_montecarlo.png", dpi=150)
    print("\nSaved plot to phase4_montecarlo.png")
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

    print("=" * 78)
    print(f"PHASE 4: WHAT IF THE REAL WORLD IS NOT PERFECT?   target = {target_km:.0f} km")
    print("=" * 78)
    print("\nAssumed 1-sigma errors (about 2 of every 3 flights stay inside these):")
    for _, _, size, name in ERROR_SOURCES:
        print(f"  {name:<24} +/- {size}")

    if not run_tests(base):
        print("\nSOME TESTS FAILED. Do not trust the results below.")
        sys.exit(1)

    print("\nSolving for the planned kick angles (Phase 3) ...")
    kicks, r_peak = solve_kicks(base, target_km)
    if kicks is None:
        print(f"{target_km:.0f} km is beyond the maximum range ({r_peak:.1f} km). "
              "Pick a closer target.")
        sys.exit(2)
    for name, k in kicks.items():
        print(f"  {name:<20} kick = {k:.6f} deg")

    print("\n" + "=" * 78)
    print("ERROR BUDGET: how far does each error alone move the landing?")
    print("=" * 78)
    budgets = []
    for name, k in kicks.items():
        nominal, rows, rss = error_budget(base, k)
        print_error_budget(name, k, nominal, rows, rss)
        budgets.append((name, k, rows, rss, nominal))

    print("\n" + "=" * 78)
    print(f"MONTE CARLO: {n} flights per arc, all errors random at the same time")
    print("=" * 78)
    samples = draw_errors(n, SEED)
    print("(Both arcs get the SAME random errors, so the comparison is fair.)")
    stats = []
    for (name, k, rows, rss, nominal) in budgets:
        print(f"  flying {n} x {name} ...", flush=True)
        res = monte_carlo(base, k, samples)
        stats.append(summarize(name, k, res, nominal, target_km))

    print(f"\n{'':<28}" + "".join(f"{s['name']:>26}" for s in stats))
    print("-" * (28 + 26 * len(stats)))
    rows = [
        ("Planned kick [deg]", "kick", "{:.4f}"),
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
        print(f"{label:<28}" + "".join(f"{fmt.format(s[key]):>26}" for s in stats))
    print(f"{'Predicted scatter [km]':<28}"
          + "".join(f"{b[3]:>26.2f}" for b in budgets)
          + "   <- from the error budget")

    text_histogram({s["name"]: s["range"] for s in stats}, target_km)

    print("\n" + "=" * 78)
    print("WHAT TO TAKE FROM THIS")
    print("=" * 78)
    best = min(stats, key=lambda s: s["std"])
    worst = max(stats, key=lambda s: s["std"])
    print(f"  * The planned kick was perfect on paper, yet the landings are spread over")
    print(f"    about +/- {3 * best['std']:.0f} km (3 sigma) even for the better arc.")
    print(f"  * The more robust arc here is the {best['name']}: scatter {best['std']:.1f} km")
    print(f"    vs {worst['std']:.1f} km for the {worst['name']}.")
    print("  * A guidance law that never looks at where the rocket actually is can")
    print("    only hope for good luck. Fixing that needs FEEDBACK (the next phase).")

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            fig = make_plot(stats, [(b[0], b[1], b[2], b[3]) for b in budgets], target_km)
            plt.show()
        except ImportError:
            print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")

    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()