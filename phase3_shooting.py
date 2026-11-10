"""
phase3_shooting.py
Phase 3 of the rocket simulation: guidance by "shooting".

You choose WHERE the rocket should land. The computer finds the kick angle
that makes it land there, by flying the rocket over and over and adjusting
the kick until the miss distance is a few metres.

Put this file in the SAME folder as phase2_gravity_turn.py, then run:

    python3 phase3_shooting.py              (default target: 400 km)
    python3 phase3_shooting.py 300          (target 300 km)
    python3 phase3_shooting.py 300 --no-plot   (text only, no plot window)

How it works
------------
1. SCAN:   fly with many kick angles and record where each one lands.
           The landing distance rises, peaks, then falls, so a target can
           have TWO solutions (a steep high arc and a flat low arc), and
           anything beyond the peak has NO solution.
2. PEAK:   locate the best kick (the farthest this rocket can reach).
3. BISECT: on each side of the peak, repeatedly cut the kick interval in
           half, keeping the half that still contains the target.
4. CHECK:  re-fly each answer with a finer time step to confirm it lands
           where the search said it would.

Units: SI inside the physics, kilometres and degrees in the printouts.
"""

import sys

import numpy as np

from phase2_gravity_turn import Rocket, air_density, fly

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_TARGET_KM = 400.0
KICK_MIN = 0.001        # [deg] smallest kick the search will try
SCAN_MAX = 2.4          # [deg] largest kick the scan will try
SCAN_STEP = 0.05        # [deg] spacing of the scan
DT_SEARCH = 0.10        # [s] time step for the scan and the peak search (fast)
DT_FINE = 0.05          # [s] time step for the bisection (accurate)
DT_CHECK = 0.025        # [s] time step for the final independent check
TOLERANCE_KM = 0.005    # [km] stop bisecting when the miss is below 5 metres


# ---------------------------------------------------------------------------
# One flight -> landing distance
# ---------------------------------------------------------------------------
def flight(rocket, kick_deg, dt):
    """Fly once. Returns (ts, ys, events, crashed).

    'crashed' means the rocket went below the ground while the engine was
    still burning (the kick was so large that it ploughed into the ground),
    which is not a valid flight.
    """
    ts, ys, ev = fly(rocket, kick_deg=kick_deg, dt=dt)
    i_b = int(np.argmin(np.abs(ts - ev["burnout"])))
    crashed = bool(ys[: i_b + 1, 0].min() < -1.0)
    return ts, ys, ev, crashed


def landing_km(rocket, kick_deg, dt):
    """Landing distance in km, or NaN if the flight is not valid."""
    _, ys, _, crashed = flight(rocket, kick_deg, dt)
    return float("nan") if crashed else ys[-1, 3] / 1e3


# ---------------------------------------------------------------------------
# Search steps
# ---------------------------------------------------------------------------
def scan(rocket):
    kicks = np.arange(SCAN_STEP, SCAN_MAX + 1e-9, SCAN_STEP)
    ranges = np.array([landing_km(rocket, k, DT_SEARCH) for k in kicks])
    return kicks, ranges


def find_peak(rocket, kicks, ranges):
    """Golden-section search for the kick that lands farthest."""
    valid = np.where(~np.isnan(ranges))[0]
    i = valid[int(np.argmax(ranges[valid]))]
    a = kicks[max(i - 1, 0)]
    b = kicks[min(i + 1, len(kicks) - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = landing_km(rocket, c, DT_SEARCH)
    fd = landing_km(rocket, d, DT_SEARCH)
    for _ in range(18):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = landing_km(rocket, c, DT_SEARCH)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = landing_km(rocket, d, DT_SEARCH)
    k_best = 0.5 * (a + b)
    return k_best, landing_km(rocket, k_best, DT_FINE)


def bisect(rocket, target_km, k_a, k_b, label):
    """Halve the interval [k_a, k_b] until the landing is within tolerance.

    Works whether the range rises or falls across the interval: it only needs
    the miss (landing - target) to have opposite signs at the two ends.
    """
    f_a = landing_km(rocket, k_a, DT_FINE) - target_km
    f_b = landing_km(rocket, k_b, DT_FINE) - target_km
    if np.isnan(f_a) or np.isnan(f_b) or f_a * f_b > 0:
        return None

    print(f"\n  Searching the {label} side: kick between "
          f"{k_a:.4f} and {k_b:.4f} deg")
    print(f"  {'try':>4} {'kick [deg]':>12} {'lands at [km]':>14} {'miss [m]':>12}")
    for it in range(1, 61):
        k_mid = 0.5 * (k_a + k_b)
        f_mid = landing_km(rocket, k_mid, DT_FINE) - target_km
        print(f"  {it:>4} {k_mid:12.6f} {f_mid + target_km:14.3f} {f_mid * 1e3:+12.1f}")
        if abs(f_mid) < TOLERANCE_KM:
            return k_mid
        if f_a * f_mid < 0:
            k_b = k_mid
        else:
            k_a, f_a = k_mid, f_mid
    return 0.5 * (k_a + k_b)


# ---------------------------------------------------------------------------
# Summary of a finished solution
# ---------------------------------------------------------------------------
def describe(rocket, kick_deg, target_km):
    ts, ys, ev, _ = flight(rocket, kick_deg, DT_FINE)
    h, v, gam, x, m = ys.T
    q = 0.5 * air_density(h) * v**2
    i_b = int(np.argmin(np.abs(ts - ev["burnout"])))
    i_q = int(np.argmax(q[: i_b + 1]))
    i_a = int(np.argmax(h))

    # Independent check: same kick, half the time step.
    check_km = landing_km(rocket, kick_deg, DT_CHECK)

    return {
        "kick": kick_deg,
        "ts": ts, "h": h, "v": v, "gam": gam, "x": x,
        "ev": ev, "i_b": i_b, "i_a": i_a,
        "maxq_kpa": q[i_q] / 1e3,
        "burn_h_km": h[i_b] / 1e3,
        "burn_v": v[i_b],
        "burn_gam": np.degrees(gam[i_b]),
        "apex_km": h[i_a] / 1e3,
        "impact_km": x[-1] / 1e3,
        "miss_m": (x[-1] / 1e3 - target_km) * 1e3,
        "t_impact": ev["impact"],
        "v_impact": v[-1],
        "check_km": check_km,
        "check_diff_m": (check_km - x[-1] / 1e3) * 1e3,
    }


def print_solution_table(solutions, target_km):
    print("\n" + "=" * 78)
    print(f"RESULT: the rocket lands {target_km:.1f} km downrange")
    print("=" * 78)
    names = [s["name"] for s in solutions]
    print(f"{'':<26}" + "".join(f"{n:>26}" for n in names))
    print("-" * (26 + 26 * len(solutions)))
    rows = [
        ("Kick angle [deg]", "kick", "{:.6f}"),
        ("Burnout altitude [km]", "burn_h_km", "{:.1f}"),
        ("Burnout speed [m/s]", "burn_v", "{:.0f}"),
        ("Burnout angle [deg]", "burn_gam", "{:.1f}"),
        ("Max-Q [kPa]", "maxq_kpa", "{:.1f}"),
        ("Apex [km]", "apex_km", "{:.1f}"),
        ("Lands at [km]", "impact_km", "{:.3f}"),
        ("Miss [m]", "miss_m", "{:+.1f}"),
        ("Flight time [s]", "t_impact", "{:.0f}"),
        ("Arrival speed [m/s]", "v_impact", "{:.0f}"),
        ("Check at dt/2, lands [km]", "check_km", "{:.3f}"),
        ("Check differs by [m]", "check_diff_m", "{:+.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<26}" + "".join(f"{fmt.format(s[key]):>26}" for s in solutions))


def print_dots_table(solution):
    """Speed and position every 40 s, so the speed is visible as numbers."""
    ts, v, h, x = solution["ts"], solution["v"], solution["h"], solution["x"]
    print(f"\n  {solution['name']} (kick {solution['kick']:.4f} deg)")
    print(f"  {'t [s]':>6} {'downrange [km]':>15} {'altitude [km]':>14} {'speed [m/s]':>12}")
    t = 0.0
    while t <= ts[-1]:
        print(f"  {t:6.0f} {np.interp(t, ts, x) / 1e3:15.1f} "
              f"{np.interp(t, ts, h) / 1e3:14.1f} {np.interp(t, ts, v):12.0f}")
        t += 40.0


# ---------------------------------------------------------------------------
# Plot: flight paths with a dot every 10 s (dot spacing = speed)
# ---------------------------------------------------------------------------
def make_plot(solutions, target_km):
    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:orange"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})

    for sol, col in zip(solutions, colors):
        ts, x, h, v = sol["ts"], sol["x"], sol["h"], sol["v"]
        ax1.plot(x / 1e3, h / 1e3, color=col, lw=2,
                 label=f"{sol['name']}: kick {sol['kick']:.3f} deg, "
                       f"apex {sol['apex_km']:.0f} km")
        t_dots = np.arange(0.0, ts[-1], 10.0)
        ax1.plot(np.interp(t_dots, ts, x) / 1e3, np.interp(t_dots, ts, h) / 1e3,
                 "o", color=col, ms=4)
        ax2.plot(ts, v, color=col, lw=2, label=sol["name"])
        ax2.axvline(sol["ev"]["burnout"], color=col, ls="--", alpha=0.5)

    ax1.axvline(target_km, color="red", ls="--", label=f"target: {target_km:.0f} km")
    ax1.plot([target_km], [0], "rx", ms=12, mew=3)
    ax1.axhline(0, color="green", lw=2)
    ax1.set_xlabel("downrange distance [km]")
    ax1.set_ylabel("altitude [km]")
    ax1.set_title("Both solutions hit the same target (one dot = 10 s of flight;\n"
                  "dots far apart = fast, dots close together = slow)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("speed [m/s]")
    ax2.set_title("Speed over time (dashed = burnout)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase 3: shooting method")
    fig.tight_layout()
    fig.savefig("phase3_shoot.png", dpi=150)
    print("\nSaved plot to phase3_shoot.png")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_plot = "--no-plot" not in sys.argv
    target_km = float(args[0]) if args else DEFAULT_TARGET_KM
    if target_km <= 0:
        print("The target must be a positive distance in km.")
        sys.exit(1)

    rocket = Rocket(m0=10000.0, m_prop=7000.0, isp=280.0,
                    mass_flow=50.0, cd=0.3, area=1.767)

    print("=" * 78)
    print(f"PHASE 3: SHOOTING METHOD   target = {target_km:.1f} km downrange")
    print("=" * 78)

    # 1) Scan
    print("\nStep 1: scan. Where does the rocket land for each kick angle?")
    print("        (each # = 10 km of landing distance; blank = not a valid flight)\n")
    kicks, ranges = scan(rocket)
    print(f"  {'kick [deg]':>10} {'lands at [km]':>14}")
    for k, r in zip(kicks, ranges):
        if np.isnan(r):
            print(f"  {k:10.2f} {'invalid':>14}   (rocket hit the ground during the burn)")
        else:
            print(f"  {k:10.2f} {r:14.1f}   " + "#" * int(r / 10))

    # 2) Peak
    print("\nStep 2: find the kick that lands farthest.")
    k_peak, r_peak = find_peak(rocket, kicks, ranges)
    print(f"  Best kick = {k_peak:.4f} deg  ->  maximum range = {r_peak:.1f} km")

    if target_km > r_peak:
        print("\n" + "=" * 78)
        print(f"IMPOSSIBLE: {target_km:.1f} km is farther than this rocket can reach.")
        print(f"The best possible landing is {r_peak:.1f} km (kick {k_peak:.3f} deg).")
        print("No kick angle can fix that: the rocket simply does not have enough")
        print("speed at burnout. A bigger rocket or a second stage is needed.")
        print("=" * 78)
        sys.exit(2)

    # 3) Bisect on each side of the peak
    print("\nStep 3: bisection. Halve the kick interval until the miss is under "
          f"{TOLERANCE_KM * 1e3:.0f} m.")
    k_valid_max = kicks[~np.isnan(ranges)].max()
    k_steep = bisect(rocket, target_km, KICK_MIN, k_peak, "steep / high-arc")
    k_flat = bisect(rocket, target_km, k_peak, k_valid_max, "flat / low-arc")

    solutions = []
    if k_steep is not None:
        s = describe(rocket, k_steep, target_km)
        s["name"] = "steep (high arc)"
        solutions.append(s)
    if k_flat is not None:
        s = describe(rocket, k_flat, target_km)
        s["name"] = "flat (low arc)"
        solutions.append(s)

    if not solutions:
        print("\nNo solution found in the search range (the target may be too close).")
        sys.exit(3)
    if len(solutions) == 1:
        print(f"\nOnly one solution exists for {target_km:.1f} km "
              f"({solutions[0]['name']}); the other side cannot reach a target "
              "that close with a valid flight.")

    # 4) Results
    print_solution_table(solutions, target_km)

    print("\nPosition and speed over time:")
    for s in solutions:
        print_dots_table(s)

    # Pass/fail verification
    ok = all(abs(s["miss_m"]) < 10.0 and abs(s["check_diff_m"]) < 500.0
             for s in solutions)
    print("\n" + "=" * 78)
    print("VERIFICATION: " + ("PASSED. Each answer lands within 10 m of the target, "
                              "and halving the time step moves it by under 500 m."
                              if ok else
                              "FAILED. Do not trust these answers until you find out why."))
    print("=" * 78)

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            fig = make_plot(solutions, target_km)
            plt.show()
        except ImportError:
            print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")

    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()