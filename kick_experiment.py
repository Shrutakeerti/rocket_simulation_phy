"""
kick_experiment.py
Runs the Phase 2 gravity-turn flight for several kick angles and prints
everything as plain text in the terminal (no plot windows), including a small
character drawing of each flight path.

Put this file in the SAME folder as phase2_gravity_turn.py, then run:

    python3 kick_experiment.py

Then copy everything it prints and paste it into the chat.
"""

import numpy as np

from phase2_gravity_turn import KICK_SPEED, Rocket, air_density, fly

# Kick angles to compare (degrees). Edit this list if you want other values.
KICK_ANGLES = [0.3, 0.7, 1.0]

# Size of the character drawings.
COLS = 72
ROWS = 18


def summarize(rocket, kick_deg):
    ts, ys, ev = fly(rocket, kick_deg=kick_deg)
    h, v, gam, x, m = ys.T
    q = 0.5 * air_density(h) * v**2
    i_b = int(np.argmin(np.abs(ts - ev["burnout"])))
    i_q = int(np.argmax(q[: i_b + 1]))
    i_a = int(np.argmax(h))
    return {
        "kick_deg": kick_deg,
        "ts": ts, "h": h, "v": v, "gam": gam, "x": x,
        "t_kick": ev["kick"],
        "i_b": i_b, "i_a": i_a,
        "maxq_kpa": q[i_q] / 1e3,
        "t_maxq": ts[i_q],
        "h_burn_km": h[i_b] / 1e3,
        "x_burn_km": x[i_b] / 1e3,
        "v_burn": v[i_b],
        "gam_burn": np.degrees(gam[i_b]),
        "apex_km": h[i_a] / 1e3,
        "t_apex": ts[i_a],
        "impact_km": x[-1] / 1e3,
        "t_impact": ev["impact"],
        "v_impact": v[-1],
        "gam_impact": np.degrees(gam[-1]),
    }


def draw(run, x_max_km, h_max_km):
    """Draw the flight path as characters, same axes for every run.

    Symbols:  .  path     K  kick     B  burnout     A  apex     X  impact
    """
    grid = [[" "] * COLS for _ in range(ROWS)]

    def put(xkm, hkm, ch):
        c = int(round(xkm / x_max_km * (COLS - 1)))
        r = ROWS - 1 - int(round(hkm / h_max_km * (ROWS - 1)))
        if 0 <= c < COLS and 0 <= r < ROWS:
            grid[r][c] = ch

    x, h, ts = run["x"], run["h"], run["ts"]
    for xi, hi in zip(x, h):
        put(xi / 1e3, max(hi, 0.0) / 1e3, ".")
    i_k = int(np.argmin(np.abs(ts - run["t_kick"])))
    put(x[i_k] / 1e3, h[i_k] / 1e3, "K")
    put(x[run["i_b"]] / 1e3, h[run["i_b"]] / 1e3, "B")
    put(x[run["i_a"]] / 1e3, h[run["i_a"]] / 1e3, "A")
    put(x[-1] / 1e3, 0.0, "X")

    print(f"\n  Flight path, kick = {run['kick_deg']} deg   "
          f"(K=kick  B=burnout  A=apex  X=impact)")
    print(f"  altitude axis: 0 to {h_max_km:.0f} km (bottom to top)")
    for r in range(ROWS):
        print("  |" + "".join(grid[r]))
    print("  +" + "-" * COLS)
    print(f"   downrange axis: 0 to {x_max_km:.0f} km (left to right)")


def main():
    rocket = Rocket(
        m0=10000.0,
        m_prop=7000.0,
        isp=280.0,
        mass_flow=50.0,
        cd=0.3,
        area=1.767,
    )

    runs = [summarize(rocket, k) for k in KICK_ANGLES]

    print("=" * 78)
    print("KICK ANGLE EXPERIMENT   (same rocket, same engine, only the kick changes)")
    print(f"Kick happens when speed reaches {KICK_SPEED:.0f} m/s")
    print("=" * 78)

    # Side-by-side table
    header = f"{'kick [deg]':<16}" + "".join(f"{r['kick_deg']:>12.2f}" for r in runs)
    print("\n" + header)
    print("-" * len(header))
    rows = [
        ("Max-Q [kPa]", "maxq_kpa", "{:12.1f}"),
        ("Burnout h [km]", "h_burn_km", "{:12.1f}"),
        ("Burnout x [km]", "x_burn_km", "{:12.1f}"),
        ("Burnout v [m/s]", "v_burn", "{:12.0f}"),
        ("Burnout angle", "gam_burn", "{:12.1f}"),
        ("Apex [km]", "apex_km", "{:12.1f}"),
        ("Apex time [s]", "t_apex", "{:12.0f}"),
        ("Impact [km]", "impact_km", "{:12.1f}"),
        ("Impact time [s]", "t_impact", "{:12.0f}"),
        ("Impact speed", "v_impact", "{:12.0f}"),
        ("Impact angle", "gam_impact", "{:12.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<16}" + "".join(fmt.format(r[key]) for r in runs))
    print("(angles in degrees above the horizon; negative = pointing downward)")

    # Character drawings on common axes so they can be compared directly
    x_max = max(r["impact_km"] for r in runs) * 1.03
    h_max = max(r["apex_km"] for r in runs) * 1.05
    print("\n" + "=" * 78)
    print("DRAWINGS (all use the same axes, so you can compare them directly)")
    print("=" * 78)
    for r in runs:
        draw(r, x_max, h_max)

    # Flight-path angle over time, every 40 s, as a compact text table
    print("\n" + "=" * 78)
    print("FLIGHT-PATH ANGLE OVER TIME (degrees above horizon; 90 = straight up)")
    print("=" * 78)
    t_samples = np.arange(0, 481, 40)
    print(f"{'time [s]':>10}" + "".join(f"{('kick ' + str(r['kick_deg'])):>14}" for r in runs))
    for t in t_samples:
        line = f"{t:>10.0f}"
        for r in runs:
            if t > r["ts"][-1]:
                line += f"{'landed':>14}"
            else:
                g = np.degrees(np.interp(t, r["ts"], r["gam"]))
                line += f"{g:>14.1f}"
        print(line)

    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()
