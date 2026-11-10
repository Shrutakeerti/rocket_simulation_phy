"""
speed_check.py
Prints the rocket's speed every 20 seconds, so you can see exactly when it
speeds up and slows down. Text only, no plot windows.

Put this file in the SAME folder as phase2_gravity_turn.py, then run:

    python3 speed_check.py
"""

import numpy as np

from phase2_gravity_turn import KICK_ANGLE_DEG, Rocket, fly


def main():
    rocket = Rocket(m0=10000.0, m_prop=7000.0, isp=280.0,
                    mass_flow=50.0, cd=0.3, area=1.767)
    ts, ys, ev = fly(rocket)
    h, v, gam, x, m = ys.T

    print(f"Kick = {KICK_ANGLE_DEG} deg.  Speed of the rocket every 20 s of flight.")
    print("The bar is the speed: each # is 50 m/s.\n")
    print(f"{'t [s]':>6} {'speed [m/s]':>12} {'change in 20 s':>15} {'altitude [km]':>14}  engine")
    prev = None
    t = 0.0
    while t <= ts[-1]:
        speed = float(np.interp(t, ts, v))
        alt = float(np.interp(t, ts, h)) / 1e3
        engine = "ON " if t <= ev["burnout"] else "off"
        change = "" if prev is None else f"{speed - prev:+.0f}"
        print(f"{t:6.0f} {speed:12.0f} {change:>15} {alt:14.1f}  {engine}  " + "#" * int(speed / 50))
        prev = speed
        t += 20.0

    i_b = int(np.searchsorted(ts, ev["burnout"]))
    i_apex = int(np.argmax(h))
    i_min = i_b + int(np.argmin(v[i_b: i_apex + 1]))
    print(f"\nFastest during the burn:    {v[: i_b + 1].max():.0f} m/s (at burnout, t = {ev['burnout']:.0f} s)")
    print(f"Slowest on the way up:      {v[i_min]:.0f} m/s (t = {ts[i_min]:.0f} s, near the top of the arc)")
    print(f"Fastest on the way down:    {v[i_apex:].max():.0f} m/s")
    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()