"""
phase2_gravity_turn.py
Phase 2 of the rocket simulation: a 2D launch with a gravity turn.

WHAT YOU ARE WATCHING
---------------------
The rocket lifts off straight up. After a few seconds it gets one tiny sideways
nudge (the "kick"). After that the engine always pushes exactly along the
direction the rocket is already moving, and NOTHING ELSE steers it. Gravity
slowly pulls the flight path over toward the horizon on its own. That bending
is the "gravity turn". The rocket tips from vertical toward the horizontal
without any steering at all, and the size of the first kick decides how far.

State:  y = [h, v, gamma, x, m]
    h      altitude above the surface [m]
    v      speed (along the flight path) [m/s]
    gamma  flight-path angle above the local horizontal [rad]
           (90 deg = straight up, 0 deg = flying level)
    x      downrange distance along the surface [m]
    m      mass [kg]

Equations (round, non-rotating Earth, thrust along the velocity):
    h'     = v sin(gamma)
    x'     = v cos(gamma) * R / (R + h)
    v'     = (T - D)/m - g sin(gamma)
    gamma' = (v/r - g/v) cos(gamma)         with r = R + h
    m'     = -mass_flow

The v/r term is "centrifugal relief": the faster you fly, the less gravity
bends the path, because the Earth's surface curves away beneath you.

Units: SI everywhere (metres, seconds, kilograms, newtons, radians inside).
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G0 = 9.80665                # standard gravity [m/s^2]
MU_EARTH = 3.986004418e14   # Earth's gravitational parameter [m^3/s^2]
R_EARTH = 6.371e6           # mean Earth radius [m]
RHO0 = 1.225                # sea-level air density [kg/m^3]
H_SCALE = 8500.0            # atmosphere scale height [m]

# Settings you can play with ------------------------------------------------
KICK_SPEED = 50.0           # [m/s] fly straight up until this speed, then kick
KICK_ANGLE_DEG = 0.7        # [deg] size of the kick (try 0.3, 0.5, 1.0, 1.5)
ANIMATE = True              # play the animation window after the static plot
DT = 0.05                   # integration time step [s]


# ---------------------------------------------------------------------------
# Rocket definition
# ---------------------------------------------------------------------------
@dataclass
class Rocket:
    m0: float           # initial (wet) mass [kg]
    m_prop: float       # propellant mass [kg]
    isp: float          # specific impulse [s]
    mass_flow: float    # propellant mass flow rate [kg/s]
    cd: float           # drag coefficient
    area: float         # reference area [m^2]

    @property
    def m_dry(self):
        return self.m0 - self.m_prop

    @property
    def v_exhaust(self):
        return self.isp * G0

    @property
    def burn_time(self):
        return self.m_prop / self.mass_flow

    @property
    def thrust(self):
        return self.v_exhaust * self.mass_flow


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def gravity(h):
    return MU_EARTH / (R_EARTH + h) ** 2


def air_density(h):
    return RHO0 * np.exp(-np.maximum(h, 0.0) / H_SCALE)


def make_rhs_2d(rocket, burning=True, drag=True):
    """Right-hand side f(t, y) for the 2D gravity-turn equations."""

    def f(t, y):
        h, v, gam, x, m = y
        r = R_EARTH + h
        g = gravity(h)
        thrust = rocket.thrust if burning else 0.0
        dm = -rocket.mass_flow if burning else 0.0
        d = 0.5 * air_density(h) * rocket.cd * rocket.area * v * abs(v) if drag else 0.0

        dh = v * np.sin(gam)
        dx = v * np.cos(gam) * R_EARTH / r
        dv = (thrust - d) / m - g * np.sin(gam)
        dgam = (v / r - g / v) * np.cos(gam)
        return np.array([dh, dv, dgam, dx, dm])

    return f


def make_rhs_1d(rocket):
    """The Phase 1 vertical model, used as a regression check."""

    def f(t, y):
        h, v, m = y
        d = 0.5 * air_density(h) * rocket.cd * rocket.area * v * abs(v)
        return np.array([v, (rocket.thrust - d) / m - gravity(h), -rocket.mass_flow])

    return f


# ---------------------------------------------------------------------------
# Integrator (the RK4 you validated in Step 1)
# ---------------------------------------------------------------------------
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_fixed(f, y0, t0, t1, n_steps):
    """Integrate from t0 to t1 in exactly n_steps equal steps."""
    dt = (t1 - t0) / n_steps
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, len(y0)))
    ts[0], ys[0] = t0, y0
    y = np.array(y0, dtype=float)
    for i in range(n_steps):
        y = rk4_step(f, t0 + i * dt, y, dt)
        ts[i + 1], ys[i + 1] = t0 + (i + 1) * dt, y
    return ts, ys


def integrate_until(f, y0, t0, dt, stop, t_max=20000.0):
    """Integrate until stop(y) becomes True, then bisect to the exact moment.

    Used for events: "speed reaches the kick speed" and "rocket hits the ground".
    """
    y = np.array(y0, dtype=float)
    t = t0
    ts, ys = [t], [y.copy()]
    while t < t_max:
        y_new = rk4_step(f, t, y, dt)
        if stop(y_new):
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if stop(rk4_step(f, t, y, mid)):
                    hi = mid
                else:
                    lo = mid
            s = hi
            y_end = rk4_step(f, t, y, s)
            ts.append(t + s)
            ys.append(y_end)
            return np.array(ts), np.array(ys)
        t += dt
        y = y_new
        ts.append(t)
        ys.append(y.copy())
    raise RuntimeError("event not reached before t_max")


# ---------------------------------------------------------------------------
# The flight
# ---------------------------------------------------------------------------
def fly(rocket, kick_speed=KICK_SPEED, kick_deg=KICK_ANGLE_DEG, dt=DT,
        coast_to_ground=True):
    """Vertical rise -> kick -> gravity turn -> burnout -> coast to impact.

    Returns time array, state array and a dict of event times.
    """
    f_burn = make_rhs_2d(rocket, burning=True)
    f_coast = make_rhs_2d(rocket, burning=False)

    # 1) Vertical rise until the kick speed is reached.
    #    (Gamma' is exactly zero at gamma = 90 deg, so this is plain vertical flight.)
    y0 = np.array([0.0, 1e-3, np.pi / 2.0, 0.0, rocket.m0])
    ts1, ys1 = integrate_until(f_burn, y0, 0.0, dt, lambda y: y[1] >= kick_speed)
    t_kick = ts1[-1]

    # 2) The kick: one tiny instantaneous tilt of the velocity direction.
    y_kick = ys1[-1].copy()
    y_kick[2] -= np.radians(kick_deg)

    # 3) Gravity turn until burnout.
    tb = rocket.burn_time
    n = max(1, int(round((tb - t_kick) / dt)))
    ts2, ys2 = integrate_fixed(f_burn, y_kick, t_kick, tb, n)

    ts = np.concatenate([ts1, ts2])
    ys = np.vstack([ys1, ys2])
    events = {"kick": t_kick, "burnout": tb}

    # 4) Coast until the rocket falls back to the ground.
    if coast_to_ground:
        ts3, ys3 = integrate_until(f_coast, ys2[-1], tb, dt, lambda y: y[0] <= 0.0)
        ts = np.concatenate([ts, ts3[1:]])
        ys = np.vstack([ys, ys3[1:]])
        events["impact"] = ts3[-1]

    return ts, ys, events


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def check(name, got, expected, tol, relative=False):
    err = abs(got - expected)
    if relative:
        err /= abs(expected)
    ok = err <= tol
    kind = "rel" if relative else "abs"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.9g}, "
          f"expected {expected:.9g}, {kind} error {err:.3e}")
    return ok


def test_reduces_to_phase1(rocket):
    """With no kick the 2D code must reproduce the Phase 1 vertical flight."""
    print("\nTest 1: with kick = 0 the 2D model must match the Phase 1 model")
    ts, ys, ev = fly(rocket, kick_deg=0.0, coast_to_ground=False)
    f1 = make_rhs_1d(rocket)
    tb = rocket.burn_time
    _, ys1 = integrate_fixed(f1, [0.0, 0.0, rocket.m0], 0.0, tb, int(round(tb / DT)))
    ok1 = check("burnout altitude [m]", ys[-1, 0], ys1[-1, 0], 1.0)
    ok2 = check("burnout speed [m/s]", ys[-1, 1], ys1[-1, 1], 0.05)
    # Downrange should be exactly 0. It comes out as a fraction of a metre
    # because the flight starts at v = 0.001 m/s: the gamma equation divides by
    # v and multiplies by cos(90 deg) = 6e-17, leaving tiny numerical noise.
    ok3 = check("downrange [m]", ys[-1, 3], 0.0, 2.0)
    return ok1 and ok2 and ok3


def test_conservation(rocket):
    """No thrust, no drag: energy and angular momentum must be conserved."""
    print("\nTest 2: coasting without drag conserves energy and angular momentum")
    f = make_rhs_2d(rocket, burning=False, drag=False)
    y0 = [100e3, 3000.0, np.radians(30.0), 0.0, rocket.m_dry]
    _, ys = integrate_until(f, y0, 0.0, 0.1, lambda y: y[0] <= 0.0)
    h, v, gam = ys[:, 0], ys[:, 1], ys[:, 2]
    r = R_EARTH + h
    energy = 0.5 * v**2 - MU_EARTH / r
    ang_mom = r * v * np.cos(gam)
    d_e = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    d_l = np.max(np.abs(ang_mom - ang_mom[0])) / abs(ang_mom[0])
    ok1 = d_e < 1e-9
    ok2 = d_l < 1e-9
    print(f"  [{'PASS' if ok1 else 'FAIL'}] max relative energy drift:           {d_e:.3e}")
    print(f"  [{'PASS' if ok2 else 'FAIL'}] max relative angular-momentum drift: {d_l:.3e}")
    return ok1 and ok2


def test_step_size(rocket):
    """Halving dt must barely change the answer."""
    print("\nTest 3: the answer must not depend on the time step")
    _, ya, _ = fly(rocket, dt=0.05)
    _, yb, _ = fly(rocket, dt=0.025)
    ok1 = check("apex altitude [m]", np.max(ya[:, 0]), np.max(yb[:, 0]), 1e-3, relative=True)
    ok2 = check("impact downrange [m]", ya[-1, 3], yb[-1, 3], 1e-3, relative=True)
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Report, plots, animation
# ---------------------------------------------------------------------------
def report(rocket, ts, ys, ev):
    h, v, gam, x, m = ys.T
    q = 0.5 * air_density(h) * v**2
    i_b = int(np.argmin(np.abs(ts - ev["burnout"])))
    i_q = int(np.argmax(q[: i_b + 1]))      # Max-Q during the climb only
    i_apex = int(np.argmax(h))
    v_circ = np.sqrt(MU_EARTH / (R_EARTH + h[i_b]))

    print("\n" + "=" * 70)
    print(f"GRAVITY-TURN FLIGHT   (kick: {KICK_ANGLE_DEG} deg at {KICK_SPEED:.0f} m/s)")
    print("=" * 70)
    print(f"Kick at:     t = {ev['kick']:.1f} s")
    print(f"Max-Q (climb): {q[i_q] / 1e3:.2f} kPa at t = {ts[i_q]:.1f} s, "
          f"h = {h[i_q] / 1e3:.1f} km, flight-path angle = {np.degrees(gam[i_q]):.1f} deg")
    print(f"Burnout:     t = {ts[i_b]:.1f} s, h = {h[i_b] / 1e3:.1f} km, "
          f"downrange = {x[i_b] / 1e3:.1f} km")
    print(f"             v = {v[i_b]:.0f} m/s, flight-path angle = {np.degrees(gam[i_b]):.1f} deg")
    print(f"             speed is {100 * v[i_b] / v_circ:.0f}% of circular orbital speed "
          f"at that altitude ({v_circ:.0f} m/s)")
    print(f"Apex:        t = {ts[i_apex]:.1f} s, h = {h[i_apex] / 1e3:.1f} km")
    print(f"Impact:      t = {ev['impact']:.1f} s, downrange = {x[-1] / 1e3:.1f} km, "
          f"arrival speed = {v[-1]:.0f} m/s")
    print("-" * 70)
    print("This rocket cannot reach orbit by itself: it is one small stage with")
    print("about 3300 m/s of ideal delta-v, and orbit needs about 9400 m/s")
    print("including losses. Staging comes later.")


def make_static_plot(rocket, ts, ys, ev):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    h, v, gam, x, m = ys.T
    q = 0.5 * air_density(h) * v**2
    i_b = int(np.argmin(np.abs(ts - ev["burnout"])))

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])
    ax_traj = fig.add_subplot(gs[:, 0])

    # Trajectory coloured by speed, drawn to true scale in km.
    pts = np.column_stack([x / 1e3, h / 1e3]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", linewidth=3)
    lc.set_array(v[:-1])
    ax_traj.add_collection(lc)
    ax_traj.autoscale()
    ax_traj.set_aspect("equal", adjustable="datalim")
    fig.colorbar(lc, ax=ax_traj, label="speed [m/s]", fraction=0.04, pad=0.02)

    # Mark the key events.
    for name, t_ev, marker in [("kick", ev["kick"], "o"),
                               ("burnout", ev["burnout"], "s")]:
        i = int(np.argmin(np.abs(ts - t_ev)))
        ax_traj.plot(x[i] / 1e3, h[i] / 1e3, marker, color="k", ms=8)
        ax_traj.annotate(name, (x[i] / 1e3, h[i] / 1e3), textcoords="offset points",
                         xytext=(10, 8))
    i_a = int(np.argmax(h))
    ax_traj.plot(x[i_a] / 1e3, h[i_a] / 1e3, "^", color="k", ms=8)
    ax_traj.annotate("apex", (x[i_a] / 1e3, h[i_a] / 1e3), textcoords="offset points",
                     xytext=(10, 8))
    ax_traj.axhline(0, color="green", lw=2)
    ax_traj.set_xlabel("downrange distance [km]")
    ax_traj.set_ylabel("altitude [km]")
    ax_traj.set_title("Flight path (true scale)")
    ax_traj.grid(True, alpha=0.3)

    panels = [
        (gs[0, 1], np.degrees(gam), "flight-path angle [deg]"),
        (gs[0, 2], v, "speed [m/s]"),
        (gs[1, 1], np.where(np.arange(len(ts)) <= i_b, q / 1e3, np.nan),
         "dynamic pressure [kPa] (climb only)"),
        (gs[1, 2], h / 1e3, "altitude [km]"),
    ]
    for pos, data, label in panels:
        ax = fig.add_subplot(pos)
        ax.plot(ts, data)
        ax.axvline(ev["kick"], color="gray", ls=":")
        ax.axvline(ev["burnout"], color="gray", ls="--")
        ax.set_ylabel(label)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 2: gravity turn  (dotted = kick, dashed = burnout)")
    fig.tight_layout()
    fig.savefig("phase2_flight.png", dpi=150)
    print("\nSaved plot to phase2_flight.png")
    return fig


def make_animation(rocket, ts, ys, ev):
    """Play the flight in (roughly) real time with a velocity arrow."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    h, v, gam, x, m = ys.T
    n_frames = 400
    idx = np.linspace(0, len(ts) - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x / 1e3, h / 1e3, color="lightgray", lw=1)
    ax.axhline(0, color="green", lw=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("downrange distance [km]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05 * x.max() / 1e3, 1.05 * x.max() / 1e3)
    ax.set_ylim(-0.08 * h.max() / 1e3, 1.1 * h.max() / 1e3)

    trail, = ax.plot([], [], color="tab:orange", lw=2)
    dot, = ax.plot([], [], "o", color="red", ms=9)
    arrow_len = 0.12 * h.max() / 1e3
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))
    text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace")

    def update(k):
        i = idx[k]
        trail.set_data(x[: i + 1] / 1e3, h[: i + 1] / 1e3)
        dot.set_data([x[i] / 1e3], [h[i] / 1e3])
        dx = arrow_len * np.cos(gam[i])
        dy = arrow_len * np.sin(gam[i])
        arrow.xy = (x[i] / 1e3 + dx, h[i] / 1e3 + dy)
        arrow.set_position((x[i] / 1e3, h[i] / 1e3))
        phase = ("vertical rise" if ts[i] < ev["kick"]
                 else "gravity turn (engine on)" if ts[i] < ev["burnout"]
                 else "coasting (engine off)")
        text.set_text(f"t = {ts[i]:6.1f} s   {phase}\n"
                      f"h = {h[i] / 1e3:7.1f} km   v = {v[i]:6.0f} m/s\n"
                      f"flight-path angle = {np.degrees(gam[i]):5.1f} deg")
        return trail, dot, text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=25, blit=False)
    fig._anim = anim   # keep a reference so it is not garbage collected
    return fig, update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Same generic sounding rocket as Phase 1.
    rocket = Rocket(
        m0=10000.0,
        m_prop=7000.0,
        isp=280.0,
        mass_flow=50.0,
        cd=0.3,
        area=1.767,
    )

    results = [
        test_reduces_to_phase1(rocket),
        test_conservation(rocket),
        test_step_size(rocket),
    ]
    print("\n" + "=" * 70)
    if all(results):
        print("ALL VALIDATION TESTS PASSED.")
    else:
        print("SOME TESTS FAILED. Do not trust the flight below.")
    print("=" * 70)
    assert all(results), "validation failed"

    ts, ys, ev = fly(rocket)
    report(rocket, ts, ys, ev)

    try:
        import matplotlib.pyplot as plt

        fig_static = make_static_plot(rocket, ts, ys, ev)
        fig_anim = make_animation(rocket, ts, ys, ev)[0] if ANIMATE else None
        plt.show()
    except ImportError:
        print("\n(matplotlib not installed, skipping plots: pip install matplotlib)")
