"""
phase1_ascent.py
Phase 1 of the rocket simulation: a 1D vertical ascent.

State:  y = [h, v, m]   (altitude [m], vertical speed [m/s], mass [kg])

    h' = v
    v' = (T - D)/m - g(h)
    m' = -mass_flow          (while the engine burns)

    T = Isp * g0 * mass_flow            (vacuum thrust, constant Isp for now)
    D = 0.5 * rho(h) * Cd * A * v|v|    (drag, always opposes motion)
    rho(h) = rho0 * exp(-h / H)         (simple exponential atmosphere)
    g(h)   = mu / (R_earth + h)^2       (inverse-square gravity)

The file has two parts:
  1. Validation tests against exact analytic answers (these must all PASS).
  2. A full flight with drag and inverse-square gravity.

Units: SI everywhere (metres, seconds, kilograms, newtons).
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G0 = 9.80665            # standard gravity [m/s^2]
MU_EARTH = 3.986004418e14   # Earth's gravitational parameter [m^3/s^2]
R_EARTH = 6.371e6       # mean Earth radius [m]
RHO0 = 1.225            # sea-level air density [kg/m^3]
H_SCALE = 8500.0        # atmosphere scale height [m] (crude, replaced later)


# ---------------------------------------------------------------------------
# Rocket definition
# ---------------------------------------------------------------------------
@dataclass
class Rocket:
    m0: float           # initial (wet) mass [kg]
    m_prop: float       # propellant mass [kg]
    isp: float          # specific impulse [s]
    mass_flow: float    # propellant mass flow rate [kg/s]
    cd: float           # drag coefficient (constant for now)
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
# Physics models
# ---------------------------------------------------------------------------
def gravity(h, mode):
    if mode == "none":
        return 0.0
    if mode == "const":
        return G0
    if mode == "inverse_square":
        return MU_EARTH / (R_EARTH + h) ** 2
    raise ValueError(f"unknown gravity mode: {mode}")


def air_density(h):
    return RHO0 * np.exp(-np.maximum(h, 0.0) / H_SCALE)


def drag_force(h, v, rocket):
    """Drag force magnitude with sign, so it always opposes the motion."""
    return 0.5 * air_density(h) * rocket.cd * rocket.area * v * abs(v)


def make_rhs(rocket, gravity_mode="inverse_square", drag=True, burning=True):
    """Build the function f(t, y) = dy/dt for the chosen physics options."""

    def f(t, y):
        h, v, m = y
        thrust = rocket.thrust if burning else 0.0
        dm_dt = -rocket.mass_flow if burning else 0.0
        d = drag_force(h, v, rocket) if drag else 0.0
        dv_dt = (thrust - d) / m - gravity(h, gravity_mode)
        return np.array([v, dv_dt, dm_dt])

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


def integrate_phase(f, y0, t0, t1, n_steps):
    """Integrate from t0 to t1 in exactly n_steps equal steps.

    The engine switches off exactly at the end of the burn phase. Integrating
    the burn and the coast as separate phases avoids stepping across the
    discontinuity in the thrust, which would ruin RK4's accuracy.
    """
    dt = (t1 - t0) / n_steps
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, len(y0)))
    ts[0] = t0
    ys[0] = y0
    y = np.array(y0, dtype=float)
    for i in range(n_steps):
        t = t0 + i * dt
        y = rk4_step(f, t, y, dt)
        ts[i + 1] = t0 + (i + 1) * dt
        ys[i + 1] = y
    return ts, ys


def coast_to_apex(f, t0, y0, dt, t_max=5000.0):
    """Integrate until the vertical speed reaches zero (the apex).

    Once a step overshoots v = 0, bisection on the step size finds the exact
    apex time. Returns (times, states, t_apex, y_apex).
    """
    y = np.array(y0, dtype=float)
    if y[1] <= 0.0:
        raise ValueError("coast_to_apex needs an initial upward speed (v > 0)")
    t = t0
    ts = [t]
    ys = [y.copy()]
    while t < t_max:
        y_new = rk4_step(f, t, y, dt)
        if y_new[1] <= 0.0:
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rk4_step(f, t, y, mid)[1] > 0.0:
                    lo = mid
                else:
                    hi = mid
            s = 0.5 * (lo + hi)
            y_apex = rk4_step(f, t, y, s)
            ts.append(t + s)
            ys.append(y_apex)
            return np.array(ts), np.array(ys), t + s, y_apex
        t += dt
        y = y_new
        ts.append(t)
        ys.append(y.copy())
    raise RuntimeError("apex not reached before t_max")


def trapezoid(x, t):
    """Trapezoidal integral of x(t) (written out to avoid numpy version issues)."""
    return float(np.sum(0.5 * (x[1:] + x[:-1]) * np.diff(t)))


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def check(name, got, expected, tol, relative=False):
    err = abs(got - expected)
    if relative:
        err /= abs(expected)
    ok = err <= tol
    status = "PASS" if ok else "FAIL"
    kind = "rel" if relative else "abs"
    print(f"  [{status}] {name}: got {got:.9g}, expected {expected:.9g}, "
          f"{kind} error {err:.3e}")
    return ok


def test_tsiolkovsky(rocket):
    print("\nTest 1: Tsiolkovsky rocket equation (no gravity, no drag)")
    f = make_rhs(rocket, gravity_mode="none", drag=False, burning=True)
    tb = rocket.burn_time
    n = int(round(tb / 0.1))
    _, ys = integrate_phase(f, [0.0, 0.0, rocket.m0], 0.0, tb, n)
    v_exact = rocket.v_exhaust * np.log(rocket.m0 / rocket.m_dry)
    ok1 = check("final speed [m/s]", ys[-1, 1], v_exact, 1e-9, relative=True)
    ok2 = check("final mass [kg]", ys[-1, 2], rocket.m_dry, 1e-6)
    return ok1 and ok2


def test_ballistic_constant_g(rocket):
    print("\nTest 2: ballistic shot, constant gravity, no drag, no thrust")
    f = make_rhs(rocket, gravity_mode="const", drag=False, burning=False)
    v0 = 300.0
    _, _, t_apex, y_apex = coast_to_apex(f, 0.0, [0.0, v0, rocket.m0], dt=0.1)
    ok1 = check("apex time [s]", t_apex, v0 / G0, 1e-6)
    ok2 = check("apex height [m]", y_apex[0], v0**2 / (2.0 * G0), 1e-6)
    return ok1 and ok2


def burn_with_gravity_exact(rocket):
    """Closed-form speed and height at burnout, constant g, no drag.

    m(t) = m0 - mdot*t
    v(t) = ve*ln(m0/m) - g*t
    h(t) = ve*(t - (m/mdot)*ln(m0/m)) - g*t^2/2
    """
    ve, mdot = rocket.v_exhaust, rocket.mass_flow
    tb = rocket.burn_time
    m, m0 = rocket.m_dry, rocket.m0
    v = ve * np.log(m0 / m) - G0 * tb
    h = ve * (tb - (m / mdot) * np.log(m0 / m)) - 0.5 * G0 * tb**2
    return v, h


def test_burn_with_gravity(rocket):
    print("\nTest 3: powered ascent, constant gravity, no drag (closed form)")
    f = make_rhs(rocket, gravity_mode="const", drag=False, burning=True)
    tb = rocket.burn_time
    n = int(round(tb / 0.1))
    _, ys = integrate_phase(f, [0.0, 0.0, rocket.m0], 0.0, tb, n)
    v_exact, h_exact = burn_with_gravity_exact(rocket)
    ok1 = check("burnout speed [m/s]", ys[-1, 1], v_exact, 1e-6)
    ok2 = check("burnout height [m]", ys[-1, 0], h_exact, 1e-4)
    return ok1 and ok2


def test_convergence(rocket):
    print("\nTest 3b: RK4 convergence on the rocket equations")
    f = make_rhs(rocket, gravity_mode="const", drag=False, burning=True)
    tb = rocket.burn_time
    v_exact, _ = burn_with_gravity_exact(rocket)
    steps = [14, 28, 56, 112]
    errors = []
    print(f"  {'dt [s]':>10} {'speed error [m/s]':>20} {'observed order':>16}")
    for i, n in enumerate(steps):
        _, ys = integrate_phase(f, [0.0, 0.0, rocket.m0], 0.0, tb, n)
        errors.append(abs(ys[-1, 1] - v_exact))
        order = "-" if i == 0 else f"{np.log2(errors[i - 1] / errors[i]):.3f}"
        print(f"  {tb / n:10.4f} {errors[-1]:20.3e} {order:>16}")
    ratio = errors[0] / errors[-1]
    ok = ratio > 500.0
    print(f"  [{'PASS' if ok else 'FAIL'}] error shrank by {ratio:.0f}x "
          f"over 3 halvings (order 4 would give about 4096x)")
    return ok


def test_inverse_square_energy(rocket):
    print("\nTest 4: inverse-square gravity, coasting (energy and apex)")
    f = make_rhs(rocket, gravity_mode="inverse_square", drag=False, burning=False)
    v0 = 2000.0
    _, ys, _, y_apex = coast_to_apex(f, 0.0, [0.0, v0, rocket.m0], dt=0.1)

    # Specific energy E = v^2/2 - mu/r must stay constant.
    h, v = ys[:, 0], ys[:, 1]
    energy = 0.5 * v**2 - MU_EARTH / (R_EARTH + h)
    drift = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    ok1 = drift < 1e-9
    print(f"  [{'PASS' if ok1 else 'FAIL'}] max relative energy drift: {drift:.3e}")

    # Analytic apex from energy conservation: v^2/2 - mu/r0 = -mu/r_max
    r0 = R_EARTH
    r_max = 1.0 / (1.0 / r0 - v0**2 / (2.0 * MU_EARTH))
    ok2 = check("apex height [m]", y_apex[0], r_max - r0, 1e-3)
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Full flight: thrust + drag + inverse-square gravity
# ---------------------------------------------------------------------------
def fly(rocket, dt=0.05):
    f_burn = make_rhs(rocket, "inverse_square", drag=True, burning=True)
    f_coast = make_rhs(rocket, "inverse_square", drag=True, burning=False)

    tb = rocket.burn_time
    ts_b, ys_b = integrate_phase(
        f_burn, [0.0, 0.0, rocket.m0], 0.0, tb, int(round(tb / dt))
    )

    if ys_b[-1, 1] <= 0.0:
        raise RuntimeError(
            "Rocket is not moving upward at burnout "
            "(thrust-to-weight below 1? there is no ground model yet)."
        )

    ts_c, ys_c, t_apex, y_apex = coast_to_apex(f_coast, tb, ys_b[-1], dt)

    ts = np.concatenate([ts_b, ts_c[1:]])
    ys = np.vstack([ys_b, ys_c[1:]])
    return ts_b, ys_b, ts, ys, t_apex, y_apex


def report_and_plot(rocket):
    print("\n" + "=" * 70)
    print("FULL FLIGHT: thrust + drag + inverse-square gravity")
    print("=" * 70)

    ts_b, ys_b, ts, ys, t_apex, y_apex = fly(rocket)
    h, v, m = ys[:, 0], ys[:, 1], ys[:, 2]

    # Dynamic pressure and Max-Q
    q = 0.5 * air_density(h) * v**2
    i_q = int(np.argmax(q))

    # Delta-v budget during the burn
    hb, vb, mb = ys_b[:, 0], ys_b[:, 1], ys_b[:, 2]
    g_arr = MU_EARTH / (R_EARTH + hb) ** 2
    drag_decel = 0.5 * air_density(hb) * rocket.cd * rocket.area * vb * np.abs(vb) / mb
    gravity_loss = trapezoid(g_arr, ts_b)
    drag_loss = trapezoid(drag_decel, ts_b)
    ideal_dv = rocket.v_exhaust * np.log(rocket.m0 / rocket.m_dry)
    v_burnout = vb[-1]

    twr = rocket.thrust / (rocket.m0 * G0)
    print(f"Thrust:                {rocket.thrust / 1e3:10.1f} kN")
    print(f"Thrust-to-weight:      {twr:10.3f}")
    print(f"Burn time:             {rocket.burn_time:10.1f} s")
    print("-" * 70)
    print(f"Max-Q:                 {q[i_q] / 1e3:10.2f} kPa "
          f"at t = {ts[i_q]:.1f} s, h = {h[i_q] / 1e3:.2f} km, v = {v[i_q]:.0f} m/s")
    print(f"Burnout:               t = {ts_b[-1]:.1f} s, h = {hb[-1] / 1e3:.2f} km, "
          f"v = {v_burnout:.1f} m/s")
    print(f"Apex:                  t = {t_apex:.1f} s, h = {y_apex[0] / 1e3:.2f} km")
    print("-" * 70)
    print("Delta-v budget during the burn:")
    print(f"  ideal (Tsiolkovsky)  {ideal_dv:10.1f} m/s")
    print(f"  - gravity loss       {gravity_loss:10.1f} m/s")
    print(f"  - drag loss          {drag_loss:10.1f} m/s")
    print(f"  = predicted burnout  {ideal_dv - gravity_loss - drag_loss:10.1f} m/s")
    print(f"  simulated burnout    {v_burnout:10.1f} m/s")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axes[0, 0].plot(ts, h / 1e3)
        axes[0, 0].set_ylabel("altitude [km]")
        axes[0, 1].plot(ts, v)
        axes[0, 1].set_ylabel("vertical speed [m/s]")
        axes[1, 0].plot(ts, q / 1e3)
        axes[1, 0].axvline(ts[i_q], color="r", ls="--", label="Max-Q")
        axes[1, 0].set_ylabel("dynamic pressure [kPa]")
        axes[1, 0].legend()
        axes[1, 1].plot(ts, m)
        axes[1, 1].set_ylabel("mass [kg]")
        for ax in axes[1]:
            ax.set_xlabel("time [s]")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.axvline(ts_b[-1], color="gray", ls=":")
        fig.suptitle("Phase 1: 1D vertical ascent (dotted line = burnout)")
        fig.tight_layout()
        fig.savefig("phase1_flight.png", dpi=150)
        print("\nSaved plot to phase1_flight.png")
        plt.show()
    except ImportError:
        print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A generic single-stage sounding rocket (made-up but plausible numbers).
    rocket = Rocket(
        m0=10000.0,        # kg, total at liftoff
        m_prop=7000.0,     # kg of propellant
        isp=280.0,         # s
        mass_flow=50.0,    # kg/s  -> burn time 140 s
        cd=0.3,
        area=1.767,        # m^2 (1.5 m diameter)
    )

    results = [
        test_tsiolkovsky(rocket),
        test_ballistic_constant_g(rocket),
        test_burn_with_gravity(rocket),
        test_convergence(rocket),
        test_inverse_square_energy(rocket),
    ]

    print("\n" + "=" * 70)
    if all(results):
        print("ALL VALIDATION TESTS PASSED. The physics can be trusted so far.")
    else:
        print("SOME TESTS FAILED. Do not build on this until you find out why.")
    print("=" * 70)
    assert all(results), "validation failed"

    report_and_plot(rocket)
