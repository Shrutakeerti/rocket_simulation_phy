"""
rk4_check.py
Step 1 of the rocket simulation: build the integrator and prove it works.

Test problem: harmonic oscillator  x'' = -x,  x(0) = 1, v(0) = 0
Exact solution: x(t) = cos(t), v(t) = -sin(t)

We compare explicit Euler (order 1) against RK4 (order 4) by halving the
time step repeatedly and measuring how fast the error shrinks.

Note: the observed order only reaches its theoretical value once dt is small
enough (the "asymptotic regime"), so the pass/fail checks look at the finest
refinement levels only.
"""

import numpy as np


# ---------------------------------------------------------------------------
# System definition: y = [x, v],  dy/dt = [v, -x]
# ---------------------------------------------------------------------------
def f(t, y):
    return np.array([y[1], -y[0]])


# ---------------------------------------------------------------------------
# Integrators (single step)
# ---------------------------------------------------------------------------
def euler_step(f, t, y, dt):
    return y + dt * f(t, y)


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(step, f, y0, t_end, dt):
    """Fixed-step integration from t = 0 to t_end. Returns the final state."""
    n = int(round(t_end / dt))
    t = 0.0
    y = np.array(y0, dtype=float)
    for _ in range(n):
        y = step(f, t, y, dt)
        t += dt
    return y


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------
def convergence_study(step, name, t_end=10.0, dt0=0.2, levels=6):
    y0 = [1.0, 0.0]
    exact = np.array([np.cos(t_end), -np.sin(t_end)])

    print(f"\n{name}")
    print(f"{'dt':>12} {'error in x':>14} {'observed order':>16}")

    errors = []
    dts = [dt0 / 2**k for k in range(levels)]
    for dt in dts:
        y = integrate(step, f, y0, t_end, dt)
        err = abs(y[0] - exact[0])
        errors.append(err)

    orders = [None]
    for i in range(1, len(errors)):
        orders.append(np.log2(errors[i - 1] / errors[i]))

    for dt, err, order in zip(dts, errors, orders):
        order_str = "-" if order is None else f"{order:.3f}"
        print(f"{dt:12.5f} {err:14.3e} {order_str:>16}")

    return dts, errors, orders


def energy_drift(step, name, t_end=100.0, dt=0.05):
    """Total energy E = (x^2 + v^2)/2 should stay at 0.5 for this system."""
    y = np.array([1.0, 0.0])
    t = 0.0
    n = int(round(t_end / dt))
    for _ in range(n):
        y = step(f, t, y, dt)
        t += dt
    energy = 0.5 * (y[0] ** 2 + y[1] ** 2)
    print(f"{name:>8}: energy after t={t_end:.0f} s = {energy:.6f}  (exact: 0.500000)")
    return energy


if __name__ == "__main__":
    dts_e, errs_e, ord_e = convergence_study(euler_step, "Explicit Euler (expect order ~1)")
    dts_r, errs_r, ord_r = convergence_study(rk4_step, "RK4 (expect order ~4)")

    print("\nEnergy conservation test (dt = 0.05):")
    e_euler = energy_drift(euler_step, "Euler")
    e_rk4 = energy_drift(rk4_step, "RK4")

    # Automatic pass/fail checks: this is your first "validation test".
    # Only the two finest refinement levels are checked, because the observed
    # order converges to the theoretical order as dt -> 0.
    assert all(0.95 < o < 1.10 for o in ord_e[-2:]), "Euler order is not ~1"
    assert all(3.90 < o < 4.10 for o in ord_r[-2:]), "RK4 order is not ~4"
    assert abs(e_rk4 - 0.5) < 1e-3, "RK4 does not conserve energy well enough"
    assert e_euler > 1.0, "Euler was expected to gain energy on this problem"
    print("\nAll checks passed: Euler is order 1, RK4 is order 4, "
          "RK4 conserves energy, Euler does not.")

    # Optional log-log plot of error vs dt
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(dts_e, errs_e, "o-", label="Euler (slope 1)")
        plt.loglog(dts_r, errs_r, "s-", label="RK4 (slope 4)")
        plt.xlabel("time step dt [s]")
        plt.ylabel("error in x at t = 10 s")
        plt.title("Convergence study")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.show()
    except ImportError:
        print("(matplotlib not installed, skipping plot)")