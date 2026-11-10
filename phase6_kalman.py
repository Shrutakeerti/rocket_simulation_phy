"""
phase6_kalman.py
Phase 6 of the rocket simulation: the rocket no longer knows where it is.

In Phase 5 the guidance computer was handed the exact height, speed and
flight-path angle. Real rockets only have SENSORS, and every sensor is wrong
in its own way:

    accelerometer  (10 Hz)  measures the push of the engine and drag, but has
                            a constant error (bias) and random noise
    gyroscope      (10 Hz)  measures how fast the rocket is turning, same
                            problems; integrating it makes the error grow
    GPS             (1 Hz)  measures position and velocity with NO drift, but
                            noisy, and only once a second

Each one alone is bad in a different way. A Kalman filter combines them:
the IMU carries the estimate between GPS fixes, and GPS keeps the IMU drift
under control. We feed the Phase 5 cutoff controller with four different
"views of reality" and measure where the rocket lands:

    perfect    exact state (the Phase 5 result, the best case)
    gps-hold   the last GPS fix, held until the next one (up to 0.9 s old)
    imu-only   integrate the accelerometer and gyro from the start (dead reckoning)
    ekf        extended Kalman filter fusing IMU and GPS

Put this file in the SAME folder as phase2_gravity_turn.py,
phase3_shooting.py, phase4_montecarlo.py and phase5_feedback.py, then run:

    python3 phase6_kalman.py                 (target 400 km, 30 flights per case)
    python3 phase6_kalman.py 380             (another target; use 350 to 420)
    python3 phase6_kalman.py 400 --n=100     (more flights)
    python3 phase6_kalman.py 400 --no-plot   (text only)

Honest limits: the sensor models are simple (no GPS delay or dropout, no
vibration), the gyro measures the flight-path turn rate directly (the thrust
is assumed to point along the velocity), and the simulation is 2D.
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from phase2_gravity_turn import (KICK_SPEED, MU_EARTH, R_EARTH, Rocket,
                                 gravity, integrate_fixed, integrate_until)
from phase4_montecarlo import (NO_ERRORS, SEED, draw_errors, make_rhs,
                               perturbed_rocket)
from phase5_feedback import (DT_MC, GUIDANCE_DT, RESERVE, TERMINAL_DT,
                             predict_range_km, solve_kicks)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_TARGET_KM = 400.0
DEFAULT_N = 30
GPS_EVERY = 10          # a GPS fix every 10 steps = every 1.0 s

# 1-sigma sensor errors
ACC_NOISE = 0.05                    # [m/s^2] accelerometer noise per 0.1 s sample
ACC_BIAS = 0.02                     # [m/s^2] accelerometer bias (constant in a flight)
GYRO_NOISE = np.radians(0.02)       # [rad/s] gyro noise per sample
GYRO_BIAS = np.radians(0.005)       # [rad/s] gyro bias (constant in a flight)
GPS_POS = 10.0                      # [m] GPS position noise
GPS_VEL = 0.5                       # [m/s] GPS velocity noise

# What the filter believes at the start (end of the vertical rise)
P0_DIAG = np.array([10.0 ** 2, 0.5 ** 2, np.radians(0.05) ** 2, 10.0 ** 2,
                    ACC_BIAS ** 2, GYRO_BIAS ** 2])
# Process noise added per 0.1 s step (how much the filter allows itself to be wrong)
Q_DIAG = np.array([0.02 ** 2, (ACC_NOISE * 0.1) ** 2, (GYRO_NOISE * 0.1) ** 2,
                   0.02 ** 2, (2e-5) ** 2, (1e-7) ** 2])
R_GPS = np.diag([GPS_POS ** 2, GPS_POS ** 2, GPS_VEL ** 2, GPS_VEL ** 2])

MODES = ["perfect", "gps-hold", "imu-only", "ekf"]


# ---------------------------------------------------------------------------
# Inertial navigation model and the Kalman filter
# ---------------------------------------------------------------------------
# State of the filter: [h, v, gamma, x, accel bias, gyro bias]
def ins_deriv(s, a_meas, w_meas):
    """How the filter believes the state changes, driven by the IMU readings."""
    h, v, g, x, ba, bg = s
    sg, cg = math.sin(g), math.cos(g)
    r = R_EARTH + h
    grav = MU_EARTH / (r * r)
    return np.array([v * sg,
                     (a_meas - ba) - grav * sg,
                     w_meas - bg,
                     v * cg * R_EARTH / r,
                     0.0, 0.0])


def ins_jacobian(s):
    """Exact derivative of ins_deriv with respect to the state."""
    h, v, g, x, ba, bg = s
    sg, cg = math.sin(g), math.cos(g)
    r = R_EARTH + h
    grav = MU_EARTH / (r * r)
    A = np.zeros((6, 6))
    A[0, 1] = sg
    A[0, 2] = v * cg
    A[1, 0] = 2.0 * grav * sg / r
    A[1, 2] = -grav * cg
    A[1, 4] = -1.0
    A[2, 5] = -1.0
    A[3, 0] = -v * cg * R_EARTH / (r * r)
    A[3, 1] = cg * R_EARTH / r
    A[3, 2] = -v * sg * R_EARTH / r
    return A


def gps_model(s):
    """What a perfect GPS would read for state s: [h, x, vertical speed, horizontal speed]."""
    h, v, g, x = s[0], s[1], s[2], s[3]
    r = R_EARTH + h
    return np.array([h, x, v * math.sin(g), v * math.cos(g) * R_EARTH / r])


def gps_jacobian(s):
    h, v, g = s[0], s[1], s[2]
    sg, cg = math.sin(g), math.cos(g)
    r = R_EARTH + h
    H = np.zeros((4, 6))
    H[0, 0] = 1.0
    H[1, 3] = 1.0
    H[2, 1] = sg
    H[2, 2] = v * cg
    H[3, 0] = -v * cg * R_EARTH / (r * r)
    H[3, 1] = cg * R_EARTH / r
    H[3, 2] = -v * sg * R_EARTH / r
    return H


class Navigator:
    """Extended Kalman filter (and, with use_covariance=False, plain dead reckoning)."""

    def __init__(self, s0, P0, use_covariance=True):
        self.s = np.array(s0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.use_cov = use_covariance

    def predict(self, a_meas, w_meas, dt):
        s = self.s
        k1 = ins_deriv(s, a_meas, w_meas)
        k2 = ins_deriv(s + 0.5 * dt * k1, a_meas, w_meas)
        k3 = ins_deriv(s + 0.5 * dt * k2, a_meas, w_meas)
        k4 = ins_deriv(s + dt * k3, a_meas, w_meas)
        s_new = s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if self.use_cov:
            A = ins_jacobian(0.5 * (s + s_new))
            Adt = A * dt
            F = np.eye(6) + Adt + 0.5 * (Adt @ Adt)
            self.P = F @ self.P @ F.T + np.diag(Q_DIAG) * (dt / 0.1)
        self.s = s_new

    def update_gps(self, z):
        H = gps_jacobian(self.s)
        innovation = z - gps_model(self.s)
        S = H @ self.P @ H.T + R_GPS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ innovation
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_GPS @ K.T   # Joseph form (stays symmetric)


def gps_to_state(z, m):
    """Turn one raw GPS reading into (h, v, gamma, x, m) with no filtering at all."""
    h, x, v_vert, v_horiz_ground = z
    v_horiz = v_horiz_ground * (R_EARTH + h) / R_EARTH
    return np.array([h, math.hypot(v_vert, v_horiz), math.atan2(v_vert, v_horiz), x, m])


# ---------------------------------------------------------------------------
# One closed-loop flight with sensors
# ---------------------------------------------------------------------------
def true_sensor_signals(f_burn, t, y):
    """What an ideal accelerometer and gyro would read for the true state y."""
    d = f_burn(t, y)
    a_specific = d[1] + gravity(y[0]) * math.sin(y[2])   # acceleration from engine + drag
    return a_specific, d[2]


def fly_sensed(base, p, kick_deg, target_km, mode, sensor_seed, record=False,
               noise_scale=1.0, dt=DT_MC):
    """Phase 5's cutoff controller, but it only sees what its sensors tell it.

    mode: 'perfect', 'gps-hold', 'imu-only' or 'ekf'.
    """
    rocket = perturbed_rocket(base, p)
    rho_scale = 1.0 + p["rho"]
    f_rise = make_rhs(rocket, rho_scale, p["wind"], burning=True, hold_vertical=True)
    f_burn = make_rhs(rocket, rho_scale, p["wind"], burning=True)
    f_coast = make_rhs(rocket, rho_scale, p["wind"], burning=False)

    y0 = np.array([0.0, 1e-3, np.pi / 2.0, 0.0, rocket.m0])
    ts1, ys1 = integrate_until(f_rise, y0, 0.0, dt, lambda y: y[1] >= KICK_SPEED)
    y = ys1[-1].copy()
    y[2] -= np.radians(kick_deg + p["kick"])
    t = ts1[-1]
    t_depl = rocket.burn_time

    # Sensor errors for this flight (the same seed gives the same sensors in every mode).
    rng = np.random.default_rng(sensor_seed)
    ns = noise_scale
    b_acc = rng.normal(0.0, ACC_BIAS) * ns
    b_gyro = rng.normal(0.0, GYRO_BIAS) * ns
    init_err = rng.normal(0.0, 1.0, 4) * np.sqrt(P0_DIAG[:4]) * ns
    acc_noise = rng.normal(0.0, ACC_NOISE * ns, 3000)
    gyro_noise = rng.normal(0.0, GYRO_NOISE * ns, 3000)
    gps_noise = rng.normal(0.0, 1.0, (400, 4)) * np.array(
        [GPS_POS, GPS_POS, GPS_VEL, GPS_VEL]) * ns

    s0 = np.array([y[0] + init_err[0], y[1] + init_err[1], y[2] + init_err[2],
                   y[3] + init_err[3], 0.0, 0.0])
    P0 = np.diag(P0_DIAG)
    nav = Navigator(s0, P0, use_covariance=(mode == "ekf"))
    held = np.array([s0[0], s0[1], s0[2], s0[3], y[4]])   # for gps-hold before the first fix
    step_count = 0
    prev_rates = true_sensor_signals(f_burn, t, y)
    rec = {"t": [], "err": [], "sig": [], "nees": []} if record else None

    def estimate(y_true):
        """The state the guidance computer believes (it knows its own mass exactly)."""
        if mode == "perfect":
            return y_true.copy()
        if mode == "gps-hold":
            out = held.copy()
        else:
            out = np.array([nav.s[0], nav.s[1], nav.s[2], nav.s[3], y_true[4]])
        return out

    def process_chunk(ts_, ys_):
        """Run the sensors and the navigation filter over the steps of one chunk."""
        nonlocal prev_rates, step_count, held
        for j in range(1, len(ts_)):
            rates = true_sensor_signals(f_burn, ts_[j], ys_[j])
            if mode != "perfect":
                a_meas = 0.5 * (prev_rates[0] + rates[0]) + b_acc + acc_noise[step_count]
                w_meas = 0.5 * (prev_rates[1] + rates[1]) + b_gyro + gyro_noise[step_count]
                if mode in ("imu-only", "ekf"):
                    nav.predict(a_meas, w_meas, ts_[j] - ts_[j - 1])
                step_count += 1
                if step_count % GPS_EVERY == 0:
                    z = gps_model(ys_[j]) + gps_noise[step_count // GPS_EVERY]
                    if mode == "ekf":
                        nav.update_gps(z)
                    held = gps_to_state(z, ys_[j][4])
                if record:
                    truth6 = np.array([ys_[j][0], ys_[j][1], ys_[j][2], ys_[j][3],
                                       b_acc, b_gyro])
                    if mode == "gps-hold":
                        est6 = np.array([held[0], held[1], held[2], held[3], 0.0, 0.0])
                    else:
                        est6 = nav.s
                    e = truth6 - est6
                    rec["t"].append(ts_[j])
                    rec["err"].append(e)
                    if mode == "ekf":
                        rec["sig"].append(np.sqrt(np.diag(nav.P)))
                        rec["nees"].append(float(e @ np.linalg.solve(nav.P, e)))
            prev_rates = rates

    seg_y = [ys1]
    r_prev, t_prev = None, None
    t_cut = None
    cycle = GUIDANCE_DT
    decision_est = None
    decision_true = None
    while True:
        est = estimate(y)
        r_now = predict_range_km(est, base)
        if r_now >= target_km:
            t_cut = t
            decision_est, decision_true = est, y.copy()
            break
        if r_prev is not None and r_now > r_prev:
            rate = (r_now - r_prev) / (t - t_prev)
            time_to_go = (target_km - r_now) / rate
            if time_to_go <= 3.0 * GUIDANCE_DT:
                cycle = TERMINAL_DT
            if time_to_go <= 2.0 * cycle:
                t_cut = min(t + time_to_go, t_depl)
                decision_est, decision_true = est, y.copy()
                n = max(1, int(round((t_cut - t) / dt)))
                _, ys = integrate_fixed(f_burn, y, t, t_cut, n)
                seg_y.append(ys)
                y = ys[-1]
                break
        if t >= t_depl - 1e-9:
            t_cut = t_depl
            decision_est, decision_true = est, y.copy()
            break
        r_prev, t_prev = r_now, t
        t_next = min(t + cycle, t_depl)
        n = max(1, int(round((t_next - t) / dt)))
        ts_, ys = integrate_fixed(f_burn, y, t, t_next, n)
        seg_y.append(ys)
        process_chunk(ts_, ys)
        y = ys[-1]
        t = t_next

    burn_ys = np.vstack(seg_y)
    crashed = bool(burn_ys[:, 0].min() < -1.0)
    ts3, ys3 = integrate_until(f_coast, y, t_cut, dt, lambda yy: yy[0] <= 0.0)

    # How wrong was the guidance computer's picture at the moment it decided?
    err_h = decision_est[0] - decision_true[0]
    err_v = decision_est[1] - decision_true[1]
    err_g = np.degrees(decision_est[2] - decision_true[2])
    pred_err = predict_range_km(decision_est, base) - predict_range_km(decision_true, base)
    out = {
        "range_km": float("nan") if crashed else ys3[-1, 3] / 1e3,
        "t_cut": t_cut,
        "saturated": bool(t_cut >= t_depl - 1e-6),
        "err_h": err_h, "err_v": err_v, "err_g": err_g, "pred_err": pred_err,
    }
    if record:
        out["rec"] = {k: np.array(v) for k, v in rec.items()}
    return out


def run_one(args):
    base, p, kick_deg, target_km, mode, seed = args
    return fly_sensed(base, p, kick_deg, target_km, mode, seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(base, kick, target_km):
    print("\nTests")
    results = []

    print("\nTest 1: the filter's Jacobians must match finite differences")
    s = np.array([40e3, 1500.0, np.radians(35.0), 30e3, 0.01, 0.0001])
    eps = 1e-6
    num_A = np.zeros((6, 6))
    num_H = np.zeros((4, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps * max(1.0, abs(s[i]))
        num_A[:, i] = (ins_deriv(s + d, 2.0, 0.001) - ins_deriv(s - d, 2.0, 0.001)) / (2 * d[i])
        num_H[:, i] = (gps_model(s + d) - gps_model(s - d)) / (2 * d[i])
    err_A = np.max(np.abs(num_A - ins_jacobian(s)))
    err_H = np.max(np.abs(num_H - gps_jacobian(s)))
    ok = err_A < 1e-6 and err_H < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] largest difference: process model {err_A:.2e}, "
          f"GPS model {err_H:.2e}")
    results.append(ok)

    print("\nTest 2: with perfect sensors the navigation filter must track the truth")
    r = fly_sensed(base, NO_ERRORS, kick, target_km, "ekf", 1, record=True, noise_scale=0.0)
    e = r["rec"]["err"]
    max_err = np.max(np.abs(e[:, :4]), axis=0)
    ok = max_err[0] < 1.0 and max_err[1] < 0.01 and max_err[2] < 1e-5 and max_err[3] < 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] largest error over the burn: height {max_err[0]:.3f} m, "
          f"speed {max_err[1]:.4f} m/s, angle {np.degrees(max_err[2]):.5f} deg, "
          f"downrange {max_err[3]:.3f} m")
    results.append(ok)

    print("\nTest 3: is the filter honest about its own uncertainty? (consistency)")
    print("        average NEES over 12 flights should be close to 6 (the number of states)")
    nees = []
    for i in range(12):
        ri = fly_sensed(base, NO_ERRORS, kick, target_km, "ekf", 100 + i, record=True)
        nees.append(np.mean(ri["rec"]["nees"]))
    avg = float(np.mean(nees))
    ok = 3.5 < avg < 9.0
    print(f"  [{'PASS' if ok else 'FAIL'}] average NEES = {avg:.2f} "
          "(much larger than 6 = overconfident filter, much smaller = too timid)")
    results.append(ok)

    print("\nTest 4: the filter must beat each raw sensor on speed error")
    errs = {m: [] for m in ("gps-hold", "imu-only", "ekf")}
    for i in range(12):
        for m in errs:
            errs[m].append(fly_sensed(base, NO_ERRORS, kick, target_km, m, 200 + i)["err_v"])
    rms = {m: float(np.sqrt(np.mean(np.square(v)))) for m, v in errs.items()}
    ok = rms["ekf"] < rms["gps-hold"] and rms["ekf"] < rms["imu-only"]
    print(f"  [{'PASS' if ok else 'FAIL'}] RMS speed error at the decision: "
          f"ekf {rms['ekf']:.2f} m/s, gps-hold {rms['gps-hold']:.2f} m/s, "
          f"imu-only {rms['imu-only']:.2f} m/s")
    results.append(ok)
    return all(results)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(base, kick_deg, target_km, samples, mode):
    jobs = [(base, p, kick_deg, target_km, mode, SEED + 1000 + i)
            for i, p in enumerate(samples)]
    workers = max(1, (os.cpu_count() or 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_one, jobs, chunksize=2))
    except Exception:
        return [run_one(j) for j in jobs]


def stats_of(res, target_km):
    rng = np.array([r["range_km"] for r in res])
    ok = ~np.isnan(rng)
    rng = rng[ok]
    dev = rng - target_km
    pick = lambda key: np.array([r[key] for r in res])[ok]
    return {
        "range": rng,
        "mean": rng.mean(),
        "bias": rng.mean() - target_km,
        "std": rng.std(ddof=1),
        "rms_miss": float(np.sqrt(np.mean(dev ** 2))),
        "min": rng.min(),
        "max": rng.max(),
        "within5": 100.0 * np.mean(np.abs(dev) <= 5.0),
        "within10": 100.0 * np.mean(np.abs(dev) <= 10.0),
        "saturated": int(pick("saturated").sum()),
        "rms_h": float(np.sqrt(np.mean(pick("err_h") ** 2))),
        "rms_v": float(np.sqrt(np.mean(pick("err_v") ** 2))),
        "rms_g": float(np.sqrt(np.mean(pick("err_g") ** 2))),
        "rms_pred": float(np.sqrt(np.mean(pick("pred_err") ** 2))),
    }


def print_histogram(arc_name, data, target_km, bins=17):
    devs = {m: np.asarray(data[m]["range"]) - target_km for m in MODES}
    span = float(np.ceil(max(np.abs(d).max() for d in devs.values()) / 5.0) * 5.0)
    span = max(span, 5.0)
    edges = np.linspace(-span, span, bins + 1)
    counts = {m: np.histogram(devs[m], edges)[0] for m in MODES}
    scale = max(1, int(np.ceil(max(c.max() for c in counts.values()) / 12.0)))
    print(f"\n  {arc_name}: landing error histogram (each # = {scale} flight(s))")
    print(f"  {'error range [km]':>20}  " + "".join(f"{m:<14}" for m in MODES))
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        bars = "".join(f"{'#' * int(np.ceil(counts[m][i] / scale)):<14}" for m in MODES)
        marker = "<- target" if lo <= 0.0 < hi else ""
        print(f"  {lo:+9.1f} to {hi:+8.1f}  {bars}{marker}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(results, demo, target_km):
    import matplotlib.pyplot as plt

    colors = {"perfect": "tab:gray", "gps-hold": "tab:orange",
              "imu-only": "tab:red", "ekf": "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    arcs = list(results)

    for row, arc in enumerate(arcs):
        ax = axes[0, row]
        data = results[arc]
        lo = min(data[m]["range"].min() for m in MODES) - 2
        hi = max(data[m]["range"].max() for m in MODES) + 2
        bins = np.linspace(lo, hi, 40)
        for m in MODES:
            ax.hist(data[m]["range"], bins=bins, alpha=0.5, color=colors[m],
                    label=f"{m}: {data[m]['rms_miss']:.1f} km")
        ax.axvline(target_km, color="k", ls="--")
        ax.set_title(f"{arc}: where did the rockets land?")
        ax.set_xlabel("landing distance [km downrange]")
        ax.set_ylabel("number of flights")
        ax.legend(title="total miss (RMS)")
        ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    width = 0.38
    xs = np.arange(len(MODES))
    for i, arc in enumerate(arcs):
        ax.bar(xs + (i - 0.5) * width, [results[arc][m]["rms_miss"] for m in MODES], width,
               label=arc)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODES)
    ax.set_yscale("log")
    ax.set_ylabel("total miss, RMS [km] (log scale)")
    ax.set_title("What each view of reality costs")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    titles = [("speed error [m/s]", 1, 1.0), ("flight-path angle error [deg]", 2, np.degrees(1.0)),
              ("height error [m]", 0, 1.0)]
    for col, (title, idx, scale) in enumerate(titles):
        ax = axes[1, col]
        for m in ("gps-hold", "imu-only", "ekf"):
            r = demo[m]
            ax.plot(r["t"], -r["err"][:, idx] * scale, color=colors[m], label=m, lw=1.4)
        sig = demo["ekf"]["sig"][:, idx] * scale
        t = demo["ekf"]["t"]
        ax.fill_between(t, -2 * sig, 2 * sig, color="tab:green", alpha=0.2,
                        label="ekf's own +/-2 sigma")
        ax.set_title(f"One flight: {title}")
        ax.set_xlabel("time [s]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 6: noisy sensors and the Kalman filter")
    fig.tight_layout()
    fig.savefig("phase6_kalman.png", dpi=150)
    print("\nSaved plot to phase6_kalman.png")
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
    print(f"PHASE 6: NOISY SENSORS AND A KALMAN FILTER   target = {target_km:.0f} km")
    print("=" * 78)
    print("\nSensor errors (1 sigma):")
    print(f"  accelerometer   noise {ACC_NOISE} m/s^2 per 0.1 s, bias {ACC_BIAS} m/s^2 (10 Hz)")
    print(f"  gyroscope       noise {np.degrees(GYRO_NOISE):.3f} deg/s per sample, "
          f"bias {np.degrees(GYRO_BIAS):.3f} deg/s (10 Hz)")
    print(f"  GPS             position {GPS_POS:.0f} m, velocity {GPS_VEL} m/s, 1 fix per second")

    print("\nSolving for the planned kick angles (Phase 5 plan) ...")
    kicks, r_peak = solve_kicks(base, target_km, plan_cut)
    if kicks is None:
        print(f"{target_km:.0f} km is beyond the maximum range ({r_peak:.1f} km). "
              "Pick a closer target (350 to 420 km works).")
        sys.exit(2)
    for name, k in kicks.items():
        print(f"  {name:<20} kick = {k:.6f} deg")

    k_test = list(kicks.values())[-1]
    if not run_tests(base, k_test, target_km):
        print("\nSOME TESTS FAILED. Do not trust the results below.")
        sys.exit(1)

    samples = draw_errors(n, SEED)
    print("\n" + "=" * 78)
    print(f"MONTE CARLO: {n} flights per case, same physical errors as before, plus sensor errors")
    print("=" * 78)
    print("(This takes a few minutes: every flight runs a navigation filter 10 times a second.)")

    results = {}
    for name, k in kicks.items():
        results[name] = {}
        for mode in MODES:
            print(f"  {name}: flying {n} x {mode} ...", flush=True)
            results[name][mode] = stats_of(monte_carlo(base, k, target_km, samples, mode),
                                           target_km)

    for name, data in results.items():
        print("\n" + "-" * 78)
        print(f"{name}   (kick {kicks[name]:.4f} deg)")
        print("-" * 78)
        print(f"{'':<34}" + "".join(f"{m:>11}" for m in MODES))
        rows = [
            ("Landing scatter, 1 sigma [km]", "std", "{:.2f}"),
            ("Average miss (bias) [km]", "bias", "{:+.2f}"),
            ("Total miss, RMS [km]", "rms_miss", "{:.2f}"),
            ("Closest landing [km]", "min", "{:.1f}"),
            ("Farthest landing [km]", "max", "{:.1f}"),
            ("Within 5 km of target [%]", "within5", "{:.0f}"),
            ("Within 10 km of target [%]", "within10", "{:.0f}"),
            ("Flights that ran out of fuel", "saturated", "{:d}"),
            ("Picture error: height [m, RMS]", "rms_h", "{:.1f}"),
            ("Picture error: speed [m/s, RMS]", "rms_v", "{:.2f}"),
            ("Picture error: angle [deg, RMS]", "rms_g", "{:.3f}"),
            ("Predicted-range error [km, RMS]", "rms_pred", "{:.2f}"),
        ]
        for label, key, fmt in rows:
            print(f"{label:<34}" + "".join(f"{fmt.format(data[m][key]):>11}" for m in MODES))
        print_histogram(name, data, target_km)

    print("\n" + "=" * 78)
    print("WHAT TO TAKE FROM THIS")
    print("=" * 78)
    for name, data in results.items():
        line = ", ".join(f"{m} {data[m]['rms_miss']:.2f}" for m in MODES)
        print(f"  * {name}: total miss (RMS, km) -> {line}")
    print("  * Scatter alone can mislead: a view that is wrong in the SAME direction every")
    print("    time (gps-hold lands short on average) can still show a small scatter.")
    print("    The total miss counts both the spread and the average error.")
    print("  * gps-hold fails mainly because its picture is a staircase: it jumps once a")
    print("    second, which fools the controller's estimate of how fast the range is")
    print("    changing, so it cuts the engine too early.")
    print("  * imu-only fails because its small errors accumulate (drift) for 130 seconds.")
    print("  * 'Picture error' is what the guidance computer got wrong about its own state")
    print("    at the moment it decided to cut the engine. Landing error follows from it.")
    print()
    print("  IMPORTANT: these sensors and this filter are idealised. The filter's model")
    print("  matches the simulated truth, GPS never drops out or lags, and nothing")
    print("  vibrates. Real navigation is harder. The point here is the method.")

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            print("\nRunning one detailed flight for the plot ...", flush=True)
            k_demo = list(kicks.values())[-1]
            demo = {}
            for mode in ("gps-hold", "imu-only", "ekf"):
                demo[mode] = fly_sensed(base, samples[0], k_demo, target_km, mode,
                                        SEED + 1000, record=True)["rec"]
            fig = make_plot(results, demo, target_km)
            plt.show()
        except ImportError:
            print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")

    print("\nDone. Copy everything above and paste it into the chat.")


if __name__ == "__main__":
    main()