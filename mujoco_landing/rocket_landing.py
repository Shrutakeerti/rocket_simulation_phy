#!/usr/bin/env python3
"""
rocket_landing.py
A SpaceX-style propulsive booster landing in the MuJoCo physics engine.

What is simulated (a real rigid body, not a point mass):
    * a 42 m booster, 6 degrees of freedom, with inertia that changes as propellant burns
    * one engine: gimbals +/-5 deg, throttle 40..100 % (it can NOT hover: minimum
      thrust is bigger than the weight, so the landing burn must be timed exactly)
    * cold-gas reaction control (RCS) for attitude, grid fins as a steering force
    * aerodynamic drag with a centre of pressure (engines-first flight is stable)
    * four landing legs with real contact against the ground, friction and tipping
    * wind, and errors in thrust, propellant, drag, air density and initial state

The flight: aerodynamic descent engines-first -> the autopilot lights the engine at the
last moment ("suicide burn") -> steers with the engine gimbal -> touches down.

Run it (all of these are separate commands):
    python3 rocket_landing.py                   tests + one landing + report + plots
    python3 rocket_landing.py --view            interactive 3D window (real-time physics)
    python3 rocket_landing.py --mc 50           50 randomised landings, success rate
    python3 rocket_landing.py --video out.mp4   render a video (needs imageio, see below)
    python3 rocket_landing.py --no-plot         text only
    python3 rocket_landing.py --seed 7          a different random landing (with --mc/--view/--video)
    python3 rocket_landing.py --mc 50 --stress 2   errors twice as large: where does it break?

Install:  pip install mujoco numpy matplotlib      (video also needs: pip install imageio imageio-ffmpeg)

Honest limits (v1): the autopilot sees the TRUE state (the Kalman filter from Phase 6 is
not plugged in yet); the flight starts at 10 km already engines-first, so the flip, boostback
and hypersonic re-entry are not simulated; the grid fins are an ideal force without torque;
the vehicle numbers are plausible but made up, not real Falcon 9 data.
"""

import argparse
import math
import sys
import time

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Vehicle and environment (SI units, z is up)
# ---------------------------------------------------------------------------
G0 = 9.80665
H_TOT, R_BODY = 42.0, 1.85            # booster height and radius [m]
COM_H = 15.0                          # centre of mass above the base [m]; body origin = COM
D_CG = H_TOT / 2.0 - COM_H            # geometric centre above the COM [m]
DRY_MASS, PROP0 = 25000.0, 5000.0     # [kg] dry mass and landing propellant
T_MAX, THR_MIN, ISP = 845e3, 0.40, 282.0   # one engine: max thrust [N], min throttle, Isp [s]
THR_SLEW = 1.0                        # throttle can change by 100 % per second
GIMBAL_MAX = math.radians(5.0)
RCS_MAX = np.array([150e3, 150e3, 40e3])   # RCS torque limit per body axis [N m]
LEG_R, FOOT_R = 8.0, 0.7              # foot circle radius and foot sphere radius [m]
Z_STAND = COM_H + 0.3 + FOOT_R        # COM height when standing on the legs [m]
CD_AX, CD_LAT = 0.8, 1.1              # drag coefficients (along the axis / sideways)
A_AX, A_LAT = math.pi * R_BODY ** 2, 2.0 * R_BODY * H_TOT
Z_CP = D_CG                           # centre of pressure above the COM [m]
S_GRID = 6.0                          # effective grid-fin area [m^2]: force = dynamic pressure * S_GRID
RHO0, H_SCALE = 1.225, 8500.0
WIND_PEAK_H, WIND_WIDTH = 6000.0, 5000.0
PAD_RADIUS = 30.0
DT_PHYS, DT_CTRL = 0.005, 0.02

# What counts as a good landing
OK_VZ, OK_VLAT, OK_TILT, OK_RATE = 3.0, 2.0, 8.0, 0.3   # m/s, m/s, deg, rad/s


def inertia(m):
    ixx = m * ((3.0 * R_BODY ** 2 + H_TOT ** 2) / 12.0 + D_CG ** 2)
    return np.array([ixx, ixx, 0.5 * m * R_BODY ** 2])


def build_xml():
    legs = ""
    for k in range(4):
        a = math.radians(45 + 90 * k)
        ca, sa = math.cos(a), math.sin(a)
        top = f"{1.7 * ca:.3f} {1.7 * sa:.3f} {-COM_H + 3.0:.3f}"
        foot = f"{LEG_R * ca:.3f} {LEG_R * sa:.3f} {-COM_H - 0.3:.3f}"
        legs += (f'<geom name="leg{k}" type="capsule" fromto="{top} {foot}" size="0.28" '
                 f'rgba=".10 .10 .11 1" friction="1 .005 .0001"/>\n'
                 f'      <geom name="foot{k}" type="sphere" pos="{foot}" size="{FOOT_R}" '
                 f'rgba=".08 .08 .08 1" friction="1.2 .005 .0001"/>\n')
    fins = ""
    for k, (px, py, ang) in enumerate([(2.4, 0, 0), (-2.4, 0, 0), (0, 2.4, 90), (0, -2.4, 90)]):
        fins += (f'<geom type="box" pos="{px} {py} 24" euler="0 0 {ang}" size="0.07 0.95 0.65" '
                 f'rgba=".15 .15 .17 1" contype="0" conaffinity="0"/>\n      ')
    ixx, _, izz = inertia(DRY_MASS + PROP0)
    return f"""
<mujoco model="booster">
  <compiler angle="degree"/>
  <option timestep="{DT_PHYS}" gravity="0 0 -{G0}" integrator="RK4"/>
  <statistic extent="150" center="0 0 40"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/>
          <map znear="0.002" zfar="300"/><headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".55 .72 .92" rgb2=".06 .11 .24" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".20 .24 .28" rgb2=".25 .29 .34" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="60 60" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="400 -300 800" dir="-.4 .3 -.8" diffuse=".9 .9 .85" castshadow="true"/>
    <geom name="ground" type="plane" size="3000 3000 0.1" material="grid" friction="1 .005 .0001"/>
    <geom name="pad" type="cylinder" pos="0 0 0.03" size="{PAD_RADIUS} 0.03" rgba=".34 .36 .40 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="0 0 0.07" size="{PAD_RADIUS * 0.55} 0.03" rgba=".80 .80 .82 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="0 0 0.11" size="{PAD_RADIUS * 0.50} 0.03" rgba=".34 .36 .40 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="0 0 0.15" size="3 0.03" rgba=".95 .35 .1 1" contype="0" conaffinity="0"/>
    <camera name="ground" mode="targetbody" target="rocket" pos="-160 -230 20"/>
    <body name="rocket" pos="0 0 {Z_STAND}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{DRY_MASS + PROP0}" diaginertia="{ixx} {ixx} {izz}"/>
      <geom name="hull" type="cylinder" pos="0 0 {D_CG}" size="{R_BODY} {H_TOT / 2}" rgba=".94 .94 .96 1"/>
      <geom type="cylinder" pos="0 0 {-COM_H + 3.0}" size="{R_BODY + 0.01} 3.0" rgba=".20 .20 .22 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 {H_TOT - COM_H - 2.0}" size="{R_BODY + 0.01} 2.0" rgba=".13 .13 .15 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 {-COM_H + 6.5}" size="{R_BODY + 0.012} 0.3" rgba=".78 .12 .10 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 {H_TOT - COM_H + 0.4}" size="1.2 0.4" rgba=".6 .6 .64 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 {-COM_H - 0.2}" size="1.0 0.2" rgba=".12 .12 .12 1" contype="0" conaffinity="0"/>
      {legs}
      {fins}
      <geom name="flame" type="ellipsoid" pos="0 0 {-COM_H - 8}" size="0.9 0.9 8" rgba="1 .62 .15 0" contype="0" conaffinity="0"/>
    </body>
    <camera name="chase" mode="trackcom" pos="-70 -110 10" xyaxes="0.843 -0.538 0 0.0 0.0 1.0"/>
  </worldbody>
</mujoco>
"""


def rot_to_tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


# ---------------------------------------------------------------------------
# The lander: physics, actuators, atmosphere
# ---------------------------------------------------------------------------
DEFAULT_ENV = dict(
    thrust_err=0.0, prop_err=0.0, dry_err=0.0, cd_scale=1.0, rho_scale=1.0,
    wind=(0.0, 0.0), vacuum=False,
    pos0=(-300.0, 150.0, 10000.0), vel0=(25.0, -12.0, -300.0),
    tilt0_deg=8.0, tilt_az_deg=40.0, rate0=(0.02, -0.01, 0.0), gravity_on=True,
)


class Lander:
    def __init__(self, env=None):
        self.env = dict(DEFAULT_ENV)
        if env:
            self.env.update(env)
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        m = self.model
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rocket")
        self.gid_ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.gid_flame = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "flame")
        self.gid_feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot{k}") for k in range(4)}
        self.gid_legs = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"leg{k}") for k in range(4)}
        self.gid_hull = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "hull")
        if self.env["vacuum"]:
            self.env["rho_scale"] = 0.0
        self.reset()

    # -- setup ------------------------------------------------------------
    def reset(self):
        e, m, d = self.env, self.model, self.data
        mujoco.mj_resetData(m, d)
        self.dry = DRY_MASS * (1.0 + e["dry_err"])
        self.prop = PROP0 * (1.0 + e["prop_err"])
        self.thr = 0.0                      # actual throttle (0 = engine off)
        self.gimbal = np.zeros(2)
        d.qpos[0:3] = e["pos0"]
        axis = np.array([-math.sin(math.radians(e["tilt_az_deg"])),
                         math.cos(math.radians(e["tilt_az_deg"])), 0.0])
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, axis, math.radians(e["tilt0_deg"]))
        d.qpos[3:7] = quat
        d.qvel[0:3] = e["vel0"]
        d.qvel[3:6] = e["rate0"]
        self._set_mass()
        mujoco.mj_forward(m, d)

    def mass(self):
        return self.dry + self.prop

    def _set_mass(self):
        mm = self.mass()
        self.model.body_mass[self.bid] = mm
        self.model.body_inertia[self.bid] = inertia(mm)

    # -- state ------------------------------------------------------------
    def state(self):
        d = self.data
        R = d.xmat[self.bid].reshape(3, 3).copy()
        return dict(t=d.time, p=d.qpos[0:3].copy(), v=d.qvel[0:3].copy(), R=R,
                    w=d.qvel[3:6].copy(), m=self.mass(), prop=self.prop, thr=self.thr)

    def wind_at(self, h):
        f = 0.3 + 0.7 * math.exp(-(((h - WIND_PEAK_H) / WIND_WIDTH) ** 2))
        return np.array([self.env["wind"][0] * f, self.env["wind"][1] * f, 0.0])

    def rho(self, h):
        return self.env["rho_scale"] * RHO0 * math.exp(-max(h, 0.0) / H_SCALE)

    def contacts(self):
        feet, hull, legs = 0, False, 0
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.gid_ground not in (g1, g2):
                continue
            other = g2 if g1 == self.gid_ground else g1
            if other in self.gid_feet:
                feet += 1
            elif other in self.gid_legs:
                legs += 1
            elif other == self.gid_hull:
                hull = True
        return feet, legs, hull

    # -- forces -----------------------------------------------------------
    def aero(self, R, v, h):
        wind = self.wind_at(h)
        vb = R.T @ (v - wind)
        rho = self.rho(h)
        sl = math.hypot(vb[0], vb[1])
        cd = self.env["cd_scale"]
        Fb = np.array([-0.5 * rho * CD_LAT * cd * A_LAT * sl * vb[0],
                       -0.5 * rho * CD_LAT * cd * A_LAT * sl * vb[1],
                       -0.5 * rho * CD_AX * cd * A_AX * abs(vb[2]) * vb[2]])
        tb = np.cross([0.0, 0.0, Z_CP], Fb)
        return R @ Fb, R @ tb

    def apply_forces(self, rcs_b, grid_dir):
        """Set the external force and torque on the booster for the next physics step."""
        d = self.data
        R = d.xmat[self.bid].reshape(3, 3)
        v, h = d.qvel[0:3], d.qpos[2] - Z_STAND
        F_aero, tau_aero = self.aero(R, v, h)
        # engine
        T = self.thr * T_MAX * (1.0 + self.env["thrust_err"]) if self.prop > 0 else 0.0
        gx, gy = self.gimbal
        tb = np.array([math.sin(gy), -math.sin(gx) * math.cos(gy), math.cos(gx) * math.cos(gy)])
        F_eng_b = T * tb
        tau_eng_b = np.cross([0.0, 0.0, -COM_H], F_eng_b)
        # grid fins: an ideal steering force limited by dynamic pressure
        q = 0.5 * self.rho(h) * float(v @ v)
        F_grid = np.zeros(3)
        if grid_dir is not None:
            n = np.linalg.norm(grid_dir)
            if n > 1e-9:
                F_grid = grid_dir / n * min(n, q * S_GRID)
        d.xfrc_applied[self.bid, 0:3] = R @ F_eng_b + F_aero + F_grid
        d.xfrc_applied[self.bid, 3:6] = R @ (tau_eng_b + rcs_b) + tau_aero
        return T

    # -- one control interval ---------------------------------------------
    def advance(self, thr_cmd, gimbal_cmd, rcs_b, grid_dir, dt=DT_CTRL):
        """Advance the physics by dt with these commands held constant."""
        # engine dynamics: ignition jumps to minimum throttle, then slews
        if thr_cmd <= 0.0 or self.prop <= 0.0:
            self.thr = 0.0
        else:
            if self.thr == 0.0:
                self.thr = THR_MIN
            step = THR_SLEW * dt
            self.thr = float(np.clip(thr_cmd, self.thr - step, self.thr + step))
            self.thr = float(np.clip(self.thr, THR_MIN, 1.0))
        self.gimbal = np.clip(np.asarray(gimbal_cmd, float), -GIMBAL_MAX, GIMBAL_MAX)
        rcs_b = np.clip(np.asarray(rcs_b, float), -RCS_MAX, RCS_MAX)
        n_sub = int(round(dt / DT_PHYS))
        for _ in range(n_sub):
            self.apply_forces(rcs_b, grid_dir)
            mujoco.mj_step(self.model, self.data)
        if self.thr > 0.0:
            mdot = self.thr * T_MAX / (ISP * G0)
            self.prop = max(0.0, self.prop - mdot * dt)
        self._set_mass()
        # MuJoCo only applies a changed body mass after mj_setConst (found by testing:
        # calling it every 0.5 s left the mass up to 0.5 s out of date).
        mujoco.mj_setConst(self.model, self.scratch)
        self._update_flame()

    def _update_flame(self):
        m = self.model
        if self.thr > 0.0:
            half = 3.0 + 9.0 * self.thr
            m.geom_size[self.gid_flame] = (0.9 + 0.5 * self.thr, 0.9 + 0.5 * self.thr, half)
            m.geom_pos[self.gid_flame] = (0.0, 0.0, -COM_H - 0.5 - half)
            m.geom_rgba[self.gid_flame][3] = 0.85
        else:
            m.geom_rgba[self.gid_flame][3] = 0.0


# ---------------------------------------------------------------------------
# The autopilot
# ---------------------------------------------------------------------------
A_REF_FRAC = 0.70     # plan the burn at 70 % of the deceleration the engine could give
V_TOUCH = 2.0         # target touchdown speed [m/s]
KV = 3.0              # vertical speed feedback [1/s]
KPOS, KVEL = 0.12, 0.6   # lateral guidance gains: position -> speed target, speed -> acceleration
TILT_MAX = math.radians(10.0)
ATT_WN, ATT_ZETA = 1.0, 1.0


class Autopilot:
    """Guidance and control. It uses the true state and only NOMINAL vehicle numbers."""

    def __init__(self):
        self.phase = "DESCENT"
        self.t_ignite = None
        self.last = {}

    def attitude_cmd(self, st, n_des, thr_on):
        """Torque to point the body z axis along n_des; split between gimbal and RCS."""
        R, w, m = st["R"], st["w"], st["m"]
        zb = R[:, 2]
        e_b = R.T @ np.cross(zb, n_des)
        I = inertia(m)
        kp, kd = ATT_WN ** 2, 2.0 * ATT_ZETA * ATT_WN
        tau = np.array([I[0] * (kp * e_b[0] - kd * w[0]),
                        I[1] * (kp * e_b[1] - kd * w[1]),
                        I[2] * (-0.8 * w[2])])
        gimbal = np.zeros(2)
        rcs = tau.copy()
        if thr_on:
            T = max(st["thr"] * T_MAX, 1.0)
            gimbal = np.array([-tau[0] / (COM_H * T), -tau[1] / (COM_H * T)])
            gimbal = np.clip(np.arcsin(np.clip(gimbal, -1, 1)), -GIMBAL_MAX, GIMBAL_MAX)
            got = np.array([-COM_H * T * math.sin(gimbal[0]), -COM_H * T * math.sin(gimbal[1])])
            rcs[0:2] = tau[0:2] - got
        return gimbal, rcs

    def update(self, lander, st):
        p, v, m = st["p"], st["v"], st["m"]
        h = p[2] - Z_STAND
        vz = v[2]
        rho = lander.env["rho_scale"] * 0 + RHO0 * math.exp(-max(h, 0) / H_SCALE)   # nominal air
        a_net_max = T_MAX / m - G0
        a_ref = A_REF_FRAC * a_net_max
        feet, legs, hull = lander.contacts()
        thr_cmd, grid_dir = 0.0, None
        info = dict(a_ref=a_ref, h=h)

        if feet + legs > 0 or hull:
            self.phase = "LANDED"

        if self.phase == "DESCENT":
            speed = np.linalg.norm(v)
            n_des = -v / speed if speed > 30.0 else np.array([0.0, 0.0, 1.0])
            t_go = max(h / max(-vz, 20.0), 2.0)
            a_xy = -6.0 * p[0:2] / t_go ** 2 - 4.0 * v[0:2] / t_go
            grid_dir = np.array([m * a_xy[0], m * a_xy[1], 0.0])
            h_ign = vz * vz / (2.0 * a_ref) + 0.5 * abs(vz)
            if vz < 0 and h <= h_ign and st["prop"] > 0:
                self.phase = "BURN"
                self.t_ignite = st["t"]
            gimbal, rcs = self.attitude_cmd(st, n_des, False)

        if self.phase == "BURN":
            v_ref = -math.sqrt(V_TOUCH ** 2 + 2.0 * a_ref * max(h, 0.0))   # constant deceleration a_ref
            a_z = a_ref + KV * (v_ref - vz)
            drag_up = 0.5 * rho * CD_AX * A_AX * vz * vz          # nominal drag, pushes up
            T_z = m * (a_z + G0) - drag_up
            T_z = max(T_z, 0.2 * m * G0)
            # Lateral guidance: steer toward a sideways velocity that points at the pad centre,
            # and fade that target to zero near the ground (stopping matters more than the last
            # few metres: the pad is 60 m wide). Gains are kept low because the attitude loop
            # needs about a second to tilt the thrust (a fast gain would make it oscillate).
            w_pos = min(1.0, max(0.0, (h - 20.0) / 150.0))
            v_des = np.clip(-KPOS * w_pos * p[0:2], -15.0, 15.0)
            a_xy = -KVEL * (v[0:2] - v_des)
            F_xy = m * a_xy
            # grid fins do the sideways work while the air is thick; the engine tilt does the rest
            q_nom = 0.5 * rho * float(v @ v)
            fg = np.linalg.norm(F_xy)
            F_grid = F_xy * min(1.0, q_nom * S_GRID / fg) if fg > 1e-9 else np.zeros(2)
            grid_dir = np.array([F_grid[0], F_grid[1], 0.0])
            F_xy = F_xy - F_grid
            lim = T_z * math.tan(TILT_MAX)
            nf = np.linalg.norm(F_xy)
            if nf > lim:
                F_xy = F_xy * lim / nf
            F_des = np.array([F_xy[0], F_xy[1], T_z])
            thr_cmd = float(np.clip(np.linalg.norm(F_des) / T_MAX, THR_MIN, 1.0))
            n_des = F_des / np.linalg.norm(F_des)
            # do not hover or climb: cut the engine if we are nearly stopped close to the ground
            if h < 3.0 and vz > -0.4:
                thr_cmd = 0.0
            gimbal, rcs = self.attitude_cmd(st, n_des, st["thr"] > 0.0)
            info.update(v_ref=v_ref, T_z=T_z)

        if self.phase == "LANDED":
            thr_cmd = 0.0
            gimbal, rcs = np.zeros(2), np.zeros(3)

        self.last = info
        return thr_cmd, gimbal, rcs, grid_dir


# ---------------------------------------------------------------------------
# Running a landing
# ---------------------------------------------------------------------------
class Flight:
    def __init__(self, env=None, record=True):
        self.L = Lander(env)
        self.ap = Autopilot()
        self.record = record
        self.log = {k: [] for k in ["t", "p", "v", "tilt", "w", "thr", "gimbal", "m", "phase",
                                    "vref", "q", "h"]}
        self.touch = None
        self.t_settle_end = None
        self.done = False

    def step(self):
        L, ap = self.L, self.ap
        st = L.state()
        thr_cmd, gimbal, rcs, grid = ap.update(L, st)
        L.advance(thr_cmd, gimbal, rcs, grid)
        if self.record:
            h = st["p"][2] - Z_STAND
            for k, val in [("t", st["t"]), ("p", st["p"]), ("v", st["v"]),
                           ("tilt", rot_to_tilt_deg(st["R"])), ("w", st["w"]), ("thr", L.thr),
                           ("gimbal", np.degrees(L.gimbal)), ("m", st["m"]), ("phase", ap.phase),
                           ("vref", ap.last.get("v_ref", np.nan)),
                           ("q", 0.5 * L.rho(h) * float(st["v"] @ st["v"])), ("h", h)]:
                self.log[k].append(val)
        feet, legs, hull = L.contacts()
        if self.touch is None and (feet + legs > 0 or hull):
            s = L.state()
            self.touch = dict(t=s["t"], vz=s["v"][2], vlat=float(np.hypot(s["v"][0], s["v"][1])),
                              tilt=rot_to_tilt_deg(s["R"]), rate=float(np.linalg.norm(s["w"])),
                              miss=float(np.hypot(s["p"][0], s["p"][1])), prop=s["prop"],
                              hull=hull)
            self.t_settle_end = s["t"] + 8.0
        z = L.data.qpos[2]
        if self.t_settle_end is not None and L.data.time >= self.t_settle_end:
            self.done = True
        if L.data.time > 200.0 or z < 0.5 or not np.all(np.isfinite(L.data.qpos)):
            self.done = True

    def run(self):
        while not self.done:
            self.step()
        return self.result()

    def result(self):
        L = self.L
        s = L.state()
        fin_tilt = rot_to_tilt_deg(s["R"])
        fin_miss = float(np.hypot(s["p"][0], s["p"][1]))
        res = dict(touched=self.touch is not None, ignite=self.ap.t_ignite, prop_left=L.prop,
                   final_tilt=fin_tilt, final_miss=fin_miss, final_z=float(s["p"][2]))
        if self.touch:
            tc = self.touch
            checks = dict(vz=abs(tc["vz"]) <= OK_VZ, vlat=tc["vlat"] <= OK_VLAT,
                          tilt=tc["tilt"] <= OK_TILT, rate=tc["rate"] <= OK_RATE,
                          pad=tc["miss"] <= PAD_RADIUS, upright=fin_tilt <= 10.0 and s["p"][2] > 12.0,
                          nohull=not tc["hull"])
            res.update(touch=tc, checks=checks, success=all(checks.values()))
        else:
            res.update(checks={}, success=False)
        return res


# ---------------------------------------------------------------------------
# Tests of the physics plumbing against exact answers
# ---------------------------------------------------------------------------
def _ok(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def run_tests():
    print("\nTests: is the physics set up correctly?")
    res = []

    # 1. free fall in vacuum
    L = Lander(dict(vacuum=True, pos0=(0, 0, 5000.0), vel0=(0, 0, 0), tilt0_deg=0.0, rate0=(0, 0, 0)))
    for _ in range(500):
        L.advance(0.0, [0, 0], np.zeros(3), None)
    t, z, vz = L.data.time, L.data.qpos[2], L.data.qvel[2]
    res.append(_ok("free fall in vacuum", abs(z - (5000.0 - 0.5 * G0 * t * t)) < 1e-4 and abs(vz + G0 * t) < 1e-6,
                   f"after {t:.1f} s: height error {abs(z - (5000.0 - 0.5 * G0 * t * t)):.2e} m, "
                   f"speed error {abs(vz + G0 * t):.2e} m/s"))

    # 2. rocket equation with gravity (vacuum, full throttle, straight up)
    L = Lander(dict(vacuum=True, pos0=(0, 0, 100.0), vel0=(0, 0, 0), tilt0_deg=0.0, rate0=(0, 0, 0)))
    m0 = L.mass()
    for _ in range(int(10.0 / DT_CTRL)):
        L.advance(1.0, [0, 0], np.zeros(3), None)
    t = L.data.time
    ve = ISP * G0
    v_exact = ve * math.log(m0 / L.mass()) - G0 * t
    res.append(_ok("rocket equation + gravity (10 s burn)", abs(L.data.qvel[2] - v_exact) < 0.15,
                   f"speed {L.data.qvel[2]:.3f} m/s, exact {v_exact:.3f} m/s, "
                   f"propellant used {m0 - L.mass():.1f} kg"))

    # 3. gimbal torque (constant minimum throttle so the thrust does not ramp)
    L = Lander(dict(vacuum=True, pos0=(0, 0, 1000.0), vel0=(0, 0, 0), tilt0_deg=0.0, rate0=(0, 0, 0)))
    gx = math.radians(3.0)
    for _ in range(int(0.2 / DT_CTRL)):
        L.advance(THR_MIN, [gx, 0], np.zeros(3), None)
    T = L.thr * T_MAX
    alpha_exact = -COM_H * T * math.sin(gx) / inertia(L.mass())[0]
    w_exact = alpha_exact * L.data.time
    w_sim = L.data.qvel[3]
    res.append(_ok("engine gimbal torque", abs(w_sim - w_exact) / abs(w_exact) < 0.01,
                   f"rotation rate {w_sim:.5f} rad/s, expected {w_exact:.5f} rad/s"))

    # 4. engines-first flight is aerodynamically stable (restoring torque)
    L = Lander(dict(pos0=(0, 0, 3000.0), vel0=(0, 0, -250.0), tilt0_deg=10.0, tilt_az_deg=0.0, rate0=(0, 0, 0)))
    R = L.data.xmat[L.bid].reshape(3, 3)
    _, tau = L.aero(R, L.data.qvel[0:3], 3000.0 - Z_STAND)
    zb = R[:, 2]
    restoring = float(np.dot(np.cross(zb, np.array([0, 0, 1.0])), tau))
    res.append(_ok("aerodynamic stability (centre of pressure above the COM)", restoring > 0,
                   f"torque pushes the nose back toward the velocity line (restoring component {restoring:.1f} N m)"))

    # 5. drag force matches the formula
    L = Lander(dict(tilt0_deg=0.0, rate0=(0, 0, 0), vel0=(0, 0, -200.0), pos0=(0, 0, 2000.0)))
    R = L.data.xmat[L.bid].reshape(3, 3)
    F, _ = L.aero(R, np.array([0, 0, -200.0]), 2000.0 - Z_STAND)
    rho = RHO0 * math.exp(-(2000.0 - Z_STAND) / H_SCALE)
    F_exact = 0.5 * rho * CD_AX * A_AX * 200.0 ** 2
    res.append(_ok("axial drag formula", abs(F[2] - F_exact) / F_exact < 1e-9,
                   f"{F[2] / 1e3:.2f} kN vs {F_exact / 1e3:.2f} kN"))

    # 6. legs hold the booster up on the ground
    L = Lander(dict(vacuum=True, pos0=(0, 0, Z_STAND + 0.5), vel0=(0, 0, 0), tilt0_deg=0.0, rate0=(0, 0, 0)))
    for _ in range(int(4.0 / DT_CTRL)):
        L.advance(0.0, [0, 0], np.zeros(3), None)
    z, tilt = L.data.qpos[2], rot_to_tilt_deg(L.data.xmat[L.bid].reshape(3, 3))
    res.append(_ok("standing on the legs", abs(z - Z_STAND) < 0.15 and tilt < 0.5,
                   f"COM height {z:.3f} m (standing height {Z_STAND:.3f} m), tilt {tilt:.3f} deg"))
    return all(res)


# ---------------------------------------------------------------------------
# Reports and plots
# ---------------------------------------------------------------------------
def print_report(fl, res, title="LANDING REPORT"):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    ap = fl.ap
    print(f"Engine ignition:        t = {res['ignite']:.1f} s" if res["ignite"] else "Engine never ignited")
    if res["touched"]:
        tc = res["touch"]
        print(f"Touchdown:              t = {tc['t']:.1f} s")
        print(f"  vertical speed        {abs(tc['vz']):6.2f} m/s   (limit {OK_VZ})")
        print(f"  sideways speed        {tc['vlat']:6.2f} m/s   (limit {OK_VLAT})")
        print(f"  tilt                  {tc['tilt']:6.2f} deg   (limit {OK_TILT})")
        print(f"  rotation rate         {tc['rate']:6.3f} rad/s (limit {OK_RATE})")
        print(f"  distance from centre  {tc['miss']:6.1f} m     (pad radius {PAD_RADIUS})")
        print(f"After 8 s:              tilt {res['final_tilt']:.1f} deg, COM height {res['final_z']:.1f} m "
              f"(standing: {Z_STAND:.1f} m)")
    print(f"Propellant left:        {res['prop_left']:.0f} kg of {PROP0:.0f} kg")
    verdict = "SUCCESS: soft upright landing on the pad" if res["success"] else "FAILURE"
    if not res["success"] and res["checks"]:
        verdict += " (" + ", ".join(k for k, v in res["checks"].items() if not v) + ")"
    print(f"Verdict:                {verdict}")


def make_plots(fl, res, path="rocket_landing.png"):
    import matplotlib.pyplot as plt

    lg = {k: np.array(v) for k, v in fl.log.items() if k != "phase"}
    phase = np.array(fl.log["phase"])
    t = lg["t"]
    ign = res["ignite"] or t[-1]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    p = lg["p"]
    ax[0, 0].plot(t, lg["h"] / 1e3)
    ax[0, 0].set_ylabel("height above the pad [km]")
    ax[0, 1].plot(t, lg["v"][:, 2], label="vertical speed")
    ax[0, 1].plot(t, lg["vref"], "--", label="planned speed")
    ax[0, 1].set_ylabel("vertical speed [m/s]")
    ax[0, 1].legend()
    ax[0, 2].plot(p[:, 0], p[:, 1], lw=1.6)
    ax[0, 2].add_patch(plt.Circle((0, 0), PAD_RADIUS, color="gray", alpha=0.35))
    ax[0, 2].plot([p[0, 0]], [p[0, 1]], "go")
    ax[0, 2].plot([p[-1, 0]], [p[-1, 1]], "rx", ms=10)
    ax[0, 2].set_aspect("equal")
    ax[0, 2].set_title("ground track (grey = landing pad)")
    ax[0, 2].set_xlabel("x [m]")
    ax[0, 2].set_ylabel("y [m]")
    burn = (phase == "BURN")
    ax[1, 0].plot(t, lg["thr"] * 100, label="throttle [%]")
    ax[1, 0].plot(t, lg["gimbal"][:, 0], label="gimbal x [deg]")
    ax[1, 0].plot(t, lg["gimbal"][:, 1], label="gimbal y [deg]")
    ax[1, 0].legend()
    ax[1, 0].set_ylabel("engine commands")
    ax[1, 1].plot(t, lg["tilt"])
    ax[1, 1].set_ylabel("tilt from vertical [deg]")
    ax[1, 2].plot(t, lg["m"])
    ax[1, 2].set_ylabel("mass [kg]")
    for a in ax.flat:
        if a is not ax[0, 2]:
            a.axvspan(ign, t[-1], color="orange", alpha=0.10)
            a.set_xlabel("time [s]")
        a.grid(True, alpha=0.3)
    ax[0, 0].set_title("height (orange = engine burning / on the ground)")
    fig.suptitle("MuJoCo booster landing, " + ("SUCCESS" if res["success"] else "FAILURE"))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nSaved plot to {path}")
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def random_env(rng, k=1.0):
    """Random errors for one landing. k scales every spread (k = 2 means twice as bad)."""
    return dict(
        pos0=(-300.0 + rng.normal(0, 100 * k), 150.0 + rng.normal(0, 100 * k), 10000.0),
        vel0=(25.0 + rng.normal(0, 6 * k), -12.0 + rng.normal(0, 6 * k), -300.0 + rng.normal(0, 15 * k)),
        tilt0_deg=abs(rng.normal(8.0, 4.0 * k)), tilt_az_deg=rng.uniform(0, 360),
        rate0=tuple(rng.normal(0, 0.02 * k, 3)),
        wind=(rng.normal(0, 6.0 * k), rng.normal(0, 6.0 * k)),
        thrust_err=rng.normal(0, 0.015 * k), prop_err=rng.normal(0, 0.03 * k),
        dry_err=rng.normal(0, 0.01 * k), cd_scale=max(0.4, rng.normal(1.0, 0.10 * k)),
        rho_scale=max(0.5, rng.normal(1.0, 0.05 * k)),
    )


def monte_carlo(n, seed, stress=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for i in range(n):
        fl = Flight(random_env(rng, stress), record=False)
        res = fl.run()
        rows.append(res)
        tc = res.get("touch")
        print(f"  run {i + 1:3d}/{n}: {'OK  ' if res['success'] else 'FAIL'} "
              + (f"vz {abs(tc['vz']):5.2f} m/s, sideways {tc['vlat']:5.2f} m/s, tilt {tc['tilt']:5.2f} deg, "
                 f"miss {tc['miss']:5.1f} m, fuel left {res['prop_left']:4.0f} kg" if tc else "never touched down"),
              flush=True)
    ok = [r for r in rows if r["success"]]
    print("\n" + "=" * 74)
    print(f"MONTE CARLO (errors x{stress:g}): {len(ok)} of {n} landings succeeded  ({100.0 * len(ok) / n:.0f} %)   "
          f"[{time.time() - t0:.0f} s]")
    print("=" * 74)
    touched = [r for r in rows if r["touched"]]
    if touched:
        def stat(f):
            a = np.array([f(r) for r in touched])
            return f"mean {a.mean():6.2f}, worst {a.max():6.2f}"
        print(f"touchdown vertical speed [m/s]: {stat(lambda r: abs(r['touch']['vz']))}")
        print(f"touchdown sideways speed [m/s]: {stat(lambda r: r['touch']['vlat'])}")
        print(f"touchdown tilt [deg]:           {stat(lambda r: r['touch']['tilt'])}")
        print(f"distance from pad centre [m]:   {stat(lambda r: r['touch']['miss'])}")
        print(f"propellant left [kg]:           mean {np.mean([r['prop_left'] for r in touched]):6.0f}, "
              f"least {min(r['prop_left'] for r in touched):6.0f}")
    fails = {}
    for r in rows:
        if not r["success"]:
            keys = [k for k, v in r["checks"].items() if not v] or ["never landed"]
            for k in keys:
                fails[k] = fails.get(k, 0) + 1
    if fails:
        print("failure reasons (a run can have several): " + ", ".join(f"{k}: {v}" for k, v in fails.items()))
    return rows


# ---------------------------------------------------------------------------
# Interactive viewer and video
# ---------------------------------------------------------------------------
def run_viewer(env, speed):
    import mujoco.viewer

    fl = Flight(env, record=False)
    L = fl.L
    with mujoco.viewer.launch_passive(L.model, L.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = L.bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 120.0, 135.0, -8.0
        print("Viewer open. Drag to rotate, scroll to zoom. Close the window to quit.")
        t_next, last_print = time.time(), -1.0
        while viewer.is_running() and not fl.done:
            fl.step()
            if L.data.time - last_print >= 2.0:
                s = L.state()
                print(f"t={s['t']:6.1f} s  height={s['p'][2] - Z_STAND:8.1f} m  speed={np.linalg.norm(s['v']):6.1f} m/s  "
                      f"throttle={L.thr * 100:4.0f} %  fuel={s['prop']:5.0f} kg  phase={fl.ap.phase}")
                last_print = L.data.time
            viewer.sync()
            t_next += DT_CTRL / speed
            time.sleep(max(0.0, t_next - time.time()))
        res = fl.result()
    print_report(fl, res)


def render_video(env, path, camera="chase", fps=24, width=960, height=544):
    """Render the landing to a video: the long fall is sped up, the last 600 m is real time."""
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    fl = Flight(env, record=False)
    L = fl.L
    renderer = mujoco.Renderer(L.model, height, width)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    t_frame, n = 0.0, 0
    while not fl.done:
        fl.step()
        if fl.touch and L.data.time > fl.touch["t"] + 5.0:
            break
        h = L.data.qpos[2] - Z_STAND
        speedup = 6.0 if h > 600.0 else 1.0
        while t_frame <= L.data.time:
            renderer.update_scene(L.data, camera=camera)
            img = Image.fromarray(renderer.render())
            s = L.state()
            lines = [f"t = {s['t']:5.1f} s" + (f"   (x{speedup:.0f} speed)" if speedup > 1 else "   (real time)"),
                     f"height {h:7.1f} m    speed {np.linalg.norm(s['v']):5.1f} m/s",
                     f"throttle {L.thr * 100:3.0f} %    fuel {s['prop']:4.0f} kg    {fl.ap.phase}"]
            d = ImageDraw.Draw(img)
            d.rectangle((8, 8, 330, 82), fill=(10, 14, 22))
            for i, ln in enumerate(lines):
                d.text((14, 12 + 22 * i), ln, fill=(235, 240, 248), font=font)
            writer.append_data(np.asarray(img))
            n += 1
            t_frame += speedup / fps
    writer.close()
    print(f"Saved {n} frames ({n / fps:.0f} s of video) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SpaceX-style booster landing in MuJoCo")
    ap.add_argument("--view", action="store_true", help="open the interactive 3D viewer")
    ap.add_argument("--mc", type=int, default=0, help="run N randomised landings")
    ap.add_argument("--video", type=str, default="", help="write a video file")
    ap.add_argument("--seed", type=int, default=-1, help="random landing (with --view/--video)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--stress", type=float, default=1.0, help="scale all random errors (e.g. 2 = twice as bad)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("SPACEX-STYLE BOOSTER LANDING IN THE MUJOCO PHYSICS ENGINE")
    print("=" * 74)
    print(f"mujoco {mujoco.__version__}   vehicle: {DRY_MASS:.0f} kg dry + {PROP0:.0f} kg propellant, "
          f"thrust {T_MAX / 1e3:.0f} kN, throttle {THR_MIN * 100:.0f}-100 %")
    w = (DRY_MASS + PROP0) * G0
    print(f"weight {w / 1e3:.0f} kN at the start, minimum thrust {T_MAX * THR_MIN / 1e3:.0f} kN "
          f"-> thrust/weight at minimum throttle = {T_MAX * THR_MIN / w:.2f}  (cannot hover)")

    if not args.no_tests and not run_tests():
        print("\nSOME TESTS FAILED. Do not trust the results below.")
        sys.exit(1)

    env = None
    if args.seed >= 0:
        env = random_env(np.random.default_rng(args.seed), args.stress)

    if args.mc:
        monte_carlo(args.mc, args.seed if args.seed >= 0 else 1, args.stress)
        return
    if args.video:
        render_video(env, args.video)
        return
    if args.view:
        run_viewer(env, args.speed)
        return

    fl = Flight(env)
    res = fl.run()
    print_report(fl, res)
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            make_plots(fl, res)
            plt.show()
        except ImportError:
            print("(matplotlib not installed, skipping plot)")


if __name__ == "__main__":
    main()