#!/usr/bin/env python3
"""
scripts/03_generate_mjcf.py
Aster D51 + 8-20B 煤水车全列闭环 MJCF 编译器 (V2-GEAR 精密走行与华氏全金属连杆版)。
严格读取 configs/d51_mass_props.json，通过旋转变换与平行轴定理生成精确 <inertial fullinertia="..."/>：
  1. 修复 4-12 逆转横轴双摇臂真实 11.0mm 力臂，并通过双圆嵌套求根使左右逆转吊杆 (2-10) 严格等长 (11.0mm) 零残差闭合；
  2. 补齐第 3 主动轮 7-10 回动曲柄、2-5 配气导轨托架、华氏五杆精密不锈钢网格、6-3 给水泵偏心环及 19-2/19-11 铸铁台车架；
  3. 实现 7-12 / 7-15 四轴耦合连杆前/后铰接节 (Knuckle Pin)，彻底释放四轴跷跷板均衡梁独立垂向悬挂自由度。
"""

import argparse
import json
import math
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def full_to_mat3(fi: List[float]) -> np.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = fi
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)


def mat3_to_full_str(m: np.ndarray) -> str:
    ixx, iyy, izz = max(1e-9, m[0, 0]), max(1e-9, m[1, 1]), max(1e-9, m[2, 2])
    return f"{ixx:.9e} {iyy:.9e} {izz:.9e} {m[0,1]:.9e} {m[0,2]:.9e} {m[1,2]:.9e}"


def rot_y_mat(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def attach_part_inertial(
    body_el: ET.Element,
    mp: Dict[str, Any],
    offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_y_rad: float = 0.0,
) -> None:
    """将单个 CAD 零件关于质心的惯量经绕 Y 轴旋转与平移后写入 body 的 <inertial>。"""
    R = rot_y_mat(pitch_y_rad)
    com_local = R @ np.array(mp["com_m"], dtype=np.float64) + np.array(offset_xyz, dtype=np.float64)
    I_rot = R @ full_to_mat3(mp["fullinertia"]) @ R.T
    ET.SubElement(
        body_el,
        "inertial",
        pos=f"{com_local[0]:.6f} {com_local[1]:.6f} {com_local[2]:.6f}",
        mass=f"{mp['mass_kg']:.6f}",
        fullinertia=mat3_to_full_str(I_rot),
    )


def attach_composite_inertial(
    body_el: ET.Element, components: List[Tuple[float, np.ndarray, np.ndarray]]
) -> None:
    """利用平行轴定理合并同一刚体下多个子零件的真实偏心质心与 3x3 惯量张量。"""
    total_m = sum(m for m, _, _ in components)
    com_total = sum(m * c for m, c, _ in components) / total_m
    I_total = np.zeros((3, 3), dtype=np.float64)
    for m, c, I_c in components:
        d = c - com_total
        I_total += I_c + m * (float(np.dot(d, d)) * np.eye(3) - np.outer(d, d))
    ET.SubElement(
        body_el,
        "inertial",
        pos=f"{com_total[0]:.6f} {com_total[1]:.6f} {com_total[2]:.6f}",
        mass=f"{total_m:.6f}",
        fullinertia=mat3_to_full_str(I_total),
    )


def intersect_circles(
    c1: Tuple[float, float], r1: float, c2: Tuple[float, float], r2: float, pick_upper: bool = False
) -> Tuple[float, float]:
    dx, dz = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dz)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0.0:
        raise ValueError(f"连杆几何无法闭合: d={d*1000:.2f}mm")
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(0.0, r1**2 - a**2))
    mx, mz = c1[0] + a * (dx / d), c1[1] + a * (dz / d)
    rx, rz = -dz * (h / d), dx * (h / d)
    p_a, p_b = (mx + rx, mz + rz), (mx - rx, mz - rz)
    return (p_a if p_a[1] >= p_b[1] else p_b) if pick_upper else (p_a if p_a[1] <= p_b[1] else p_b)


def exp_link_slot_tilt(theta_rad: float, cfg: Dict) -> float:
    """月牙板从 t=0 姿态转到摆动中位所需的角度 (rad)，用于把圆弧滑块槽对齐到中位铅垂方向。"""
    s = 1e-3
    vg_cfg = cfg["walschaerts_valve_gear"]
    trun = (vg_cfg["trunnion_x"] * s, vg_cfg["trunnion_z"] * s)
    r_ret = vg_cfg["return_crank_radius"] * s

    def tail_angle(th_ret: float) -> float:
        e = intersect_circles(
            (r_ret * math.cos(th_ret), r_ret * math.sin(th_ret)), vg_cfg["eccentric_rod_length"] * s,
            trun, vg_cfg["exp_link_tail_radius"] * s, pick_upper=False,
        )
        return math.atan2(e[1] - trun[1], e[0] - trun[0])

    th_fdc = math.radians(vg_cfg["return_crank_phase_deg"])
    mid = 0.5 * (tail_angle(th_fdc) + tail_angle(th_fdc + math.pi))
    return mid - tail_angle(theta_rad + th_fdc)


def _solve_comb_for_die_pin(
    u0: Tuple[float, float], s_die: Tuple[float, float], cfg: Dict
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """给定十字头下耳轴 u0 与滑块销 s_die，精确求根合并杆摆角 phi，返回 (v_pin, m_pin, u1_pin, rad_lift)。"""
    s = 1e-3
    rc_cfg, vg_cfg = cfg["cylinder_and_rods"], cfg["walschaerts_valve_gear"]
    z_valve = rc_cfg["valve_axis_z"] * s
    L_rad = vg_cfg["radius_rod_length"] * s
    L_union = vg_cfg["union_link_length"] * s
    L_comb = vg_cfg["comb_lever_total_length"] * s
    L_mv = vg_cfg["comb_lever_lap_lead_offset"] * s
    L_back = vg_cfg["radius_rod_back_ext"] * s

    def residual(phi: float):
        mz = z_valve - L_mv * math.cos(phi)
        mx = s_die[0] + math.sqrt(max(0.0, L_rad**2 - (mz - s_die[1]) ** 2))
        vx = mx + L_mv * math.sin(phi)
        u1 = (vx - L_comb * math.sin(phi), z_valve - L_comb * math.cos(phi))
        return math.hypot(u1[0] - u0[0], u1[1] - u0[1]) - L_union, (vx, z_valve), (mx, mz), u1

    grid = [-0.75 + i * 1.5 / 200.0 for i in range(201)]
    vals = [residual(g)[0] for g in grid]
    lo, hi = -0.3, 0.3
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] <= 0:
            lo, hi = grid[i], grid[i + 1]
            break

    f_lo = residual(lo)[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)[0]
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    _, v_pin, m_pin, u1_pin = residual(0.5 * (lo + hi))
    dir_x, dir_z = (m_pin[0] - s_die[0]) / L_rad, (m_pin[1] - s_die[1]) / L_rad
    rad_lift = (s_die[0] - L_back * dir_x, s_die[1] - L_back * dir_z)
    return v_pin, m_pin, u1_pin, rad_lift


def solve_side_pose(theta_rad: float, cfg: Dict) -> Dict[str, Any]:
    """
    求解单侧驱动与华氏配气机构在 t=0 (j_reverser=0 中立位) 的机器精度零残差位姿:
      - 严格锁死左右逆转摇臂孔距 reverser_arm_length = 11.0mm;
      - 严格锁死左右逆转吊杆孔距 lift_link_length = 11.0mm (外层二分求根圆弧滑块初始微倾角 q_die)。
    """
    s = 1e-3
    rc_cfg, vg_cfg = cfg["cylinder_and_rods"], cfg["walschaerts_valve_gear"]
    r_crank, L_main = rc_cfg["crank_radius"] * s, rc_cfg["main_rod_length"] * s
    z_cyl = rc_cfg["cylinder_center_z"] * s
    L_rad = vg_cfg["radius_rod_length"] * s

    cx, cz = r_crank * math.cos(theta_rad), r_crank * math.sin(theta_rad)
    px, pz = cx + math.sqrt(L_main**2 - (z_cyl - cz) ** 2), z_cyl
    u0 = (px, pz - vg_cfg["crosshead_drop"] * s)

    theta_ret = theta_rad + math.radians(vg_cfg["return_crank_phase_deg"])
    r_ret = vg_cfg["return_crank_radius"] * s
    rx, rz = r_ret * math.cos(theta_ret), r_ret * math.sin(theta_ret)

    trun = (vg_cfg["trunnion_x"] * s, vg_cfg["trunnion_z"] * s)
    e_tail = intersect_circles(
        (rx, rz), vg_cfg["eccentric_rod_length"] * s, trun, vg_cfg["exp_link_tail_radius"] * s, pick_upper=False
    )

    # 先求前死点 (theta=0, s_die=trun) 处心向棒基准仰角 alpha_dc，锁死圆弧月牙槽几何圆心
    px_dc = r_crank + L_main
    u0_dc = (px_dc, z_cyl - vg_cfg["crosshead_drop"] * s)
    _, m_dc, _, _ = _solve_comb_for_die_pin(u0_dc, trun, cfg)
    alpha_dc = math.atan2(-(m_dc[1] - trun[1]), m_dc[0] - trun[0])
    slot_tilt = exp_link_slot_tilt(theta_rad, cfg) + alpha_dc
    cx_arc = L_rad * math.cos(slot_tilt)
    cz_arc = -L_rad * math.sin(slot_tilt)

    arm_tip = (
        (vg_cfg["reverser_shaft_x"] + vg_cfg["reverser_arm_length"]) * s,
        vg_cfg["reverser_shaft_z"] * s,
    )
    L_lift = vg_cfg["lift_link_length"] * s

    def lift_closure_err(q_die: float):
        c, sn = math.cos(q_die), math.sin(q_die)
        vx0, vz0 = -cx_arc, -cz_arc
        s_die = (trun[0] + cx_arc + (vx0 * c + vz0 * sn), trun[1] + cz_arc + (-vx0 * sn + vz0 * c))
        v_p, m_p, u1_p, rl_p = _solve_comb_for_die_pin(u0, s_die, cfg)
        err = math.hypot(rl_p[0] - arm_tip[0], rl_p[1] - arm_tip[1]) - L_lift
        return err, s_die, v_p, m_p, u1_p, rl_p

    lo_q, hi_q = -0.06, 0.06
    f_lo_q = lift_closure_err(lo_q)[0]
    for _ in range(60):
        mid_q = 0.5 * (lo_q + hi_q)
        f_mid_q = lift_closure_err(mid_q)[0]
        if f_lo_q * f_mid_q <= 0.0:
            hi_q = mid_q
        else:
            lo_q, f_lo_q = mid_q, f_mid_q

    q_die_sol = 0.5 * (lo_q + hi_q)
    _, s_die, v_pin, m_pin, u1_pin, rad_lift = lift_closure_err(q_die_sol)

    return {
        "crank": (cx, cz),
        "crosshead": (px, pz),
        "drop_arm": u0,
        "ret_crank": (rx, rz),
        "exp_trunnion": trun,
        "exp_tail": e_tail,
        "die_pin": s_die,
        "q_die": q_die_sol,
        "arc_center_local": (cx_arc, cz_arc),
        "arm_tip": arm_tip,
        "valve_pin": v_pin,
        "comb_mid": m_pin,
        "comb_bot": u1_pin,
        "rad_lift": rad_lift,
    }


def add_rail_wheel_colliders(
    wbody: ET.Element,
    name_prefix: str,
    r_tread: float,
    y_gauge_half: float,
    use_explicit_inertial: bool,
    is_driving: bool = True,
) -> None:
    m_in = "0" if use_explicit_inertial else "0.038"
    m_out = "0" if use_explicit_inertial else "0.016"
    m_flg = "0" if use_explicit_inertial else "0.010"
    alpha = "0" if use_explicit_inertial else "1"
    grp = "3" if use_explicit_inertial else "0"
    tread_fric = "0.46 0.003 0.0001" if is_driving else "0.14 0.002 0.0001"
    tread_prio = "0" if is_driving else "1"

    for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
        y_tread_in = sgn * (y_gauge_half - 0.0006)
        y_tread_mid = sgn * (y_gauge_half + 0.0008)
        y_tread_out = sgn * (y_gauge_half + 0.0024)
        ET.SubElement(
            wbody, "geom", name=f"{name_prefix}_tread_in_{s_tag}", type="cylinder",
            fromto=f"0 {y_tread_in:.6f} 0  0 {y_tread_mid:.6f} 0",
            size=f"{r_tread + 0.00008:.6f}", contype="2", conaffinity="1",
            priority=tread_prio, friction=tread_fric,
            rgba=f"0.18 0.18 0.20 {alpha}", group=grp, mass=m_in
        )
        ET.SubElement(
            wbody, "geom", name=f"{name_prefix}_tread_out_{s_tag}", type="cylinder",
            fromto=f"0 {y_tread_mid:.6f} 0  0 {y_tread_out:.6f} 0",
            size=f"{r_tread - 0.00005:.6f}", contype="2", conaffinity="1",
            priority=tread_prio, friction=tread_fric,
            rgba=f"0.16 0.16 0.18 {alpha}", group=grp, mass=m_out
        )
        y_flg_in = sgn * (y_gauge_half - 0.0018)
        y_flg_out = sgn * (y_gauge_half - 0.0011)
        ET.SubElement(
            wbody, "geom", name=f"{name_prefix}_flange_{s_tag}", type="cylinder",
            fromto=f"0 {y_flg_in:.6f} 0  0 {y_flg_out:.6f} 0",
            size=f"{r_tread + 0.00165:.6f}", contype="2", conaffinity="1", priority="1", friction="0.06 0.001 0.0001",
            rgba=f"0.25 0.25 0.28 {alpha}", group=grp, mass=m_flg
        )


def add_r2m_circular_track(world: ET.Element, R_center: float, y_gauge_half: float, n_seg: int = 144) -> None:
    rail_half_w = 0.0008
    r_outer_rail = R_center + (y_gauge_half + rail_half_w)
    r_inner_rail = R_center - (y_gauge_half + rail_half_w)
    n_rail_seg = max(n_seg, 216)
    dphi = (2.0 * math.pi) / n_rail_seg
    z_rail_axis = 0.004 - rail_half_w

    track_root = ET.SubElement(world, "body", name="r2m_track_layout", pos="0 0 0")
    for i in range(n_rail_seg):
        phi0 = 0.5 * math.pi - i * dphi
        phi1 = 0.5 * math.pi - (i + 1) * dphi
        phi_mid = 0.5 * (phi0 + phi1)
        yaw = phi_mid - 0.5 * math.pi
        for tag, r_rail in [("outer_L", r_outer_rail), ("inner_R", r_inner_rail)]:
            x0, y0 = r_rail * math.cos(phi0), -R_center + r_rail * math.sin(phi0)
            x1, y1 = r_rail * math.cos(phi1), -R_center + r_rail * math.sin(phi1)
            ET.SubElement(
                track_root, "geom", name=f"rail_{tag}_{i:03d}", type="capsule",
                fromto=f"{x0:.6f} {y0:.6f} {z_rail_axis:.6f}  {x1:.6f} {y1:.6f} {z_rail_axis:.6f}",
                size=f"{rail_half_w:.6f}",
                contype="1", conaffinity="2", friction="0.46 0.002 0.0001", rgba="0.70 0.72 0.75 1"
            )
        if i % 2 == 0:
            sx, sy = R_center * math.cos(phi_mid), -R_center + R_center * math.sin(phi_mid)
            ET.SubElement(
                track_root, "geom", type="box",
                pos=f"{sx:.6f} {sy:.6f} 0.0006", euler=f"0 0 {yaw:.6f}",
                size="0.0045 0.036 0.0006", rgba="0.28 0.20 0.14 1", contype="0", conaffinity="0"
            )


def build_mjcf(cfg: Dict, mass_props: Dict[str, Any], mode: str = "roller", include_tender: bool = True) -> str:
    s = 1e-3
    parts_mp = mass_props["parts"]
    rho_sus = mass_props["meta"]["densities_kg_m3"]["stainless_SUS"]
    rho_iron = mass_props["meta"]["densities_kg_m3"]["cast_iron_FC"]

    w_cfg = cfg["wheels_and_axles"]
    rc_cfg = cfg["cylinder_and_rods"]
    vg_cfg = cfg["walschaerts_valve_gear"]
    b_cfg = cfg["boiler_and_aux"]
    t_cfg = cfg["tender_and_track"]

    r_drv = 0.5 * w_cfg["driving_wheel_diameter"] * s
    r_trk = 0.5 * w_cfg["leading_wheel_diameter"] * s
    r_tdr = 0.5 * t_cfg["tender_wheel_diameter"] * s
    r_crank = rc_cfg["crank_radius"] * s
    y_gauge_half = 0.5 * cfg["meta"]["track_gauge"] * s

    mj = ET.Element("mujoco", model="aster_d51_consist")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", meshdir="meshes")
    ET.SubElement(mj, "option", timestep="0.0005", integrator="implicitfast", gravity="0 0 -9.81")

    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "geom", condim="3", solref="0.0015 1.0", solimp="0.95 0.99 0.0001")
    ET.SubElement(default, "joint", solreflimit="0.002 1.0", solimplimit="0.98 0.995 0.0001")
    ET.SubElement(default, "equality", solref="0.00045 1.0", solimp="0.997 0.9997 0.000002")

    mesh_dir = Path("models/meshes")
    asset = ET.SubElement(mj, "asset")
    for m_name in parts_mp.keys():
        if (mesh_dir / f"{m_name}.stl").exists():
            ET.SubElement(asset, "mesh", name=f"m_{m_name}", file=f"{m_name}.stl", scale="0.001 0.001 0.001")
    ET.SubElement(asset, "material", name="mat_black_iron", rgba="0.14 0.14 0.16 1", specular="0.6", shininess="0.5")
    ET.SubElement(asset, "material", name="mat_machined_steel", rgba="0.78 0.80 0.84 1", specular="0.9", shininess="0.8")
    ET.SubElement(asset, "material", name="mat_brass", rgba="0.82 0.62 0.24 1", specular="0.85", shininess="0.75")
    ET.SubElement(asset, "material", name="mat_copper", rgba="0.78 0.43 0.27 1", specular="0.88", shininess="0.78")
    ET.SubElement(asset, "material", name="mat_silicone_hose", rgba="0.92 0.94 0.96 0.68", specular="0.4", shininess="0.3")
    ET.SubElement(asset, "material", name="mat_viton_hose", rgba="0.11 0.11 0.13 1", specular="0.25", shininess="0.2")
    ET.SubElement(asset, "material", name="mat_pyrex_glass", rgba="0.82 0.95 0.99 0.36", specular="0.95", shininess="0.92")
    ET.SubElement(asset, "material", name="mat_gauge_dial", rgba="0.96 0.96 0.94 1", specular="0.35", shininess="0.25")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 -1.5 2.2", dir="0 0.4 -0.9", diffuse="0.92 0.92 0.92")
    ET.SubElement(
        world, "geom", name="ground", type="plane", size="4.5 4.5 0.05",
        rgba="0.16 0.18 0.20 1", contype="0", conaffinity="0"
    )

    on_rails = mode in ("track", "curve_r2m")
    rail_top_z = 0.004
    z_loco_axle = (r_drv + 0.020) if mode == "roller" else (rail_top_z + r_drv + 0.0002)

    if mode == "track":
        rail_half_w = 0.0008
        for side_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            y_r = sgn * (y_gauge_half + rail_half_w)
            ET.SubElement(
                world, "geom", name=f"rail_{side_tag}", type="box",
                pos=f"1.2 {y_r:.6f} 0.002", size=f"3.5 {rail_half_w:.6f} 0.002",
                contype="1", conaffinity="2", friction="0.46 0.002 0.0001", rgba="0.70 0.72 0.75 1"
            )
    elif mode == "curve_r2m":
        add_r2m_circular_track(world, t_cfg["curve_radius_mm"] * s, y_gauge_half, int(t_cfg["curve_segments"]))

    # ==================== A. 蒸汽机车车架、2-5 配气托架与锅炉上部总成 ====================
    R_curve = t_cfg["curve_radius_mm"] * s
    if mode == "curve_r2m":
        x_drv_mid = 0.5 * (w_cfg["axle_x"]["drive_1"] + w_cfg["axle_x"]["drive_4"]) * s
        half_dwb = 0.5 * (w_cfg["axle_x"]["drive_1"] - w_cfg["axle_x"]["drive_4"]) * s
        sag_drv = R_curve - math.sqrt(R_curve**2 - half_dwb**2)
        loco_wx, loco_wy = -x_drv_mid, -0.5 * sag_drv
    else:
        loco_wx, loco_wy = 0.0, 0.0

    frame = ET.SubElement(world, "body", name="loco_frame", pos=f"{loco_wx:.6f} {loco_wy:.6f} {z_loco_axle:.6f}")
    if on_rails:
        ET.SubElement(frame, "freejoint", name="loco_free")

    y_vg_abs = rc_cfg["cylinder_center_y"] * s + 0.0035
    tx_m, tz_m = vg_cfg["trunnion_x"] * s, vg_cfg["trunnion_z"] * s

    mp_boil = parts_mp["boiler_superstructure"]
    mp_cu_core = parts_mp["boiler_firebox_flues"]
    mp_sh_pipe = parts_mp["superheater_piping"]
    mp_exh_pipe = parts_mp["exhaust_blast_pipe"]
    mp_blw_17_3 = parts_mp["blower_nozzle_17_3"]
    mp_pet = parts_mp["smokebox_petticoat"]
    mp_tsv = parts_mp["tandem_safety_valves"]
    mp_fw_pipe = parts_mp["feedwater_piping"]
    mp_lub = parts_mp["displacement_lubricator"]
    mp_burner = parts_mp["alcohol_burner"]
    mp_aux_valves = parts_mp["boiler_aux_valves"]
    mp_eq_brakes = parts_mp["chassis_equalizer_brakes"]
    mp_cylL, mp_cylR = parts_mp["cylinder_block_L"], parts_mp["cylinder_block_R"]
    mp_sbL, mp_sbR = parts_mp["slide_bar_L"], parts_mp["slide_bar_R"]
    mp_ptsL, mp_ptsR = parts_mp["piston_tail_sheath_L"], parts_mp["piston_tail_sheath_R"]
    mp_vssL, mp_vssR = parts_mp["valve_spindle_sheath_L"], parts_mp["valve_spindle_sheath_R"]
    mp_cab_bh = parts_mp["cab_backhead_plate"]
    mp_cab_dk = parts_mp["cab_deck_and_mounts"]
    mp_sg_tube = parts_mp["cab_sight_glass_tube"]
    mp_pg_dial = parts_mp["cab_pg_dial"]
    mp_rev_stand = parts_mp["cab_reverser_stand"]
    mp_mbL, mp_mbR = parts_mp["motion_bracket_L"], parts_mp["motion_bracket_R"]
    cyl_pos_L = np.array([rc_cfg["cylinder_center_x"] * s, +rc_cfg["cylinder_center_y"] * s, rc_cfg["cylinder_center_z"] * s])
    cyl_pos_R = np.array([rc_cfg["cylinder_center_x"] * s, -rc_cfg["cylinder_center_y"] * s, rc_cfg["cylinder_center_z"] * s])
    mb_pos_L = np.array([tx_m, +y_vg_abs, tz_m])
    mb_pos_R = np.array([tx_m, -y_vg_abs, tz_m])
    rev_stand_pos = np.array([-0.127500, +0.032500, 0.020000])
    sg_tube_pos = np.array([-0.118500, -0.031300, 0.046000])
    pg_dial_pos = np.array([-0.122050, -0.015400, 0.067040])

    m_side_plate = (0.330 * 0.0035 * 0.016) * rho_iron
    I_side_plate = (m_side_plate / 12.0) * np.diag([0.0035**2 + 0.016**2, 0.330**2 + 0.016**2, 0.330**2 + 0.0035**2])
    m_boiler_water = 0.210  # 50% 标准水位锅水质量 (210 ml)
    r_core, L_core = 0.5 * b_cfg["boiler_front_id"] * s, 0.204
    I_boiler_water = np.diag([
        0.5 * m_boiler_water * (r_core**2),
        (m_boiler_water / 12.0) * (3.0 * (r_core**2) + L_core**2),
        (m_boiler_water / 12.0) * (3.0 * (r_core**2) + L_core**2),
    ])
    m_front_saddle = 0.235
    I_front_saddle = (m_front_saddle / 12.0) * np.diag([0.042**2 + 0.018**2, 0.055**2 + 0.018**2, 0.055**2 + 0.042**2])
    attach_composite_inertial(
        frame,
        [
            (mp_boil["mass_kg"], np.array(mp_boil["com_m"]), full_to_mat3(mp_boil["fullinertia"])),
            (mp_cu_core["mass_kg"], np.array(mp_cu_core["com_m"]), full_to_mat3(mp_cu_core["fullinertia"])),
            (mp_sh_pipe["mass_kg"], np.array(mp_sh_pipe["com_m"]), full_to_mat3(mp_sh_pipe["fullinertia"])),
            (mp_exh_pipe["mass_kg"], np.array(mp_exh_pipe["com_m"]), full_to_mat3(mp_exh_pipe["fullinertia"])),
            (mp_blw_17_3["mass_kg"], np.array(mp_blw_17_3["com_m"]), full_to_mat3(mp_blw_17_3["fullinertia"])),
            (mp_pet["mass_kg"], np.array(mp_pet["com_m"]), full_to_mat3(mp_pet["fullinertia"])),
            (mp_tsv["mass_kg"], np.array(mp_tsv["com_m"]), full_to_mat3(mp_tsv["fullinertia"])),
            (mp_fw_pipe["mass_kg"], np.array(mp_fw_pipe["com_m"]), full_to_mat3(mp_fw_pipe["fullinertia"])),
            (mp_lub["mass_kg"], np.array(mp_lub["com_m"]), full_to_mat3(mp_lub["fullinertia"])),
            (mp_burner["mass_kg"], np.array(mp_burner["com_m"]), full_to_mat3(mp_burner["fullinertia"])),
            (mp_aux_valves["mass_kg"], np.array(mp_aux_valves["com_m"]), full_to_mat3(mp_aux_valves["fullinertia"])),
            (mp_cab_bh["mass_kg"], np.array(mp_cab_bh["com_m"]), full_to_mat3(mp_cab_bh["fullinertia"])),
            (mp_cab_dk["mass_kg"], np.array(mp_cab_dk["com_m"]), full_to_mat3(mp_cab_dk["fullinertia"])),
            (mp_sg_tube["mass_kg"], np.array(mp_sg_tube["com_m"]) + sg_tube_pos, full_to_mat3(mp_sg_tube["fullinertia"])),
            (mp_pg_dial["mass_kg"], np.array(mp_pg_dial["com_m"]) + pg_dial_pos, full_to_mat3(mp_pg_dial["fullinertia"])),
            (mp_eq_brakes["mass_kg"], np.array(mp_eq_brakes["com_m"]), full_to_mat3(mp_eq_brakes["fullinertia"])),
            (mp_cylL["mass_kg"], np.array(mp_cylL["com_m"]) + cyl_pos_L, full_to_mat3(mp_cylL["fullinertia"])),
            (mp_cylR["mass_kg"], np.array(mp_cylR["com_m"]) + cyl_pos_R, full_to_mat3(mp_cylR["fullinertia"])),
            (mp_sbL["mass_kg"], np.array(mp_sbL["com_m"]) + cyl_pos_L, full_to_mat3(mp_sbL["fullinertia"])),
            (mp_sbR["mass_kg"], np.array(mp_sbR["com_m"]) + cyl_pos_R, full_to_mat3(mp_sbR["fullinertia"])),
            (mp_ptsL["mass_kg"], np.array(mp_ptsL["com_m"]) + cyl_pos_L, full_to_mat3(mp_ptsL["fullinertia"])),
            (mp_ptsR["mass_kg"], np.array(mp_ptsR["com_m"]) + cyl_pos_R, full_to_mat3(mp_ptsR["fullinertia"])),
            (mp_vssL["mass_kg"], np.array(mp_vssL["com_m"]) + cyl_pos_L, full_to_mat3(mp_vssL["fullinertia"])),
            (mp_vssR["mass_kg"], np.array(mp_vssR["com_m"]) + cyl_pos_R, full_to_mat3(mp_vssR["fullinertia"])),
            (mp_rev_stand["mass_kg"], np.array(mp_rev_stand["com_m"]) + rev_stand_pos, full_to_mat3(mp_rev_stand["fullinertia"])),
            (mp_mbL["mass_kg"], np.array(mp_mbL["com_m"]) + mb_pos_L, full_to_mat3(mp_mbL["fullinertia"])),
            (mp_mbR["mass_kg"], np.array(mp_mbR["com_m"]) + mb_pos_R, full_to_mat3(mp_mbR["fullinertia"])),
            (m_side_plate, np.array([0.020, +0.0135, 0.004]), I_side_plate),
            (m_side_plate, np.array([0.020, -0.0135, 0.004]), I_side_plate),
            (m_boiler_water, np.array([0.010, 0.0, 0.042]), I_boiler_water),
            (m_front_saddle, np.array([0.145, 0.0, 0.008]), I_front_saddle),
        ],
    )
    for s_tag, sgn_f in [("L", +1.0), ("R", -1.0)]:
        ET.SubElement(
            frame, "geom", name=f"main_chassis_{s_tag}", type="box",
            pos=f"0.020 {sgn_f * 0.0135:.6f} 0.004", size="0.165 0.00175 0.008",
            material="mat_black_iron", rgba="0.12 0.12 0.14 1", mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            frame, "geom", name=f"motion_bracket_vis_{s_tag}", type="mesh",
            mesh=f"m_motion_bracket_{s_tag}", material="mat_black_iron",
            pos=f"{tx_m:.6f} {sgn_f * y_vg_abs:.6f} {tz_m:.6f}", mass="0", contype="0", conaffinity="0"
        )
    # 8-6 / 8-7 车架尾部牵引横梁框与 8-8 牵引立销 (衔接两侧主车架板，X-Ray 模式下保持完整实体不透明)
    ET.SubElement(
        frame, "geom", name="loco_rear_dragbox", type="box",
        pos="-0.144000 0.000000 -0.002900", size="0.006200 0.015000 0.001600",
        material="mat_black_iron", rgba="0.12 0.12 0.14 1", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="loco_drawbar_pin_vis", type="cylinder",
        fromto="-0.146000 0.000000 -0.009200  -0.146000 0.000000 -0.001200", size="0.001800",
        material="mat_brass", rgba="0.82 0.62 0.24 1", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="chassis_equalizer_brakes_vis", type="mesh",
        mesh="m_chassis_equalizer_brakes", material="mat_black_iron",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    # P0-D + Stage 14 真实四回路锅炉内脏、烟室独立分件与车架内双水路/置换注油器参数化 CAD
    ET.SubElement(
        frame, "geom", name="boiler_copper_core_vis", type="mesh",
        mesh="m_boiler_firebox_flues", material="mat_copper",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="superheater_piping_vis", type="mesh",
        mesh="m_superheater_piping", material="mat_machined_steel",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="exhaust_blast_pipe_vis", type="mesh",
        mesh="m_exhaust_blast_pipe", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="blower_nozzle_17_3_vis", type="mesh",
        mesh="m_blower_nozzle_17_3", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="smokebox_petticoat_vis", type="mesh",
        mesh="m_smokebox_petticoat", material="mat_machined_steel",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="tandem_safety_valves_vis", type="mesh",
        mesh="m_tandem_safety_valves", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="feedwater_piping_vis", type="mesh",
        mesh="m_feedwater_piping", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="displacement_lubricator_vis", type="mesh",
        mesh="m_displacement_lubricator", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="alcohol_burner_vis", type="mesh",
        mesh="m_alcohol_burner", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="boiler_aux_valves_vis", type="mesh",
        mesh="m_boiler_aux_valves", material="mat_brass",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="cab_backhead_plate_vis", type="mesh",
        mesh="m_cab_backhead_plate", material="mat_black_iron",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="cab_deck_and_mounts_vis", type="mesh",
        mesh="m_cab_deck_and_mounts", material="mat_black_iron",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="cab_sight_glass_tube_vis", type="mesh",
        mesh="m_cab_sight_glass_tube", material="mat_pyrex_glass",
        pos="-0.118500 -0.031300 0.046000", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="cab_pg_dial_vis", type="mesh",
        mesh="m_cab_pg_dial", material="mat_gauge_dial",
        pos="-0.122050 -0.015400 0.067040", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="cab_reverser_stand_vis", type="mesh",
        mesh="m_cab_reverser_stand", material="mat_black_iron",
        pos="-0.127500 0.032500 0.020000", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(
        frame, "geom", name="boiler_Super_vis", type="mesh",
        mesh="m_boiler_superstructure", material="mat_black_iron",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )

    equality = ET.SubElement(mj, "equality")
    tendon = ET.SubElement(mj, "tendon")
    actuator = ET.SubElement(mj, "actuator")
    sensor = ET.SubElement(mj, "sensor")

    # 预先求解左右两侧 t=0 位姿 (用于第 3 主动轮 7-10 回动曲柄与两侧闭环装配)
    q_rad_R = math.radians(w_cfg["quartering_deg_R_minus_L"])
    side_poses = {
        "L": solve_side_pose(0.0, cfg),
        "R": solve_side_pose(q_rad_R, cfg),
    }

    # 1. 四对 Boxpok 驱动动轮 (第 3 主动轮采用加长曲柄销 + 7-10 回动曲柄复合惯量)
    axle_xs = [
        (1, w_cfg["axle_x"]["drive_1"] * s, w_cfg["axle_lateral_play_half"]["drive_1"] * s),
        (2, w_cfg["axle_x"]["drive_2"] * s, w_cfg["axle_lateral_play_half"]["drive_2"] * s),
        (3, w_cfg["axle_x"]["drive_3"] * s, w_cfg["axle_lateral_play_half"]["drive_3"] * s),
        (4, w_cfg["axle_x"]["drive_4"] * s, max(0.0004, w_cfg["axle_lateral_play_half"]["drive_4"] * s)),
    ]
    y_w_off = y_gauge_half + 0.0008
    m_axle_rod = math.pi * (0.003**2) * (2.0 * y_gauge_half) * rho_sus
    I_axle_rod = np.diag([1e-6, 0.5 * m_axle_rod * (0.003**2), 1e-6])

    wheel_bodies: Dict[int, ET.Element] = {}
    for idx, ax_x, lat_play in axle_xs:
        abox = ET.SubElement(frame, "body", name=f"axlebox_{idx}", pos=f"{ax_x:.6f} 0 0")
        attach_part_inertial(abox, parts_mp["driving_axlebox"])
        ET.SubElement(
            abox, "joint", name=f"j_susp_z{idx}", type="slide", axis="0 0 1",
            springref="-0.0017" if on_rails else "0",
            limited="true", range="-0.0035 0.0035", stiffness="950", damping="14"
        )
        ET.SubElement(
            abox, "joint", name=f"j_lat_y{idx}", type="slide", axis="0 1 0",
            limited="true", range=f"{-lat_play:.6f} {lat_play:.6f}", stiffness="85", damping="3.5"
        )
        ET.SubElement(
            abox, "geom", name=f"abox_vis_{idx}", type="mesh", mesh="m_driving_axlebox",
            material="mat_brass", contype="0", conaffinity="0", mass="0"
        )

        wbody = ET.SubElement(abox, "body", name=f"wheelset_{idx}", pos="0 0 0")
        ET.SubElement(wbody, "joint", name=f"j_wheel_{idx}", type="hinge", axis="0 1 0", damping="0.0012")

        is_main = idx == 3
        w_key_L = "main_driving_wheel_L" if is_main else "driving_wheel_L"
        w_key_R = "main_driving_wheel_R" if is_main else "driving_wheel_R"
        mp_wL, mp_wR = parts_mp[w_key_L], parts_mp[w_key_R]
        w_comps = [
            (mp_wL["mass_kg"], np.array(mp_wL["com_m"]) + np.array([0.0, +y_w_off, 0.0]), full_to_mat3(mp_wL["fullinertia"])),
            (mp_wR["mass_kg"], np.array(mp_wR["com_m"]) + np.array([0.0, -y_w_off, 0.0]), full_to_mat3(mp_wR["fullinertia"])),
            (m_axle_rod, np.zeros(3), I_axle_rod),
        ]
        if is_main:
            for s_tag, sgn in [("L", +1.0), ("R", -1.0)]:
                cx_m, cz_m = side_poses[s_tag]["crank"]
                mp_rc = parts_mp[f"return_crank_{s_tag}"]
                rc_off = np.array([cx_m, sgn * y_vg_abs, cz_m])
                w_comps.append((mp_rc["mass_kg"], np.array(mp_rc["com_m"]) + rc_off, full_to_mat3(mp_rc["fullinertia"])))
        attach_composite_inertial(wbody, w_comps)

        ET.SubElement(
            wbody, "geom", type="cylinder", fromto=f"0 {-y_gauge_half:.6f} 0  0 {y_gauge_half:.6f} 0",
            size="0.0028", rgba="0.3 0.3 0.3 1", mass="0", contype="0", conaffinity="0"
        )
        add_rail_wheel_colliders(wbody, f"drv_{idx}", r_drv, y_gauge_half, use_explicit_inertial=True)
        for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            m_w = f"m_main_driving_wheel_{s_tag}" if is_main else f"m_driving_wheel_{s_tag}"
            ET.SubElement(
                wbody, "geom", name=f"wheel_vis_{idx}_{s_tag}", type="mesh",
                mesh=m_w, material="mat_black_iron",
                pos=f"0 {sgn * y_w_off:.6f} 0", contype="0", conaffinity="0", mass="0"
            )
            if is_main:
                cx_m, cz_m = side_poses[s_tag]["crank"]
                ET.SubElement(
                    wbody, "geom", name=f"return_crank_vis_{s_tag}", type="mesh",
                    mesh=f"m_return_crank_{s_tag}", material="mat_machined_steel",
                    pos=f"{cx_m:.6f} {sgn * y_vg_abs:.6f} {cz_m:.6f}", contype="0", conaffinity="0", mass="0"
                )
        wheel_bodies[idx] = wbody

    # O3-3: イラスト 3 前后两组动轮跷跷板均衡梁 (Equalizer <tendon>)
    eq_springlen = "-0.0017" if on_rails else "0"
    for eq_name, j_a, j_b in [("tendon_eq_front", "j_susp_z1", "j_susp_z2"), ("tendon_eq_rear", "j_susp_z3", "j_susp_z4")]:
        ft = ET.SubElement(
            tendon, "fixed", name=eq_name, stiffness="2100", damping="28", springlength=eq_springlen
        )
        ET.SubElement(ft, "joint", joint=j_a, coef="0.5")
        ET.SubElement(ft, "joint", joint=j_b, coef="0.5")

    # 2. 19-2 铸铁先导径向台车架 (含 20-6 R2m/R3m 双孔支座) 与 19-11 Delta 式从轮铸铁台车架
    tz_truck = -(r_drv - r_trk)
    mp_tL, mp_tR = parts_mp["truck_wheel_L"], parts_mp["truck_wheel_R"]
    y_tw_off = y_gauge_half + 0.0006

    lead_axle_x = w_cfg["axle_x"]["leading"] * s
    lead_pivot_x = lead_axle_x - (0.052 if mode == "curve_r2m" else 0.062)
    lead_arm_len = lead_axle_x - lead_pivot_x
    if mode == "curve_r2m":
        lead_init_yaw = -0.080
        lead_arm = ET.SubElement(
            frame, "body", name="leading_truck_arm",
            pos=f"{lead_pivot_x:.6f} 0 {tz_truck:.6f}", euler=f"0 0 {lead_init_yaw:.6f}"
        )
        ET.SubElement(
            lead_arm, "joint", name="j_leading_yaw", type="hinge", axis="0 0 1",
            ref=f"{lead_init_yaw:.6f}", springref="0",
            limited="true", range="-0.28 0.28", stiffness="0.35", damping="0.22"
        )
    else:
        lead_arm = ET.SubElement(frame, "body", name="leading_truck_arm", pos=f"{lead_pivot_x:.6f} 0 {tz_truck:.6f}")
        ET.SubElement(
            lead_arm, "joint", name="j_leading_yaw", type="hinge", axis="0 0 1",
            limited="true", range="-0.28 0.28", stiffness="0.35", damping="0.22"
        )
    attach_part_inertial(lead_arm, parts_mp["leading_truck_frame"], offset_xyz=(lead_arm_len, 0.0, 0.0))
    ET.SubElement(
        lead_arm, "joint", name="j_leading_susp", type="slide", axis="0 0 1",
        springref="-0.0069" if on_rails else "0",
        limited="true", range="-0.004 0.004", stiffness="1050", damping="16"
    )
    ET.SubElement(
        lead_arm, "geom", name="lead_frame_vis", type="mesh",
        mesh="m_leading_truck_frame", material="mat_black_iron",
        pos=f"{lead_arm_len:.6f} 0 0", contype="0", conaffinity="0", mass="0"
    )
    lead_ws = ET.SubElement(lead_arm, "body", name="leading_wheelset", pos=f"{lead_arm_len:.6f} 0 0")
    ET.SubElement(lead_ws, "joint", name="j_leading_spin", type="hinge", axis="0 1 0", damping="0.0004")
    attach_composite_inertial(
        lead_ws,
        [
            (mp_tL["mass_kg"], np.array(mp_tL["com_m"]) + np.array([0.0, +y_tw_off, 0.0]), full_to_mat3(mp_tL["fullinertia"])),
            (mp_tR["mass_kg"], np.array(mp_tR["com_m"]) + np.array([0.0, -y_tw_off, 0.0]), full_to_mat3(mp_tR["fullinertia"])),
        ],
    )
    add_rail_wheel_colliders(lead_ws, "lead", r_trk, y_gauge_half, use_explicit_inertial=True, is_driving=False)

    trail_axle_x = w_cfg["axle_x"]["trailing"] * s
    trail_pivot_x = trail_axle_x + 0.054
    trail_arm_dx = trail_axle_x - trail_pivot_x
    if mode == "curve_r2m":
        wx_ta = loco_wx + trail_axle_x
        wy_ta_curve = -R_curve + math.sqrt(R_curve**2 - wx_ta**2)
        trail_init_yaw = math.asin(max(-0.24, min(0.24, (wy_ta_curve - loco_wy) / trail_arm_dx)))
        trail_arm = ET.SubElement(
            frame, "body", name="trailing_truck_arm",
            pos=f"{trail_pivot_x:.6f} 0 {tz_truck:.6f}", euler=f"0 0 {trail_init_yaw:.6f}"
        )
        ET.SubElement(
            trail_arm, "joint", name="j_trailing_yaw", type="hinge", axis="0 0 1",
            ref=f"{trail_init_yaw:.6f}", springref="0",
            limited="true", range="-0.26 0.26", stiffness="0.40", damping="0.05"
        )
    else:
        trail_arm = ET.SubElement(frame, "body", name="trailing_truck_arm", pos=f"{trail_pivot_x:.6f} 0 {tz_truck:.6f}")
        ET.SubElement(
            trail_arm, "joint", name="j_trailing_yaw", type="hinge", axis="0 0 1",
            limited="true", range="-0.26 0.26", stiffness="0.40", damping="0.05"
        )
    attach_part_inertial(trail_arm, parts_mp["trailing_truck_frame"])
    ET.SubElement(
        trail_arm, "joint", name="j_trailing_susp", type="slide", axis="0 0 1",
        springref="-0.0055" if on_rails else "0",
        limited="true", range="-0.004 0.004", stiffness="1050", damping="16"
    )
    ET.SubElement(
        trail_arm, "geom", name="trail_frame_vis", type="mesh",
        mesh="m_trailing_truck_frame", material="mat_black_iron",
        pos="0 0 0", contype="0", conaffinity="0", mass="0"
    )
    trail_ws = ET.SubElement(trail_arm, "body", name="trailing_wheelset", pos=f"{trail_arm_dx:.6f} 0 0")
    ET.SubElement(trail_ws, "joint", name="j_trailing_spin", type="hinge", axis="0 1 0", damping="0.0004")
    attach_composite_inertial(
        trail_ws,
        [
            (mp_tL["mass_kg"], np.array(mp_tL["com_m"]) + np.array([0.0, +y_tw_off, 0.0]), full_to_mat3(mp_tL["fullinertia"])),
            (mp_tR["mass_kg"], np.array(mp_tR["com_m"]) + np.array([0.0, -y_tw_off, 0.0]), full_to_mat3(mp_tR["fullinertia"])),
        ],
    )
    add_rail_wheel_colliders(trail_ws, "trail", r_trk, y_gauge_half, use_explicit_inertial=True, is_driving=False)

    for ws_elem, pfx in [(lead_ws, "lead"), (trail_ws, "trail")]:
        for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            ET.SubElement(
                ws_elem, "geom", name=f"{pfx}_vis_{s_tag}", type="mesh",
                mesh=f"m_truck_wheel_{s_tag}", material="mat_black_iron",
                pos=f"0 {sgn * y_tw_off:.6f} 0", contype="0", conaffinity="0", mass="0"
            )

    # 3. 中央轴动给水泵 (6-3 黄铜偏心轮箍 + 6-5 黄铜柱塞)、4-12 逆转器横轴、8-18 逆转长拉杆与 8-5 齿轮排液阀联动
    play_m = float(rc_cfg.get("crank_pin_axial_play_mm", 0.35)) * s
    r_pump, x_pump = b_cfg["axle_pump_eccentric"] * s, b_cfg["axle_pump_x"] * s
    pump_ram = ET.SubElement(frame, "body", name="axle_pump_ram", pos=f"{x_pump:.6f} 0 0")
    ET.SubElement(
        pump_ram, "joint", name="j_pump_ram", type="slide", axis="1 0 0",
        limited="true", range="-0.012 0.012", damping="0.4"
    )
    ET.SubElement(
        pump_ram, "geom", type="cylinder", fromto="-0.008 0 0  0.008 0 0",
        size="0.0016", material="mat_brass", rgba="0.82 0.62 0.24 1", density="8500", contype="0", conaffinity="0"
    )
    ET.SubElement(pump_ram, "site", name="s_pump_pin", pos="0 0 0")

    pump_strap = ET.SubElement(wheel_bodies[3], "body", name="pump_eccentric_rod", pos=f"{r_pump:.6f} 0 0")
    attach_part_inertial(pump_strap, parts_mp["axle_pump_eccentric"])
    ET.SubElement(
        pump_strap, "joint", name="j_pump_pin_play", type="slide", axis="0 1 0",
        limited="true", range=f"{-play_m:.6f} {play_m:.6f}", damping="0.04"
    )
    ET.SubElement(pump_strap, "joint", name="j_pump_ecc", type="hinge", axis="0 1 0")
    ET.SubElement(
        pump_strap, "geom", name="pump_ecc_vis", type="mesh",
        mesh="m_axle_pump_eccentric", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )
    ET.SubElement(pump_strap, "site", name="s_pump_rod_end", pos=f"{x_pump - r_pump:.6f} 0 0")
    ET.SubElement(equality, "connect", name="loop_axle_pump", site1="s_pump_rod_end", site2="s_pump_pin")

    rev_x, rev_z = vg_cfg["reverser_shaft_x"] * s, vg_cfg["reverser_shaft_z"] * s
    L_rev_arm = vg_cfg["reverser_arm_length"] * s
    L_lift_m = vg_cfg["lift_link_length"] * s
    rev_shaft = ET.SubElement(frame, "body", name="reverser_shaft", pos=f"{rev_x:.6f} 0 {rev_z:.6f}")
    attach_part_inertial(rev_shaft, parts_mp["reverser_shaft"])
    ET.SubElement(
        rev_shaft, "joint", name="j_reverser", type="hinge", axis="0 1 0",
        limited="true", range="-0.48 0.48", armature="0.0015", damping="0.6"
    )
    ET.SubElement(
        rev_shaft, "geom", name="reverser_shaft_vis", type="mesh",
        mesh="m_reverser_shaft", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
    )

    # Tier 3: 8-18 左侧 (+Y 左驾司机位) 逆转长拉杆 (通过零齿隙 <equality><joint> 定比联动，零新增执行器)
    reach_body = ET.SubElement(frame, "body", name="cab_reach_rod", pos=f"{rev_x:.6f} 0.033500 {rev_z + 0.007:.6f}")
    attach_part_inertial(reach_body, parts_mp["cab_reach_rod"])
    ET.SubElement(
        reach_body, "joint", name="j_reach_rod", type="slide", axis="1 0 0",
        limited="true", range="-0.005 0.005", damping="0.02"
    )
    ET.SubElement(
        reach_body, "geom", name="cab_reach_rod_vis", type="mesh",
        mesh="m_cab_reach_rod", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
    )
    ET.SubElement(
        equality, "joint", name="couple_reach_rod",
        joint1="j_reach_rod", joint2="j_reverser", polycoef="0 0.007 0 -0.001167 0"
    )

    # Tier 3: 8-6 -> 8-5 -> 7-1~3 -> 7-5 -> 1-19 -> 1-18 -> 1-16 八级气缸排液阀完整联动机构 (イラスト 1, 7, 8 & 図 1.3/8.1: 100% 全直拉杆 8-5 + 向上摇臂 7-1)
    dc_body = ET.SubElement(frame, "body", name="drain_cock_linkage", pos="0.124200 0.000000 -0.003800")
    attach_part_inertial(dc_body, parts_mp["drain_cock_linkage"])
    ET.SubElement(
        dc_body, "joint", name="j_drain_cock", type="hinge", axis="0 -1 0",
        limited="true", range="-0.32 0.32", stiffness="0.08", damping="0.008"
    )
    ET.SubElement(
        dc_body, "geom", name="drain_cock_vis", type="mesh",
        mesh="m_drain_cock_linkage", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    dp_body = ET.SubElement(frame, "body", name="drain_pull_rod", pos="0 0 0")
    attach_part_inertial(dp_body, parts_mp["drain_pull_rod"])
    ET.SubElement(
        dp_body, "joint", name="j_drain_pull", type="slide", axis="1 0 0",
        limited="true", range="-0.005 0.005", damping="0.02"
    )
    ET.SubElement(
        dp_body, "geom", name="drain_pull_rod_vis", type="mesh",
        mesh="m_drain_pull_rod", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
    )
    ET.SubElement(
        equality, "joint", name="couple_drain_rod",
        joint1="j_drain_pull", joint2="j_drain_cock", polycoef="0 -0.0114 0 0.001900 0"
    )

    # P3-CAB: 驾驶室原厂全真值可动操纵阀件、螺旋逆转机滑块与天窗刚体链 (イラスト 8, 11, 12, 13, 17, 21 & p45)
    rev_wheel = ET.SubElement(frame, "body", name="cab_reverser_wheel", pos="-0.127500 0.032500 0.034500")
    attach_part_inertial(rev_wheel, parts_mp["cab_reverser_wheel"])
    ET.SubElement(rev_wheel, "joint", name="j_reverser_wheel", type="hinge", axis="1 0 0", damping="0.002")
    ET.SubElement(
        rev_wheel, "geom", name="cab_reverser_wheel_vis", type="mesh",
        mesh="m_cab_reverser_wheel", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
    )
    ET.SubElement(
        equality, "joint", name="couple_reverser_wheel",
        joint1="j_reverser_wheel", joint2="j_reverser", polycoef="0 10.0 0 0 0"
    )

    rev_nut = ET.SubElement(frame, "body", name="cab_reverser_nut", pos="-0.127500 0.032500 0.034500")
    attach_part_inertial(rev_nut, parts_mp["cab_reverser_nut"])
    ET.SubElement(
        rev_nut, "joint", name="j_reverser_nut", type="slide", axis="1 0 0",
        limited="true", range="-0.005 0.005", damping="0.02"
    )
    ET.SubElement(
        rev_nut, "geom", name="cab_reverser_nut_vis", type="mesh",
        mesh="m_cab_reverser_nut", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )
    ET.SubElement(
        equality, "joint", name="couple_reverser_nut",
        joint1="j_reverser_nut", joint2="j_reverser", polycoef="0 0.007 0 -0.001167 0"
    )

    # 11-11 顶中 (-119.5, 0.0, +68.2mm) 加减阀长柄: q=0 水平指向左驾 (+Y) 为全闭，向上扫掠 (0 -> +1.15 rad) 开启主蒸汽
    reg_lev = ET.SubElement(frame, "body", name="cab_regulator_lever", pos="-0.119500 0.000000 0.068200")
    attach_part_inertial(reg_lev, parts_mp["cab_regulator_lever"])
    ET.SubElement(
        reg_lev, "joint", name="j_cab_regulator", type="hinge", axis="1 0 0",
        limited="true", range="0.0 1.15", damping="0.003"
    )
    ET.SubElement(
        reg_lev, "geom", name="cab_regulator_vis", type="mesh",
        mesh="m_cab_regulator_lever", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # 11-13 左上内 (+Y=+5.6, Z=+61.4mm, X=-120.2mm) 11-14 通风阀直向后三辐圆形黄铜手轮
    blw_whl = ET.SubElement(
        frame, "body", name="cab_blower_wheel",
        pos="-0.120200 0.005600 0.061400"
    )
    attach_part_inertial(blw_whl, parts_mp["cab_blower_wheel"])
    ET.SubElement(
        blw_whl, "joint", name="j_cab_blower", type="hinge", axis="1 0 0",
        limited="true", range="-6.28 6.28", damping="0.001"
    )
    ET.SubElement(
        blw_whl, "geom", name="cab_blower_vis", type="mesh",
        mesh="m_cab_blower_wheel", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # 11-3 左上外 (+Y=+14.5, Z=+62.0mm) 11-1 汽笛阀底部上翘触发拨柄
    whi_lev = ET.SubElement(frame, "body", name="cab_whistle_lever", pos="-0.116200 0.014500 0.062000")
    attach_part_inertial(whi_lev, parts_mp["cab_whistle_lever"])
    ET.SubElement(
        whi_lev, "joint", name="j_cab_whistle", type="hinge", axis="1 0 0",
        limited="true", range="-0.45 0.45", stiffness="0.04", damping="0.002"
    )
    ET.SubElement(
        whi_lev, "geom", name="cab_whistle_vis", type="mesh",
        mesh="m_cab_whistle_lever", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # 1-1/1-2/1-3 右侧外缘 (-Y=-31.3mm) 水面计下座同轴一字槽排污螺塞
    sg_drn = ET.SubElement(frame, "body", name="cab_sight_drain", pos="-0.118500 -0.031300 0.027800")
    attach_part_inertial(sg_drn, parts_mp["cab_sight_drain"])
    ET.SubElement(
        sg_drn, "joint", name="j_cab_sight_drain", type="slide", axis="0 0 -1",
        limited="true", range="0.0 0.0018", stiffness="0.25", damping="0.008"
    )
    ET.SubElement(
        sg_drn, "geom", name="cab_sight_drain_vis", type="mesh",
        mesh="m_cab_sight_drain", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # PG-ø20 右上内 (-Y=-15.4mm, Z=+67.15mm) 上仰 24° 压力表指针 (顺时针随表压增大正偏转)
    pg_ndl = ET.SubElement(frame, "body", name="cab_pressure_needle", pos="-0.122580 -0.015400 0.067150")
    attach_part_inertial(pg_ndl, parts_mp["cab_pressure_needle"])
    ET.SubElement(
        pg_ndl, "joint", name="j_cab_pg_needle", type="hinge", axis="0.913545 0.000000 -0.406737",
        limited="true", range="-2.5 2.5", damping="0.0005"
    )
    ET.SubElement(
        pg_ndl, "geom", name="cab_pg_needle_vis", type="mesh",
        mesh="m_cab_pressure_needle", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
    )

    # 11-17 左床下 (+Y=+18.5mm, Z=+14.5mm) 11-16 锅炉排水阀一字槽针阀螺杆 (イラスト 11, 13)
    bd_scr = ET.SubElement(frame, "body", name="cab_blowdown_screw", pos="-0.118000 0.018500 0.014500")
    attach_part_inertial(bd_scr, parts_mp["cab_blowdown_screw"])
    ET.SubElement(
        bd_scr, "joint", name="j_cab_blowdown", type="hinge", axis="1 0 0",
        limited="true", range="-6.28 6.28", damping="0.001"
    )
    ET.SubElement(
        bd_scr, "geom", name="cab_blowdown_vis", type="mesh",
        mesh="m_cab_blowdown_screw", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # 12-1 低位中下底铰蝴蝶式铸造火室门: 绕底部水平 Y 轴销 12-3 (-115.4, 0.0, +22.0mm) 向后下方翻开 (q ∈ [0, +1.45])
    fb_door = ET.SubElement(frame, "body", name="cab_firebox_door", pos="-0.115400 0.000000 0.022000")
    attach_part_inertial(fb_door, parts_mp["cab_firebox_door"])
    ET.SubElement(
        fb_door, "joint", name="j_cab_fire_door", type="hinge", axis="0 -1 0",
        limited="true", range="0.0 1.45", damping="0.003"
    )
    ET.SubElement(
        fb_door, "geom", name="cab_fire_door_vis", type="mesh",
        mesh="m_cab_firebox_door", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
    )

    # 6-9 / 8-7 驾驶室右后地板 (-134.5, -8.8, Z=+23.8mm) 同轴旁通回水阀立杆手柄 (向下直插床下 4-17)
    byp_lev = ET.SubElement(frame, "body", name="cab_bypass_lever", pos="-0.134500 -0.008800 0.023800")
    attach_part_inertial(byp_lev, parts_mp["cab_bypass_lever"])
    ET.SubElement(
        byp_lev, "joint", name="j_cab_bypass", type="hinge", axis="0 0 1",
        limited="true", range="-1.57 1.57", damping="0.002"
    )
    ET.SubElement(
        byp_lev, "geom", name="cab_bypass_vis", type="mesh",
        mesh="m_cab_bypass_lever", material="mat_brass", contype="0", conaffinity="0", mass="0"
    )

    # 21-15 驾驶室顶棚中央天窗通风盖 (前铰于 (-119.5, 0.0, +82.2mm))
    rf_htc = ET.SubElement(frame, "body", name="cab_roof_hatch", pos="-0.119500 0.000000 0.082200")
    attach_part_inertial(rf_htc, parts_mp["cab_roof_hatch"])
    ET.SubElement(
        rf_htc, "joint", name="j_cab_roof_hatch", type="hinge", axis="0 1 0",
        limited="true", range="0.0 2.36", damping="0.008"
    )
    ET.SubElement(
        rf_htc, "geom", name="cab_roof_hatch_vis", type="mesh",
        mesh="m_cab_roof_hatch", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
    )

    # 4. 左右双侧气缸、上置亮面滑杆、三段铰接耦合连杆、主连杆、十字头、平面滑阀与华氏五杆全金属 CAD 闭环
    L_union_m = vg_cfg["union_link_length"] * s
    L_comb_m = vg_cfg["comb_lever_total_length"] * s
    L_mv_m = vg_cfg["comb_lever_lap_lead_offset"] * s
    L_ecc_m = vg_cfg["eccentric_rod_length"] * s
    L_rad_m = vg_cfg["radius_rod_length"] * s
    L_back_m = vg_cfg["radius_rod_back_ext"] * s
    k_off_m = w_cfg.get("coupling_knuckle_offset", 8.5) * s

    sides = [("L", +1.0, 0.0), ("R", -1.0, q_rad_R)]
    for tag, sgn, theta in sides:
        pose = side_poses[tag]
        y_cyl = sgn * (rc_cfg["cylinder_center_y"] * s)
        y_coup = sgn * (y_gauge_half + 0.0055)
        y_vg = sgn * y_vg_abs

        ET.SubElement(
            frame, "geom", name=f"cyl_housing_{tag}", type="mesh",
            mesh=f"m_cylinder_block_{tag}", material="mat_black_iron",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} {rc_cfg['cylinder_center_z']*s:.6f}",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            frame, "geom", name=f"slide_bar_vis_{tag}", type="mesh",
            mesh=f"m_slide_bar_{tag}", material="mat_machined_steel",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} {rc_cfg['cylinder_center_z']*s:.6f}",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            frame, "geom", name=f"piston_tail_sheath_vis_{tag}", type="mesh",
            mesh=f"m_piston_tail_sheath_{tag}", material="mat_machined_steel",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} {rc_cfg['cylinder_center_z']*s:.6f}",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            frame, "geom", name=f"valve_spindle_sheath_vis_{tag}", type="mesh",
            mesh=f"m_valve_spindle_sheath_{tag}", material="mat_machined_steel",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} {rc_cfg['cylinder_center_z']*s:.6f}",
            mass="0", contype="0", conaffinity="0"
        )

        cx, cz = pose["crank"]
        for idx, _, _ in axle_xs:
            ET.SubElement(
                wheel_bodies[idx], "site", name=f"s_crank_pin_{idx}_{tag}",
                pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}", size="0.002"
            )

        # P0-A: 7-12 / 7-15 三段铰接式四轴耦合连杆 (主中段 Axle3-Axle2 + 前铰接节 Axle1 + 后铰接节 Axle4)
        dx_1 = w_cfg["axle_x"]["drive_1"] * s
        dx_2 = w_cfg["axle_x"]["drive_2"] * s
        dx_4 = w_cfg["axle_x"]["drive_4"] * s

        coup_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"coupling_rod_{tag}", pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}"
        )
        attach_part_inertial(coup_rod, parts_mp["coupling_rod"])
        ET.SubElement(coup_rod, "joint", name=f"j_coup3_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            coup_rod, "geom", name=f"coupling_rod_vis_{tag}", type="mesh",
            mesh="m_coupling_rod", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )

        def add_coup_lat_bushing(parent_el: ET.Element, target_idx: int, local_dx: float) -> None:
            bush = ET.SubElement(parent_el, "body", name=f"coup_bush_{target_idx}_{tag}", pos=f"{local_dx:.6f} 0 0")
            ET.SubElement(
                bush, "joint", name=f"j_coup_lat_{target_idx}_{tag}", type="slide", axis="0 1 0",
                limited="true", range="-0.0016 0.0016", damping="0.25"
            )
            ET.SubElement(
                bush, "geom", type="cylinder", fromto="0 -0.001 0  0 0.001 0", size="0.002",
                density="8500", group="3", rgba="0 0 0 0", contype="0", conaffinity="0"
            )
            ET.SubElement(bush, "site", name=f"s_coup_hole_{target_idx}_{tag}", pos="0 0 0")
            ET.SubElement(
                equality, "connect", name=f"loop_coup_{target_idx}_{tag}",
                site1=f"s_coup_hole_{target_idx}_{tag}", site2=f"s_crank_pin_{target_idx}_{tag}"
            )

        add_coup_lat_bushing(coup_rod, 2, dx_2)

        x_kf = dx_2 + k_off_m
        coup_front = ET.SubElement(coup_rod, "body", name=f"coupling_rod_front_{tag}", pos=f"{x_kf:.6f} 0 0")
        attach_part_inertial(coup_front, parts_mp["coupling_rod_front"])
        ET.SubElement(
            coup_front, "joint", name=f"j_coup_knuckle_f_{tag}", type="hinge", axis="0 1 0",
            limited="true", range="-0.15 0.15", damping="0.0005"
        )
        ET.SubElement(
            coup_front, "geom", name=f"coupling_rod_f_vis_{tag}", type="mesh",
            mesh="m_coupling_rod_front", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        add_coup_lat_bushing(coup_front, 1, dx_1 - x_kf)

        x_kr = -k_off_m
        coup_rear = ET.SubElement(coup_rod, "body", name=f"coupling_rod_rear_{tag}", pos=f"{x_kr:.6f} 0 0")
        attach_part_inertial(coup_rear, parts_mp["coupling_rod_rear"])
        ET.SubElement(
            coup_rear, "joint", name=f"j_coup_knuckle_r_{tag}", type="hinge", axis="0 1 0",
            limited="true", range="-0.15 0.15", damping="0.0005"
        )
        ET.SubElement(
            coup_rear, "geom", name=f"coupling_rod_r_vis_{tag}", type="mesh",
            mesh="m_coupling_rod_rear", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        add_coup_lat_bushing(coup_rear, 4, dx_4 - x_kr)

        # 主连杆 (7-11 / 7-16, 含 Tier 2 曲柄销 ±0.35mm 轴向微游隙释放 j_lat_y3)
        px, pz = pose["crosshead"]
        pitch_y = -math.atan2(pz - cz, px - cx)
        main_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"main_rod_{tag}", pos=f"{cx:.6f} {y_cyl:.6f} {cz:.6f}"
        )
        attach_part_inertial(main_rod, parts_mp["main_rod"], pitch_y_rad=pitch_y)
        ET.SubElement(
            main_rod, "joint", name=f"j_main_pin_play_{tag}", type="slide", axis="0 1 0",
            limited="true", range=f"{-play_m:.6f} {play_m:.6f}", damping="0.04"
        )
        ET.SubElement(main_rod, "joint", name=f"j_main_crank_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            main_rod, "geom", name=f"main_rod_vis_{tag}", type="mesh",
            mesh="m_main_rod", material="mat_machined_steel",
            euler=f"0 {pitch_y:.6f} 0", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(main_rod, "site", name=f"s_mainrod_small_{tag}", pos=f"{px - cx:.6f} 0 {pz - cz:.6f}")

        # 1-28 黄铜活塞 + 1-31 十字头总成
        p_mid_x = rc_cfg["main_rod_length"] * s
        slide_min, slide_max = (p_mid_x - r_crank - 0.002) - px, (p_mid_x + r_crank + 0.002) - px
        piston_body = ET.SubElement(
            frame, "body", name=f"piston_crosshead_{tag}", pos=f"{px:.6f} {y_cyl:.6f} {pz:.6f}"
        )
        attach_part_inertial(piston_body, parts_mp[f"piston_crosshead_{tag}"])
        ET.SubElement(
            piston_body, "joint", name=f"j_piston_{tag}", type="slide", axis="1 0 0",
            limited="true", range=f"{slide_min:.6f} {slide_max:.6f}", damping="0.35", frictionloss="0.12"
        )
        ET.SubElement(
            piston_body, "geom", name=f"piston_ch_vis_{tag}", type="mesh",
            mesh=f"m_piston_crosshead_{tag}", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(piston_body, "site", name=f"s_crosshead_pin_{tag}", pos="0 0 0")
        ET.SubElement(
            equality, "connect", name=f"loop_main_drive_{tag}",
            site1=f"s_mainrod_small_{tag}", site2=f"s_crosshead_pin_{tag}"
        )

        # 华氏五杆精密不锈钢 CAD 刚体链 (1-27 先导短连杆 + 1-23 合并杆 + 2-12/19 偏心棒 + 2-1 心向棒 + 2-10 吊杆)
        u0x, u0z = pose["drop_arm"]
        u1x, u1z = pose["comb_bot"]
        mx, mz = pose["comb_mid"]
        vx, vz = pose["valve_pin"]

        pitch_union = -math.atan2(u1z - u0z, u1x - u0x)
        union_body = ET.SubElement(
            piston_body, "body", name=f"union_link_{tag}",
            pos=f"{u0x - px:.6f} {y_vg - y_cyl:.6f} {u0z - pz:.6f}",
            euler=f"0 {pitch_union:.6f} 0"
        )
        attach_part_inertial(union_body, parts_mp[f"union_link_{tag}"])
        ET.SubElement(union_body, "joint", name=f"j_union_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            union_body, "geom", name=f"union_link_vis_{tag}", type="mesh",
            mesh=f"m_union_link_{tag}", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )

        pitch_comb = math.atan2(vx - u1x, vz - u1z)
        comb_body = ET.SubElement(
            union_body, "body", name=f"comb_lever_{tag}",
            pos=f"{L_union_m:.6f} 0 0", euler=f"0 {pitch_comb - pitch_union:.6f} 0"
        )
        attach_part_inertial(comb_body, parts_mp["combination_lever"])
        ET.SubElement(comb_body, "joint", name=f"j_comb_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            comb_body, "geom", name=f"comb_lever_vis_{tag}", type="mesh",
            mesh="m_combination_lever", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(comb_body, "site", name=f"s_comb_radius_{tag}", pos=f"0 0 {L_comb_m - L_mv_m:.6f}")
        ET.SubElement(comb_body, "site", name=f"s_comb_valve_{tag}", pos=f"0 0 {L_comb_m:.6f}")

        # 1-6 平面滑阀 + 1-22 阀芯棒总成
        valve_body = ET.SubElement(frame, "body", name=f"slide_valve_{tag}", pos=f"{vx:.6f} {y_vg:.6f} {vz:.6f}")
        attach_part_inertial(valve_body, parts_mp[f"slide_valve_{tag}"])
        ET.SubElement(
            valve_body, "joint", name=f"j_valve_{tag}", type="slide", axis="1 0 0",
            limited="true", range="-0.008 0.008", damping="0.12", frictionloss="0.04"
        )
        ET.SubElement(
            valve_body, "geom", name=f"valve_vis_{tag}", type="mesh",
            mesh=f"m_slide_valve_{tag}", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(valve_body, "site", name=f"s_valve_clevis_{tag}", pos="0 0 0")
        ET.SubElement(
            equality, "connect", name=f"loop_valve_{tag}", site1=f"s_comb_valve_{tag}", site2=f"s_valve_clevis_{tag}"
        )

        rx, rz = pose["ret_crank"]
        etx, etz = pose["exp_tail"]
        tx, tz = pose["exp_trunnion"]
        sdx, sdz = pose["die_pin"]
        q_die_0 = float(pose["q_die"])
        cx_arc, cz_arc = pose["arc_center_local"]
        slot_tilt = math.atan2(-cz_arc, cx_arc)
        rlx, rlz = pose["rad_lift"]
        atx, atz = pose["arm_tip"]

        pitch_ecc = -math.atan2(etz - rz, etx - rx)
        ecc_body = ET.SubElement(
            wheel_bodies[3], "body", name=f"eccentric_rod_{tag}",
            pos=f"{rx:.6f} {y_vg:.6f} {rz:.6f}", euler=f"0 {pitch_ecc:.6f} 0"
        )
        attach_part_inertial(ecc_body, parts_mp[f"eccentric_rod_{tag}"])
        ET.SubElement(
            ecc_body, "joint", name=f"j_ecc_pin_play_{tag}", type="slide", axis="0 1 0",
            limited="true", range=f"{-play_m:.6f} {play_m:.6f}", damping="0.03"
        )
        ET.SubElement(ecc_body, "joint", name=f"j_retcrank_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            ecc_body, "geom", name=f"ecc_rod_vis_{tag}", type="mesh",
            mesh=f"m_eccentric_rod_{tag}", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(ecc_body, "site", name=f"s_ecc_front_{tag}", pos=f"{L_ecc_m:.6f} 0 0")

        # Gate 3: 半月板 CAD 已按精确 exp_link_tail_cant_deg 构造下尾耳，按 slot_tilt 旋转后滑槽与尾孔同时实现 0.0000° 对齐！
        exp_body = ET.SubElement(frame, "body", name=f"expansion_link_{tag}", pos=f"{tx:.6f} {y_vg:.6f} {tz:.6f}")
        attach_part_inertial(exp_body, parts_mp[f"expansion_link_{tag}"], pitch_y_rad=slot_tilt)
        ET.SubElement(exp_body, "joint", name=f"j_explink_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            exp_body, "geom", name=f"exp_link_vis_{tag}", type="mesh",
            mesh=f"m_expansion_link_{tag}", material="mat_machined_steel",
            euler=f"0 {slot_tilt:.6f} 0", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(exp_body, "site", name=f"s_exp_tail_{tag}", pos=f"{etx - tx:.6f} 0 {etz - tz:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_eccentric_{tag}", site1=f"s_ecc_front_{tag}", site2=f"s_exp_tail_{tag}"
        )

        # O3-1: 2-3 圆弧月牙槽等效铰接 (R = radius_rod_length)，滑块网格按 slot_tilt 与圆弧滑槽严格同心
        slot_lim = vg_cfg["die_slot_half_travel"] * s
        ang_lim = math.asin(min(0.95, slot_lim / L_rad_m))
        die_body = ET.SubElement(
            exp_body, "body", name=f"die_block_{tag}",
            pos=f"{sdx - tx:.6f} 0 {sdz - tz:.6f}", euler=f"0 {q_die_0:.6f} 0"
        )
        attach_part_inertial(die_body, parts_mp[f"die_block_{tag}"], pitch_y_rad=slot_tilt)
        ET.SubElement(
            die_body, "joint", name=f"j_die_shift_{tag}", type="hinge", axis="0 1 0",
            pos=f"{cx_arc:.6f} 0 {cz_arc:.6f}", ref=f"{q_die_0:.6f}",
            limited="true", range=f"{-ang_lim:.6f} {ang_lim:.6f}", damping="0.00035"
        )
        ET.SubElement(
            die_body, "geom", name=f"die_vis_{tag}", type="mesh",
            mesh=f"m_die_block_{tag}", material="mat_brass",
            euler=f"0 {slot_tilt:.6f} 0", contype="0", conaffinity="0", mass="0"
        )

        pitch_rad = -math.atan2(mz - sdz, mx - sdx)
        rad_body = ET.SubElement(
            die_body, "body", name=f"radius_rod_{tag}",
            pos="0 0 0", euler=f"0 {pitch_rad - q_die_0:.6f} 0"
        )
        attach_part_inertial(rad_body, parts_mp[f"radius_rod_{tag}"])
        ET.SubElement(rad_body, "joint", name=f"j_radius_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            rad_body, "geom", name=f"radius_rod_vis_{tag}", type="mesh",
            mesh=f"m_radius_rod_{tag}", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(rad_body, "site", name=f"s_rad_front_{tag}", pos=f"{L_rad_m:.6f} 0 0")
        ET.SubElement(rad_body, "site", name=f"s_rad_lift_{tag}", pos=f"{-L_back_m:.6f} 0 0")
        ET.SubElement(
            equality, "connect", name=f"loop_radius_{tag}", site1=f"s_rad_front_{tag}", site2=f"s_comb_radius_{tag}"
        )

        # P0-A: 左右严格等长 11.0mm 逆转摇臂 (L_rev_arm) 与 11.0mm 逆转吊杆 (L_lift_m)
        pitch_lift = math.atan2(-(rlx - atx), -(rlz - atz))
        lift_arm = ET.SubElement(
            rev_shaft, "body", name=f"lift_link_{tag}",
            pos=f"{L_rev_arm:.6f} {y_vg:.6f} 0", euler=f"0 {pitch_lift:.6f} 0"
        )
        attach_part_inertial(lift_arm, parts_mp[f"lift_link_{tag}"])
        ET.SubElement(lift_arm, "joint", name=f"j_lift_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            lift_arm, "geom", name=f"lift_link_vis_{tag}", type="mesh",
            mesh=f"m_lift_link_{tag}", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(lift_arm, "site", name=f"s_lift_bot_{tag}", pos=f"0 0 {-L_lift_m:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_reverser_{tag}", site1=f"s_lift_bot_{tag}", site2=f"s_rad_lift_{tag}"
        )

        ET.SubElement(actuator, "general", name=f"steam_force_{tag}", joint=f"j_piston_{tag}", gainprm="1")
        ET.SubElement(sensor, "jointpos", name=f"piston_pos_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointvel", name=f"piston_vel_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointpos", name=f"valve_pos_{tag}", joint=f"j_valve_{tag}")

    ET.SubElement(actuator, "position", name="reverser_ctrl", joint="j_reverser", kp="60", ctrlrange="-0.46 0.46")
    ET.SubElement(sensor, "jointvel", name="wheel3_omega", joint="j_wheel_3")
    ET.SubElement(sensor, "jointvel", name="pump_ram_vel", joint="j_pump_ram")

    # ==================== B. 四轴双转向架煤水车与牵引拉杆 ====================
    if include_tender:
        x_pin_loco = t_cfg["drawbar_loco_pin_x"] * s
        L_drawbar = t_cfg["drawbar_length"] * s
        x_pin_tdr_local = t_cfg["drawbar_tender_pin_local_x"] * s
        base_rail_z = rail_top_z if on_rails else (z_loco_axle - r_drv)
        z_tdr_deck = base_rail_z + (2.0 * r_tdr) - 0.009
        half_span = 0.5 * t_cfg["bogie_center_span"] * s
        half_wb = 0.5 * t_cfg["bogie_wheelbase"] * s

        pin_loco_wx = loco_wx + x_pin_loco
        pin_loco_wy = loco_wy

        if mode == "curve_r2m":
            R_tdr = math.sqrt(R_curve**2 - half_wb**2 - half_span**2)
            phi_loco_pin = math.atan2(pin_loco_wy + R_curve, pin_loco_wx)
            phi_tdr = phi_loco_pin + (L_drawbar + x_pin_tdr_local) / R_curve
            tdr_wx = R_tdr * math.cos(phi_tdr)
            tdr_wy = -R_curve + R_tdr * R_curve / R_curve * math.sin(phi_tdr)
            tdr_yaw = phi_tdr - 0.5 * math.pi
        else:
            tdr_wx, tdr_wy, tdr_yaw = pin_loco_wx - L_drawbar - x_pin_tdr_local, 0.0, 0.0

        pin_tdr_wx = tdr_wx + x_pin_tdr_local * math.cos(tdr_yaw)
        pin_tdr_wy = tdr_wy + x_pin_tdr_local * math.sin(tdr_yaw)
        db_dx, db_dy = pin_tdr_wx - pin_loco_wx, pin_tdr_wy - pin_loco_wy
        db_len = math.hypot(db_dx, db_dy)
        db_yaw = math.atan2(db_dy, db_dx)

        drawbar = ET.SubElement(frame, "body", name="drawbar_8_8", pos=f"{x_pin_loco:.6f} 0 -0.006")
        if on_rails:
            ET.SubElement(drawbar, "joint", name="j_drawbar_yaw", type="hinge", axis="0 0 1", armature="0.0005", damping="0.02")
            ET.SubElement(drawbar, "joint", name="j_drawbar_pitch", type="hinge", axis="0 1 0", armature="0.0005", damping="0.02")
        ET.SubElement(
            drawbar, "geom", name="drawbar_bar_vis", type="box",
            pos=f"{0.5 * db_dx:.6f} {0.5 * db_dy:.6f} 0", euler=f"0 0 {db_yaw:.6f}",
            size=f"{0.5 * db_len:.6f} 0.002800 0.001250",
            material="mat_machined_steel", rgba="0.72 0.74 0.77 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(
            drawbar, "geom", name="drawbar_eye_f_vis", type="cylinder",
            fromto="0 0 -0.001500  0 0 0.001500", size="0.003500",
            material="mat_machined_steel", rgba="0.72 0.74 0.77 1", mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            drawbar, "geom", name="drawbar_eye_r_vis", type="cylinder",
            fromto=f"{db_dx:.6f} {db_dy:.6f} -0.001500  {db_dx:.6f} {db_dy:.6f} 0.001500", size="0.003200",
            material="mat_machined_steel", rgba="0.72 0.74 0.77 1", mass="0", contype="0", conaffinity="0"
        )
        # イラスト 34: 机车↔煤水车 3 根真实跨车弹性软管 (上置白色硅胶供醇 ST-ø7 + 下置斜跨黑色氟胶给水/回水 BT-ø5 x2 + 34-8 活接)
        sfx_h = "_r2m" if mode == "curve_r2m" else ""
        ET.SubElement(
            drawbar, "geom", name="umb_hose_fuel", type="mesh",
            mesh=f"m_umb_hose_fuel{sfx_h}", material="mat_silicone_hose",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            drawbar, "geom", name="umb_hose_feed", type="mesh",
            mesh=f"m_umb_hose_feed{sfx_h}", material="mat_viton_hose",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            drawbar, "geom", name="umb_union_34_8_vis", type="mesh",
            mesh=f"m_umb_union_34_8{sfx_h}", material="mat_brass",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(
            drawbar, "geom", name="umb_hose_return", type="mesh",
            mesh=f"m_umb_hose_return{sfx_h}", material="mat_viton_hose",
            mass="0", contype="0", conaffinity="0"
        )
        ET.SubElement(drawbar, "site", name="s_drawbar_rear", pos=f"{db_dx:.6f} {db_dy:.6f} 0")

        tender = ET.SubElement(
            world, "body", name="tender_body",
            pos=f"{tdr_wx:.6f} {tdr_wy:.6f} {z_tdr_deck:.6f}", euler=f"0 0 {tdr_yaw:.6f}"
        )
        if on_rails:
            ET.SubElement(tender, "freejoint", name="tender_free")
        mp_tb = parts_mp["tender_body"]
        mp_ti = parts_mp["tender_internals"]
        mp_tr_lad = parts_mp["tender_rear_ladder"]
        m_tdr_fluid = 0.375  # 272 ml 标准 80% 补水 + 130 ml 燃料酒精
        I_tdr_fluid = (m_tdr_fluid / 12.0) * np.diag([0.070**2 + 0.036**2, 0.200**2 + 0.036**2, 0.200**2 + 0.070**2])
        attach_composite_inertial(
            tender,
            [
                (mp_tb["mass_kg"], np.array(mp_tb["com_m"]), full_to_mat3(mp_tb["fullinertia"])),
                (mp_ti["mass_kg"], np.array(mp_ti["com_m"]), full_to_mat3(mp_ti["fullinertia"])),
                (mp_tr_lad["mass_kg"], np.array(mp_tr_lad["com_m"]), full_to_mat3(mp_tr_lad["fullinertia"])),
                (m_tdr_fluid, np.array([-0.015, 0.0, 0.022]), I_tdr_fluid),
            ],
        )
        z_pin_local = (z_loco_axle - 0.006) - z_tdr_deck
        ET.SubElement(tender, "site", name="s_tender_front_pin", pos=f"{x_pin_tdr_local:.6f} 0 {z_pin_local:.6f}")
        ET.SubElement(
            tender, "geom", name="tender_internals_vis", type="mesh",
            mesh="m_tender_internals", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(
            tender, "geom", name="tender_body_vis", type="mesh",
            mesh="m_tender_body", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(
            tender, "geom", name="tender_rear_ladder_vis", type="mesh",
            mesh="m_tender_rear_ladder", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
        )

        # P3-TENDER: 煤水车原厂真 1-DOF 四连杆手摇给水泵 (31-5 + 31-7x2 + 31-3 + 31-2) 与可动功能件 (イラスト 31-37, Z_max < +62.0mm 零穿顶!)
        hp_lnk = ET.SubElement(tender, "body", name="tender_hp_links", pos="-0.047000 0.004000 0.031700")
        attach_part_inertial(hp_lnk, parts_mp["tender_hp_links"])
        ET.SubElement(
            hp_lnk, "joint", name="j_tender_hp_link", type="hinge", axis="0 1 0",
            limited="true", range="-0.35 0.35", damping="0.004"
        )
        ET.SubElement(
            hp_lnk, "geom", name="tender_hp_links_vis", type="mesh",
            mesh="m_tender_hp_links", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )

        hp_lev = ET.SubElement(hp_lnk, "body", name="tender_handpump_lever", pos="0.020000 0.000000 0.000000")
        attach_part_inertial(hp_lev, parts_mp["tender_handpump_lever"])
        ET.SubElement(
            hp_lev, "joint", name="j_tender_handpump", type="hinge", axis="0 1 0",
            limited="true", range="-0.45 0.45", damping="0.008"
        )
        ET.SubElement(
            hp_lev, "geom", name="tender_handpump_vis", type="mesh",
            mesh="m_tender_handpump_lever", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(hp_lev, "site", name="s_hp_lever_bot", pos="0.000000 0.000000 -0.010000")

        hp_ram = ET.SubElement(tender, "body", name="tender_handpump_ram", pos="-0.027000 0.004000 0.021700")
        attach_part_inertial(hp_ram, parts_mp["tender_handpump_ram"])
        ET.SubElement(
            hp_ram, "joint", name="j_tender_hp_ram", type="slide", axis="1 0 0",
            limited="true", range="-0.008 0.008", damping="0.04"
        )
        ET.SubElement(
            hp_ram, "geom", name="tender_hp_ram_vis", type="mesh",
            mesh="m_tender_handpump_ram", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(hp_ram, "site", name="s_hp_ram_pin", pos="0.000000 0.000000 0.000000")
        ET.SubElement(
            equality, "connect", name="loop_tender_handpump",
            site1="s_hp_lever_bot", site2="s_hp_ram_pin",
            solref="0.0015 1.0", solimp="0.995 0.9995 0.00005"
        )

        drn_plg = ET.SubElement(tender, "body", name="tender_drain_plug", pos="-0.041000 -0.010300 0.015200")
        attach_part_inertial(drn_plg, parts_mp["tender_drain_plug"])
        ET.SubElement(
            drn_plg, "joint", name="j_tender_drain_plug", type="slide", axis="0 0 1",
            limited="true", range="0.0 0.0055", damping="0.05"
        )
        ET.SubElement(
            drn_plg, "geom", name="tender_drain_plug_vis", type="mesh",
            mesh="m_tender_drain_plug", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )

        coal_cov = ET.SubElement(tender, "body", name="tender_coal_cover", pos="-0.045000 0.000000 0.062600")
        attach_part_inertial(coal_cov, parts_mp["tender_coal_cover"])
        ET.SubElement(
            coal_cov, "joint", name="j_tender_coal_cover", type="slide", axis="0 0 1",
            limited="true", range="0.0 0.032", damping="0.05"
        )
        ET.SubElement(
            coal_cov, "geom", name="tender_coal_cover_vis", type="mesh",
            mesh="m_tender_coal_cover", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
        )

        wt_lid = ET.SubElement(tender, "body", name="tender_water_lid", pos="-0.085000 0.000000 0.062000")
        attach_part_inertial(wt_lid, parts_mp["tender_water_lid"])
        ET.SubElement(
            wt_lid, "joint", name="j_tender_water_lid", type="hinge", axis="0 1 0",
            limited="true", range="0.0 1.65", damping="0.003"
        )
        ET.SubElement(
            wt_lid, "geom", name="tender_water_lid_vis", type="mesh",
            mesh="m_tender_water_lid", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
        )

        fv_tee = ET.SubElement(tender, "body", name="tender_fuel_valve", pos="0.023000 -0.006800 0.066600")
        attach_part_inertial(fv_tee, parts_mp["tender_fuel_valve"])
        ET.SubElement(
            fv_tee, "joint", name="j_tender_fuel_valve", type="hinge", axis="0 0 1",
            limited="true", range="-1.57 1.57", damping="0.002"
        )
        ET.SubElement(
            fv_tee, "geom", name="tender_fuel_valve_vis", type="mesh",
            mesh="m_tender_fuel_valve", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )

        fc_cap = ET.SubElement(tender, "body", name="tender_fuel_cap", pos="0.034800 0.005600 0.065800")
        attach_part_inertial(fc_cap, parts_mp["tender_fuel_cap"])
        ET.SubElement(
            fc_cap, "joint", name="j_tender_fuel_cap", type="slide", axis="0 0 1",
            limited="true", range="0.0 0.006", damping="0.05"
        )
        ET.SubElement(
            fc_cap, "geom", name="tender_fuel_cap_vis", type="mesh",
            mesh="m_tender_fuel_cap", material="mat_brass", contype="0", conaffinity="0", mass="0"
        )

        z_bogie_local = (base_rail_z + r_tdr + (0.0002 if on_rails else 0.0)) - z_tdr_deck

        for b_idx, bx in [(1, +half_span), (2, -half_span)]:
            if mode == "curve_r2m":
                bogie_yaw = -math.asin(bx / R_curve)
                bogie = ET.SubElement(
                    tender, "body", name=f"tender_bogie_{b_idx}",
                    pos=f"{bx:.6f} 0 {z_bogie_local:.6f}", euler=f"0 0 {bogie_yaw:.6f}"
                )
                attach_part_inertial(bogie, parts_mp["tender_bogie_frame"])
                ET.SubElement(
                    bogie, "joint", name=f"j_tbogie_{b_idx}_yaw", type="hinge", axis="0 0 1",
                    ref=f"{bogie_yaw:.6f}", damping="0.015"
                )
            else:
                bogie = ET.SubElement(tender, "body", name=f"tender_bogie_{b_idx}", pos=f"{bx:.6f} 0 {z_bogie_local:.6f}")
                attach_part_inertial(bogie, parts_mp["tender_bogie_frame"])
                if on_rails:
                    ET.SubElement(bogie, "joint", name=f"j_tbogie_{b_idx}_yaw", type="hinge", axis="0 0 1", damping="0.015")
            if on_rails:
                ET.SubElement(
                    bogie, "joint", name=f"j_tbogie_{b_idx}_pitch", type="hinge", axis="0 1 0",
                    limited="true", range="-0.08 0.08", stiffness="0.6", damping="0.05"
                )
                ET.SubElement(
                    bogie, "joint", name=f"j_tbogie_{b_idx}_roll", type="hinge", axis="1 0 0",
                    limited="true", range="-0.05 0.05", stiffness="1.2", damping="0.08"
                )
            ET.SubElement(
                bogie, "geom", name=f"tbogie_vis_{b_idx}", type="mesh",
                mesh="m_tender_bogie_frame", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
            )

            for a_sub, ax_dx in [(1, +half_wb), (2, -half_wb)]:
                w_num = (b_idx - 1) * 2 + a_sub
                t_abox = ET.SubElement(bogie, "body", name=f"tender_axlebox_{w_num}", pos=f"{ax_dx:.6f} 0 0")
                attach_part_inertial(t_abox, parts_mp["tender_axlebox"])
                if on_rails:
                    ET.SubElement(
                        t_abox, "joint", name=f"j_tsusp_z{w_num}", type="slide", axis="0 0 1",
                        limited="true", range="-0.002 0.002", stiffness="900", damping="12"
                    )
                ET.SubElement(
                    t_abox, "geom", name=f"tabox_vis_{w_num}", type="mesh",
                    mesh="m_tender_axlebox", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
                )

                t_ws = ET.SubElement(t_abox, "body", name=f"tender_wheelset_{w_num}", pos="0 0 0")
                if on_rails:
                    ET.SubElement(t_ws, "joint", name=f"j_twheel_{w_num}", type="hinge", axis="0 1 0", damping="0.0005")
                attach_composite_inertial(
                    t_ws,
                    [
                        (mp_tL["mass_kg"], np.array(mp_tL["com_m"]) + np.array([0.0, +y_tw_off, 0.0]), full_to_mat3(mp_tL["fullinertia"])),
                        (mp_tR["mass_kg"], np.array(mp_tR["com_m"]) + np.array([0.0, -y_tw_off, 0.0]), full_to_mat3(mp_tR["fullinertia"])),
                    ],
                )
                if on_rails:
                    add_rail_wheel_colliders(t_ws, f"tdr_{w_num}", r_tdr, y_gauge_half, use_explicit_inertial=True, is_driving=False)
                for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
                    ET.SubElement(
                        t_ws, "geom", name=f"tdr_vis_{w_num}_{s_tag}", type="mesh",
                        mesh=f"m_truck_wheel_{s_tag}", material="mat_black_iron",
                        pos=f"0 {sgn * y_tw_off:.6f} 0", contype="0", conaffinity="0", mass="0"
                    )

        if on_rails:
            ET.SubElement(
                equality, "connect", name="loop_tender_drawbar", site1="s_drawbar_rear", site2="s_tender_front_pin",
                solref="0.0004 1.0", solimp="0.998 0.9998 0.000001"
            )
            ET.SubElement(sensor, "framepos", name="loco_world_pos", objtype="body", objname="loco_frame")
            ET.SubElement(sensor, "jointpos", name="leading_yaw_rad", joint="j_leading_yaw")
            ET.SubElement(sensor, "jointpos", name="drawbar_yaw_rad", joint="j_drawbar_yaw")

    # [契约硬校验] 严禁任何运动刚体缺少真实材料惯量！
    for body_el in mj.iter("body"):
        has_joint = (body_el.find("joint") is not None) or (body_el.find("freejoint") is not None)
        has_inertial = body_el.find("inertial") is not None
        has_mass_geom = any(
            (g.get("density") is not None) or (float(g.get("mass", "0.0")) > 0.0)
            for g in body_el.findall("geom")
        )
        if has_joint and (not has_inertial) and (not has_mass_geom):
            raise RuntimeError(f"违反 GEMINI.md 契约: 运动刚体 '{body_el.get('name')}' 未分配真实材料质量!")

    raw_xml = ET.tostring(mj, encoding="utf-8")
    return minidom.parseString(raw_xml).toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(description="Aster D51 MJCF 编译器 (精确材料密度积分惯量版)")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--mass-props", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--out", type=Path, default=Path("models/d51_full.xml"))
    parser.add_argument("--mode", choices=["roller", "track", "curve_r2m"], default="roller")
    parser.add_argument("--no-tender", action="store_true")
    args = parser.parse_args()

    if not args.mass_props.exists():
        raise FileNotFoundError(f"未找到 {args.mass_props}，请先运行 python scripts/02_build_cad_meshes.py 生成精确材料惯量库！")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mass_props = json.loads(args.mass_props.read_text(encoding="utf-8"))
    xml_str = build_mjcf(cfg, mass_props, mode=args.mode, include_tender=not args.no_tender)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml_str, encoding="utf-8")
    print(f"[DONE] 已注入精确材料惯量并生成模型: {args.out} (模式: {args.mode})")


if __name__ == "__main__":
    main()
