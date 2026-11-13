#!/usr/bin/env python3
"""
scripts/03_generate_mjcf.py
Aster D51 + 8-20B 煤水车全列闭环 MJCF 编译器 (真实金属材料惯量积分版)。
严格读取 configs/d51_mass_props.json，通过旋转变换与平行轴定理生成精确 <inertial fullinertia="..."/>，
彻底废除任何统一空壳质量兜底。
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
    # 确保对称主对角正定裕度
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
    """利用平行轴定理合并同一刚体下多个子零件 (如左轮 + 右轮 + 钢车轴) 的真实偏心质心与 3x3 惯量张量。"""
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


def solve_side_pose(theta_rad: float, cfg: Dict) -> Dict[str, Tuple[float, float]]:
    s = 1e-3
    rc_cfg, vg_cfg = cfg["cylinder_and_rods"], cfg["walschaerts_valve_gear"]
    r_crank, L_main = rc_cfg["crank_radius"] * s, rc_cfg["main_rod_length"] * s
    z_cyl, z_valve = rc_cfg["cylinder_center_z"] * s, rc_cfg["valve_axis_z"] * s

    cx, cz = r_crank * math.cos(theta_rad), r_crank * math.sin(theta_rad)
    px, pz = cx + math.sqrt(L_main**2 - (z_cyl - cz) ** 2), z_cyl

    theta_ret = theta_rad + math.radians(vg_cfg["return_crank_phase_deg"])
    r_ret = vg_cfg["return_crank_radius"] * s
    rx, rz = r_ret * math.cos(theta_ret), r_ret * math.sin(theta_ret)

    trun = (vg_cfg["trunnion_x"] * s, vg_cfg["trunnion_z"] * s)
    e_tail = intersect_circles(
        (rx, rz), vg_cfg["eccentric_rod_length"] * s, trun, vg_cfg["exp_link_tail_radius"] * s, pick_upper=False
    )

    u0 = (px, pz - vg_cfg["crosshead_drop"] * s)
    L_rad = vg_cfg["radius_rod_length"] * s
    L_union = vg_cfg["union_link_length"] * s
    L_comb = vg_cfg["comb_lever_total_length"] * s
    L_mv = vg_cfg["comb_lever_lap_lead_offset"] * s

    def residual(phi: float):
        mz = z_valve - L_mv * math.cos(phi)
        mx = trun[0] + math.sqrt(max(0.0, L_rad**2 - (mz - trun[1]) ** 2))
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
    L_back = vg_cfg["radius_rod_back_ext"] * s
    dir_x, dir_z = (m_pin[0] - trun[0]) / L_rad, (m_pin[1] - trun[1]) / L_rad
    rad_lift = (trun[0] - L_back * dir_x, trun[1] - L_back * dir_z)

    return {
        "crank": (cx, cz), "crosshead": (px, pz), "drop_arm": u0,
        "ret_crank": (rx, rz), "exp_trunnion": trun, "exp_tail": e_tail,
        "valve_pin": v_pin, "comb_mid": m_pin, "comb_bot": u1_pin, "rad_lift": rad_lift,
    }


def exp_link_slot_tilt(theta_rad: float, cfg: Dict) -> float:
    """月牙板从 t=0 姿态转到摆动中位所需的角度 (rad)，用于把滑块槽对齐到中位铅垂方向。"""
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

    swing = [tail_angle(2.0 * math.pi * i / 720) for i in range(720)]
    mid = 0.5 * (max(swing) + min(swing))
    return mid - tail_angle(theta_rad + math.radians(vg_cfg["return_crank_phase_deg"]))


def add_rail_wheel_colliders(
    wbody: ET.Element,
    name_prefix: str,
    r_tread: float,
    y_gauge_half: float,
    use_explicit_inertial: bool,
    is_driving: bool = True,
) -> None:
    """生成双阶锥形踏面 + 轮缘碰撞体；当刚体已注入 CAD 精确 <inertial> 时碰撞体设 mass=0 防止重复计重。"""
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
    z_rail_axis = 0.004 - rail_half_w  # 轨顶高度严格保持 4.0mm，圆弧轨头半径 0.8mm 共球心无缝衔接

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
    ET.SubElement(default, "equality", solref="0.0015 1.0", solimp="0.95 0.995 0.00005")

    mesh_dir = Path("models/meshes")
    asset = ET.SubElement(mj, "asset")
    for m_name in parts_mp.keys():
        if (mesh_dir / f"{m_name}.stl").exists():
            ET.SubElement(asset, "mesh", name=f"m_{m_name}", file=f"{m_name}.stl", scale="0.001 0.001 0.001")
    ET.SubElement(asset, "material", name="mat_black_iron", rgba="0.14 0.14 0.16 1", specular="0.6", shininess="0.5")
    ET.SubElement(asset, "material", name="mat_machined_steel", rgba="0.78 0.80 0.84 1", specular="0.9", shininess="0.8")
    ET.SubElement(asset, "material", name="mat_brass", rgba="0.82 0.62 0.24 1", specular="0.85", shininess="0.75")

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

    # ==================== A. 蒸汽机车车架与锅炉上部总成 ====================
    R_curve = t_cfg["curve_radius_mm"] * s
    if mode == "curve_r2m":
        # 将四联耦合动轮组弦中点对准 R=2.0m 曲线顶弧，使 4 对动轮初始全部处于 ±1.1mm 轮缘游隙内零穿模
        x_drv_mid = 0.5 * (w_cfg["axle_x"]["drive_1"] + w_cfg["axle_x"]["drive_4"]) * s
        half_dwb = 0.5 * (w_cfg["axle_x"]["drive_1"] - w_cfg["axle_x"]["drive_4"]) * s
        sag_drv = R_curve - math.sqrt(R_curve**2 - half_dwb**2)
        loco_wx, loco_wy = -x_drv_mid, -0.5 * sag_drv
    else:
        loco_wx, loco_wy = 0.0, 0.0

    frame = ET.SubElement(world, "body", name="loco_frame", pos=f"{loco_wx:.6f} {loco_wy:.6f} {z_loco_axle:.6f}")
    if on_rails:
        ET.SubElement(frame, "freejoint", name="loco_free")

    # 精确合成：CAD 黄铜锅炉外壳 + 左右黄铜气缸体 + 钢制双侧棒式主车架 + 铜火管束(0.46kg)与50%锅水(0.21kg)
    mp_boil = parts_mp["boiler_superstructure"]
    mp_cylL, mp_cylR = parts_mp["cylinder_block_L"], parts_mp["cylinder_block_R"]
    cyl_pos_L = np.array([rc_cfg["cylinder_center_x"] * s, +rc_cfg["cylinder_center_y"] * s, rc_cfg["cylinder_center_z"] * s])
    cyl_pos_R = np.array([rc_cfg["cylinder_center_x"] * s, -rc_cfg["cylinder_center_y"] * s, rc_cfg["cylinder_center_z"] * s])
    m_side_plate = (0.330 * 0.0035 * 0.016) * rho_iron
    I_side_plate = (m_side_plate / 12.0) * np.diag([0.0035**2 + 0.016**2, 0.330**2 + 0.016**2, 0.330**2 + 0.0035**2])
    m_boil_core = 0.460 + 0.210  # 铜锅筒火管束 + 50% 水位液态水 (严格对应 thermo_1d.py)
    r_core, L_core = 0.5 * b_cfg["boiler_front_id"] * s, 0.204
    I_boil_core = np.diag([
        0.5 * m_boil_core * (r_core**2),
        (m_boil_core / 12.0) * (3.0 * (r_core**2) + L_core**2),
        (m_boil_core / 12.0) * (3.0 * (r_core**2) + L_core**2),
    ])
    m_front_saddle = 0.340  # 铸铁烟室鞍座 + 前端梁配重 + 置换注油器总成 (使整机质心位于第2~3动轮间)
    I_front_saddle = (m_front_saddle / 12.0) * np.diag([0.042**2 + 0.018**2, 0.055**2 + 0.018**2, 0.055**2 + 0.042**2])
    attach_composite_inertial(
        frame,
        [
            (mp_boil["mass_kg"], np.array(mp_boil["com_m"]), full_to_mat3(mp_boil["fullinertia"])),
            (mp_cylL["mass_kg"], np.array(mp_cylL["com_m"]) + cyl_pos_L, full_to_mat3(mp_cylL["fullinertia"])),
            (mp_cylR["mass_kg"], np.array(mp_cylR["com_m"]) + cyl_pos_R, full_to_mat3(mp_cylR["fullinertia"])),
            (m_side_plate, np.array([0.020, +0.0135, 0.004]), I_side_plate),
            (m_side_plate, np.array([0.020, -0.0135, 0.004]), I_side_plate),
            (m_boil_core, np.array([0.0225, 0.0, 0.044]), I_boil_core),
            (m_front_saddle, np.array([0.145, 0.0, 0.008]), I_front_saddle),
        ],
    )
    for s_tag, sgn_f in [("L", +1.0), ("R", -1.0)]:
        ET.SubElement(
            frame, "geom", name=f"main_chassis_{s_tag}", type="box",
            pos=f"0.020 {sgn_f * 0.0135:.6f} 0.004", size="0.165 0.00175 0.008",
            rgba="0.12 0.12 0.14 1", mass="0", contype="0", conaffinity="0"
        )
    ET.SubElement(
        frame, "geom", name="boiler_Super_vis", type="mesh",
        mesh="m_boiler_superstructure", material="mat_black_iron",
        pos="0 0 0", mass="0", contype="0", conaffinity="0"
    )

    equality = ET.SubElement(mj, "equality")
    actuator = ET.SubElement(mj, "actuator")
    sensor = ET.SubElement(mj, "sensor")

    # 1. 四对 Boxpok 驱动动轮 (含 5-9 黄铜轴箱精确质量 + 左右动轮月牙配重块真实偏心惯量合成)
    axle_xs = [
        (1, w_cfg["axle_x"]["drive_1"] * s, w_cfg["axle_lateral_play_half"]["drive_1"] * s),
        (2, w_cfg["axle_x"]["drive_2"] * s, w_cfg["axle_lateral_play_half"]["drive_2"] * s),
        (3, w_cfg["axle_x"]["drive_3"] * s, w_cfg["axle_lateral_play_half"]["drive_3"] * s),
        (4, w_cfg["axle_x"]["drive_4"] * s, max(0.0004, w_cfg["axle_lateral_play_half"]["drive_4"] * s)),
    ]
    mp_wL, mp_wR = parts_mp["driving_wheel_L"], parts_mp["driving_wheel_R"]
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
            limited="true", range="-0.0035 0.0035", stiffness="2000", damping="28"
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
        # 平行轴定理精确合成左轮(0°配重)+右轮(-90°配重)+不锈钢车轴
        attach_composite_inertial(
            wbody,
            [
                (mp_wL["mass_kg"], np.array(mp_wL["com_m"]) + np.array([0.0, +y_w_off, 0.0]), full_to_mat3(mp_wL["fullinertia"])),
                (mp_wR["mass_kg"], np.array(mp_wR["com_m"]) + np.array([0.0, -y_w_off, 0.0]), full_to_mat3(mp_wR["fullinertia"])),
                (m_axle_rod, np.zeros(3), I_axle_rod),
            ],
        )
        ET.SubElement(
            wbody, "geom", type="cylinder", fromto=f"0 {-y_gauge_half:.6f} 0  0 {y_gauge_half:.6f} 0",
            size="0.0028", rgba="0.3 0.3 0.3 1", mass="0", contype="0", conaffinity="0"
        )
        add_rail_wheel_colliders(wbody, f"drv_{idx}", r_drv, y_gauge_half, use_explicit_inertial=True)
        for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            ET.SubElement(
                wbody, "geom", name=f"wheel_vis_{idx}_{s_tag}", type="mesh",
                mesh=f"m_driving_wheel_{s_tag}", material="mat_black_iron",
                pos=f"0 {sgn * y_w_off:.6f} 0", contype="0", conaffinity="0", mass="0"
            )
        wheel_bodies[idx] = wbody

    # 2. 径向摆臂先导台车 (19-2 铸铁摆臂实体赋密度) 与从轮台车
    tz_truck = -(r_drv - r_trk)
    mp_tL, mp_tR = parts_mp["truck_wheel_L"], parts_mp["truck_wheel_R"]
    y_tw_off = y_gauge_half + 0.0006

    lead_axle_x = w_cfg["axle_x"]["leading"] * s
    lead_pivot_x = lead_axle_x - (0.052 if mode == "curve_r2m" else 0.062)
    lead_arm_len = lead_axle_x - lead_pivot_x
    if mode == "curve_r2m":
        # 兼容 R2m(52mm) 与 R3m(62mm, ±0.090rad) 两种支点孔位，使先导轮 t=0 落于轨面中心
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
    ET.SubElement(
        lead_arm, "joint", name="j_leading_susp", type="slide", axis="0 0 1",
        springref="-0.0065" if on_rails else "0",
        limited="true", range="-0.004 0.004", stiffness="1050", damping="16"
    )
    ET.SubElement(
        lead_arm, "geom", name="lead_arm_bar", type="box",
        pos=f"{0.5 * lead_arm_len:.6f} 0 0", size=f"{0.5 * lead_arm_len:.6f} 0.012 0.0025",
        rgba="0.18 0.18 0.20 1", density=f"{rho_iron:.1f}", contype="0", conaffinity="0"
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
    ET.SubElement(
        trail_arm, "joint", name="j_trailing_susp", type="slide", axis="0 0 1",
        springref="-0.0055" if on_rails else "0",
        limited="true", range="-0.004 0.004", stiffness="1050", damping="16"
    )
    ET.SubElement(
        trail_arm, "geom", name="trail_arm_bar", type="box",
        pos=f"{0.5 * trail_arm_dx:.6f} 0 0", size=f"{0.5 * abs(trail_arm_dx):.6f} 0.014 0.003",
        rgba="0.18 0.18 0.20 1", density=f"{rho_iron:.1f}", contype="0", conaffinity="0"
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

    # 3. 中央轴动给水泵 (6-5 黄铜柱塞) 与 4-12 逆转器横轴 (精确不锈钢 CAD 惯量)
    r_pump, x_pump = b_cfg["axle_pump_eccentric"] * s, b_cfg["axle_pump_x"] * s
    pump_ram = ET.SubElement(frame, "body", name="axle_pump_ram", pos=f"{x_pump:.6f} 0 0")
    ET.SubElement(
        pump_ram, "joint", name="j_pump_ram", type="slide", axis="1 0 0",
        limited="true", range="-0.012 0.012", damping="0.4"
    )
    ET.SubElement(
        pump_ram, "geom", type="cylinder", fromto="-0.008 0 0  0.008 0 0",
        size="0.0016", density="8500", contype="0", conaffinity="0"
    )
    ET.SubElement(pump_ram, "site", name="s_pump_pin", pos="0 0 0")

    pump_strap = ET.SubElement(wheel_bodies[3], "body", name="pump_eccentric_rod", pos=f"{r_pump:.6f} 0 0")
    ET.SubElement(pump_strap, "joint", name="j_pump_ecc", type="hinge", axis="0 1 0")
    ET.SubElement(
        pump_strap, "geom", type="capsule", fromto=f"0 0 0  {x_pump - r_pump:.6f} 0 0",
        size="0.0012", rgba="0.75 0.65 0.25 1", density="8500", contype="0", conaffinity="0"
    )
    ET.SubElement(pump_strap, "site", name="s_pump_rod_end", pos=f"{x_pump - r_pump:.6f} 0 0")
    ET.SubElement(equality, "connect", name="loop_axle_pump", site1="s_pump_rod_end", site2="s_pump_pin")

    rev_x, rev_z = vg_cfg["reverser_shaft_x"] * s, vg_cfg["reverser_shaft_z"] * s
    rev_shaft = ET.SubElement(frame, "body", name="reverser_shaft", pos=f"{rev_x:.6f} 0 {rev_z:.6f}")
    attach_part_inertial(rev_shaft, parts_mp["reverser_shaft"])
    # 计入驾驶室螺杆逆转机 (11-18) 经逆转棒 (4-13) 折算的等效传动惯量与自锁阻尼，防止 kp=80 在 dt=0.5ms 下显式发散
    ET.SubElement(
        rev_shaft, "joint", name="j_reverser", type="hinge", axis="0 1 0",
        limited="true", range="-0.50 0.50", armature="0.0015", damping="0.6"
    )
    ET.SubElement(
        rev_shaft, "geom", name="reverser_shaft_vis", type="mesh",
        mesh="m_reverser_shaft", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
    )

    # 4. 左右双侧气缸、连杆、十字头、平面滑阀与月牙滑块精确材料惯量闭环
    sides = [("L", +1.0, 0.0), ("R", -1.0, math.radians(w_cfg["quartering_deg_R_minus_L"]))]
    for tag, sgn, theta in sides:
        pose = solve_side_pose(theta, cfg)
        y_cyl = sgn * (rc_cfg["cylinder_center_y"] * s)
        y_coup = sgn * (y_gauge_half + 0.0055)
        y_vg = sgn * (rc_cfg["cylinder_center_y"] * s + 0.0035)

        ET.SubElement(
            frame, "geom", name=f"cyl_housing_{tag}", type="mesh",
            mesh=f"m_cylinder_block_{tag}", material="mat_black_iron",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} {rc_cfg['cylinder_center_z']*s:.6f}",
            mass="0", contype="0", conaffinity="0"
        )

        cx, cz = pose["crank"]
        for idx, _, _ in axle_xs:
            ET.SubElement(
                wheel_bodies[idx], "site", name=f"s_crank_pin_{idx}_{tag}",
                pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}", size="0.002"
            )
        coup_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"coupling_rod_{tag}", pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}"
        )
        attach_part_inertial(coup_rod, parts_mp["coupling_rod"])
        ET.SubElement(coup_rod, "joint", name=f"j_coup3_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            coup_rod, "geom", name=f"coupling_rod_vis_{tag}", type="mesh",
            mesh="m_coupling_rod", material="mat_machined_steel", contype="0", conaffinity="0", mass="0"
        )
        for target_idx in [1, 2, 4]:
            dx_i = w_cfg["axle_x"][f"drive_{target_idx}"] * s
            # 5-10 曲柄销轴向游隙衬套: 锁死 (X,Z) 旋转相位同时释放图 5.1 动轮轴箱横动自由度 (j_lat_y1..4)
            bush = ET.SubElement(coup_rod, "body", name=f"coup_bush_{target_idx}_{tag}", pos=f"{dx_i:.6f} 0 0")
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

        px, pz = pose["crosshead"]
        pitch_y = -math.atan2(pz - cz, px - cx)
        main_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"main_rod_{tag}", pos=f"{cx:.6f} {y_cyl:.6f} {cz:.6f}"
        )
        attach_part_inertial(main_rod, parts_mp["main_rod"], pitch_y_rad=pitch_y)
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

        # 华氏细杆件 (按 SUS 不锈钢真实密度 7930 kg/m³ 自动体积积分)
        u0x, u0z = pose["drop_arm"]
        u1x, u1z = pose["comb_bot"]
        mx, mz = pose["comb_mid"]
        vx, vz = pose["valve_pin"]

        union_body = ET.SubElement(
            piston_body, "body", name=f"union_link_{tag}", pos=f"{u0x - px:.6f} {y_vg - y_cyl:.6f} {u0z - pz:.6f}"
        )
        ET.SubElement(union_body, "joint", name=f"j_union_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            union_body, "geom", type="capsule", fromto=f"0 0 0  {u1x - u0x:.6f} 0 {u1z - u0z:.6f}",
            size="0.0011", rgba="0.75 0.75 0.8 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        comb_body = ET.SubElement(union_body, "body", name=f"comb_lever_{tag}", pos=f"{u1x - u0x:.6f} 0 {u1z - u0z:.6f}")
        ET.SubElement(comb_body, "joint", name=f"j_comb_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            comb_body, "geom", type="capsule", fromto=f"0 0 0  {vx - u1x:.6f} 0 {vz - u1z:.6f}",
            size="0.0012", rgba="0.82 0.82 0.85 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(comb_body, "site", name=f"s_comb_radius_{tag}", pos=f"{mx - u1x:.6f} 0 {mz - u1z:.6f}")
        ET.SubElement(comb_body, "site", name=f"s_comb_valve_{tag}", pos=f"{vx - u1x:.6f} 0 {vz - u1z:.6f}")

        # 1-6 平面滑阀 + 1-22 阀芯棒总成 (左右独立 Plane.XZ 镜像 CAD 实体与精确复合惯量)
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
        rlx, rlz = pose["rad_lift"]

        ecc_body = ET.SubElement(
            wheel_bodies[3], "body", name=f"eccentric_rod_{tag}", pos=f"{rx:.6f} {y_vg:.6f} {rz:.6f}"
        )
        ET.SubElement(ecc_body, "joint", name=f"j_retcrank_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            ecc_body, "geom", type="capsule", fromto=f"0 0 0  {etx - rx:.6f} 0 {etz - rz:.6f}",
            size="0.0012", rgba="0.72 0.72 0.76 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(ecc_body, "site", name=f"s_ecc_front_{tag}", pos=f"{etx - rx:.6f} 0 {etz - rz:.6f}")

        exp_pitch_y = math.atan2(-(etx - tx), -(etz - tz))
        exp_body = ET.SubElement(frame, "body", name=f"expansion_link_{tag}", pos=f"{tx:.6f} {y_vg:.6f} {tz:.6f}")
        attach_part_inertial(exp_body, parts_mp["expansion_link"], pitch_y_rad=exp_pitch_y)
        ET.SubElement(exp_body, "joint", name=f"j_explink_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            exp_body, "geom", name=f"exp_link_vis_{tag}", type="mesh",
            mesh="m_expansion_link", material="mat_machined_steel",
            euler=f"0 {exp_pitch_y:.6f} 0", contype="0", conaffinity="0", mass="0"
        )
        ET.SubElement(exp_body, "site", name=f"s_exp_tail_{tag}", pos=f"{etx - tx:.6f} 0 {etz - tz:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_eccentric_{tag}", site1=f"s_ecc_front_{tag}", site2=f"s_exp_tail_{tag}"
        )

        # 2-3 月牙滑块 (真实 ~0.34g 黄铜滑块质量；保留 exp_link_slot_tilt 修正右侧月牙板 t=0 偏角)
        slot_lim = vg_cfg["die_slot_half_travel"] * s
        slot_tilt = exp_link_slot_tilt(theta, cfg)
        die_body = ET.SubElement(exp_body, "body", name=f"die_block_{tag}", pos="0 0 0")
        attach_part_inertial(die_body, parts_mp["die_block"], pitch_y_rad=exp_pitch_y)
        ET.SubElement(
            die_body, "joint", name=f"j_die_shift_{tag}", type="slide",
            axis=f"{math.sin(slot_tilt):.6f} 0 {math.cos(slot_tilt):.6f}",
            limited="true", range=f"{-slot_lim:.6f} {slot_lim:.6f}", damping="0.08"
        )
        ET.SubElement(
            die_body, "geom", name=f"die_vis_{tag}", type="mesh",
            mesh="m_die_block", material="mat_brass",
            euler=f"0 {exp_pitch_y:.6f} 0", contype="0", conaffinity="0", mass="0"
        )

        rad_body = ET.SubElement(die_body, "body", name=f"radius_rod_{tag}", pos="0 0 0")
        ET.SubElement(rad_body, "joint", name=f"j_radius_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            rad_body, "geom", type="capsule", fromto=f"{rlx - tx:.6f} 0 {rlz - tz:.6f}  {mx - tx:.6f} 0 {mz - tz:.6f}",
            size="0.0011", rgba="0.80 0.80 0.84 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(rad_body, "site", name=f"s_rad_front_{tag}", pos=f"{mx - tx:.6f} 0 {mz - tz:.6f}")
        ET.SubElement(rad_body, "site", name=f"s_rad_lift_{tag}", pos=f"{rlx - tx:.6f} 0 {rlz - tz:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_radius_{tag}", site1=f"s_rad_front_{tag}", site2=f"s_comb_radius_{tag}"
        )

        lift_arm = ET.SubElement(rev_shaft, "body", name=f"lift_link_{tag}", pos=f"{rlx - rev_x:.6f} {y_vg:.6f} 0")
        ET.SubElement(lift_arm, "joint", name=f"j_lift_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            lift_arm, "geom", type="capsule", fromto=f"0 0 0  0 0 {rlz - rev_z:.6f}",
            size="0.0010", rgba="0.7 0.7 0.72 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(lift_arm, "site", name=f"s_lift_bot_{tag}", pos=f"0 0 {rlz - rev_z:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_reverser_{tag}", site1=f"s_lift_bot_{tag}", site2=f"s_rad_lift_{tag}"
        )

        ET.SubElement(actuator, "general", name=f"steam_force_{tag}", joint=f"j_piston_{tag}", gainprm="1")
        ET.SubElement(sensor, "jointpos", name=f"piston_pos_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointvel", name=f"piston_vel_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointpos", name=f"valve_pos_{tag}", joint=f"j_valve_{tag}")

    ET.SubElement(actuator, "position", name="reverser_ctrl", joint="j_reverser", kp="40", ctrlrange="-0.45 0.45")
    ET.SubElement(sensor, "jointvel", name="wheel3_omega", joint="j_wheel_3")
    ET.SubElement(sensor, "jointvel", name="pump_ram_vel", joint="j_pump_ram")

    # ==================== B. 四轴双转向架煤水车与牵引拉杆 ====================
    if include_tender and on_rails:
        x_pin_loco = t_cfg["drawbar_loco_pin_x"] * s
        L_drawbar = t_cfg["drawbar_length"] * s
        x_pin_tdr_local = t_cfg["drawbar_tender_pin_local_x"] * s
        z_tdr_deck = rail_top_z + (2.0 * r_tdr) - 0.009
        half_span = 0.5 * t_cfg["bogie_center_span"] * s
        half_wb = 0.5 * t_cfg["bogie_wheelbase"] * s

        pin_loco_wx = loco_wx + x_pin_loco
        pin_loco_wy = loco_wy

        if mode == "curve_r2m":
            R_tdr = math.sqrt(R_curve**2 - half_wb**2 - half_span**2)
            phi_loco_pin = math.atan2(pin_loco_wy + R_curve, pin_loco_wx)
            phi_tdr = phi_loco_pin + (L_drawbar + x_pin_tdr_local) / R_curve
            tdr_wx = R_tdr * math.cos(phi_tdr)
            tdr_wy = -R_curve + R_tdr * math.sin(phi_tdr)
            tdr_yaw = phi_tdr - 0.5 * math.pi
        else:
            tdr_wx, tdr_wy, tdr_yaw = pin_loco_wx - L_drawbar - x_pin_tdr_local, 0.0, 0.0

        pin_tdr_wx = tdr_wx + x_pin_tdr_local * math.cos(tdr_yaw)
        pin_tdr_wy = tdr_wy + x_pin_tdr_local * math.sin(tdr_yaw)
        db_dx, db_dy = pin_tdr_wx - pin_loco_wx, pin_tdr_wy - pin_loco_wy

        drawbar = ET.SubElement(frame, "body", name="drawbar_8_8", pos=f"{x_pin_loco:.6f} 0 -0.006")
        ET.SubElement(drawbar, "joint", name="j_drawbar_yaw", type="hinge", axis="0 0 1", damping="0.02")
        ET.SubElement(drawbar, "joint", name="j_drawbar_pitch", type="hinge", axis="0 1 0", damping="0.02")
        ET.SubElement(
            drawbar, "geom", type="capsule", fromto=f"0 0 0  {db_dx:.6f} {db_dy:.6f} 0",
            size="0.0022", rgba="0.22 0.22 0.25 1", density=f"{rho_sus:.1f}", contype="0", conaffinity="0"
        )
        ET.SubElement(drawbar, "site", name="s_drawbar_rear", pos=f"{db_dx:.6f} {db_dy:.6f} 0")

        tender = ET.SubElement(
            world, "body", name="tender_body",
            pos=f"{tdr_wx:.6f} {tdr_wy:.6f} {z_tdr_deck:.6f}", euler=f"0 0 {tdr_yaw:.6f}"
        )
        ET.SubElement(tender, "freejoint", name="tender_free")
        # 煤水车初始总重 = CAD 积分黄铜板壳空重 + 320ml 水 + 130ml 酒精 (0.422 kg) 平行轴合成惯量
        mp_tb = parts_mp["tender_body"]
        m_tdr_fluid = 0.422
        I_tdr_fluid = (m_tdr_fluid / 12.0) * np.diag([0.070**2 + 0.036**2, 0.200**2 + 0.036**2, 0.200**2 + 0.070**2])
        attach_composite_inertial(
            tender,
            [
                (mp_tb["mass_kg"], np.array(mp_tb["com_m"]), full_to_mat3(mp_tb["fullinertia"])),
                (m_tdr_fluid, np.array([-0.015, 0.0, 0.022]), I_tdr_fluid),
            ],
        )
        z_pin_local = (z_loco_axle - 0.006) - z_tdr_deck
        ET.SubElement(tender, "site", name="s_tender_front_pin", pos=f"{x_pin_tdr_local:.6f} 0 {z_pin_local:.6f}")
        ET.SubElement(
            tender, "geom", name="tender_body_vis", type="mesh",
            mesh="m_tender_body", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
        )

        z_bogie_local = (rail_top_z + r_tdr + 0.0002) - z_tdr_deck

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
                ET.SubElement(bogie, "joint", name=f"j_tbogie_{b_idx}_yaw", type="hinge", axis="0 0 1", damping="0.015")
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
                ET.SubElement(
                    t_abox, "joint", name=f"j_tsusp_z{w_num}", type="slide", axis="0 0 1",
                    limited="true", range="-0.002 0.002", stiffness="900", damping="12"
                )
                ET.SubElement(
                    t_abox, "geom", name=f"tabox_vis_{w_num}", type="mesh",
                    mesh="m_tender_axlebox", material="mat_black_iron", contype="0", conaffinity="0", mass="0"
                )

                t_ws = ET.SubElement(t_abox, "body", name=f"tender_wheelset_{w_num}", pos="0 0 0")
                ET.SubElement(t_ws, "joint", name=f"j_twheel_{w_num}", type="hinge", axis="0 1 0", damping="0.0005")
                attach_composite_inertial(
                    t_ws,
                    [
                        (mp_tL["mass_kg"], np.array(mp_tL["com_m"]) + np.array([0.0, +y_tw_off, 0.0]), full_to_mat3(mp_tL["fullinertia"])),
                        (mp_tR["mass_kg"], np.array(mp_tR["com_m"]) + np.array([0.0, -y_tw_off, 0.0]), full_to_mat3(mp_tR["fullinertia"])),
                    ],
                )
                add_rail_wheel_colliders(t_ws, f"tdr_{w_num}", r_tdr, y_gauge_half, use_explicit_inertial=True, is_driving=False)
                for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
                    ET.SubElement(
                        t_ws, "geom", name=f"tdr_vis_{w_num}_{s_tag}", type="mesh",
                        mesh=f"m_truck_wheel_{s_tag}", material="mat_black_iron",
                        pos=f"0 {sgn * y_tw_off:.6f} 0", contype="0", conaffinity="0", mass="0"
                    )

        ET.SubElement(
            equality, "connect", name="loop_tender_drawbar", site1="s_drawbar_rear", site2="s_tender_front_pin"
        )
        ET.SubElement(sensor, "framepos", name="loco_world_pos", objtype="body", objname="loco_frame")
        ET.SubElement(sensor, "jointpos", name="leading_yaw_rad", joint="j_leading_yaw")
        ET.SubElement(sensor, "jointpos", name="drawbar_yaw_rad", joint="j_drawbar_yaw")

    # [契约硬校验] 严禁任何运动刚体缺少真实材料惯量！若漏设直接抛错提示而非盲目兜底
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