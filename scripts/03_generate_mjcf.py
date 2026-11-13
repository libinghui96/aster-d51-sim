#!/usr/bin/env python3
"""
读取 configs/d51_kinematics.json，精确求解左右 90° 正交相位下的闭环初始位姿，
自动编译输出完整的 Aster D51 MuJoCo 物理模型 (models/d51_full.xml)。
"""

import argparse
import json
import math
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple


def intersect_circles(
    c1: Tuple[float, float], r1: float, c2: Tuple[float, float], r2: float, pick_upper: bool = False
) -> Tuple[float, float]:
    """求解 XZ 平面内两圆交点，用于精确闭合偏心杆与月牙板尾耳等四杆环路。"""
    dx, dz = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dz)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0.0:
        raise ValueError(f"连杆几何无法闭合: 圆心距 d={d*1000:.2f}mm, r1={r1*1000:.2f}mm, r2={r2*1000:.2f}mm")
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(0.0, r1**2 - a**2))
    mx = c1[0] + a * (dx / d)
    mz = c1[1] + a * (dz / d)
    rx = -dz * (h / d)
    rz = dx * (h / d)
    p_a = (mx + rx, mz + rz)
    p_b = (mx - rx, mz - rz)
    return (p_a if p_a[1] >= p_b[1] else p_b) if pick_upper else (p_a if p_a[1] <= p_b[1] else p_b)


def solve_side_pose(theta_rad: float, cfg: Dict) -> Dict[str, Tuple[float, float]]:
    """
    给定单侧动轮主曲柄相位角 theta_rad (弧度)，在米制 (m) 下精确求解该侧全部运动构件在 t=0 的铰接点 (x, z)。
    此时逆转器处于中立位 N (滑块 2-3 重合于月牙板定轴 trunnion)。
    """
    s = 1e-3  # mm -> m
    rc_cfg = cfg["cylinder_and_rods"]
    vg_cfg = cfg["walschaerts_valve_gear"]

    r_crank = rc_cfg["crank_radius"] * s
    L_main = rc_cfg["main_rod_length"] * s
    z_cyl = rc_cfg["cylinder_center_z"] * s
    z_valve = rc_cfg["valve_axis_z"] * s

    # 1. 主曲柄销 C 与十字头主销 P (沿 z = z_cyl 直线滑动)
    cx, cz = r_crank * math.cos(theta_rad), r_crank * math.sin(theta_rad)
    px = cx + math.sqrt(L_main**2 - (z_cyl - cz) ** 2)
    pz = z_cyl

    # 2. 回动曲柄销 R (距轮心 7.0mm, 滞后/超前 90°) 与月牙板尾耳 E_tail
    theta_ret = theta_rad + math.radians(vg_cfg["return_crank_phase_deg"])
    r_ret = vg_cfg["return_crank_radius"] * s
    rx, rz = r_ret * math.cos(theta_ret), r_ret * math.sin(theta_ret)

    trun = (vg_cfg["trunnion_x"] * s, vg_cfg["trunnion_z"] * s)
    r_tail = vg_cfg["exp_link_tail_radius"] * s
    L_ecc = vg_cfg["eccentric_rod_length"] * s
    e_tail = intersect_circles((rx, rz), L_ecc, trun, r_tail, pick_upper=False)

    # 3. 合并杆 (1-23) 倾角 phi 一维求根：同时满足心向棒长度 L_rad 与短连杆长度 L_union
    u0 = (px, pz - vg_cfg["crosshead_drop"] * s)
    L_rad = vg_cfg["radius_rod_length"] * s
    L_union = vg_cfg["union_link_length"] * s
    L_comb = vg_cfg["comb_lever_total_length"] * s
    L_mv = vg_cfg["comb_lever_lap_lead_offset"] * s

    def residual(phi: float) -> Tuple[float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        # 顶孔 V 位于滑阀轴线 z = z_valve 上；次上孔 M 接心向棒前端；底孔 U1 接短连杆
        mz = z_valve - L_mv * math.cos(phi)
        rad_dx = math.sqrt(max(0.0, L_rad**2 - (mz - trun[1]) ** 2))
        mx = trun[0] + rad_dx
        vx = mx + L_mv * math.sin(phi)
        u1 = (vx - L_comb * math.sin(phi), z_valve - L_comb * math.cos(phi))
        err = (u1[0] - u0[0]) - math.copysign(
            math.hypot(u1[0] - u0[0], u1[1] - u0[1]), u1[0] - u0[0]
        )
        dist_err = math.hypot(u1[0] - u0[0], u1[1] - u0[1]) - L_union
        return dist_err, (vx, z_valve), (mx, mz), u1

    # 二分法扫描合并杆摆角 phi in [-0.75, +0.75] rad
    lo, hi = -0.75, 0.75
    f_lo = residual(lo)[0]
    # 若单调区间偏移，先做粗网格定位变号区间
    grid = [lo + i * (hi - lo) / 200.0 for i in range(201)]
    vals = [residual(g)[0] for g in grid]
    best_i = min(range(len(grid) - 1), key=lambda i: abs(vals[i]))
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] <= 0:
            lo, hi = grid[i], grid[i + 1]
            break
    else:
        lo = max(-0.75, grid[best_i] - 0.02)
        hi = min(0.75, grid[best_i] + 0.02)

    f_lo = residual(lo)[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)[0]
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    _, v_pin, m_pin, u1_pin = residual(0.5 * (lo + hi))

    # 心向棒后端吊杆孔 (在 D -> M 反方向延伸 L_back)
    L_back = vg_cfg["radius_rod_back_ext"] * s
    dir_x = (m_pin[0] - trun[0]) / L_rad
    dir_z = (m_pin[1] - trun[1]) / L_rad
    rad_lift = (trun[0] - L_back * dir_x, trun[1] - L_back * dir_z)

    return {
        "crank": (cx, cz),
        "crosshead": (px, pz),
        "drop_arm": u0,
        "ret_crank": (rx, rz),
        "exp_trunnion": trun,
        "exp_tail": e_tail,
        "valve_pin": v_pin,
        "comb_mid": m_pin,
        "comb_bot": u1_pin,
        "rad_lift": rad_lift,
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


def fmt(vec) -> str:
    return " ".join(f"{v:.6f}" for v in vec)


def build_mjcf(cfg: Dict, mode: str = "roller") -> str:
    s = 1e-3  # mm -> m
    w_cfg = cfg["wheels_and_axles"]
    rc_cfg = cfg["cylinder_and_rods"]
    vg_cfg = cfg["walschaerts_valve_gear"]
    b_cfg = cfg["boiler_and_aux"]

    r_drv = 0.5 * w_cfg["driving_wheel_diameter"] * s
    r_trk = 0.5 * w_cfg["leading_wheel_diameter"] * s
    r_crank = rc_cfg["crank_radius"] * s
    y_gauge_half = 0.5 * cfg["meta"]["track_gauge"] * s

    mj = ET.Element("mujoco", model="aster_d51_full")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mj, "option", timestep="0.0005", integrator="implicitfast", gravity="0 0 -9.81")

    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "geom", condim="3", friction="0.6 0.005 0.0001")
    ET.SubElement(default, "equality", solref="0.0015 1.0", solimp="0.95 0.995 0.00005")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 0.6", dir="0 0.5 -0.8", diffuse="0.9 0.9 0.9")
    ET.SubElement(world, "geom", name="ground", type="plane", size="2 2 0.05", rgba="0.18 0.20 0.22 1")

    # 若为轨道模式，生成两条间距 45mm 的钢轨与车架 6-DOF freejoint；滚轮台模式则架高悬空测试
    z_base = (r_drv + 0.020) if mode == "roller" else (r_drv + 0.004)
    if mode == "track":
        for side_tag, y_r in [("L", y_gauge_half), ("R", -y_gauge_half)]:
            ET.SubElement(
                world, "geom", name=f"rail_{side_tag}", type="box",
                pos=f"0 {y_r:.6f} 0.002", size="1.8 0.0012 0.002", rgba="0.65 0.65 0.68 1"
            )

    frame = ET.SubElement(world, "body", name="loco_frame", pos=f"0 0 {z_base:.6f}")
    if mode == "track":
        ET.SubElement(frame, "freejoint", name="loco_free")

    # 主车架、锥形锅筒与烟室主体外观及集中质量 (整机约 3.6kg)
    ET.SubElement(
        frame, "geom", name="main_chassis", type="box",
        pos="0.020 0 0.004", size="0.165 0.016 0.008", rgba="0.12 0.12 0.14 1", mass="0.8"
    )
    ET.SubElement(
        frame, "geom", name="boiler_barrel", type="cylinder",
        fromto="-0.115 0 0.044  0.145 0 0.044", size=f"{0.5 * b_cfg['smokebox_od'] * s:.6f}",
        rgba="0.10 0.10 0.12 1", mass="2.2"
    )

    equality = ET.SubElement(mj, "equality")
    actuator = ET.SubElement(mj, "actuator")
    sensor = ET.SubElement(mj, "sensor")

    # 1. 构建 4 对驱动动轮 (wheelset_1 .. wheelset_4)
    axle_xs = [
        (1, w_cfg["axle_x"]["drive_1"] * s, w_cfg["axle_lateral_play_half"]["drive_1"] * s),
        (2, w_cfg["axle_x"]["drive_2"] * s, w_cfg["axle_lateral_play_half"]["drive_2"] * s),
        (3, w_cfg["axle_x"]["drive_3"] * s, w_cfg["axle_lateral_play_half"]["drive_3"] * s),
        (4, w_cfg["axle_x"]["drive_4"] * s, w_cfg["axle_lateral_play_half"]["drive_4"] * s),
    ]
    wheel_bodies: Dict[int, ET.Element] = {}
    for idx, ax_x, lat_play in axle_xs:
        abox = ET.SubElement(frame, "body", name=f"axlebox_{idx}", pos=f"{ax_x:.6f} 0 0")
        ET.SubElement(
            abox, "joint", name=f"j_susp_z{idx}", type="slide", axis="0 0 1",
            limited="true", range="-0.0025 0.0025", stiffness="1500", damping="18"
        )
        ET.SubElement(
            abox, "joint", name=f"j_lat_y{idx}", type="slide", axis="0 1 0",
            limited="true", range=f"{-lat_play:.6f} {lat_play:.6f}", stiffness="120", damping="4"
        )
        wbody = ET.SubElement(abox, "body", name=f"wheelset_{idx}", pos="0 0 0")
        ET.SubElement(wbody, "joint", name=f"j_wheel_{idx}", type="hinge", axis="0 1 0", damping="0.0015")
        ET.SubElement(
            wbody, "geom", type="cylinder", fromto=f"0 {-y_gauge_half:.6f} 0  0 {y_gauge_half:.6f} 0",
            size="0.003", rgba="0.3 0.3 0.3 1", mass="0.015"
        )
        for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            y_in = sgn * (y_gauge_half - 0.0015)
            y_out = sgn * (y_gauge_half + 0.0015)
            ET.SubElement(
                wbody, "geom", name=f"wheel_{idx}_{s_tag}", type="cylinder",
                fromto=f"0 {y_in:.6f} 0  0 {y_out:.6f} 0", size=f"{r_drv:.6f}",
                rgba="0.16 0.16 0.18 1", mass="0.065"
            )
        wheel_bodies[idx] = wbody

    # 先导转向架 (19-9) 与从轮转向架 (19-14)
    for t_name, tx in [("leading", w_cfg["axle_x"]["leading"] * s), ("trailing", w_cfg["axle_x"]["trailing"] * s)]:
        tz = -(r_drv - r_trk)
        tbody = ET.SubElement(frame, "body", name=f"{t_name}_truck", pos=f"{tx:.6f} 0 {tz:.6f}")
        ET.SubElement(
            tbody, "joint", name=f"j_{t_name}_lat", type="slide", axis="0 1 0",
            limited="true", range="-0.008 0.008", stiffness="80", damping="3"
        )
        ET.SubElement(tbody, "joint", name=f"j_{t_name}_spin", type="hinge", axis="0 1 0", damping="0.0005")
        for s_tag, sgn in [("L", 1.0), ("R", -1.0)]:
            ET.SubElement(
                tbody, "geom", type="cylinder",
                fromto=f"0 {sgn*(y_gauge_half-0.001):.6f} 0  0 {sgn*(y_gauge_half+0.001):.6f} 0",
                size=f"{r_trk:.6f}", rgba="0.2 0.2 0.22 1", mass="0.025"
            )

    # 2. 中央轴动给水泵 (6-1..6-5, 由第 3 动轮偏心套驱动)
    r_pump = b_cfg["axle_pump_eccentric"] * s
    x_pump = b_cfg["axle_pump_x"] * s
    pump_ram = ET.SubElement(frame, "body", name="axle_pump_ram", pos=f"{x_pump:.6f} 0 0")
    ET.SubElement(
        pump_ram, "joint", name="j_pump_ram", type="slide", axis="1 0 0",
        limited="true", range="-0.012 0.012", damping="0.4"
    )
    ET.SubElement(pump_ram, "geom", type="cylinder", fromto="-0.008 0 0  0.008 0 0", size="0.0016", mass="0.005")
    ET.SubElement(pump_ram, "site", name="s_pump_pin", pos="0 0 0")

    pump_strap = ET.SubElement(wheel_bodies[3], "body", name="pump_eccentric_rod", pos=f"{r_pump:.6f} 0 0")
    ET.SubElement(pump_strap, "joint", name="j_pump_ecc", type="hinge", axis="0 1 0")
    ET.SubElement(
        pump_strap, "geom", type="capsule", fromto=f"0 0 0  {x_pump - r_pump:.6f} 0 0",
        size="0.0012", rgba="0.75 0.65 0.25 1", mass="0.006"
    )
    ET.SubElement(pump_strap, "site", name="s_pump_rod_end", pos=f"{x_pump - r_pump:.6f} 0 0")
    ET.SubElement(equality, "connect", name="loop_axle_pump", site1="s_pump_rod_end", site2="s_pump_pin")

    # 3. 统一逆转器横轴 (4-12 逆転器アームシャフト)
    rev_x, rev_z = vg_cfg["reverser_shaft_x"] * s, vg_cfg["reverser_shaft_z"] * s
    rev_shaft = ET.SubElement(frame, "body", name="reverser_shaft", pos=f"{rev_x:.6f} 0 {rev_z:.6f}")
    ET.SubElement(
        rev_shaft, "joint", name="j_reverser", type="hinge", axis="0 1 0",
        limited="true", range="-0.50 0.50", damping="0.3"
    )

    # 4. 左右双侧气缸、主/耦合连杆与华氏配气机构 (L: 0°, R: -90°)
    sides = [
        ("L", +1.0, 0.0),
        ("R", -1.0, math.radians(w_cfg["quartering_deg_R_minus_L"])),
    ]

    for tag, sgn, theta in sides:
        pose = solve_side_pose(theta, cfg)
        y_cyl = sgn * (rc_cfg["cylinder_center_y"] * s)
        y_coup = sgn * (y_gauge_half + 0.0055)
        y_vg = sgn * (rc_cfg["cylinder_center_y"] * s + 0.0035)

        # 4.1 静态气缸块与滑阀室壳体
        ET.SubElement(
            frame, "geom", name=f"cyl_housing_{tag}", type="box",
            pos=f"{rc_cfg['cylinder_center_x']*s:.6f} {y_cyl:.6f} 0.008",
            size="0.018 0.011 0.016", rgba="0.22 0.22 0.25 0.45", contype="0", conaffinity="0"
        )

        # 4.2 四轴耦合连杆 (挂载于第 3 主动轮曲柄销，连接第 1, 2, 4 动轮曲柄销)
        cx, cz = pose["crank"]
        for idx, ax_x, _ in axle_xs:
            ET.SubElement(
                wheel_bodies[idx], "site", name=f"s_crank_pin_{idx}_{tag}",
                pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}", size="0.002"
            )
        coup_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"coupling_rod_{tag}", pos=f"{cx:.6f} {y_coup:.6f} {cz:.6f}"
        )
        ET.SubElement(coup_rod, "joint", name=f"j_coup3_{tag}", type="hinge", axis="0 1 0")
        dx_f = w_cfg["axle_x"]["drive_1"] * s
        dx_r = w_cfg["axle_x"]["drive_4"] * s
        ET.SubElement(
            coup_rod, "geom", type="capsule", fromto=f"{dx_r:.6f} 0 0  {dx_f:.6f} 0 0",
            size="0.0016", rgba="0.78 0.78 0.82 1", mass="0.018", contype="0", conaffinity="0"
        )
        for target_idx in [1, 2, 4]:
            dx_i = w_cfg["axle_x"][f"drive_{target_idx}"] * s
            ET.SubElement(coup_rod, "site", name=f"s_coup_hole_{target_idx}_{tag}", pos=f"{dx_i:.6f} 0 0")
            ET.SubElement(
                equality, "connect", name=f"loop_coup_{target_idx}_{tag}",
                site1=f"s_coup_hole_{target_idx}_{tag}", site2=f"s_crank_pin_{target_idx}_{tag}"
            )

        # 4.3 主连杆 (7-11/7-16): 从第 3 轮主曲柄销连往十字头主销
        px, pz = pose["crosshead"]
        main_rod = ET.SubElement(
            wheel_bodies[3], "body", name=f"main_rod_{tag}", pos=f"{cx:.6f} {y_cyl:.6f} {cz:.6f}"
        )
        ET.SubElement(main_rod, "joint", name=f"j_main_crank_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            main_rod, "geom", type="capsule",
            fromto=f"0 0 0  {px - cx:.6f} 0 {pz - cz:.6f}",
            size="0.0018", rgba="0.82 0.82 0.86 1", mass="0.015", contype="0", conaffinity="0"
        )
        ET.SubElement(
            main_rod, "site", name=f"s_mainrod_small_{tag}", pos=f"{px - cx:.6f} 0 {pz - cz:.6f}"
        )

        # 4.4 活塞 + 十字头 (PR-15) + 短连杆 (1-27) + 合并杆 (1-23)
        # 滑动关节 range 以气缸中心对应的中性行程点做偏移，确保左右正负行程完全对称
        p_mid_x = math.sqrt((rc_cfg["main_rod_length"] * s) ** 2)
        slide_min = (p_mid_x - r_crank - 0.002) - px
        slide_max = (p_mid_x + r_crank + 0.002) - px

        piston_body = ET.SubElement(
            frame, "body", name=f"piston_crosshead_{tag}", pos=f"{px:.6f} {y_cyl:.6f} {pz:.6f}"
        )
        ET.SubElement(
            piston_body, "joint", name=f"j_piston_{tag}", type="slide", axis="1 0 0",
            limited="true", range=f"{slide_min:.6f} {slide_max:.6f}", damping="0.35", frictionloss="0.12"
        )
        p_head_dx = rc_cfg["cylinder_center_x"] * s - p_mid_x
        ET.SubElement(
            piston_body, "geom", name=f"piston_disc_{tag}", type="cylinder",
            fromto=f"{p_head_dx - 0.002:.6f} 0 0  {p_head_dx + 0.002:.6f} 0 0",
            size=f"{0.5 * rc_cfg['bore'] * s:.6f}", rgba="0.80 0.58 0.20 1", mass="0.016",
            contype="0", conaffinity="0"
        )
        ET.SubElement(
            piston_body, "geom", type="cylinder", fromto=f"0 0 0  {p_head_dx:.6f} 0 0",
            size="0.0015", rgba="0.9 0.9 0.9 1", mass="0.008", contype="0", conaffinity="0"
        )
        ET.SubElement(piston_body, "site", name=f"s_crosshead_pin_{tag}", pos="0 0 0")
        ET.SubElement(
            equality, "connect", name=f"loop_main_drive_{tag}",
            site1=f"s_mainrod_small_{tag}", site2=f"s_crosshead_pin_{tag}"
        )

        # 短连杆 (1-27) 与合并杆 (1-23)
        u0x, u0z = pose["drop_arm"]
        u1x, u1z = pose["comb_bot"]
        mx, mz = pose["comb_mid"]
        vx, vz = pose["valve_pin"]

        union_body = ET.SubElement(
            piston_body, "body", name=f"union_link_{tag}",
            pos=f"{u0x - px:.6f} {y_vg - y_cyl:.6f} {u0z - pz:.6f}"
        )
        ET.SubElement(union_body, "joint", name=f"j_union_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            union_body, "geom", type="capsule",
            fromto=f"0 0 0  {u1x - u0x:.6f} 0 {u1z - u0z:.6f}",
            size="0.001", rgba="0.75 0.75 0.8 1", mass="0.003", contype="0", conaffinity="0"
        )
        comb_body = ET.SubElement(
            union_body, "body", name=f"comb_lever_{tag}", pos=f"{u1x - u0x:.6f} 0 {u1z - u0z:.6f}"
        )
        ET.SubElement(comb_body, "joint", name=f"j_comb_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            comb_body, "geom", type="capsule",
            fromto=f"0 0 0  {vx - u1x:.6f} 0 {vz - u1z:.6f}",
            size="0.0011", rgba="0.82 0.82 0.85 1", mass="0.004", contype="0", conaffinity="0"
        )
        ET.SubElement(
            comb_body, "site", name=f"s_comb_radius_{tag}", pos=f"{mx - u1x:.6f} 0 {mz - u1z:.6f}"
        )
        ET.SubElement(
            comb_body, "site", name=f"s_comb_valve_{tag}", pos=f"{vx - u1x:.6f} 0 {vz - u1z:.6f}"
        )

        # 4.5 平面滑阀组件 (1-6 滑り弁 + 1-22 弁芯棒)
        valve_body = ET.SubElement(
            frame, "body", name=f"slide_valve_{tag}", pos=f"{vx:.6f} {y_vg:.6f} {vz:.6f}"
        )
        ET.SubElement(
            valve_body, "joint", name=f"j_valve_{tag}", type="slide", axis="1 0 0",
            limited="true", range="-0.008 0.008", damping="0.12", frictionloss="0.04"
        )
        v_box_dx = rc_cfg["cylinder_center_x"] * s - vx
        ET.SubElement(
            valve_body, "geom", name=f"valve_block_{tag}", type="box",
            pos=f"{v_box_dx:.6f} {y_cyl - y_vg:.6f} 0", size="0.0045 0.004 0.002",
            rgba="0.85 0.55 0.20 1", mass="0.006", contype="0", conaffinity="0"
        )
        ET.SubElement(valve_body, "site", name=f"s_valve_clevis_{tag}", pos="0 0 0")
        ET.SubElement(
            equality, "connect", name=f"loop_valve_{tag}",
            site1=f"s_comb_valve_{tag}", site2=f"s_valve_clevis_{tag}"
        )

        # 4.6 回动曲柄偏心杆 (2-12/2-19) -> 月牙加减链 (2-4) -> 滑块 (2-3) -> 心向棒 (2-1)
        rx, rz = pose["ret_crank"]
        etx, etz = pose["exp_tail"]
        tx, tz = pose["exp_trunnion"]
        rlx, rlz = pose["rad_lift"]

        ecc_body = ET.SubElement(
            wheel_bodies[3], "body", name=f"eccentric_rod_{tag}", pos=f"{rx:.6f} {y_vg:.6f} {rz:.6f}"
        )
        ET.SubElement(ecc_body, "joint", name=f"j_retcrank_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            ecc_body, "geom", type="capsule",
            fromto=f"0 0 0  {etx - rx:.6f} 0 {etz - rz:.6f}",
            size="0.0011", rgba="0.72 0.72 0.76 1", mass="0.005", contype="0", conaffinity="0"
        )
        ET.SubElement(ecc_body, "site", name=f"s_ecc_front_{tag}", pos=f"{etx - rx:.6f} 0 {etz - rz:.6f}")

        exp_body = ET.SubElement(
            frame, "body", name=f"expansion_link_{tag}", pos=f"{tx:.6f} {y_vg:.6f} {tz:.6f}"
        )
        ET.SubElement(exp_body, "joint", name=f"j_explink_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            exp_body, "geom", type="capsule",
            fromto=f"{etx - tx:.6f} 0 {etz - tz:.6f}  0 0 0.007",
            size="0.0014", rgba="0.65 0.65 0.70 1", mass="0.005", contype="0", conaffinity="0"
        )
        ET.SubElement(
            exp_body, "site", name=f"s_exp_tail_{tag}", pos=f"{etx - tx:.6f} 0 {etz - tz:.6f}"
        )
        ET.SubElement(
            equality, "connect", name=f"loop_eccentric_{tag}",
            site1=f"s_ecc_front_{tag}", site2=f"s_exp_tail_{tag}"
        )

        slot_lim = vg_cfg["die_slot_half_travel"] * s
        # R 侧 t=0 时月牙板偏离中位 ~31°，槽向若取世界竖直会使换档后滑阀中心整体偏移
        slot_tilt = exp_link_slot_tilt(theta, cfg)
        die_body = ET.SubElement(exp_body, "body", name=f"die_block_{tag}", pos="0 0 0")
        ET.SubElement(
            die_body, "joint", name=f"j_die_shift_{tag}", type="slide",
            axis=f"{math.sin(slot_tilt):.6f} 0 {math.cos(slot_tilt):.6f}",
            limited="true", range=f"{-slot_lim:.6f} {slot_lim:.6f}", damping="0.08"
        )
        rad_body = ET.SubElement(die_body, "body", name=f"radius_rod_{tag}", pos="0 0 0")
        ET.SubElement(rad_body, "joint", name=f"j_radius_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            rad_body, "geom", type="capsule",
            fromto=f"{rlx - tx:.6f} 0 {rlz - tz:.6f}  {mx - tx:.6f} 0 {mz - tz:.6f}",
            size="0.001", rgba="0.80 0.80 0.84 1", mass="0.004", contype="0", conaffinity="0"
        )
        ET.SubElement(rad_body, "site", name=f"s_rad_front_{tag}", pos=f"{mx - tx:.6f} 0 {mz - tz:.6f}")
        ET.SubElement(rad_body, "site", name=f"s_rad_lift_{tag}", pos=f"{rlx - tx:.6f} 0 {rlz - tz:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_radius_{tag}",
            site1=f"s_rad_front_{tag}", site2=f"s_comb_radius_{tag}"
        )

        # 4.7 逆转摇臂与吊杆 (2-10)，在 t=0 精确对齐心向棒后端孔 rlx, rlz
        lift_arm = ET.SubElement(
            rev_shaft, "body", name=f"lift_link_{tag}",
            pos=f"{rlx - rev_x:.6f} {y_vg:.6f} 0"
        )
        ET.SubElement(lift_arm, "joint", name=f"j_lift_{tag}", type="hinge", axis="0 1 0")
        ET.SubElement(
            lift_arm, "geom", type="capsule",
            fromto=f"0 0 0  0 0 {rlz - rev_z:.6f}",
            size="0.0009", rgba="0.7 0.7 0.72 1", mass="0.002", contype="0", conaffinity="0"
        )
        ET.SubElement(lift_arm, "site", name=f"s_lift_bot_{tag}", pos=f"0 0 {rlz - rev_z:.6f}")
        ET.SubElement(
            equality, "connect", name=f"loop_reverser_{tag}",
            site1=f"s_lift_bot_{tag}", site2=f"s_rad_lift_{tag}"
        )

        # 注册该侧蒸汽推力执行器与位移/速度传感器
        ET.SubElement(actuator, "general", name=f"steam_force_{tag}", joint=f"j_piston_{tag}", gainprm="1")
        ET.SubElement(sensor, "jointpos", name=f"piston_pos_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointvel", name=f"piston_vel_{tag}", joint=f"j_piston_{tag}")
        ET.SubElement(sensor, "jointpos", name=f"valve_pos_{tag}", joint=f"j_valve_{tag}")

    # 驾驶室逆转器位置伺服与车轮/给水泵传感器
    ET.SubElement(actuator, "position", name="reverser_ctrl", joint="j_reverser", kp="40", ctrlrange="-0.45 0.45")
    ET.SubElement(sensor, "jointvel", name="wheel3_omega", joint="j_wheel_3")
    ET.SubElement(sensor, "jointvel", name="pump_ram_vel", joint="j_pump_ram")

    # 自动检查并为所有不含自身质量 geom 的中间运动刚体 (axlebox, die_block, reverser_shaft 等) 注入真实金属惯量
    for body_el in mj.iter("body"):
        has_joint = (body_el.find("joint") is not None) or (body_el.find("freejoint") is not None)
        has_inertial = body_el.find("inertial") is not None
        has_mass_geom = any(
            float(g.get("mass", "1.0")) > 0.0 for g in body_el.findall("geom")
        )
        if has_joint and (not has_inertial) and (not has_mass_geom):
            ET.SubElement(
                body_el, "inertial", pos="0 0 0", mass="0.012", diaginertia="1e-5 1e-5 1e-5"
            )

    raw_xml = ET.tostring(mj, encoding="utf-8")
    return minidom.parseString(raw_xml).toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(description="Aster D51 全车闭环 MJCF 自动编译器")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--out", type=Path, default=Path("models/d51_full.xml"))
    parser.add_argument("--mode", choices=["roller", "track"], default="roller",
                        help="roller: 试运转滚轮台悬空模式; track: 45mm 轨道行驶模式")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    xml_str = build_mjcf(cfg, mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml_str, encoding="utf-8")
    print(f"[DONE] 已生成完整闭环模型: {args.out} (模式: {args.mode})")


if __name__ == "__main__":
    main()