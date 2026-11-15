#!/usr/bin/env python3
"""
scripts/02_build_cad_meshes.py
Aster Hobby JNR D51 (1/32, Gauge 1) 全套参数化水密 CAD 建模 + OpenCascade 材料密度积分惯量器
(P0-C/D + P2-D/E/F 全真圆弧弯管、三软管、左驾 11 项驾驶室机构、排液阀 8 级传动与煤水车过程版)。

核心规范执行 (GEMINI.md & LESSONS_LEARNED.md Gates 1-14)：
  1. 彻底废除全车管路中的直角方块 Box 弯头 (Gate 12)：统一使用 OpenCascade Wire+fillet+sweep 圆弧扫掠圆管
     (make_swept_tube, R=2.2~6.0mm) 与圆柱形中空バンジョー接头 (4-15) / 六角活接螺母 (34-8, 17-4) 建模；
     仅 7-8 排汽分流中央底座保留原厂铣削矩形铜块。
  2. 彻底废除为凑 cc==1 而捏造的虚假跨接桥杆 (Gate 11)：
     - slide_bar_L/R.stl 仅含 1-33 不锈钢单滑杆 + 1-30/2-13 支座 (移除气缸顶虚假 tie_rod)；
     - 独立建模 piston_tail_sheath_L/R.stl (20-6+20-2 活塞先导尻棒护套) 与 valve_spindle_sheath_L/R.stl (20-1 滑阀前导套)。
  3. 还原跨车 3 根弹性软管真值 (イラスト 34 & 手册 p.39, Gate 12)：
     - umb_hose_fuel(_r2m).stl: 牵引杆上方中央 ST-ø7 半透明硅胶供醇软管 (34-2 -> 23-1)；
     - umb_hose_feed(_r2m).stl + umb_union_34_8(_r2m).stl: 牵引杆下方斜跨 BT-ø5 黑色氟橡胶给水/手摇共用软管
       (煤水车右侧 -Y 34-6 -> 34-8 滚花黄铜活接 -> 机车左侧 +Y 6-11)；
     - umb_hose_return(_r2m).stl: 牵引杆下方斜跨 BT-ø5 黑色氟橡胶无压回水软管 (机车右侧 -Y 4-16 -> 煤水车左侧 +Y 31-14)。
  4. 还原日本国铁左驾 (左運転台 +Y) 逆转系、排液阀 8 级拉杆与驾驶室/煤水车全部可动操纵件 (Gate 14)：
     - reverser_shaft / cab_reach_rod / boiler_superstructure 25-7 护罩与穿墙槽全部校正至左侧 (+Y)；
     - 新增 drain_pull_rod.stl (8-6 拉柄 -> 8-5 右侧长拉杆 -> 7-1~3 摇臂) 与中心对称 drain_cock_linkage.stl 联动；
     - 新增 cab_reverser_stand, cab_reverser_wheel, cab_regulator_lever, cab_blower_wheel, cab_whistle_lever,
       cab_pressure_needle, cab_firebox_door, cab_bypass_lever 及 tender_handpump_lever, tender_water_lid,
       tender_fuel_valve, tender_fuel_cap。
  5. 火室底燃烧器按 イラスト 23 校正为纵向中央 1x3 直列三头立式 ø10mm 燃烧器 (23-1)。
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from build123d import (
    Box,
    Circle,
    Cone,
    Cylinder,
    Edge,
    Plane,
    PolarLocations,
    Pos,
    RegularPolygon,
    Rot,
    SlotOverall,
    Vector,
    Wire,
    export_stl,
    extrude,
    fillet,
    mirror,
    sweep,
)

# 原厂材质密度表 (kg/m³)
RHO_COPPER = 8940.0             # 无氧紫铜 C1100 (内火室、顶板、5根烟管、管板)
RHO_BRASS = 8500.0              # 黄铜 C3604/BS (气缸、滑阀、6-5 轴动泵、4-2 注油箱、阀件、轴箱、滑块、偏心环)
RHO_STAINLESS = 7930.0          # 不锈钢 SUS304 (1-33 滑杆、20-1/2 护套、17-5 双U过热器、连杆系、华氏五杆、逆转轴/拉杆)
RHO_CAST_IRON = 7350.0          # 铸铁/碳钢 FC/S45C (Boxpok 动轮、镂空配气桁架、均衡梁闸瓦、台车架、转向架)
RHO_COMPOSITE_VALVE = 8250.0    # 1-6 黄铜滑阀 + 1-22 不锈钢阀芯叉杆复合体
RHO_COMPOSITE_PISTON = 8200.0   # 1-28 黄铜活塞 + SUS 活塞杆 + 1-31 十字头 + 1-32 滑块夹板复合体
RHO_BURNER_ASSY = 8200.0        # 23-1 1x3 三头黄铜燃烧器 + 陶瓷纤维灯芯总成
RHO_SILICONE = 1180.0           # ST-ø7 半透明耐热硅胶软管
RHO_VITON = 1820.0              # BT-ø5 耐热黑色氟橡胶软管
RHO_PYREX_GLASS = 2230.0        # 11-7 耐热硼硅玻璃水面计管 (ø4.0mm)
RHO_GAUGE_DIAL = 2700.0         # PG-ø20 白釉刻度表盘片


def compute_occ_mass_props(solid: Any, density_kg_m3: float) -> Dict[str, Any]:
    """调用 OpenCascade BRepGProp 精确积分体积、偏心质心与 3x3 惯量张量 (建模 mm -> 物理 SI m/kg)。"""
    props = GProp_GProps()
    wrapped = solid.wrapped if hasattr(solid, "wrapped") else solid
    BRepGProp.VolumeProperties_s(wrapped, props)

    vol_mm3 = float(props.Mass())
    com = props.CentreOfMass()
    mat = props.MatrixOfInertia()

    mass_kg = vol_mm3 * 1e-9 * density_kg_m3
    scale_i = 1e-15 * density_kg_m3

    ixx = max(1e-9, float(mat.Value(1, 1)) * scale_i)
    iyy = max(1e-9, float(mat.Value(2, 2)) * scale_i)
    izz = max(1e-9, float(mat.Value(3, 3)) * scale_i)
    ixy = float(mat.Value(1, 2)) * scale_i
    ixz = float(mat.Value(1, 3)) * scale_i
    iyz = float(mat.Value(2, 3)) * scale_i

    return {
        "density_kg_m3": density_kg_m3,
        "volume_cm3": round(vol_mm3 * 1e-3, 4),
        "mass_kg": mass_kg,
        "com_m": [float(com.X()) * 1e-3, float(com.Y()) * 1e-3, float(com.Z()) * 1e-3],
        "fullinertia": [ixx, iyy, izz, ixy, ixz, iyz],
    }


def make_swept_tube(pts: Sequence[Tuple[float, float, float]], r_tube: float, bend_r: float):
    """沿任意 2D/3D 空间折线构造 G1 相切圆弧弯角 (Edge.make_three_point_arc) 并扫掠圆截面金属弯管，彻底取代失真的直角方块弯头！"""
    vecs = [Vector(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    n = len(vecs)
    if n < 2:
        raise ValueError("make_swept_tube requires at least 2 points")
    if n == 2:
        w = Wire([Edge.make_line(vecs[0], vecs[1])])
    else:
        corners = []
        for i in range(1, n - 1):
            seg_in = vecs[i - 1] - vecs[i]
            seg_out = vecs[i + 1] - vecs[i]
            L_in, L_out = seg_in.length, seg_out.length
            u, v = seg_in.normalized(), seg_out.normalized()
            cos_th = max(-1.0, min(1.0, u.dot(v)))
            theta = math.acos(cos_th)
            if theta > math.radians(177.0) or theta < math.radians(5.0):
                corners.append((vecs[i], vecs[i], None))
            else:
                alpha = 0.5 * theta
                d_max = 0.45 * min(L_in, L_out)
                r_i = min(max(r_tube + 0.35, bend_r), d_max * math.tan(alpha))
                r_i = max(r_tube + 0.12, r_i)
                d_i = r_i / math.tan(alpha)
                a_pt = vecs[i] + u * d_i
                b_pt = vecs[i] + v * d_i
                bis = (u + v).normalized()
                m_pt = vecs[i] + bis * (r_i * (1.0 / math.sin(alpha) - 1.0))
                corners.append((a_pt, b_pt, m_pt))

        edges = []
        cur_pt = vecs[0]
        for a_pt, b_pt, m_pt in corners:
            if (a_pt - cur_pt).length > 1e-4:
                edges.append(Edge.make_line(cur_pt, a_pt))
            if m_pt is not None:
                edges.append(Edge.make_three_point_arc(a_pt, m_pt, b_pt))
            cur_pt = b_pt
        if (vecs[-1] - cur_pt).length > 1e-4:
            edges.append(Edge.make_line(cur_pt, vecs[-1]))
        w = Wire(edges)

    prof = Plane(origin=w @ 0, z_dir=w % 0) * Circle(r_tube)
    return sweep(prof, path=w)


def make_spline_hose(
    pts: Sequence[Tuple[float, float, float]],
    t_start: Tuple[float, float, float],
    t_end: Tuple[float, float, float],
    r_hose: float,
):
    """构建自然悬垂或对角 S 型平滑样条弹性软管实体。"""
    vecs = [Vector(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    edge = Edge.make_spline(vecs, tangents=(Vector(*t_start), Vector(*t_end)))
    w = Wire([edge])
    prof = Plane(origin=w @ 0, z_dir=w % 0) * Circle(r_hose)
    return sweep(prof, path=w)


def make_hex_nut(
    pos: Tuple[float, float, float],
    axis_dir: Tuple[float, float, float],
    hex_r: float,
    length: float,
):
    """构建居中对齐的六角活接螺母 / 中空螺栓头 (17-4, 34-8, 4-15)。"""
    ax = Vector(*axis_dir).normalized()
    p0 = Vector(*pos) - ax * (0.5 * length)
    return extrude(Plane(origin=p0, z_dir=ax) * RegularPolygon(radius=hex_r, side_count=6), amount=length)


def make_banjo_eye(
    pos: Tuple[float, float, float],
    axis_dir: Tuple[float, float, float],
    r_eye: float,
    h_eye: float,
    hex_r: float,
):
    """构建原厂圆柱形バンジョー环眼 + 4-15 六角中空管螺栓头总成 (零方块！)。"""
    ax = Vector(*axis_dir).normalized()
    p_base = Vector(*pos) - ax * (0.5 * h_eye)
    eye = extrude(Plane(origin=p_base, z_dir=ax) * Circle(r_eye), amount=h_eye)
    bolt_pos = Vector(*pos) + ax * (0.5 * h_eye + 0.45)
    bolt = make_hex_nut((bolt_pos.X, bolt_pos.Y, bolt_pos.Z), axis_dir, hex_r=hex_r, length=1.2)
    return eye + bolt


def make_stepped_pin_L(
    x: float,
    z: float,
    y_nut_inner: float,
    y_head_outer: float,
    r_shank: float,
    r_head: float,
    r_nut: float,
    slot_w: float = 0.42,
):
    head_t = 0.75
    nut_t = 0.65
    y_span = (y_head_outer - 0.25) - (y_nut_inner + 0.25)
    y_mid = 0.5 * ((y_head_outer - 0.25) + (y_nut_inner + 0.25))
    shank = Pos(x, y_mid, z) * Rot(90, 0, 0) * Cylinder(radius=r_shank, height=max(0.8, y_span))
    head = Pos(x, y_head_outer - 0.5 * head_t, z) * Rot(90, 0, 0) * Cylinder(radius=r_head, height=head_t)
    slot_cut = Pos(x, y_head_outer + 0.05, z) * Box(2.0 * r_head + 0.8, 0.50, slot_w)
    nut = Pos(x, y_nut_inner + 0.5 * nut_t, z) * Rot(90, 0, 0) * Cylinder(radius=r_nut, height=nut_t)
    return shank + (head - slot_cut) + nut


def make_connecting_bar_xy(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    z_c: float,
    thick_y: float,
    height_z: float,
):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    yaw_deg = math.degrees(math.atan2(dy, dx))
    bar = Pos(0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), z_c) * Rot(0, 0, yaw_deg) * Box(L, thick_y, height_z)
    c0 = Pos(p0[0], p0[1], z_c) * Cylinder(radius=0.5 * thick_y, height=height_z)
    c1 = Pos(p1[0], p1[1], z_c) * Cylinder(radius=0.5 * thick_y, height=height_z)
    return bar + c0 + c1


# ==================== 1. 动轮系、主连杆、三段铰接耦合连杆与 7-10 回动曲柄 ====================

def build_boxpok_wheel(cfg: Dict, phase_deg: float, is_left: bool, is_main_driver: bool = False):
    d_tread = float(cfg["wheels_and_axles"]["driving_wheel_diameter"])
    r_tread = 0.5 * d_tread
    r_flange = r_tread + 1.00
    r_rim_inner = r_tread - 2.4
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    w_tire, w_flange, w_web = 3.8, 1.0, (2.8 if is_main_driver else 2.4)

    tire = Cylinder(radius=r_tread, height=w_tire)
    flange = Pos(0, 0, -0.5 * w_tire + 0.5 * w_flange) * Cylinder(radius=r_flange, height=w_flange)
    wheel = tire + flange - Cylinder(radius=r_rim_inner, height=w_tire + 2.0)

    web = Cylinder(radius=r_rim_inner + 0.3, height=w_web)
    hole_angles = [42, 75, 108, 252, 285, 318]
    r_hole_ring = 0.54 * (r_rim_inner + 4.5)
    for ang_deg in hole_angles:
        ang_rad = math.radians(ang_deg)
        hx, hy = r_hole_ring * math.cos(ang_rad), r_hole_ring * math.sin(ang_rad)
        web -= Pos(hx, hy, 0) * Cylinder(radius=2.1 if is_main_driver else 2.3, height=w_web + 2.0)
        hx2, hy2 = (r_rim_inner - 2.6) * math.cos(ang_rad), (r_rim_inner - 2.6) * math.sin(ang_rad)
        web -= Pos(hx2, hy2, 0) * Cylinder(radius=1.05 if is_main_driver else 1.15, height=w_web + 2.0)
    wheel += web

    cw_thick = w_tire if is_main_driver else (w_tire - 0.3)
    cw_cut_x = 9.2 if is_main_driver else 7.5
    cw_outer = Cylinder(radius=r_rim_inner + 0.25, height=cw_thick)
    cw_cutter_arc = Pos(cw_cut_x, 0, 0) * Cylinder(radius=r_rim_inner - 1.5, height=w_tire + 2.0)
    cw_half_mask = Pos(r_flange, 0, 0) * Box(2.0 * r_flange + 5.0, 2.0 * r_flange + 5.0, w_tire + 2.0)
    wheel += cw_outer - cw_cutter_arc - cw_half_mask

    boss_z = 0.5 * w_tire
    center_hub = Cylinder(radius=4.8 if is_main_driver else 4.6, height=w_tire + 0.6)
    pad_h = 7.8 if is_main_driver else 7.0
    crank_pad = Pos(0, 0, -0.5 * w_web) * extrude(
        Pos(0.5 * r_crank, 0) * SlotOverall(width=r_crank + 7.8, height=pad_h),
        amount=boss_z + 0.5 * w_web + 0.5,
    )
    if is_main_driver:
        pin_coup = Pos(r_crank, 0, 3.8) * Cylinder(radius=1.75, height=4.6)
        pin_main = Pos(r_crank, 0, 7.7) * Cylinder(radius=1.70, height=4.2)
        pin_ret = Pos(r_crank, 0, 10.5) * Cylinder(radius=1.40, height=2.0)
        crank_pin = pin_coup + pin_main + pin_ret
    else:
        crank_pin = Pos(r_crank, 0, boss_z + 1.70) * Cylinder(radius=1.72, height=3.8)
        cap = Pos(r_crank, 0, boss_z + 3.50) * Cylinder(radius=2.15, height=0.50)
        cap -= Pos(r_crank, 0, boss_z + 3.70) * Box(4.6, 0.40, 0.35)
        crank_pin += cap

    wheel = wheel + center_hub + crank_pad + crank_pin - Cylinder(radius=2.75, height=w_tire + 10.0)

    wheel_mj = Rot(-90, 0, 0) * wheel
    if not is_left:
        wheel_mj = mirror(wheel_mj, about=Plane.XZ)
    if abs(phase_deg) > 1e-6:
        wheel_mj = Rot(0, -phase_deg, 0) * wheel_mj
    return wheel_mj


def build_return_crank(cfg: Dict, is_left: bool = True):
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    r_ret = float(cfg["walschaerts_valve_gear"]["return_crank_radius"])
    ph_ret = math.radians(float(cfg["walschaerts_valve_gear"]["return_crank_phase_deg"]))
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    dx = r_ret * math.cos(ph_ret) - r_crank
    dz = r_ret * math.sin(ph_ret) - 0.0
    arm_len = math.hypot(dx, dz)
    pitch_deg = math.degrees(-math.atan2(dz, dx))

    y_web = -1.45
    clamp_hub = Pos(0, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=3.0, height=1.4)
    main_pin_cap = Pos(0, y_web + 0.70, 0) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=0.45)
    pinch_lug = Pos(-2.4, y_web, 1.35) * Box(3.0, 1.4, 2.7)
    pinch_bolt = Pos(-2.8, y_web, 2.7) * Cylinder(radius=0.65, height=1.2)
    pinch_slit = Pos(-3.1, y_web, 0.0) * Box(3.0, 2.2, 0.45)
    web = Pos(0.5 * arm_len, y_web, 0) * Box(arm_len, 1.2, 3.4)
    ecc_hub = Pos(arm_len, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=1.4)

    ecc_pin = Pos(arm_len, 0.35, 0) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=2.4)
    ecc_cap = Pos(arm_len, 1.55, 0) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=0.55)
    ecc_cap -= Pos(arm_len, 1.76, 0) * Box(4.0, 0.36, 0.40)

    crank = clamp_hub + main_pin_cap + pinch_lug + pinch_bolt + web + ecc_hub + ecc_pin + ecc_cap - pinch_slit
    crank -= Pos(0, y_web - 0.6, 0) * Rot(90, 0, 0) * Cylinder(radius=1.42, height=1.8)
    crank_L = Rot(0, pitch_deg, 0) * crank

    if is_left:
        return crank_L
    return Rot(0, -q_deg, 0) * mirror(crank_L, about=Plane.XZ)


def build_main_rod(cfg: Dict):
    L_main = float(cfg["cylinder_and_rods"]["main_rod_length"])
    d_small = float(cfg["cylinder_and_rods"]["crosshead_pin_dia"])
    thickness_y, big_boss_y, small_boss_y = 1.38, 1.96, 1.80
    h_big, h_small, h_shank = 6.2, 4.5, 3.1

    big_end = Box(8.0, big_boss_y, h_big)
    big_oil_cup = Pos(0, 0, 0.5 * h_big + 0.5) * Cylinder(radius=1.15, height=1.4)
    small_end = Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * h_small, height=small_boss_y)
    shank = Pos(0.5 * L_main, 0, 0) * Box(L_main - 1.5, thickness_y, h_shank)
    rod = big_end + big_oil_cup + small_end + shank

    flute_len = L_main - 14.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * L_main, sgn * (0.5 * thickness_y - 0.16), 0) * Box(flute_len, 0.36, h_shank - 1.2)

    rod -= Rot(90, 0, 0) * Cylinder(radius=1.72, height=big_boss_y + 4.0)
    rod -= Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * d_small + 0.02, height=small_boss_y + 4.0)
    return rod


def build_coupling_rod_main(cfg: Dict):
    w_cfg = cfg["wheels_and_axles"]
    x_d2 = float(w_cfg["axle_x"]["drive_2"])
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    x_kr, x_kf = -k_off, x_d2 + k_off
    shank_t, boss_t, shank_h = 1.80, 2.20, 3.3

    rod = Pos(0.5 * x_d2, 0, 0) * Box(x_d2, shank_t, shank_h)
    for x_pos, r_b in [(0.0, 3.9), (x_d2, 3.3)]:
        rod += Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=r_b, height=boss_t)
        rod += Pos(x_pos, 0, r_b + 0.4) * Cylinder(radius=1.0, height=1.3)

    tongue_t = 1.05
    rod += Pos(0.5 * x_kr, 0, 0) * Box(abs(x_kr) + 1.0, shank_t, shank_h - 0.2)
    rod += Pos(x_kr, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=tongue_t)
    rod += Pos(x_d2 + 0.5 * k_off, 0, 0) * Box(k_off + 1.0, shank_t, shank_h - 0.2)
    rod += Pos(x_kf, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=tongue_t)

    flute_len = x_d2 - 12.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * x_d2, sgn * (0.5 * shank_t - 0.20), 0) * Box(flute_len, 0.45, shank_h - 1.2)

    for x_pos in (0.0, x_d2):
        rod -= Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.76, height=boss_t + 4.0)
    for x_k in (x_kr, x_kf):
        rod -= Pos(x_k, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=boss_t + 4.0)
    return rod


def build_coupling_rod_end(cfg: Dict, is_front: bool = True):
    w_cfg = cfg["wheels_and_axles"]
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    dx = (float(w_cfg["axle_x"]["drive_1"]) - (float(w_cfg["axle_x"]["drive_2"]) + k_off)) if is_front else (
        float(w_cfg["axle_x"]["drive_4"]) - (-k_off)
    )
    L_seg = abs(dx)
    sgn_x = 1.0 if dx >= 0 else -1.0
    shank_t, boss_t, shank_h, boss_r = 1.80, 2.20, 3.2, 3.3

    clevis = Rot(90, 0, 0) * Cylinder(radius=2.2, height=boss_t) + Pos(sgn_x * 1.8, 0, 0) * Box(3.6, boss_t, 4.0)
    clevis -= Pos(-sgn_x * 0.8, 0, 0) * Box(4.6, 1.25, 6.0)
    knuckle_pin = Rot(90, 0, 0) * Cylinder(radius=0.93, height=boss_t + 0.24) + (
        Rot(90, 0, 0) * Cylinder(radius=1.35, height=boss_t + 0.24)
        - Rot(90, 0, 0) * Cylinder(radius=1.40, height=boss_t - 0.16)
    )

    shank = Pos(0.5 * dx, 0, 0) * Box(L_seg, shank_t, shank_h)
    wheel_boss = Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=boss_r, height=boss_t)
    oil_cup = Pos(dx, 0, boss_r + 0.4) * Cylinder(radius=0.95, height=1.2)
    rod = clevis + knuckle_pin + shank + wheel_boss + oil_cup

    flute_len = L_seg - 10.5
    if flute_len > 6.0:
        for sy in (-1.0, 1.0):
            rod -= Pos(0.5 * dx, sy * (0.5 * shank_t - 0.20), 0) * Box(flute_len, 0.45, shank_h - 1.2)

    rod -= Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.74, height=boss_t + 4.0)
    return rod


# ==================== 2. 华氏五杆、单剪叠搭与全套台阶销栓 ====================

def build_eccentric_rod(cfg: Dict, is_left: bool = True):
    vg = cfg["walschaerts_valve_gear"]
    L_ecc = float(vg["eccentric_rod_length"])
    y_rear = float(vg["y_layer_offsets"]["eccentric_rod_rear_dy"])
    y_front = float(vg["y_layer_offsets"]["eccentric_rod_front_lap_dy"])

    rear_boss = (
        Pos(0, y_rear, 0) * Rot(90, 0, 0) * Cylinder(radius=2.20, height=1.35)
        + Pos(0, y_rear, 2.30) * Cylinder(radius=0.75, height=0.85)
    )
    front_flat_eye = Pos(L_ecc, y_front, 0) * Rot(90, 0, 0) * Cylinder(radius=2.00, height=1.15)
    shank_rear = make_connecting_bar_xy((1.2, y_rear), (21.5, y_rear), 0.0, thick_y=1.05, height_z=2.30)
    shank_front = make_connecting_bar_xy((21.5, y_rear), (L_ecc - 1.0, y_front), 0.0, thick_y=1.05, height_z=2.30)

    rod_L = rear_boss + shank_rear + shank_front + front_flat_eye
    rod_L -= Pos(0, y_rear, 0) * Rot(90, 0, 0) * Cylinder(radius=1.18, height=5.0)
    rod_L -= Pos(L_ecc, y_front, 0) * Rot(90, 0, 0) * Cylinder(radius=1.02, height=5.0)
    return rod_L if is_left else mirror(rod_L, about=Plane.XZ)


def build_expansion_link(cfg: Dict, is_left: bool = True):
    vg = cfg["walschaerts_valve_gear"]
    R_arc = float(vg["radius_rod_length"])
    r_tail = float(vg["exp_link_tail_radius"])
    cant_rad = math.radians(float(vg["exp_link_tail_cant_deg"]))
    slot_half = float(vg["die_slot_half_travel"])

    thick_y = 1.90
    slot_w = 1.8
    body_w = 3.8
    z_top = slot_half + 1.9
    z_bot = -(slot_half + 1.9)

    outer_ring = Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc + 0.5 * body_w, height=thick_y
    ) - Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc - 0.5 * body_w, height=thick_y + 2.0
    )
    crescent_body = outer_ring & Box(18.0, thick_y + 2.0, z_top - z_bot)

    x_cap_top = R_arc - math.sqrt(max(0.0, R_arc**2 - z_top**2))
    x_cap_bot = R_arc - math.sqrt(max(0.0, R_arc**2 - z_bot**2))
    top_cap = Pos(x_cap_top, 0, z_top) * Rot(90, 0, 0) * Cylinder(radius=0.5 * body_w, height=thick_y)
    bot_cap = Pos(x_cap_bot, 0, z_bot) * Rot(90, 0, 0) * Cylinder(radius=0.5 * body_w, height=thick_y)
    crescent_body = crescent_body + top_cap + bot_cap

    slot_ring = Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc + 0.5 * slot_w, height=thick_y + 2.0
    ) - Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc - 0.5 * slot_w, height=thick_y + 4.0
    )
    z_slot_end = slot_half + 0.8
    x_slot_end = R_arc - math.sqrt(max(0.0, R_arc**2 - z_slot_end**2))
    slot_cut = (slot_ring & Box(18.0, thick_y + 4.0, 2.0 * z_slot_end)) + (
        Pos(x_slot_end, 0, z_slot_end) * Rot(90, 0, 0) * Cylinder(radius=0.5 * slot_w, height=thick_y + 2.0)
    ) + (
        Pos(x_slot_end, 0, -z_slot_end) * Rot(90, 0, 0) * Cylinder(radius=0.5 * slot_w, height=thick_y + 2.0)
    )
    crescent_body -= slot_cut

    x_tail = r_tail * math.sin(cant_rad)
    z_tail = -r_tail * math.cos(cant_rad)
    y_ear = float(vg["y_layer_offsets"]["expansion_link_tail_ear_dy"])
    ear_t = 1.40
    dx_n = x_tail - x_cap_bot
    dz_n = z_tail - (z_bot + 0.4)
    neck_len = math.hypot(dx_n, dz_n)
    neck_pitch = math.degrees(math.atan2(-dx_n, -dz_n))
    tail_neck = (
        Pos(0.5 * (x_cap_bot + x_tail), y_ear, 0.5 * (z_bot + 0.4 + z_tail))
        * Rot(0, neck_pitch, 0)
        * Box(2.9, ear_t, neck_len + 1.0)
    )
    tail_boss = Pos(x_tail, y_ear, z_tail) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=ear_t)
    pin_2_2 = make_stepped_pin_L(
        x=x_tail,
        z=z_tail,
        y_nut_inner=float(vg["y_layer_offsets"]["stepped_pin_2_2_nut_dy"]),
        y_head_outer=float(vg["y_layer_offsets"]["stepped_pin_2_2_head_dy"]),
        r_shank=1.00,
        r_head=1.80,
        r_nut=1.50,
    )

    standoff_f = Pos(+1.42, 1.45, 0) * Box(0.95, 1.40, 4.4)
    standoff_r = Pos(-1.42, 1.45, 0) * Box(0.95, 1.40, 4.4)
    trun_bridge = Pos(0.0, 2.50, 0) * Box(3.8, 0.95, 4.6)
    trun_pin = Pos(0.0, 3.95, 0) * Rot(90, 0, 0) * Cylinder(radius=1.72, height=2.2)
    trun_cap = Pos(0.0, 5.10, 0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=0.55)

    link_L = crescent_body + tail_neck + tail_boss + pin_2_2 + standoff_f + standoff_r + trun_bridge + trun_pin + trun_cap
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


def build_die_block(cfg: Dict, is_left: bool = True):
    block = Box(3.6, 1.85, 3.1)
    pin_2_21 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-2.85, y_head_outer=+1.68, r_shank=0.85, r_head=1.38, r_nut=1.30, slot_w=0.36
    )
    part_L = block + pin_2_21
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_radius_rod(cfg: Dict, is_left: bool = True):
    vg = cfg["walschaerts_valve_gear"]
    L_rad = float(vg["radius_rod_length"])
    L_back = float(vg["radius_rod_back_ext"])
    y_bar = float(vg["y_layer_offsets"]["radius_rod_bar_dy"])
    t_bar = 0.96

    back_eye = Pos(-L_back, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=t_bar)
    die_hub = Pos(0.0, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=1.90, height=t_bar)
    rear_span = Pos(-0.5 * L_back, y_bar, 0) * Box(L_back, t_bar, 2.15)
    mid_span = Pos(23.5, y_bar, 0) * Box(47.0, t_bar, 2.35)
    trans_span = make_connecting_bar_xy((46.5, y_bar), (L_rad - 2.0, 0.0), 0.0, thick_y=1.05, height_z=2.35)

    front_clevis = (
        Pos(L_rad, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.12, height=3.3)
        + Pos(L_rad - 1.2, 0, 0) * Box(2.4, 3.3, 2.16)
        - Pos(L_rad + 0.5, 0, 0) * Box(3.8, 1.72, 5.0)
    )
    pin_1_21 = make_stepped_pin_L(
        x=L_rad, z=0.0, y_nut_inner=-2.25, y_head_outer=+2.25, r_shank=0.75, r_head=1.04, r_nut=1.00
    )

    rod_L = back_eye + die_hub + rear_span + mid_span + trans_span + front_clevis + pin_1_21
    rod_L -= Pos(0.0, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=0.87, height=4.0)
    rod_L -= Pos(-L_back, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=0.87, height=4.0)
    return rod_L if is_left else mirror(rod_L, about=Plane.XZ)


def build_combination_lever(cfg: Dict):
    vg = cfg["walschaerts_valve_gear"]
    L_comb = float(vg["comb_lever_total_length"])
    L_mv = float(vg["comb_lever_lap_lead_offset"])
    z_mid = L_comb - L_mv
    t_web = 1.50

    bot_eye = Rot(90, 0, 0) * Cylinder(radius=1.80, height=t_web)
    lower_shank = Pos(0, 0, 0.5 * z_mid) * Box(2.25, t_web, z_mid)
    mid_boss = Pos(0, 0, z_mid) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=t_web)
    upper_neck = Pos(0, 0, z_mid + 0.5 * L_mv) * Box(2.05, t_web, L_mv)
    top_boss = Pos(0, 0, L_comb) * Rot(90, 0, 0) * Cylinder(radius=1.30, height=t_web)

    lever = bot_eye + lower_shank + mid_boss + upper_neck + top_boss
    lever -= Rot(90, 0, 0) * Cylinder(radius=0.85, height=5.0)
    for zp in (z_mid, L_comb):
        lever -= Pos(0, 0, zp) * Rot(90, 0, 0) * Cylinder(radius=0.76, height=5.0)
    return lever


def build_union_link(cfg: Dict, is_left: bool = True):
    L_un = float(cfg["walschaerts_valve_gear"]["union_link_length"])
    clevis_0 = (
        Rot(90, 0, 0) * Cylinder(radius=1.85, height=3.5)
        + Pos(1.2, 0, 0) * Box(2.4, 3.5, 3.2)
        - Pos(-0.6, 0, 0) * Box(3.8, 2.05, 4.5)
    )
    clevis_1 = (
        Pos(L_un, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.68, height=3.4)
        + Pos(L_un - 1.2, 0, 0) * Box(2.4, 3.4, 3.0)
        - Pos(L_un + 0.6, 0, 0) * Box(3.8, 1.70, 4.5)
    )
    bar = Pos(0.5 * L_un, 0, 0) * Box(L_un - 1.6, 1.45, 2.2)
    pin_0 = make_stepped_pin_L(0.0, 0.0, -2.40, +2.40, r_shank=0.88, r_head=1.55, r_nut=1.40)
    pin_1 = make_stepped_pin_L(L_un, 0.0, -2.35, +2.35, r_shank=0.88, r_head=1.50, r_nut=1.25)
    link_L = clevis_0 + clevis_1 + bar + pin_0 + pin_1
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


def build_lift_link(cfg: Dict, is_left: bool = True):
    vg = cfg["walschaerts_valve_gear"]
    L_lift = float(vg["lift_link_length"])
    y_bot = float(vg["y_layer_offsets"]["lift_link_bot_clevis_dy"])

    top_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.80, height=3.3)
        + Pos(0, 0, -1.1) * Box(3.0, 3.3, 2.2)
        - Pos(0, 0, 0.6) * Box(4.0, 1.95, 3.4)
    )
    bot_clevis = (
        Pos(0, y_bot, -L_lift) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=2.6)
        + Pos(0, y_bot, -L_lift + 1.1) * Box(2.9, 2.6, 2.2)
        - Pos(0, y_bot, -L_lift - 0.6) * Box(4.0, 1.25, 3.4)
    )
    ang_x = math.degrees(math.atan2(y_bot, L_lift - 2.2))
    strap = Pos(0, 0.5 * y_bot, -0.5 * L_lift) * Rot(ang_x, 0, 0) * Box(2.1, 1.25, math.hypot(L_lift - 1.6, y_bot))

    pin_top = make_stepped_pin_L(0.0, 0.0, -2.30, +2.30, r_shank=0.85, r_head=1.50, r_nut=1.35)
    pin_bot = make_stepped_pin_L(0.0, -L_lift, y_bot - 1.95, y_bot + 1.95, r_shank=0.85, r_head=1.45, r_nut=1.30)
    link_L = top_clevis + bot_clevis + strap + pin_top + pin_bot
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


# ==================== 3. 配气桁架、去假桥独立滑杆与前盖护套、左驾操纵与排液阀 8 级传动 ====================

def build_motion_bracket(cfg: Dict, is_left: bool = True):
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    tx, tz = float(vg["trunnion_x"]), float(vg["trunnion_z"])
    y_vg = float(rc["cylinder_center_y"]) + 3.5

    dy_chassis = 14.5 - y_vg
    dy_slidebar = float(rc["cylinder_center_y"]) - y_vg
    dx_bar_tail = 69.0 - tx
    dz_bar_top = 8.4 - tz

    frame_pad = Pos(0, dy_chassis + 0.8, 11.2) * Box(9.5, 2.0, 8.2)

    y_truss_lo, y_truss_hi = dy_chassis + 0.5, +5.0
    truss_span_y = y_truss_hi - y_truss_lo
    truss_ym = 0.5 * (y_truss_lo + y_truss_hi)
    truss_zm = 12.4
    truss_h = 5.2

    plate_f = Pos(+2.2, truss_ym, truss_zm) * Box(1.3, truss_span_y, truss_h)
    plate_r = Pos(-2.2, truss_ym, truss_zm) * Box(1.3, truss_span_y, truss_h)
    top_chord = Pos(0.0, truss_ym, truss_zm + 0.5 * truss_h - 0.50) * Box(5.7, truss_span_y, 1.0)
    bot_chord = Pos(0.0, truss_ym, truss_zm - 0.5 * truss_h + 0.50) * Box(5.7, truss_span_y, 1.0)
    truss = plate_f + plate_r + top_chord + bot_chord

    for yh in (-17.2, -12.0, -6.8, -1.6):
        truss -= Pos(0.0, yh, truss_zm) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=9.0)

    outer_leg = Pos(0.0, +3.95, 6.2) * Box(5.0, 1.85, 12.4)
    bearing_housing = Pos(0.0, +3.95, 0.0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=1.85)
    bearing_cap_lug = Pos(0.0, +3.95, -2.2) * Box(6.6, 1.85, 1.8)
    cap_bolt_f = Pos(+2.5, +3.95, -2.5) * Cylinder(radius=0.55, height=2.2)
    cap_bolt_r = Pos(-2.5, +3.95, -2.5) * Cylinder(radius=0.55, height=2.2)

    arm_horiz = Pos(0.5 * dx_bar_tail, dy_slidebar - 0.55, 11.4) * Box(dx_bar_tail + 2.4, 1.6, 2.2)
    z_drop_lo = dz_bar_top + 0.5
    arm_drop = Pos(dx_bar_tail, dy_slidebar - 0.55, 0.5 * (11.4 + z_drop_lo)) * Box(2.4, 1.6, 11.4 - z_drop_lo + 1.2)
    guide_clamp = Pos(dx_bar_tail, dy_slidebar, z_drop_lo) * Box(4.2, 2.8, 1.5)

    bracket_L = (
        frame_pad + truss + outer_leg + bearing_housing + bearing_cap_lug + cap_bolt_f + cap_bolt_r
        + arm_horiz + arm_drop + guide_clamp
        - Pos(0.0, +3.95, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.78, height=3.2)
    )
    return bracket_L if is_left else mirror(bracket_L, about=Plane.XZ)


def build_slide_bar(cfg: Dict, is_left: bool = True):
    """
    Gate 11 修复版 1-33 上置亮面不锈钢单滑杆 + 1-30 前支座 + 2-13 后吊耳座 (SUS304):
      - 彻底删除曾跨越气缸顶部的虚假 tie_rod 桥杆！
      - 前盖活塞先导护套 (20-2) 与滑阀导套 (20-1) 已拆分为真实独立零件网格。
    """
    x_bar_f, x_bar_r = -16.8, -58.0
    bar_len = x_bar_f - x_bar_r
    bar = Pos(0.5 * (x_bar_f + x_bar_r), 0.0, 6.5) * Box(bar_len, 2.8, 1.7)
    front_lug = Pos(-16.8, 0.0, 5.5) * Box(2.2, 3.2, 3.8)
    rear_lug = Pos(-57.0, 0.0, 7.6) * Box(3.6, 2.8, 1.2) + Pos(-57.0, 0.0, 8.5) * Cylinder(radius=0.85, height=1.2)
    part_L = bar + front_lug + rear_lug
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_tail_sheath(cfg: Dict, is_left: bool = True):
    """20-6 / 20-2 / 20-8 气缸前盖活塞先导尻棒护套 (Tail-Rod Sheath, 直接嵌合气缸前盖 x_local=+17.2mm, cc=1)。"""
    stroke = 2.0 * float(cfg["cylinder_and_rods"]["crank_radius"])
    cyl_len = stroke + 12.0  # 32.0 mm -> front_cover front face at +17.3mm
    flange_base = Pos(0.5 * cyl_len + 1.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=1.8)
    sheath_tube = Pos(0.5 * cyl_len + 6.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.85, height=10.2)
    tip_cone = Pos(0.5 * cyl_len + 12.4, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=1.85, top_radius=1.1, height=1.8)
    part_L = flange_base + sheath_tube + tip_cone
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_valve_spindle_sheath(cfg: Dict, is_left: bool = True):
    """20-1 蒸汽室前盖阀芯先导护套 (Valve-Spindle Front Guide Sheath, 直接嵌合蒸汽室前壁 x_local=+15.4mm, cc=1)。"""
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    flange = Pos(+15.6, 0.0, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.6)
    tube = Pos(+19.8, 0.0, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.40, height=7.4)
    tip = Pos(+24.0, 0.0, z_valve) * Rot(0, 90, 0) * Cone(bottom_radius=1.40, top_radius=0.90, height=1.4)
    part_L = flange + tube + tip
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_crosshead(cfg: Dict, is_left: bool = True):
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    bore = float(rc["bore"])
    L_main = float(rc["main_rod_length"])
    cyl_cx = float(rc["cylinder_center_x"])
    drop_z = float(vg["crosshead_drop"])
    p_head_dx = (cyl_cx + 4.2) - L_main

    piston_disc = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore - 0.1, height=3.2)
    ring_groove = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * (
        Cylinder(radius=0.5 * bore + 0.5, height=1.1) - Cylinder(radius=0.5 * bore - 0.9, height=1.6)
    )
    piston_disc -= ring_groove
    piston_rod = Pos(0.5 * p_head_dx + 1.2, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.5, height=p_head_dx - 1.2)
    # 活塞前伸尻棒先导端 (伸入 20-2 护套内)
    tail_rod = Pos(p_head_dx + 5.8, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=9.2)

    ch_hub = Pos(-1.4, 0, 0) * Box(6.4, 3.60, 6.8) - Pos(-2.2, 0, 0) * Box(6.4, 2.05, 7.6)
    shoe_outer = Pos(-2.6, +0.20, 6.45) * Box(8.4, 4.00, 5.0)
    bar_slot = Pos(-2.6, 0.0, 6.40) * Box(11.0, 3.08, 3.1)
    under_open = Pos(-2.6, 0.0, 4.30) * Box(11.0, 2.0, 1.6)
    top_shoe = shoe_outer - bar_slot - under_open
    neck = Pos(-1.4, 0.0, 3.6) * Box(5.4, 3.40, 1.8)

    drop_root = Pos(0, +0.90, -2.6) * Box(3.4, 1.80, 2.4)
    drop_step = Pos(0, +2.20, -4.4) * Rot(+38, 0, 0) * Box(3.4, 1.40, 3.8)
    drop_tongue = Pos(0, +3.50, -7.4) * Box(3.4, 1.50, 4.6)
    drop_boss = Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=1.50)
    pin_7_13 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-1.80, y_head_outer=+2.45, r_shank=1.48, r_head=2.15, r_nut=1.75, slot_w=0.45
    )

    part_L = (
        piston_disc + piston_rod + tail_rod + ch_hub + neck + top_shoe
        + drop_root + drop_step + drop_tongue + drop_boss + pin_7_13
    )
    part_L -= Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=0.90, height=6.0)
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_slide_valve_assembly(cfg: Dict, is_left: bool = True):
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    v_dx = float(rc["cylinder_center_x"]) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))

    clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.12, height=3.3)
        + Pos(1.3, 0, 0) * Box(2.6, 3.3, 2.16)
        - Pos(-0.7, 0, 0) * Box(3.4, 1.72, 6.0)
    )
    pin_1_24 = make_stepped_pin_L(0.0, 0.0, -2.25, +2.25, r_shank=0.75, r_head=1.04, r_nut=1.00)

    rod_len = v_dx + 20.0
    stem = Pos(0.5 * rod_len - 2.0, -3.5, 0) * Rot(0, 90, 0) * Cylinder(radius=1.0, height=rod_len)
    cross_arm = Pos(1.5, -1.75, 0) * Box(2.1, 5.5, 2.16)

    valve_block = Pos(v_dx, -3.5, -0.5) * Box(10.2, 9.6, 5.2)
    exhaust_cavity = Pos(v_dx, -3.5, -2.2) * Box(4.8, 6.6, 2.4)
    valve_block -= exhaust_cavity

    assembly_L = clevis + pin_1_24 + cross_arm + stem + valve_block
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_reverser_shaft(cfg: Dict):
    """
    4-12 逆转器横轴 + 2-11/2-18 左右等长 11.0mm 提升臂 + 左走台外侧 (+Y=+33.5mm) 向上逆转摇臂 (Gate 14 日本国铁左驾校正!):
      - 左侧操纵臂置于 y=+33.5mm, 向上高 7.0mm (接左侧 8-18 逆转长拉杆，藏于左走台 25-7 护罩内)。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5
    arm_len_x = float(vg["reverser_arm_length"])

    main_bar = Rot(90, 0, 0) * Cylinder(radius=1.4, height=2.0 * y_vg + 3.0)
    reach_hub = Pos(0, +33.5, 0) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=2.2)
    reach_lever = Pos(0, +33.5, 3.5) * Box(3.0, 1.8, 7.0)
    reach_tip = Pos(0, +33.5, 7.0) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=1.8)

    arms = None
    for sgn in (-1.0, 1.0):
        hub = Pos(0, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=2.2)
        arm = Pos(0.5 * arm_len_x, sgn * y_vg, 0) * Box(arm_len_x, 1.75, 3.0)
        tip_boss = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=1.9, height=1.75)
        tip_hole = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=5.0)
        sub = hub + arm + tip_boss - tip_hole
        arms = sub if arms is None else (arms + sub)

    return main_bar + reach_hub + reach_lever + reach_tip + arms


def build_cab_reach_rod(cfg: Dict):
    """
    8-18 左侧 (+Y=+33.5mm) 逆转长拉杆 + 8-20 前叉接头 + 8-17 螺杆滑块尾耳 (Gate 14 左驾校正!):
      - 局部原点 (0,0,0) 位于 reverser_shaft 左操纵臂顶点 (X=25.5, Y=+33.5, Z=33.5 mm)；
      - 沿左走台外侧向后直通驾驶室左前侧 8-12~17 螺旋逆转机座 (-148.5mm -> X=-123.0mm)。
    """
    L_rod = 148.5
    front_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.9, height=3.6)
        + Pos(-1.6, 0, 0) * Box(3.2, 3.6, 3.4)
        - Pos(+0.6, 0, 0) * Box(3.8, 1.95, 4.6)
    )
    pin_L = make_stepped_pin_L(0.0, 0.0, -2.35, +2.35, r_shank=0.85, r_head=1.45, r_nut=1.35)
    shaft = Pos(-0.5 * L_rod, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=L_rod)
    guide_collar = Pos(-72.0, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=2.6)
    # 尾端向内 (-Y) 偏置衔接驾驶室左侧 8-17 丝杠螺母滑块 (Y=+31.5mm)
    rear_nut_8_17 = (
        Pos(-L_rod, -1.0, 0.0) * Box(5.2, 3.2, 3.2)
        + Pos(-L_rod - 1.5, -2.0, 1.9) * Box(4.2, 2.8, 5.0)
        + Pos(-L_rod - 1.5, -2.0, 3.8) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=4.4)
    )
    return front_clevis + pin_L + shaft + guide_collar + rear_nut_8_17


def build_drain_cock_linkage(cfg: Dict):
    """
    7-5 排液凸轮横轴 + 7-4/7-6 凸轮 + 1-19 x4 剪式摇板 + 1-18 x4 顶针 + 1-16 x4 气缸底排液嘴 (100% Y 对称, イラスト 1 & 7):
      - 局部原点 (0,0,0) 对准 (X=129.0, Y=0.0, Z=-14.2 mm)；
      - 右侧内端 (y=-18.6mm) 伸出 7-1~3 驱动曲柄耳衔接纵向长拉杆 8-5，左侧镜像配重耳保持严格 |min_y+max_y|=0 & com_y=0！
    """
    y_cock = 30.8
    cross_shaft = Pos(0.0, 0.0, -3.6) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=43.6)
    sides = None
    for sgn in (-1.0, 1.0):
        yc = sgn * y_cock
        outer_shaft = Pos(0.0, sgn * 26.35, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.05, height=10.7)
        crank_cheek = Pos(0.0, sgn * 21.3, -1.8) * Box(2.1, 1.5, 5.6)
        horn_7_1 = (
            Pos(0.0, sgn * 18.6, -1.2) * Box(2.0, 1.4, 5.8)
            + Pos(0.0, sgn * 18.6, 1.6) * Rot(90, 0, 0) * Cylinder(radius=1.5, height=1.6)
        )
        cam_7_4 = Pos(0.0, yc, 0.8) * Box(2.6, 1.9, 2.6)
        scissor_1_19 = Pos(0.0, yc, 1.3) * Box(12.4, 1.5, 1.5)
        plunger_bar = Pos(-4.2, sgn * (y_cock - 2.6), 0.0) * Box(8.4, 1.4, 1.5) + Pos(
            0.0, sgn * (y_cock - 1.3), 0.0
        ) * Box(2.2, 2.6, 1.8)
        cock_rear = (
            Pos(-5.0, yc, 1.6) * Cylinder(radius=1.75, height=2.8)
            + Pos(-5.0, yc, 0.6) * Cylinder(radius=2.30, height=1.0)
            + Pos(-5.0, yc, -0.9) * Cylinder(radius=0.85, height=3.0)
        )
        cock_front = (
            Pos(+5.0, yc, 1.6) * Cylinder(radius=1.75, height=2.8)
            + Pos(+5.0, yc, 0.6) * Cylinder(radius=2.30, height=1.0)
            + Pos(+5.0, yc, -0.9) * Cylinder(radius=0.85, height=3.0)
        )
        sub = outer_shaft + crank_cheek + horn_7_1 + cam_7_4 + scissor_1_19 + plunger_bar + cock_rear + cock_front
        sides = sub if sides is None else (sides + sub)
    return cross_shaft + sides


def build_drain_pull_rod(cfg: Dict):
    """
    8-6 驾驶室右前地板竖向摇柄 + 8-10 地板支座槽配合 + 8-5 右侧全长排液拉杆 (イラスト 8):
      - 挂载于 loco_frame 原点 (0,0,0) 的滑动刚体 (j_drain_pull 沿 X 平移，与 j_drain_cock 耦合)；
      - 沿右侧无干涉走廊 Y=-18.6mm, Z=+7.8mm 贯通，后端在车底 (-119.5, -18.6, +14.8) 向上竖起 8-6 驾驶室操纵立柄穿过 8-11 地板槽至 Z=+29.8mm！
    """
    pts = [
        (-121.0, -18.6, 14.8),
        (-116.0, -18.6, 14.8),
        (-114.0, -18.6, 7.8),
        (+123.5, -18.6, 7.8),
        (+125.5, -18.6, -12.6),
        (+129.0, -18.6, -12.6),
    ]
    rod = make_swept_tube(pts, r_tube=0.72, bend_r=2.0)
    lever_8_6 = (
        Pos(-119.5, -18.6, 22.0) * Box(1.8, 1.3, 15.2)
        + Pos(-119.5, -18.6, 29.8) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.6)
    )
    front_eye = Pos(+129.0, -18.6, -12.6) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=2.2)
    return rod + lever_8_6 + front_eye


# ==================== 3B. 驾驶室 1:1 原厂全功能内构与煤水车全功能可动阀件网格 ====================

def build_cab_backhead_plate():
    """
    11-9 ボイラバックプレート (哑光黑铸铁锅炉后封头背板) + 12-5 x3 压板耳座 + 12-2 双窥火圆孔护板 (100% Y 对称, cc=1):
      - 紧贴紫铜锅筒后端 X=-112.0mm，下底落座于 8-11 驾驶室地板 Z=+20.0mm，彻底消除驾驶室仪表悬空！
      - 下中 12-2 护板在 Z=+38.0mm, Y=±2.5mm 贯穿双 ø2.9mm 圆形观火孔，直通火室内部 23-1 火焰。
    """
    lower_skirt = Pos(-113.2, 0.0, 32.3) * Box(2.4, 50.4, 25.4)
    upper_arch = Pos(-113.2, 0.0, 45.0) * Rot(0, 90, 0) * Cylinder(radius=25.2, height=2.4)
    clamp_top = Pos(-113.6, 0.0, 68.8) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=3.2)
    clamp_l = Pos(-113.6, +23.2, 28.0) * Rot(0, 90, 0) * Cylinder(radius=2.6, height=3.2)
    clamp_r = Pos(-113.6, -23.2, 28.0) * Rot(0, 90, 0) * Cylinder(radius=2.6, height=3.2)
    shield_12_2 = (
        Pos(-114.4, 0.0, 36.5) * Box(1.8, 20.4, 17.6)
        + Pos(-115.0, +6.2, 38.0) * Cylinder(radius=1.4, height=8.2)
        + Pos(-115.0, -6.2, 38.0) * Cylinder(radius=1.4, height=8.2)
    )
    peep_l = Pos(-113.5, +2.5, 38.0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=8.0)
    peep_r = Pos(-113.5, -2.5, 38.0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=8.0)
    return (lower_skirt + upper_arch + clamp_top + clamp_l + clamp_r + shield_12_2) - peep_l - peep_r


def build_cab_deck_and_mounts():
    """
    8-11 運転室床 + 8-21 後部渡り板 + 4-6/6-4 車架支座 + 8-10 右前ドレン支座 + 4-8 右後逆止弁支座 (cc=1):
      - Z_top = +20.0mm，为 8-12~16 左驾逆转机、8-7 右后旁通阀、8-6 右前排液柄与 11-9 背板提供真实物理落脚基座！
    """
    deck_8_11 = Pos(-130.8, 0.0, 19.1) * Box(34.8, 68.4, 1.8)
    apron_8_21 = Pos(-152.5, 0.0, 19.5) * Box(10.2, 42.0, 1.2)
    risers = (
        Pos(-141.0, +13.5, 14.8) * Box(5.0, 3.2, 7.6)
        + Pos(-141.0, -13.5, 14.8) * Box(5.0, 3.2, 7.6)
        + Pos(-128.0, +13.5, 14.8) * Box(5.0, 3.2, 7.6)
        + Pos(-128.0, -13.5, 14.8) * Box(5.0, 3.2, 7.6)
    )
    drain_slot_cut = Pos(-119.5, -18.6, 19.1) * Box(9.2, 5.2, 5.0)
    bracket_8_10 = (
        Pos(-119.5, -15.3, 21.4) * Box(7.6, 1.6, 3.6)
        + Pos(-119.5, -21.9, 21.4) * Box(7.6, 1.6, 3.6)
    )
    stand_4_8 = (
        Pos(-114.0, -6.2, 16.4) * Box(5.6, 6.4, 4.4)
        - Pos(-113.6, -6.2, 15.8) * Cylinder(radius=2.9, height=6.0)
    )
    return (deck_8_11 + apron_8_21 + risers + bracket_8_10 + stand_4_8) - drain_slot_cut


def build_cab_reverser_stand():
    """8-12/8-13 驾驶室左前侧 (+Y) 螺旋逆转机前后立柱支座 + 8-14 丝杠 + 8-15 刻度扇梁 (落座于 Z=+20.0mm 8-11 地板, 局部原点 (-123.0, +31.5, 33.5))。"""
    foot_plate = Pos(0.0, 0.0, -13.1) * Box(15.6, 5.2, 1.6)
    pillar_f = Pos(+5.4, 0.0, -4.2) * Box(2.6, 3.4, 18.6)
    pillar_r = Pos(-5.4, 0.0, -4.2) * Box(2.6, 3.4, 18.6)
    screw_8_14 = Pos(-0.8, 0.0, 3.8) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=15.0)
    guide_bar = Pos(0.0, 0.0, 1.0) * Rot(0, 90, 0) * Cylinder(radius=0.85, height=12.6)
    scale_8_15 = Pos(0.0, -1.5, 5.1) * Box(12.2, 1.0, 1.8)
    return foot_plate + pillar_f + pillar_r + screw_8_14 + guide_bar + scale_8_15


def build_cab_reverser_wheel():
    """8-16 驾驶室左侧螺旋逆转机手轮 (局部原点对准丝杠后端，绕 X 轴多圈旋转)。"""
    rim = Rot(0, 90, 0) * (Cylinder(radius=4.8, height=1.2) - Cylinder(radius=3.6, height=3.0))
    hub = Rot(0, 90, 0) * Cylinder(radius=1.65, height=2.2)
    spokes = Box(0.9, 8.8, 0.9) + Box(0.9, 0.9, 8.8)
    crank_handle = Pos(-1.5, 4.1, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.70, height=2.6)
    return rim + hub + spokes + crank_handle


def build_cab_regulator_lever():
    """
    11-11 驾驶室后壁中央加减阀手柄 (取扱説明書 p.30: 局部 q=0 关闭位水平指向左侧 +Y 机关士席，开启时绕 X 轴向上摆动划过 11-10 扇形齿板！)
    """
    hub = Rot(0, 90, 0) * Cylinder(radius=1.8, height=2.2)
    arm = Pos(-0.8, +5.8, 0.0) * Box(1.2, 11.8, 1.9)
    latch_lug = Pos(-0.2, +7.6, 0.0) * Box(1.6, 1.8, 2.7)
    grip = Pos(-2.1, +11.2, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.10, height=3.4)
    return hub + arm + latch_lug + grip


def build_cab_blower_wheel():
    """
    11-13 驾驶室左上部通风阀滚花圆柱黄铜旋钮 (イラスト 11 & ill_p45: ø5.0 x 4.6mm 滚花旋钮，沿左后 35° 阀杆轴线旋转，绝非辐条手轮！)
    """
    stem = Pos(-1.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.05, height=3.8)
    knob = Pos(-4.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.55, height=4.6)
    collar = Pos(-2.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.85, height=1.2)
    knurl_hex = make_hex_nut((-5.4, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=2.68, length=2.0)
    return stem + knob + collar + knurl_hex


def build_cab_whistle_lever():
    """11-3 驾驶室右上部 (-Y) 弹簧复位汽笛压柄 (绕 X 轴按压吹响车底 11-8 长汽笛)。"""
    hub = Rot(0, 90, 0) * Cylinder(radius=1.7, height=2.2)
    lever = Pos(-0.8, -3.4, 1.0) * Rot(-16, 0, 0) * Box(1.0, 7.0, 1.5)
    tip = Pos(-0.8, -6.6, 2.0) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=1.8)
    return hub + lever + tip


def build_cab_sight_glass_tube():
    """11-7 驾驶室左侧 (+Y=+10.5mm) 透明耐热硼硅玻璃水面计管 (ø4.0mm x 18.0mm 贯通空心管, mat_pyrex_glass, cc=1)。"""
    return Cylinder(radius=1.95, height=18.0) - Cylinder(radius=1.12, height=20.0)


def build_cab_sight_drain():
    """1-2 / 1-1 左侧水面计下座 (11-6) 底部弹簧排污冲洗顶针 (水面計テスト, 沿 Z 轴上推排汽冲洗)。"""
    stem = Cylinder(radius=0.85, height=4.4)
    button = Pos(0.0, 0.0, -2.0) * Cylinder(radius=1.65, height=1.4)
    return stem + button


def build_cab_pg_dial():
    """PG-ø20 右上部 (-Y=-11.8mm, Z=+58.0mm) 上仰 38° 真实 ø20mm 压力表白底刻度盘片 (独立材质 mat_gauge_dial)。"""
    return Rot(0, +38, 0) * Rot(0, 90, 0) * Cylinder(radius=8.4, height=0.8)


def build_cab_pressure_needle():
    """PG-ø20 压力表上仰 38° 黑色指针 (绕表盘法向 (-cos38°, 0, +sin38°) 随波登管压力偏转)。"""
    needle_flat = (
        Rot(0, 90, 0) * Cylinder(radius=1.15, height=0.9)
        + Pos(-0.35, 0.0, 2.7) * Box(0.45, 0.65, 6.0)
        + Pos(-0.35, 0.0, -1.3) * Box(0.45, 1.05, 2.2)
    )
    return Rot(0, +38, 0) * needle_flat


def build_cab_blowdown_screw():
    """11-17 驾驶室右下 (-Y=-11.5mm, Z=+24.8mm) 锅炉排污针阀一字槽/滚花旋柄 (绕 X 轴旋开泄放锅水)。"""
    stem = Rot(0, 90, 0) * Cylinder(radius=1.15, height=3.8)
    head = Pos(-2.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.35, height=2.2)
    slot = Pos(-3.2, 0.0, 0.0) * Box(0.8, 5.2, 0.65)
    return stem + (head - slot)


def build_cab_firebox_door():
    """12-1 驾驶室下中 JNR D51 矩形圆角铸造火室门 (绕左侧 +Y=+6.2mm 立销 12-3 向左后开启，露出 12-2 双观火圆孔)。"""
    hinge_barrel = Cylinder(radius=1.25, height=8.6)
    strap = Pos(-0.5, -3.2, 0.0) * Box(1.2, 6.8, 3.2)
    door_plate = Pos(-0.7, -6.2, 0.0) * Box(1.6, 11.4, 9.2)
    boss_ring = Pos(-1.4, -6.2, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=0.9)
    latch_handle = Pos(-1.9, -10.8, 0.6) * Box(1.4, 3.6, 1.3)
    return hinge_barrel + strap + door_plate + boss_ring + latch_handle


def build_cab_bypass_lever():
    """6-9 / 8-7 驾驶室右后地板 (+20.0mm) 旁通阀立杆与水平拨柄 (イラスト 6 & 8: 绕 Z 轴旋转控制床下 4-17 旁通回水)。"""
    stem_6_9 = Pos(0.0, 0.0, -3.2) * Cylinder(radius=1.0, height=7.2)
    collar = Cylinder(radius=1.9, height=1.5)
    handle_8_7 = Pos(-3.2, 0.0, 0.8) * Box(6.8, 1.6, 1.2) + Pos(-6.4, 0.0, 1.5) * Cylinder(radius=1.1, height=2.4)
    return stem_6_9 + collar + handle_8_7


def build_cab_roof_hatch():
    """21-15 ベンチレーター + 21-28 驾驶室顶棚前铰链可掀天窗总成 (ill_p45: 铰链位于 (-115.5, 0, +88.8)，绕 +Y 轴向前翻开直视驾驶室全貌)。"""
    L_h, W_h = 41.2, 35.4
    arch_outer = (Pos(-0.5 * L_h, 0.0, -36.8) * Rot(0, 90, 0) * Cylinder(radius=42.0, height=L_h)) & (
        Pos(-0.5 * L_h, 0.0, -2.0) * Box(L_h, W_h, 12.8)
    )
    arch_inner = Pos(-0.5 * L_h, 0.0, -36.8) * Rot(0, 90, 0) * Cylinder(radius=40.2, height=L_h + 2.0)
    panel = arch_outer - arch_inner
    vent_21_15 = (
        Pos(-19.5, 0.0, +4.8) * Box(14.5, 16.5, 2.8)
        + Pos(-19.5, 0.0, +6.3) * Box(16.2, 18.2, 1.0)
    )
    hinge_l = Pos(0.0, +12.0, +3.0) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=4.8)
    hinge_r = Pos(0.0, -12.0, +3.0) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=4.8)
    return panel + vent_21_15 + hinge_l + hinge_r


def build_tender_handpump_lever():
    """31-3 手摇给水泵摇臂套筒 + 37-12 着脱加长黄铜手柄 (自 35-1 敞口水槽底部 (-20, +8, +13.5) 竖直伸出水槽口，绕 Y 轴往复打水)。"""
    clevis = Rot(90, 0, 0) * Cylinder(radius=2.1, height=3.8)
    arm_31_3 = Pos(0.8, 0.0, 9.0) * Rot(0, 6, 0) * Box(2.6, 2.4, 18.5)
    socket = Pos(1.8, 0.0, 19.5) * Rot(0, 6, 0) * Cylinder(radius=2.0, height=6.5)
    ext_37_12 = Pos(3.6, 0.0, 36.0) * Rot(0, 6, 0) * Cylinder(radius=1.30, height=30.0)
    knob = Pos(5.2, 0.0, 51.5) * Rot(0, 6, 0) * Cylinder(radius=2.2, height=3.4)
    return clevis + arm_31_3 + socket + ext_37_12 + knob


def build_tender_handpump_ram():
    """31-2 水槽底浸渍横型プランジャ (ø7.0mm, PN7-1.9) + 31-7 双侧连杆耳 (随 j_tender_handpump 沿 X 轴水平往复压水)。"""
    plunger = Pos(-5.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.45, height=13.0)
    groove = Pos(-9.5, 0.0, 0.0) * Rot(0, 90, 0) * (Cylinder(radius=3.55, height=1.6) - Cylinder(radius=2.85, height=2.0))
    link_bar = Pos(+4.5, 0.0, 0.0) * Box(9.5, 4.4, 2.8)
    pin = Pos(+8.5, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.2, height=5.4)
    return (plunger - groove) + link_bar + pin


def build_tender_drain_plug():
    """31-11 水槽栓 (上部引抜式垂直排水ロッド + PS6-1.9 密封肩): 竖立于 35-1 敞口水槽右侧 (-38, -8)，向上提拉开启 31-9 底部放水！"""
    tip = Pos(0.0, 0.0, -3.0) * Cylinder(radius=1.45, height=6.6)
    seal_collar = Pos(0.0, 0.0, 0.4) * Cylinder(radius=3.1, height=2.2)
    shaft = Pos(0.0, 0.0, 20.5) * Cylinder(radius=1.25, height=40.0)
    pull_knob = Pos(0.0, 0.0, 41.2) * Cylinder(radius=2.8, height=3.6) + make_hex_nut((0.0, 0.0, 42.5), (0.0, 0.0, 1.0), hex_r=2.4, length=1.8)
    return tip + seal_collar + shaft + pull_knob


def build_tender_coal_cover():
    """36-1 + 36-2 煤水车前部可掀/可拆 L 型石炭槽盖板 (盖于 36-4 酒精箱上方，掀开即可操作 36-5 密封盖与 36-6 T 字燃料针阀)。"""
    L_c, W_c = 104.0, 55.6
    deck = Pos(-0.5 * L_c, 0.0, 0.0) * Box(L_c, W_c, 1.5)
    front_gate_36_2 = Pos(-2.0, 0.0, -3.8) * Box(1.8, 38.0, 8.6)
    rear_slope = Pos(-L_c + 6.0, 0.0, 2.2) * Rot(0, -18, 0) * Box(13.0, W_c - 1.0, 1.5)
    rib_1 = Pos(-28.0, 0.0, 1.2) * Box(2.2, W_c - 4.0, 1.8)
    rib_2 = Pos(-62.0, 0.0, 1.2) * Box(2.2, W_c - 4.0, 1.8)
    hinge_l = Pos(0.0, +22.0, 0.4) * Rot(90, 0, 0) * Cylinder(radius=1.3, height=5.0)
    hinge_r = Pos(0.0, -22.0, 0.4) * Rot(90, 0, 0) * Cylinder(radius=1.3, height=5.0)
    return deck + front_gate_36_2 + rear_slope + rib_1 + rib_2 + hinge_l + hinge_r


def build_tender_water_lid():
    """35-6 煤水车后甲板圆形检修盖 (位于敞口水槽后方 X=-76mm 处，绕前缘 Y 轴掀起)。"""
    hinge = Rot(90, 0, 0) * Cylinder(radius=1.1, height=5.4)
    arm = Pos(-4.2, 0.0, 0.4) * Box(8.4, 3.4, 1.2)
    lid_disc = Pos(-9.0, 0.0, 0.3) * Cylinder(radius=8.6, height=1.5)
    lip = Pos(-9.0, 0.0, -0.5) * Cylinder(radius=6.4, height=1.2)
    knob = Pos(-16.0, 0.0, 1.5) * Cylinder(radius=1.4, height=2.0)
    return hinge + arm + lid_disc + lip + knob


def build_tender_fuel_valve():
    """36-6 煤水车煤槽前部酒精针阀 T 字手柄 (绕 Z 轴逆时针旋开 90°)。"""
    stem = Pos(0.0, 0.0, -1.8) * Cylinder(radius=1.1, height=4.2)
    hub = Cylinder(radius=1.9, height=1.8)
    t_bar = Pos(0.0, 0.0, 0.8) * Rot(90, 0, 0) * Cylinder(radius=1.05, height=9.6)
    return stem + hub + t_bar


def build_tender_fuel_cap():
    """36-5 煤水车马氏瓶酒精箱气密滚花旋盖 (带 PN10 O 型圈密封肩)。"""
    plug = Pos(0.0, 0.0, -1.4) * Cylinder(radius=2.5, height=2.8)
    cap = Pos(0.0, 0.0, 0.8) * Cylinder(radius=4.8, height=2.4)
    knurl_hex = make_hex_nut((0.0, 0.0, 1.6), (0.0, 0.0, 1.0), hex_r=3.2, length=1.6)
    return plug + cap + knurl_hex


# ==================== 3C. 机车↔煤水车 3 根跨车弹性软管与 34-8 活接 ====================

def _compute_drawbar_hose_targets_mm(cfg: Dict, is_r2m: bool) -> Dict[str, Tuple[float, float, float]]:
    """精确复刻 03_generate_mjcf.py 中 roller 与 curve_r2m 姿态下煤水车前隔板三嘴在 drawbar_8_8 局部系下的毫米坐标。"""
    w_cfg = cfg["wheels_and_axles"]
    t_cfg = cfg["tender_and_track"]
    r_drv = 0.5 * float(w_cfg["driving_wheel_diameter"])
    r_tdr = 0.5 * float(t_cfg["tender_wheel_diameter"])
    x_pin_loco = float(t_cfg["drawbar_loco_pin_x"])          # -146.0 mm
    L_drawbar = float(t_cfg["drawbar_length"])               # 34.0 mm
    x_pin_tdr_local = float(t_cfg["drawbar_tender_pin_local_x"])  # +116.0 mm
    half_span = 0.5 * float(t_cfg["bogie_center_span"])
    half_wb = 0.5 * float(t_cfg["bogie_wheelbase"])
    R_curve = float(t_cfg["curve_radius_mm"])                # 2000.0 mm

    if is_r2m:
        x_drv_mid = 0.5 * (float(w_cfg["axle_x"]["drive_1"]) + float(w_cfg["axle_x"]["drive_4"]))
        half_dwb = 0.5 * (float(w_cfg["axle_x"]["drive_1"]) - float(w_cfg["axle_x"]["drive_4"]))
        sag_drv = R_curve - math.sqrt(R_curve**2 - half_dwb**2)
        loco_wx, loco_wy = -x_drv_mid, -0.5 * sag_drv
        z_loco_axle = 4.0 + r_drv + 0.2
        base_rail_z = 4.0
        pin_loco_wx = loco_wx + x_pin_loco
        pin_loco_wy = loco_wy
        R_tdr = math.sqrt(R_curve**2 - half_wb**2 - half_span**2)
        phi_loco_pin = math.atan2(pin_loco_wy + R_curve, pin_loco_wx)
        phi_tdr = phi_loco_pin + (L_drawbar + x_pin_tdr_local) / R_curve
        tdr_wx = R_tdr * math.cos(phi_tdr)
        tdr_wy = -R_curve + R_tdr * math.sin(phi_tdr)
        tdr_yaw = phi_tdr - 0.5 * math.pi
    else:
        loco_wx, loco_wy = 0.0, 0.0
        z_loco_axle = r_drv + 20.0
        base_rail_z = z_loco_axle - r_drv
        pin_loco_wx = loco_wx + x_pin_loco
        pin_loco_wy = loco_wy
        tdr_wx = pin_loco_wx - L_drawbar - x_pin_tdr_local
        tdr_wy = 0.0
        tdr_yaw = 0.0

    z_tdr_deck = base_rail_z + 2.0 * r_tdr - 9.0
    z_db_origin = z_loco_axle - 6.0

    def to_db_local(tx_l: float, ty_l: float, tz_l: float) -> Tuple[float, float, float]:
        wx = tdr_wx + tx_l * math.cos(tdr_yaw) - ty_l * math.sin(tdr_yaw)
        wy = tdr_wy + tx_l * math.sin(tdr_yaw) + ty_l * math.cos(tdr_yaw)
        wz = z_tdr_deck + tz_l
        return (wx - pin_loco_wx, wy - pin_loco_wy, wz - z_db_origin)

    return {
        "fuel_tdr": to_db_local(113.2, 0.0, +7.7),
        "feed_tdr": to_db_local(113.2, -7.2, -5.1),
        "ret_tdr": to_db_local(113.2, +7.2, -5.1),
    }


def build_umb_hose_fuel(cfg: Dict, is_r2m: bool = False):
    """1) ST-ø7 半透明白色硅胶供醇软管 (牵引杆 8-8 上方中央 Y=0, 23-1 <-> 34-2 サンプ)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["fuel_tdr"]
    p0 = (-3.6, 0.0, +9.6)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), min(p0[2], p2[2]) - 1.5)
    yaw_tdr = math.atan2(p2[1] - p0[1], p2[0] - p0[0]) + math.pi
    t_end = (-math.cos(yaw_tdr - math.pi), -math.sin(yaw_tdr - math.pi), 0.0)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), t_end, r_hose=2.45)


def build_umb_hose_feed(cfg: Dict, is_r2m: bool = False):
    """2) BT-ø5 黑色氟橡胶给水/手摇泵共用软管 (牵引杆下方斜跨: 机车左侧 +Y 6-11 <-> 煤水车右侧 -Y 34-6)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["feed_tdr"]
    p0 = (-3.6, +7.2, -3.2)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -7.8)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), r_hose=1.82)


def build_umb_union_34_8(cfg: Dict, is_r2m: bool = False):
    """34-8 滚花黄铜共用给水软管活接螺母 + 两侧 HC-1 弹簧卡箍总成 (套接于 umb_hose_feed 中段, cc=1)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["feed_tdr"]
    p0 = (-3.6, +7.2, -3.2)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -7.8)
    dx, dy, dz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    ax = Vector(dx, dy, dz).normalized()
    sleeve = extrude(Plane(origin=Vector(*pm) - ax * 3.8, z_dir=ax) * Circle(2.25), amount=7.6)
    nut_34_8 = make_hex_nut(pm, (ax.X, ax.Y, ax.Z), hex_r=3.15, length=3.2)
    hc1_f = extrude(Plane(origin=Vector(*pm) + ax * 2.2, z_dir=ax) * Circle(2.55), amount=1.1)
    hc1_r = extrude(Plane(origin=Vector(*pm) - ax * 3.3, z_dir=ax) * Circle(2.55), amount=1.1)
    return sleeve + nut_34_8 + hc1_f + hc1_r


def build_umb_hose_return(cfg: Dict, is_r2m: bool = False):
    """3) BT-ø5 黑色氟橡胶旁通无压回水软管 (牵引杆下方斜跨: 机车右侧 -Y 4-16 <-> 煤水车左侧 +Y 31-14)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["ret_tdr"]
    p0 = (-3.6, -7.2, -3.2)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -3.4)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), r_hose=1.82)


# ==================== 3D. 均衡梁闸瓦、台车架、气缸体与锅炉外衣 ====================

def build_chassis_equalizer_brakes(cfg: Dict):
    w_cfg = cfg["wheels_and_axles"]
    ax_xs = [
        float(w_cfg["axle_x"]["drive_1"]),
        float(w_cfg["axle_x"]["drive_2"]),
        float(w_cfg["axle_x"]["drive_3"]),
        float(w_cfg["axle_x"]["drive_4"]),
    ]

    tie_f = Pos(0.5 * (ax_xs[0] + ax_xs[1]), 0.0, 13.2) * Box(4.0, 29.6, 2.2)
    tie_r = Pos(0.5 * (ax_xs[2] + ax_xs[3]), 0.0, 13.2) * Box(4.0, 29.6, 2.2)
    assy = tie_f + tie_r

    for sgn in (-1.0, 1.0):
        y_eq = sgn * 10.7
        y_frame_out = sgn * 15.6

        for ax in ax_xs:
            sp1 = Pos(ax, y_eq, 12.8) * Box(18.0, 1.5, 1.6)
            sp2 = Pos(ax, y_eq, 14.0) * Box(12.5, 1.5, 1.4)
            buckle = Pos(ax, y_eq, 13.3) * Box(3.2, 2.0, 3.0)
            spring_lug = Pos(ax, sgn * 12.45, 12.9) * Box(3.2, 4.8, 1.9)
            assy += sp1 + sp2 + buckle + spring_lug

        for xm in (0.5 * (ax_xs[0] + ax_xs[1]), 0.5 * (ax_xs[2] + ax_xs[3])):
            rocker = Pos(xm, y_eq, 12.1) * Box(31.0, 1.4, 2.4)
            fulcrum = Pos(xm, y_eq, 12.9) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=2.2)
            assy += rocker + fulcrum

        hanger_xs = []
        for idx_a, ax in enumerate(ax_xs):
            if idx_a == 0:
                sx, sz, rot_p, hx = ax - 19.0, -19.6, -44.0, ax - 19.5
            else:
                sx, sz, rot_p, hx = ax + 19.0, -19.6, +44.0, ax + 19.5
            hanger_xs.append(hx)
            shoe = Pos(sx, sgn * 23.0, sz) * Rot(0, rot_p, 0) * Box(2.0, 2.2, 6.4)
            hanger_v = Pos(hx, y_frame_out, -11.6) * Box(2.0, 1.6, 17.4)
            cross_pin = Pos(hx, 0.5 * (y_frame_out + sgn * 23.0), sz - 0.3) * Rot(90, 0, 0) * Cylinder(
                radius=1.00, height=abs(sgn * 23.0 - y_frame_out) + 1.4
            )
            pad_f = Pos(hx, sgn * 14.2, -3.2) * Box(3.6, 2.6, 3.0)
            riser = Pos(hx, sgn * 14.2, 4.8) * Box(2.4, 1.6, 17.6)
            assy += shoe + hanger_v + cross_pin + pad_f + riser

        x_lo = min(min(hanger_xs), min(ax_xs)) - 2.5
        x_hi = max(max(hanger_xs), max(ax_xs)) + 2.5
        long_tie = Pos(0.5 * (x_lo + x_hi), sgn * 14.2, 12.8) * Box(x_hi - x_lo, 1.6, 1.9)
        assy += long_tie

    return assy


def build_axle_pump_eccentric(cfg: Dict):
    b_cfg = cfg["boiler_and_aux"]
    L_rod = abs(float(b_cfg["axle_pump_x"]) - float(b_cfg["axle_pump_eccentric"]))

    sheave_strap = Rot(90, 0, 0) * Cylinder(radius=6.3, height=3.2) - Rot(90, 0, 0) * Cylinder(radius=2.85, height=5.0)
    flange_ring = Rot(90, 0, 0) * Cylinder(radius=5.1, height=4.0) - Rot(90, 0, 0) * Cylinder(radius=2.85, height=6.0)
    neck_rod = Pos(-0.5 * (L_rod + 4.5), 0, 0) * Box(L_rod - 4.0, 2.1, 2.9)
    ram_clevis = (
        Pos(-L_rod, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.3, height=4.0)
        - Pos(-L_rod - 0.8, 0, 0) * Box(4.2, 2.1, 5.5)
    )
    ram_pin = Pos(-L_rod, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=4.6)
    return sheave_strap + flange_ring + neck_rod + ram_clevis + ram_pin


def build_leading_truck_frame(cfg: Dict):
    y_in = 0.5 * float(cfg["meta"]["track_gauge"]) - 5.2
    cross_bolster = Box(9.0, 2.0 * y_in + 4.0, 4.6)
    pivot_tongue = Pos(-57.0, 0, 0) * Box(20.0, 11.0, 4.2)
    boss_r2m = Pos(-52.0, 0, 0) * Cylinder(radius=3.6, height=5.2)
    boss_r3m = Pos(-62.0, 0, 0) * Cylinder(radius=3.6, height=5.2)

    arms = None
    for sgn in (-1.0, 1.0):
        jbox = (
            Pos(0, sgn * y_in, 0.4) * Box(8.6, 4.2, 8.2)
            - Pos(0, sgn * y_in, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=7.0)
        )
        beam = Pos(-26.0, sgn * 10.6, 0) * Rot(0, 0, sgn * +14.5) * Box(55.0, 3.4, 3.8)
        sub = jbox + beam
        arms = sub if arms is None else (arms + sub)

    frame = cross_bolster + pivot_tongue + boss_r2m + boss_r3m + arms
    frame -= Pos(-52.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Pos(-62.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Rot(90, 0, 0) * Cylinder(radius=2.1, height=2.0 * y_in + 8.0)
    return frame


def build_trailing_truck_frame(cfg: Dict):
    dx_ax = -54.0
    y_out = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.3
    pivot_nose = Cylinder(radius=4.5, height=5.0) + Pos(-6.5, 0, 0) * Box(13.0, 10.0, 4.4)
    front_transom = Pos(-34.5, 0, 0.5) * Box(3.4, 2.0 * y_out + 2.4, 4.0)

    sides = None
    for sgn in (-1.0, 1.0):
        flare_arm = Pos(-21.0, sgn * 15.8, 0.5) * Rot(0, 0, sgn * -42.7) * Box(39.0, 3.4, 4.2)
        side_beam = Pos(-47.0, sgn * y_out, 0.5) * Box(28.0, 3.4, 4.2)
        abox = (
            Pos(dx_ax, sgn * y_out, 0.4) * Box(10.0, 4.4, 9.2)
            + Pos(dx_ax, sgn * (y_out + 1.2), 3.6) * Box(12.0, 3.6, 3.0)
            - Pos(dx_ax, sgn * y_out, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=7.0)
        )
        sub = flare_arm + side_beam + abox
        sides = sub if sides is None else (sides + sub)

    return pivot_nose + front_transom + sides - Cylinder(radius=1.8, height=8.0)


def build_driving_axlebox(cfg: Dict):
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) - 6.5
    tie = Pos(0, 0, 5.2) * Box(4.0, 2.0 * y_box, 2.0)
    pair = tie
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0.5) * Box(10.5, 5.2, 11.5)
        flange = Pos(0, sgn * y_box, 0.5) * Box(12.5, 2.2, 13.0)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=8.0)
        pair += blk + flange - bore
    return pair


def build_tender_axlebox(cfg: Dict):
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2
    tie = Pos(0, 0, 2.8) * Box(3.0, 2.0 * y_box, 1.6)
    pair = tie
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0) * Box(7.0, 3.8, 7.6)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=1.9, height=6.0)
        pair += blk - bore
    return pair


def build_cylinder_block(cfg: Dict, is_left: bool = True):
    bore = float(cfg["cylinder_and_rods"]["bore"])
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    cyl_od = bore + 7.0

    main_barrel = Pos(+5.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od, height=21.6)
    rear_neck = (
        Pos(-8.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=5.8, height=7.8)
        + Pos(-13.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.6)
        - Pos(-9.8, -7.20, 0.0) * Box(14.0, 10.0, 16.0)
        - Pos(-9.8, +6.85, 0.0) * Box(14.0, 10.0, 16.0)
    )
    saddle_block = Pos(+5.2, -5.5, 4.5) * Box(20.0, 14.0, cyl_od + 3.0)
    front_cover = Pos(+16.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od + 0.8, height=2.2)

    for b in (
        Pos(+17.4, 0.0, 0.0)
        * Rot(0, 90, 0)
        * PolarLocations(radius=0.5 * bore + 2.2, count=8)
        * Cylinder(radius=0.85, height=1.2)
    ):
        front_cover += b

    steam_chest = Pos(+4.8, 0.5, z_valve) * Box(21.2, 15.0, 11.0)
    inlet_boss_1_20 = Pos(0.0, -3.0, z_valve + 4.8) * Cylinder(radius=2.4, height=2.6)
    drain_boss_r = Pos(+3.0 - 5.0, -1.7, -10.5) * Cylinder(radius=2.1, height=3.2)
    drain_boss_f = Pos(+3.0 + 5.0, -1.7, -10.5) * Cylinder(radius=2.1, height=3.2)
    gland_gp5 = Pos(-6.2, 0.0, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.9, height=1.8)

    bore_cut = (
        Pos(+5.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=19.6)
        + Pos(-9.5, -0.15, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=13.0)
    )
    assembly_L = (
        main_barrel + rear_neck + saddle_block + front_cover
        + steam_chest + inlet_boss_1_20 + drain_boss_r + drain_boss_f + gland_gp5 - bore_cut
    )
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_boiler_superstructure(cfg: Dict, heavy_equipped: bool = True):
    """
    锅炉烟室驾驶室外衣总成 (Gate 14 左驾校正: 8-18 拉杆穿墙槽与 25-7 走台护罩全部移至左侧 +Y!):
      - 外边界由两侧除烟板 28-11/12 严格锁定在 [-40.8, +40.8] mm，保证 |min_y + max_y| = 0.000mm！
    """
    b_cfg = cfg["boiler_and_aux"]
    od_sb = float(b_cfg["smokebox_od"])
    r_sb, L_sb = 0.5 * od_sb, float(b_cfg["smokebox_length"])
    fb_l, fb_w, fb_h = [float(v) for v in b_cfg["firebox_lwh"]]
    z_axis, x_sb_front = 44.0, 162.0
    x_sb_rear = x_sb_front - L_sb
    x_fb_rear = -112.0
    x_fb_front = x_fb_rear + fb_l

    sb_x0, sb_x1 = x_sb_rear - 0.5, x_sb_front
    smokebox = Pos(0.5 * (sb_x0 + sb_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb, height=sb_x1 - sb_x0)
    sb_saddle = Pos(136.0, 0, 21.5) * Box(36.0, 26.0, 14.0)
    door_ring = Pos(x_sb_front + 0.4, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 1.0, height=2.4)
    door_cone1 = Pos(x_sb_front + 2.3, 0, z_axis) * Rot(0, 90, 0) * Cone(
        bottom_radius=r_sb - 3.0, top_radius=r_sb - 11.5, height=2.6
    )
    door_cone2 = Pos(x_sb_front + 4.1, 0, z_axis) * Rot(0, 90, 0) * Cone(
        bottom_radius=r_sb - 11.2, top_radius=5.5, height=1.8
    )
    door_handle = (
        Pos(x_sb_front + 5.6, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=2.0)
        + Pos(x_sb_front + 6.4, 0, z_axis + 2.2) * Box(1.0, 1.2, 5.2)
    )
    hinge_u = Pos(x_sb_front + 2.2, 10.5, z_axis + 9.5) * Box(2.4, 21.0, 2.0)
    hinge_l = Pos(x_sb_front + 2.2, 10.5, z_axis - 9.5) * Box(2.4, 21.0, 2.0)
    headlight = Pos(x_sb_front + 0.8, 0, z_axis + r_sb + 4.2) * Rot(0, 90, 0) * Cylinder(radius=4.4, height=9.2)
    headlight_ped = Pos(x_sb_front - 1.0, 0, z_axis + r_sb + 0.5) * Box(5.0, 5.0, 4.0)
    feedwater_heater = Pos(x_sb_front - 14.0, 0, z_axis + r_sb + 2.2) * Rot(90, 0, 0) * Cylinder(radius=5.2, height=44.0)
    fwh_brackets = (
        Pos(x_sb_front - 14.0, -14.0, z_axis + r_sb - 1.0) * Box(6.0, 3.0, 5.0)
        + Pos(x_sb_front - 14.0, +14.0, z_axis + r_sb - 1.0) * Box(6.0, 3.0, 5.0)
    )

    x_chimney = x_sb_front - 32.0
    chimney = Pos(x_chimney, 0, z_axis + r_sb + 5.5) * Cylinder(radius=7.8, height=16.0)
    if heavy_equipped:
        chimney += Pos(x_chimney - 2.0, 0, z_axis + r_sb + 10.5) * Box(22.0, 19.0, 13.0)
        chimney += Pos(x_chimney - 10.0, 0, z_axis + r_sb + 14.0) * Box(12.0, 17.0, 6.0)

    b_x0, b_x1 = x_fb_rear, x_sb_rear + 1.0
    r_barrel = 28.4
    barrel = Pos(0.5 * (b_x0 + b_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=b_x1 - b_x0)
    for bx in [72.0, 38.0, -16.0, -60.0]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.6, height=2.2)

    dome_base = Pos(22.0, 0, z_axis + r_barrel - 3.0) * extrude(SlotOverall(width=72.0, height=28.0), amount=14.0)
    firebox = Pos(0.5 * ((x_fb_front + 1.0) + x_fb_rear), 0, z_axis - 11.0) * Box(fb_l + 1.0, fb_w, fb_h)

    cab_len, cab_w = 54.0, 74.0
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0
    cab_box = Pos(cab_x, 0, z_axis + 4.5) * Box(cab_len, cab_w, 53.0)
    cab_roof = (Pos(cab_x - 2.0, 0, z_axis + 8.0) * Rot(0, 90, 0) * Cylinder(radius=42.0, height=cab_len + 8.0)) & (
        Pos(cab_x - 2.0, 0, z_axis + 35.0) * Box(cab_len + 8.0, cab_w + 4.0, 22.0)
    )
    cab_roof_hollow = (
        Pos(cab_x - 2.5, 0, z_axis + 8.0) * Rot(0, 90, 0) * Cylinder(radius=40.2, height=cab_len - 3.0)
    ) & (Pos(cab_x - 2.5, 0, 77.0) * Box(cab_len - 3.0, cab_w - 4.0, 24.0))
    roof_hatch_cut = Pos(-136.5, 0.0, 86.0) * Box(41.8, 36.0, 16.0)
    cab_trim = None
    for sgn_c in (-1.0, 1.0):
        visor_21_11 = Pos(cab_x + 2.0, sgn_c * 37.6, z_axis + 24.0) * Box(29.0, 2.2, 1.2)
        handrail_30_4 = (
            Pos(cab_x - 0.5 * cab_len + 0.6, sgn_c * 24.2, z_axis + 4.0) * Cylinder(radius=0.75, height=28.0)
            + Pos(cab_x - 0.5 * cab_len + 1.2, sgn_c * 24.2, z_axis + 17.0) * Box(2.2, 2.0, 1.6)
            + Pos(cab_x - 0.5 * cab_len + 1.2, sgn_c * 24.2, z_axis - 9.0) * Box(2.2, 2.0, 1.6)
        )
        ct = visor_21_11 + handrail_30_4
        cab_trim = ct if cab_trim is None else (cab_trim + ct)
    # 驾驶室前壁左侧 (+Y=+33.5mm) 留出 8-18 逆转长拉杆穿墙孔 + 顶棚中央 21-15/28 可掀天窗开口 + 后壁火室门圆孔保持贯通
    cab = (
        cab_box + cab_roof + cab_trim
        - Pos(cab_x - 2.5, 0, z_axis + 3.5) * Box(cab_len - 3.0, cab_w - 4.0, 51.0)
        - cab_roof_hollow
        - roof_hatch_cut
        - Pos(cab_x + 2.0, 0, z_axis + 14.5) * Box(28.0, cab_w + 10.0, 18.0)
        - Pos(cab_x - 0.5 * cab_len, 0, z_axis + 2.5) * Box(8.0, 46.0, 48.0)
        - Pos(-118.0, +33.5, 33.5) * Box(26.0, 6.2, 7.6)
    )

    walk_x0, walk_x1 = x_fb_rear + 1.0, x_sb_front + 6.0
    walk_len = walk_x1 - walk_x0
    walk_xm = 0.5 * (walk_x0 + walk_x1)
    walkways = None
    for sgn in (-1.0, 1.0):
        board = Pos(walk_xm, sgn * 34.4, 30.8) * Box(walk_len, 10.2, 1.5)
        inward_web = Pos(walk_xm, sgn * 27.5, 30.8) * Box(walk_len - 8.0, 5.0, 1.5)
        rev_cut = Pos(30.5, sgn * 35.1, 30.8) * Box(26.0, 11.4, 5.0)
        board_sub = board + inward_web - rev_cut

        defl = Pos(131.5, sgn * 40.2, 49.2) * Box(59.0, 1.2, 38.0)
        defl -= Pos(161.0, sgn * 40.2, 68.2) * Rot(0, -38, 0) * Box(18.0, 4.0, 18.0)
        defl_stay_u = Pos(144.0, sgn * 35.0, 61.0) * Box(1.8, 11.0, 1.8)
        defl_stay_l = Pos(112.0, sgn * 35.0, 58.0) * Box(1.8, 11.0, 1.8)

        dummy_15_3 = (
            Pos(58.0, sgn * 28.2, 44.0) * Rot(90, 0, 0) * Cylinder(radius=3.0, height=3.6)
            + Pos(58.0, sgn * 29.8, 45.2) * Cylinder(radius=2.1, height=4.8)
        )
        dummy_16_2 = Pos(-78.0, sgn * 24.4, 24.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.4)
        air_res = Pos(-18.0, sgn * 27.6, 36.2) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=34.0)

        part = board_sub + defl + defl_stay_u + defl_stay_l + dummy_15_3 + dummy_16_2 + air_res
        walkways = part if walkways is None else (walkways + part)

    front_apron = (
        Pos(x_sb_front + 3.0, 0.0, 24.0) * Box(2.2, 76.0, 14.5)
        + Pos(x_sb_front + 6.5, 0.0, 17.2) * Box(8.5, 76.0, 1.8)
    )
    hatch_25_3 = (
        Pos(165.0, 0.0, 18.4) * Box(6.4, 9.2, 1.2)
        + Pos(162.0, 0.0, 19.0) * Rot(90, 0, 0) * Cylinder(radius=0.7, height=8.0)
    )

    comp_22_11 = (
        Pos(-46.0, +33.2, 37.5) * Cylinder(radius=3.4, height=12.5)
        + Pos(-46.0, +33.2, 41.5) * Cylinder(radius=3.9, height=3.2)
        + Pos(-46.0, +28.0, 37.5) * Box(5.0, 10.0, 4.0)
    )
    silencer_28_8 = (
        Pos(118.0, +34.2, 34.2) * Rot(0, 90, 0) * Cylinder(radius=2.5, height=11.0)
        + Pos(118.0, +31.0, 32.2) * Box(4.0, 6.5, 3.0)
    )
    dummy_whistle_30_2 = Pos(-48.0, -7.5, z_axis + r_barrel + 3.0) * Cylinder(radius=1.8, height=7.5)
    gen_15_8 = (
        Pos(-84.0, -25.5, 62.5) * Rot(0, 90, 0) * Cylinder(radius=3.0, height=9.5)
        + Pos(-84.0, -21.0, 59.5) * Box(5.0, 8.5, 5.0)
    )
    # 25-7 左走台 (+Y=+34.2mm) 逆转摇臂桥式护罩 (避让左侧 8-18 拉杆且 |Y|<=39.5mm < 40.8mm)
    cover_25_7 = (
        Pos(22.5, +34.2, 35.4) * Box(14.0, 10.6, 9.8)
        - Pos(22.5, +34.2, 34.0) * Box(17.0, 8.2, 9.4)
    )

    outer_solid = (
        smokebox + sb_saddle + door_ring + door_cone1 + door_cone2 + door_handle + hinge_u + hinge_l
        + headlight + headlight_ped + feedwater_heater + fwh_brackets + chimney
        + barrel + dome_base + firebox + cab + walkways + front_apron + hatch_25_3
        + comp_22_11 + silencer_28_8 + dummy_whistle_30_2 + gen_15_8 + cover_25_7
    )

    wheel_cuts = None
    for ax in (93.0, 46.5, 0.0, -46.5):
        for sgn in (-1.0, 1.0):
            wc = Pos(ax, sgn * 22.5, 0.0) * Rot(90, 0, 0) * Cylinder(radius=25.8, height=9.2)
            wheel_cuts = wc if wheel_cuts is None else (wheel_cuts + wc)

    bore_barrel = (
        Pos(0.5 * (-109.5 + 93.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=24.2, height=93.5 - (-109.5))
        + (
            Pos(0.5 * (-109.5 + 93.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=26.8, height=93.5 - (-109.5))
            & Pos(0.0, 0.0, 56.5) * Box(220.0, 60.0, 58.0)
        )
        - Pos(25.5, 0, 24.0) * Box(9.0, 56.0, 16.0)
    )
    bore_smokebox = (
        Pos(0.5 * (92.5 + 119.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=24.2, height=119.5 - 92.5)
        + (
            Pos(0.5 * (92.5 + 160.2), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 1.6, height=160.2 - 92.5)
            & Pos(126.0, 0.0, 56.5) * Box(80.0, 70.0, 58.0)
        )
        + Pos(0.5 * (119.5 + 160.2), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 2.0, height=160.2 - 119.5)
    )
    bore_firebox = (
        Pos(0.5 * (x_fb_front + x_fb_rear), 0, z_axis - 13.5) * Box(fb_l - 6.0, 33.6, fb_h)
        + Pos(0.5 * (x_fb_front + x_fb_rear), 0, 44.0) * Box(fb_l - 4.0, 45.0, 33.0)
    )
    bore_dome_cab = (
        Pos(22.0, 0, 72.0) * Box(64.0, 23.0, 18.0)
        + Pos(cab_x - 2.5, 0, z_axis + 26.0) * Box(cab_len - 4.0, 54.0, 16.0)
        + Pos(136.0, 0, 21.5) * Box(31.0, 21.0, 10.0)
    )
    bore_chimney = Pos(x_chimney, 0, z_axis + r_sb + 6.0) * Cylinder(radius=6.0, height=28.0)
    rev_shaft_tunnel = Pos(25.5, 0, 26.5) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=62.0)
    firehole_outer_cut = Pos(-111.0, 0.0, 38.0) * Rot(0, 90, 0) * Cylinder(radius=5.2, height=8.0)

    return (
        outer_solid
        - wheel_cuts
        - bore_barrel
        - bore_smokebox
        - bore_firebox
        - bore_dome_cab
        - bore_chimney
        - rev_shaft_tunnel
        - firehole_outer_cut
    )


# ==================== 4. 四回路圆弧扫掠炉内管网、车架内三软管水路、置换注油器、1x3 三头燃烧器与煤水车内脏 ====================

def build_boiler_firebox_flues(cfg: Dict, pip_cfg: Dict):
    pv = pip_cfg["boiler_pressure_vessel"]
    z_axis = float(pv["centerline_z_mm"])
    r_in = 0.5 * float(pv["barrel_id_mm"])
    r_out = r_in + 0.8
    x_rear, x_front = [float(v) for v in pv["barrel_x_range_mm"]]

    fb = pv["inner_firebox"]
    fb_x0, fb_x1 = [float(v) for v in fb["x_range_mm"]]
    fb_len = fb_x1 - fb_x0
    fb_xm = 0.5 * (fb_x0 + fb_x1)
    fb_w = float(fb["width_mm"])
    fb_z0 = float(fb["floor_z_mm"])
    z_crown = float(fb["crown_sheet_z_mm"])
    fb_h = z_crown - fb_z0
    t_cu = float(fb["wall_thickness_mm"])

    fl5 = pv["five_flue_tubes"]
    fl_x0, fl_x1 = [float(v) for v in fl5["x_range_mm"]]
    fl_end_x = fl_x1 + 2.2
    fl_len = fl_end_x - (fl_x0 - 0.5)
    fl_xm = 0.5 * (fl_end_x + (fl_x0 - 0.5))

    up_spec = fl5["upper_superheater_flues"]
    lo_spec = fl5["lower_fire_tubes"]

    full_len = x_front - x_rear
    full_xm = 0.5 * (x_front + x_rear)
    outer_barrel = Pos(full_xm, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_out, height=full_len)
    fb_legs = Pos(fb_xm, 0, 0.5 * (fb_z0 + z_axis)) * Box(fb_len + 3.2, fb_w + 1.6, z_axis - fb_z0)
    under_fb_trim = Pos(0.5 * (x_rear - 2.0 + fb_x1), 0, 0.5 * (16.0 + fb_z0)) * Box(
        (fb_x1 - (x_rear - 2.0)), 60.0, fb_z0 - 16.0
    )
    barrel_cavity = Pos(0.5 * (fb_x1 + 2.4 + x_front - 2.4), 0, z_axis) * Rot(0, 90, 0) * Cylinder(
        radius=r_in, height=(x_front - 2.4) - (fb_x1 + 2.4)
    )
    steam_dome_cavity = Pos(0.5 * (x_rear + 2.2 + fb_x1 + 4.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(
        radius=r_in - 0.4, height=(fb_x1 + 4.0) - (x_rear + 2.2)
    ) & Pos(0.5 * (x_rear + fb_x1 + 4.0), 0, z_crown + 13.0) * Box(
        (fb_x1 + 4.0 - x_rear) + 4.0, 58.0, 24.0
    )
    fb_cavity = Pos(fb_xm, 0, fb_z0 + 0.5 * (fb_h - t_cu) - 1.0) * Box(
        fb_len - 2.0 * t_cu, fb_w - 2.0 * t_cu, fb_h - t_cu + 2.0
    )
    shell_body = (outer_barrel + fb_legs) - under_fb_trim - barrel_cavity - steam_dome_cavity - fb_cavity
    mud_ring = Pos(fb_xm, 0, fb_z0 + 1.0) * Box(fb_len + 3.6, fb_w + 2.4, 2.0) - Pos(
        fb_xm, 0, fb_z0 + 1.0
    ) * Box(fb_len - 2.0 * t_cu, fb_w - 2.0 * t_cu, 4.0)
    firehole_ring = Pos(x_rear - 0.4, 0, z_crown - 9.5) * Rot(0, 90, 0) * Cylinder(radius=6.6, height=4.4)

    stays = None
    for sx in (-95.0, -78.0, -61.0):
        for sy in (-12.0, -4.0, 4.0, 12.0):
            b = Pos(sx, sy, z_crown + 0.6) * Cylinder(radius=1.1, height=2.2)
            stays = b if stays is None else (stays + b)

    flues_outer = None
    flues_bore = None
    all_tubes = [
        (float(yz[0]), float(yz[1]), 0.5 * float(up_spec["od_mm"]), 0.5 * float(up_spec["id_mm"]))
        for yz in up_spec["yz_centers_mm"]
    ] + [
        (float(yz[0]), float(yz[1]), 0.5 * float(lo_spec["od_mm"]), 0.5 * float(lo_spec["id_mm"]))
        for yz in lo_spec["yz_centers_mm"]
    ]
    for yf, zf, r_o, r_i in all_tubes:
        fo = Pos(fl_xm, yf, zf) * Rot(0, 90, 0) * Cylinder(radius=r_o, height=fl_len)
        bead = Pos(fl_end_x - 0.5, yf, zf) * Rot(0, 90, 0) * Cylinder(radius=r_o + 0.65, height=1.2)
        fb_cut = Pos(fl_xm, yf, zf) * Rot(0, 90, 0) * Cylinder(radius=r_i, height=fl_len + 8.0)
        flues_outer = (fo + bead) if flues_outer is None else (flues_outer + fo + bead)
        flues_bore = fb_cut if flues_bore is None else (flues_bore + fb_cut)

    firehole_bore = Pos(x_rear + 1.0, 0, z_crown - 9.5) * Rot(0, 90, 0) * Cylinder(radius=4.8, height=12.0)
    rev_liner_cut = Pos(25.5, 0, 26.5) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=58.0)
    feed_bush_13_5 = Pos(-42.5, 0.0, 17.0) * Cylinder(radius=2.4, height=2.6)
    crown_bush_13_5 = Pos(-105.5, 0.0, 70.6) * Cylinder(radius=2.4, height=2.4)

    return (
        shell_body + mud_ring + firehole_ring + stays + flues_outer + feed_bush_13_5 + crown_bush_13_5
        - flues_bore - firehole_bore - rev_liner_cut
    )


def build_superheater_piping(cfg: Dict, pip_cfg: Dict):
    """
    圆弧扫掠版炉内干汽管、双U四程过热器、烟室 17-3 通风下垂管与 7-8 排汽总成 (Gate 12 & Gate 16):
      - 炉内干汽管自锅顶 13-5 衬套 (-105.5, 0.0, 68.5) 平滑弯入 Z=60.5mm 纵贯至烟室 17-6 空心螺栓；
      - 17-3 烟室通风下垂管自烟室后隔板 (94.0, +6.2, 58.0) 衔接外部 13-8 通风管后下垂汇入 20-7 喷嘴环！
    """
    sp = pip_cfg["superheater_and_steam_piping"]
    pv = pip_cfg["boiler_pressure_vessel"]
    sh = sp["superheater_4pass"]
    br = sp["branch_steam_pipes"]
    ej = sp["smokebox_draft_ejector"]
    reg = sp["cab_regulator_11_10"]

    z_pipe = float(reg["internal_pipe_z_mm"])
    r_dry = 0.5 * float(reg["internal_pipe_od_mm"])
    x_turn = float(sh["x_turn_in_flue_mm"])
    x_hdr = float(sh["x_smokebox_header_mm"])
    r_sh = 1.32

    up_yz = pv["five_flue_tubes"]["upper_superheater_flues"]["yz_centers_mm"]

    # 1. 锅顶 13-5 衬套 (-105.5, 0.0, 68.5) -> 圆弧扫掠炉内干汽管直通烟室 17-6
    crown_union = make_hex_nut((-105.5, 0.0, 68.5), (0.0, 0.0, 1.0), hex_r=2.35, length=2.0)
    pipe_13_9 = make_swept_tube(
        [(-105.5, 0.0, 68.5), (-103.2, 0.0, z_pipe), (x_hdr, 0.0, z_pipe)],
        r_tube=r_dry,
        bend_r=2.4,
    )

    # 2. 烟室 17-6 单空心螺栓过热集箱 (圆腰槽座 + 六角螺栓头，零方块弯头) + 17-5 双 U 四程圆弧弯管
    bolt_17_6 = (
        Pos(x_hdr, 0.0, 57.2) * Cylinder(radius=3.2, height=9.6)
        + make_hex_nut((x_hdr, 0.0, 62.2), (0.0, 0.0, 1.0), hex_r=4.2, length=2.4)
        + Pos(x_hdr, 0.0, 51.5) * Rot(0, 0, 90) * extrude(SlotOverall(width=28.8, height=6.8), amount=4.4)
    )
    sh_loops = None
    x_r1 = x_hdr - 1.5
    x_r2 = x_hdr + 1.5
    for yz in up_yz:
        yf, zf = float(yz[0]), float(yz[1])
        sgn = 1.0 if yf > 0 else -1.0
        y_a, z_a = yf - sgn * 1.75, zf + 1.75
        y_b, z_b = yf + sgn * 1.75, zf - 1.75
        u_loop = make_swept_tube(
            [
                (x_r1, y_a, 53.8),
                (x_r1, y_a, z_a),
                (x_turn, y_a, z_a),
                (x_turn, y_b, z_b),
                (x_r2, y_b, z_b),
                (x_r2, y_b, 53.8),
            ],
            r_tube=r_sh,
            bend_r=1.95,
        )
        sh_loops = u_loop if sh_loops is None else (sh_loops + u_loop)

    # 3. 17-4 六角活接 -> 13-4 圆弧下垂主蒸汽管 -> 7-9 圆柱四通 + 左右蒸汽室圆弧给汽管与 1-20 バンジョー接头
    tx_lub, _, tz_lub = [float(v) for v in br["lubricator_tee_xyz_mm"]]
    z_in_7_9 = 22.8
    y_in_7_9 = 29.5
    x_chest = float(br["x_chest_entry_mm"])

    pipe_13_4 = (
        make_swept_tube(
            [(x_hdr, 0.0, 52.2), (tx_lub, 0.0, 52.2), (tx_lub, 0.0, tz_lub - 0.8)],
            r_tube=1.72,
            bend_r=5.2,
        )
        + make_hex_nut((x_hdr + 5.0, 0.0, 52.2), (1.0, 0.0, 0.0), hex_r=2.6, length=2.8)
        + Pos(tx_lub, 0.0, 0.5 * (tz_lub + z_in_7_9)) * Cylinder(radius=2.6, height=(z_in_7_9 - tz_lub) + 3.8)
    )
    pipes_7_9 = None
    for sgn in (-1.0, 1.0):
        yi = sgn * y_in_7_9
        branch = make_swept_tube(
            [(tx_lub, 0.0, z_in_7_9), (tx_lub, yi, z_in_7_9), (x_chest, yi, z_in_7_9)],
            r_tube=1.25,
            bend_r=3.8,
        ) + make_banjo_eye((x_chest, yi, z_in_7_9), (0.0, 0.0, 1.0), r_eye=2.05, h_eye=2.3, hex_r=1.65)
        pipes_7_9 = branch if pipes_7_9 is None else (pipes_7_9 + branch)

    # 4. 7-8 排汽歧管 (原厂中央矩形铣削铜座 + 左右圆形排汽横管与圆柱法兰环) + 20-7 ø4.2mm 乏汽喷嘴
    x_ch = float(ej["chimney_axis_x_mm"])
    z_tip = float(ej["blast_nozzle_tip_z_mm"])
    r_orifice = 0.5 * float(ej["blast_nozzle_orifice_dia_mm"])
    z_exh = 12.0

    exh_7_8 = Pos(x_ch, 0.0, z_exh) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=67.6)
    for sgn in (-1.0, 1.0):
        exh_7_8 += Pos(x_ch, sgn * 20.0, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.45, height=2.4)
        exh_7_8 += Pos(x_ch, sgn * 33.4, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=1.4)

    blast_col = (
        Pos(x_ch, 0.0, z_exh) * Box(5.8, 6.2, 4.0)
        + Pos(x_ch, 0.0, 0.5 * (z_exh + z_tip)) * Cone(bottom_radius=4.3, top_radius=3.1, height=z_tip - z_exh)
    )
    blower_ring = Pos(x_ch, 0.0, z_tip - 1.2) * Cylinder(radius=4.7, height=1.8)
    blast_bore = Pos(x_ch, 0.0, z_tip - 4.0) * Cylinder(radius=r_orifice, height=10.0)

    # 5. 烟室隔板活接 (94.0, +6.2, 58.0) -> 17-3 圆弧下垂通风细管 (衔接外部 13-8 通风管，零方块！)
    x_tp_front = 96.4
    z_blw_low = 20.8
    blower_bulkhead_union = make_hex_nut((94.2, +6.2, 58.0), (1.0, 0.0, 0.0), hex_r=2.05, length=2.2)
    drop_17_3 = make_swept_tube(
        [
            (93.6, +6.2, 58.0),
            (x_tp_front, +6.2, 58.0),
            (x_tp_front, +6.2, z_blw_low),
            (x_ch - 6.5, +3.4, z_blw_low),
            (x_ch, +3.4, z_blw_low),
            (x_ch, +3.4, z_tip - 1.1),
        ],
        r_tube=1.02,
        bend_r=2.2,
    )

    # 6. 18-28 烟囱文丘里裙罩与圆柱支撑杆 (零方块！)
    z_pet = float(ej["petticoat_entry_z_mm"])
    r_pet_th = 0.5 * float(ej["petticoat_throat_dia_mm"])
    r_pet_fl = 0.5 * float(ej["petticoat_flare_dia_mm"])
    z_ch_top = 91.2
    petticoat_outer = (
        Pos(x_ch, 0, z_pet + 5.5) * Cone(bottom_radius=r_pet_fl, top_radius=r_pet_th + 1.0, height=11.0)
        + Pos(x_ch, 0, 0.5 * (z_pet + 10.5 + z_ch_top)) * Cylinder(radius=r_pet_th + 1.0, height=z_ch_top - (z_pet + 10.5))
        + Pos(x_ch, 0, z_ch_top - 0.8) * Cylinder(radius=r_pet_th + 2.1, height=1.6)
        + Pos(x_ch, 0, 74.0) * Cylinder(radius=r_pet_th + 2.6, height=2.2)
    )
    petticoat_inner = (
        Pos(x_ch, 0, z_pet + 5.5) * Cone(bottom_radius=r_pet_fl - 1.1, top_radius=r_pet_th, height=11.8)
        + Pos(x_ch, 0, 0.5 * (z_pet + 10.0 + z_ch_top + 2.0)) * Cylinder(radius=r_pet_th, height=(z_ch_top + 2.0) - (z_pet + 10.0))
    )
    pet_stay = make_swept_tube(
        [(x_hdr, 0.0, 61.5), (x_hdr, 0.0, 73.8), (x_ch - r_pet_th + 0.5, 0.0, 73.8)],
        r_tube=1.35,
        bend_r=3.2,
    )

    return (
        crown_union + pipe_13_9 + bolt_17_6 + sh_loops + pipe_13_4 + pipes_7_9
        + exh_7_8 + (blast_col + blower_ring - blast_bore)
        + blower_bulkhead_union + drop_17_3 + pet_stay + (petticoat_outer - petticoat_inner)
    )


def build_feedwater_piping(cfg: Dict, pip_cfg: Dict):
    """
    圆弧扫掠版车架内给水闭环 (|Y| <= 11.0mm, イラスト 4, 6, 8, 13):
      - 左侧 (+Y=+7.2mm) 6-11 共用给水嘴 -> 6-10 吸水管 -> 6-5 中央轴动泵；
      - 手摇泵与轴动泵吐出管 6-6/6-8 在右后床下汇入 4-8 支座内的 13-2/13-7 逆止阀总成 (-113.6, -6.2, +15.8)；
      - 经 13-11 前向炉底给水管进入火室前锅筒底部衬套 (-42.5, 0.0, +17.0)；
      - 分流经 8-11 地板下 4-17 旁通阀 (-124.0, -8.4, +14.8) -> 4-16 右侧 (-Y=-7.2mm) 回水尾嘴。
    """
    fw = pip_cfg["feedwater_piping_topology"]
    ax_loop = fw["axle_and_shared_hand_pump_loop"]
    x_pump = float(ax_loop["pump_x_mm"])
    x_tw = x_pump - 16.8  # -38.8 mm

    # 1. 6-5 中央轴动泵黄铜阀体 + 双 SB-ø3 止回阀塔与六角顶底闷盖
    pump_barrel = Pos(x_pump - 15.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.6, height=7.2)
    gland_nut = make_hex_nut((x_pump - 11.4, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=4.1, length=2.0)
    check_tower = Pos(x_tw, 0.0, 1.5) * Cylinder(radius=3.0, height=16.5)
    top_hex_cap = make_hex_nut((x_tw, 0.0, 10.4), (0.0, 0.0, 1.0), hex_r=3.7, length=2.2)
    bot_hex_cap = make_hex_nut((x_tw, 0.0, -7.4), (0.0, 0.0, 1.0), hex_r=3.7, length=2.0)
    pump_bracket = Pos(x_pump - 15.0, 0.0, 4.2) * Box(5.4, 21.0, 2.4)
    ram_bore = Pos(x_pump - 14.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.90, height=8.4)
    pump_6_5 = pump_barrel + gland_nut + check_tower + top_hex_cap + bot_hex_cap + pump_bracket - ram_bore

    # 2. 6-11 左后吸水/手摇汇流嘴 (-150.0, +7.2, -9.2) -> 6-10 圆弧扫掠吸水管入 6-5 底部
    pipe_6_10 = (
        make_swept_tube(
            [
                (-150.8, +7.2, -9.2),
                (-142.5, +7.2, -9.2),
                (-134.5, +7.2, +9.2),
                (x_tw, +7.2, +9.2),
                (x_tw, +7.2, -5.2),
                (x_tw, 0.0, -5.2),
            ],
            r_tube=1.22,
            bend_r=3.0,
        )
        + Pos(-149.2, +7.2, -9.2) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=3.8)
        + make_hex_nut((-147.2, +7.2, -9.2), (1.0, 0.0, 0.0), hex_r=2.35, length=2.0)
    )

    # 3. 31-5 手摇泵高压支路自 (+7.2, +9.2) 跨接至右侧 (-6.2, +10.2) 给水总管
    cy = -6.2
    shared_hp_cross = (
        make_swept_tube(
            [(-118.0, +7.2, +9.2), (-113.6, +7.2, +10.2), (-113.6, cy, +10.2)],
            r_tube=1.18,
            bend_r=2.2,
        )
        + Pos(-113.6, +1.0, +10.2) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=4.0)
    )

    # 4. 6-6 向后送水管 -> 4-8 支座内 13-2 逆止阀体 + 13-7 六角顶盖 (-113.6, -6.2, +15.8) -> 13-11 前向炉底入炉管至 (-42.5, 0.0, +17.0)
    pipe_6_6 = make_swept_tube(
        [(x_tw, 0.0, +10.2), (x_tw, cy, +10.2), (-113.6, cy, +10.2), (-113.6, cy, +15.8)],
        r_tube=1.20,
        bend_r=2.6,
    )
    clack_13_2_13_7 = (
        Pos(-113.6, cy, +15.4) * Cylinder(radius=2.65, height=6.4)
        + make_hex_nut((-113.6, cy, +18.8), (0.0, 0.0, 1.0), hex_r=2.95, length=1.8)
    )
    pipe_13_11_forward = (
        make_swept_tube(
            [
                (-113.6, cy, +15.4),
                (-48.0, cy, +15.4),
                (-42.5, 0.0, +15.4),
                (-42.5, 0.0, +17.2),
            ],
            r_tube=1.18,
            bend_r=2.6,
        )
        + make_banjo_eye((-42.5, 0.0, +16.8), (0.0, 0.0, 1.0), r_eye=2.1, h_eye=2.2, hex_r=1.65)
    )

    # 5. 4-17 驾驶室 8-11 地板下旁通阀 (-124.0, -8.4, +14.8) -> 4-16 圆弧回水管至右后嘴 (-150.0, -7.2, -9.2)
    bx, by, bz = -124.0, -8.4, 14.8
    bypass_branch = make_swept_tube(
        [(-113.6, cy, +13.6), (-118.0, by, +14.8), (bx, by, bz + 1.2)],
        r_tube=1.12,
        bend_r=2.0,
    )
    valve_4_17 = (
        Pos(bx, by, bz) * Cylinder(radius=2.25, height=6.2)
        + make_banjo_eye((bx, by, bz - 1.4), (1.0, 0.0, 0.0), r_eye=2.0, h_eye=2.2, hex_r=1.6)
    )
    pipe_4_16_return = (
        make_swept_tube(
            [
                (bx, by, bz - 1.4),
                (-129.0, -7.2, +9.0),
                (-136.0, -7.2, +9.0),
                (-143.0, -7.2, -9.2),
                (-150.8, -7.2, -9.2),
            ],
            r_tube=1.18,
            bend_r=2.8,
        )
        + Pos(-149.2, -7.2, -9.2) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=3.8)
        + make_hex_nut((-147.2, -7.2, -9.2), (1.0, 0.0, 0.0), hex_r=2.35, length=2.0)
    )

    return (
        pump_6_5 + pipe_6_10 + shared_hp_cross + pipe_6_6
        + clack_13_2_13_7 + pipe_13_11_forward + bypass_branch + valve_4_17 + pipe_4_16_return
    )


def build_displacement_lubricator(cfg: Dict, pip_cfg: Dict):
    lub = pip_cfg["displacement_lubricator_system"]
    x0, x1 = [float(v) for v in lub["x_range_mm"]]
    xm = 0.5 * (x0 + x1)
    zc = float(lub["z_center_mm"])
    r_tank = 0.5 * float(lub["tank_od_mm"])
    cap_x, _, cap_z = [float(v) for v in lub["cap_xyz_mm"]]
    tx, _, tz = [float(v) for v in lub["oil_delivery_tee_xyz_mm"]]

    tank_4_2 = Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_tank, height=x1 - x0)
    saddle_cradle = Pos(xm, 0.0, zc + 1.5) * Box(12.0, 16.4, 4.2)
    neck_4_1 = Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z)) * Cylinder(radius=2.8, height=cap_z - zc + 1.0)
    cap_4_1 = Pos(cap_x - 1.5, 0.0, cap_z) * Cylinder(radius=3.8, height=2.2)

    oil_pipe = make_swept_tube(
        [(x0 + 2.0, 0.0, zc + 3.2), (tx, 0.0, zc + 3.2), (tx, 0.0, tz + 0.6)],
        r_tube=1.10,
        bend_r=3.6,
    )
    union_nut = make_hex_nut((tx, 0.0, tz - 2.2), (0.0, 0.0, 1.0), hex_r=2.15, length=2.2)
    return tank_4_2 + saddle_cradle + neck_4_1 + cap_4_1 + oil_pipe + union_nut


def build_boiler_aux_valves(cfg: Dict, pip_cfg: Dict):
    """
    驾驶室原厂 1:1 全铜阀座与外管网总成 (イラスト 11, 13, 14, 29 & ill_p45.jpg):
      - 彻底废除驾驶室内部任何虚构 H 型支撑杆！所有后壁阀件经银焊在 11-9 背板内部的环形衬套座圈 (X=-112.8mm) 真实物理连通 (cc=1)；
      - 左驾 (+Y=+10.5mm): 11-4/11-5/11-6 水面计上下黄铜座 + 13-10/13-6 接口 (中部留空供独立透明硼硅玻璃管 11-7 桥接)；
      - 中上 (Y=0, Z=58.5mm): 11-15 分配タレット + 11-10 扇形卡齿板 + 13-9 外露S管入锅顶衬套 -> 28-6x2 双安全阀；
      - 左上 (+Y=+4.8mm, Z=58.5mm): 左后斜 35° 通风阀体 11-14 + 外衣夹层 13-8 通风管直抵烟室隔板 (94.0, +6.2, 58.0)；
      - 右上 (-Y=-11.8mm, Z=58.0mm): 13-1 下部 180° U 型冷凝虹吸弯管 + 上仰 38° 真实 ø20mm 压力表黄铜表圈；
      - 右中 (-Y=-12.5mm, Z=43.5mm): 11-1/11-2 汽笛阀 -> 13-3 汽笛管 -> 火室前锅筒底 11-8 长谐振真汽笛 (X∈[-19, +23], Y=-5.6, Z=+15.0)；
      - 右下 (-Y=-11.5mm, Z=24.8mm): 11-16 锅炉后壁右下部排污针阀座。
    """
    bom = pip_cfg["pumps_and_valves_bom"]
    aux = pip_cfg["superheater_and_steam_piping"]["auxiliary_steam_lines"]
    sv = bom["twin_safety_valves_28_6"]
    wg = bom["water_gauge_11_7"]
    wh = aux["functional_whistle"]
    bd = aux["backhead_blowdown"]

    x_ring = -112.8  # 嵌于 11-9 铸造背板 [-114.4, -112.0] 内部的银焊衬套法兰环

    # 1. 嵌于 11-9 背板内部的后封头银焊法兰环 (外观不可见，真实连通后壁各螺纹衬套，零驾驶室裸露假杆！)
    brazed_backhead_ring = Pos(x_ring, 0.0, 44.0) * Rot(0, 90, 0) * (
        Cylinder(radius=19.8, height=1.2) - Cylinder(radius=17.4, height=3.0)
    )

    # 2. 11-15 蒸汽分配タレット + 11-10 加減弁扇形卡齿板 + 13-9 外部 S 管 -> 28-6x2 双安全阀
    turret_11_15 = (
        Pos(-114.8, 0.0, 58.5) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=22.4)
        + Pos(-114.2, 0.0, 58.5) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=4.6)
        + make_hex_nut((-116.2, 0.0, 58.5), (1.0, 0.0, 0.0), hex_r=2.3, length=1.6)
        + Pos(0.5 * (x_ring - 114.8), 0.0, 59.8) * Box(abs(-114.8 - x_ring) + 1.4, 3.6, 4.4)
    )
    quadrant_11_10 = (
        Pos(-116.1, 0.0, 58.5) * Rot(0, 90, 0) * (Cylinder(radius=8.8, height=1.1) - Cylinder(radius=6.8, height=3.0))
        & Pos(-116.1, +4.6, 63.1) * Box(3.0, 9.6, 9.6)
    ) + Pos(-115.6, +7.6, 58.5) * Box(1.8, 2.2, 1.8) + Pos(-115.6, 0.0, 66.1) * Box(1.8, 1.8, 2.2)

    sv_x = float(sv["x_mm"])
    sv_1 = Pos(sv_x, +3.8, 76.2) * Cylinder(radius=2.3, height=9.2)
    sv_2 = Pos(sv_x, -3.8, 76.2) * Cylinder(radius=2.3, height=9.2)
    sv_base = Pos(sv_x, 0.0, 69.7) * Rot(0, 0, 90) * extrude(SlotOverall(width=12.6, height=5.6), amount=3.6)
    spipe_13_9_and_crown = (
        make_swept_tube(
            [(-114.8, 0.0, 60.5), (-112.2, 0.0, 68.5), (-105.5, 0.0, 68.5), (sv_x, 0.0, 69.8)],
            r_tube=1.35,
            bend_r=2.8,
        )
        + make_banjo_eye((-105.5, 0.0, 68.8), (0.0, 0.0, 1.0), r_eye=2.2, h_eye=2.2, hex_r=1.75)
    )

    # 3. 11-14 左后斜 35° 通风阀座 + 13-8 外衣夹层通风管直抵烟室隔板 (94.0, +6.2, 58.0)
    blower_boss_11_14 = (
        Pos(-115.4, +4.8, 58.5) * Rot(0, 0, -35) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=4.4)
        + Pos(-114.8, +4.8, 58.5) * Cylinder(radius=1.8, height=3.8)
    )
    pipe_13_8_ext = (
        make_swept_tube(
            [
                (-114.8, +4.8, 60.0),
                (-110.5, +6.2, 67.6),
                (89.5, +6.2, 67.6),
                (94.0, +6.2, 58.0),
            ],
            r_tube=1.02,
            bend_r=3.0,
        )
        + make_banjo_eye((94.0, +6.2, 58.0), (1.0, 0.0, 0.0), r_eye=1.95, h_eye=2.2, hex_r=1.55)
    )

    # 4. 左侧 (+Y=+10.5mm) 11-4/11-5/11-6 水面计上下黄铜座 (中间完全留空给透明玻璃管 cab_sight_glass_tube_vis!)
    yw = float(wg["y_mm"])  # +10.5 mm
    wg_top_11_5 = (
        Pos(-116.2, yw, 56.3) * Cylinder(radius=2.35, height=3.6)
        + make_hex_nut((-116.2, yw, 58.4), (0.0, 0.0, 1.0), hex_r=1.95, length=1.6)
        + make_swept_tube([(-114.8, yw, 58.5), (-116.2, yw, 56.8)], r_tube=1.15, bend_r=1.2)
        + Pos(-114.4, yw, 56.3) * Rot(0, 90, 0) * Cylinder(radius=1.65, height=3.8)
    )
    wg_bot_11_6 = (
        Pos(-116.2, yw, 38.7) * Cylinder(radius=2.35, height=3.8)
        + make_hex_nut((-116.2, yw, 36.6), (0.0, 0.0, 1.0), hex_r=1.85, length=1.4)
        + Pos(0.5 * (x_ring - 116.2), yw, 38.7) * Rot(0, 90, 0) * Cylinder(radius=1.55, height=abs(-116.2 - x_ring) + 1.2)
        + Pos(x_ring, yw + 4.2, 38.7) * Box(1.2, 7.6, 2.8)
    )

    # 5. 右上部 (-Y=-11.8mm, Z=+58.0mm) 13-1 下部 180° U 型冷凝虹吸弯管 + 上仰 38° 真实 ø20mm 压力表黄铜表圈
    yp, zp = -11.8, 58.0
    pg_bezel_outer = Pos(-120.0, yp, zp) * Rot(0, +38, 0) * Rot(0, 90, 0) * Cylinder(radius=9.8, height=3.0)
    pg_bezel_recess = Pos(-120.9, yp, zp + 0.7) * Rot(0, +38, 0) * Rot(0, 90, 0) * Cylinder(radius=8.45, height=2.2)
    pg_bezel = pg_bezel_outer - pg_bezel_recess
    siphon_13_1 = make_swept_tube(
        [
            (-114.8, -9.8, 58.5),
            (-115.6, yp, 55.0),
            (-115.6, yp, 46.8),
            (-119.2, yp, 46.8),
            (-119.6, yp, 57.4),
        ],
        r_tube=1.05,
        bend_r=1.45,
    ) + Pos(-119.5, yp, 56.2) * Cylinder(radius=1.85, height=3.8)

    # 6. 右中 (-Y=-12.5mm, Z=43.5mm) 11-1/2 汽笛阀 -> 13-3 汽笛管 -> 锅筒前下 11-8 长谐振真汽笛
    wy_v, wz_v = -12.5, 43.5
    wx, wy, wz = [float(v) for v in wh["whistle_xyz_mm"]]  # (2.0, -5.6, 15.0)
    wh_valve_11_2 = (
        Pos(-115.0, wy_v, wz_v) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=4.8)
        + make_hex_nut((-116.4, wy_v, wz_v), (1.0, 0.0, 0.0), hex_r=1.95, length=1.4)
        + Pos(x_ring, wy_v - 3.2, wz_v) * Box(1.2, 6.8, 2.8)
    )
    pipe_13_3 = make_swept_tube(
        [
            (-114.2, wy_v, wz_v),
            (-111.5, -9.2, 22.0),
            (-104.0, wy, wz),
            (-19.0, wy, wz),
        ],
        r_tube=1.02,
        bend_r=3.2,
    )
    whistle_11_8 = (
        Pos(wx, wy, wz) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=42.0)
        + Pos(-17.5, wy, wz) * Rot(0, 90, 0) * Cylinder(radius=3.4, height=3.6)
        - Pos(-12.5, wy, wz - 2.4) * Box(3.2, 4.8, 3.2)
    )

    # 7. 右下 (-Y=-11.5mm, Z=24.8mm) 11-16 锅炉排污针阀座
    bdx, bdy, bdz = [float(v) for v in bd["xyz_mm"]]
    blowdown_11_16 = (
        Pos(bdx, bdy, bdz) * Rot(0, 90, 0) * Cylinder(radius=2.25, height=4.6)
        + make_hex_nut((bdx - 1.6, bdy, bdz), (1.0, 0.0, 0.0), hex_r=2.05, length=1.4)
        + Pos(bdx - 0.4, bdy, bdz - 2.4) * Cylinder(radius=1.10, height=4.4)
        + Pos(x_ring, bdy, bdz + 3.2) * Box(1.2, 3.6, 7.6)
    )

    return (
        brazed_backhead_ring + turret_11_15 + quadrant_11_10 + sv_1 + sv_2 + sv_base
        + spipe_13_9_and_crown + blower_boss_11_14 + pipe_13_8_ext
        + wg_top_11_5 + wg_bot_11_6 + pg_bezel + siphon_13_1
        + wh_valve_11_2 + pipe_13_3 + whistle_11_8 + blowdown_11_16
    )


def build_alcohol_burner(cfg: Dict, pip_cfg: Dict):
    """
    23-1 火室底纵向中央 1x3 直列三头立式 10mm 燃烧器 + 23-2 托架 (イラスト 23, 100% 中心对称!):
      - 3 只灯芯杯沿 Y=0.0mm 纵向直列于 X = [-93.0, -77.0, -61.0] mm；
      - 后部中央供醇圆管平滑圆弧下弯至牵引杆上方 X=-150.0, Y=0.0, Z=+3.6mm 接 ST-ø7 半透明硅胶软管！
    """
    ab = pip_cfg["alcohol_burner_system"]
    wick_xs = [float(x) for x in ab["wick_x_mm"]]
    r_cup_out = 0.5 * float(ab["wick_cup_od_mm"])
    r_wick = 0.5 * float(ab["wick_dia_mm"])
    z_top = float(ab["wick_top_z_mm"])
    r_tube = 0.5 * float(ab["supply_tube_od_mm"])

    xm = 0.5 * (wick_xs[0] + wick_xs[-1])
    holder_23_2 = Pos(xm, 0.0, 12.2) * Box((wick_xs[-1] - wick_xs[0]) + 15.0, 18.0, 2.6)
    rear_lug = Pos(wick_xs[0] - 9.0, 0.0, 13.0) * Box(6.0, 15.0, 3.8)

    cups = None
    for wx in wick_xs:
        cup_shell = Pos(wx, 0.0, 0.5 * (11.5 + z_top)) * Cylinder(radius=r_cup_out, height=z_top - 11.5)
        cup_well = Pos(wx, 0.0, z_top - 2.0) * Cylinder(radius=r_wick, height=4.6)
        wick_core = Pos(wx, 0.0, z_top - 2.0) * Cylinder(radius=r_wick - 0.8, height=5.8)
        one = (cup_shell - cup_well) + wick_core
        cups = one if cups is None else (cups + one)

    feed_pipe = (
        make_swept_tube(
            [
                (wick_xs[-1], 0.0, 11.8),
                (-136.0, 0.0, 11.8),
                (-143.0, 0.0, +3.6),
                (-150.8, 0.0, +3.6),
            ],
            r_tube=r_tube,
            bend_r=3.0,
        )
        + Pos(-149.2, 0.0, +3.6) * Rot(0, 90, 0) * Cylinder(radius=2.55, height=3.8)
    )
    return holder_23_2 + rear_lug + cups + feed_pipe


def build_tender_body(cfg: Dict):
    """
    8-20B 四轴煤水车外衣壳体 (イラスト 32, 33, 35 — 100% Y 对称, |min_y+max_y|=0, com_y=0):
      - 32-10 台枠 + 32-3 前梁中央燃料ホースガイド环；
      - 35-1 水槽上板中央后部 48x35mm (X∈[-58,-10], Y∈[-17.5,+17.5]) 常时敞口加水/观水湾 + 35-5x2 左右防波挡板；
      - 后甲板 X=-85.5mm 圆形检修口 (配 35-6 铰接盖)；前部石炭槽下沉嵌放 36-4 酒精箱并由 36-1/2 石炭盖遮蔽。
    """
    t_cfg = cfg["tender_and_track"]
    L, W, H = float(t_cfg["tender_body_length"]), float(t_cfg["tender_body_width"]), float(t_cfg["tender_body_height"])
    half_span = 0.5 * float(t_cfg["bogie_center_span"])

    base_plate = Pos(0, 0, 2.4) * Box(L + 4.0, W + 2.0, 2.0)
    front_beam = Pos(0.5 * L + 1.2, 0, 0.5) * Box(3.6, W + 2.0, 6.5)
    rear_beam = Pos(-0.5 * L - 1.2, 0, 0.5) * Box(3.6, W + 2.0, 6.5)
    drawbar_lug = Pos(0.5 * L + 2.6, 0, -1.9) * Box(5.2, 12.0, 4.2) - Pos(0.5 * L + 3.4, 0, -1.9) * Box(4.2, 7.2, 2.6)
    hose_guide_32_3 = (
        Pos(0.5 * L + 2.2, 0.0, +7.6) * Rot(0, 90, 0) * (Cylinder(radius=4.2, height=1.6) - Cylinder(radius=3.1, height=4.0))
        + Pos(0.5 * L + 1.8, 0.0, +4.2) * Box(2.0, 3.6, 3.2)
    )

    tank_outer = Pos(-4.0, 0, 3.0 + 0.5 * H) * Box(L - 8.0, W, H)
    tank_inner_hollow = Pos(-4.0, 0, 12.0 + 0.5 * (H - 11.0)) * Box(L - 12.0, W - 3.6, H - 11.0)
    coal_bunker_cut = Pos(50.0, 0, 3.0 + H - 7.0) * Box(108.0, W - 14.0, 16.0)
    open_bay_35_1_cut = Pos(-34.0, 0.0, 3.0 + H - 2.0) * Box(48.0, 35.0, 12.0)
    tank_shell = tank_outer - tank_inner_hollow - coal_bunker_cut - open_bay_35_1_cut

    bunker_and_coaming = None
    for sgn in (-1.0, 1.0):
        wall = Pos(48.0, sgn * (0.5 * W - 6.5), 3.0 + H + 3.2) * Box(112.0, 1.6, 8.2)
        fence_35_5 = Pos(-34.0, sgn * 18.2, 3.0 + H + 2.6) * Box(50.0, 1.4, 5.4)
        sub = wall + fence_35_5
        bunker_and_coaming = sub if bunker_and_coaming is None else (bunker_and_coaming + sub)
    bay_lips = (
        Pos(-58.6, 0.0, 3.0 + H + 1.2) * Box(1.4, 37.8, 2.6)
        + Pos(-9.4, 0.0, 3.0 + H + 1.2) * Box(1.4, 37.8, 2.6)
    )

    bogie_wells = None
    for bx in (-half_span, +half_span):
        bolster_rel = Pos(bx, 0, -0.5) * Box(24.0, 66.0, 5.0)
        sub = bolster_rel
        for sgn in (-1.0, 1.0):
            wheel_arch = Pos(bx, sgn * 23.2, 5.5) * Box(86.0, 9.6, 16.6)
            side_rel = Pos(bx, sgn * 29.8, 0.5) * Box(76.0, 11.0, 6.6)
            sub += wheel_arch + side_rel
        bogie_wells = sub if bogie_wells is None else (bogie_wells + sub)

    rear_coaming = Pos(-85.5, 0, 3.0 + H + 0.6) * Cylinder(radius=9.2, height=2.0)
    rear_hole = Pos(-85.5, 0, 3.0 + H - 2.0) * Cylinder(radius=7.0, height=10.0)
    return (
        base_plate + front_beam + rear_beam + drawbar_lug + hose_guide_32_3
        + tank_shell + bunker_and_coaming + bay_lips + rear_coaming
        - bogie_wells - rear_hole
    )


def build_tender_internals(cfg: Dict):
    """
    圆弧扫掠版煤水车 1:1 内部真值总成 (イラスト 31, 34, 36, 37):
      - 36-4 前部 130ml 密闭马氏瓶酒精箱 (36-5 气密注醇口 X=+28, 36-6 针阀座 X=+68) -> 台枠中央下部 34-2 恒液位サンプ (X=+10) + 斜切补气管 + 130mm 前出供醇管至 (+113.2, 0.0, +7.6)；
      - 31-5 水槽底浸渍横型 ø7x16mm 手摇泵缸体 (-38, +8, +14) + M6 双 SB-ø4 止回阀塔 -> 31-13 铜管 -> 31-12 右下隔壁 -> 34-6 床下水管 + 4-15 バンジョー -> (+113.2, -7.2, -5.1)；
      - 左下 (+113.2, +7.2, -5.1) -> 34-6 床下回水管 -> 31-14 敞口湾左前高位 50° 鹅颈弯头防虹吸回水立管 (喷嘴位于 (-21.5, +5.5, +46.8)，回水柱肉眼直视！)；
      - 31-9 敞口湾右底排水衬套 (-38, -8, +7.5)，配合 31-11 上部提拉放水塞杆实现底部重力放水！
    """
    floor_spine = Pos(8.0, 0.0, 5.8) * Box(208.0, 20.0, 2.4)
    sump_34_2 = (
        Pos(10.0, 0.0, 0.6) * Cylinder(radius=5.2, height=11.6)
        + Pos(10.0, +4.2, 3.8) * Cylinder(radius=1.4, height=5.0)
        + Pos(10.0, -4.2, 3.8) * Cylinder(radius=1.4, height=5.0)
    )
    drawbar_clevis_31_2 = (
        Pos(109.5, 0.0, -1.9) * Box(17.5, 10.4, 4.6)
        - Pos(115.2, 0.0, -1.9) * Box(7.5, 6.8, 3.0)
        + Pos(116.0, 0.0, -1.9) * Cylinder(radius=1.8, height=5.2)
    )

    alc_outer = Pos(54.0, 0.0, 29.0) * Box(80.0, 36.0, 44.5)
    alc_hollow = Pos(54.0, 0.0, 29.0) * Box(77.4, 33.4, 41.9)
    filler_neck_36_5 = Pos(28.0, 0.0, 52.2) * Cylinder(radius=4.2, height=3.8)
    filler_bore = Pos(28.0, 0.0, 52.0) * Cylinder(radius=2.55, height=8.0)
    alc_tank_36_4 = (alc_outer + filler_neck_36_5) - alc_hollow - filler_bore
    valve_boss_36_6 = Pos(68.0, 0.0, 52.8) * Cylinder(radius=2.5, height=4.4)
    mariotte_tubes = (
        make_swept_tube([(68.0, +1.8, 8.0), (10.0, +1.8, 8.0), (10.0, +1.8, 0.5)], r_tube=1.45, bend_r=2.8)
        + make_swept_tube([(28.0, -1.8, 12.0), (10.0, -1.8, 8.0), (10.0, -1.8, 1.8)], r_tube=1.35, bend_r=2.8)
    )

    # 34-2 中央サンプ -> 130mm 前向供醇管至牵引杆上方中央嘴 (+113.2, 0.0, +7.7)
    fuel_barb_34_2 = (
        make_swept_tube([(10.0, 0.0, +2.2), (18.0, 0.0, +7.7), (114.2, 0.0, +7.7)], r_tube=2.10, bend_r=3.6)
        + Pos(112.0, 0.0, +7.7) * Rot(0, 90, 0) * Cylinder(radius=2.95, height=2.2)
    )

    # 31-5 水槽底浸渍横型プランジャ手摇泵缸体 (-38, +8, +14) + M6 双止回阀塔 + 摇臂支耳 (-20, +8)
    hand_pump_31_5 = (
        Pos(-39.5, +8.0, 14.0) * Rot(0, 90, 0) * Cylinder(radius=5.2, height=19.0)
        + Pos(-39.5, +8.0, 8.8) * Box(20.0, 11.4, 4.6)
        + Pos(-46.5, +8.0, 19.5) * Cylinder(radius=3.8, height=12.5)
        + make_hex_nut((-46.5, +8.0, 26.2), (0.0, 0.0, 1.0), hex_r=4.2, length=2.2)
        + Pos(-20.0, +8.0, 10.2) * Box(5.6, 7.6, 7.2)
        - Pos(-34.0, +8.0, 14.0) * Rot(0, 90, 0) * Cylinder(radius=3.55, height=10.0)
    )
    pipe_31_13_and_34_6 = (
        make_swept_tube(
            [
                (-46.5, +8.0, +22.0),
                (-46.5, -7.2, +22.0),
                (-46.5, -7.2, +7.2),
                (98.0, -7.2, +7.2),
                (106.5, -7.2, -5.1),
                (114.2, -7.2, -5.1),
            ],
            r_tube=1.32,
            bend_r=2.8,
        )
        + make_banjo_eye((98.0, -7.2, +7.2), (0.0, 1.0, 0.0), r_eye=2.1, h_eye=2.4, hex_r=1.65)
        + Pos(112.0, -7.2, -5.1) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.6)
    )

    # 左下 (+Y=+7.2, z=-5.1) 34-6 回水管 -> 31-14 敞口湾左前高位 50° 鹅颈防虹吸回水立管！
    return_standpipe_31_14 = (
        make_swept_tube(
            [
                (114.2, +7.2, -5.1),
                (106.5, +7.2, -5.1),
                (98.0, +7.2, +7.2),
                (-14.5, +7.2, +7.2),
                (-14.5, +6.5, +49.2),
                (-18.5, +5.8, +49.2),
                (-21.5, +5.5, +46.2),
            ],
            r_tube=1.32,
            bend_r=2.4,
        )
        + make_banjo_eye((98.0, +7.2, +7.2), (0.0, 1.0, 0.0), r_eye=2.1, h_eye=2.4, hex_r=1.65)
        + Pos(112.0, +7.2, -5.1) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.6)
    )

    # 31-9 敞口湾右底排水衬套 (-38, -8) (同轴迎接 31-11 提拉放水塞)
    drain_bush_31_9 = (
        Pos(-38.0, -8.0, 8.2) * Cylinder(radius=3.8, height=4.4)
        + make_hex_nut((-38.0, -8.0, 10.2), (0.0, 0.0, 1.0), hex_r=3.6, length=1.6)
    )
    lower_hose_bracket = Pos(109.5, 0.0, -2.5) * Box(3.2, 18.2, 7.6)

    return (
        floor_spine + sump_34_2 + drawbar_clevis_31_2
        + alc_tank_36_4 + valve_boss_36_6 + mariotte_tubes + fuel_barb_34_2
        + hand_pump_31_5 + pipe_31_13_and_34_6 + return_standpipe_31_14
        + drain_bush_31_9 + lower_hose_bracket
    )


def build_tender_bogie_frame(cfg: Dict):
    """イラスト 31: 31-26 铸铁侧架 + 31-27 前后三椭圆减重孔端梁 + 31-29 枕梁 + 31-25 中央板簧束 (100% Y 对称)。"""
    wb = float(cfg["tender_and_track"]["bogie_wheelbase"])
    half_wb = 0.5 * wb
    y_frame = 0.5 * float(cfg["tender_and_track"].get("track_gauge", cfg["meta"]["track_gauge"])) + 5.2

    bolster = Pos(0, 0, 2.0) * Box(14.0, 2.0 * y_frame + 1.0, 5.5)
    end_beams_31_27 = None
    for ex_sgn in (-1.0, 1.0):
        ex = ex_sgn * (half_wb + 7.8)
        eb = Pos(ex, 0.0, 1.5) * Box(1.8, 2.0 * y_frame + 1.6, 5.0)
        for yh in (-11.0, 0.0, +11.0):
            eb -= Pos(ex, yh, 1.5) * Rot(0, 90, 0) * Rot(0, 0, 90) * extrude(SlotOverall(width=5.8, height=2.4), amount=4.0, both=True)
        end_beams_31_27 = eb if end_beams_31_27 is None else (end_beams_31_27 + eb)

    sides = None
    for sgn in (-1.0, 1.0):
        yf = sgn * y_frame
        leaf_31_25 = (
            Pos(0, sgn * (y_frame + 1.3), -1.2) * Box(21.0, 2.3, 3.8)
            + Pos(0, sgn * (y_frame + 1.6), -2.6) * Box(14.0, 2.3, 1.6)
            + Pos(0, sgn * (y_frame + 1.5), -1.2) * Box(3.2, 2.8, 4.4)
        )
        side_part = Pos(0, yf, 1.5) * Box(wb + 17.4, 2.4, 5.5) + leaf_31_25
        for ax_x in (-half_wb, +half_wb):
            side_part += Pos(ax_x, sgn * (y_frame + 1.4), 0.0) * Box(7.5, 4.2, 8.2)
            side_part += Pos(ax_x - 4.8, sgn * (y_frame + 1.0), 2.5) * Cylinder(radius=1.6, height=5.5)
            side_part += Pos(ax_x + 4.8, sgn * (y_frame + 1.0), 2.5) * Cylinder(radius=1.6, height=5.5)
            side_part -= Pos(ax_x, yf, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=10.0)
        sides = side_part if sides is None else (sides + side_part)
    return bolster + end_beams_31_27 + sides


def build_spoked_truck_wheel(cfg: Dict, is_left: bool = True):
    r_tread = 0.5 * float(cfg["wheels_and_axles"]["leading_wheel_diameter"])
    w_tire = 3.4
    rim = (
        Cylinder(radius=r_tread, height=w_tire)
        + Pos(0, 0, -0.5 * w_tire + 0.45) * Cylinder(radius=r_tread + 1.3, height=0.9)
        - Cylinder(radius=r_tread - 2.0, height=w_tire + 2.0)
    )
    disc = Cylinder(radius=r_tread - 1.7, height=1.8)
    for ang_deg in range(0, 360, 45):
        ang = math.radians(ang_deg)
        disc -= Pos(7.2 * math.cos(ang), 7.2 * math.sin(ang), 0) * Cylinder(radius=1.45, height=4.0)

    wheel = rim + disc + Cylinder(radius=3.5, height=w_tire + 0.4) - Cylinder(radius=1.8, height=w_tire + 6.0)
    wheel_mj = Rot(-90, 0, 0) * wheel
    return wheel_mj if is_left else mirror(wheel_mj, about=Plane.XZ)


def subdivide_stl_x_span(stl_path: Path, max_dx: float = 1.85, z_min_filter: float = 52.0) -> None:
    raw = stl_path.read_bytes()
    if len(raw) < 84:
        return
    header = raw[:80]
    n_tri = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + n_tri * 50:
        return
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(raw, dtype=dt, offset=84, count=n_tri)["v"].astype(np.float64)

    pts = tris.reshape(-1, 3)
    q = np.round(pts * 1e5).astype(np.int64)
    _, first_idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    V = pts[first_idx].copy()
    F = inv.reshape(-1, 3)

    for _ in range(8):
        e01 = np.sort(F[:, [0, 1]], axis=1)
        e12 = np.sort(F[:, [1, 2]], axis=1)
        e20 = np.sort(F[:, [2, 0]], axis=1)
        all_edges = np.unique(np.vstack([e01, e12, e20]), axis=0)
        u, v = all_edges[:, 0], all_edges[:, 1]
        mask = (
            (np.abs(V[u, 0] - V[v, 0]) > max_dx)
            & (np.maximum(V[u, 2], V[v, 2]) > z_min_filter)
            & (np.maximum(np.abs(V[u, 1]), np.abs(V[v, 1])) < 16.0)
        )
        split_edges = all_edges[mask]
        if len(split_edges) == 0:
            break

        mids = 0.5 * (V[split_edges[:, 0]] + V[split_edges[:, 1]])
        base_idx = len(V)
        V = np.vstack([V, mids])
        edge_to_mid: Dict[Tuple[int, int], int] = {
            (int(eu), int(ev)): base_idx + i for i, (eu, ev) in enumerate(split_edges)
        }

        new_F: List[Tuple[int, int, int]] = []
        for a, b, c in F.tolist():
            mab = edge_to_mid.get((min(a, b), max(a, b)), -1)
            mbc = edge_to_mid.get((min(b, c), max(b, c)), -1)
            mca = edge_to_mid.get((min(c, a), max(c, a)), -1)
            cnt = (mab >= 0) + (mbc >= 0) + (mca >= 0)
            if cnt == 0:
                new_F.append((a, b, c))
            elif cnt == 1:
                if mbc >= 0:
                    a, b, c, mab = b, c, a, mbc
                elif mca >= 0:
                    a, b, c, mab = c, a, b, mca
                new_F.append((a, mab, c))
                new_F.append((mab, b, c))
            elif cnt == 2:
                if mca < 0:
                    p0, p1, p2, m01, m12 = a, b, c, mab, mbc
                elif mab < 0:
                    p0, p1, p2, m01, m12 = b, c, a, mbc, mca
                else:
                    p0, p1, p2, m01, m12 = c, a, b, mca, mab
                new_F.append((m01, p1, m12))
                if abs(V[p0, 0] - V[m12, 0]) <= abs(V[p2, 0] - V[m01, 0]):
                    new_F.append((p0, m01, m12))
                    new_F.append((p0, m12, p2))
                else:
                    new_F.append((p0, m01, p2))
                    new_F.append((m01, m12, p2))
            else:
                new_F.append((a, mab, mca))
                new_F.append((mab, b, mbc))
                new_F.append((mca, mbc, c))
                new_F.append((mab, mbc, mca))
        F = np.asarray(new_F, dtype=np.int64)

    out_tris = V[F]
    v0, v1, v2 = out_tris[:, 0], out_tris[:, 1], out_tris[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.maximum(nlen, 1e-15))

    out_arr = np.zeros(len(out_tris), dtype=dt)
    out_arr["n"] = nrm.astype(np.float32)
    out_arr["v"] = out_tris.astype(np.float32)
    with stl_path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(out_tris)))
        f.write(out_arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Aster D51 P2 水密圆弧管路、三软管、驾驶室与煤水车 CAD 建模与惯量积分器")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--piping-config", type=Path, default=Path("configs/d51_boiler_piping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/meshes"))
    parser.add_argument("--mass-out", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--standard-stack", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pip_cfg = json.loads(args.piping_config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    tasks: List[Tuple[str, float, float, Callable[[], Any]]] = [
        ("driving_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, False)),
        ("driving_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, False)),
        ("main_driving_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, True)),
        ("main_driving_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, True)),
        ("return_crank_L", 0.018, RHO_STAINLESS, lambda: build_return_crank(cfg, True)),
        ("return_crank_R", 0.018, RHO_STAINLESS, lambda: build_return_crank(cfg, False)),
        ("main_rod", 0.02, RHO_STAINLESS, lambda: build_main_rod(cfg)),
        ("coupling_rod", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_main(cfg)),
        ("coupling_rod_front", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, True)),
        ("coupling_rod_rear", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, False)),
        ("eccentric_rod_L", 0.018, RHO_STAINLESS, lambda: build_eccentric_rod(cfg, True)),
        ("eccentric_rod_R", 0.018, RHO_STAINLESS, lambda: build_eccentric_rod(cfg, False)),
        ("expansion_link_L", 0.018, RHO_STAINLESS, lambda: build_expansion_link(cfg, True)),
        ("expansion_link_R", 0.018, RHO_STAINLESS, lambda: build_expansion_link(cfg, False)),
        ("die_block_L", 0.018, RHO_BRASS, lambda: build_die_block(cfg, True)),
        ("die_block_R", 0.018, RHO_BRASS, lambda: build_die_block(cfg, False)),
        ("radius_rod_L", 0.018, RHO_STAINLESS, lambda: build_radius_rod(cfg, True)),
        ("radius_rod_R", 0.018, RHO_STAINLESS, lambda: build_radius_rod(cfg, False)),
        ("combination_lever", 0.018, RHO_STAINLESS, lambda: build_combination_lever(cfg)),
        ("union_link_L", 0.018, RHO_STAINLESS, lambda: build_union_link(cfg, True)),
        ("union_link_R", 0.018, RHO_STAINLESS, lambda: build_union_link(cfg, False)),
        ("lift_link_L", 0.018, RHO_STAINLESS, lambda: build_lift_link(cfg, True)),
        ("lift_link_R", 0.018, RHO_STAINLESS, lambda: build_lift_link(cfg, False)),
        ("motion_bracket_L", 0.03, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, True)),
        ("motion_bracket_R", 0.03, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, False)),
        ("slide_bar_L", 0.025, RHO_STAINLESS, lambda: build_slide_bar(cfg, True)),
        ("slide_bar_R", 0.025, RHO_STAINLESS, lambda: build_slide_bar(cfg, False)),
        ("piston_tail_sheath_L", 0.025, RHO_STAINLESS, lambda: build_piston_tail_sheath(cfg, True)),
        ("piston_tail_sheath_R", 0.025, RHO_STAINLESS, lambda: build_piston_tail_sheath(cfg, False)),
        ("valve_spindle_sheath_L", 0.025, RHO_STAINLESS, lambda: build_valve_spindle_sheath(cfg, True)),
        ("valve_spindle_sheath_R", 0.025, RHO_STAINLESS, lambda: build_valve_spindle_sheath(cfg, False)),
        ("cab_reach_rod", 0.025, RHO_STAINLESS, lambda: build_cab_reach_rod(cfg)),
        ("drain_cock_linkage", 0.025, RHO_BRASS, lambda: build_drain_cock_linkage(cfg)),
        ("drain_pull_rod", 0.035, RHO_STAINLESS, lambda: build_drain_pull_rod(cfg)),
        ("cab_backhead_plate", 0.04, RHO_CAST_IRON, build_cab_backhead_plate),
        ("cab_deck_and_mounts", 0.04, RHO_BRASS, build_cab_deck_and_mounts),
        ("cab_reverser_stand", 0.03, RHO_BRASS, build_cab_reverser_stand),
        ("cab_reverser_wheel", 0.025, RHO_STAINLESS, build_cab_reverser_wheel),
        ("cab_regulator_lever", 0.025, RHO_BRASS, build_cab_regulator_lever),
        ("cab_blower_wheel", 0.025, RHO_BRASS, build_cab_blower_wheel),
        ("cab_whistle_lever", 0.025, RHO_BRASS, build_cab_whistle_lever),
        ("cab_sight_glass_tube", 0.02, RHO_PYREX_GLASS, build_cab_sight_glass_tube),
        ("cab_sight_drain", 0.02, RHO_BRASS, build_cab_sight_drain),
        ("cab_pg_dial", 0.025, RHO_GAUGE_DIAL, build_cab_pg_dial),
        ("cab_pressure_needle", 0.02, RHO_STAINLESS, build_cab_pressure_needle),
        ("cab_blowdown_screw", 0.02, RHO_BRASS, build_cab_blowdown_screw),
        ("cab_firebox_door", 0.025, RHO_BRASS, build_cab_firebox_door),
        ("cab_bypass_lever", 0.025, RHO_BRASS, build_cab_bypass_lever),
        ("cab_roof_hatch", 0.045, RHO_BRASS, build_cab_roof_hatch),
        ("tender_handpump_lever", 0.03, RHO_BRASS, build_tender_handpump_lever),
        ("tender_handpump_ram", 0.025, RHO_BRASS, build_tender_handpump_ram),
        ("tender_drain_plug", 0.025, RHO_BRASS, build_tender_drain_plug),
        ("tender_coal_cover", 0.05, RHO_BRASS, build_tender_coal_cover),
        ("tender_water_lid", 0.03, RHO_BRASS, build_tender_water_lid),
        ("tender_fuel_valve", 0.025, RHO_BRASS, build_tender_fuel_valve),
        ("tender_fuel_cap", 0.025, RHO_BRASS, build_tender_fuel_cap),
        ("umb_hose_fuel", 0.03, RHO_SILICONE, lambda: build_umb_hose_fuel(cfg, False)),
        ("umb_hose_fuel_r2m", 0.03, RHO_SILICONE, lambda: build_umb_hose_fuel(cfg, True)),
        ("umb_hose_feed", 0.03, RHO_VITON, lambda: build_umb_hose_feed(cfg, False)),
        ("umb_hose_feed_r2m", 0.03, RHO_VITON, lambda: build_umb_hose_feed(cfg, True)),
        ("umb_union_34_8", 0.025, RHO_BRASS, lambda: build_umb_union_34_8(cfg, False)),
        ("umb_union_34_8_r2m", 0.025, RHO_BRASS, lambda: build_umb_union_34_8(cfg, True)),
        ("umb_hose_return", 0.03, RHO_VITON, lambda: build_umb_hose_return(cfg, False)),
        ("umb_hose_return_r2m", 0.03, RHO_VITON, lambda: build_umb_hose_return(cfg, True)),
        ("chassis_equalizer_brakes", 0.035, RHO_CAST_IRON, lambda: build_chassis_equalizer_brakes(cfg)),
        ("axle_pump_eccentric", 0.02, RHO_BRASS, lambda: build_axle_pump_eccentric(cfg)),
        ("leading_truck_frame", 0.03, RHO_CAST_IRON, lambda: build_leading_truck_frame(cfg)),
        ("trailing_truck_frame", 0.03, RHO_CAST_IRON, lambda: build_trailing_truck_frame(cfg)),
        ("piston_crosshead_L", 0.025, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, True)),
        ("piston_crosshead_R", 0.025, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, False)),
        ("slide_valve_L", 0.02, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, True)),
        ("slide_valve_R", 0.02, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, False)),
        ("reverser_shaft", 0.025, RHO_STAINLESS, lambda: build_reverser_shaft(cfg)),
        ("driving_axlebox", 0.03, RHO_BRASS, lambda: build_driving_axlebox(cfg)),
        ("tender_axlebox", 0.03, RHO_CAST_IRON, lambda: build_tender_axlebox(cfg)),
        ("cylinder_block_L", 0.04, RHO_BRASS, lambda: build_cylinder_block(cfg, True)),
        ("cylinder_block_R", 0.04, RHO_BRASS, lambda: build_cylinder_block(cfg, False)),
        ("boiler_superstructure", 0.075, RHO_BRASS, lambda: build_boiler_superstructure(cfg, not args.standard_stack)),
        ("boiler_firebox_flues", 0.045, RHO_COPPER, lambda: build_boiler_firebox_flues(cfg, pip_cfg)),
        ("superheater_piping", 0.045, RHO_STAINLESS, lambda: build_superheater_piping(cfg, pip_cfg)),
        ("feedwater_piping", 0.04, RHO_BRASS, lambda: build_feedwater_piping(cfg, pip_cfg)),
        ("displacement_lubricator", 0.035, RHO_BRASS, lambda: build_displacement_lubricator(cfg, pip_cfg)),
        ("boiler_aux_valves", 0.04, RHO_BRASS, lambda: build_boiler_aux_valves(cfg, pip_cfg)),
        ("alcohol_burner", 0.035, RHO_BURNER_ASSY, lambda: build_alcohol_burner(cfg, pip_cfg)),
        ("tender_body", 0.08, RHO_BRASS, lambda: build_tender_body(cfg)),
        ("tender_internals", 0.055, RHO_BRASS, lambda: build_tender_internals(cfg)),
        ("tender_bogie_frame", 0.04, RHO_CAST_IRON, lambda: build_tender_bogie_frame(cfg)),
        ("truck_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, True)),
        ("truck_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, False)),
    ]

    expected_stls = {f"{k}.stl" for k, _, _, _ in tasks}
    for old_stl in args.out_dir.glob("*.stl"):
        if old_stl.name not in expected_stls:
            old_stl.unlink()

    mass_db: Dict[str, Any] = {
        "meta": {
            "stage": "P2-C~G (V2-SWEPT-TUBE-CAB-TENDER-AI)",
            "units": "SI (kg, m, kg*m^2)",
            "densities_kg_m3": {
                "copper_C1100": RHO_COPPER,
                "brass_BS": RHO_BRASS,
                "stainless_SUS": RHO_STAINLESS,
                "cast_iron_FC": RHO_CAST_IRON,
                "silicone_ST": RHO_SILICONE,
                "viton_BT": RHO_VITON,
            },
        },
        "parts": {},
    }

    for idx, (key, tol, rho, builder) in enumerate(tasks, start=1):
        stl_path = args.out_dir / f"{key}.stl"
        print(f"[CAD+MASS] {idx:02d}/{len(tasks)} 构建并积分 {key}.stl (ρ={rho:.0f} kg/m³) ...")
        solid = builder()
        if key in ("expansion_link_L", "expansion_link_R", "die_block_L", "die_block_R"):
            ang_tol = 0.20
        elif "wheel" in key:
            ang_tol = 0.28
        else:
            ang_tol = 0.36
        export_stl(solid, str(stl_path), tolerance=tol * 1.35, angular_tolerance=ang_tol)
        if key == "boiler_superstructure":
            subdivide_stl_x_span(stl_path, max_dx=1.85, z_min_filter=71.0)
        mp = compute_occ_mass_props(solid, rho)
        mass_db["parts"][key] = mp
        print(
            f"           -> 质量 = {mp['mass_kg']*1000:6.2f} g | "
            f"质心 = [{mp['com_m'][0]*1000:+.1f}, {mp['com_m'][1]*1000:+.1f}, {mp['com_m'][2]*1000:+.1f}] mm"
        )

    args.mass_out.parent.mkdir(parents=True, exist_ok=True)
    args.mass_out.write_text(json.dumps(mass_db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] 已导出 {len(tasks)} 个水密 STL 网格，并将精确材料惯量库写入: {args.mass_out}")


if __name__ == "__main__":
    main()
