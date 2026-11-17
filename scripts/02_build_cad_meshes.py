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
    Face,
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
    Gate 11 & Gate 20 1-33 上置亮面不锈钢单滑杆 + 1-30 前缸盖支座 + 2-13 后吊耳座 (イラスト 1, 2, SUS304):
      - 严格终止于气缸后盖 1-29 颈部 (x_max = -9.6mm < 0.0mm，零跨缸假桥杆!)；
      - 前盖活塞先导护套 (20-6+20-2) 与滑阀导套 (20-1) 为独立黑色零件网格。
    """
    x_bar_f, x_bar_r = -10.2, -58.0
    bar_len = x_bar_f - x_bar_r
    bar = Pos(0.5 * (x_bar_f + x_bar_r), 0.0, 6.5) * Box(bar_len, 2.8, 1.75)
    front_lug_1_30 = (
        Pos(-10.8, 0.0, 5.2) * Box(2.4, 2.6, 3.4)
        + Pos(-10.8, 0.0, 7.6) * Cylinder(radius=0.75, height=0.8)
    )
    rear_lug_2_13 = Pos(-57.0, 0.0, 7.6) * Box(3.6, 2.8, 1.2) + Pos(-57.0, 0.0, 8.5) * Cylinder(radius=0.85, height=1.2)
    part_L = bar + front_lug_1_30 + rear_lug_2_13
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_tail_sheath(cfg: Dict, is_left: bool = True):
    """20-6 方形底座 + 20-2 细长封闭圆筒活塞先导尻棒护套 (イラスト 20, 同轴 y=0, z=0 嵌合气缸前盖 x=+17.8mm, cc=1)。"""
    collar_20_6 = (
        Pos(+18.4, 0.0, 0.0) * Box(1.8, 4.4, 4.4)
        + Pos(+18.4, 0.0, +2.3) * Cylinder(radius=0.65, height=0.8)
    )
    tube_20_2 = Pos(+22.6, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.42, height=7.2)
    tip_cap = Pos(+26.4, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=1.42, top_radius=0.85, height=0.8)
    part_L = collar_20_6 + tube_20_2 + tip_cap
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_valve_spindle_sheath(cfg: Dict, is_left: bool = True):
    """20-1 台阶封闭式滑阀前导套 (イラスト 1, 20, 同轴对准真实外移滑阀轴线 y=+3.5mm [|Y|=36.0mm], z=+18.5mm, cc=1)。"""
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    y_valve = +3.5
    body_20_1 = Pos(+18.2, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=3.05, height=3.4)
    set_screw = Pos(+18.3, y_valve, z_valve + 3.15) * Cylinder(radius=0.80, height=1.1)
    nose_20_1 = Pos(+21.1, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.00, height=3.0)
    tip_cap = Pos(+22.8, y_valve, z_valve) * Rot(0, 90, 0) * Cone(bottom_radius=2.00, top_radius=1.35, height=0.6)
    part_L = body_20_1 + set_screw + nose_20_1 + tip_cap
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_crosshead(cfg: Dict, is_left: bool = True):
    """
    1-28 黄铜活塞 (双 PR-15 活塞环槽 + 前端螺母，零前伸尻棒!) + ø3.0 不锈钢活塞杆 + 1-31/1-32 封闭抱轨十字头 (イラスト 1):
      - イラスト 1 真值核实：活塞前端无先导尻棒 (20-2 为实车外观护套)，彻底根除捅穿前护套伪影！
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    bore = float(rc["bore"])
    L_main = float(rc["main_rod_length"])
    cyl_cx = float(rc["cylinder_center_x"])
    drop_z = float(vg["crosshead_drop"])
    p_head_dx = (cyl_cx + 4.2) - L_main  # +38.2 mm

    piston_disc = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore - 0.1, height=4.2)
    for gx in (p_head_dx - 1.05, p_head_dx + 1.05):
        ring_groove = Pos(gx, 0, 0) * Rot(0, 90, 0) * (
            Cylinder(radius=0.5 * bore + 0.5, height=0.80) - Cylinder(radius=0.5 * bore - 0.95, height=1.2)
        )
        piston_disc -= ring_groove
    front_nut = make_hex_nut((p_head_dx + 2.45, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=2.15, length=1.1)
    piston_rod = Pos(0.5 * (p_head_dx + 1.8) + 0.9, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.5, height=p_head_dx)

    ch_hub = Pos(-1.4, 0, 0) * Box(6.4, 3.60, 6.8) - Pos(-2.2, 0, 0) * Box(6.4, 2.05, 7.6)
    shoe_outer = Pos(-2.4, +0.15, 6.50) * Box(9.2, 4.10, 4.4)
    bar_slot = Pos(-2.4, 0.0, 6.50) * Box(11.4, 2.96, 1.95)
    under_open = Pos(-2.4, 0.0, 4.85) * Box(6.2, 1.90, 1.8)
    end_tabs_1_32 = (
        Pos(+1.9, +1.85, 6.50) * Box(0.9, 1.0, 4.6)
        + Pos(-6.7, +1.85, 6.50) * Box(0.9, 1.0, 4.6)
    )
    top_shoe = (shoe_outer + end_tabs_1_32) - bar_slot - under_open
    neck = Pos(-1.4, 0.0, 3.7) * Box(5.4, 3.40, 2.0)

    drop_root = Pos(0, +0.90, -2.6) * Box(3.4, 1.80, 2.4)
    drop_step = Pos(0, +2.20, -4.4) * Rot(+38, 0, 0) * Box(3.4, 1.40, 3.8)
    drop_tongue = Pos(0, +3.50, -7.4) * Box(3.4, 1.50, 4.6)
    drop_boss = Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=1.50)
    pin_7_13 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-1.80, y_head_outer=+2.45, r_shank=1.48, r_head=2.15, r_nut=1.75, slot_w=0.45
    )

    part_L = (
        piston_disc + front_nut + piston_rod + ch_hub + neck + top_shoe
        + drop_root + drop_step + drop_tongue + drop_boss + pin_7_13
    )
    part_L -= Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=0.90, height=6.0)
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_slide_valve_assembly(cfg: Dict, is_left: bool = True):
    """
    1-25 阀杆叉头 + 1-24 销栓 + N-M2 锁紧螺母 + 1-22 全直 ø2.0mm 不锈钢阀芯棒 + 1-5 浮动滑阀框 + 1-6 外进气平面 D 型滑阀 (イラスト 1, 20):
      - 彻底删除虚构的 3.5mm 横向折弯狗腿 (cross_arm)！
      - 1-25、1-22、1-5、1-6 与前导套 20-1 严格同轴位于配气平面 (y_local=0 -> |Y|=36.0mm, Z=+18.5mm)！
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    v_dx = (float(rc["cylinder_center_x"]) + 4.2) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))  # +19.7 mm

    clevis_1_25 = (
        Rot(90, 0, 0) * Cylinder(radius=1.12, height=3.3)
        + Pos(1.4, 0, 0) * Box(2.8, 3.3, 2.16)
        - Pos(-0.7, 0, 0) * Box(3.4, 1.72, 6.0)
    )
    pin_1_24 = make_stepped_pin_L(0.0, 0.0, -2.25, +2.25, r_shank=0.75, r_head=1.04, r_nut=1.00)
    nut_m2 = make_hex_nut((3.2, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=1.35, length=1.1)

    x_spindle_start = 2.4
    x_spindle_tip = v_dx + 6.6  # +26.3mm -> X_max <= +141.0mm (x_cyl <= +15.0mm, 严密收于 20-1 封闭套筒内!)
    spindle_1_22 = Pos(0.5 * (x_spindle_start + x_spindle_tip), 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(
        radius=1.00, height=x_spindle_tip - x_spindle_start
    )

    # 1-5 黄铜滑阀框 (Valve Buckle) + 顶部双螺钉头 + 1-6 外进气平面 D 型滑阀 (底面排气腔 z_seat=-3.7mm -> Z_world=+14.8mm)
    buckle_1_5 = (
        Pos(v_dx, 0.0, +0.3) * Box(9.4, 5.2, 3.6)
        + Pos(v_dx - 2.8, 0.0, +2.35) * Cylinder(radius=0.85, height=0.8)
        + Pos(v_dx + 2.8, 0.0, +2.35) * Cylinder(radius=0.85, height=0.8)
    )
    d_valve_1_6 = (
        Pos(v_dx, 0.0, -2.1) * Box(10.6, 7.2, 3.2)
        + Pos(v_dx, 0.0, -0.2) * Box(8.4, 6.2, 2.2)
        - Pos(v_dx, 0.0, -3.05) * Box(5.2, 4.8, 1.5)
    )

    assembly_L = clevis_1_25 + pin_1_24 + nut_m2 + spindle_1_22 + buckle_1_5 + d_valve_1_6
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_reverser_shaft(cfg: Dict):
    """
    4-12 逆转器下弯腹横轴 + 2-11/2-18 左右等长 11.0mm 提升臂 + 左走台内侧 (+Y=+27.5mm) 向上逆转摇臂 (イラスト 4, 図 4.2, イラスト 25):
      - 中央段 (|Y| <= 17.5mm) 下弯至 z_local=-15.0mm (Z=+11.5mm)，从紫铜锅筒底部 (Z_min=+16.7mm) 下方无干涉绕过；
      - 左侧向上摇臂严格置于 y=+27.5mm (内收于 25-7 窄护罩内，彻底消除外凸穿走台失真!)，顶点 z=+7.0mm 接扁钢逆转长拉杆 8-18。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5  # 36.0 mm
    arm_len_x = float(vg["reverser_arm_length"])
    y_end = y_vg + 1.2

    cranked_bar = make_swept_tube(
        [
            (0.0, -y_end, 0.0),
            (0.0, -25.8, 0.0),
            (-1.1, -17.5, -15.0),
            (-1.1, +17.5, -15.0),
            (0.0, +25.8, 0.0),
            (0.0, +y_end, 0.0),
        ],
        r_tube=1.22,
        bend_r=1.8,
    )
    reach_hub = Pos(0, +32.5, 0) * Rot(90, 0, 0) * Cylinder(radius=2.2, height=2.0)
    reach_lever = Pos(0, +32.5, 3.5) * Box(2.6, 1.6, 7.0)
    reach_tip = Pos(0, +32.5, 7.0) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=1.6)

    arms = None
    for sgn in (-1.0, 1.0):
        hub = Pos(0, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=2.1)
        arm = Pos(0.5 * arm_len_x, sgn * y_vg, 0) * Box(arm_len_x, 1.75, 3.0)
        tip_boss = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=1.9, height=1.75)
        tip_hole = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=5.0)
        sub = hub + arm + tip_boss - tip_hole
        arms = sub if arms is None else (arms + sub)

    return cranked_bar + reach_hub + reach_lever + reach_tip + arms


def build_cab_reach_rod(cfg: Dict):
    """
    8-18 左侧扁钢拱曲逆转长拉杆 (イラスト 8, 17, 25 & Manual p.43 实车多图联立真值):
      - 局部原点 (0,0,0) 位于 reverser_shaft 左走台向上摇臂顶点 (X=+25.5, Y=+32.5, Z=+33.5 mm)，与驾驶室螺旋逆转机严格共面 (Y=+32.5mm)；
      - 采用原厂 2.3x0.98mm 立式扁钢条 (绝非圆管!)，在中段 X∈[-70,-24]mm 平滑向上拱起至 z=+12.6mm (Z=+46.1mm)
        跨越左走台梯形隆起桥台 (Z=+42.2mm)，穿过驾驶室前壁窄竖缝，
        后端 (-153.0, 0.0, +2.0) -> 世界系 (-127.5, +32.5, +35.5mm) 同轴铰接窗高螺旋逆转机下垂双连杆 8-20x2！
    """
    L_rod = 153.0
    front_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.70, height=3.0)
        + Pos(-1.5, 0, 0) * Box(3.0, 3.0, 2.9)
        - Pos(+0.6, 0, 0) * Box(3.4, 1.72, 4.4)
    )
    pin_f = make_stepped_pin_L(0.0, 0.0, -1.90, +1.90, r_shank=0.80, r_head=1.30, r_nut=1.20)

    # XZ 侧视平滑拱曲立式扁钢条 (0.98 x 2.30 mm, 前/后端深入叉眼确保 cc==1，全程净空 >1.1mm!)
    xz_pts_top = [
        Vector(-2.0, 0.0, +1.15),
        Vector(-22.0, 0.0, +2.80),
        Vector(-45.5, 0.0, +3.60),
        Vector(-58.0, 0.0, +12.60),
        Vector(-88.0, 0.0, +12.60),
        Vector(-102.0, 0.0, +4.20),
        Vector(-L_rod + 0.6, 0.0, +3.15),
    ]
    xz_pts_bot = [Vector(p.X, 0.0, p.Z - 2.30) for p in reversed(xz_pts_top)]
    e_top_xz = Edge.make_spline(xz_pts_top)
    e_bot_xz = Edge.make_spline(xz_pts_bot)
    w_xz = Wire([e_top_xz, Edge.make_line(xz_pts_top[-1], xz_pts_bot[0]), e_bot_xz, Edge.make_line(xz_pts_bot[-1], xz_pts_top[0])])
    flat_blade = extrude(Face(w_xz), amount=0.49, dir=(0.0, 1.0, 0.0), both=True)

    rear_eye = (
        Pos(-L_rod, 0.0, +2.0) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=1.5)
        + make_stepped_pin_L(-L_rod, +2.0, -1.95, +1.95, r_shank=0.80, r_head=1.30, r_nut=1.20)
    )
    return front_clevis + pin_f + flat_blade + rear_eye


def build_drain_cock_linkage(cfg: Dict):
    """
    7-5 排液凸轮横轴 + 7-1~3 向上驱动摇臂 + 7-4/7-6 凸轮 + 1-19 x4 叠搭剪式摇板 + 1-18 x4 顶针 + 1-16 x4 排液嘴 (100% Y 对称, イラスト 1, 7 & 図 1.3):
      - 局部原点 (0,0,0) 对准 7-5 横轴中心 (X=+124.2, Y=0.0, Z=-3.8 mm)；
      - 7-1~3 摇臂自 Z=-3.8mm 向上伸至 z_local=+13.4mm (Z=+9.6mm, Y=±17.6mm) 衔接全直水平拉杆 8-5；
      - 前侧 1-19 摇板按 図 1.3 半叠搭于后侧 1-19 之上，1-16 排液嘴严格位于缸底 Z_world <= -10.2mm。
    """
    y_cock = 30.8
    cross_shaft_7_5 = Rot(90, 0, 0) * Cylinder(radius=0.95, height=46.8)
    sides = None
    for sgn in (-1.0, 1.0):
        yc = sgn * y_cock
        horn_7_1 = (
            Pos(0.0, sgn * 17.6, +5.9) * Box(2.0, 1.3, 12.2)
            + Pos(0.0, sgn * 17.6, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=1.8)
            + Pos(0.0, sgn * 17.6, +11.8) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=1.8)
        )
        cam_7_4 = (
            Pos(0.0, sgn * 22.4, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)
            + Pos(0.0, sgn * 22.8, -4.8) * Box(2.4, 2.0, 9.8)
        )
        scissor_rear = Pos(-1.0, sgn * 26.8, -9.7) * Box(6.8, 8.6, 1.2)
        scissor_front = Pos(+4.0, sgn * 26.8, -8.8) * Box(6.8, 8.6, 1.2)
        overlap_hub = Pos(+1.2, sgn * 23.4, -9.0) * Box(3.8, 3.0, 2.8)
        cock_rear = (
            Pos(-3.4, yc, -10.0) * Cylinder(radius=1.58, height=3.4)
            + Pos(-3.4, yc, -11.4) * Cylinder(radius=2.05, height=1.0)
            + Pos(-3.4, yc, -12.6) * Cylinder(radius=0.82, height=2.4)
        )
        cock_front = (
            Pos(+6.6, yc, -9.8) * Cylinder(radius=1.58, height=3.4)
            + Pos(+6.6, yc, -11.4) * Cylinder(radius=2.05, height=1.0)
            + Pos(-3.4 + 10.0, yc, -12.6) * Cylinder(radius=0.82, height=2.4)
        )
        sub = horn_7_1 + cam_7_4 + scissor_rear + scissor_front + overlap_hub + cock_rear + cock_front
        sides = sub if sides is None else (sides + sub)
    return cross_shaft_7_5 + sides


def build_drain_pull_rod(cfg: Dict):
    """
    8-5 右侧 100% 全直水平排液拉杆 + 8-6 驾驶室床下右后双台阶车削拉钮 (イラスト 7, 8 & 図 8.1):
      - 零折弯细不锈钢圆杆沿 (Y=-17.6mm, Z=+8.0mm) 自驾驶室床下 8-6 (-132.0) 穿过 8-10 冲压薄导片孔 (-121.0) 直抵 7-1 摇臂顶端 (+124.2, -17.6, +8.0)！
    """
    x0, x1 = -131.5, +124.2
    rod_8_5 = Pos(0.5 * (x0 + x1), -17.6, 8.0) * Rot(0, 90, 0) * Cylinder(radius=0.62, height=x1 - x0)
    knob_8_6 = (
        Pos(-132.4, -17.6, 8.0) * Rot(0, 90, 0) * Cylinder(radius=1.75, height=2.6)
        + Pos(-130.5, -17.6, 8.0) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=1.8)
    )
    front_clevis = Pos(+124.2, -17.6, 8.0) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=1.8)
    return rod_8_5 + knob_8_6 + front_clevis


# ==================== 3B. 驾驶室 1:1 原厂全功能内构与煤水车全功能可动阀件网格 ====================

def build_cab_backhead_plate():
    """
    11-9 ボイラバックプレート + 12-5 x3 压板 + 12-2 观火孔框与底铰耳 + 前折排水阀侧翼 (イラスト 11, 12 & p45, 100% Y 对称, cc=1):
      - 紧贴紫铜锅筒后端 X=-112.0mm，下底坐于 8-11 地板 Z=+20.0mm；
      - 左右床下设向前 (+X) 90° 折弯侧翼片 (Y=±18.5mm)，左翼承载横向 (+Y) 11-16/17 排水针阀；
      - 彻底删除背板虚假月牙螺钉凸块与穿光双孔，仅保留中央真实圆形投火口与 12-2 底铰框。
    """
    lower_skirt = Pos(-113.2, 0.0, 32.5) * Box(2.2, 53.6, 25.0)
    upper_arch = Pos(-113.2, 0.0, 44.5) * Rot(0, 90, 0) * Cylinder(radius=26.8, height=2.2)
    underfloor_flanges = (
        Pos(-113.2, +18.0, 16.2) * Box(2.2, 3.6, 8.6)
        + Pos(-115.0, +18.8, 16.2) * Box(4.2, 1.5, 8.2)
        + Pos(-113.2, -18.0, 16.2) * Box(2.2, 3.6, 8.6)
        + Pos(-115.0, -18.8, 16.2) * Box(4.2, 1.5, 8.2)
    )
    clamp_top = Pos(-113.6, 0.0, 69.6) * Rot(0, 90, 0) * Cylinder(radius=2.4, height=3.0)
    clamp_l = Pos(-113.6, +23.8, 28.0) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=3.0)
    clamp_r = Pos(-113.6, -23.8, 28.0) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=3.0)
    shield_12_2 = (
        Pos(-114.2, 0.0, 28.8) * Box(1.6, 13.8, 13.4)
        + Pos(-115.1, +5.0, 22.0) * Rot(90, 0, 0) * Cylinder(radius=1.40, height=2.0)
        + Pos(-115.1, -5.0, 22.0) * Rot(90, 0, 0) * Cylinder(radius=1.40, height=2.0)
        + Pos(-115.0, 0.0, 35.8) * Box(1.8, 2.8, 1.4)
    )
    firehole_cut = Pos(-113.5, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=4.4, height=8.0)
    return (lower_skirt + upper_arch + underfloor_flanges + clamp_top + clamp_l + clamp_r + shield_12_2) - firehole_cut


def build_cab_deck_and_mounts():
    """
    8-11 運転室床 (宽 76.4mm, Z_top=+20.0mm) + 8-21 左前延长片 + 4-6/4-7 折弯梁 + 8-10 冲压薄片ドレン导片 + 4-8 旁通支座 (イラスト 4, 8 & 図 8.1, cc=1):
      - 删除地板表面虚构凸环 bypass_boss 与床下笨重方块立柱；
      - 8-10 按 図 8.1 还原为 1.3mm 薄冲压倒 L 形导片 (-121.0, -17.6, +9.6 导孔)。
    """
    deck_8_11 = (
        Pos(-130.75, 0.0, 19.35) * Box(34.5, 76.4, 1.3)
        - Pos(-145.5, 0.0, 19.35) * Box(6.0, 27.0, 4.0)
    )
    strip_8_21 = Pos(-115.2, +32.5, 19.5) * Box(3.6, 9.2, 1.0)
    channels_4_6_7 = (
        Pos(-131.0, +13.6, 16.2) * Box(28.0, 1.3, 5.6)
        + Pos(-131.0, -13.6, 16.2) * Box(28.0, 1.3, 5.6)
    )
    bypass_hole = Pos(-134.5, -8.8, 19.5) * Cylinder(radius=1.45, height=6.0)
    riser_6_8_slot = Pos(-116.0, -9.1, 19.35) * Box(4.2, 3.8, 5.0)
    guide_8_10 = (
        Pos(-121.0, -17.6, 13.2) * Box(1.3, 3.8, 12.2)
        + Pos(-119.8, -17.6, 18.4) * Box(3.4, 3.8, 1.2)
        - Pos(-121.0, -17.6, 8.0) * Rot(0, 90, 0) * Cylinder(radius=1.05, height=4.0)
    )
    stand_4_8 = Pos(-131.2, -8.8, 17.2) * Box(9.6, 3.8, 3.2)
    return (deck_8_11 + strip_8_21 + channels_4_6_7 + guide_8_10 + stand_4_8) - bypass_hole - riser_6_8_slot


def build_cab_reverser_stand():
    """
    8-12 (前立柱) + 8-13 (后A字座) + 8-15 x2 (双侧多孔水平分度导条) 日本国铁左驾窗高螺旋逆转机座 (イラスト 8, 17 & p45, cc=1):
      - 挂载于驾驶室左地板 (-127.5, +32.5, +20.0mm)，底脚 z=0 零间隙贴合 8-11 地板表面；
      - 丝杠轴承与 8-15 分度导条中心升高至窗台原厂高度 z=+21.2mm (Z=+41.2mm)！
    """
    z_scr = 21.2
    base_pad = Pos(0.0, 0.0, +0.6) * Box(20.6, 5.0, 1.2)
    post_8_12_f = (
        Pos(+9.0, 0.0, 0.5 * (z_scr + 1.6)) * Box(2.2, 3.6, z_scr + 1.6)
        + Pos(+9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.4)
    )
    ped_8_13_r = (
        Pos(-9.0, 0.0, 0.5 * (z_scr + 1.6)) * Box(2.4, 4.6, z_scr + 1.6)
        + Pos(-9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=2.6)
        - Pos(-9.0, 0.0, +9.5) * Rot(0, 90, 0) * extrude(SlotOverall(width=9.0, height=2.4), amount=5.0, both=True)
    )
    bars_8_15 = (
        Pos(0.0, -1.75, z_scr) * Box(18.0, 0.85, 2.0)
        + Pos(0.0, +1.75, z_scr) * Box(18.0, 0.85, 2.0)
    )
    for hx in (-4.5, -1.5, +1.5, +4.5):
        bars_8_15 -= Pos(hx, 0.0, z_scr) * Rot(90, 0, 0) * Cylinder(radius=0.48, height=6.0)
    bore_8_14 = Pos(0.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=24.0)
    return (base_pad + post_8_12_f + ped_8_13_r + bars_8_15) - bore_8_14


def build_cab_reverser_wheel():
    """
    8-14 水平逆转丝杠 + 8-16 原厂双柄直径摇臂曲柄 (イラスト 8, 17 & p45, 绕 X 轴旋转!):
      - 彻底删除失真的四辐大圆手轮！按 イラスト 8 & p45 还原为直径双柄曲柄杆 (总跨度 9.6mm，完全收于驾驶室左内壁 Y<+38.0mm 内，零穿墙!)。
    """
    screw_8_14 = Pos(-0.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.08, height=22.4)
    hub = Pos(-11.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.90, height=2.0)
    crank_arm_8_16 = (
        Pos(-11.5, 0.0, 0.0) * Box(1.1, 1.9, 9.2)
        + Pos(-11.5, 0.0, +4.4) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=1.1)
        + Pos(-11.5, 0.0, -4.4) * Rot(0, 90, 0) * Cylinder(radius=1.10, height=1.1)
    )
    grip_main = Pos(-13.2, 0.0, +4.4) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=3.2)
    grip_bal = Pos(-12.6, 0.0, -4.4) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=1.8)
    nut_ret = make_hex_nut((-12.8, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=1.30, length=0.9)
    return screw_8_14 + hub + crank_arm_8_16 + grip_main + grip_bal + nut_ret


def build_cab_reverser_nut():
    """
    8-17 丝杠行走螺母滑块 + 8-20 x2 对称下垂双扁连杆 + 8-19 销栓 (イラスト 8, cc=1):
      - 局部原点位于窗高丝杠中心 (-127.5, +32.5, +41.2mm)，沿 X 轴随丝杠平移；
      - 对称双扁连杆 (y=±0.95mm) 夹持 Y=+32.5mm 的 8-18 扁钢拉杆尾眼，底销 z=-5.7mm (Z=+35.5mm) 同轴铰接！
    """
    nut_8_17 = Box(3.6, 2.8, 3.6) + Pos(0.0, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=3.4)
    link_in = Pos(0.0, -0.95, -2.85) * Box(2.2, 0.75, 7.2)
    link_out = Pos(0.0, +0.95, -2.85) * Box(2.2, 0.75, 7.2)
    bot_pin = Pos(0.0, 0.0, -5.7) * Rot(90, 0, 0) * Cylinder(radius=0.98, height=3.2)
    return nut_8_17 + link_in + link_out + bot_pin


def build_cab_regulator_lever():
    """
    11-11 驾驶室后壁顶中 (-121.2, 0.0, +68.5mm) 镂空扁平黄铜加减阀长柄 (イラスト 11 & p45):
      - 删除末端垂直后伸圆木桩伪影，还原原厂带指环圆孔的扁平黄铜操纵柄；q=0 水平指向 +Y，开启向上摆动。
    """
    hub = Rot(0, 90, 0) * Cylinder(radius=1.75, height=2.0)
    blade = (
        Pos(-0.4, +6.6, 0.0) * Box(1.0, 13.2, 2.0)
        + Pos(-0.4, +13.4, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=1.0)
        - Pos(-0.4, +13.4, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.68, height=3.0)
        - Pos(-0.4, +8.5, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.48, height=3.0)
    )
    ret_nut = make_hex_nut((-1.1, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=1.20, length=0.8)
    return hub + blade + ret_nut


def build_cab_blower_wheel():
    """
    11-13 驾驶室 11-15 左下体一体阀杆 (-118.8, +5.1, +64.2mm) ø6.2mm 三辐圆形黄铜通风手轮 (イラスト 11 & p45):
      - 同轴向前旋入 11-15 左下一体阀座 11-14，与上方 11-11 加减阀柄保持 >1.8mm 全行程安全净空！
    """
    stem = Pos(+1.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.88, height=3.4)
    hub = Rot(0, 90, 0) * Cylinder(radius=1.35, height=1.3)
    rim = Rot(0, 90, 0) * (Cylinder(radius=3.10, height=0.95) - Cylinder(radius=2.22, height=3.0))
    spokes = None
    for ang_d in (0.0, 120.0, 240.0):
        sp = Rot(ang_d, 0, 0) * Pos(0.0, +1.28, 0.0) * Box(0.68, 2.35, 0.68)
        spokes = sp if spokes is None else (spokes + sp)
    ret_nut = make_hex_nut((-0.65, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=1.05, length=0.75)
    return stem + hub + rim + spokes + ret_nut


def build_cab_whistle_lever():
    """
    11-3 驾驶室左上 11-1 汽笛阀底部 (-115.8, +18.2, +62.0mm) 镂空扁弧形触发拨柄 (イラスト 11, 12, 13 & p45)。
    """
    hub = Rot(0, 90, 0) * Cylinder(radius=1.45, height=1.6)
    arm_swept = make_swept_tube(
        [(0.0, 0.0, 0.0), (-0.3, +4.4, +1.3), (-0.3, +8.2, +4.8)],
        r_tube=0.68,
        bend_r=2.0,
    )
    tip = (
        Pos(-0.3, +8.4, +5.0) * Rot(0, 90, 0) * Cylinder(radius=1.25, height=1.1)
        - Pos(-0.3, +8.4, +5.0) * Rot(0, 90, 0) * Cylinder(radius=0.52, height=3.0)
    )
    return hub + arm_swept + tip


def build_cab_sight_glass_tube():
    """11-7 驾驶室右侧外缘 (-118.5, -31.3, +46.0mm) 透明耐热硼硅玻璃水面计管 (OD 4.0mm x H 17.0mm 空心管, cc=1)。"""
    return Cylinder(radius=2.00, height=17.0) - Cylinder(radius=1.15, height=19.0)


def build_cab_sight_drain():
    """13-5 #3 六角中空螺母 + (1-1/1-2/1-3) 右侧水面计下座同轴底部一字槽提升阀螺塞 (-118.5, -31.3, +28.8mm)。"""
    hex_13_5 = make_hex_nut((0.0, 0.0, +0.9), (0.0, 0.0, 1.0), hex_r=2.25, length=1.8)
    stem = Pos(0.0, 0.0, +1.8) * Cylinder(radius=1.20, height=2.8)
    cap_1_1 = Pos(0.0, 0.0, -0.9) * Cylinder(radius=2.05, height=2.0) - Pos(0.0, 0.0, -1.75) * Box(4.8, 0.60, 0.75)
    return hex_13_5 + stem + cap_1_1


def build_cab_pg_dial():
    """PG-ø20 右内上 (-122.5, -15.4, +66.8mm) 严格垂直朝后 (-X, 0.0° 零仰角!) ø20mm 压力表白底刻度盘片 (イラスト 13 & p44 边缘差分标定)。"""
    return Rot(0, 90, 0) * Cylinder(radius=8.4, height=0.8)


def build_cab_pressure_needle():
    """PG-ø20 垂直表盘黑色指针 (绕水平法向 axis='1 0 0' 在垂直 YZ 平面内顺时针随表压正偏转)。"""
    return (
        Rot(0, 90, 0) * Cylinder(radius=1.15, height=0.9)
        + Pos(-0.35, 0.0, 2.7) * Box(0.45, 0.65, 6.0)
        + Pos(-0.35, 0.0, -1.3) * Box(0.45, 1.05, 2.2)
    )


def build_cab_blowdown_screw():
    """
    11-17 驾驶室左床下 11-9 前折侧翼处 (-115.5, +20.6, +15.8mm) 横向 (+Y) 一字槽锅炉排水针阀螺杆 (イラスト 11, 13 & Manual p.43):
      - 严格修正 90° 朝向错误！针阀螺杆轴线沿 Y 轴 (axis='0 1 0')，一字螺丝刀槽朝向机车左外侧 (+Y)，供司炉从左床下横向旋开排污！
    """
    stem = Pos(0.0, -1.2, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.12, height=3.2)
    head = Pos(0.0, +0.9, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.15, height=1.8)
    slot = Pos(0.0, +1.75, 0.0) * Box(4.8, 0.75, 0.62)
    return stem + (head - slot)


def build_cab_firebox_door():
    """
    12-1 驾驶室低位中下底铰蝴蝶式铸造火室门 (イラスト 12 & p45):
      - 绕底部水平 Y 轴销 12-3 (-115.4, 0.0, +22.0mm) 向后下方翻开；实心浮雕门板闭合时严密遮光，零漏火伪影！
    """
    pin_12_3 = Rot(90, 0, 0) * Cylinder(radius=1.12, height=12.6)
    door_plate = (
        Pos(-0.6, 0.0, +6.7) * Box(1.5, 12.2, 13.0)
        + Pos(-0.6, 0.0, +6.8) * Rot(0, 90, 0) * Cylinder(radius=5.6, height=1.6)
    )
    boss_ring = Pos(-1.35, 0.0, +6.8) * Rot(0, 90, 0) * (Cylinder(radius=3.8, height=0.8) - Cylinder(radius=2.2, height=1.2))
    top_latch = Pos(-1.2, 0.0, +13.4) * Box(1.8, 2.4, 1.4)
    return pin_12_3 + door_plate + boss_ring + top_latch


def build_cab_bypass_lever():
    """
    6-9 / 8-7 驾驶室右后地板 (-134.5, -8.8, +23.8mm) 同轴旁通阀立杆与车削圆锥水平拨柄 (イラスト 6, 8 & p45):
      - 6-9 立杆向下延伸穿过 8-11 地板直插床下 4-17 旁通阀顶塔，零悬空！
    """
    stem_6_9 = Pos(0.0, 0.0, -4.2) * Cylinder(radius=1.05, height=8.8)
    collar = Pos(0.0, 0.0, -2.9) * Cylinder(radius=1.85, height=1.8)
    hub = Cylinder(radius=1.75, height=2.0)
    rod_8_7 = Pos(-3.8, 0.0, +0.2) * Rot(0, -90, 0) * Cone(bottom_radius=0.95, top_radius=0.68, height=7.2)
    ball_tip = Pos(-7.4, 0.0, +0.2) * Cylinder(radius=1.05, height=1.6)
    return stem_6_9 + collar + hub + rod_8_7 + ball_tip


def build_cab_roof_hatch():
    """21-15 ベンチレーター (驾驶室顶棚中央天窗通风盖, イラスト 21, 28, 前铰于 (-119.5, 0, +82.2mm))。"""
    L_v, W_v = 24.5, 19.6
    lid_plate = Pos(-0.5 * L_v, 0.0, +1.2) * Box(L_v, W_v, 1.4)
    clerestory = (
        Pos(-0.5 * L_v, 0.0, +2.5) * Box(L_v - 2.4, W_v - 2.6, 2.0)
        + Pos(-0.5 * L_v, 0.0, +3.6) * Box(L_v + 0.8, W_v + 0.8, 1.0)
    )
    hinge_l = Pos(0.0, +7.2, +0.8) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=3.6)
    hinge_r = Pos(0.0, -7.2, +0.8) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=3.6)
    return lid_plate + clerestory + hinge_l + hinge_r


def build_tender_hp_links():
    """
    31-7 x2 手摇泵平行摇动支点双连杆 + 31-8 x2 销栓 (イラスト 31, 34, 37 真 1-DOF 四连杆机构!, cc=1):
      - 后端枢接于 31-5 泵体前上耳 (-47.0, +4.0, +31.7mm)，孔距 L=20.0mm 向前伸至浮动支点 (+20.0, 0.0, 0.0) -> (-27.0, +4.0, +31.7mm)！
    """
    L = 20.0
    pin_rear = Rot(90, 0, 0) * Cylinder(radius=1.35, height=6.4) + make_hex_nut((0.0, +2.8, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.1) + make_hex_nut((0.0, -2.8, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.1)
    pin_front = Pos(L, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=6.4) + make_hex_nut((L, +2.8, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.1) + make_hex_nut((L, -2.8, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.1)
    bar_l = make_connecting_bar_xy((0.0, +1.85), (L, +1.85), 0.0, thick_y=1.15, height_z=4.6)
    bar_r = make_connecting_bar_xy((0.0, -1.85), (L, -1.85), 0.0, thick_y=1.15, height_z=4.6)
    return pin_rear + pin_front + bar_l + bar_r


def build_tender_handpump_lever():
    """
    31-3 手摇给水泵浮动摇臂立杆 (イラスト 31, 34, 37 & p47 真四连杆浮动支点件):
      - 局部原点 (0,0,0) 位于中部 31-8#1 浮动支点 (-27.0, +4.0, +31.7mm)；
      - 下端 z=-10.0mm 扁舌插于 31-2 柱塞前端割槽内由 31-1 销铰接；
      - 彻底删除粗大空心圆筒伪影！按 イラスト 31 & p47 还原为细长 ø4.0mm 黄铜立柱 (顶面微浅插孔供 37-12 加长杆插拔，Z_tdr=+58.2mm < +62.0mm 甲板)。
    """
    mid_hub = Rot(90, 0, 0) * Cylinder(radius=2.1, height=2.3)
    lower_tang = Pos(0.0, 0.0, -5.0) * Box(4.0, 2.2, 10.4) + Pos(0.0, 0.0, -10.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)
    upper_shank = Pos(0.0, 0.0, +7.5) * Box(3.4, 2.2, 15.0)
    upper_post = (
        Pos(0.0, 0.0, +20.2) * Cylinder(radius=2.0, height=12.6)
        - Pos(0.0, 0.0, +25.0) * Cylinder(radius=0.95, height=4.0)
    )
    return mid_hub + lower_tang + upper_shank + upper_post


def build_tender_handpump_ram():
    """
    31-2 水槽底浸渍 ø7.0x26.0mm 横型黄铜柱塞 + 前端割槽与 31-1 双螺母横销 (イラスト 31):
      - 局部原点 (0,0,0) 位于前端 31-1 销孔中心 (-27.0, +4.0, +21.7mm)，柱塞向后 (-X) 伸入 31-5 缸孔 [-67.0, -45.0]mm！
    """
    plunger = Pos(-12.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.45, height=25.0)
    groove = Pos(-22.5, 0.0, 0.0) * Rot(0, 90, 0) * (Cylinder(radius=3.60, height=1.8) - Cylinder(radius=2.80, height=2.2))
    front_clevis = (
        Pos(-1.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.45, height=5.6)
        - Pos(+0.5, 0.0, 0.0) * Box(5.8, 2.55, 8.0)
    )
    pin_31_1 = (
        Rot(90, 0, 0) * Cylinder(radius=1.25, height=10.2)
        + make_hex_nut((0.0, +4.3, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.2)
        + make_hex_nut((0.0, -4.3, 0.0), (0.0, 1.0, 0.0), hex_r=1.9, length=1.2)
    )
    return (plunger - groove + front_clevis) + pin_31_1


def build_tender_drain_plug():
    """
    31-11 水槽栓 (イラスト 31 上部引抜式垂直排水ロッド + PS6-1.9 密封肩):
      - 落座于水箱底 31-9 衬套 (-41.0, -10.3, +15.2mm)，总高 36.5mm -> 提拉 +5.0mm 开启时顶端 Z_tdr=+56.7mm < +62.0mm 上板，零穿顶！
    """
    tip = Pos(0.0, 0.0, -1.8) * Cylinder(radius=2.2, height=4.2)
    seal_collar = Pos(0.0, 0.0, +0.8) * Cylinder(radius=2.9, height=2.0)
    shaft = Pos(0.0, 0.0, +18.0) * Cylinder(radius=1.25, height=34.0)
    pull_knob = Pos(0.0, 0.0, +35.5) * Cylinder(radius=3.2, height=2.4)
    return tip + seal_collar + shaft + pull_knob


def build_tender_coal_cover():
    """
    36-1 着脱式角丸給水口蓋 (67.4x35.8mm, R=3.2mm 下折包边) + 36-2 中央 33.2mm 宽石炭後仕切板 (イラスト 35, 36 & p47):
      - 彻底根除 77.6mm 十字飞翼失真！イラスト 35 + 36 多图联立核实：左右两侧横隔板为焊死在 35-1 上的 35-5 x2，
        可提起的 36-2 仅跨越中央加水口宽度 W=33.2mm (Y∈[-16.6,+16.6]mm, X=-52.1mm)，闭合时与左右 35-5 x2 共面成完整后煤壁！
    """
    L_c, W_c, R_c = 67.4, 35.8, 3.2
    dx_c, dy_c = 0.5 * L_c - R_c, 0.5 * W_c - R_c

    def _rounded_plate(lx: float, wy: float, rc: float, hz: float, z0: float):
        ax, ay = 0.5 * lx - rc, 0.5 * wy - rc
        core = Pos(0.0, 0.0, z0) * Box(lx - 2.0 * rc, wy, hz) + Pos(0.0, 0.0, z0) * Box(lx, wy - 2.0 * rc, hz)
        for sx in (-1.0, +1.0):
            for sy in (-1.0, +1.0):
                core += Pos(sx * ax, sy * ay, z0) * Cylinder(radius=rc, height=hz)
        return core

    cap_outer = _rounded_plate(L_c, W_c, R_c, 2.1, -0.35)
    cap_cavity = _rounded_plate(L_c - 2.2, W_c - 2.2, max(1.0, R_c - 1.1), 1.6, -0.90)
    cap_36_1 = cap_outer - cap_cavity
    front_lip = Pos(+0.5 * L_c + 0.9, 0.0, +0.25) * Box(2.4, 12.0, 0.9)
    wall_36_2 = Pos(-7.1, 0.0, +6.1) * Box(1.5, 33.2, 11.7)
    return cap_36_1 + front_lip + wall_36_2


def build_tender_fuel_valve():
    """
    36-6 煤水车前部 36-4 酒精箱 (+21.7, -6.0, +66.6mm) 燃料针阀 T 字手柄 + 槽内全长垂直黄铜针阀杆 (イラスト 36 & Manual p.11)。
    """
    long_needle_stem = Pos(0.0, 0.0, -20.5) * Cylinder(radius=1.15, height=41.5)
    needle_cone = Pos(0.0, 0.0, -41.6) * Rot(180, 0, 0) * Cone(bottom_radius=1.15, top_radius=0.25, height=1.8)
    hub = Cylinder(radius=1.9, height=1.8)
    t_bar = Pos(0.0, 0.0, +0.8) * Rot(90, 0, 0) * Cylinder(radius=1.05, height=10.0)
    return long_needle_stem + needle_cone + hub + t_bar


def build_tender_fuel_cap():
    """36-5 煤水车前部 36-4 酒精箱 (+32.7, +6.5, +65.8mm) 平顶气密滚花旋盖 (删除虚假顶面六角螺母, イラスト 36 & p47)。"""
    plug = Pos(0.0, 0.0, -1.6) * Cylinder(radius=2.8, height=3.2)
    cap = Pos(0.0, 0.0, +0.9) * Cylinder(radius=5.1, height=2.6)
    chamfer_top = Pos(0.0, 0.0, +2.35) * Cone(bottom_radius=5.1, top_radius=4.5, height=0.4)
    return plug + cap + chamfer_top


# ==================== 3C. 机车↔煤水车 3 根跨车弹性软管与机车端 34-8 活接 ====================

def _compute_drawbar_hose_targets_mm(cfg: Dict, is_r2m: bool) -> Dict[str, Tuple[float, float, float]]:
    """精确复刻 03_generate_mjcf.py 中 roller 与 curve_r2m 姿态下煤水车前端梁三嘴在 drawbar_8_8 局部系下的毫米坐标。"""
    w_cfg = cfg["wheels_and_axles"]
    t_cfg = cfg["tender_and_track"]
    r_drv = 0.5 * float(w_cfg["driving_wheel_diameter"])
    r_tdr = 0.5 * float(t_cfg["tender_wheel_diameter"])
    x_pin_loco = float(t_cfg["drawbar_loco_pin_x"])          # -146.0 mm
    L_drawbar = float(t_cfg["drawbar_length"])               # 34.0 mm
    x_pin_tdr_local = float(t_cfg["drawbar_tender_pin_local_x"])  # +102.0 mm
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
        "fuel_tdr": to_db_local(106.0, 0.0, +7.0),
        "feed_tdr": to_db_local(108.5, -21.3, -5.1),
        "ret_tdr": to_db_local(108.5, +21.3, -5.1),
    }


def build_umb_hose_fuel(cfg: Dict, is_r2m: bool = False):
    """1) ST-ø7 半透明白色硅胶供醇软管 (牵引杆 8-8 上方中央 Y=0, 23-1 <-> 34-2 サンプ + 32-3 导环)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["fuel_tdr"]
    p0 = (-3.6, 0.0, +9.6)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), min(p0[2], p2[2]) - 1.2)
    yaw_tdr = math.atan2(p2[1] - p0[1], p2[0] - p0[0]) + math.pi
    t_end = (-math.cos(yaw_tdr - math.pi), -math.sin(yaw_tdr - math.pi), 0.0)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), t_end, r_hose=2.45)


def build_umb_hose_feed(cfg: Dict, is_r2m: bool = False):
    """2) BT-ø5 黑色氟橡胶给水/手摇泵共用软管 (牵引杆下方斜跨: 机车左侧 +Y 6-11 <-> 煤水车右侧 -Y=-21.3mm 34-6)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["feed_tdr"]
    p0 = (-4.2, +7.2, -3.2)
    p2 = (tgt[0] + 0.4, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -7.8)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), r_hose=1.82)


def build_umb_union_34_8(cfg: Dict, is_r2m: bool = False):
    """
    34-7 口金 + 34-8 ポンプジョイント滚花活接螺母 + HC-1 弹簧卡箍 (イラスト 34, 37 & p45/p47 真值校正: 位于机车端 6-11 接头处，绝非悬在半空中段！cc=1)。
    """
    p_loco = (-4.6, +7.2, -3.2)
    ax = Vector(1.0, 0.0, 0.0)
    tail_34_7 = extrude(Plane(origin=Vector(*p_loco) - ax * 3.6, z_dir=ax) * Circle(2.25), amount=5.4)
    nut_34_8 = (
        Pos(p_loco[0] + 0.6, p_loco[1], p_loco[2]) * Rot(0, 90, 0) * Cylinder(radius=3.25, height=3.4)
        + make_hex_nut((p_loco[0] - 0.8, p_loco[1], p_loco[2]), (1.0, 0.0, 0.0), hex_r=2.85, length=1.4)
    )
    hc1_clamp = extrude(Plane(origin=Vector(*p_loco) - ax * 3.0, z_dir=ax) * Circle(2.60), amount=1.2)
    return tail_34_7 + nut_34_8 + hc1_clamp


def build_umb_hose_return(cfg: Dict, is_r2m: bool = False):
    """3) BT-ø5 黑色氟橡胶旁通无压回水软管 (牵引杆下方斜跨: 机车右侧 -Y 4-16 <-> 煤水车左侧 +Y=+21.3mm 34-6)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["ret_tdr"]
    p0 = (-3.6, -7.2, -3.2)
    p2 = (tgt[0] + 0.4, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -3.6)
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
    """
    1-29 主气缸 + 20-4 外包板轮廓 + 1-7/1-14 镂空外进气蒸汽室 + 10-1 螺钉顶盖 + 1-9 三气口镜面 + 20-8 后导套 + 内壁 1-20 给/排汽承插口 (イラスト 1, 7, 10, 20):
      - 还原 JNR D51 气缸外包板特征轮廓 (外侧平齐面 |Y_outer|=40.3mm -> y_local=+7.8mm，下部圆弧缸肚，上部内收斜肩)；
      - 蒸汽室内部真实镂空并刻出 1-9 三条进/排气道口 (Z_seat=+14.8mm)，经前/后阀杆孔与内壁 1-20 进汽孔自然贯通 (确保单连通 cc==1!)；
      - 后部阀杆填料压盖 GP-5 + 20-8 护套与前部 20-1 同轴位于 y_local=+3.5mm (|Y|=36.0mm, Z=+18.5mm)，并与合并杆保持 >1.5mm 全包络净空！
    """
    bore = float(cfg["cylinder_and_rods"]["bore"])
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    y_valve = +3.5
    r_barrel = 10.6

    # 1. 下部圆弧主缸筒 + 20-4 外侧竖直外包壁 (y_max=+7.8mm) + 内侧鞍座块
    main_barrel = Pos(+5.3, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=22.2)
    rear_neck = (
        Pos(-8.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=5.6, height=7.6)
        + Pos(-12.9, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.5)
        - Pos(-9.8, -7.20, 0.0) * Box(14.0, 10.0, 16.0)
        - Pos(-9.8, +6.85, 0.0) * Box(14.0, 10.0, 16.0)
    )
    saddle_block = Pos(+4.8, -6.2, 6.2) * Box(21.4, 12.2, 23.6)
    outer_clad_20_4 = (
        Pos(+5.1, +1.8, 11.2) * Box(22.6, 12.0, 21.6)
        - Pos(+5.1, +7.6, -7.8) * Rot(+36, 0, 0) * Box(26.0, 8.0, 10.0)
    )
    front_cover = Pos(+17.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=10.1, height=1.6)
    for b in (
        Pos(+18.0, 0.0, 0.0)
        * Rot(0, 90, 0)
        * PolarLocations(radius=8.4, count=6)
        * Cylinder(radius=0.72, height=0.8)
    ):
        front_cover += b

    # 2. 1-7 / 1-14 外进气蒸汽室壳体 + 10-1 四螺栓矩形顶盖 (同轴包容 y_local=+3.5mm 的 D 型滑阀与滑阀框)
    steam_chest = Pos(+5.1, +2.4, 18.2) * Box(23.4, 10.8, 10.2)
    lid_10_1 = Pos(+5.1, +2.6, 23.6) * Box(23.8, 10.6, 1.3)
    for sx_b in (-9.2, +9.2):
        for sy_b in (-3.4, +3.4):
            lid_10_1 += make_hex_nut((5.1 + sx_b, 2.6 + sy_b, 24.55), (0.0, 0.0, 1.0), hex_r=0.95, length=0.8)

    # 3. 内侧竖壁 1-20 上下进/排汽承插口 + 后部同轴 GP-5/20-8 阀杆压盖 + 底部排液阀座
    spigot_inlet_1_20 = (
        Pos(-2.5, -9.2, z_valve) * Rot(90, 0, 0) * Cylinder(radius=2.9, height=8.8)
        + make_hex_nut((-2.5, -11.2, z_valve), (0.0, 1.0, 0.0), hex_r=3.3, length=2.0)
    )
    spigot_exh_1_20 = (
        Pos(+4.0, -10.8, 12.5) * Rot(90, 0, 0) * Cylinder(radius=2.7, height=5.6)
        + make_hex_nut((+4.0, -11.4, 12.5), (0.0, 1.0, 0.0), hex_r=3.0, length=1.8)
    )
    drain_boss_r = Pos(-5.2, -1.7, -10.4) * Cylinder(radius=1.68, height=3.6)
    drain_boss_f = Pos(+4.8, -1.7, -10.4) * Cylinder(radius=1.68, height=3.6)
    gland_gp5_20_8 = (
        Pos(-7.5, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=1.8)
        + make_hex_nut((-7.1, y_valve, z_valve), (1.0, 0.0, 0.0), hex_r=2.30, length=1.2)
    )

    # 4. 真实内部气缸膛孔 + 蒸汽室内腔 (Z_seat=+14.8mm) + 1-9 三气道凹槽 + 贯通阀杆/进汽孔 (保障 cc==1)
    chest_hollow = (
        Pos(+5.0, +2.7, 18.3) * Box(19.8, 7.6, 7.0)
        + Pos(+8.2, +2.7, 14.3) * Box(1.6, 5.2, 1.6)
        + Pos(+4.2, +2.7, 14.1) * Box(2.0, 5.2, 2.0)
        + Pos(+0.2, +2.7, 14.3) * Box(1.6, 5.2, 1.6)
    )
    spindle_bore = Pos(+4.5, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=28.0)
    inlet_bore = Pos(-2.5, -6.5, z_valve) * Rot(90, 0, 0) * Cylinder(radius=1.85, height=15.0)
    bore_cut = (
        Pos(+5.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=21.2)
        + Pos(-9.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.62, height=13.2)
        + chest_hollow
        + spindle_bore
        + inlet_bore
    )
    assembly_L = (
        main_barrel + rear_neck + saddle_block + outer_clad_20_4 + front_cover
        + steam_chest + lid_10_1 + spigot_inlet_1_20 + spigot_exh_1_20
        + drain_boss_r + drain_boss_f + gland_gp5_20_8 - bore_cut
    )
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_boiler_superstructure(cfg: Dict, heavy_equipped: bool = True):
    """
    锅炉烟室、流线型カマボコ砂箱汽包、车头斜裙板/前端梁/车钩/带观察窗除烟板、走台隆起桥台与驾驶室外衣总成 (イラスト 15, 16, 18, 21, 25, 28, 29, 30 & Manual p.43):
      - 外边界由两侧除烟板 28-11/12 严格锁定在 [-40.8, +40.8] mm (|min_y + max_y| = 0.000mm)；
      - 彻底删除烟囱顶部违和方形积木块，还原圆润微锥翻边烟囱 18-25、球冠烟室门 18-14 (+Y 双铰链/4辐手轮/D51牌)、34mm 宽给水加热器 28-13 与前伸 LP-1 头灯；
      - 还原 45° 斜切前渡板 21-18/24 + 下层前甲板 (25-3 活门) + 前端梁 8-1/8-2 + 车钩 29-2 + 前踏板 25-4x2 (从正前方 100% 遮蔽内部黄色 4-2 油箱，同时侧面完整露出气缸前盖与 20-1/20-2 护套!)；
      - 还原两侧走台中段梯形隆起桥台 (X∈[-70.2,-23.8], Z=+42.4mm)，左桥台下收纳 22-11 双联空压机，前部走台内侧设窄小 25-7 护罩！
    """
    b_cfg = cfg["boiler_and_aux"]
    od_sb = float(b_cfg["smokebox_od"])
    r_sb, L_sb = 0.5 * od_sb, float(b_cfg["smokebox_length"])
    fb_l, fb_w, fb_h = [float(v) for v in b_cfg["firebox_lwh"]]
    z_axis, x_sb_front = 44.0, 162.0
    x_sb_rear = x_sb_front - L_sb
    x_fb_rear = -112.0
    x_fb_front = x_fb_rear + fb_l

    # 1. 烟室筒体 + 18-22 门圈 + 18-14 球冠凸面烟室门 (+Y 双锥形铰链带 18-4x2、4辐锁紧轮 18-12/13、[D51] 号码牌 18-18、8 爪卡扣)
    sb_x0, sb_x1 = x_sb_rear - 0.5, x_sb_front
    smokebox = Pos(0.5 * (sb_x0 + sb_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb, height=sb_x1 - sb_x0)
    sb_saddle = Pos(136.0, 0, 21.5) * Box(36.0, 26.0, 14.0)
    door_ring_18_22 = Pos(x_sb_front + 0.5, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 0.8, height=2.6)
    door_dish_18_14 = (
        Pos(x_sb_front + 2.2, 0, z_axis) * Rot(0, 90, 0) * Cone(bottom_radius=25.2, top_radius=20.6, height=1.8)
        + Pos(x_sb_front + 3.7, 0, z_axis) * Rot(0, 90, 0) * Cone(bottom_radius=20.6, top_radius=13.5, height=1.5)
        + Pos(x_sb_front + 4.9, 0, z_axis) * Rot(0, 90, 0) * Cone(bottom_radius=13.5, top_radius=5.2, height=1.1)
    )
    for cl_b in (
        Pos(x_sb_front + 1.8, 0, z_axis)
        * Rot(0, 90, 0)
        * PolarLocations(radius=25.4, count=8)
        * Box(1.4, 1.4, 1.4)
    ):
        door_dish_18_14 += cl_b
    plate_d51_18_18 = Pos(x_sb_front + 4.3, 0.0, z_axis + 7.2) * Box(1.8, 10.6, 3.8)
    wheel_18_12_13 = (
        Pos(x_sb_front + 5.7, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=1.6, height=1.8)
        + Pos(x_sb_front + 6.3, 0, z_axis) * Rot(0, 90, 0) * (Cylinder(radius=3.4, height=0.8) - Cylinder(radius=2.5, height=2.0))
        + Pos(x_sb_front + 6.3, 0, z_axis) * Box(0.7, 6.4, 0.7)
        + Pos(x_sb_front + 6.3, 0, z_axis) * Box(0.7, 0.7, 6.4)
        + Pos(x_sb_front + 6.8, -1.2, z_axis - 2.6) * Rot(-26, 0, 0) * Box(0.9, 1.0, 5.4)
    )
    hinge_straps_18_4 = (
        Pos(x_sb_front + 2.1, +14.5, z_axis + 9.2) * Rot(0, +7, 0) * Box(2.0, 20.0, 2.1)
        + Pos(x_sb_front + 2.1, +14.5, z_axis - 9.2) * Rot(0, +7, 0) * Box(2.0, 20.0, 2.1)
        + Pos(x_sb_front + 1.4, +24.8, z_axis) * Cylinder(radius=1.15, height=22.0)
        + Pos(x_sb_front + 2.6, -22.5, z_axis) * Cylinder(radius=0.65, height=9.0)
        + Pos(x_sb_front + 1.8, -22.5, z_axis + 4.0) * Box(2.0, 1.4, 1.2)
        + Pos(x_sb_front + 1.8, -22.5, z_axis - 4.0) * Box(2.0, 1.4, 1.2)
    )

    # 2. 28-13 横置给水加热器 (宽 34.0mm, Y∈[-17,+17]) + 前高置 LP-1 头灯 + 圆润翻边烟囱 18-25 (彻底删除顶部方块!)
    feedwater_heater = (
        Pos(150.5, 0.0, z_axis + r_sb + 4.0) * Rot(90, 0, 0) * Cylinder(radius=5.3, height=34.0)
        + Pos(150.5, +17.2, z_axis + r_sb + 4.0) * Rot(90, 0, 0) * Cylinder(radius=5.8, height=1.2)
        + Pos(150.5, -17.2, z_axis + r_sb + 4.0) * Rot(90, 0, 0) * Cylinder(radius=5.8, height=1.2)
    )
    fwh_brackets = (
        Pos(150.5, -11.5, z_axis + r_sb - 0.5) * Box(6.4, 2.6, 5.6)
        + Pos(150.5, +11.5, z_axis + r_sb - 0.5) * Box(6.4, 2.6, 5.6)
    )
    headlight = (
        Pos(164.0, 0.0, z_axis + r_sb + 9.6) * Rot(0, 90, 0) * Cylinder(radius=4.8, height=8.2)
        + Pos(168.4, 0.0, z_axis + r_sb + 9.6) * Rot(0, 90, 0) * Cylinder(radius=5.2, height=1.2)
    )
    headlight_ped = Pos(161.2, 0.0, z_axis + r_sb + 3.2) * Box(5.6, 5.2, 9.2)

    x_chimney = x_sb_front - 32.0  # +130.0 mm
    chimney = (
        Pos(x_chimney, 0.0, z_axis + r_sb - 0.8) * Cylinder(radius=9.8, height=3.2)
        + Pos(x_chimney, 0.0, z_axis + r_sb + 9.6) * Cone(bottom_radius=8.1, top_radius=8.5, height=18.8)
        + Pos(x_chimney, 0.0, z_axis + r_sb + 19.3) * Cylinder(radius=9.0, height=1.4)
    )

    # 3. 锅炉外衣 (r_barrel=30.2mm + 4 道箍带) + JNR D51 流线型カマボコ汽包砂箱罩 29-11
    b_x0, b_x1 = x_fb_rear, x_sb_rear + 1.0
    r_barrel = 30.2
    barrel = Pos(0.5 * (b_x0 + b_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=b_x1 - b_x0)
    for bx in [74.0, 38.0, -14.0, -58.0]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.55, height=1.8)

    dome_29_11 = (
        Pos(22.0, 0, z_axis + r_barrel - 5.0) * extrude(SlotOverall(width=76.0, height=25.6), amount=16.5)
        + Pos(54.0, 0, z_axis + r_barrel + 3.0) * Rot(0, +32, 0) * Box(22.0, 23.0, 14.0)
        - Pos(68.0, 0, z_axis + r_barrel + 14.5) * Rot(0, +28, 0) * Box(28.0, 30.0, 14.0)
        + Pos(12.0, 0, z_axis + r_barrel + 11.8) * Cylinder(radius=3.2, height=1.2)
        + Pos(36.0, 0, z_axis + r_barrel + 11.8) * Cylinder(radius=2.8, height=1.2)
    )
    firebox = Pos(0.5 * ((x_fb_front + 1.0) + x_fb_rear), 0, z_axis - 11.0) * Box(fb_l + 1.0, fb_w, fb_h)

    # 4. 驾驶室外壳 21-28 (38.6x79.2mm, 左前窄竖缝供扁钢 8-18 穿过，下缘侧裙板遮蔽床下空洞，零顶壁大缺口!)
    cab_len, cab_w = 38.6, 79.2
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0  # -129.3 mm -> X ∈ [-148.6, -110.0]
    cab_box = Pos(cab_x, 0, 45.0) * Box(cab_len, cab_w, 52.0)
    cab_roof = (Pos(cab_x - 1.5, 0, z_axis + 2.0) * Rot(0, 90, 0) * Cylinder(radius=41.5, height=cab_len + 7.0)) & (
        Pos(cab_x - 1.5, 0, 76.5) * Box(cab_len + 7.0, cab_w + 2.0, 22.0)
    )
    cab_roof_hollow = (
        Pos(cab_x - 3.5, 0, z_axis + 2.0) * Rot(0, 90, 0) * Cylinder(radius=39.8, height=cab_len + 5.0)
    ) & (Pos(cab_x - 3.5, 0, 74.5) * Box(cab_len + 5.0, cab_w - 3.6, 24.0))
    rear_doorway_cut = (
        Pos(cab_x - 0.5 * cab_len, 0, 44.5) * Box(10.0, 47.0, 56.0)
        + (
            Pos(cab_x - 0.5 * cab_len, 0, z_axis + 2.0) * Rot(0, 90, 0) * Cylinder(radius=39.8, height=12.0)
            & Pos(cab_x - 0.5 * cab_len, 0, 73.0) * Box(12.0, 47.0, 24.0)
        )
    )
    roof_hatch_cut = Pos(-131.25, 0.0, 84.0) * Box(23.5, 18.6, 14.0)
    cab_trim = None
    for sgn_c in (-1.0, 1.0):
        visor_21_11 = Pos(cab_x + 1.5, sgn_c * 39.6, z_axis + 23.0) * Box(23.0, 2.0, 1.2)
        lower_skirt_21_28 = Pos(-135.5, sgn_c * 38.9, 16.0) * Box(25.0, 1.4, 7.2)
        handrail_30_4 = (
            Pos(cab_x - 0.5 * cab_len + 0.6, sgn_c * 24.8, z_axis + 4.0) * Cylinder(radius=0.75, height=28.0)
            + Pos(cab_x - 0.5 * cab_len + 1.2, sgn_c * 24.8, z_axis + 17.0) * Box(2.2, 2.0, 1.6)
            + Pos(cab_x - 0.5 * cab_len + 1.2, sgn_c * 24.8, z_axis - 9.0) * Box(2.2, 2.0, 1.6)
        )
        ct = visor_21_11 + lower_skirt_21_28 + handrail_30_4
        cab_trim = ct if cab_trim is None else (cab_trim + ct)

    cab = (
        cab_box + cab_roof + cab_trim
        - Pos(cab_x - 0.2, 0, 45.0) * Box(cab_len - 3.4, cab_w - 3.2, 56.0)
        - cab_roof_hollow
        - roof_hatch_cut
        - Pos(cab_x + 1.5, 0, z_axis + 14.0) * Box(21.5, cab_w + 10.0, 17.0)
        - rear_doorway_cut
        - Pos(-111.5, +32.5, 36.5) * Box(8.0, 2.8, 6.2)
    )

    # 5. 左右走台 + 中段梯形隆起桥台 (X∈[-70.2,-23.8], Z=+42.2mm) + 带观察方窗与双圆撑杆全尺寸除烟板 28-11/12 (|Y_outer|=40.8mm)
    walk_x0, walk_x1 = x_fb_rear + 1.0, 151.0
    walk_len = walk_x1 - walk_x0
    walk_xm = 0.5 * (walk_x0 + walk_x1)
    walkways = None
    for sgn in (-1.0, 1.0):
        board = Pos(walk_xm, sgn * 34.4, 30.8) * Box(walk_len, 10.2, 1.5)
        inward_web = Pos(walk_xm, sgn * 27.5, 30.8) * Box(walk_len - 8.0, 5.0, 1.5)
        # 梯形隆起走台桥 (跨越第 3/4 动轮上方，供扁钢 8-18 在上方拱跃，下方收纳 22-11 空压机) + 吊杆上叉头避让长槽
        bridge_cut = Pos(-47.0, sgn * 34.8, 30.8) * Box(36.0, 12.0, 4.0)
        lift_slot_cut = Pos(36.2, sgn * 36.0, 30.8) * Box(12.4, 5.9, 4.0)
        rev_hood_cut = Pos(25.3, +32.5, 30.8) * Box(10.6, 5.2, 4.0) if sgn > 0 else None
        bridge_top = Pos(-47.0, sgn * 34.2, 41.5) * Box(30.0, 9.8, 1.4)
        bridge_ramp_f = Pos(-28.8, sgn * 34.2, 36.2) * Rot(0, +52, 0) * Box(14.2, 9.8, 1.4)
        bridge_ramp_r = Pos(-65.2, sgn * 34.2, 36.2) * Rot(0, -52, 0) * Box(14.2, 9.8, 1.4)
        bridge_tie = Pos(-47.0, sgn * 28.5, 36.2) * Box(30.0, 1.8, 11.4)
        board_base = (board + inward_web) - bridge_cut - lift_slot_cut
        if rev_hood_cut is not None:
            board_base = board_base - rev_hood_cut
        board_sub = board_base + bridge_top + bridge_ramp_f + bridge_ramp_r + bridge_tie

        # 28-11 / 28-12 除烟板: X∈[+96.0, +176.0], Z∈[+31.2, +85.2], 前上 45° 倒角 + 内折上缘 + 矩形检查窗 + 双 ø1.5mm 圆横撑
        defl = (
            Pos(136.0, sgn * 40.25, 58.2) * Box(80.0, 1.1, 54.0)
            + Pos(131.0, sgn * 39.2, 84.6) * Box(68.0, 3.0, 1.2)
            - Pos(172.0, sgn * 40.25, 81.2) * Rot(0, -42, 0) * Box(22.0, 5.0, 22.0)
            - Pos(120.0, sgn * 40.25, 45.8) * Box(19.0, 5.0, 10.2)
            + Pos(174.2, sgn * 39.4, 48.0) * Cylinder(radius=0.65, height=22.0)
        )
        defl_stays = (
            Pos(148.0, sgn * 34.8, 67.5) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=11.2)
            + Pos(116.0, sgn * 34.8, 67.5) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=11.2)
        )

        # 21-18/24 车头两侧 45° 斜降渡板阶梯 (X=+150..+174, Z=+31.2 -> +9.6, 完全避开下方气缸前盖与护套!)
        front_slope = Pos(162.0, sgn * 33.4, 20.4) * Rot(0, +42, 0) * Box(32.5, 12.6, 1.5)
        for st_x, st_z in [(155.5, 26.5), (162.0, 20.6), (168.5, 14.7)]:
            front_slope += Pos(st_x, sgn * 33.4, st_z) * Box(3.2, 12.2, 1.1)

        dummy_15_3 = (
            Pos(58.0, sgn * 29.8, 44.0) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=3.4)
            + Pos(58.0, sgn * 31.2, 45.2) * Cylinder(radius=1.9, height=4.6)
        )
        dummy_16_2 = Pos(-78.0, sgn * 24.4, 24.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.4)
        air_res = Pos(-3.0, sgn * 28.8, 35.6) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=32.0)

        part = board_sub + defl + defl_stays + front_slope + dummy_15_3 + dummy_16_2 + air_res
        walkways = part if walkways is None else (walkways + part)

    # 6. 下层前甲板 (Z=+9.6mm) + 25-3 注油器活门 + 全宽前端梁 8-1/8-2 + 自动车钩 29-2 + 28-3 提钩杆 + 25-4x2 前踏板 (正视 100% 遮挡 4-2 油箱!)
    front_deck_and_beam = (
        Pos(159.5, 0.0, 20.4) * Rot(0, +42, 0) * Box(32.0, 42.0, 1.5)
        + Pos(179.5, 0.0, 9.6) * Box(15.0, 79.2, 1.6)
        + Pos(186.5, 0.0, 6.2) * Box(3.2, 79.2, 8.4)
        + Pos(186.5, 0.0, 6.2) * Rot(90, 0, 0) * Cylinder(radius=0.65, height=76.0)
        + Pos(189.2, 0.0, 6.0) * Box(4.2, 9.6, 5.4)
        + Pos(192.4, 0.0, 6.0) * Cylinder(radius=3.4, height=5.8)
    )
    for sgn_st in (-1.0, 1.0):
        front_deck_and_beam += (
            Pos(186.5, sgn_st * 38.6, -4.0) * Box(2.4, 1.1, 13.6)
            + Pos(186.5, sgn_st * 35.4, -4.0) * Box(2.4, 1.1, 13.6)
            + Pos(186.5, sgn_st * 37.0, -10.4) * Box(2.6, 4.3, 1.1)
        )
    hatch_25_3 = (
        Pos(164.0, 0.0, 15.8) * Rot(0, +42, 0) * Box(7.2, 9.6, 1.2)
    )

    # 7. 左桥台下收纳 22-11 双联空压机 (X=-47.0) + 窄式封闭前脸 25-7 逆转摇臂罩 (X=+25.5, Y=+27.5)
    comp_22_11 = (
        Pos(-47.0, +34.4, 28.5) * Cylinder(radius=3.1, height=10.8)
        + Pos(-47.0, +34.4, 31.8) * Cylinder(radius=3.5, height=2.4)
        + Pos(-47.0, +31.6, 33.8) * Box(4.8, 6.8, 3.6)
    )
    silencer_28_8 = (
        Pos(118.0, +34.2, 34.2) * Rot(0, 90, 0) * Cylinder(radius=2.5, height=11.0)
        + Pos(118.0, +31.0, 32.2) * Box(4.0, 6.5, 3.0)
    )
    dummy_whistle_30_2 = Pos(-48.0, -7.5, z_axis + r_barrel + 2.5) * Cylinder(radius=1.8, height=7.2)
    gen_15_8 = (
        Pos(-84.0, -26.5, 63.2) * Rot(0, 90, 0) * Cylinder(radius=3.0, height=9.5)
        + Pos(-84.0, -21.8, 60.2) * Box(5.0, 8.5, 5.0)
    )
    cover_25_7 = (
        Pos(25.5, +32.5, 34.6) * Box(12.2, 6.8, 6.6)
        - Pos(24.3, +32.5, 33.8) * Box(12.2, 5.2, 6.6)
    )

    outer_solid = (
        smokebox + sb_saddle + door_ring_18_22 + door_dish_18_14 + plate_d51_18_18 + wheel_18_12_13 + hinge_straps_18_4
        + headlight + headlight_ped + feedwater_heater + fwh_brackets + chimney
        + barrel + dome_29_11 + firebox + cab + walkways + front_deck_and_beam + hatch_25_3
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
            Pos(0.5 * (-109.5 + 93.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=28.2, height=93.5 - (-109.5))
            & Pos(0.0, 0.0, 56.5) * Box(220.0, 60.0, 58.0)
        )
    )
    bore_smokebox = (
        Pos(0.5 * (92.5 + 119.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=24.2, height=119.5 - 92.5)
        + (
            Pos(0.5 * (92.5 + 160.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 1.8, height=160.0 - 92.5)
            & Pos(126.0, 0.0, 56.5) * Box(80.0, 70.0, 58.0)
        )
        + Pos(0.5 * (119.5 + 160.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 2.1, height=160.0 - 119.5)
    )
    bore_firebox = (
        Pos(0.5 * (x_fb_front + x_fb_rear), 0, z_axis - 13.5) * Box(fb_l - 6.0, 33.6, fb_h)
        + Pos(0.5 * (x_fb_front + x_fb_rear), 0, 40.0) * Box(fb_l - 2.0, 45.4, 38.0)
    )
    bore_dome_cab = (
        Pos(22.0, 0, 72.0) * Box(64.0, 21.0, 18.0)
        + Pos(cab_x - 1.8, 0, z_axis + 24.0) * Box(cab_len - 3.5, 58.0, 16.0)
        + Pos(136.0, 0, 21.5) * Box(31.0, 21.0, 10.0)
        + Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=4.1, height=30.0)
    )
    bore_chimney = Pos(x_chimney, 0, z_axis + r_sb + 6.0) * Cylinder(radius=6.0, height=34.0)
    rev_outer_slots = (
        Pos(25.5, +25.0, 26.5) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=16.0)
        + Pos(25.5, -25.0, 26.5) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=16.0)
    )
    rear_steam_grommet = Pos(-110.5, 0.0, 71.2) * Rot(0, 90, 0) * Cylinder(radius=2.4, height=6.0)
    safety_valve_holes = (
        Pos(-46.0, 0.0, 73.0) * Cylinder(radius=3.2, height=12.0)
        + Pos(-33.0, 0.0, 73.0) * Cylinder(radius=3.2, height=12.0)
    )
    firehole_outer_cut = Pos(-111.0, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=5.2, height=8.0)

    return (
        outer_solid
        - wheel_cuts
        - bore_barrel
        - bore_smokebox
        - bore_firebox
        - bore_dome_cab
        - bore_chimney
        - rev_outer_slots
        - rear_steam_grommet
        - safety_valve_holes
        - firehole_outer_cut
    )


# ==================== 4. 原厂真值四回路圆弧扫掠管网、车架内三软管、置换注油器、1x3 燃烧器与 8-20B 煤水车 ====================

def build_boiler_firebox_flues(cfg: Dict, pip_cfg: Dict):
    """
    12-6 无氧紫铜银焊受压锅筒 + 5 本煙管 + 干式火室罩接口 (イラスト 12, 16, 17, 100% Y 对称!):
      - 锅筒后端顶面 X=-98.5mm 设主蒸汽集汽座 (接 13-4)；
      - 锅筒顶中线 Y=0 纵列双安全阀衬套 X=-46.0 / -33.0mm (接 28-6x2)；
      - 锅筒底中线 X=-42.5mm 设排水衬套 (接 13-11)；彻底取消逆转横轴穿炉圆管孔！
    """
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
    firehole_ring = Pos(x_rear - 0.4, 0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=6.2, height=4.4)

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

    firehole_bore = Pos(x_rear + 1.0, 0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=4.6, height=12.0)
    blowdown_bush_bot = Pos(-42.5, 0.0, 17.0) * Cylinder(radius=2.5, height=2.6)
    steam_boss_rear = Pos(-98.5, 0.0, 71.2) * Cylinder(radius=3.8, height=3.8)
    sv_bushes_tandem = (
        Pos(-46.0, 0.0, 70.6) * Cylinder(radius=2.9, height=2.6)
        + Pos(-33.0, 0.0, 70.6) * Cylinder(radius=2.9, height=2.6)
    )

    return (
        shell_body + mud_ring + firehole_ring + stays + flues_outer
        + blowdown_bush_bot + steam_boss_rear + sv_bushes_tandem
        - flues_bore - firehole_bore
    )


def build_superheater_piping(cfg: Dict, pip_cfg: Dict):
    """
    原厂 100% 物理连通主蒸汽受热与给汽歧管单连通体 (イラスト 7, 12, 14, 17, 20 & 図 17.1, cc=1):
      - 炉内干汽管自后封头接头 (-112.8, -3.2, +48.0) 在铜锅筒内升至 (0.0, +58.8) 直通前管板 (+92.0)；
      - 17-7 ジャムナット + 17-8 16.0mm 六角アダプタ (+92..+108.5) + 17-6 中空ボルト (+108.5, 0, +58.8)；
      - 17-5 上下丸型クロスヘッダ (Z=+48.35 / +44.65, Y∈[-12.2,+12.2]) + **100% 水平直插零狗腿折弯**双U四程过热管；
      - 中央下垂管位于 20-7 喷嘴**后方**，经 17-4 蒸気吸入口 45° 锁入 **7-9 单根水平直通给汽歧管** (X=+123.5, Z=+18.5, 后高前低!)，
        并在 +Y=+7.5mm 底部设 4-2 置换注油毛细管专用螺座！
    """
    pv = pip_cfg["boiler_pressure_vessel"]
    sh = pip_cfg["superheater_and_steam_piping"]["superheater_4pass"]
    z_pipe = 58.8
    r_dry = 2.0
    x_turn = float(sh["x_turn_in_flue_mm"])  # -24.0 mm
    x_hdr = 108.5
    r_sh = 1.25
    up_yz = pv["five_flue_tubes"]["upper_superheater_flues"]["yz_centers_mm"]

    # 1. 后封头螺柱 (-112.8, -3.2, +48.0) -> 炉内干汽管 (Y=0, Z=+58.8) -> 前管板 17-7 + 17-8 16mm 六角接管 (+92.0..+108.5)
    backhead_stud = make_hex_nut((-112.8, -3.2, 48.0), (1.0, 0.0, 0.0), hex_r=2.05, length=2.0)
    dry_pipe_internal = make_swept_tube(
        [
            (-113.6, -3.2, 48.0),
            (-108.5, -3.2, 48.0),
            (-100.0, 0.0, z_pipe),
            (+93.0, 0.0, z_pipe),
        ],
        r_tube=r_dry,
        bend_r=2.8,
    )
    adapter_17_7_8 = (
        make_hex_nut((+93.4, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.75, length=2.0)
        + make_hex_nut((+100.8, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.55, length=13.4)
    )

    # 2. 17-6 中空螺栓 + 17-5 上环眼、上下丸型集箱 (Z=+48.35 / +44.65) 与 100% 水平直插双U四程过热管 (零狗腿折弯!)
    z_hdr_hi, z_hdr_lo = 48.35, 44.65
    banjo_17_6 = (
        Pos(x_hdr, 0.0, z_pipe) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=3.0)
        + make_hex_nut((x_hdr + 2.2, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.6, length=1.8)
        + Pos(x_hdr, 0.0, 0.5 * (z_hdr_hi + z_pipe)) * Cylinder(radius=1.55, height=z_pipe - z_hdr_hi + 1.2)
    )
    hdr_upper = Pos(x_hdr, 0.0, z_hdr_hi) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=24.8)
    hdr_lower = Pos(x_hdr, 0.0, z_hdr_lo) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=24.8)

    sh_loops = None
    for yz in up_yz:
        yf = float(yz[0])
        u_loop = make_swept_tube(
            [
                (x_hdr, yf, z_hdr_hi),
                (x_turn, yf, z_hdr_hi),
                (x_turn, yf, z_hdr_lo),
                (x_hdr, yf, z_hdr_lo),
            ],
            r_tube=r_sh,
            bend_r=1.65,
        )
        sh_loops = u_loop if sh_loops is None else (sh_loops + u_loop)

    # 3. 17-5 一体中央垂直下垂管 -> 17-4 45° 中空螺栓 -> 7-9 单根水平直通给汽歧管 (X=+123.5, Z=+18.5) + 右偏 (+7.5mm) 注油螺座
    drop_leg_17_5 = make_swept_tube(
        [
            (x_hdr, 0.0, z_hdr_lo),
            (x_hdr, 0.0, 26.5),
            (119.5, 0.0, 21.8),
            (123.5, 0.0, 18.5),
        ],
        r_tube=1.60,
        bend_r=4.0,
    )
    bolt_17_4 = (
        Pos(120.6, 0.0, 21.0) * Rot(0, 52, 0) * Cylinder(radius=2.8, height=3.0)
        + make_hex_nut((119.0, 0.0, 22.3), (-0.788, 0.0, +0.616), hex_r=2.4, length=1.7)
    )
    manifold_7_9 = (
        Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=3.55, height=41.6)
        + Pos(123.5, +19.5, 18.5) * Rot(90, 0, 0) * Cylinder(radius=3.95, height=2.2)
        + Pos(123.5, -19.5, 18.5) * Rot(90, 0, 0) * Cylinder(radius=3.95, height=2.2)
        + Pos(123.5, +7.5, 16.4) * Cylinder(radius=2.15, height=3.8)
    )

    return backhead_stud + dry_pipe_internal + adapter_17_7_8 + banjo_17_6 + hdr_upper + hdr_lower + sh_loops + drop_leg_17_5 + bolt_17_4 + manifold_7_9


def build_exhaust_blast_pipe(cfg: Dict, pip_cfg: Dict):
    """
    7-8 前低位排汽横歧管 (X=+130.0, Z=+12.5mm) + 20-7 底部旋入式光滑锥形乏汽喷嘴 (イラスト 7, 20 & 図 20.1, 100% Y 对称, cc=1):
      - 按 図 20.1 修正 20-7 安装结构：一字螺丝刀法兰头位于 7-8 方块**正下方** (Z=+8.2mm)，上方为光滑车削黄铜锥管 (彻底删除顶部假六角螺母!)。
    """
    ej = pip_cfg["superheater_and_steam_piping"]["smokebox_draft_ejector"]
    x_ch = float(ej["chimney_axis_x_mm"])     # +130.0 mm
    z_tip = float(ej["blast_nozzle_tip_z_mm"])  # +35.0 mm
    r_orifice = 0.5 * float(ej["blast_nozzle_orifice_dia_mm"])  # 2.1 mm
    z_exh = 12.5

    block_7_8 = Pos(x_ch, 0.0, z_exh) * Box(8.2, 8.8, 7.6)
    pipes_7_8 = Pos(x_ch, 0.0, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.45, height=41.6)
    collars = (
        Pos(x_ch, +19.5, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.95, height=2.0)
        + Pos(x_ch, -19.5, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.95, height=2.0)
    )
    bottom_head_20_7 = (
        Pos(x_ch, 0.0, 8.1) * Cylinder(radius=3.5, height=1.6)
        - Pos(x_ch, 0.0, 7.3) * Box(7.4, 0.85, 0.8)
    )
    nozzle_20_7 = Pos(x_ch, 0.0, 0.5 * (16.0 + z_tip)) * Cone(
        bottom_radius=3.55, top_radius=2.95, height=z_tip - 16.0
    )
    orifice_bore = Pos(x_ch, 0.0, z_tip - 4.5) * Cylinder(radius=r_orifice, height=10.0)
    return block_7_8 + pipes_7_8 + collars + bottom_head_20_7 + nozzle_20_7 - orifice_bore


def build_blower_nozzle_17_3(cfg: Dict, pip_cfg: Dict):
    """
    铜锅筒内部中空通风撑管 (-112.8..+93.5) + 烟室 17-3 通風ノズル 低位水平走线 + 180° J 字上弯喷嘴 (イラスト 14, 17, 20 & 図 20.2, cc=1):
      - 严格自过热集箱**下方** (+93.5, +4.0, +40.5mm，上下烟管之间空白区) 穿出前管板，绝不横切 17-5 过热器！
      - 降至 Z=+29.5mm 后 100% 水平直行至 20-7 后侧，再按 図 20.2 做紧凑 180° 底部 U 弯竖直向上指向烟囱中心 (+130.0, +3.6, +35.2mm)！
    """
    rear_stud = make_hex_nut((-112.8, +5.1, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=2.0)
    front_union = make_hex_nut((+93.6, +4.0, 40.5), (1.0, 0.0, 0.0), hex_r=1.95, length=2.2)
    stay_and_j_nozzle = make_swept_tube(
        [
            (-113.6, +5.1, 48.0),
            (-104.0, +4.0, 40.5),
            (+96.2, +4.0, 40.5),
            (+96.2, +4.0, 29.5),
            (+125.0, +3.6, 29.5),
            (+125.0, +3.6, 24.8),
            (+130.0, +3.6, 24.8),
            (+130.0, +3.6, 35.2),
        ],
        r_tube=1.00,
        bend_r=2.2,
    )
    return rear_stud + front_union + stay_and_j_nozzle


def build_smokebox_petticoat(cfg: Dict, pip_cfg: Dict):
    """
    18-28 ペチコート + 18-27 煙室座 (イラスト 18, 100% Y 对称, 零虚假支撑杆, cc=1):
      - 修正夸张大喇叭斗失真！按 イラスト 18 还原为 ø13.2mm 直筒衬套 + 底部微翻边 ø15.2mm (Z=+57.5..+91.2mm)，与后方 17-5 过热集箱保持 >9.5mm 净空！
    """
    ej = pip_cfg["superheater_and_steam_piping"]["smokebox_draft_ejector"]
    x_ch = float(ej["chimney_axis_x_mm"])
    z_pet = float(ej["petticoat_entry_z_mm"])     # +57.5 mm
    r_pet_th = 0.5 * float(ej["petticoat_throat_dia_mm"])  # 6.5 mm -> inner 5.6, outer 6.6
    r_pet_fl = 0.5 * float(ej["petticoat_flare_dia_mm"])   # 7.6 mm
    z_ch_top = 91.2
    petticoat_outer = (
        Pos(x_ch, 0, z_pet + 1.4) * Cone(bottom_radius=r_pet_fl, top_radius=r_pet_th + 0.1, height=2.8)
        + Pos(x_ch, 0, 0.5 * (z_pet + 2.6 + z_ch_top)) * Cylinder(radius=r_pet_th + 0.1, height=z_ch_top - (z_pet + 2.6))
        + Pos(x_ch, 0, z_ch_top - 0.8) * Cylinder(radius=r_pet_th + 1.4, height=1.6)
        + Pos(x_ch, 0, 74.2) * Cylinder(radius=9.4, height=2.0)
    )
    petticoat_inner = (
        Pos(x_ch, 0, z_pet + 1.4) * Cone(bottom_radius=r_pet_fl - 0.9, top_radius=r_pet_th - 0.8, height=3.4)
        + Pos(x_ch, 0, 0.5 * (z_pet + 2.2 + z_ch_top + 2.0)) * Cylinder(radius=r_pet_th - 0.8, height=(z_ch_top + 2.0) - (z_pet + 2.2))
    )
    return petticoat_outer - petticoat_inner


def build_tandem_safety_valves(cfg: Dict, pip_cfg: Dict):
    """
    28-6 x2 纵列双安全阀 (イラスト 12, 28, Y=0.0mm 中线前后串联于 X=-46.0 / -33.0mm，100% 零外接管线!, cc=1)。
    """
    pad = Pos(-39.5, 0.0, 70.8) * extrude(SlotOverall(width=20.0, height=6.4), amount=2.2)
    valves = None
    for sx in (-46.0, -33.0):
        v = (
            make_hex_nut((sx, 0.0, 73.2), (0.0, 0.0, 1.0), hex_r=2.8, length=2.0)
            + Pos(sx, 0.0, 76.8) * Cylinder(radius=2.35, height=7.4)
            - Pos(sx, 0.0, 79.8) * Cylinder(radius=1.35, height=2.2)
        )
        valves = v if valves is None else (valves + v)
    return pad + valves


def build_feedwater_piping(cfg: Dict, pip_cfg: Dict):
    """
    原厂车架内给水闭环 + 驾驶室穿地板短立管 6-8 (|Y| <= 11.0mm, イラスト 4, 6, 8, 13, cc=1):
      - 左后 (+Y=+7.2mm) 6-11 口金 -> 6-10 吸水管 -> 6-5 中央轴动泵；
      - 6-6 向后吐出管直通驾驶室右后地板下方 4-17 旁通阀 (-134.5, -8.8, +14.8, 与地板上 6-9/8-7 同轴!)；
      - 4-16 回水管接右后尾嘴 (-150.0, -7.2, -9.2)；
      - **6-8 短给水立管**自 4-17 横口引至 (-116.0, -9.1)，穿过 8-11 地板短升至 Z=+26.4mm 直承后封头**右下低位** 13-2 逆止阀 (彻底消除横穿表盘下方长黄管失真!)。
    """
    fw = pip_cfg["feedwater_piping_topology"]
    ax_loop = fw["axle_and_shared_hand_pump_loop"]
    x_pump = float(ax_loop["pump_x_mm"])
    x_tw = x_pump - 16.8  # -38.8 mm

    pump_barrel = Pos(x_pump - 15.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.6, height=7.2)
    gland_nut = make_hex_nut((x_pump - 11.4, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=4.1, length=2.0)
    check_tower = Pos(x_tw, 0.0, 1.5) * Cylinder(radius=3.0, height=16.5)
    top_hex_cap = make_hex_nut((x_tw, 0.0, 10.4), (0.0, 0.0, 1.0), hex_r=3.7, length=2.2)
    bot_hex_cap = make_hex_nut((x_tw, 0.0, -7.4), (0.0, 0.0, 1.0), hex_r=3.7, length=2.0)
    pump_bracket = Pos(x_pump - 15.0, 0.0, 4.2) * Box(5.4, 21.0, 2.4)
    ram_bore = Pos(x_pump - 14.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.90, height=8.4)
    pump_6_5 = pump_barrel + gland_nut + check_tower + top_hex_cap + bot_hex_cap + pump_bracket - ram_bore

    pipe_6_10 = (
        make_swept_tube(
            [
                (-150.8, +7.2, -9.2),
                (-142.5, +7.2, -9.2),
                (-134.5, +7.2, +8.5),
                (x_tw, +7.2, +8.5),
                (x_tw, +7.2, -5.2),
                (x_tw, 0.0, -5.2),
            ],
            r_tube=1.22,
            bend_r=3.0,
        )
        + Pos(-149.2, +7.2, -9.2) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=3.8)
        + make_hex_nut((-147.2, +7.2, -9.2), (1.0, 0.0, 0.0), hex_r=2.35, length=2.0)
    )

    bx, by, bz = -134.5, -8.8, 14.8
    pipe_6_6 = make_swept_tube(
        [
            (x_tw, 0.0, +10.2),
            (x_tw, by, +13.5),
            (-126.5, by, +14.8),
            (bx, by, bz),
        ],
        r_tube=1.20,
        bend_r=2.6,
    )
    valve_4_17 = (
        Pos(-130.8, by, bz) * Box(10.2, 3.6, 4.2)
        + Pos(bx, by, bz + 1.8) * Cylinder(radius=1.80, height=6.4)
        + make_banjo_eye((-127.0, -9.0, bz), (0.0, 0.0, 1.0), r_eye=1.75, h_eye=2.0, hex_r=1.45)
    )

    # 6-8 短给水立管 (|-9.1 - 1.80| = 10.90mm <= 11.0mm! 直插低位右下 13-2 底部 Z=+26.0mm)
    riser_6_8 = (
        make_swept_tube(
            [
                (-127.0, -9.0, bz),
                (-116.0, -9.1, +16.0),
                (-116.0, -9.1, +26.8),
            ],
            r_tube=1.15,
            bend_r=2.4,
        )
        + make_hex_nut((-116.0, -9.1, +25.8), (0.0, 0.0, 1.0), hex_r=1.80, length=2.0)
    )

    pipe_4_16_return = (
        make_swept_tube(
            [
                (-131.5, by, bz - 1.6),
                (-138.5, -7.2, +6.2),
                (-143.5, -7.2, -9.2),
                (-150.8, -7.2, -9.2),
            ],
            r_tube=1.18,
            bend_r=2.4,
        )
        + Pos(-149.2, -7.2, -9.2) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=3.8)
        + make_hex_nut((-147.2, -7.2, -9.2), (1.0, 0.0, 0.0), hex_r=2.35, length=2.0)
    )

    return pump_6_5 + pipe_6_10 + pipe_6_6 + valve_4_17 + riser_6_8 + pipe_4_16_return


def build_displacement_lubricator(cfg: Dict, pip_cfg: Dict):
    """
    4-2 前车架中央卧式置换油箱 + 4-1 顶盖 + 左偏 (+Y=+7.5mm) 避让式毛细出油管 (イラスト 4, 7, 20):
      - 彻底消除 Y=0 出油管直撞 7-8 排汽座失真！出油毛细管沿车架内侧 +Y=+7.5mm (|Y|<=9.6mm<=10.5mm) 绕过 7-8，垂直向上锁入后方 7-9 给汽管右偏油口 (+123.5, +7.5, +16.2mm)！
    """
    lub = pip_cfg["displacement_lubricator_system"]
    x0, x1 = [float(v) for v in lub["x_range_mm"]]
    xm = 0.5 * (x0 + x1)
    zc = float(lub["z_center_mm"])
    r_tank = 0.5 * float(lub["tank_od_mm"])
    cap_x, _, cap_z = [float(v) for v in lub["cap_xyz_mm"]]
    tx, ty, tz = [float(v) for v in lub["oil_delivery_tee_xyz_mm"]]  # (123.5, +7.5, 16.2)

    tank_4_2 = Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_tank, height=x1 - x0)
    saddle_cradle = Pos(xm, 0.0, zc + 1.5) * Box(12.0, 19.2, 4.2)  # ±9.6mm 保持 |min_y+max_y|=0 且 <=10.5mm!
    neck_4_1 = Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z)) * Cylinder(radius=2.8, height=cap_z - zc + 1.0)
    cap_4_1 = Pos(cap_x - 1.5, 0.0, cap_z) * Cylinder(radius=3.8, height=2.2)

    oil_pipe = make_swept_tube(
        [(x0 + 2.0, +2.6, zc), (x0 - 2.5, ty, zc), (tx, ty, zc), (tx, ty, tz + 0.6)],
        r_tube=1.05,
        bend_r=2.8,
    )
    union_nut = make_hex_nut((tx, ty, tz - 1.0), (0.0, 0.0, 1.0), hex_r=1.85, length=1.8)
    return tank_4_2 + saddle_cradle + neck_4_1 + cap_4_1 + oil_pipe + union_nut


def build_boiler_aux_valves(cfg: Dict, pip_cfg: Dict):
    """
    驾驶室后封头与火室侧原厂多图联立 100% 真值铜阀座与外管网单连通总成 (イラスト 11, 12, 13, 14 & Manual p.43/p.45, cc=1):
      1. 13-4 主蒸気供給管 + 11-15 横置蒸汽分配器 + 11-10 双耳填料压盖 + 一体左下 11-14 通风阀座；
      2. 13-9 加減管 (-X バンジョー -> 后封头 (-112.8, -3.2, +48.0)) + 13-8 通風管 (-> (-112.8, +5.1, +48.0))；
      3. PG-ø20 上仰 24° 黄铜表圈 (-121.5, -15.4, +66.8) + 紧凑 13-1 180° 垂直大 U 弯虹吸管 (谷底 Z=+42.4mm，高居火门上方!)；
      4. **13-2 逆止弁 + 13-7 六角顶盖**: 校正至观火孔右下低位衬套 (-115.8, -9.1, +30.5mm)，下承短立管 6-8；
      5. 右外缘 (-118.5, -31.3) 11-4~6 水面计上下座 + **同轴 13-5#1/#2 垂直中空六角螺栓与バンジョー环** (零侧向直角焊接!) + 13-6/13-10；
      6. **11-1 左端汽笛阀** (-115.8, +18.2, +66.5, 带顶置 (1-1) 螺帽) -> **100% 全直垂降 13-3 汽笛管** -> **11-8 左火室颊下真汽笛**；
      7. **11-16 左床下横向 (+Y) 锅炉排水针阀体** (-115.5, +19.0, +15.8): 针阀孔横向朝左外侧 (+Y 接 11-17)，排污嘴垂直朝下 (-Z)，内口接 13-11 前伸至锅底 (-42.5, 0, +17.0)！
    """
    wg = pip_cfg["pumps_and_valves_bom"]["water_gauge_11_7"]
    x_ring = -113.0  # 隐埋于 11-9 背板内部的银焊内衬环 (外观不可见，保障全件单连通 cc=1)

    brazed_backhead_ring = (
        Pos(x_ring, 0.0, 49.5) * Rot(0, 90, 0) * (Cylinder(radius=19.2, height=1.3) - Cylinder(radius=15.0, height=3.0))
        + Pos(x_ring, +18.2, 32.5) * Box(1.5, 5.2, 37.0)
        + Pos(x_ring, -15.5, 31.5) * Box(1.6, 16.0, 4.0)
        + Pos(x_ring, 0.0, 66.0) * Box(1.6, 6.0, 5.0)
    )

    # 1 & 2. 13-4 主蒸気供給管 + 11-15 蒸気分配器 (含一体左下 11-14 通风阀颈) + 11-10 双耳压盖
    jumper_13_4 = (
        make_swept_tube(
            [(-98.5, 0.0, 73.2), (-102.5, 0.0, 73.2), (-111.5, 0.0, 69.5), (-115.8, 0.0, 69.5)],
            r_tube=1.42,
            bend_r=2.8,
        )
        + make_banjo_eye((-98.5, 0.0, 73.2), (0.0, 0.0, 1.0), r_eye=2.7, h_eye=2.3, hex_r=2.0)
        + make_banjo_eye((-115.8, 0.0, 69.5), (0.0, 0.0, 1.0), r_eye=2.5, h_eye=2.1, hex_r=1.9)
    )
    manifold_11_15 = (
        Pos(-115.8, +2.0, 66.5) * Box(5.4, 26.8, 5.6)
        + Pos(-114.2, 0.0, 66.5) * Box(3.0, 7.6, 4.0)
    )
    gland_collar_11_10 = (
        Pos(-119.2, 0.0, 68.5) * Box(2.0, 9.2, 3.0)
        + Pos(-120.2, 0.0, 68.5) * Rot(0, 90, 0) * Cylinder(radius=2.0, height=2.2)
        + Pos(-120.4, +3.5, 68.5) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=1.0)
        + Pos(-120.4, -3.5, 68.5) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=1.0)
    )

    # 3. 13-9 加減管 (11-15 后侧 -X バンジョー -> 后封头炉内干汽管接头 (-112.8, -3.2, +48.0))
    jumper_13_9 = (
        make_banjo_eye((-118.2, -3.2, 65.2), (-1.0, 0.0, 0.0), r_eye=2.0, h_eye=1.8, hex_r=1.55)
        + make_swept_tube(
            [(-118.2, -3.2, 65.2), (-118.2, -3.2, 48.0), (-112.8, -3.2, 48.0)],
            r_tube=1.30,
            bend_r=2.2,
        )
        + make_hex_nut((-114.2, -3.2, 48.0), (1.0, 0.0, 0.0), hex_r=1.95, length=1.8)
    )

    # 4. 11-15 左下一体通风阀颈 11-14 (+5.1, +64.2) + 13-8 通風管 (-> (-112.8, +5.1, +48.0))
    blower_11_14_and_13_8 = (
        Pos(-117.2, +5.1, 64.2) * Rot(0, 90, 0) * Cylinder(radius=1.70, height=3.2)
        + make_hex_nut((-115.8, +5.1, 62.6), (0.0, 0.0, 1.0), hex_r=1.70, length=1.5)
        + make_swept_tube(
            [(-115.8, +5.1, 63.2), (-115.8, +5.1, 48.0), (-112.8, +5.1, 48.0)],
            r_tube=1.08,
            bend_r=2.0,
        )
        + make_hex_nut((-114.2, +5.1, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=1.8)
    )

    # 5. PG-ø20 严格垂直朝后 (-X, 0.0° 零仰角!) 黄铜表壳 (-122.0, -15.4, +66.8) + 13-1 同轴正交底入 180° 垂直大 U 弯虹吸管
    yp, zp = -15.4, 66.8
    pg_bezel = (
        Pos(-122.0, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=9.8, height=3.6)
        + Pos(-122.0, yp, zp - 10.6) * Cylinder(radius=1.95, height=3.4)
        - Pos(-123.6, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=8.45, height=2.4)
    )
    siphon_13_1 = (
        make_banjo_eye((-118.2, -0.8, 65.4), (-1.0, 0.0, 0.0), r_eye=1.70, h_eye=1.6, hex_r=1.35)
        + make_swept_tube(
            [
                (-118.2, -0.8, 65.4),
                (-119.5, -0.8, 46.8),
                (-120.8, -8.1, 42.4),
                (-122.0, yp, 46.8),
                (-122.0, yp, 57.5),
            ],
            r_tube=0.98,
            bend_r=4.2,
        )
        + make_hex_nut((-122.0, yp, 55.2), (0.0, 0.0, 1.0), hex_r=1.80, length=2.0)
    )

    # 6. 13-2 逆止弁 + 13-7 六角顶盖: 校正至观火孔右下低位衬套 (-115.8, -9.1, +30.5)，底部 Z=+26.0 直承短立管 6-8！
    clack_13_2_7 = (
        Pos(-115.8, -9.1, 30.5) * Cylinder(radius=2.45, height=8.8)
        + make_hex_nut((-115.8, -9.1, 35.3), (0.0, 0.0, 1.0), hex_r=2.75, length=1.9)
        + Pos(-114.2, -9.1, 30.5) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=4.4)
        + make_hex_nut((-113.8, -9.1, 30.5), (1.0, 0.0, 0.0), hex_r=2.40, length=1.5)
    )

    # 7. 右外缘 (-118.5, -31.3, Z_c=+46.0) 11-4~6 水面计上下座 + 同轴垂直 13-5#1/#2 バンジョー螺栓 + 13-6/13-10
    yw = float(wg["y_mm"])  # -31.3 mm
    wg_top = (
        Pos(-118.5, yw, 57.2) * Cylinder(radius=2.35, height=5.2)
        + make_hex_nut((-118.5, yw, 55.1), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + make_banjo_eye((-118.5, yw, 60.6), (0.0, 0.0, 1.0), r_eye=2.35, h_eye=2.0, hex_r=1.85)
        + make_swept_tube(
            [(-118.5, yw, 60.6), (-118.5, -18.0, 60.6), (-108.0, -17.5, 60.5)],
            r_tube=1.15,
            bend_r=2.6,
        )
        + make_banjo_eye((-108.0, -17.5, 60.5), (0.0, +0.5, +0.86), r_eye=2.05, h_eye=1.9, hex_r=1.55)
        + Pos(x_ring, -16.0, 57.0) * Box(1.6, 5.6, 6.8)
    )
    wg_bot = (
        Pos(-118.5, yw, 33.6) * Cylinder(radius=2.35, height=6.4)
        + make_hex_nut((-118.5, yw, 36.9), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + Pos(-118.5, yw, 30.2) * Cylinder(radius=2.35, height=2.0)
        + make_swept_tube(
            [(-118.5, yw, 30.2), (-118.5, -25.2, 30.2), (-110.5, -23.5, 33.5), (-48.0, -23.5, 35.0)],
            r_tube=1.15,
            bend_r=2.6,
        )
        + make_banjo_eye((-48.0, -23.5, 35.0), (0.0, +1.0, 0.0), r_eye=2.05, h_eye=1.9, hex_r=1.55)
    )

    # 8. 11-1 左端汽笛阀 (-115.8, +18.2, +66.5, 带顶置 (1-1) 一字螺帽) -> 100% 全直垂直 13-3 汽笛管 -> 11-8 左颊下真汽笛
    wh_valve_11_1 = (
        Pos(-115.8, +18.2, 66.5) * Box(4.4, 4.8, 8.8)
        + Pos(-115.8, +15.5, 66.5) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)
        + Pos(-115.8, +18.2, 71.5) * Cylinder(radius=1.85, height=1.6)
        - Pos(-115.8, +18.2, 72.2) * Box(3.8, 0.55, 0.6)
    )
    pipe_13_3 = (
        make_banjo_eye((-118.2, +18.2, 65.5), (-1.0, 0.0, 0.0), r_eye=1.75, h_eye=1.8, hex_r=1.38)
        + make_swept_tube(
            [
                (-115.8, +18.2, 65.5),
                (-118.2, +18.2, 65.5),
                (-118.2, +18.2, 25.5),
                (-113.5, +20.2, 25.5),
            ],
            r_tube=1.00,
            bend_r=1.8,
        )
        + make_banjo_eye((-117.6, +18.2, 25.5), (1.0, 0.0, 0.0), r_eye=1.75, h_eye=1.8, hex_r=1.38)
    )
    whistle_11_8 = (
        make_hex_nut((-114.6, +20.2, 25.5), (1.0, 0.0, 0.0), hex_r=2.85, length=1.8)
        + Pos(-92.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=3.10, height=44.4)
        - Pos(-102.0, +20.2, 23.2) * Box(3.0, 4.8, 3.0)
        + Pos(x_ring, +19.5, 25.5) * Box(1.6, 4.0, 4.0)
    )

    # 9. 11-16 左床下横向 (+Y) 锅炉排水针阀体 (-115.5, +19.0, +15.8) + 垂直向下排污嘴 (-Z) + 13-11 前伸配水管
    blowdown_11_16_and_13_11 = (
        Pos(-115.5, +18.6, 15.8) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=4.4)
        + Pos(-113.2, +18.2, 16.5) * Box(3.2, 2.2, 4.0)
        + Pos(-115.5, +19.2, 13.4) * Cylinder(radius=1.02, height=4.0)
        + make_banjo_eye((-115.5, +16.2, 15.8), (0.0, -1.0, 0.0), r_eye=2.0, h_eye=1.8, hex_r=1.50)
        + make_swept_tube(
            [
                (-115.5, +16.2, 15.8),
                (-108.0, +16.2, 15.2),
                (-72.0, +16.2, 15.2),
                (-54.0, +7.2, 15.4),
                (-42.5, 0.0, 15.6),
                (-42.5, 0.0, 17.2),
            ],
            r_tube=1.18,
            bend_r=3.2,
        )
        + make_banjo_eye((-42.5, 0.0, 16.8), (0.0, 0.0, 1.0), r_eye=2.15, h_eye=2.2, hex_r=1.65)
    )

    return (
        brazed_backhead_ring + jumper_13_4 + manifold_11_15 + gland_collar_11_10
        + jumper_13_9 + blower_11_14_and_13_8 + pg_bezel + siphon_13_1
        + clack_13_2_7 + wg_top + wg_bot
        + wh_valve_11_1 + pipe_13_3 + whistle_11_8 + blowdown_11_16_and_13_11
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
    8-20B 四轴煤水车外衣壳体与台架总成 (イラスト 32, 33, 35 & p45-p47 — 100% Y 对称, |min_y+max_y|=0, com_y=0, cc=1):
      - 32-10 平坦槽边底架 + **32-11x2 海鸥翼下沉枕梁 + 32-12x2 心皿ボス** + 32-1/32-14 前后端梁 + (28-2) 后车钩 + 四角踏板；
      - **33-10 原厂平滑 S 曲线阶梯煤库侧壁** (消除折线锯齿!) + **35-5 x2 左右固定横向后挡煤板 (X=-52.1, Y=±[16.7,39.8])**；
      - **35-1 主甲板 (Z_tdr=+62.0mm)**: 前部 36-4 方孔 + 中后部圆角防波给水口凸缘 + **后部 X=-89.5 固定一体实心装饰人孔铸件 35-6** (删除虚假铰链水箱小圆盖刚体!)。
    """
    t_cfg = cfg["tender_and_track"]
    L, W = float(t_cfg["tender_body_length"]), float(t_cfg["tender_body_width"])  # 200.7, 81.5 mm
    half_span = 0.5 * float(t_cfg["bogie_center_span"])  # 56.25 mm

    web_32_10 = Pos(0.0, 0.0, 14.2) * Box(L + 5.6, W, 1.4)
    sk_L = (
        Pos(0.0, +(0.5 * W - 0.7), 13.05) * Box(L + 5.6, 1.4, 3.7)
        + Pos(+100.8, +39.2, 6.2) * Box(3.2, 1.2, 10.2)
        + Pos(+100.8, +37.8, 1.6) * Box(3.2, 3.6, 1.0)
        + Pos(-102.2, +39.0, 6.6) * Box(2.8, 1.2, 9.4)
        + Pos(-102.2, +37.6, 2.4) * Box(2.8, 3.6, 1.0)
    )
    skirts_and_steps = sk_L + mirror(sk_L, about=Plane.XZ)

    front_beam_32_1 = Pos(+0.5 * L + 2.8, 0.0, 7.2) * Box(2.8, W, 13.4)
    rear_beam_32_14 = Pos(-0.5 * L - 2.8, 0.0, 8.2) * Box(2.8, W, 11.4)
    hose_guide_32_3 = (
        Pos(+104.0, 0.0, +7.0) * Rot(0, 90, 0) * (Cylinder(radius=4.1, height=1.6) - Cylinder(radius=2.9, height=4.0))
        + Pos(+104.0, 0.0, +11.2) * Box(1.8, 3.6, 3.2)
    )
    rear_coupler_28_2 = (
        Pos(-106.5, 0.0, 7.2) * Box(6.4, 9.2, 5.2)
        + Pos(-111.0, 0.0, 7.2) * Cylinder(radius=3.6, height=5.8)
    )

    bolsters_32_11_12 = None
    for bx in (-half_span, +half_span):
        saddle = Pos(bx, 0.0, 3.4) * Box(11.6, 17.6, 1.6)
        boss_32_12 = Pos(bx, 0.0, 1.55) * Cylinder(radius=6.8, height=2.1) + Pos(bx, 0.0, 0.1) * Cylinder(radius=3.8, height=1.8)
        wing_L = (
            Pos(bx, +25.5, 12.9) * Box(11.6, 13.0, 1.6)
            + Pos(bx, +13.8, 8.15) * Rot(+38.0, 0, 0) * Box(11.6, 14.6, 1.6)
        )
        b_one = saddle + boss_32_12 + wing_L + mirror(wing_L, about=Plane.XZ)
        bolsters_32_11_12 = b_one if bolsters_32_11_12 is None else (bolsters_32_11_12 + b_one)

    shell_outer = Pos(0.0, 0.0, 38.4) * Box(L, W, 47.2)
    shell_hollow = Pos(0.0, 0.0, 38.1) * Box(L - 3.2, W - 3.2, 45.2)
    sump_floor_cut = Pos(+28.4, 0.0, 13.5) * Box(31.0, 29.6, 6.0)
    cutout_1_alc = Pos(+29.75, 0.0, 61.5) * Box(61.5, 61.6, 6.0)
    cutout_2_water = Pos(-45.0, 0.0, 61.5) * Box(63.0, 31.6, 8.0)
    front_gangway_cut = Pos(+93.6, 0.0, 61.5) * Box(14.2, 28.0, 4.0)

    # 给水敞口周围一体式 1.8mm 圆角防波缘 (内孔严格由 cutout_2_water 切净，零遮挡 [-76.5,-13.5]x[-15.8,+15.8])
    coaming_ring = (
        Pos(-45.0, 0.0, 62.6) * Box(65.4 - 5.2, 34.0, 2.4)
        + Pos(-45.0, 0.0, 62.6) * Box(65.4, 34.0 - 5.2, 2.4)
    )
    for sx_c in (-1.0, +1.0):
        for sy_c in (-1.0, +1.0):
            coaming_ring += Pos(-45.0 + sx_c * (32.7 - 2.6), sy_c * (17.0 - 2.6), 62.6) * Cylinder(radius=2.6, height=2.4)

    tank_shell = (shell_outer + coaming_ring) - shell_hollow - sump_floor_cut - cutout_1_alc - cutout_2_water - front_gangway_cut

    # 33-10 平滑三次样条 S 曲线煤库侧壁 + 35-5 固定横向后煤壁 + 35-2 前斜煤槽壁 + 33-3 前阶梯 + 37-2 前 U 扶手 (单侧构造后精确 XZ 镜像确保 com_y==0!)
    yw_in = 0.5 * W - 1.70
    s_top_pts = [
        Vector(-74.0, 0.0, 61.9),
        Vector(-68.5, 0.0, 62.8),
        Vector(-60.5, 0.0, 69.2),
        Vector(-54.8, 0.0, 74.8),
        Vector(-52.1, 0.0, 75.55),
    ]
    w_bunker_xz = Wire([
        Edge.make_line(Vector(-74.0, 0.0, 61.2), s_top_pts[0]),
        Edge.make_spline(s_top_pts),
        Edge.make_line(s_top_pts[-1], Vector(+91.4, 0.0, 75.55)),
        Edge.make_line(Vector(+91.4, 0.0, 75.55), Vector(+91.4, 0.0, 63.2)),
        Edge.make_line(Vector(+91.4, 0.0, 63.2), Vector(+100.35, 0.0, 63.2)),
        Edge.make_line(Vector(+100.35, 0.0, 63.2), Vector(+100.35, 0.0, 61.2)),
        Edge.make_line(Vector(+100.35, 0.0, 61.2), Vector(-74.0, 0.0, 61.2)),
    ])
    side_wall_L = Pos(0.0, yw_in, 0.0) * extrude(Face(w_bunker_xz), amount=1.70, dir=(0.0, 1.0, 0.0))
    fixed_partition_35_5_L = Pos(-52.1, +28.25, 68.525) * Box(1.5, 23.1, 14.05)
    v_wall_35_2_L = Pos(+80.0, +26.2, 68.5) * Rot(0, 0, -63.8) * Box(29.6, 1.5, 13.2)
    handrail_37_2_L = make_swept_tube(
        [(+99.4, +39.1, 61.8), (+99.4, +39.1, 74.6), (+94.6, +39.1, 74.6), (+94.6, +39.1, 62.2)],
        r_tube=0.62,
        bend_r=1.3,
    )
    front_rungs_33_3_L = (
        Pos(+97.5, +40.4, 34.0) * Box(2.8, 1.4, 1.0)
        + Pos(+97.5, +40.4, 48.0) * Box(2.8, 1.4, 1.0)
        + Pos(+98.2, +37.5, 18.2) * Rot(90, 0, 0) * Cylinder(radius=1.3, height=3.0)
    )
    bunker_L = side_wall_L + fixed_partition_35_5_L + v_wall_35_2_L + handrail_37_2_L + front_rungs_33_3_L
    bunker_sides = bunker_L + mirror(bunker_L, about=Plane.XZ)

    coal_gate_33_4 = Pos(+86.5, 0.0, 66.8) * Box(1.8, 26.8, 10.2)
    static_manhole_35_6 = (
        Pos(-89.5, 0.0, 62.6) * Cylinder(radius=6.6, height=1.6)
        + Pos(-89.5, 0.0, 63.6) * Cylinder(radius=5.3, height=1.0)
        + Pos(-83.0, 0.0, 62.6) * Box(2.4, 3.0, 1.4)
        + Pos(-96.0, 0.0, 62.6) * Box(2.2, 2.2, 1.4)
    )

    return (
        web_32_10 + skirts_and_steps + front_beam_32_1 + rear_beam_32_14 + hose_guide_32_3
        + rear_coupler_28_2 + bolsters_32_11_12 + tank_shell + bunker_sides + coal_gate_33_4 + static_manhole_35_6
    )


def build_tender_rear_ladder():
    """
    35-12 x2 左后甲板双弧形扶手 + 33-11 x3 左后独立 U 型梯级 (イラスト 33, 35 & p45/p47 原厂左后 +Y 乘务员扶梯, cc=1):
      - 将单连通暗筋移入煤水车后壁厚度内部 (X=-99.4mm)，外部仅呈现 3 根干净利落的独立 U 形梯级与顶部双弯管扶手！
    """
    hidden_inner_spine = Pos(-99.4, +32.5, 42.5) * Box(1.2, 8.6, 40.2)
    rungs = None
    for zr in (24.5, 37.5, 50.5):
        rg = make_swept_tube(
            [(-99.2, +28.8, zr), (-102.0, +28.8, zr), (-102.0, +36.2, zr), (-99.2, +36.2, zr)],
            r_tube=0.62,
            bend_r=1.0,
        )
        rungs = rg if rungs is None else (rungs + rg)
    grabs = None
    for yg in (+28.8, +36.2):
        gb = make_swept_tube(
            [(-99.4, yg, 61.2), (-99.4, yg, 69.2), (-91.8, yg, 69.2), (-91.8, yg, 61.2)],
            r_tube=0.65,
            bend_r=1.4,
        )
        grabs = gb if grabs is None else (grabs + gb)
    hidden_top_tie = Pos(-95.6, +32.5, 61.4) * Box(8.8, 8.6, 0.9)
    return hidden_inner_spine + rungs + grabs + hidden_top_tie


def build_tender_internals(cfg: Dict):
    """
    8-20B 煤水车 100% 原厂真值内部双箱与直列管网总成 (イラスト 31, 34, 36, 37 & p47, 零虚构支架腿, cc=1):
      1. **36-4 无腿上法兰吊挂式角型黄铜酒精箱** (上法兰 66x68mm 坐于 35-1 表面 Z_tdr=+62.0mm, 带**双平行 -45° 压筋**; 槽体 60x60x37.5mm)；
      2. **34-2 同轴矩形恒液位サンプ** (马氏管垂直直降零 S 弯!)；
      3. **31-10 后部独立敞口黄铜水箱** (与 36-4 保持洁净空气隔舱!)；
      4. **31-5 横型手摇泵体** + **31-12 x2** + **31-13** + **31-14 防虹吸鹅颈回水立管** + **31-9** + **34-6 x2 床下直管**。
    """
    alc_outer = Pos(+30.2, 0.0, +43.25) * Box(60.0, 60.0, 37.5)
    alc_hollow = Pos(+30.2, 0.0, +43.4) * Box(57.8, 57.8, 35.2)
    alc_top_flange = (
        Pos(+30.8, 0.0, +62.6) * Box(65.6, 67.6, 1.2)
        + Pos(+24.5, -6.0, +63.3) * Rot(0, 0, -45) * Box(52.0, 1.8, 0.7)
        + Pos(+36.5, +6.0, +63.3) * Rot(0, 0, -45) * Box(52.0, 1.8, 0.7)
    )
    filler_neck_36_5 = Pos(+32.7, +6.5, +64.2) * Cylinder(radius=4.3, height=3.2)
    filler_bore = Pos(+32.7, +6.5, +62.5) * Cylinder(radius=2.85, height=8.0)
    valve_boss_36_6 = Pos(+21.7, -6.0, +64.6) * Cylinder(radius=2.6, height=4.0)
    alc_tank_36_4 = (alc_outer + alc_top_flange + filler_neck_36_5 + valve_boss_36_6) - alc_hollow - filler_bore

    sump_box_34_2 = (
        Pos(+28.4, 0.0, +7.1) * Box(30.2, 28.0, 11.2)
        - Pos(+28.4, 0.0, +7.9) * Box(27.8, 25.6, 10.4)
        + Pos(+28.9, 0.0, +12.4) * Box(16.0, 33.2, 1.6)
        + Pos(+28.4, +14.8, +15.0) * Cylinder(radius=2.1, height=6.4)
        + Pos(+28.4, -14.8, +15.0) * Cylinder(radius=2.1, height=6.4)
    )
    mariotte_vertical_tubes = (
        Pos(+21.7, -6.0, +14.0) * Cylinder(radius=1.55, height=24.0)
        + Pos(+32.7, +6.5, +14.0) * Cylinder(radius=1.45, height=24.0)
    )
    drawbar_pocket_32_5_8 = (
        Pos(+93.5, 0.0, +12.6) * Box(18.0, 16.0, 2.2)
        + Pos(+101.2, 0.0, -1.9) * Box(7.6, 10.4, 4.6)
        - Pos(+103.2, 0.0, -1.9) * Box(5.0, 6.8, 3.0)
        + Pos(+102.0, 0.0, +4.8) * Cylinder(radius=1.85, height=17.6)
    )
    fuel_tube_st7 = (
        make_swept_tube(
            [(+43.0, 0.0, +5.2), (+56.25, +2.2, +6.5), (+96.0, 0.0, +7.0), (+107.2, 0.0, +7.0)],
            r_tube=2.15,
            bend_r=4.0,
        )
        + Pos(+93.5, 0.0, +9.8) * Box(4.0, 5.2, 5.6)
        + Pos(+105.0, 0.0, +7.0) * Rot(0, 90, 0) * Cylinder(radius=2.85, height=2.2)
    )

    wt_liner = Pos(-46.75, 0.0, +37.7) * Box(84.5, 76.4, 47.4) - Pos(-46.75, 0.0, +38.5) * Box(82.1, 74.0, 46.6)

    hand_pump_31_5 = (
        Pos(-56.0, +4.0, +21.7) * Rot(0, 90, 0) * Cylinder(radius=5.4, height=22.0)
        + Pos(-56.0, +4.0, +16.4) * Box(18.0, 10.6, 4.8)
        + Pos(-47.0, +4.0, +27.8) * Box(4.2, 2.5, 11.2)
        + Pos(-47.0, +4.0, +31.7) * Rot(90, 0, 0) * Cylinder(radius=2.4, height=2.6)
        + Pos(-69.5, +4.0, +28.0) * Cylinder(radius=4.4, height=25.6)
        + make_hex_nut((-69.5, +4.0, +41.2), (0.0, 0.0, 1.0), hex_r=4.2, length=2.2)
        - Pos(-49.5, +4.0, +21.7) * Rot(0, 90, 0) * Cylinder(radius=3.55, height=12.0)
    )

    bulkheads_31_12 = None
    for sy in (-1.0, +1.0):
        bh = (
            Pos(-16.4, sy * 15.2, +16.2) * Cylinder(radius=3.5, height=9.2)
            + make_hex_nut((-16.4, sy * 15.2, +18.2), (0.0, 0.0, 1.0), hex_r=4.4, length=1.8)
            + make_banjo_eye((-16.4, sy * 15.2, +11.8), (0.0, 0.0, -1.0), r_eye=2.5, h_eye=2.2, hex_r=1.9)
        )
        bulkheads_31_12 = bh if bulkheads_31_12 is None else (bulkheads_31_12 + bh)

    pipe_31_13 = (
        make_hex_nut((-69.5, +0.2, +35.7), (0.0, 1.0, 0.0), hex_r=2.15, length=2.0)
        + make_swept_tube(
            [
                (-69.5, +2.0, +35.7),
                (-69.5, -15.2, +35.7),
                (-16.4, -15.2, +35.7),
                (-16.4, -15.2, +19.5),
            ],
            r_tube=1.35,
            bend_r=4.2,
        )
        + make_hex_nut((-16.4, -15.2, +21.2), (0.0, 0.0, 1.0), hex_r=2.15, length=2.0)
    )
    standpipe_31_14 = (
        make_hex_nut((-16.4, +15.2, +21.2), (0.0, 0.0, 1.0), hex_r=2.15, length=2.0)
        + make_swept_tube(
            [
                (-16.4, +15.2, +19.5),
                (-16.4, +15.2, +54.2),
                (-22.0, +12.5, +55.8),
                (-26.0, +11.0, +53.7),
            ],
            r_tube=1.35,
            bend_r=3.4,
        )
    )
    drain_bush_31_9 = (
        Pos(-41.0, -10.3, +14.2) * Cylinder(radius=4.4, height=5.2)
        + make_hex_nut((-41.0, -10.3, +16.4), (0.0, 0.0, 1.0), hex_r=4.0, length=1.5)
    )

    pipes_34_6 = None
    for sy in (-1.0, +1.0):
        yp_in = sy * 15.2
        yp_out = sy * 21.3
        p6 = (
            make_swept_tube(
                [
                    (-16.4, yp_in, +13.0),
                    (+95.0, yp_in, +13.0),
                    (+101.5, yp_out, +9.2),
                    (+103.5, yp_out, -5.1),
                    (+109.5, yp_out, -5.1),
                ],
                r_tube=1.20,
                bend_r=2.8,
            )
            + Pos(+28.4, yp_in, +13.4) * Box(5.0, 3.0, 2.0)
            + Pos(+103.5, yp_out, +3.6) * Box(2.4, 4.2, 19.2)
            + make_hex_nut((+106.4, yp_out, -5.1), (1.0, 0.0, 0.0), hex_r=2.55, length=1.8)
            + Pos(+108.2, yp_out, -5.1) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=2.6)
        )
        pipes_34_6 = p6 if pipes_34_6 is None else (pipes_34_6 + p6)

    front_beam_tie = (
        Pos(+103.2, 0.0, +11.6) * Box(2.8, 45.8, 2.2)
        + Pos(+103.2, 0.0, +9.2) * Box(2.6, 6.0, 5.4)
    )

    return (
        alc_tank_36_4 + sump_box_34_2 + mariotte_vertical_tubes + drawbar_pocket_32_5_8 + fuel_tube_st7
        + wt_liner + hand_pump_31_5 + bulkheads_31_12 + pipe_31_13 + standpipe_31_14 + drain_bush_31_9
        + pipes_34_6 + front_beam_tie
    )


def build_tender_bogie_frame(cfg: Dict):
    """イラスト 31 & 36: 31-26 铸铁侧架 + 31-27 三长圆孔端梁 + 31-29 枕梁 + 36-3 心皿座金 + 31-25 板簧 + 轴箱耳下置 31-28 螺旋弹簧 (100% Y 对称)。"""
    wb = float(cfg["tender_and_track"]["bogie_wheelbase"])
    half_wb = 0.5 * wb
    y_frame = 0.5 * float(cfg["tender_and_track"].get("track_gauge", cfg["meta"]["track_gauge"])) + 5.2

    bolster = (
        Pos(0, 0, 2.25) * Box(14.0, 2.0 * y_frame + 1.0, 5.5)
        + Pos(0, 0, -1.1) * Cylinder(radius=7.8, height=1.2)
        + make_hex_nut((0.0, 0.0, -2.2), (0.0, 0.0, 1.0), hex_r=2.9, length=1.6)
    )
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
            side_part += Pos(ax_x, sgn * (y_frame + 1.2), 2.8) * Box(12.2, 3.4, 1.8)
            side_part += Pos(ax_x - 4.8, sgn * (y_frame + 1.2), -0.4) * Cylinder(radius=1.45, height=4.8)
            side_part += Pos(ax_x + 4.8, sgn * (y_frame + 1.2), -0.4) * Cylinder(radius=1.45, height=4.8)
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
        ("cab_reverser_nut", 0.025, RHO_BRASS, build_cab_reverser_nut),
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
        ("tender_hp_links", 0.025, RHO_BRASS, build_tender_hp_links),
        ("tender_handpump_lever", 0.03, RHO_BRASS, build_tender_handpump_lever),
        ("tender_handpump_ram", 0.025, RHO_BRASS, build_tender_handpump_ram),
        ("tender_drain_plug", 0.025, RHO_BRASS, build_tender_drain_plug),
        ("tender_coal_cover", 0.05, RHO_BRASS, build_tender_coal_cover),
        ("tender_fuel_valve", 0.025, RHO_BRASS, build_tender_fuel_valve),
        ("tender_fuel_cap", 0.025, RHO_BRASS, build_tender_fuel_cap),
        ("tender_rear_ladder", 0.03, RHO_BRASS, build_tender_rear_ladder),
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
        ("exhaust_blast_pipe", 0.035, RHO_BRASS, lambda: build_exhaust_blast_pipe(cfg, pip_cfg)),
        ("blower_nozzle_17_3", 0.035, RHO_BRASS, lambda: build_blower_nozzle_17_3(cfg, pip_cfg)),
        ("smokebox_petticoat", 0.04, RHO_STAINLESS, lambda: build_smokebox_petticoat(cfg, pip_cfg)),
        ("tandem_safety_valves", 0.03, RHO_BRASS, lambda: build_tandem_safety_valves(cfg, pip_cfg)),
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
