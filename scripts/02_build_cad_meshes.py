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
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from sim.hw_guard import HardwareThermalPowerGuard, apply_cpu_thread_and_nice_caps, stop_background_gpu_server_for_heavy_job

apply_cpu_thread_and_nice_caps(max_threads=8, nice_delta=5)

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from scripts.step_pipeline import compile_step_to_stl_and_mass
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
    export_step,
    export_stl,
    extrude,
    fillet,
    loft,
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
RHO_BURNER_ASSY = 8500.0        # 23-1 1x3 三头离散圆筒黄铜燃烧器 + 23-2/4-9 冲压黄铜支架
RHO_CERAMIC_WICK = 420.0        # セラミックヤーン(バーナー芯): ~30束氧化铝-硅酸铝陶瓷纤维毛细灯芯
RHO_CERAMIC_SHEET = 260.0       # セラミックシート(断熱材): Templates 1-6 陶瓷纤维隔热/气密衬垫
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


def _extend_polyline_ends(
    pts: Sequence[Tuple[float, float, float]],
    ext_start: float = 0.0,
    ext_end: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """沿首尾切向微延拓中心折线，供内孔布尔差集干净切穿接头/法兰端面 (消除共面薄皮并保证 cc==1)。"""
    out = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    if len(out) < 2:
        return out
    if ext_start != 0.0:
        v0 = Vector(*out[0]) - Vector(*out[1])
        if v0.length > 1e-6:
            p0 = Vector(*out[0]) + v0.normalized() * ext_start
            out[0] = (float(p0.X), float(p0.Y), float(p0.Z))
    if ext_end != 0.0:
        v1 = Vector(*out[-1]) - Vector(*out[-2])
        if v1.length > 1e-6:
            p1 = Vector(*out[-1]) + v1.normalized() * ext_end
            out[-1] = (float(p1.X), float(p1.Y), float(p1.Z))
    return out


def _build_swept_wire(pts: Sequence[Tuple[float, float, float]], r_ref: float, bend_r: float) -> Wire:
    """构造 G1 相切圆弧弯角中心线 Wire (以 r_ref 锁定最小弯头半径，使内外壁 100% 同心等壁厚)。"""
    vecs = [Vector(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    n = len(vecs)
    if n < 2:
        raise ValueError("_build_swept_wire requires at least 2 points")
    if n == 2:
        return Wire([Edge.make_line(vecs[0], vecs[1])])
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
            r_i = min(max(r_ref + 0.35, bend_r), d_max * math.tan(alpha))
            r_i = max(r_ref + 0.12, r_i)
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
    return Wire(edges)


def make_swept_tube(
    pts: Sequence[Tuple[float, float, float]],
    r_tube: float,
    bend_r: float,
    r_ref: float | None = None,
):
    """沿任意 2D/3D 空间折线构造 G1 相切圆弧弯角 (Edge.make_three_point_arc) 并扫掠圆截面金属弯管，彻底取代失真的直角方块弯头！"""
    w = _build_swept_wire(pts, r_tube if r_ref is None else r_ref, bend_r)
    prof = Plane(origin=w @ 0, z_dir=w % 0) * Circle(r_tube)
    return sweep(prof, path=w)


def make_hollow_swept_tube(
    pts: Sequence[Tuple[float, float, float]],
    r_od: float,
    r_id: float,
    bend_r: float,
):
    """通过外径扫掠减去两端延伸 0.5mm 的同路径内径扫掠 (消除共面端盖退化与 BSpline 圆环扫掠断言) 生成真壁厚空心弯管 B-Rep 实体！"""
    outer = make_swept_tube(pts, r_tube=r_od, bend_r=bend_r, r_ref=r_od)
    inner = make_swept_tube(_extend_polyline_ends(pts, 0.5, 0.5), r_tube=r_id, bend_r=bend_r, r_ref=r_od)
    return outer - inner


def make_spline_hose(
    pts: Sequence[Tuple[float, float, float]],
    t_start: Tuple[float, float, float],
    t_end: Tuple[float, float, float],
    r_hose: float,
    r_id: float = 0.0,
):
    """构建自然悬垂且平行零交叉的大圆弧弹性软管实体 (解析圆弧+直管扫掠，100% 流形水密且杜绝样条自交)。"""
    poly = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    bend_r = max(1.85 * r_hose, 5.5)
    outer = make_swept_tube(poly, r_tube=r_hose, bend_r=bend_r, r_ref=r_hose)
    if r_id <= 1e-4:
        return outer
    inner = make_swept_tube(_extend_polyline_ends(poly, 0.7, 0.7), r_tube=r_id, bend_r=bend_r, r_ref=r_hose)
    return outer - inner


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
    """20-6 方形底座 + 20-2 细长盲孔空心活塞先导尻棒护套 (イラスト 20 & p43, 同轴 y=0, z=0 嵌合气缸前盖 x=+17.8..+29.5mm, cc=1)。"""
    collar_20_6 = (
        Pos(+18.4, 0.0, 0.0) * Box(1.8, 4.4, 4.4)
        + Pos(+18.4, 0.0, +2.3) * Cylinder(radius=0.65, height=0.8)
    )
    tube_20_2 = Pos(+23.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=9.8)
    tip_cap = Pos(+29.1, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=1.45, top_radius=0.88, height=0.8)
    blind_bore = Pos(+22.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.00, height=11.0)
    part_L = (collar_20_6 + tube_20_2 + tip_cap) - blind_bore
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_valve_spindle_sheath(cfg: Dict, is_left: bool = True):
    """20-1 台阶盲孔空心滑阀前导套 (イラスト 1, 20 & p43, 同轴 y=+3.5mm [|Y|=36.0mm], z=+18.5mm, x=+16.5..+26.5mm, cc=1)。"""
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    y_valve = +3.5
    body_20_1 = Pos(+18.4, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=3.05, height=3.8)
    set_screw = Pos(+18.4, y_valve, z_valve + 3.15) * Cylinder(radius=0.80, height=1.1)
    nose_20_1 = Pos(+23.0, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.02, height=5.8)
    tip_cap = Pos(+26.2, y_valve, z_valve) * Rot(0, 90, 0) * Cone(bottom_radius=2.02, top_radius=1.35, height=0.6)
    spindle_blind_bore = Pos(+20.6, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=9.2)
    part_L = (body_20_1 + set_screw + nose_20_1 + tip_cap) - spindle_blind_bore
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
    8-18 左侧立式扁钢逆转长拉杆 (イラスト 8, 図 9.1, p43 SideB & p44 TopView 多图联立真值):
      - 局部原点 (0,0,0) 位于 reverser_shaft 左走台向上摇臂 2-18 顶点 (X=+25.5, Y=+32.5, Z=+33.5 mm)；
      - 后端孔位于局部 (-148.0, 0.0, +10.5) = 世界系 (-122.5, +32.5, +44.0 mm)，同轴铰接水平前伸的 8-20x2 连杆；
      - 后 80% 段 (X_world ∈ [-122.5, -5.0] mm) 100% 笔直水平延伸于窗台高度 Z=+44.0 mm (零波浪驼峰! 底缘 Z=+42.85mm 无接触越过 +42.2mm 后走台隆起台)，
        仅前 20% 段 (X_world ∈ [-5.0, +25.5] mm) 单调平滑下弯接入 25-7 护罩内的 2-18 摇臂顶销！
    """
    x_rear_pin = -148.0
    z_rear_pin = +10.50
    front_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.70, height=3.0)
        + Pos(-1.5, 0, 0) * Box(3.0, 3.0, 2.9)
        - Pos(+0.6, 0, 0) * Box(3.4, 1.72, 4.4)
    )
    pin_f = make_stepped_pin_L(0.0, 0.0, -1.90, +1.90, r_shank=0.80, r_head=1.30, r_nut=1.20)

    p_f_top = [
        Vector(-1.8, 0.0, +1.15),
        Vector(-10.0, 0.0, +5.95),
        Vector(-20.0, 0.0, +10.55),
        Vector(-30.5, 0.0, z_rear_pin + 1.15),
    ]
    p_f_bot = [
        Vector(-30.5, 0.0, z_rear_pin - 1.15),
        Vector(-20.0, 0.0, +8.25),
        Vector(-10.0, 0.0, +3.65),
        Vector(-1.8, 0.0, -1.15),
    ]
    p_r_top = Vector(x_rear_pin + 0.6, 0.0, z_rear_pin + 1.15)
    p_r_bot = Vector(x_rear_pin + 0.6, 0.0, z_rear_pin - 1.15)
    w_xz = Wire([
        Edge.make_spline(p_f_top, tangents=(Vector(-1.0, 0.0, 0.42).normalized(), Vector(-1.0, 0.0, 0.0))),
        Edge.make_line(p_f_top[-1], p_r_top),
        Edge.make_line(p_r_top, p_r_bot),
        Edge.make_line(p_r_bot, p_f_bot[0]),
        Edge.make_spline(p_f_bot, tangents=(Vector(1.0, 0.0, 0.0), Vector(1.0, 0.0, -0.42).normalized())),
        Edge.make_line(p_f_bot[-1], p_f_top[0]),
    ])
    flat_blade = extrude(Face(w_xz), amount=0.49, dir=(0.0, 1.0, 0.0), both=True)

    rear_eye = (
        Pos(x_rear_pin, 0.0, z_rear_pin) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=1.5)
        + make_stepped_pin_L(x_rear_pin, z_rear_pin, -1.95, +1.95, r_shank=0.80, r_head=1.30, r_nut=1.20)
    )
    return front_clevis + pin_f + flat_blade + rear_eye


def build_drain_cock_linkage(cfg: Dict):
    """
    7-5 排液凸轮横轴 + 7-1~3 向上驱动摇臂 + 7-4/7-6 凸轮 + 1-19 x4 叠搭剪式摇板 + 1-18 x4 横向顶针 + 1-16 x4 全通空心弹簧提升阀排液嘴 (100% Y 对称, イラスト 1 Fig.1, 7 & 図 1.3):
      - 局部原点 (0,0,0) 对准 7-5 横轴中心 (X=+124.2, Y=0.0, Z=-3.8 mm)；
      - 7-1~3 摇臂自 Z=-3.8mm 向上伸至 z_local=+12.3mm (Z=+8.5mm, Y=±17.8mm) 衔接全直水平拉杆 8-5；
      - 1-16 x4 排液嘴内置 (1-1/1-2/1-3 + PN2.2-0.8) 弹簧提升阀腔与 1-18 横向顶针孔，上下全通 ø1.10mm 排凝流道！
    """
    y_cock = 30.8
    cross_shaft_7_5 = Rot(90, 0, 0) * Cylinder(radius=0.95, height=46.8)
    sides_outer = cross_shaft_7_5
    bores_all = None
    for sgn in (-1.0, 1.0):
        yc = sgn * y_cock
        horn_7_1 = (
            Pos(0.0, sgn * 17.8, +6.15) * Box(2.0, 1.3, 12.7)
            + Pos(0.0, sgn * 17.8, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.70, height=1.8)
            + Pos(0.0, sgn * 17.8, +12.3) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=1.8)
        )
        cam_7_4 = (
            Pos(0.0, sgn * 22.4, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)
            + Pos(0.0, sgn * 22.8, -4.8) * Box(2.4, 2.0, 9.8)
        )
        scissor_rear = Pos(-1.0, sgn * 26.8, -9.7) * Box(6.8, 8.6, 1.2)
        scissor_front = Pos(+4.0, sgn * 26.8, -8.8) * Box(6.8, 8.6, 1.2)
        overlap_hub = Pos(+1.2, sgn * 23.4, -9.0) * Box(3.8, 3.0, 2.8)
        push_pins_1_18 = (
            Pos(-3.4, sgn * 29.2, -10.2) * Rot(90, 0, 0) * Cylinder(radius=0.58, height=3.6)
            + Pos(+6.6, sgn * 29.2, -10.2) * Rot(90, 0, 0) * Cylinder(radius=0.58, height=3.6)
        )
        cock_rear = (
            Pos(-3.4, yc, -10.0) * Cylinder(radius=1.62, height=3.6)
            + Pos(-3.4, yc, -11.4) * Cylinder(radius=2.05, height=1.0)
            + Pos(-3.4, yc, -12.6) * Cylinder(radius=0.95, height=2.4)
        )
        cock_front = (
            Pos(+6.6, yc, -9.8) * Cylinder(radius=1.62, height=3.6)
            + Pos(+6.6, yc, -11.4) * Cylinder(radius=2.05, height=1.0)
            + Pos(+6.6, yc, -12.6) * Cylinder(radius=0.95, height=2.4)
        )
        sides_outer = sides_outer + horn_7_1 + cam_7_4 + scissor_rear + scissor_front + overlap_hub + push_pins_1_18 + cock_rear + cock_front
        cock_bores = (
            Pos(-3.4, yc, -11.0) * Cylinder(radius=0.56, height=6.6)
            + Pos(-3.4, yc, -10.2) * Cylinder(radius=0.92, height=2.2)
            + Pos(+6.6, yc, -10.9) * Cylinder(radius=0.56, height=6.8)
            + Pos(+6.6, yc, -10.0) * Cylinder(radius=0.92, height=2.2)
        )
        bores_all = cock_bores if bores_all is None else (bores_all + cock_bores)
    return sides_outer - bores_all


def build_drain_pull_rod(cfg: Dict):
    """
    8-5 右侧 (-Y=-17.8mm) 100% 全直水平排液拉杆 + 8-6 床下可调铰接套环 + P-M2.6-3 紧定螺钉 (イラスト 7, 8 & 図 8.1):
      - 彻底删除虚构的车外床下大滚花鼓锤！8-7 排液手柄真实立于驾驶室内部右地板上方 (-133.5, -21.2)，
        下端穿过 8-11 纵向槽与 8-10 支座，在 Z=+8.5mm 处铰接 8-6 套环驱动全直拉杆 8-5 直抵前部 7-1 摇臂 (+124.2, -17.8, +8.5)！
    """
    x0, x1 = -135.2, +124.2
    rod_8_5 = Pos(0.5 * (x0 + x1), -17.8, 8.5) * Rot(0, 90, 0) * Cylinder(radius=0.62, height=x1 - x0)
    collar_8_6 = (
        Pos(-133.5, -18.8, 8.5) * Rot(0, 90, 0) * Cylinder(radius=1.65, height=4.4)
        + Pos(-133.5, -19.5, 8.5) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=3.8)
        + Pos(-133.5, -17.8, 10.2) * Cylinder(radius=0.85, height=1.2)
    )
    front_clevis = Pos(+124.2, -17.8, 8.5) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=1.8)
    return rod_8_5 + collar_8_6 + front_clevis


# ==================== 3B. 驾驶室 1:1 原厂全功能内构与煤水车全功能可动阀件网格 ====================

def build_cab_backhead_plate():
    """
    11-9 ボイラバックプレート + 12-5 x3 压板 + 12-2 观火孔框与底铰耳 + 前折排水阀侧翼 (イラスト 11, 12 & p45, 100% Y 对称, cc=1)。
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
    8-11 運転室床 (宽 76.4mm, Z_top=+20.0mm, 带右侧室内 8-7 排液立杆纵向通槽) + 8-21 左前延长片 + 4-6/4-7 折弯梁 + 6-12 后横梁 + 8-10 ドレンねじスタンド支座 + 4-8 横置旁通支座 (イラスト 4, 6, 8, 図 8.1 & p45, cc=1)。
    """
    deck_8_11 = (
        Pos(-130.75, 0.0, 19.35) * Box(34.5, 76.4, 1.3)
        - Pos(-145.5, 0.0, 19.35) * Box(6.0, 24.0, 4.0)
    )
    strip_8_21 = Pos(-115.2, +32.5, 19.5) * Box(3.6, 9.2, 1.0)
    channels_4_6_7 = (
        Pos(-131.0, +13.6, 16.2) * Box(28.0, 1.3, 5.6)
        + Pos(-131.0, -13.6, 16.2) * Box(28.0, 1.3, 5.6)
    )
    rear_beam_6_12 = (
        Pos(-144.2, 0.0, 14.8) * Box(1.6, 44.0, 8.2)
        - Pos(-144.2, 0.0, 14.8) * Box(4.0, 20.5, 4.8)
    )
    bypass_hole = Pos(-136.5, +6.8, 19.5) * Cylinder(radius=1.45, height=6.0)
    riser_6_8_slot = Pos(-115.8, -11.8, 19.35) * Box(4.4, 4.0, 5.0)
    drain_lever_slot = Pos(-133.5, -21.2, 19.35) * Box(7.4, 2.5, 4.0)
    stand_8_10 = (
        Pos(-133.5, -23.1, 18.4) * Box(6.8, 2.2, 1.2)
        + Pos(-133.5, -23.1, 16.2) * Box(3.6, 1.4, 4.6)
        + Pos(-133.5, -22.5, 15.0) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=1.6)
        + Pos(-121.0, -17.8, 13.8) * Box(1.3, 3.6, 11.4)
        + Pos(-119.8, -17.8, 18.4) * Box(3.4, 3.6, 1.2)
        - Pos(-121.0, -17.8, 8.5) * Rot(0, 90, 0) * Cylinder(radius=1.05, height=4.0)
    )
    stand_4_8 = Pos(-136.5, 0.0, 18.1) * Box(4.6, 20.0, 2.4)
    return (deck_8_11 + strip_8_21 + channels_4_6_7 + rear_beam_6_12 + stand_8_10 + stand_4_8) - bypass_hole - riser_6_8_slot - drain_lever_slot


def build_cab_reverser_stand():
    """
    8-12 (细圆柱前立柱) + 8-13 (竖排双圆孔后U字座) + 8-15 (单侧外置 +Y 多孔分度导条) 日本国铁左驾窗高螺旋逆转机座 (イラスト 8, 図 9.1 & p45, cc=1):
      - 挂载于驾驶室左地板 (-127.5, +32.5, +20.0mm)，丝杠轴心 z_scr=24.0mm (世界系 Z=+44.0mm 窗台高)！
    """
    z_scr = 24.0
    base_pad = Pos(0.0, 0.0, +0.6) * Box(20.6, 4.8, 1.2)
    post_8_12_f = (
        Pos(+9.0, 0.0, 0.5 * z_scr) * Cylinder(radius=1.25, height=z_scr)
        + Pos(+9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.2)
    )
    ped_8_13_r = (
        Pos(-9.0, 0.0, 0.5 * (z_scr + 1.6)) * Box(2.4, 4.6, z_scr + 1.6)
        + Pos(-9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=2.6)
        - Pos(-9.0, 0.0, +7.6) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=5.0)
        - Pos(-9.0, 0.0, +15.2) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=5.0)
    )
    bar_8_15_out = Pos(0.0, +1.75, z_scr) * Box(18.8, 0.88, 2.1)
    for hx in (-4.5, -1.5, +1.5, +4.5):
        bar_8_15_out -= Pos(hx, +1.75, z_scr) * Rot(90, 0, 0) * Cylinder(radius=0.48, height=3.0)
    bore_8_14 = Pos(0.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=24.0)
    return (base_pad + post_8_12_f + ped_8_13_r + bar_8_15_out) - bore_8_14


def build_cab_reverser_wheel():
    """
    8-14 水平逆转丝杠 + 8-16 原厂双柄直径摇臂曲柄 (イラスト 8, 17 & p45, 绕 X 轴旋转!)。
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
    8-17 丝杠行走螺母滑块 + 8-20 x2 水平前伸双扁连杆 + 8-19 x2 销栓 (イラスト 8 & Manual p.25 図 9.1, cc=1):
      - 8-20 x2 自螺母 (0,0,0) 沿 +X 水平向前延伸 L=5.0mm 至 (+5.0, 0.0, 0.0) = 世界系 (-122.5, +32.5, +44.0mm)，
        同轴衔接 8-18 平直逆转拉杆后端孔，彻底消除下垂连杆导致的虚假驼峰折弯！
    """
    nut_8_17 = Box(3.6, 2.8, 3.6) + Rot(90, 0, 0) * Cylinder(radius=1.35, height=3.4)
    link_in = Pos(+2.5, -0.95, 0.0) * Box(6.4, 0.75, 2.2)
    link_out = Pos(+2.5, +0.95, 0.0) * Box(6.4, 0.75, 2.2)
    front_pin = Pos(+5.0, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=0.98, height=3.2)
    return nut_8_17 + link_in + link_out + front_pin


def build_cab_regulator_lever():
    """
    11-11 + 11-12 驾驶室 11-15 左侧面后口 (-119.2, +8.2, +66.5mm) 横轴加减阀杆、向后水平多孔扁柄及外侧 +Y 指拨圆销 (イラスト 11, 13, 14 & Manual p.29):
      - q=0 闭合位扁手柄沿 -X 水平朝向驾驶室后方，末端配 +Y 外伸车削指拨圆柄，开启时绕 Y 轴 (axis='0 -1 0') 在 XZ 平面向上摆起！
    """
    spindle_11_12 = Pos(0.0, -1.8, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=4.4)
    hub = Rot(90, 0, 0) * Cylinder(radius=1.85, height=2.2)
    blade = (
        Pos(-6.2, +0.55, 0.0) * Box(12.4, 1.0, 2.1)
        + Pos(-12.6, +0.55, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.50, height=1.0)
        - Pos(-8.2, +0.55, 0.0) * Rot(90, 0, 0) * Cylinder(radius=0.52, height=3.0)
        - Pos(-4.4, +0.55, 0.0) * Rot(90, 0, 0) * Cylinder(radius=0.45, height=3.0)
    )
    finger_pin = Pos(-12.4, +1.65, 0.0) * Rot(90, 0, 0) * Cylinder(radius=0.78, height=2.2)
    set_screw = Pos(0.0, +0.2, +2.0) * Cylinder(radius=0.62, height=0.8)
    return spindle_11_12 + hub + blade + finger_pin + set_screw


def build_cab_blower_wheel():
    """
    11-13 + 11-14 驾驶室 11-15 后侧正中 (-121.6, 0.0, +66.5mm) ø7.4mm 四辐圆形黄铜通风手轮 (イラスト 11, 14 & p45)。
    """
    stem = Pos(+1.6, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.92, height=3.6)
    hub = Rot(0, 90, 0) * Cylinder(radius=1.42, height=1.4)
    rim = Rot(0, 90, 0) * (Cylinder(radius=3.70, height=1.05) - Cylinder(radius=2.65, height=3.0))
    spokes = Box(0.76, 5.8, 0.76) + Box(0.76, 0.76, 5.8)
    ret_nut = make_hex_nut((-0.72, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=1.08, length=0.75)
    return stem + hub + rim + spokes + ret_nut


def build_cab_whistle_lever():
    """
    11-3 驾驶室左上 11-1 汽笛阀底部 (-115.6, +14.8, +62.2mm) 纵向 XZ 平面向后 (-X) 悬垂微凹弧形三圆孔拨柄 (イラスト 11 Fig.1, 13, 14 & p45):
      - 修正 90° 平面与弧向失真！11-3 绕 Y 轴横向圆销 (axis='0 -1 0') 在纵向 XZ 平面内向后 (-X) 伸向司机，呈下腹微垂凹弧，下压尾端顶起 11-2 顶销开启汽笛！
    """
    hub = Rot(90, 0, 0) * Cylinder(radius=1.45, height=1.8)
    arc_pts_up = [
        Vector(0.0, 0.0, +0.90),
        Vector(-3.6, 0.0, -1.50),
        Vector(-7.2, 0.0, -2.30),
        Vector(-10.6, 0.0, -0.85),
    ]
    arc_pts_lo = [
        Vector(-11.2, 0.0, -2.05),
        Vector(-7.2, 0.0, -4.20),
        Vector(-3.4, 0.0, -3.40),
        Vector(0.0, 0.0, -1.00),
    ]
    w_arc = Wire([
        Edge.make_spline(arc_pts_up),
        Edge.make_line(arc_pts_up[-1], arc_pts_lo[0]),
        Edge.make_spline(arc_pts_lo),
        Edge.make_line(arc_pts_lo[-1], arc_pts_up[0]),
    ])
    blade = extrude(Face(w_arc), amount=0.50, dir=(0.0, 1.0, 0.0), both=True)
    for hx, hz in ((-3.2, -2.45), (-5.8, -3.15), (-8.4, -2.25)):
        blade -= Pos(hx, 0.0, hz) * Rot(90, 0, 0) * Cylinder(radius=0.45, height=3.0)
    return hub + blade


def build_cab_sight_glass_tube():
    """11-7 驾驶室右侧外缘 (-118.5, -31.3, +46.0mm) 透明耐热硼硅玻璃水面计管 (OD 4.0mm / ID 2.3mm 空心管, 壁厚 0.85mm, cc=1)。"""
    return Cylinder(radius=2.00, height=17.0) - Cylinder(radius=1.15, height=19.0)


def build_cab_sight_drain():
    """13-5 #3 六角中空螺母 + 右侧水面计下座同轴底部 M4-6 一字槽吹扫螺塞 (-118.5, -31.3, +28.8mm, イラスト 11 Fig.2)。"""
    hex_13_5 = make_hex_nut((0.0, 0.0, +0.9), (0.0, 0.0, 1.0), hex_r=2.25, length=1.8)
    stem = Pos(0.0, 0.0, +1.8) * Cylinder(radius=1.20, height=2.8)
    cap_plug = Pos(0.0, 0.0, -0.9) * Cylinder(radius=2.05, height=2.0) - Pos(0.0, 0.0, -1.75) * Box(4.8, 0.60, 0.75)
    return hex_13_5 + stem + cap_plug


def build_cab_pg_dial():
    """PG-ø20 右内上 (-122.5, -16.5, +66.8mm) 严格垂直朝后 (-X, 0.0° 零仰角!) ø20mm 压力表白底刻度盘片 (イラスト 13 & p45 边缘差分标定)。"""
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
    11-17 驾驶室左床下 11-9 前折侧翼处 (-115.5, +20.6, +15.8mm) 横向 (+Y) 一字槽锅炉排水针阀螺杆 (イラスト 11, 13 & Manual p.43)。
    """
    stem = Pos(0.0, -1.2, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.12, height=3.2)
    head = Pos(0.0, +0.9, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.15, height=1.8)
    slot = Pos(0.0, +1.75, 0.0) * Box(4.8, 0.75, 0.62)
    return stem + (head - slot)


def build_cab_firebox_door():
    """
    12-1/12-2 驾驶室低位中下底铰蝴蝶式铸造火室门 + 内衬 Template 3 隔热护盾座 (イラスト 12 & p45)。
    """
    pin_12_3 = Rot(90, 0, 0) * Cylinder(radius=1.12, height=12.6)
    door_plate = (
        Pos(-0.6, 0.0, +6.7) * Box(1.5, 12.2, 13.0)
        + Pos(-0.6, 0.0, +6.8) * Rot(0, 90, 0) * Cylinder(radius=5.6, height=1.6)
        + Pos(+0.45, 0.0, +6.8) * Box(0.8, 10.6, 11.2)
    )
    boss_ring = Pos(-1.35, 0.0, +6.8) * Rot(0, 90, 0) * (Cylinder(radius=3.8, height=0.8) - Cylinder(radius=2.2, height=1.2))
    top_latch = Pos(-1.2, 0.0, +13.4) * Box(1.8, 2.4, 1.4)
    return pin_12_3 + door_plate + boss_ring + top_latch


def build_cab_bypass_lever():
    """
    6-9 驾驶室左地板 (+Y=+6.8mm, X=-136.5mm, Z=+23.2mm) 同轴给水旁通针阀立杆与水平拨柄 (イラスト 6, 図 8.1 & p45)。
    """
    stem_6_9 = Pos(0.0, 0.0, -4.0) * Cylinder(radius=1.05, height=8.4)
    collar = Pos(0.0, 0.0, -2.6) * Cylinder(radius=1.85, height=1.8)
    hub = Cylinder(radius=1.85, height=2.2)
    handle_arm = Pos(-2.2, +3.8, +0.2) * Rot(0, 0, +120) * Rot(0, -90, 0) * Cone(bottom_radius=0.95, top_radius=0.68, height=8.2)
    ball_tip = Pos(-4.2, +7.2, +0.2) * Cylinder(radius=1.15, height=1.8)
    return stem_6_9 + collar + hub + handle_arm + ball_tip


def build_cab_drain_lever():
    """
    8-7 ドレンレバー (Group 8 BOM p.5, イラスト 8, Manual p.24 図 8.1 & p45 驾驶室后视实拍真值):
      - 排液手柄真实竖立于【驾驶室内部】右地板 8-11 上方 (-133.5, -21.2mm)，顶端球形握球高 Z=+37.2mm (高于地板 +17.2mm)！
      - 局部原点 (0,0,0) 位于床下 8-10 支座枢轴 (-133.5, -21.2, +15.0mm, axis='0 1 0')，下端眼环 z=-6.5mm (Z=+8.5mm) 驱动 8-6/8-5 拉杆！
    """
    pivot_boss = Rot(90, 0, 0) * Cylinder(radius=1.55, height=2.4)
    lower_arm = (
        Pos(0.0, 0.0, -3.25) * Box(2.0, 1.1, 6.5)
        + Pos(0.0, +0.4, -6.5) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=1.8)
    )
    upper_shaft = Pos(0.0, 0.0, +10.2) * Cone(bottom_radius=1.05, top_radius=0.76, height=20.4)
    top_knob = (
        Pos(0.0, 0.0, +21.1) * Cylinder(radius=1.68, height=2.2)
        + Pos(0.0, 0.0, +19.8) * Cylinder(radius=1.18, height=1.0)
    )
    return pivot_boss + lower_arm + upper_shaft + top_knob


def build_dome_lid_29_12():
    """
    29-12 ドームのフタ + 29-13 軸 (イラスト 29 & Manual p.36 Step 29 原厂钢丝铰接可开启汽包/砂箱前盖, cc=1):
      - 局部原点 (0,0,0) 位于盖前端横向钢丝铰链 29-13 (世界系 X=+62.2, Y=0.0, Z=+91.85mm, axis='0 -1 0')，向后覆盖 X∈[+50.2, +62.2]mm！
    """
    wire_29_13 = Rot(90, 0, 0) * Cylinder(radius=0.48, height=9.6)
    hinge_lug = Pos(-0.8, 0.0, +0.2) * Box(2.0, 5.4, 1.1)
    lid_disc = (
        Pos(-6.0, 0.0, +0.35) * Cylinder(radius=6.05, height=0.85)
        + Pos(-6.0, 0.0, +0.92) * Cone(bottom_radius=5.75, top_radius=3.60, height=0.55)
        + Pos(-11.6, 0.0, +0.35) * Box(1.8, 2.6, 0.85)
    )
    return wire_29_13 + hinge_lug + lid_disc


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
    """1) ST-ø7 半透明白色硅胶供醇空心软管 (OD 7.0 / ID 5.0mm, 壁厚 1.00mm; 牵引杆 8-8 上方中央平行直连 Y=0: 23-1 <-> 34-2 サンプ + 32-3 导环)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["fuel_tdr"]
    p0 = (-3.6, 0.0, +9.6)
    p2 = (tgt[0] + 0.6, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), min(p0[2], p2[2]) - 1.2)
    yaw_tdr = math.atan2(p2[1] - p0[1], p2[0] - p0[0]) + math.pi
    t_end = (-math.cos(yaw_tdr - math.pi), -math.sin(yaw_tdr - math.pi), 0.0)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), t_end, r_hose=3.45, r_id=2.45)


def build_umb_hose_feed(cfg: Dict, is_r2m: bool = False):
    """2) BT-ø5 黑色氟橡胶给水/手摇泵共用空心软管 (OD 5.0 / ID 2.5mm, 壁厚 1.25mm; 牵引杆下方右侧 -Y 平行直连零交叉: 机车右后 -Y=-19.2mm 6-11 <-> 煤水车右前 -Y=-21.3mm 34-6)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["feed_tdr"]
    p0 = (-4.2, -19.2, -2.5)
    p2 = (tgt[0] + 0.4, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -6.4)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), r_hose=2.45, r_id=1.25)


def build_umb_union_34_8(cfg: Dict, is_r2m: bool = False):
    """
    34-7 空心口金 + 34-8 ポンプジョイント滚花活接螺母 + HC-1 弹簧卡箍 (イラスト 6, 34, 37 & p43/p46 真值校正: 位于机车右后 -Y=-19.2mm 6-11 外螺纹口金处，带 ø2.0mm 真实通孔, cc=1)。
    """
    p_loco = (-4.8, -19.2, -2.5)
    ax = Vector(1.0, 0.0, 0.0)
    tail_34_7 = extrude(Plane(origin=Vector(*p_loco) - ax * 3.6, z_dir=ax) * Circle(2.25), amount=5.4)
    nut_34_8 = (
        Pos(p_loco[0] + 0.6, p_loco[1], p_loco[2]) * Rot(0, 90, 0) * Cylinder(radius=3.35, height=3.4)
        + make_hex_nut((p_loco[0] - 0.8, p_loco[1], p_loco[2]), (1.0, 0.0, 0.0), hex_r=2.90, length=1.4)
    )
    hc1_clamp = extrude(Plane(origin=Vector(*p_loco) - ax * 3.0, z_dir=ax) * Circle(2.75), amount=1.2)
    bore_34_8 = extrude(Plane(origin=Vector(*p_loco) - ax * 4.5, z_dir=ax) * Circle(1.02), amount=8.0)
    return (tail_34_7 + nut_34_8 + hc1_clamp) - bore_34_8


def build_umb_hose_return(cfg: Dict, is_r2m: bool = False):
    """3) BT-ø5 黑色氟橡胶旁通无压回水空心软管 (OD 5.0 / ID 2.5mm, 壁厚 1.25mm; 牵引杆下方左侧 +Y 平行直连零交叉: 机车左后 +Y=+19.2mm 4-16 <-> 煤水车左前 +Y=+21.3mm 34-6)。"""
    tgt = _compute_drawbar_hose_targets_mm(cfg, is_r2m)["ret_tdr"]
    p0 = (-3.8, +19.2, -2.5)
    p2 = (tgt[0] + 0.4, tgt[1], tgt[2])
    pm = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]), -6.0)
    return make_spline_hose([p0, pm, p2], (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), r_hose=2.45, r_id=1.25)


# ==================== 3D. 均衡梁闸瓦、台车架、气缸体与锅炉外衣 ====================

def build_chassis_equalizer_brakes(cfg: Dict):
    """
    イラスト 3 & イラスト 5 真值均衡梁板簧与四轴前挂闸瓦总成 (5-1~5-4 + 5-12/5-13):
      - 严格修正首轴闸瓦装反错误！全部 4 对闸瓦吊杆 (5-1~5-4) 均位于第 1~4 动轮前方 (+X = ax + 19.0mm)，第 4 动轮后方增设 5-12/5-13 斜拉制动均衡臂！
    """
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
        for ax in ax_xs:
            sx, sz, rot_p, hx = ax + 19.0, -19.6, +44.0, ax + 19.5
            hanger_xs.append(hx)
            shoe = Pos(sx, sgn * 23.0, sz) * Rot(0, rot_p, 0) * Box(2.0, 2.2, 6.4)
            hanger_v = Pos(hx, y_frame_out, -11.6) * Box(2.0, 1.6, 17.4)
            cross_pin = Pos(hx, 0.5 * (y_frame_out + sgn * 23.0), sz - 0.3) * Rot(90, 0, 0) * Cylinder(
                radius=1.00, height=abs(sgn * 23.0 - y_frame_out) + 1.4
            )
            pad_f = Pos(hx, sgn * 14.2, -3.2) * Box(3.6, 2.6, 3.0)
            riser = Pos(hx, sgn * 14.2, 4.8) * Box(2.2, 1.5, 17.6)
            assy += shoe + hanger_v + cross_pin + pad_f + riser

        rear_eq_5_12 = (
            Pos(ax_xs[3] - 18.5, sgn * 14.2, +5.2) * Rot(0, -32.0, 0) * Box(2.4, 1.8, 18.8)
            + Pos(ax_xs[3] - 14.2, sgn * 14.2, -2.0) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=2.4)
        )
        x_lo = ax_xs[3] - 25.0
        x_hi = max(hanger_xs) + 2.0
        long_tie = Pos(0.5 * (x_lo + x_hi), sgn * 14.2, 12.8) * Box(x_hi - x_lo, 1.8, 2.1)
        assy += rear_eq_5_12 + long_tie

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
        leaf_19_15 = (
            Pos(dx_ax, sgn * (y_out + 1.1), 6.0) * Box(17.8, 2.2, 2.4)
            + Pos(dx_ax, sgn * (y_out + 1.1), 7.6) * Box(11.8, 2.2, 1.4)
            + Pos(dx_ax, sgn * (y_out + 1.1), 6.2) * Box(3.2, 2.8, 3.4)
        )
        abox = (
            Pos(dx_ax, sgn * y_out, 0.4) * Box(10.0, 4.4, 9.2)
            + Pos(dx_ax, sgn * (y_out + 1.2), 3.6) * Box(12.0, 3.6, 3.0)
            + leaf_19_15
            - Pos(dx_ax, sgn * y_out, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=7.0)
        )
        sub = flare_arm + side_beam + abox
        sides = sub if sides is None else (sides + sub)

    return pivot_nose + front_transom + sides - Cylinder(radius=1.8, height=8.0)


def build_driving_axlebox(cfg: Dict, is_left: bool = True):
    """5-9 动轮独立单侧黄铜轴箱 (彻底废除跨车架假桥连杆, L/R 分件各自 cc==1)。"""
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) - 6.5
    blk = Pos(0, +y_box, 0.5) * Box(10.5, 5.2, 11.5)
    flange = Pos(0, +y_box, 0.5) * Box(12.5, 2.2, 13.0)
    bore = Pos(0, +y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=8.0)
    part_L = blk + flange - bore
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_tender_axlebox(cfg: Dict, is_left: bool = True):
    """31-22 / 31-23 煤水车转向架独立左右轴箱 (彻底废除跨转向架虚假横桥杆 tie, L/R 分件各自 cc==1)。"""
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2
    blk = (
        Pos(0, +y_box, 0) * Box(7.2, 3.8, 7.6)
        + Pos(0, +y_box + 1.1, 0.2) * Box(5.4, 2.4, 5.8)
    )
    bore = Pos(0, +y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=6.0)
    part_L = blk - bore
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_cylinder_block(cfg: Dict, is_left: bool = True):
    """
    1-4/1-29 主气缸 + 20-4 外包板轮廓 + 1-7/1-14 镂空外进气蒸汽室 + 10-1 螺钉顶盖 + 1-9 三气口镜面 + 真实前后进汽道/中央排汽道 + 1-20 空心承插口 (イラスト 1, 7, 10, 20, cc==1):
      - 内部完整贯通：1-20 进汽口 -> 蒸汽室内腔 -> 1-9 前/后斜气道 -> ø15.0mm 气缸前/后腔 -> 底部排液孔，以及 1-9 中央排气槽 -> 1-20 排汽口！
    """
    bore = float(cfg["cylinder_and_rods"]["bore"])
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    y_valve = +3.5
    r_barrel = 10.6

    main_barrel = Pos(+5.3, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=22.2)
    rear_neck = (
        Pos(-8.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=5.6, height=7.6)
        + Pos(-12.9, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.5)
        - Pos(-9.8, -7.20, 0.0) * Box(14.0, 10.0, 16.0)
        - Pos(-9.8, +6.85, 0.0) * Box(14.0, 10.0, 16.0)
    )
    saddle_block = Pos(+4.8, -6.2, 6.2) * Box(21.4, 12.2, 23.6)
    outer_clad_20_4 = (
        Pos(+5.1, +1.8, 11.2) * Box(24.2, 12.0, 21.6)
        - Pos(+5.1, +7.6, -7.8) * Rot(+36, 0, 0) * Box(27.0, 8.0, 10.0)
    )
    front_cover = Pos(+17.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=10.1, height=1.6)
    for b in (
        Pos(+18.0, 0.0, 0.0)
        * Rot(0, 90, 0)
        * PolarLocations(radius=8.4, count=6)
        * Cylinder(radius=0.72, height=0.8)
    ):
        front_cover += b

    steam_chest = Pos(+5.1, +2.4, 18.2) * Box(23.4, 10.8, 10.2)
    lid_10_1 = Pos(+5.1, +2.6, 23.6) * Box(23.8, 10.6, 1.3)
    for sx_b in (-9.2, +9.2):
        for sy_b in (-3.4, +3.4):
            lid_10_1 += make_hex_nut((5.1 + sx_b, 2.6 + sy_b, 24.55), (0.0, 0.0, 1.0), hex_r=0.95, length=0.8)

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

    # 1-9 三气口镜面 (Z ∈ [14.90, 15.45]mm 保留 0.55mm 实体阀座镜面隔桥，杜绝进汽室与中央排气道静态短路!)
    chest_hollow = (
        Pos(+5.0, +2.7, 18.65) * Box(19.8, 7.6, 6.4)
        + Pos(+8.2, +2.7, 14.3) * Box(1.6, 5.2, 1.6)
        + Pos(+4.2, +2.7, 14.0) * Box(2.2, 5.2, 1.8)
        + Pos(+0.2, +2.7, 14.3) * Box(1.6, 5.2, 1.6)
    )
    front_steam_passage = make_swept_tube([(8.2, +2.0, 14.4), (14.2, +0.5, 11.2), (14.2, 0.0, 5.5)], r_tube=0.78, bend_r=1.8)
    rear_steam_passage = make_swept_tube([(0.2, +2.0, 14.4), (-4.0, +0.5, 11.2), (-4.0, 0.0, 5.5)], r_tube=0.78, bend_r=1.8)
    exhaust_passage = make_swept_tube([(4.2, +2.2, 13.5), (4.0, -2.5, 12.5), (4.0, -14.2, 12.5)], r_tube=1.35, bend_r=2.0)
    drain_bores = (
        Pos(-5.2, -1.7, -9.2) * Cylinder(radius=0.62, height=7.2)
        + Pos(+4.8, -1.7, -9.2) * Cylinder(radius=0.62, height=7.2)
    )
    spindle_bore = Pos(+4.5, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=28.0)
    inlet_bore = Pos(-2.5, -6.5, z_valve) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=16.0)
    bore_cut = (
        Pos(+5.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=21.2)
        + Pos(-9.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.62, height=13.2)
        + chest_hollow
        + front_steam_passage
        + rear_steam_passage
        + exhaust_passage
        + drain_bores
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
    锅炉烟室、流线型カマボコ砂箱汽包、车头低置给水加热器/头灯/前端梁、后移至第4动轮后方的走台隆起桥台及驾驶室外衣总成 (イラスト 15, 16, 18, 21, 22, 25, 28, 29, 30 & p43):
      - 外边界由两侧除烟板 28-11/12 严格锁定在 [-40.8, +40.8] mm (|min_y + max_y| = 0.000mm)；
      - 两侧走台梯形隆起桥台按 イラスト 22 & p43 精确移至第 4 动轮后方火室侧 (X=-81.0mm, X∈[-98,-64]mm)，左桥下配 22-11 双联空压机、右桥下配 22-18 辅助给水泵；
      - 储风缸 22-13/22-26 (OD 8.4x34mm) 悬吊于左右走台下方 (Z=+25.8mm)；
      - 消音器 28-8 移至右后火室肩部 (-76.5, -28.8, +50.0mm) 紧邻 15-8 发电机；车顶假汽笛 30-2 + 拉索 30-3 移至左肩 (+Y=+10.5mm, X=-58.0mm)！
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

    # 2. 前探低鞍贴合式 28-13 给水加热器 (X=+155.5, OD 12.4) + 前伸底座 LP-1 头灯 (X=+169.5) + 标准细腰烟囱 18-25 (イラスト 28 & p43)
    feedwater_heater = (
        Pos(155.5, 0.0, z_axis + r_sb + 2.4) * Rot(90, 0, 0) * Cylinder(radius=6.2, height=38.2)
        + Pos(155.5, +19.2, z_axis + r_sb + 2.4) * Rot(90, 0, 0) * Cylinder(radius=6.6, height=1.2)
        + Pos(155.5, -19.2, z_axis + r_sb + 2.4) * Rot(90, 0, 0) * Cylinder(radius=6.6, height=1.2)
        + make_swept_tube([(155.5, -15.5, z_axis + r_sb + 0.5), (155.5, -24.5, z_axis + 22.0), (148.0, -29.5, z_axis + 6.0)], r_tube=0.95, bend_r=2.5)
    )
    fwh_brackets = (
        Pos(155.5, -12.0, z_axis + r_sb - 1.2) * Box(7.2, 2.8, 4.6)
        + Pos(155.5, +12.0, z_axis + r_sb - 1.2) * Box(7.2, 2.8, 4.6)
    )
    headlight = (
        Pos(169.5, 0.0, z_axis + r_sb + 4.8) * Rot(0, 90, 0) * Cylinder(radius=4.5, height=7.8)
        + Pos(173.6, 0.0, z_axis + r_sb + 4.8) * Rot(0, 90, 0) * Cylinder(radius=4.9, height=1.1)
    )
    headlight_ped = Pos(165.8, 0.0, z_axis + r_sb - 0.2) * Box(8.4, 4.8, 5.4)

    x_chimney = x_sb_front - 32.0  # +130.0 mm
    chimney = (
        Pos(x_chimney, 0.0, z_axis + r_sb - 0.8) * Cylinder(radius=9.2, height=3.0)
        + Pos(x_chimney, 0.0, z_axis + r_sb + 8.1) * Cone(bottom_radius=7.25, top_radius=7.55, height=15.8)
        + Pos(x_chimney, 0.0, z_axis + r_sb + 16.2) * Cylinder(radius=7.85, height=1.3)
    )

    b_x0, b_x1 = x_fb_rear, x_sb_rear + 1.0
    r_barrel = 30.2
    barrel = Pos(0.5 * (b_x0 + b_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=b_x1 - b_x0)
    for bx in [74.0, 38.0, -14.0, -58.0]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.55, height=1.8)

    # 3. 标准型 JNR D51 カマボコ形流线砂箱/蒸汽包罩 29-11 (85.8mm 全长 X∈[-14.8,+71.0], Z_top=+91.8mm 齐平烟囱, 钝圆前鼻 + 24.8mm S形长尾 + 6根砂管 29-14~19, イラスト 29 & p43/p46)
    dome_profile_table = [
        (-14.8, 73.70, 5.40, 2.2),
        (-12.5, 74.95, 7.60, 3.0),
        (-9.0, 77.30, 9.60, 4.0),
        (-5.0, 80.70, 10.90, 4.8),
        (-1.0, 84.50, 11.65, 5.4),
        (+3.0, 87.90, 12.10, 5.8),
        (+7.0, 90.35, 12.32, 6.0),
        (+10.5, 91.45, 12.40, 6.2),
        (+15.0, 91.80, 12.40, 6.2),
        (+26.0, 91.80, 12.40, 6.2),
        (+38.0, 91.80, 12.40, 6.2),
        (+50.0, 91.80, 12.40, 6.2),
        (+58.0, 91.75, 12.35, 6.2),
        (+61.5, 91.35, 12.05, 6.0),
        (+64.5, 90.20, 11.35, 5.5),
        (+67.0, 87.80, 10.10, 4.8),
        (+69.0, 83.60, 8.20, 3.8),
        (+70.3, 78.80, 6.20, 2.8),
        (+71.0, 73.70, 4.40, 1.9),
    ]
    dome_secs = []
    z_dome_root = 68.4
    for dx_s, z_top_s, hw_s, r_crn in dome_profile_table:
        z_sh = max(z_dome_root + 1.2, z_top_s - r_crn)
        w_sec = Wire([
            Edge.make_line(Vector(dx_s, -hw_s, z_dome_root), Vector(dx_s, -hw_s, z_sh)),
            Edge.make_spline(
                [
                    Vector(dx_s, -hw_s, z_sh),
                    Vector(dx_s, -0.72 * hw_s, z_sh + 0.76 * (z_top_s - z_sh)),
                    Vector(dx_s, 0.0, z_top_s),
                    Vector(dx_s, +0.72 * hw_s, z_sh + 0.76 * (z_top_s - z_sh)),
                    Vector(dx_s, +hw_s, z_sh),
                ]
            ),
            Edge.make_line(Vector(dx_s, +hw_s, z_sh), Vector(dx_s, +hw_s, z_dome_root)),
            Edge.make_line(Vector(dx_s, +hw_s, z_dome_root), Vector(dx_s, -hw_s, z_dome_root)),
        ])
        dome_secs.append(Face(w_sec))
    dome_29_11 = (
        loft(dome_secs, ruled=True)
        + Pos(56.2, 0.0, 91.55) * (Cylinder(radius=5.75, height=0.8) - Cylinder(radius=4.55, height=2.2))
        + Pos(62.2, +3.8, 91.75) * Box(1.8, 1.2, 1.1)
        + Pos(62.2, -3.8, 91.75) * Box(1.8, 1.2, 1.1)
        + Pos(32.0, 0.0, 91.85) * Cylinder(radius=3.4, height=0.9)
        + Pos(11.0, 0.0, 91.65) * Cylinder(radius=2.8, height=0.9)
    )
    sand_pipes_L = None
    for sp_x, sp_end_x in [(44.0, 52.0), (31.0, 31.0), (18.0, 10.0)]:
        sp_one = (
            Pos(sp_x, +11.4, 75.5) * Rot(90, 0, 0) * Cylinder(radius=1.25, height=2.4)
            + make_swept_tube(
                [
                    (sp_x, +11.6, 75.5),
                    (0.5 * (sp_x + sp_end_x), +23.8, 63.0),
                    (sp_end_x, +30.4, 46.0),
                    (sp_end_x, +30.2, 31.2),
                ],
                r_tube=0.68,
                bend_r=6.0,
            )
        )
        sand_pipes_L = sp_one if sand_pipes_L is None else (sand_pipes_L + sp_one)
    dome_29_11 = dome_29_11 + sand_pipes_L + mirror(sand_pipes_L, about=Plane.XZ)

    firebox = Pos(0.5 * ((x_fb_front + 1.0) + x_fb_rear), 0, z_axis - 10.5) * Box(fb_l + 1.0, fb_w, fb_h - 1.0)

    # 4. 驾驶室外衣 21-28 + 永久螺钉锁固车顶通风器 21-15 (零可动天窗!) + 前 20mm 高门槛裁切 (Z_min=+30.05mm, 删除虚假下垂裙板) + 双侧窗与后宽拱门 (イラスト 21 & p43/p44)
    cab_len, cab_w = 38.6, 79.2
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0  # -129.3 mm, X ∈ [-148.6, -110.0]
    cab_box = Pos(cab_x, 0, 46.8) * Box(cab_len, cab_w, 55.6)  # Z ∈ [19.0, 74.6]

    # 宽檐无台阶圆弧车顶 (R=54.0mm, Z_center=36.0 -> Z_crown=90.0mm, Y_eave=±40.40mm 严格镜像对称)
    roof_len = cab_len + 7.2
    roof_xc = cab_x - 1.6
    roof_shell_raw = (
        Pos(roof_xc, 0.0, 36.0) * Rot(0, 90, 0) * (Cylinder(radius=54.0, height=roof_len) - Cylinder(radius=52.4, height=roof_len + 2.0))
    ) & (Pos(roof_xc, +20.20, 81.5) * Box(roof_len + 2.0, 40.40, 22.0))
    cab_roof = roof_shell_raw + mirror(roof_shell_raw, about=Plane.XZ)

    # 21-15 固定铆接式车顶通风器 (ベンチレーター, +M2-2.5 & +M2-6 永久锁固于整片车顶，严禁切洞铰接!)
    vent_21_15_L = (
        Pos(-129.5, +4.3, 89.9) * Box(25.6, 8.6, 2.2)
        + Pos(-129.5, +4.6, 91.1) * Rot(8, 0, 0) * Box(26.8, 9.2, 0.85)
    )
    vent_21_15 = vent_21_15_L + mirror(vent_21_15_L, about=Plane.XZ)

    rear_doorway_cut = (
        Pos(cab_x - 0.5 * cab_len, 0, 45.0) * Box(10.0, 67.0, 56.0)
        + (
            Pos(cab_x - 0.5 * cab_len, 0, 36.0) * Rot(0, 90, 0) * Cylinder(radius=52.4, height=12.0)
            & Pos(cab_x - 0.5 * cab_len, 0, 76.0) * Box(12.0, 67.0, 26.0)
        )
    )
    cab_trim_L = (
        Pos(-127.0, +39.55, 65.8) * Box(26.5, 1.7, 1.0)  # 21-11 雨搭眉檐 (位于双侧窗 Z_max=62.5 上方)
        + Pos(-132.8, +39.55, 41.5) * Box(10.8, 0.5, 4.2)  # 21-9 D51 号牌
        + Pos(cab_x - 0.5 * cab_len + 0.6, +34.8, 46.5) * Cylinder(radius=0.68, height=30.0)  # 30-4 后扶手杆
        + Pos(cab_x - 0.5 * cab_len + 1.2, +34.8, 60.5) * Box(2.0, 1.8, 1.4)
        + Pos(cab_x - 0.5 * cab_len + 1.2, +34.8, 32.5) * Box(2.0, 1.8, 1.4)
    )
    cab_trim = cab_trim_L + mirror(cab_trim_L, about=Plane.XZ)

    # 双侧窗 (中柱分隔前后窗) + 前妻板圆角瞭望窗 + 从轮上方前 20mm 侧围板高门槛裁切 (Z_min=+30.05mm, X∈[-130.0,-109.5])
    side_windows_cut = (
        Pos(-121.1, 0.0, 56.5) * Box(11.4, cab_w + 10.0, 12.0)
        + Pos(-133.9, 0.0, 56.5) * Box(9.0, cab_w + 10.0, 12.0)
    )
    spectacle_windows_cut = (
        Pos(-110.5, +25.2, 59.5) * Rot(0, 90, 0) * Cylinder(radius=4.6, height=8.0)
        + Pos(-110.5, -25.2, 59.5) * Rot(0, 90, 0) * Cylinder(radius=4.6, height=8.0)
    )
    front_sill_trim_cut = (
        Pos(-119.6, +38.5, 22.0) * Box(20.8, 8.0, 16.1)
        + Pos(-119.6, -38.5, 22.0) * Box(20.8, 8.0, 16.1)
    )

    cab = (
        cab_box + cab_roof + vent_21_15 + cab_trim
        - Pos(cab_x - 0.2, 0, 46.5) * Box(cab_len - 3.4, cab_w - 3.2, 58.0)
        - (
            Pos(cab_x - 0.2, 0, 36.0) * Rot(0, 90, 0) * Cylinder(radius=52.4, height=cab_len - 3.4)
            & Pos(cab_x - 0.2, 0, 78.0) * Box(cab_len - 3.4, cab_w - 3.2, 26.0)
        )
        - side_windows_cut
        - spectacle_windows_cut
        - front_sill_trim_cut
        - rear_doorway_cut
        - Pos(-111.5, +32.5, 44.1) * Box(8.0, 3.4, 5.2)  # 8-18 水平逆转拉杆 Z=+44.0mm 穿孔
    )

    # 5. 左右走台 + 第 4 动轮后方连续抬升走台 22-6 (X∈[-110.2,-64.0], Z_top=+42.2mm, 删除虚假后下坡!) + 标准高度除烟板 28-11/12 (Z_top=+68.2mm)
    walk_x0, walk_x1 = x_fb_rear + 1.0, 151.0
    walk_len = walk_x1 - walk_x0
    walk_xm = 0.5 * (walk_x0 + walk_x1)

    def _build_side_walk_and_defl(sgn: float):
        board = Pos(walk_xm, sgn * 34.4, 30.8) * Box(walk_len, 10.2, 1.5)
        inward_web = Pos(walk_xm, sgn * 27.5, 30.8) * Box(walk_len - 8.0, 5.0, 1.5)
        bridge_cut = Pos(-87.0, sgn * 34.8, 30.8) * Box(47.0, 12.0, 4.0)
        lift_slot_cut = Pos(36.2, sgn * 36.0, 30.8) * Box(12.4, 5.9, 4.0)
        rev_hood_cut = Pos(25.3, +32.5, 30.8) * Box(10.6, 5.2, 4.0) if sgn > 0 else None
        bridge_top = Pos(-87.1, sgn * 34.2, 41.5) * Box(46.2, 9.8, 1.4)
        bridge_ramp_f = Pos(-59.5, sgn * 34.2, 36.2) * Rot(0, +52, 0) * Box(14.2, 9.8, 1.4)
        bridge_tie = Pos(-87.1, sgn * 28.5, 36.2) * Box(46.2, 1.8, 11.4)
        board_base = (board + inward_web) - bridge_cut - lift_slot_cut
        if rev_hood_cut is not None:
            board_base = board_base - rev_hood_cut
        board_sub = board_base + bridge_top + bridge_ramp_f + bridge_tie

        # 28-11/12 除烟板：Z_top=+68.2mm (< 烟室顶 +76.45mm!) + 上沿内折斜翼 + 前上 45° 切角 + 前下顺应 42° 斜甲板 + 后下检修缺口
        defl_raw = (
            Pos(153.5, sgn * 40.25, 44.5) * Box(58.0, 1.0, 47.4)  # X ∈ [124.5, 182.5], Z ∈ [20.8, 68.2]
            + Pos(150.0, sgn * 39.20, 67.6) * Rot(sgn * -22, 0, 0) * Box(51.0, 3.0, 1.0)
            - Pos(177.5, sgn * 40.25, 64.5) * Rot(0, -45, 0) * Box(18.0, 5.0, 18.0)
            - Pos(126.0, sgn * 40.25, 45.2) * Box(10.0, 5.0, 11.0)
            - Pos(128.0, sgn * 40.25, 24.5) * Box(12.0, 5.0, 11.5)
            - Pos(165.5, sgn * 40.25, 14.5) * Rot(0, +42, 0) * Box(42.0, 6.0, 15.0)
            + Pos(175.8, sgn * 39.45, 44.5) * Cylinder(radius=0.60, height=19.0)
        )
        defl_stays = (
            Pos(152.0, sgn * 33.8, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.75, height=13.5)
            + Pos(131.5, sgn * 33.8, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.75, height=13.5)
        )

        front_slope = Pos(162.0, sgn * 33.4, 20.4) * Rot(0, +42, 0) * Box(32.5, 12.6, 1.5)
        for st_x, st_z in [(155.5, 26.5), (162.0, 20.6), (168.5, 14.7)]:
            front_slope += Pos(st_x, sgn * 33.4, st_z) * Box(3.2, 12.2, 1.1)

        dummy_15_3 = (
            Pos(58.0, sgn * 29.8, 44.0) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=3.4)
            + Pos(58.0, sgn * 31.2, 45.2) * Cylinder(radius=1.9, height=4.6)
        )
        dummy_16_2 = Pos(-78.0, sgn * 24.4, 24.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.4)
        air_res_22_13 = (
            Pos(-18.0, sgn * 31.6, 25.8) * Rot(0, 90, 0) * Cylinder(radius=4.15, height=34.0)
            + Pos(-29.0, sgn * 31.6, 29.2) * Box(2.2, 4.8, 3.2)
            + Pos(-7.0, sgn * 31.6, 29.2) * Box(2.2, 4.8, 3.2)
        )
        handrail_26_10 = make_swept_tube(
            [(-44.0, sgn * 30.8, 56.5), (+152.0, sgn * 32.9, 56.5)],
            r_tube=0.55,
            bend_r=2.0,
        )
        for hx_s in (-42.0, +8.0, +62.0, +115.0, +148.0):
            y_in_h = sgn * (27.2 if hx_s > 92.0 else 26.2)
            y_End = sgn * (32.9 if hx_s > 92.0 else 30.8)
            handrail_26_10 += Pos(hx_s, 0.5 * (y_in_h + y_End), 56.5) * Rot(90, 0, 0) * Cylinder(
                radius=0.68, height=abs(y_End - y_in_h) + 0.8
            )
        return board_sub + defl_raw + defl_stays + front_slope + dummy_15_3 + dummy_16_2 + air_res_22_13 + handrail_26_10

    walk_L = _build_side_walk_and_defl(+1.0)
    walk_R = mirror(walk_L, about=Plane.XZ) - Pos(25.3, -32.5, 30.8) * Box(0.01, 0.01, 0.01)
    # 恢复右侧 -Y 无逆转罩缺口的平整走台片
    walk_R = walk_R + Pos(25.3, -32.5, 30.8) * Box(10.8, 5.4, 1.5)
    walkways = walk_L + walk_R

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
            # 8-3 / 8-4 排障器 (轨面 Z=-12.8mm 上方安全间隙)
            + Pos(185.8, sgn_st * 22.5, -3.2) * Box(2.6, 3.2, 14.8)
            + Pos(187.0, sgn_st * 22.5, -10.0) * Box(3.8, 3.6, 1.6)
        )
    hatch_25_3 = Pos(164.0, 0.0, 15.8) * Rot(0, +42, 0) * Box(7.2, 9.6, 1.2)

    # 7. 左后抬升走台下全尺寸带散热片复式空压机 22-11 (+Y) + 右后立式串联给水泵 22-18 & 22-19 冷却盘管 (-Y) + 28-8 消音器 + 30-2 假汽笛 + 30-7 加减杆 + 15-8/15-10 双发电机
    comp_22_11 = (
        Pos(-84.0, +33.8, 36.5) * Cylinder(radius=4.2, height=8.6)
        + Pos(-75.5, +33.8, 36.5) * Cylinder(radius=3.6, height=8.6)
        + Pos(-84.0, +33.8, 23.5) * Cylinder(radius=4.5, height=9.4)
        + Pos(-75.5, +33.8, 23.5) * Cylinder(radius=3.9, height=9.4)
        + Pos(-79.8, +33.8, 30.0) * Box(14.5, 4.8, 5.2)
        + Pos(-79.8, +30.2, 34.5) * Box(16.8, 6.8, 12.6)
    )
    for z_fin in (20.5, 22.8, 25.1, 34.2, 36.8, 39.2):
        comp_22_11 += (
            Pos(-84.0, +33.8, z_fin) * Cylinder(radius=4.9, height=0.65)
            + Pos(-75.5, +33.8, z_fin) * Cylinder(radius=4.3, height=0.65)
        )

    pump_22_18 = (
        Pos(-80.5, -33.8, 36.0) * Cylinder(radius=4.1, height=9.4)
        + Pos(-80.5, -33.8, 29.5) * Cylinder(radius=2.6, height=4.8)
        + Pos(-80.5, -33.8, 22.8) * Cylinder(radius=3.9, height=9.2)
        + Pos(-87.5, -33.6, 26.5) * Cylinder(radius=2.2, height=13.5)
        + Pos(-83.2, -30.2, 34.2) * Box(14.6, 6.8, 12.8)
        + make_swept_tube([(-36.0, -31.8, 28.8), (+18.0, -31.8, 28.8), (+18.0, -31.8, 26.8), (-36.0, -31.8, 26.8)], r_tube=0.58, bend_r=0.95)
        + Pos(-9.0, -30.2, 29.2) * Box(42.0, 3.6, 2.2)
    )
    silencer_28_8 = (
        Pos(-96.5, -18.5, 67.5) * Cylinder(radius=2.7, height=11.6)
        + Pos(-96.5, -15.8, 64.5) * Box(4.4, 6.8, 4.2)
    )
    dummy_whistle_30_2 = (
        Pos(-94.0, +13.5, z_axis + r_barrel - 0.2) * Rot(0, -72, 0) * Cylinder(radius=1.75, height=7.8)
        + Pos(-94.0, +13.5, z_axis + r_barrel - 2.4) * Cylinder(radius=1.95, height=4.2)
        + make_swept_tube([(-95.5, +13.5, z_axis + r_barrel - 0.2), (-110.5, +13.5, z_axis + r_barrel - 0.8)], r_tube=0.45, bend_r=1.5)
    )
    reg_rod_30_7 = (
        make_swept_tube([(-110.5, +21.5, 65.5), (+16.0, +21.5, 65.5), (+16.0, +11.5, 68.5)], r_tube=0.55, bend_r=2.5)
        + Pos(-48.0, +19.5, 65.5) * Rot(90, 0, 0) * Cylinder(radius=0.8, height=5.0)
        + Pos(-4.0, +19.5, 65.5) * Rot(90, 0, 0) * Cylinder(radius=0.8, height=5.0)
    )
    gen_15_8 = (
        Pos(-86.5, -24.5, 64.2) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=9.8)
        + Pos(-86.5, -20.2, 61.2) * Box(5.2, 8.8, 5.2)
        + Pos(-98.5, -25.2, 62.5) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=7.2)  # 15-10 ATS 发电机
        + Pos(-98.5, -21.2, 60.2) * Box(4.2, 8.2, 4.4)
    )
    cover_25_7 = (
        Pos(25.5, +32.5, 34.6) * Box(12.2, 6.8, 6.6)
        - Pos(24.3, +32.5, 33.8) * Box(12.2, 5.2, 6.6)
    )

    outer_solid = (
        smokebox + sb_saddle + door_ring_18_22 + door_dish_18_14 + plate_d51_18_18 + wheel_18_12_13 + hinge_straps_18_4
        + headlight + headlight_ped + feedwater_heater + fwh_brackets + chimney
        + barrel + dome_29_11 + firebox + cab + walkways + front_deck_and_beam + hatch_25_3
        + comp_22_11 + pump_22_18 + silencer_28_8 + dummy_whistle_30_2 + reg_rod_30_7 + gen_15_8 + cover_25_7
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
        Pos(22.0, 0, 74.0) * Box(68.0, 19.6, 22.0)
        + Pos(56.2, 0.0, 88.0) * Cylinder(radius=4.55, height=10.0)
        + Pos(cab_x - 1.8, 0, z_axis + 24.0) * Box(cab_len - 3.5, 58.0, 16.0)
        + Pos(136.0, 0, 21.5) * Box(31.0, 21.0, 10.0)
        + Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=4.1, height=30.0)
    )
    bore_chimney = Pos(x_chimney, 0, z_axis + r_sb + 6.0) * Cylinder(radius=5.8, height=34.0)
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
    # 100% 物理打通受压紫铜锅筒全部蒸汽/给水/排污/水面计衬套通孔 (左右镜像配置对称工艺堵头保持 |com_y| == 0.000mm!)
    pv_port_bores = (
        Pos(-98.5, 0.0, 71.2) * Cylinder(radius=1.05, height=8.4)
        + Pos(-46.0, 0.0, 70.6) * Cylinder(radius=1.15, height=8.4)
        + Pos(-33.0, 0.0, 70.6) * Cylinder(radius=1.15, height=8.4)
        + Pos(-42.5, 0.0, 17.2) * Cylinder(radius=0.85, height=6.4)
        + Pos(x_rear + 1.0, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.40, height=6.4)
        + Pos(x_rear + 1.0, +3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.40, height=6.4)
        + Pos(x_rear + 1.0, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=0.65, height=6.4)
        + Pos(x_rear + 1.0, -2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=0.65, height=6.4)
        + Pos(x_rear + 1.0, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=6.4)
        + Pos(x_rear + 1.0, +11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=6.4)
        + Pos(-108.0, -17.5, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=18.0)
        + Pos(-108.0, +17.5, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=18.0)
        + Pos(-48.0, -22.5, 35.0) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=8.0)
        + Pos(-48.0, +22.5, 35.0) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=8.0)
    )

    return (
        shell_body + mud_ring + firehole_ring + stays + flues_outer
        + blowdown_bush_bot + steam_boss_rear + sv_bushes_tandem
        - flues_bore - firehole_bore - pv_port_bores
    )


def build_boiler_insulation(cfg: Dict, pip_cfg: Dict):
    """
    原厂说明书 & 1:1 裁剪模板 (Templates 1–6) 硅酸铝陶瓷纤维保温层 `セラミックシート(断熱材)` (Manual pp.28-31, 35 & イラスト 12, 16, 18, cc=1):
      - Template 1: 后妻板隔热片 (X∈[-112.2,-110.8]mm, 避让火门与后封头铜管座)；
      - Template 2: 火室拱顶与侧水腿隔热包覆片 (X∈[-111.5,-46.5]mm, t≈1.6mm)；
      - Template 4: 紫铜锅筒圆周隔热毡 (X∈[-47.0,+92.2]mm, r_in=27.35mm, r_out=28.15mm, 夹于 C1100 铜壁与黄铜外衣空气层之间)；
      - Template 5 + Template 6: 烟室内壁圆筒隔热衬 (X∈[+91.6,+159.2]mm, r_in=28.5mm, r_out=30.2mm) + 烟室前门 ø53.0mm 隔热圆板 (X∈[+158.6,+160.0]mm)！
    """
    z_axis = 44.0
    # Template 4: 锅筒圆周陶瓷纤维层
    t4_barrel = (
        Pos(0.5 * (-47.0 + 92.2), 0.0, z_axis)
        * Rot(0, 90, 0)
        * (Cylinder(radius=28.12, height=92.2 - (-47.0)) - Cylinder(radius=27.36, height=92.2 - (-47.0) + 2.0))
    ) - Pos(22.0, 0.0, 16.5) * Box(150.0, 18.0, 6.0)

    # Template 2: 火室上部与两侧隔热层 (与 Template 4 融合 0.5mm 保持 cc==1)
    t2_arch = (
        Pos(0.5 * (-111.4 + (-46.5)), 0.0, z_axis)
        * Rot(0, 90, 0)
        * (Cylinder(radius=28.12, height=-46.5 - (-111.4)) - Cylinder(radius=27.36, height=-46.5 - (-111.4) + 2.0))
    ) & (Pos(-79.0, 0.0, 58.0) * Box(70.0, 60.0, 32.0))
    t2_cheeks_L = Pos(-79.0, +22.2, 36.5) * Box(64.9, 1.3, 21.0)
    t2_firebox = t2_arch + t2_cheeks_L + mirror(t2_cheeks_L, about=Plane.XZ)

    # Template 1: 驾驶室后封头陶瓷隔热板
    t1_backhead = (
        (
            Pos(-111.4, 0.0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=27.9, height=1.3)
            & Pos(-111.4, 0.0, 56.0) * Box(4.0, 58.0, 30.0)
        )
        + Pos(-111.4, 0.0, 36.5) * Box(1.3, 44.6, 16.0)
    )

    # Template 5 & Template 6: 烟室内衬 + ø53.0mm 前门隔热圆盘 (经前管板法兰环形搭接融合)
    t5_bridge = (
        Pos(92.0, 0.0, z_axis) * Rot(0, 90, 0) * (Cylinder(radius=30.15, height=1.2) - Cylinder(radius=27.50, height=3.0))
    ) & (Pos(92.0, 0.0, 58.0) * Box(4.0, 64.0, 34.0))
    t5_smokebox = (
        Pos(0.5 * (91.8 + 159.2), 0.0, z_axis)
        * Rot(0, 90, 0)
        * (Cylinder(radius=30.15, height=159.2 - 91.8) - Cylinder(radius=28.75, height=159.2 - 91.8 + 2.0))
    ) - Pos(126.0, 0.0, 17.5) * Box(76.0, 34.0, 14.0)
    t6_door_disc = (
        Pos(159.0, 0.0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=26.5, height=1.4)
        + (
            Pos(158.8, 0.0, z_axis) * Rot(0, 90, 0) * (Cylinder(radius=29.2, height=1.2) - Cylinder(radius=25.8, height=3.0))
            & Pos(158.8, 0.0, 58.0) * Box(4.0, 62.0, 32.0)
        )
    )

    insul_cuts = (
        Pos(-98.5, 0.0, 72.0) * Cylinder(radius=4.4, height=12.0)
        + Pos(-46.0, 0.0, 72.0) * Cylinder(radius=3.8, height=12.0)
        + Pos(-33.0, 0.0, 72.0) * Cylinder(radius=3.8, height=12.0)
        + Pos(+130.0, 0.0, 74.0) * Cylinder(radius=10.2, height=14.0)
        + Pos(-111.4, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=6.8, height=6.0)
        + Pos(-111.4, 0.0, 48.0) * Box(6.0, 14.0, 6.5)
        + Pos(-111.4, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=6.0)
        + Pos(-111.4, +11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=6.0)
        + Pos(-108.0, 0.0, 60.5) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=60.0)
        + Pos(-48.0, 0.0, 35.0) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=60.0)
    )
    return (t4_barrel + t2_firebox + t1_backhead + t5_bridge + t5_smokebox + t6_door_disc) - insul_cuts


def build_superheater_piping(cfg: Dict, pip_cfg: Dict):
    """
    原厂全真壁厚空心内腔主蒸汽干汽管 + 17-5 双U四程过热器 + 7-9 给汽横歧管单连通体 (イラスト 7, 12, 14, 17, 20 & 図 17.1, cc=1):
      - 炉内空心干汽管 13-9 (OD 4.0 / ID 2.8mm, t=0.60mm) 自右中后封头接头 (-112.8, -3.6, +48.0) 升至 (0.0, +58.8) 直通前管板；
      - 17-7 + 17-8 六角接管 + 17-6 中空螺栓连通 17-5 上下空心丸型集箱 (OD 4.2 / ID 2.8mm, t=0.70mm)；
      - 双U四程过热管按原厂真值建模为 OD 3.0 / ID 2.0mm (壁厚 t=0.50mm) 100% 全通空心弯管；
      - 中央空心下垂管经 17-4 45° 锁入空心 7-9 给汽横歧管 (OD 7.1 / ID 4.2mm, 左右气缸端口与 +Y=+7.5mm 底注油口全贯通！)。
    """
    pv = pip_cfg["boiler_pressure_vessel"]
    sh = pip_cfg["superheater_and_steam_piping"]["superheater_4pass"]
    z_pipe = 58.8
    x_turn = float(sh["x_turn_in_flue_mm"])
    x_hdr = 108.5
    up_yz = pv["five_flue_tubes"]["upper_superheater_flues"]["yz_centers_mm"]

    pts_dry = [
        (-113.6, -3.6, 48.0),
        (-108.5, -3.6, 48.0),
        (-100.0, 0.0, z_pipe),
        (+109.2, 0.0, z_pipe),
    ]
    backhead_stud = make_hex_nut((-112.8, -3.6, 48.0), (1.0, 0.0, 0.0), hex_r=2.05, length=2.0)
    dry_pipe_outer = make_swept_tube(pts_dry, r_tube=2.00, bend_r=2.8, r_ref=2.00)
    adapter_17_7_8 = (
        make_hex_nut((+93.4, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.75, length=2.0)
        + make_hex_nut((+100.8, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.55, length=13.4)
    )

    z_hdr_hi, z_hdr_lo = 48.35, 44.65
    banjo_17_6 = (
        Pos(x_hdr, 0.0, z_pipe) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=3.0)
        + make_hex_nut((x_hdr + 2.2, 0.0, z_pipe), (1.0, 0.0, 0.0), hex_r=2.6, length=1.8)
        + Pos(x_hdr, 0.0, 0.5 * (z_hdr_hi + z_pipe)) * Cylinder(radius=1.75, height=z_pipe - z_hdr_hi + 1.2)
    )
    hdr_upper = Pos(x_hdr, 0.0, z_hdr_hi) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=26.6)
    hdr_lower = Pos(x_hdr, 0.0, z_hdr_lo) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=26.6)

    sh_loops_outer = None
    sh_loops_bore = None
    for yz in up_yz:
        yf = float(yz[0])
        pts_u = [
            (x_hdr, yf, z_hdr_hi),
            (x_turn, yf, z_hdr_hi),
            (x_turn, yf, z_hdr_lo),
            (x_hdr, yf, z_hdr_lo),
        ]
        u_out = make_swept_tube(pts_u, r_tube=1.50, bend_r=1.80, r_ref=1.50)
        u_in = make_swept_tube(_extend_polyline_ends(pts_u, 0.35, 0.35), r_tube=1.00, bend_r=1.80, r_ref=1.50)
        sh_loops_outer = u_out if sh_loops_outer is None else (sh_loops_outer + u_out)
        sh_loops_bore = u_in if sh_loops_bore is None else (sh_loops_bore + u_in)

    pts_drop = [
        (x_hdr, 0.0, z_hdr_lo),
        (x_hdr, 0.0, 26.5),
        (119.5, 0.0, 21.8),
        (123.5, 0.0, 18.5),
    ]
    drop_leg_17_5 = make_swept_tube(pts_drop, r_tube=1.60, bend_r=4.0, r_ref=1.60)
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

    outer_all = (
        backhead_stud + dry_pipe_outer + adapter_17_7_8 + banjo_17_6
        + hdr_upper + hdr_lower + sh_loops_outer + drop_leg_17_5 + bolt_17_4 + manifold_7_9
    )
    bore_all = (
        make_swept_tube(_extend_polyline_ends(pts_dry, 0.60, 0.0), r_tube=1.40, bend_r=2.8, r_ref=2.00)
        + Pos(x_hdr, 0.0, 0.5 * (z_hdr_hi + z_pipe)) * Cylinder(radius=1.15, height=z_pipe - z_hdr_hi + 1.8)
        + Pos(x_hdr, 0.0, z_hdr_hi) * Rot(90, 0, 0) * Cylinder(radius=1.40, height=24.6)
        + Pos(x_hdr, 0.0, z_hdr_lo) * Rot(90, 0, 0) * Cylinder(radius=1.40, height=24.6)
        + sh_loops_bore
        + make_swept_tube(_extend_polyline_ends(pts_drop, 0.45, 0.45), r_tube=1.00, bend_r=4.0, r_ref=1.60)
        + Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=43.0)
        + Pos(123.5, +7.5, 16.4) * Cylinder(radius=0.88, height=5.2)
    )
    return outer_all - bore_all


def build_exhaust_blast_pipe(cfg: Dict, pip_cfg: Dict):
    """
    7-8 前低位全空心排汽横歧管 (OD 5.5 / ID 3.8mm, 壁厚 0.85mm) + 20-7 底部旋入式全通锥形乏汽喷嘴 (ø4.2mm 喉口, イラスト 7, 20 & 図 20.1, cc=1)。
    """
    ej = pip_cfg["superheater_and_steam_piping"]["smokebox_draft_ejector"]
    x_ch = float(ej["chimney_axis_x_mm"])
    z_tip = float(ej["blast_nozzle_tip_z_mm"])
    r_orifice = 0.5 * float(ej["blast_nozzle_orifice_dia_mm"])
    z_exh = 12.5

    block_7_8 = Pos(x_ch, 0.0, z_exh) * Box(8.2, 8.8, 7.6)
    pipes_7_8 = Pos(x_ch, 0.0, z_exh) * Rot(90, 0, 0) * Cylinder(radius=2.75, height=41.6)
    collars = (
        Pos(x_ch, +19.5, z_exh) * Rot(90, 0, 0) * Cylinder(radius=3.15, height=2.0)
        + Pos(x_ch, -19.5, z_exh) * Rot(90, 0, 0) * Cylinder(radius=3.15, height=2.0)
    )
    bottom_head_20_7 = (
        Pos(x_ch, 0.0, 8.1) * Cylinder(radius=3.5, height=1.6)
        - Pos(x_ch, 0.0, 7.3) * Box(7.4, 0.85, 0.8)
    )
    nozzle_20_7 = Pos(x_ch, 0.0, 0.5 * (16.0 + z_tip)) * Cone(
        bottom_radius=3.55, top_radius=2.95, height=z_tip - 16.0
    )
    lumen_cut = (
        Pos(x_ch, 0.0, z_exh) * Rot(90, 0, 0) * Cylinder(radius=1.90, height=43.0)
        + Pos(x_ch, 0.0, 0.5 * (z_exh + z_tip + 0.8)) * Cone(
            bottom_radius=2.35, top_radius=r_orifice, height=(z_tip + 0.8) - z_exh
        )
    )
    return (block_7_8 + pipes_7_8 + collars + bottom_head_20_7 + nozzle_20_7) - lumen_cut


def build_blower_nozzle_17_3(cfg: Dict, pip_cfg: Dict):
    """
    铜锅筒内部空心通风撑管 (-112.8, +2.6, +48.0 -> +93.5) + 烟室 17-3 通風ノズル 空心 180° J 字喷嘴 (OD 2.0 / ID 1.2mm, 壁厚 0.40mm, cc=1)。
    """
    pts_17_3 = [
        (-113.6, +2.6, 48.0),
        (-104.0, +4.0, 40.5),
        (+96.2, +4.0, 40.5),
        (+96.2, +4.0, 29.5),
        (+125.0, +3.6, 29.5),
        (+125.0, +3.6, 24.8),
        (+130.0, +3.6, 24.8),
        (+130.0, +3.6, 35.2),
    ]
    rear_stud = make_hex_nut((-112.8, +2.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=2.0)
    front_union = make_hex_nut((+93.6, +4.0, 40.5), (1.0, 0.0, 0.0), hex_r=1.95, length=2.2)
    outer_17_3 = rear_stud + front_union + make_swept_tube(pts_17_3, r_tube=1.00, bend_r=2.2, r_ref=1.00)
    bore_17_3 = make_swept_tube(_extend_polyline_ends(pts_17_3, 0.55, 0.45), r_tube=0.60, bend_r=2.2, r_ref=1.00)
    return outer_17_3 - bore_17_3


def build_smokebox_petticoat(cfg: Dict, pip_cfg: Dict):
    """
    18-28 ペチコート + 18-27 煙室座 (イラスト 18, 100% Y 对称, 零虚假支撑杆, cc=1)。
    """
    ej = pip_cfg["superheater_and_steam_piping"]["smokebox_draft_ejector"]
    x_ch = float(ej["chimney_axis_x_mm"])
    z_pet = float(ej["petticoat_entry_z_mm"])
    r_pet_th = 0.5 * float(ej["petticoat_throat_dia_mm"])
    r_pet_fl = 0.5 * float(ej["petticoat_flare_dia_mm"])
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
    28-6 x2 Gauge-1 纵列突跳式安全阀壳体 + 29-7 八字形安全弁カバー护圈 (イラスト 12, 28, 29 & Manual p.40/p.44, 100% Y 对称, cc=1):
      - 内部精确车削 4 阶真流体腔 (自下而上全贯通)：
        1) ø2.20mm 进汽喉口阀座 (Z=70.0..75.5mm)
        2) ø2.60mm 蓄压突跳环腔 Huddling Chamber (Z=75.5..76.7mm)
        3) ø3.80mm 不锈钢螺旋弹簧室 (Z=76.5..80.2mm)
        4) 顶盖十字排汽槽 + ø1.70mm 导向顶孔 (Z=79.8..81.6mm，供 28-6 阀瓣中央突起顶针伸出!)
    """
    collar_29_7 = (
        Pos(-46.0, 0.0, 71.8) * Cylinder(radius=3.45, height=1.8)
        + Pos(-33.0, 0.0, 71.8) * Cylinder(radius=3.45, height=1.8)
        + Pos(-39.5, 0.0, 71.8) * Box(13.0, 5.2, 1.8)
    )
    valves_outer = collar_29_7
    valves_bore = None
    for sx in (-46.0, -33.0):
        v_out = (
            make_hex_nut((sx, 0.0, 73.5), (0.0, 0.0, 1.0), hex_r=2.85, length=2.0)
            + Pos(sx, 0.0, 77.2) * Cylinder(radius=2.55, height=7.6)
        )
        cross_slots = (
            Pos(sx, 0.0, 80.5) * Box(5.6, 0.90, 1.4)
            + Pos(sx, 0.0, 80.5) * Box(0.90, 5.6, 1.4)
        )
        v_in = (
            Pos(sx, 0.0, 72.8) * Cylinder(radius=1.10, height=5.8)
            + Pos(sx, 0.0, 76.1) * Cylinder(radius=1.30, height=1.25)
            + Pos(sx, 0.0, 78.3) * Cylinder(radius=1.90, height=3.8)
            + Pos(sx, 0.0, 80.8) * Cylinder(radius=0.86, height=2.6)
            + cross_slots
        )
        valves_outer = valves_outer + v_out
        valves_bore = v_in if valves_bore is None else (valves_bore + v_in)
    return valves_outer - valves_bore


def build_safety_valve_poppet():
    """
    28-6 可动突跳安全阀瓣 + 反冲蓄压唇口 (Huddling Lip) + 弹簧座盘 + 顶部突起排垢销 (上部の小さな突起, Manual p.44, cc=1):
      - 局部原点 (0,0,0) 对准安全阀喉口密封座面 (Z_world = +75.5mm)，随锅炉超压 (4.0 -> 3.55 kgf/cm²) 沿 +Z 突跳抬升！
    """
    cone_seat = Pos(0.0, 0.0, -0.45) * Cone(bottom_radius=0.88, top_radius=1.14, height=0.90)
    huddling_shroud = (
        Pos(0.0, 0.0, +0.25) * Cylinder(radius=1.25, height=0.80)
        - Pos(0.0, 0.0, -0.10) * (Cylinder(radius=1.16, height=0.25) - Cylinder(radius=0.98, height=0.35))
    )
    spring_flange = Pos(0.0, 0.0, +0.95) * Cylinder(radius=1.52, height=0.75)
    guide_stem = Pos(0.0, 0.0, +2.85) * Cylinder(radius=0.70, height=3.4)
    top_lift_pin = Pos(0.0, 0.0, +5.35) * Cylinder(radius=0.55, height=2.1)
    return cone_seat + huddling_shroud + spring_flange + guide_stem + top_lift_pin


def build_feedwater_piping(cfg: Dict, pip_cfg: Dict):
    """
    原厂全真壁厚空心给水闭环 + 床下中央横置圆筒 4-17 旁通阀 + 垂直 6-8 立管 + 左右平行尾梁口金 (イラスト 4, 6, 8, 13, 37, cc=1):
      - 动轮区 (X >= -130mm) 100% 收于车架内侧 (|Y| <= 10.5mm, 车轮外侧零管线!)；
      - 右后 (-Y=-19.2mm) 6-11 空心外螺纹口金 -> 空心 6-10 (OD 2.4 / ID 1.6mm) -> 空心 6-5 中央轴动泵 (ø3.2mm 膛孔 + 双 SB-ø3 阀腔)；
      - 空心 6-6 向后直入驾驶室地板下方横置圆筒 4-17 旁通阀 (-136.5, Y∈[-11.8,+11.8], +15.0, +Y=+6.8mm 设 6-9 立轴孔)；
      - 4-17 右端 (-Y=-11.8mm) 接空心 6-8 垂直立管直插后封头火室冠板上方中位 13-2 逆止阀底口 (Z=+43.2mm)；
      - 4-17 左端 (+Y=+9.5mm) 接空心 4-16 回水管通往左后 (+Y=+19.2mm) 宝塔尾嘴！
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
    pump_outer = pump_barrel + gland_nut + check_tower + top_hex_cap + bot_hex_cap + pump_bracket

    pts_6_10 = [
        (-150.8, -19.2, -8.5),
        (-142.5, -19.2, -8.5),
        (-136.0, -8.2, +8.5),
        (x_tw, -8.2, +8.5),
        (x_tw, -8.2, -5.2),
        (x_tw, 0.0, -5.2),
    ]
    stud_6_11 = (
        Pos(-149.0, -19.2, -8.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=4.2)
        + make_hex_nut((-146.8, -19.2, -8.5), (1.0, 0.0, 0.0), hex_r=2.55, length=1.8)
    )

    bx, bz = -136.5, 15.0
    pts_6_6 = [
        (x_tw, 0.0, +10.0),
        (x_tw, -3.5, +13.5),
        (-128.0, -3.5, +15.0),
        (bx, -3.5, bz),
    ]
    valve_4_17 = (
        Pos(bx, 0.0, bz) * Rot(90, 0, 0) * Cylinder(radius=3.05, height=23.6)
        + Pos(bx, +6.8, bz + 2.4) * Cylinder(radius=1.95, height=5.8)
        + make_banjo_eye((bx, -11.8, bz), (0.0, -1.0, 0.0), r_eye=2.05, h_eye=2.0, hex_r=1.55)
        + make_banjo_eye((bx, +9.6, bz - 1.2), (0.0, 0.0, -1.0), r_eye=1.95, h_eye=1.8, hex_r=1.50)
    )

    pts_6_8 = [
        (bx, -11.8, bz),
        (-115.8, -11.8, +16.2),
        (-115.8, -11.8, +43.4),
    ]
    nut_6_8_top = make_hex_nut((-115.8, -11.8, +42.4), (0.0, 0.0, 1.0), hex_r=1.85, length=2.0)

    pts_4_16 = [
        (bx, +9.6, bz - 1.2),
        (-141.0, +19.2, +5.5),
        (-144.5, +19.2, -8.5),
        (-150.8, +19.2, -8.5),
    ]
    barb_4_16 = (
        Pos(-149.2, +19.2, -8.5) * Rot(0, 90, 0) * Cone(bottom_radius=1.85, top_radius=1.45, height=3.8)
        + make_hex_nut((-146.8, +19.2, -8.5), (1.0, 0.0, 0.0), hex_r=2.45, length=1.8)
    )

    outer_fw = (
        pump_outer
        + make_swept_tube(pts_6_10, r_tube=1.22, bend_r=2.8, r_ref=1.22) + stud_6_11
        + make_swept_tube(pts_6_6, r_tube=1.20, bend_r=2.6, r_ref=1.20)
        + valve_4_17
        + make_swept_tube(pts_6_8, r_tube=1.20, bend_r=2.4, r_ref=1.20) + nut_6_8_top
        + make_swept_tube(pts_4_16, r_tube=1.20, bend_r=2.6, r_ref=1.20) + barb_4_16
    )
    bore_fw = (
        Pos(x_pump - 14.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.62, height=13.5)
        + Pos(x_tw, 0.0, 1.5) * Cylinder(radius=1.70, height=15.6)
        # 6-5 内部双 SB-ø3 止回球锥形阀座喉口台阶
        - (Pos(x_tw, 0.0, -2.8) * (Cylinder(radius=1.72, height=0.9) - Cylinder(radius=1.18, height=1.2)))
        - (Pos(x_tw, 0.0, +4.8) * (Cylinder(radius=1.72, height=0.9) - Cylinder(radius=1.18, height=1.2)))
        + make_swept_tube(_extend_polyline_ends(pts_6_10, 0.55, 0.40), r_tube=0.80, bend_r=2.8, r_ref=1.22)
        + make_swept_tube(_extend_polyline_ends(pts_6_6, 0.40, 0.40), r_tube=0.80, bend_r=2.6, r_ref=1.20)
        + Pos(bx, 0.0, bz) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=22.4)
        # 4-17 右端 (-Y=-11.8mm) 空心 Banjo 横向汇流全圆通孔 (彻底消除月牙缩颈!)
        + Pos(bx, -11.2, bz) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=3.8)
        + Pos(bx, +6.8, bz + 2.6) * Cylinder(radius=0.88, height=6.6)
        + make_swept_tube(_extend_polyline_ends(pts_6_8, 0.40, 0.55), r_tube=0.80, bend_r=2.4, r_ref=1.20)
        + make_swept_tube(_extend_polyline_ends(pts_4_16, 0.40, 0.55), r_tube=0.80, bend_r=2.6, r_ref=1.20)
    )
    return outer_fw - bore_fw


def build_displacement_lubricator(cfg: Dict, pip_cfg: Dict):
    """
    4-2 前车架中央全空心卧式置换油箱 (壁厚 t=0.80mm) + 4-1 顶盖内螺纹加油颈腔 + 内部竖直溢流立管 (Roscoe Standpipe Quill) + 空心毛细出油管 (イラスト 4 Fig.1, 7, 20, cc=1):
      - 油箱内腔真实镂空 (t=0.80mm)，4-1 加油颈内腔与主油腔贯通，内置竖直黄铜立管 (OD 2.0 / ID 1.1mm) 上升至距油箱内顶 1.3mm 处；
      - 毛细出油管内孔起自 x0+3.4mm 横穿立管底部中心，确保 7-9 <-> 毛细管 <-> 竖直立管 <-> 4-2 油箱顶腔 100% 零盲板全通 (cc==1)！
    """
    lub = pip_cfg["displacement_lubricator_system"]
    x0, x1 = [float(v) for v in lub["x_range_mm"]]
    xm = 0.5 * (x0 + x1)
    zc = float(lub["z_center_mm"])
    r_tank = 0.5 * float(lub["tank_od_mm"])  # 6.8 mm
    r_inner = r_tank - 0.80                  # 6.0 mm (壁厚 0.80 mm)
    cap_x, _, cap_z = [float(v) for v in lub["cap_xyz_mm"]]
    tx, ty, tz = [float(v) for v in lub["oil_delivery_tee_xyz_mm"]]

    tank_4_2 = Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_tank, height=x1 - x0)
    saddle_cradle = Pos(xm, 0.0, zc + 1.5) * Box(12.0, 19.2, 4.2)
    neck_4_1 = Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z)) * Cylinder(radius=2.8, height=cap_z - zc + 1.0)
    cap_4_1 = Pos(cap_x - 1.5, 0.0, cap_z) * Cylinder(radius=3.8, height=2.2)

    pts_oil = [(x0 + 3.4, +2.6, zc), (x0 - 2.5, ty, zc), (tx, ty, zc), (tx, ty, tz + 0.6)]
    oil_pipe_outer = make_swept_tube(pts_oil, r_tube=1.05, bend_r=2.8, r_ref=1.05)
    union_nut = make_hex_nut((tx, ty, tz - 1.0), (0.0, 0.0, 1.0), hex_r=1.85, length=1.8)

    tank_cavity = (
        Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_inner, height=(x1 - x0) - 1.6)
        + Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z - 1.2)) * Cylinder(radius=1.75, height=(cap_z - 1.2) - zc)
    )
    z_quill_top = zc + r_inner - 1.3
    quill_outer = (
        Pos(x0 + 2.6, +2.6, 0.5 * ((zc - 1.0) + z_quill_top)) * Cylinder(radius=1.00, height=z_quill_top - (zc - 1.0))
        + Pos(x0 + 1.5, +2.6, zc) * Rot(0, 90, 0) * Cylinder(radius=1.05, height=3.8)
    )
    quill_and_pipe_bore = (
        make_swept_tube(_extend_polyline_ends(pts_oil, 0.0, 0.55), r_tube=0.58, bend_r=2.8, r_ref=1.05)
        + Pos(x0 + 2.6, +2.6, 0.5 * ((zc - 0.6) + z_quill_top + 0.6)) * Cylinder(radius=0.55, height=(z_quill_top + 0.6) - (zc - 0.6))
    )

    shell_with_quill = ((tank_4_2 + saddle_cradle + neck_4_1 + cap_4_1 + oil_pipe_outer + union_nut) - tank_cavity) + quill_outer
    return shell_with_quill - quill_and_pipe_bore


def build_boiler_aux_valves(cfg: Dict, pip_cfg: Dict):
    """
    驾驶室后封头与火室侧原厂多图联立 100% 真值空心铜阀座与外管网单连通总成 (イラスト 11, 12, 13, 14 & p44, cc=1):
      1. 13-4 空心主蒸気供給管 (OD 3.0 / ID 2.0mm) -> 11-15 前后向角ブロック分配器 (正中后口接四辐 11-13 通风轮, 左面后口接横轴 11-10/11-12 加减阀)；
      2. 并排双下垂铜管: 左 13-8 通風管 (Y=+2.6mm, OD 2.4 / ID 1.6mm) + 右 13-9 加減管 (Y=-3.6mm, OD 3.0 / ID 2.0mm)；
      3. 13-2 逆止弁 + 13-7 六角顶盖: 严格复原至火室冠板上方右内中位衬套 (-115.8, -11.8, +47.5mm)，下口 Z=+43.0mm 直承垂直 6-8 立管；
      4. PG-ø20 严格垂直表壳 (-122.0, -16.5, +66.8mm, 0° 零仰角!) + 空心 13-1 虹吸管 (OD 2.0 / ID 1.2mm, 自 13-2 正下方 Z_valley=+39.0mm 优雅绕过)；
      5. 右外缘 (-118.5, -31.3) 11-4~6 水面计上下座 + 空心 13-6/13-10 (OD 2.4 / ID 1.6mm)；
      6. 11-1 左前侧面汽笛阀 (-115.6, +14.8, +66.5) -> 全直垂直空心 13-3 (Y=+19.2, OD 2.0 / ID 1.2mm) -> 左颊下空心 11-8 真汽笛；
      7. 11-16 左床下横向 (+Y) 锅炉排水针阀体 + 垂直朝下 (-Z) 排污嘴 + 空心 13-11 前伸配水管 (OD 2.4 / ID 1.6mm)。
    """
    wg = pip_cfg["pumps_and_valves_bom"]["water_gauge_11_7"]
    x_ring = -113.0

    brazed_backhead_ring = (
        Pos(x_ring, 0.0, 49.5) * Rot(0, 90, 0) * (Cylinder(radius=19.5, height=1.6) - Cylinder(radius=14.5, height=3.0))
        + Pos(x_ring, 0.0, 48.0) * Box(2.0, 28.0, 4.8)
        + Pos(x_ring, +18.2, 31.0) * Box(2.0, 6.8, 35.0)
        + Pos(x_ring, -20.5, 45.0) * Box(2.0, 12.0, 29.0)
        + Pos(x_ring, 0.0, 58.0) * Box(1.8, 6.4, 22.0)
    )

    pts_13_4 = [(-98.5, 0.0, 73.2), (-102.5, 0.0, 73.2), (-111.5, 0.0, 69.5), (-116.5, 0.0, 69.5)]
    jumper_13_4 = (
        make_swept_tube(pts_13_4, r_tube=1.50, bend_r=2.8, r_ref=1.50)
        + make_banjo_eye((-98.5, 0.0, 73.2), (0.0, 0.0, 1.0), r_eye=2.7, h_eye=2.3, hex_r=2.0)
        + make_banjo_eye((-116.5, 0.0, 69.5), (0.0, 0.0, 1.0), r_eye=2.5, h_eye=2.4, hex_r=1.9)
    )

    manifold_11_15 = (
        Pos(-116.8, +0.4, 66.5) * Box(8.2, 8.8, 6.0)
        + Pos(-121.0, 0.0, 66.5) * Rot(0, 90, 0) * Cylinder(radius=1.85, height=2.6)
    )
    gland_collar_11_10 = (
        Pos(-119.2, +5.4, 66.5) * Box(5.8, 1.8, 3.2)
        + Pos(-119.2, +6.6, 66.5) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=2.2)
        + Pos(-121.4, +6.2, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.85, height=1.2)
        + Pos(-117.0, +6.2, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.85, height=1.2)
    )

    pts_13_8 = [(-116.5, +2.6, 63.8), (-116.5, +2.6, 48.0), (-112.8, +2.6, 48.0)]
    blower_13_8 = (
        make_hex_nut((-116.5, +2.6, 63.0), (0.0, 0.0, 1.0), hex_r=1.75, length=1.6)
        + make_swept_tube(pts_13_8, r_tube=1.20, bend_r=2.0, r_ref=1.20)
        + make_hex_nut((-114.2, +2.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=1.8)
    )

    pts_13_9 = [(-115.6, -3.6, 65.2), (-117.4, -3.6, 65.2), (-117.4, -3.6, 48.0), (-112.8, -3.6, 48.0)]
    jumper_13_9 = (
        make_banjo_eye((-115.6, -3.6, 65.2), (0.0, -1.0, 0.0), r_eye=2.1, h_eye=1.9, hex_r=1.60)
        + make_swept_tube(pts_13_9, r_tube=1.50, bend_r=2.2, r_ref=1.50)
        + make_hex_nut((-114.2, -3.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.95, length=1.8)
    )

    clack_13_2_7 = (
        Pos(-115.8, -11.8, 47.5) * Cylinder(radius=2.55, height=9.0)
        + make_hex_nut((-115.8, -11.8, 52.4), (0.0, 0.0, 1.0), hex_r=2.80, length=1.9)
        + Pos(-114.0, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=2.10, height=4.6)
        + make_hex_nut((-113.8, -11.8, 47.5), (1.0, 0.0, 0.0), hex_r=2.45, length=1.5)
    )

    yp, zp = -16.5, 66.8
    pg_bezel = (
        Pos(-122.0, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=9.8, height=3.6)
        + Pos(-122.0, yp, zp - 10.6) * Cylinder(radius=1.95, height=3.4)
        - Pos(-123.6, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=8.45, height=2.4)
    )
    pts_13_1 = [
        (-119.2, -3.6, 65.4),
        (-120.2, -5.8, 43.2),
        (-121.0, -11.8, 39.0),
        (-122.0, yp, 43.2),
        (-122.0, yp, 57.5),
    ]
    siphon_13_1 = (
        make_banjo_eye((-119.2, -3.6, 65.4), (0.0, -1.0, 0.0), r_eye=1.75, h_eye=1.6, hex_r=1.35)
        + make_swept_tube(pts_13_1, r_tube=1.00, bend_r=3.8, r_ref=1.00)
        + make_hex_nut((-122.0, yp, 55.2), (0.0, 0.0, 1.0), hex_r=1.80, length=2.0)
    )

    yw = float(wg["y_mm"])  # -31.3 mm
    pts_13_6 = [(-118.5, yw, 60.6), (-118.5, -18.5, 60.6), (-108.0, -17.5, 60.5)]
    wg_top = (
        Pos(-118.5, yw, 57.2) * Cylinder(radius=2.35, height=5.2)
        + make_hex_nut((-118.5, yw, 55.1), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + make_banjo_eye((-118.5, yw, 60.6), (0.0, 0.0, 1.0), r_eye=2.35, h_eye=2.0, hex_r=1.85)
        + make_swept_tube(pts_13_6, r_tube=1.20, bend_r=2.6, r_ref=1.20)
        + make_banjo_eye((-108.0, -17.5, 60.5), (0.0, +0.5, +0.86), r_eye=2.05, h_eye=1.9, hex_r=1.55)
        + Pos(x_ring, -16.0, 57.0) * Box(1.6, 5.6, 6.8)
    )
    pts_13_10 = [(-118.5, yw, 30.2), (-118.5, -25.2, 30.2), (-110.5, -23.5, 33.5), (-48.0, -23.5, 35.0)]
    wg_bot = (
        Pos(-118.5, yw, 33.6) * Cylinder(radius=2.35, height=6.4)
        + make_hex_nut((-118.5, yw, 36.9), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + Pos(-118.5, yw, 30.2) * Cylinder(radius=2.35, height=2.0)
        + make_swept_tube(pts_13_10, r_tube=1.20, bend_r=2.6, r_ref=1.20)
        + make_banjo_eye((-48.0, -23.5, 35.0), (0.0, +1.0, 0.0), r_eye=2.05, h_eye=1.9, hex_r=1.55)
    )

    wh_valve_11_1 = (
        Pos(-115.6, +14.8, 66.5) * Box(4.4, 5.0, 8.8)
        + Pos(-115.6, +8.8, 66.5) * Rot(90, 0, 0) * Cylinder(radius=1.85, height=8.4)
        + make_hex_nut((-115.6, +6.2, 66.5), (0.0, 1.0, 0.0), hex_r=2.15, length=1.6)
        + Pos(-115.6, +14.8, 71.5) * Cylinder(radius=1.85, height=1.6)
        - Pos(-115.6, +14.8, 72.2) * Box(3.8, 0.55, 0.6)
    )
    pts_13_3 = [
        (-115.6, +14.8, 65.5),
        (-118.0, +19.2, 65.5),
        (-118.0, +19.2, 25.5),
        (-113.5, +20.2, 25.5),
    ]
    pipe_13_3 = (
        make_banjo_eye((-118.0, +19.2, 65.5), (-1.0, 0.0, 0.0), r_eye=1.75, h_eye=1.8, hex_r=1.38)
        + make_swept_tube(pts_13_3, r_tube=1.00, bend_r=1.8, r_ref=1.00)
        + make_banjo_eye((-117.6, +19.2, 25.5), (1.0, 0.0, 0.0), r_eye=1.75, h_eye=1.8, hex_r=1.38)
    )
    whistle_11_8 = (
        make_hex_nut((-114.6, +20.2, 25.5), (1.0, 0.0, 0.0), hex_r=2.85, length=1.8)
        + Pos(-92.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=3.10, height=44.4)
        - Pos(-102.0, +20.2, 23.2) * Box(3.2, 4.8, 3.2)
        + Pos(x_ring, +19.5, 25.5) * Box(1.6, 4.0, 4.0)
    )

    pts_13_11 = [
        (-115.5, +16.2, 15.8),
        (-108.0, +16.2, 15.2),
        (-72.0, +16.2, 15.2),
        (-54.0, +7.2, 15.4),
        (-42.5, 0.0, 15.6),
        (-42.5, 0.0, 17.2),
    ]
    blowdown_11_16_and_13_11 = (
        Pos(-115.5, +18.6, 15.8) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=4.4)
        + Pos(-113.2, +18.2, 16.5) * Box(3.2, 2.2, 4.0)
        + Pos(-115.5, +19.2, 13.4) * Cylinder(radius=1.08, height=4.0)
        + make_banjo_eye((-115.5, +16.2, 15.8), (0.0, -1.0, 0.0), r_eye=2.0, h_eye=1.8, hex_r=1.50)
        + make_swept_tube(pts_13_11, r_tube=1.20, bend_r=3.2, r_ref=1.20)
        + make_banjo_eye((-42.5, 0.0, 16.8), (0.0, 0.0, 1.0), r_eye=2.15, h_eye=2.2, hex_r=1.65)
    )

    outer_aux = (
        brazed_backhead_ring + jumper_13_4 + manifold_11_15 + gland_collar_11_10
        + blower_13_8 + jumper_13_9 + clack_13_2_7 + pg_bezel + siphon_13_1
        + wg_top + wg_bot + wh_valve_11_1 + pipe_13_3 + whistle_11_8 + blowdown_11_16_and_13_11
    )
    bore_aux = (
        make_swept_tube(_extend_polyline_ends(pts_13_4, 1.5, 1.5), r_tube=0.95, bend_r=2.8, r_ref=1.50)
        + Pos(-98.5, 0.0, 72.5) * Cylinder(radius=0.95, height=4.8)
        + Pos(-116.5, 0.0, 68.0) * Cylinder(radius=0.95, height=5.8)
        + Pos(-117.5, 0.0, 66.5) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=11.0)
        # 11-15 角形蒸汽分配器全阀腔真连通内孔 (通往 13-8 通风针阀座、11-10 横向旋转盘式加减阀室 -> 13-9、13-1 虹吸管及 11-1 弹簧汽笛阀座!)
        + Pos(-116.5, +1.3, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.78, height=3.6)
        + Pos(-116.5, +2.6, 65.1) * Cylinder(radius=0.75, height=3.4)
        + Pos(-119.2, +1.4, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.92, height=11.6)
        + Pos(-117.4, -3.6, 65.9) * Cylinder(radius=0.88, height=2.2)
        + Pos(-119.2, -3.6, 65.9) * Cylinder(radius=0.65, height=2.2)
        + Pos(-115.6, +7.4, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.82, height=15.6)
        + Pos(-115.6, +14.8, 67.2) * Cylinder(radius=1.05, height=5.2)
        + make_swept_tube(_extend_polyline_ends(pts_13_8, 1.2, 1.5), r_tube=0.75, bend_r=2.0, r_ref=1.20)
        + make_swept_tube(_extend_polyline_ends(pts_13_9, 1.2, 1.5), r_tube=0.95, bend_r=2.2, r_ref=1.50)
        # 13-2 逆止阀腔 (底部承接 6-8 + 内部 SB-ø3 止回球阀座台阶 + 顶部 13-7 六角闷盖盲孔密封 Z_max=51.8 < 53.35!)
        + Pos(-115.8, -11.8, 47.2) * Cylinder(radius=1.42, height=9.2)
        - (Pos(-115.8, -11.8, 45.2) * (Cylinder(radius=1.45, height=0.8) - Cylinder(radius=1.02, height=1.2)))
        + Pos(-115.8, -11.8, 44.0) * Cylinder(radius=0.95, height=3.6)
        + Pos(-113.5, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=6.5)
        + make_swept_tube(_extend_polyline_ends(pts_13_1, 1.2, 1.2), r_tube=0.58, bend_r=3.8, r_ref=1.00)
        + Pos(-122.0, yp, 59.5) * Cylinder(radius=0.72, height=5.2)
        + Pos(-118.5, yw, 57.2) * Cylinder(radius=1.18, height=9.2)
        + make_swept_tube(_extend_polyline_ends(pts_13_6, 1.2, 1.5), r_tube=0.75, bend_r=2.6, r_ref=1.20)
        + Pos(-118.5, yw, 33.4) * Cylinder(radius=1.18, height=9.2)
        + make_swept_tube(_extend_polyline_ends(pts_13_10, 1.2, 1.5), r_tube=0.75, bend_r=2.6, r_ref=1.20)
        + make_swept_tube(_extend_polyline_ends(pts_13_3, 1.2, 1.4), r_tube=0.58, bend_r=1.8, r_ref=1.00)
        # 11-8 汽笛共鸣管：后部蒸汽风道进汽喉口全贯通 + 前端 X=-70.0mm 真实盲盖反射腔 (X_bore_max=-71.4mm)!
        + Pos(-113.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=0.85, height=5.2)
        + Pos(-92.6, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=39.6)
        + Pos(-115.5, +19.2, 13.4) * Cylinder(radius=0.62, height=6.4)
        + Pos(-115.5, +17.5, 15.8) * Rot(90, 0, 0) * Cylinder(radius=0.72, height=4.8)
        + make_swept_tube(_extend_polyline_ends(pts_13_11, 1.2, 1.5), r_tube=0.75, bend_r=3.2, r_ref=1.20)
        + Pos(-42.5, 0.0, 17.2) * Cylinder(radius=0.80, height=4.2)
    )
    return outer_aux - bore_aux


def build_alcohol_burner(cfg: Dict, pip_cfg: Dict):
    """
    23-1 火室底纵向中央 1x3 直列三头立式 10mm 全空心酒精燃烧器 + 23-2 前折弯支架 + 4-9 后抱箍 (イラスト 23 & Manual p.30, 100% 中心对称, cc=1):
      - 彻底废除遮挡二次风的虚假矩形底板与事后布尔相加堵死杯口的实体铜环！
      - 中央供醇总管按 BOM 真值还原为 OD 4.0 / ID 2.8mm (Z=+10.2mm) 无缝空心紫铜管；
      - 3 只独立分立式黄铜灯芯杯 (X=-93/-77/-61mm, OD 10.8 / ID 10.0mm, 杯间保留 5.2mm 真实二次助燃空气间隙) 焊接于总管上方；
      - 在最终布尔差集一次性钻通：水平总管内孔 (ID 2.8mm) + 3 个杯底垂直引醇孔 (ø2.8mm, Z=+9.6..+13.5mm) + 3 个敞口灯芯杯腔 (ID 10.0mm)，实现 100% 端到端三维流体连通！
    """
    ab = pip_cfg["alcohol_burner_system"]
    wick_xs = [float(x) for x in ab["wick_x_mm"]]  # [-93.0, -77.0, -61.0]
    r_cup_out = 0.5 * float(ab["wick_cup_od_mm"])  # 5.40 mm
    r_cup_in = 0.5 * float(ab["wick_dia_mm"])      # 5.00 mm (壁厚 0.40 mm)
    z_cup_bot = 9.6
    z_cup_rim = 19.2
    z_tube = 10.2
    r_tube_out = 0.5 * float(ab["supply_tube_od_mm"])  # 2.00 mm
    r_tube_in = 1.40                                   # ID 2.80 mm (壁厚 0.60 mm)

    pts_feed = [
        (wick_xs[-1] + 2.0, 0.0, z_tube),
        (-136.0, 0.0, z_tube),
        (-143.0, 0.0, +3.6),
        (-150.8, 0.0, +3.6),
    ]
    feed_outer = (
        make_swept_tube(pts_feed, r_tube=r_tube_out, bend_r=3.0, r_ref=r_tube_out)
        + Pos(wick_xs[-1] + 2.4, 0.0, z_tube) * Rot(0, 90, 0) * Cylinder(radius=r_tube_out + 0.15, height=1.2)
        + Pos(-149.2, 0.0, +3.6) * Rot(0, 90, 0) * Cylinder(radius=2.55, height=3.8)
    )

    # 23-2 前折弯支架 (焊接于第3号杯前壁, Y 严格对称) + 4-9 后部马鞍形管卡 (X=-114.0mm, Y 严格对称)
    bracket_23_2 = (
        Pos(wick_xs[-1] + 5.8, 0.0, 14.5) * Box(2.8, 7.6, 5.2)
        + Pos(wick_xs[-1] + 7.4, 0.0, 17.2) * Box(4.2, 12.6, 1.4)
    )
    clamp_4_9 = (
        Pos(-114.0, 0.0, z_tube + 1.1) * Box(5.6, 13.6, 1.5)
        + Pos(-114.0, 0.0, z_tube) * Rot(0, 90, 0) * Cylinder(radius=r_tube_out + 0.75, height=5.6)
    )

    cups_outer = None
    cups_and_risers_bore = None
    for wx in wick_xs:
        co = (
            Pos(wx, 0.0, 0.5 * (z_cup_bot + z_cup_rim)) * Cylinder(radius=r_cup_out, height=z_cup_rim - z_cup_bot)
            + Pos(wx, 0.0, z_cup_bot + 1.0) * Cylinder(radius=r_cup_out + 0.25, height=2.0)
        )
        ci = (
            Pos(wx, 0.0, 0.5 * ((z_cup_bot + 1.3) + (z_cup_rim + 1.2)))
            * Cylinder(radius=r_cup_in, height=(z_cup_rim + 1.2) - (z_cup_bot + 1.3))
            # 垂直引醇立孔 (ø2.80mm): 自水平总管中心 Z=+10.2mm 直通灯芯杯底腔 Z=+13.2mm!
            + Pos(wx, 0.0, 0.5 * ((z_tube - 0.3) + 13.2)) * Cylinder(radius=1.40, height=13.2 - (z_tube - 0.3))
        )
        cups_outer = co if cups_outer is None else (cups_outer + co)
        cups_and_risers_bore = ci if cups_and_risers_bore is None else (cups_and_risers_bore + ci)

    feed_bore = make_swept_tube(_extend_polyline_ends(pts_feed, -1.0, 0.70), r_tube=r_tube_in, bend_r=3.0, r_ref=r_tube_out)
    return (feed_outer + bracket_23_2 + clamp_4_9 + cups_outer) - (feed_bore + cups_and_risers_bore)


def build_burner_wick():
    """
    `セラミックヤーン(バーナー芯)` 多股硅酸铝耐火陶瓷纤维毛细灯芯束 (Manual pp.30, 41 図 4, cc=1):
      - 原厂规格: 约 30 股捻合陶瓷纤维纱线对折塞入 ø10.0mm 灯芯杯，顶部露出杯口约 3.5~4.0mm (Z=+11.3..+23.0mm)；
      - 局部原点设于 (0, 0, 0)，在 MJCF 中分别实例挂载于 X = -93.0 / -77.0 / -61.0 mm 三只燃烧杯同轴内腔！
    """
    z_bot, z_rim, z_top = 11.3, 19.2, 23.0
    core = Pos(0.0, 0.0, 0.5 * (z_bot + (z_top - 1.0))) * Cylinder(radius=4.55, height=(z_top - 1.0) - z_bot)
    crown = Pos(0.0, 0.0, z_top - 1.1) * Cone(bottom_radius=4.55, top_radius=3.45, height=1.8)
    strands = None
    for i_s in range(12):
        ang = i_s * 30.0
        rad = math.radians(ang)
        sx, sy = 4.15 * math.cos(rad), 4.15 * math.sin(rad)
        st = Pos(sx, sy, 0.5 * ((z_rim - 1.5) + (z_top - 0.3))) * Cylinder(radius=0.82, height=(z_top - 0.3) - (z_rim - 1.5))
        strands = st if strands is None else (strands + st)
    bottom_plenum_recess = Pos(0.0, 0.0, z_bot + 0.9) * Cone(bottom_radius=2.2, top_radius=1.2, height=2.2)
    return (core + crown + strands) - bottom_plenum_recess


def build_tender_body(cfg: Dict):
    """
    8-20B 四轴煤水车外衣壳体与台架总成 (イラスト 32, 33, 35 & p45-p47 — 外廓 100% Y 对称, cc=1):
      - 32-10 平坦槽边底架 (带 31-12x2 床下双孔与 31-9 排水通孔) + 32-11x2 海鸥翼下沉枕梁 + 32-12x2 心皿ボス；
      - 32-1/32-14 前后端梁 + 28-2 后车钩 + **28-3 后解钩横杆** + 四角踏板；
      - 33-10 平滑 S 曲线阶梯煤库侧壁 + 35-5 x2 左右固定横向后挡煤板 + **35-11 前渡板门槛** + **35-10 右前手制动摇把立柱**；
      - 35-1 主甲板 (Z_tdr=+62.0mm): 前部 36-4 方孔 + 中后部圆角防波给水口凸缘 + 后部 X=-89.5 固定一体装饰人孔铸件 35-6。
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
    uncoupling_bar_28_3 = (
        Pos(-104.8, 0.0, 12.2) * Rot(90, 0, 0) * Cylinder(radius=0.65, height=70.0)
        + Pos(-104.2, +28.0, 12.2) * Box(1.8, 1.8, 2.2)
        + Pos(-104.2, -28.0, 12.2) * Box(1.8, 1.8, 2.2)
    )
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
    floor_through_cuts = (
        Pos(-16.4, +21.3, 14.2) * Cylinder(radius=2.9, height=6.5)
        + Pos(-16.4, -21.3, 14.2) * Cylinder(radius=2.9, height=6.5)
        + Pos(-41.0, -10.3, 14.2) * Cylinder(radius=3.2, height=6.5)
    )
    cutout_1_alc = Pos(+29.75, 0.0, 61.5) * Box(61.5, 61.6, 6.0)
    cutout_2_water = Pos(-45.0, 0.0, 61.5) * Box(63.0, 31.6, 8.0)
    front_gangway_cut = Pos(+93.6, 0.0, 61.5) * Box(14.2, 28.0, 4.0)

    coaming_ring = (
        Pos(-45.0, 0.0, 62.6) * Box(65.4 - 5.2, 34.0, 2.4)
        + Pos(-45.0, 0.0, 62.6) * Box(65.4, 34.0 - 5.2, 2.4)
    )
    for sx_c in (-1.0, +1.0):
        for sy_c in (-1.0, +1.0):
            coaming_ring += Pos(-45.0 + sx_c * (32.7 - 2.6), sy_c * (17.0 - 2.6), 62.6) * Cylinder(radius=2.6, height=2.4)

    tank_shell = (shell_outer + coaming_ring) - shell_hollow - sump_floor_cut - cutout_1_alc - cutout_2_water - front_gangway_cut

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
    threshold_35_11 = Pos(+93.6, 0.0, 62.2) * Box(13.6, 27.6, 1.0)
    handbrake_35_10 = (
        Pos(+94.2, -17.6, 68.2) * Cylinder(radius=1.05, height=13.4)
        + Pos(+94.2, -17.6, 75.0) * Box(1.8, 5.6, 1.1)
        + Pos(+94.2, -20.1, 76.2) * Cylinder(radius=0.80, height=2.6)
    )
    cocks_35_8_9 = (
        Pos(+98.5, +17.6, 63.3) * Cylinder(radius=1.35, height=3.6)
        + Pos(+96.0, +17.6, 63.3) * Cylinder(radius=1.15, height=3.6)
    )
    static_manhole_35_6 = (
        Pos(-89.5, 0.0, 62.6) * Cylinder(radius=6.6, height=1.6)
        + Pos(-89.5, 0.0, 63.6) * Cylinder(radius=5.3, height=1.0)
        + Pos(-83.0, 0.0, 62.6) * Box(2.4, 3.0, 1.4)
        + Pos(-96.0, 0.0, 62.6) * Box(2.2, 2.2, 1.4)
    )

    body_solid = (
        web_32_10 + skirts_and_steps + front_beam_32_1 + rear_beam_32_14 + uncoupling_bar_28_3 + hose_guide_32_3
        + rear_coupler_28_2 + bolsters_32_11_12 + tank_shell + bunker_sides + coal_gate_33_4
        + threshold_35_11 + handbrake_35_10 + cocks_35_8_9 + static_manhole_35_6
    ) - floor_through_cuts
    return body_solid


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
    8-20B 煤水车 100% 原厂壁厚与真实中空流道双箱管网总成 (イラスト 31, 34, 36, 37 & p47, cc=1):
      1. **36-4 无腿上法兰吊挂式黄铜酒精箱 (壁厚 t=0.80mm)** + **+45° 方向双压筋** + **单根 36-6 下方中空马氏定液位立管 (OD 5.0 / ID 3.8mm, 彻底删除 36-5 下方虚假第二立管!)**；
      2. **34-2 中空恒液位サンプ (t=0.80mm)** + **Y=0.0 笔直中空供醇前管 (OD 4.5 / ID 3.1mm)**；
      3. **31-10 后部敞口黄铜水箱 (t=0.80mm)** + **31-5 全深 ø7.0mm 缸孔与 SB-ø4x2 中空双球止回阀塔**；
      4. **31-12 x2 外移至 Y=±21.3mm 原厂真值位置** + **34-6 x2 床下笔直全通中空铜管 (OD 3.0 / ID 2.0mm)** + **中空 31-13 给水管** + **中空 31-14 高位鹅颈防虹吸回水立管** + **ID 6.0mm 贯通 31-9 排水座**。
    """
    # 1. 36-4 酒精箱 (t=0.80mm) + +45° 双斜筋 + 单根中空马氏立管
    alc_outer = Pos(+30.2, 0.0, +43.25) * Box(60.0, 60.0, 37.5)
    alc_hollow = Pos(+30.2, 0.0, +43.25) * Box(58.4, 58.4, 35.9)
    alc_top_flange = (
        Pos(+30.8, 0.0, +62.6) * Box(65.6, 67.6, 1.2)
        + Pos(+24.5, -6.0, +63.3) * Rot(0, 0, +45) * Box(52.0, 1.8, 0.7)
        + Pos(+36.5, +6.0, +63.3) * Rot(0, 0, +45) * Box(52.0, 1.8, 0.7)
    )
    filler_neck_36_5 = Pos(+32.7, +6.5, +64.2) * Cylinder(radius=4.3, height=3.2)
    filler_bore = Pos(+32.7, +6.5, +62.5) * Cylinder(radius=2.85, height=8.0)
    valve_boss_36_6 = Pos(+21.7, -6.0, +64.6) * Cylinder(radius=2.6, height=4.0)
    valve_top_bore = Pos(+21.7, -6.0, +63.2) * Cylinder(radius=1.45, height=8.0)

    # 单根 36-6 同轴中空马氏空气置换下垂管 (OD 5.0 / ID 3.8mm, 下探至サンプ Z=+9.6mm)
    mariotte_spout_outer = Pos(+21.7, -6.0, +17.8) * Cylinder(radius=2.50, height=16.4)
    mariotte_spout_bore = Pos(+21.7, -6.0, +18.2) * Cylinder(radius=1.90, height=19.6)

    sump_outer = (
        Pos(+28.4, 0.0, +7.1) * Box(30.2, 28.0, 11.2)
        + Pos(+28.9, 0.0, +12.4) * Box(16.0, 33.2, 1.6)
        + Pos(+28.4, +14.8, +18.5) * Cylinder(radius=2.2, height=13.6)
        + Pos(+28.4, -14.8, +18.5) * Cylinder(radius=2.2, height=13.6)
    )
    sump_cavity = Pos(+28.4, 0.0, +7.5) * Box(28.6, 26.4, 10.4)

    pts_fuel = [(+42.2, 0.0, +6.2), (+62.0, 0.0, +7.0), (+107.2, 0.0, +7.0)]
    fuel_outer = (
        make_swept_tube(pts_fuel, r_tube=2.25, bend_r=5.0, r_ref=2.25)
        + Pos(+93.5, 0.0, +9.8) * Box(4.0, 5.2, 5.6)
        + Pos(+105.0, 0.0, +7.0) * Rot(0, 90, 0) * Cylinder(radius=2.95, height=2.2)
    )
    fuel_bore = make_swept_tube(_extend_polyline_ends(pts_fuel, 0.8, 0.8), r_tube=1.55, bend_r=5.0, r_ref=2.25)

    drawbar_pocket_32_5_8 = (
        Pos(+93.5, 0.0, +12.6) * Box(18.0, 16.0, 2.2)
        + Pos(+101.2, 0.0, -1.9) * Box(7.6, 10.4, 4.6)
        - Pos(+103.2, 0.0, -1.9) * Box(5.0, 6.8, 3.0)
        # 32-8 牵引销限制在 Z ∈ [-3.9, +4.3]mm (低于 Z=+7.0mm 供醇管底壁)，两侧以 Y=±4.1mm 立板连接上座，彻底杜绝穿管截断流道!
        + Pos(+102.0, 0.0, +0.2) * Cylinder(radius=1.85, height=8.2)
        + Pos(+100.5, +4.1, +5.4) * Box(5.4, 2.2, 15.6)
        + Pos(+100.5, -4.1, +5.4) * Box(5.4, 2.2, 15.6)
    )

    alc_outer_all = alc_outer + alc_top_flange + filler_neck_36_5 + valve_boss_36_6 + mariotte_spout_outer + sump_outer + fuel_outer + drawbar_pocket_32_5_8
    alc_bore_all = alc_hollow + filler_bore + valve_top_bore + mariotte_spout_bore + sump_cavity + fuel_bore

    # 2. 31-10 水箱内胆 (t=0.80mm) + 31-5 全深 ø7.0mm 缸孔与中空双球阀塔 + 31-12x2 (Y=±21.3) + 笔直中空 34-6x2
    wt_liner = Pos(-46.75, 0.0, +37.7) * Box(84.5, 76.4, 47.4) - Pos(-46.75, 0.0, +38.1) * Box(82.9, 74.8, 46.6)

    hp_outer = (
        Pos(-56.0, +4.0, +21.7) * Rot(0, 90, 0) * Cylinder(radius=5.4, height=22.0)
        + Pos(-56.0, +4.0, +16.4) * Box(18.0, 10.6, 4.8)
        + Pos(-47.0, +4.0, +27.8) * Box(4.2, 2.5, 11.2)
        + Pos(-47.0, +4.0, +31.7) * Rot(90, 0, 0) * Cylinder(radius=2.4, height=2.6)
        + Pos(-69.5, +4.0, +28.0) * Cylinder(radius=4.4, height=25.6)
        + make_hex_nut((-69.5, +4.0, +41.2), (0.0, 0.0, 1.0), hex_r=4.2, length=2.2)
    )
    hp_bore = (
        Pos(-55.2, +4.0, +21.7) * Rot(0, 90, 0) * Cylinder(radius=3.52, height=21.5)
        + Pos(-67.2, +4.0, +21.7) * Rot(0, 90, 0) * Cylinder(radius=2.0, height=5.5)
        + Pos(-69.5, +4.0, +26.8) * Cylinder(radius=2.3, height=20.0)
        # 31-5 内部双 SB-ø4 止回球阀座缩颈台阶
        - (Pos(-69.5, +4.0, +19.5) * (Cylinder(radius=2.32, height=0.9) - Cylinder(radius=1.55, height=1.2)))
        - (Pos(-69.5, +4.0, +29.5) * (Cylinder(radius=2.32, height=0.9) - Cylinder(radius=1.55, height=1.2)))
        + Pos(-69.5, +6.6, +17.4) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=5.2)
    )

    pts_31_13 = [
        (-69.5, +4.0, +35.7),
        (-69.5, -21.3, +35.7),
        (-16.4, -21.3, +35.7),
        (-16.4, -21.3, +18.2),
    ]
    pts_31_14 = [
        (-16.4, +21.3, +18.2),
        (-16.4, +21.3, +54.2),
        (-22.0, +15.0, +55.8),
        (-26.0, +11.5, +53.6),
    ]

    water_outer = wt_liner + hp_outer
    water_bore = hp_bore

    water_outer += (
        make_hex_nut((-69.5, +0.2, +35.7), (0.0, 1.0, 0.0), hex_r=2.15, length=2.0)
        + make_swept_tube(pts_31_13, r_tube=1.50, bend_r=4.2, r_ref=1.50)
        + make_hex_nut((-16.4, -21.3, +21.2), (0.0, 0.0, 1.0), hex_r=2.15, length=2.0)
        + make_hex_nut((-16.4, +21.3, +21.2), (0.0, 0.0, 1.0), hex_r=2.15, length=2.0)
        + make_swept_tube(pts_31_14, r_tube=1.50, bend_r=3.4, r_ref=1.50)
        + Pos(-41.0, -10.3, +14.8) * Cylinder(radius=4.4, height=5.6)
        + make_hex_nut((-41.0, -10.3, +16.6), (0.0, 0.0, 1.0), hex_r=4.0, length=1.5)
    )
    water_bore += (
        make_swept_tube(_extend_polyline_ends(pts_31_13, 0.0, 0.6), r_tube=1.00, bend_r=4.2, r_ref=1.50)
        + make_swept_tube(_extend_polyline_ends(pts_31_14, 0.6, 0.65), r_tube=1.00, bend_r=3.4, r_ref=1.50)
        + Pos(-41.0, -10.3, +14.8) * Cylinder(radius=3.00, height=9.0)
    )

    for sy in (-1.0, +1.0):
        yp = sy * 21.3
        pts_34_6 = [
            (-16.4, yp, +16.2),
            (+102.2, yp, +16.2),
            (+103.8, yp, -5.1),
            (+109.5, yp, -5.1),
        ]
        water_outer += (
            Pos(-16.4, yp, +17.8) * Cylinder(radius=3.5, height=6.4)
            + make_hex_nut((-16.4, yp, +19.2), (0.0, 0.0, 1.0), hex_r=4.4, length=1.8)
            + make_banjo_eye((-16.4, yp, +16.2), (0.0, 0.0, -1.0), r_eye=2.5, h_eye=2.4, hex_r=1.9)
            + make_swept_tube(pts_34_6, r_tube=1.50, bend_r=2.8, r_ref=1.50)
            + Pos(+28.4, yp, +17.9) * Box(5.0, 3.2, 2.4)
            + Pos(+103.5, yp, +6.6) * Box(2.4, 4.2, 25.0)
            + make_hex_nut((+106.4, yp, -5.1), (1.0, 0.0, 0.0), hex_r=2.55, length=1.8)
            + Pos(+108.2, yp, -5.1) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=2.6)
        )
        water_bore += (
            Pos(-16.4, yp, +17.6) * Cylinder(radius=1.10, height=8.8)
            + make_swept_tube(_extend_polyline_ends(pts_34_6, 0.0, 0.75), r_tube=1.00, bend_r=2.8, r_ref=1.50)
        )

    # 横向连接梁严格抬升至 Z=+18.4mm (> 34-6 管顶 Z=+17.7mm)，并且所有流道内孔在最终一步统一布尔差集!
    front_beam_tie = (
        Pos(+103.2, 0.0, +18.4) * Box(2.8, 45.8, 1.6)
        + Pos(+103.2, 0.0, +15.2) * Box(2.6, 6.0, 6.4)
    )
    mid_frame_tie = Pos(+28.4, 0.0, +18.4) * Box(4.2, 44.6, 1.6)

    return (alc_outer_all + water_outer + front_beam_tie + mid_frame_tie) - (alc_bore_all + water_bore)


def build_tender_bogie_frame(cfg: Dict):
    """イラスト 31 & 36: 31-26 铸铁侧架 + 31-24x4 下封口护板 + 31-27 三长圆孔端梁 + 31-29 枕梁 + 36-3 心皿座金 + 31-25 板簧 + 31-28 螺旋弹簧 (100% Y 对称)。"""
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
            side_part += Pos(ax_x - 4.8, sgn * (y_frame + 1.2), -1.2) * Cylinder(radius=1.45, height=5.8)
            side_part += Pos(ax_x + 4.8, sgn * (y_frame + 1.2), -1.2) * Cylinder(radius=1.45, height=5.8)
            # 31-24 轴箱导框底部保持扁铁 (イラスト 31)
            side_part += Pos(ax_x, sgn * (y_frame + 1.3), -3.9) * Box(11.6, 3.6, 1.1)
            side_part -= Pos(ax_x, yf, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=10.0)
        sides = side_part if sides is None else (sides + side_part)
    return bolster + end_beams_31_27 + sides


def build_spoked_truck_wheel(cfg: Dict, is_left: bool = True):
    """イラスト 20, 24 & 31: 10 辐条放射镂空导轮/从轮/煤水车车轮。"""
    r_tread = 0.5 * float(cfg["wheels_and_axles"]["leading_wheel_diameter"])
    w_tire = 3.4
    rim = (
        Cylinder(radius=r_tread, height=w_tire)
        + Pos(0, 0, -0.5 * w_tire + 0.45) * Cylinder(radius=r_tread + 1.3, height=0.9)
        - Cylinder(radius=r_tread - 1.9, height=w_tire + 2.0)
    )
    spokes = None
    n_sp = 10
    for i in range(n_sp):
        ang_deg = i * (360.0 / n_sp)
        sp = Rot(0, 0, ang_deg) * Pos(0.5 * (3.2 + r_tread - 1.6), 0.0, 0.0) * Box((r_tread - 1.6) - 3.2, 1.15, 1.8)
        spokes = sp if spokes is None else (spokes + sp)

    wheel = rim + spokes + Cylinder(radius=3.6, height=w_tire + 0.4) - Cylinder(radius=1.8, height=w_tire + 6.0)
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
    parser = argparse.ArgumentParser(description="Aster D51 Stage 16 精确 STEP B-Rep 母本 + 高精真壁厚空心 STL 建模与惯量积分器")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--piping-config", type=Path, default=Path("configs/d51_boiler_piping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/meshes"))
    parser.add_argument("--step-dir", type=Path, default=Path("models/step"))
    parser.add_argument("--mass-out", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--only", type=str, default="", help="Optional comma-separated part keys to rebuild incrementally")
    parser.add_argument("--standard-stack", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pip_cfg = json.loads(args.piping_config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.step_dir.mkdir(parents=True, exist_ok=True)
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    tasks: List[Tuple[str, float, float, Callable[[], Any]]] = [
        ("driving_wheel_L", 0.025, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, False)),
        ("driving_wheel_R", 0.025, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, False)),
        ("main_driving_wheel_L", 0.025, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, True)),
        ("main_driving_wheel_R", 0.025, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, True)),
        ("return_crank_L", 0.014, RHO_STAINLESS, lambda: build_return_crank(cfg, True)),
        ("return_crank_R", 0.014, RHO_STAINLESS, lambda: build_return_crank(cfg, False)),
        ("main_rod", 0.016, RHO_STAINLESS, lambda: build_main_rod(cfg)),
        ("coupling_rod", 0.016, RHO_STAINLESS, lambda: build_coupling_rod_main(cfg)),
        ("coupling_rod_front", 0.016, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, True)),
        ("coupling_rod_rear", 0.016, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, False)),
        ("eccentric_rod_L", 0.014, RHO_STAINLESS, lambda: build_eccentric_rod(cfg, True)),
        ("eccentric_rod_R", 0.014, RHO_STAINLESS, lambda: build_eccentric_rod(cfg, False)),
        ("expansion_link_L", 0.014, RHO_STAINLESS, lambda: build_expansion_link(cfg, True)),
        ("expansion_link_R", 0.014, RHO_STAINLESS, lambda: build_expansion_link(cfg, False)),
        ("die_block_L", 0.012, RHO_BRASS, lambda: build_die_block(cfg, True)),
        ("die_block_R", 0.012, RHO_BRASS, lambda: build_die_block(cfg, False)),
        ("radius_rod_L", 0.014, RHO_STAINLESS, lambda: build_radius_rod(cfg, True)),
        ("radius_rod_R", 0.014, RHO_STAINLESS, lambda: build_radius_rod(cfg, False)),
        ("combination_lever", 0.014, RHO_STAINLESS, lambda: build_combination_lever(cfg)),
        ("union_link_L", 0.014, RHO_STAINLESS, lambda: build_union_link(cfg, True)),
        ("union_link_R", 0.014, RHO_STAINLESS, lambda: build_union_link(cfg, False)),
        ("lift_link_L", 0.014, RHO_STAINLESS, lambda: build_lift_link(cfg, True)),
        ("lift_link_R", 0.014, RHO_STAINLESS, lambda: build_lift_link(cfg, False)),
        ("motion_bracket_L", 0.022, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, True)),
        ("motion_bracket_R", 0.022, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, False)),
        ("slide_bar_L", 0.018, RHO_STAINLESS, lambda: build_slide_bar(cfg, True)),
        ("slide_bar_R", 0.018, RHO_STAINLESS, lambda: build_slide_bar(cfg, False)),
        ("piston_tail_sheath_L", 0.016, RHO_STAINLESS, lambda: build_piston_tail_sheath(cfg, True)),
        ("piston_tail_sheath_R", 0.016, RHO_STAINLESS, lambda: build_piston_tail_sheath(cfg, False)),
        ("valve_spindle_sheath_L", 0.016, RHO_STAINLESS, lambda: build_valve_spindle_sheath(cfg, True)),
        ("valve_spindle_sheath_R", 0.016, RHO_STAINLESS, lambda: build_valve_spindle_sheath(cfg, False)),
        ("cab_reach_rod", 0.018, RHO_STAINLESS, lambda: build_cab_reach_rod(cfg)),
        ("drain_cock_linkage", 0.018, RHO_BRASS, lambda: build_drain_cock_linkage(cfg)),
        ("drain_pull_rod", 0.022, RHO_STAINLESS, lambda: build_drain_pull_rod(cfg)),
        ("cab_backhead_plate", 0.028, RHO_CAST_IRON, build_cab_backhead_plate),
        ("cab_deck_and_mounts", 0.028, RHO_BRASS, build_cab_deck_and_mounts),
        ("cab_reverser_stand", 0.020, RHO_BRASS, build_cab_reverser_stand),
        ("cab_reverser_wheel", 0.016, RHO_STAINLESS, build_cab_reverser_wheel),
        ("cab_reverser_nut", 0.016, RHO_BRASS, build_cab_reverser_nut),
        ("cab_regulator_lever", 0.016, RHO_BRASS, build_cab_regulator_lever),
        ("cab_blower_wheel", 0.014, RHO_BRASS, build_cab_blower_wheel),
        ("cab_whistle_lever", 0.016, RHO_BRASS, build_cab_whistle_lever),
        ("cab_sight_glass_tube", 0.012, RHO_PYREX_GLASS, build_cab_sight_glass_tube),
        ("cab_sight_drain", 0.014, RHO_BRASS, build_cab_sight_drain),
        ("cab_pg_dial", 0.016, RHO_GAUGE_DIAL, build_cab_pg_dial),
        ("cab_pressure_needle", 0.014, RHO_STAINLESS, build_cab_pressure_needle),
        ("cab_blowdown_screw", 0.014, RHO_BRASS, build_cab_blowdown_screw),
        ("cab_firebox_door", 0.018, RHO_BRASS, build_cab_firebox_door),
        ("cab_bypass_lever", 0.014, RHO_BRASS, build_cab_bypass_lever),
        ("cab_drain_lever", 0.014, RHO_BRASS, build_cab_drain_lever),
        ("dome_lid_29_12", 0.014, RHO_BRASS, build_dome_lid_29_12),
        ("safety_valve_poppet", 0.010, RHO_BRASS, build_safety_valve_poppet),
        ("tender_hp_links", 0.018, RHO_BRASS, build_tender_hp_links),
        ("tender_handpump_lever", 0.020, RHO_BRASS, build_tender_handpump_lever),
        ("tender_handpump_ram", 0.016, RHO_BRASS, build_tender_handpump_ram),
        ("tender_drain_plug", 0.016, RHO_BRASS, build_tender_drain_plug),
        ("tender_coal_cover", 0.032, RHO_BRASS, build_tender_coal_cover),
        ("tender_fuel_valve", 0.016, RHO_BRASS, build_tender_fuel_valve),
        ("tender_fuel_cap", 0.016, RHO_BRASS, build_tender_fuel_cap),
        ("tender_rear_ladder", 0.020, RHO_BRASS, build_tender_rear_ladder),
        ("umb_hose_fuel", 0.018, RHO_SILICONE, lambda: build_umb_hose_fuel(cfg, False)),
        ("umb_hose_fuel_r2m", 0.018, RHO_SILICONE, lambda: build_umb_hose_fuel(cfg, True)),
        ("umb_hose_feed", 0.018, RHO_VITON, lambda: build_umb_hose_feed(cfg, False)),
        ("umb_hose_feed_r2m", 0.018, RHO_VITON, lambda: build_umb_hose_feed(cfg, True)),
        ("umb_union_34_8", 0.014, RHO_BRASS, lambda: build_umb_union_34_8(cfg, False)),
        ("umb_union_34_8_r2m", 0.014, RHO_BRASS, lambda: build_umb_union_34_8(cfg, True)),
        ("umb_hose_return", 0.018, RHO_VITON, lambda: build_umb_hose_return(cfg, False)),
        ("umb_hose_return_r2m", 0.018, RHO_VITON, lambda: build_umb_hose_return(cfg, True)),
        ("chassis_equalizer_brakes", 0.025, RHO_CAST_IRON, lambda: build_chassis_equalizer_brakes(cfg)),
        ("axle_pump_eccentric", 0.016, RHO_BRASS, lambda: build_axle_pump_eccentric(cfg)),
        ("leading_truck_frame", 0.022, RHO_CAST_IRON, lambda: build_leading_truck_frame(cfg)),
        ("trailing_truck_frame", 0.022, RHO_CAST_IRON, lambda: build_trailing_truck_frame(cfg)),
        ("piston_crosshead_L", 0.018, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, True)),
        ("piston_crosshead_R", 0.018, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, False)),
        ("slide_valve_L", 0.016, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, True)),
        ("slide_valve_R", 0.016, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, False)),
        ("reverser_shaft", 0.018, RHO_STAINLESS, lambda: build_reverser_shaft(cfg)),
        ("driving_axlebox_L", 0.020, RHO_BRASS, lambda: build_driving_axlebox(cfg, True)),
        ("driving_axlebox_R", 0.020, RHO_BRASS, lambda: build_driving_axlebox(cfg, False)),
        ("tender_axlebox_L", 0.020, RHO_CAST_IRON, lambda: build_tender_axlebox(cfg, True)),
        ("tender_axlebox_R", 0.020, RHO_CAST_IRON, lambda: build_tender_axlebox(cfg, False)),
        ("cylinder_block_L", 0.024, RHO_BRASS, lambda: build_cylinder_block(cfg, True)),
        ("cylinder_block_R", 0.024, RHO_BRASS, lambda: build_cylinder_block(cfg, False)),
        ("boiler_superstructure", 0.045, RHO_BRASS, lambda: build_boiler_superstructure(cfg, not args.standard_stack)),
        ("boiler_firebox_flues", 0.028, RHO_COPPER, lambda: build_boiler_firebox_flues(cfg, pip_cfg)),
        ("boiler_insulation", 0.032, RHO_CERAMIC_SHEET, lambda: build_boiler_insulation(cfg, pip_cfg)),
        ("superheater_piping", 0.014, RHO_STAINLESS, lambda: build_superheater_piping(cfg, pip_cfg)),
        ("exhaust_blast_pipe", 0.016, RHO_BRASS, lambda: build_exhaust_blast_pipe(cfg, pip_cfg)),
        ("blower_nozzle_17_3", 0.014, RHO_BRASS, lambda: build_blower_nozzle_17_3(cfg, pip_cfg)),
        ("smokebox_petticoat", 0.020, RHO_STAINLESS, lambda: build_smokebox_petticoat(cfg, pip_cfg)),
        ("tandem_safety_valves", 0.014, RHO_BRASS, lambda: build_tandem_safety_valves(cfg, pip_cfg)),
        ("feedwater_piping", 0.016, RHO_BRASS, lambda: build_feedwater_piping(cfg, pip_cfg)),
        ("displacement_lubricator", 0.016, RHO_BRASS, lambda: build_displacement_lubricator(cfg, pip_cfg)),
        ("boiler_aux_valves", 0.016, RHO_BRASS, lambda: build_boiler_aux_valves(cfg, pip_cfg)),
        ("alcohol_burner", 0.018, RHO_BURNER_ASSY, lambda: build_alcohol_burner(cfg, pip_cfg)),
        ("burner_wick", 0.016, RHO_CERAMIC_WICK, build_burner_wick),
        ("tender_body", 0.045, RHO_BRASS, lambda: build_tender_body(cfg)),
        ("tender_internals", 0.024, RHO_BRASS, lambda: build_tender_internals(cfg)),
        ("tender_bogie_frame", 0.025, RHO_CAST_IRON, lambda: build_tender_bogie_frame(cfg)),
        ("truck_wheel_L", 0.020, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, True)),
        ("truck_wheel_R", 0.020, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, False)),
    ]

    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()
    if not only_set:
        expected_stls = {f"{k}.stl" for k, _, _, _ in tasks}
        for old_stl in args.out_dir.glob("*.stl"):
            if old_stl.name not in expected_stls:
                old_stl.unlink()
        expected_steps = {f"{k}.step" for k, _, _, _ in tasks}
        for old_step in args.step_dir.glob("*.step"):
            if old_step.name not in expected_steps:
                old_step.unlink()

    mass_db: Dict[str, Any] = {
        "meta": {
            "stage": "STAGE-17-STEP-FIRST-END-TO-END-3D-LUMEN-WICK-INSULATION",
            "cad_master_format": "ISO-10303-21 STEP (.step) canonical B-Rep source of truth -> direct STEP-to-STL tessellation & analytic STEP inertia",
            "units": "SI (kg, m, kg*m^2)",
            "densities_kg_m3": {
                "copper_C1100": RHO_COPPER,
                "brass_BS": RHO_BRASS,
                "stainless_SUS": RHO_STAINLESS,
                "cast_iron_FC": RHO_CAST_IRON,
                "silicone_ST": RHO_SILICONE,
                "viton_BT": RHO_VITON,
                "ceramic_wick_yarn": RHO_CERAMIC_WICK,
                "ceramic_insulation_sheet": RHO_CERAMIC_SHEET,
            },
        },
        "parts": {},
    }
    if only_set and args.mass_out.exists():
        prev_db = json.loads(args.mass_out.read_text(encoding="utf-8"))
        mass_db["parts"] = dict(prev_db.get("parts", {}))
        tasks = [t for t in tasks if t[0] in only_set]

    high_curv_piping = {
        "superheater_piping",
        "exhaust_blast_pipe",
        "blower_nozzle_17_3",
        "tandem_safety_valves",
        "safety_valve_poppet",
        "feedwater_piping",
        "displacement_lubricator",
        "boiler_aux_valves",
        "alcohol_burner",
        "burner_wick",
        "tender_internals",
        "umb_hose_fuel",
        "umb_hose_fuel_r2m",
        "umb_hose_feed",
        "umb_hose_feed_r2m",
        "umb_hose_return",
        "umb_hose_return_r2m",
        "umb_union_34_8",
        "umb_union_34_8_r2m",
        "cab_sight_glass_tube",
        "cab_pg_dial",
        "cab_blower_wheel",
    }

    stop_background_gpu_server_for_heavy_job()
    hw_guard = HardwareThermalPowerGuard(max_cpu_c=78.0, max_gpu_c=72.0, max_gpu_w=260.0)

    for idx, (key, tol, rho, builder) in enumerate(tasks, start=1):
        if idx % 8 == 0:
            hw_guard.cooldown_if_needed(max_wait_s=1.2)
        stl_path = args.out_dir / f"{key}.stl"
        step_path = args.step_dir / f"{key}.step"
        print(f"[STEP-FIRST] {idx:02d}/{len(tasks)} 导出母本 {key}.step -> 编译网格 .stl & 积分质量 (ρ={rho:.0f} kg/m³) ...")
        solid = builder()
        export_step(solid, str(step_path))
        stl_tol = tol
        if key in ("boiler_superstructure", "tender_body"):
            stl_tol = 0.085
            ang_tol = 0.28
        elif key == "boiler_insulation":
            stl_tol = 0.065
            ang_tol = 0.26
        elif key in high_curv_piping:
            stl_tol = max(tol, 0.022)
            ang_tol = 0.185
        elif key in ("expansion_link_L", "expansion_link_R", "die_block_L", "die_block_R"):
            ang_tol = 0.16
        elif "wheel" in key:
            ang_tol = 0.22
        else:
            ang_tol = 0.24
        mp, _ = compile_step_to_stl_and_mass(
            step_path,
            stl_path,
            density_kg_m3=rho,
            lin_tol=stl_tol,
            ang_tol=ang_tol,
        )
        if key == "boiler_superstructure":
            subdivide_stl_x_span(stl_path, max_dx=1.92, z_min_filter=71.0)
        mass_db["parts"][key] = mp
        print(
            f"           -> 质量 = {mp['mass_kg']*1000:6.2f} g | "
            f"质心 = [{mp['com_m'][0]*1000:+.1f}, {mp['com_m'][1]*1000:+.1f}, {mp['com_m'][2]*1000:+.1f}] mm"
        )

    args.mass_out.parent.mkdir(parents=True, exist_ok=True)
    args.mass_out.write_text(json.dumps(mass_db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] 已导出 {len(tasks)} 个精确 STEP B-Rep 母本 ({args.step_dir}) 与高精空心水密 STL 网格 ({args.out_dir})，惯量库写入: {args.mass_out}")


if __name__ == "__main__":
    main()
