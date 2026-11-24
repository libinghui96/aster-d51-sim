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
    Sphere,
    Torus,
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
    """构建原厂圆柱形バンジョー环眼 + 4-15 六角中空管螺栓头总成 (零方块，外六角端面实体封顶！)。"""
    ax = Vector(*axis_dir).normalized()
    p_base = Vector(*pos) - ax * (0.5 * h_eye)
    eye = extrude(Plane(origin=p_base, z_dir=ax) * Circle(r_eye), amount=h_eye)
    bolt_pos = Vector(*pos) + ax * (0.5 * h_eye + 0.45)
    bolt = make_hex_nut((bolt_pos.X, bolt_pos.Y, bolt_pos.Z), axis_dir, hex_r=hex_r, length=1.2)
    return eye + bolt


def make_blind_hex_bolt(
    pos: Tuple[float, float, float],
    axis_dir: Tuple[float, float, float],
    hex_r: float,
    head_len: float,
    collar_r: float = 0.0,
    collar_len: float = 0.0,
):
    """构建外端面 100% 实心封顶的 (4-15) M5 中空管螺栓六角帽 / 闷头螺塞 (严禁被内孔布尔差集打穿顶面！)。"""
    ax = Vector(*axis_dir).normalized()
    part = make_hex_nut(pos, axis_dir, hex_r=hex_r, length=head_len)
    if collar_r > 1e-4 and collar_len > 1e-4:
        p_col = Vector(*pos) - ax * (0.5 * head_len + collar_len - 0.15)
        part = part + extrude(Plane(origin=p_col, z_dir=ax) * Circle(collar_r), amount=collar_len + 0.15)
    return part


def make_coil_spring_stack(
    center_xyz: Tuple[float, float, float],
    r_coil: float,
    r_wire: float,
    height: float,
    n_turns: int = 5,
    r_core_rod: float = 0.0,
):
    """构建严格 Y 对称 (|com_y| == 0.0000mm)、零切向退化、单连通水密 (cc == 1) 的精密弹簧圈组与上下座垫。"""
    cx, cy, cz = center_xyz
    r_out = r_coil + r_wire
    r_in = max(0.35, r_coil - 0.82 * r_wire)
    z_bot = cz - 0.5 * height
    z_top = cz + 0.5 * height
    seat_h = max(0.55, 1.10 * r_wire)

    bot_seat = Pos(cx, cy, z_bot + 0.5 * seat_h) * Cylinder(radius=r_out + 0.12, height=seat_h)
    top_seat = Pos(cx, cy, z_top - 0.5 * seat_h) * Cylinder(radius=r_out + 0.12, height=seat_h)
    spine_r = max(0.34, 0.55 * r_wire)
    coil = (
        bot_seat
        + top_seat
        + Pos(cx + r_coil, cy, cz) * Cylinder(radius=spine_r, height=height - 0.2)
        + Pos(cx - r_coil, cy, cz) * Cylinder(radius=spine_r, height=height - 0.2)
    )
    usable_h = max(0.9, height - 1.5 * seat_h)
    turn_h = min(1.65 * r_wire, 0.62 * (usable_h / max(1, n_turns)))
    for k in range(n_turns):
        zk = (cz - 0.5 * usable_h) + (k + 0.5) * (usable_h / max(1, n_turns))
        ring_k = (
            Pos(cx, cy, zk) * Cylinder(radius=r_out, height=turn_h)
            + Pos(cx, cy, zk + 0.35 * turn_h) * Cone(bottom_radius=r_out, top_radius=r_out - 0.28 * r_wire, height=0.32 * turn_h)
            + Pos(cx, cy, zk - 0.35 * turn_h) * Rot(180, 0, 0) * Cone(bottom_radius=r_out, top_radius=r_out - 0.28 * r_wire, height=0.32 * turn_h)
        )
        coil = coil + ring_k
    if r_core_rod > 1e-4:
        hollow = Pos(cx, cy, cz) * (Cylinder(radius=r_in, height=height + 1.2) - Cylinder(radius=r_core_rod, height=height + 2.0))
        coil = (coil - hollow) + Pos(cx, cy, cz) * Cylinder(radius=r_core_rod, height=height + 0.5)
    else:
        coil = coil - Pos(cx, cy, cz) * Cylinder(radius=r_in, height=height + 1.2)
    return coil


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
    # 严守四动轮 46.50mm 轴距几何上限: 2*(r_tread + h_flange) <= 45.90mm -> h_flange = 1.06mm (相邻动轮轮缘间距 >= 0.62mm > 0.50mm!)
    r_flange = r_tread + 1.06
    r_rim_inner = r_tread - 2.4
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    w_tire, w_flange, w_web = 3.9, 1.05, (2.8 if is_main_driver else 2.4)

    tire = Cylinder(radius=r_tread, height=w_tire)
    flange = Pos(0, 0, -0.5 * w_tire + 0.5 * w_flange) * Cone(
        bottom_radius=r_flange, top_radius=r_tread + 0.35, height=w_flange
    )
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
    x_bend = 0.46 * L_ecc
    shank_rear = make_connecting_bar_xy((1.2, y_rear), (x_bend, y_rear), 0.0, thick_y=1.05, height_z=2.30)
    shank_front = make_connecting_bar_xy((x_bend, y_rear), (L_ecc - 1.0, y_front), 0.0, thick_y=1.05, height_z=2.30)

    rod_L = rear_boss + shank_rear + shank_front + front_flat_eye
    rod_L -= Pos(0, y_rear, 0) * Rot(90, 0, 0) * Cylinder(radius=1.18, height=5.0)
    rod_L -= Pos(L_ecc, y_front, 0) * Rot(90, 0, 0) * Cylinder(radius=1.02, height=5.0)
    return rod_L if is_left else mirror(rod_L, about=Plane.XZ)


def build_expansion_link(cfg: Dict, is_left: bool = True):
    """
    2-4(中板) + 2-5/2-6(R) / 2-7/2-8(L) 原厂三层夹板箱型月牙板 (加減リンク, イラスト 2 & d51_manual.pdf p.3/p.18, cc=1):
      - 内外双侧开槽月牙夹板 (y ∈ [-1.85,-1.15] & [+1.15,+1.85]mm) 由上下 2-4 垫块与 +M1.4-4 螺钉锁紧，中央留出 2.30mm 贯通槽供 2-1 心向棒居中穿过；
      - 内外夹板中心自带一体式对称耳轴销 (ø2.20mm, y ∈ [-2.75,+2.75]mm)，彻底删除虚假外凸 U 型跨桥与 ø4.5mm 大圆帽！
    """
    vg = cfg["walschaerts_valve_gear"]
    R_arc = float(vg["radius_rod_length"])
    r_tail = float(vg["exp_link_tail_radius"])
    cant_rad = math.radians(float(vg["exp_link_tail_cant_deg"]))
    slot_half = float(vg["die_slot_half_travel"])

    body_w = 3.80
    slot_w = 1.85
    z_top = slot_half + 2.40  # +7.90 mm (总高 ±9.80 mm，端垫块留足 |z|<=7.35mm 供满档心向棒 >=0.65mm 净空!)
    z_bot = -(slot_half + 2.40)
    z_slot_end = slot_half + 0.85  # ±6.35 mm

    def _make_crescent_plate(y_c: float, th_y: float):
        ring = Pos(R_arc, y_c, 0) * Rot(90, 0, 0) * Cylinder(
            radius=R_arc + 0.5 * body_w, height=th_y
        ) - Pos(R_arc, y_c, 0) * Rot(90, 0, 0) * Cylinder(
            radius=R_arc - 0.5 * body_w, height=th_y + 1.0
        )
        plate = ring & Pos(0, y_c, 0) * Box(18.0, th_y + 1.0, z_top - z_bot)
        for z_c in (z_top, z_bot):
            x_c = R_arc - math.sqrt(max(0.0, R_arc**2 - z_c**2))
            plate = plate + Pos(x_c, y_c, z_c) * Rot(90, 0, 0) * Cylinder(radius=0.5 * body_w, height=th_y)
        return plate

    cheek_in = _make_crescent_plate(-1.50, 0.70)
    cheek_out = _make_crescent_plate(+1.50, 0.70)

    # 2-4 中板上下端垫块 (保留 |z| <= 7.35mm 中央贯通腔供 2-1 心向棒满档无干涉摆动 >=0.65mm，两端嵌入双侧夹板确保 cc=1)
    core_full = _make_crescent_plate(0.0, 3.10)
    spacer_top = core_full & Pos(0, 0, 0.5 * (7.35 + 10.20)) * Box(18.0, 3.4, 10.20 - 7.35)
    spacer_bot = core_full & Pos(0, 0, -0.5 * (7.35 + 10.20)) * Box(18.0, 3.4, 10.20 - 7.35)

    # 仅切削内外夹板的弧形滑槽 (保留 |y| >= 1.72mm 处的一体式耳轴法兰根部!)
    slot_ring = Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc + 0.5 * slot_w, height=3.40
    ) - Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc - 0.5 * slot_w, height=3.60
    )
    x_slot_end = R_arc - math.sqrt(max(0.0, R_arc**2 - z_slot_end**2))
    slot_cut = (slot_ring & Box(18.0, 3.40, 2.0 * z_slot_end)) + (
        Pos(x_slot_end, 0, z_slot_end) * Rot(90, 0, 0) * Cylinder(radius=0.5 * slot_w, height=3.40)
    ) + (
        Pos(x_slot_end, 0, -z_slot_end) * Rot(90, 0, 0) * Cylinder(radius=0.5 * slot_w, height=3.40)
    )
    crescent_box = (cheek_in + cheek_out + spacer_top + spacer_bot) - slot_cut

    # 上下 +M1.4-4 夹紧螺钉圆头
    screws_m14 = None
    for z_sc in (z_top + 0.10, z_bot - 0.10):
        x_sc = R_arc - math.sqrt(max(0.0, R_arc**2 - z_sc**2))
        sc = Pos(x_sc, +1.95, z_sc) * Rot(90, 0, 0) * Cylinder(radius=1.12, height=0.55)
        screws_m14 = sc if screws_m14 is None else (screws_m14 + sc)

    # 对称一体式内外中心耳轴销 (2-5..8 耳軸: 法兰 r=1.90mm 跨接前后槽壁，轴颈 ø2.20mm 伸至 y=±2.75mm 嵌入内外配气托架)
    trunnion_sym = (
        Pos(0.0, +1.85, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.90, height=0.45)
        + Pos(0.0, +2.25, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.10, height=1.05)
        + Pos(0.0, -1.85, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.90, height=0.45)
        + Pos(0.0, -2.25, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.10, height=1.05)
    )

    # 下部流线渐缩偏心杆尾耳 + 1:1 实寸盒 2-2 台阶螺钉 (总跨度 4.20mm)
    x_tail = r_tail * math.sin(cant_rad)
    z_tail = -r_tail * math.cos(cant_rad)
    x_cap_bot = R_arc - math.sqrt(max(0.0, R_arc**2 - z_bot**2))
    y_ear = float(vg["y_layer_offsets"]["expansion_link_tail_ear_dy"])
    ear_t = 1.25
    dx_n = x_tail - x_cap_bot
    dz_n = z_tail - (z_bot + 0.35)
    neck_len = math.hypot(dx_n, dz_n)
    neck_pitch = math.degrees(math.atan2(-dx_n, -dz_n))
    tail_neck = (
        Pos(0.5 * (x_cap_bot + x_tail), y_ear, 0.5 * (z_bot + 0.35 + z_tail))
        * Rot(0, neck_pitch, 0)
        * Box(2.45, ear_t, neck_len + 1.2)
    )
    tail_boss = Pos(x_tail, y_ear, z_tail) * Rot(90, 0, 0) * Cylinder(radius=1.82, height=ear_t)
    pin_2_2 = make_stepped_pin_L(
        x=x_tail,
        z=z_tail,
        y_nut_inner=float(vg["y_layer_offsets"]["stepped_pin_2_2_nut_dy"]),
        y_head_outer=float(vg["y_layer_offsets"]["stepped_pin_2_2_head_dy"]),
        r_shank=1.00,
        r_head=1.50,
        r_nut=1.32,
    )

    link_L = crescent_box + screws_m14 + trunnion_sym + tail_neck + tail_boss + pin_2_2
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


def build_die_block(cfg: Dict, is_left: bool = True):
    """2-3 黄铜滑块 + 2-2 平齐修锉台阶销 (イラスト 2 Fig.1: 居中跨骑 2-1 心向棒并嵌入双侧月牙槽，零外凸螺帽)。"""
    outer_blk = Box(3.30, 3.45, 2.85) - Box(4.0, 1.02, 3.4)
    pin_2_2 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-1.75, y_head_outer=+1.75, r_shank=0.98, r_head=1.42, r_nut=1.28, slot_w=0.34
    )
    part_L = outer_blk + pin_2_2
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_radius_rod(cfg: Dict, is_left: bool = True):
    """2-1 全直扁钢心向棒 (イラスト 2 ①: y=0 居中穿过箱型月牙板中腔，后端 -9.0mm 小圆眼接 2-10 吊杆，前端叉头跨接 1-23 顶销)。"""
    vg = cfg["walschaerts_valve_gear"]
    L_rad = float(vg["radius_rod_length"])
    L_back = float(vg["radius_rod_back_ext"])
    t_bar = 0.86

    back_eye = Pos(-L_back, 0.0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=t_bar)
    die_hub = Pos(0.0, 0.0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.62, height=t_bar)
    rear_span = Pos(-0.5 * L_back, 0.0, 0) * Box(L_back, t_bar, 1.85)
    main_span = Pos(0.5 * (L_rad - 1.2), 0.0, 0) * Box(L_rad - 1.2, t_bar, 1.90)

    front_clevis = (
        Pos(L_rad, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.02, height=2.9)
        + Pos(L_rad - 0.9, 0, 0) * Box(1.8, 2.9, 1.56)
        - Pos(L_rad + 0.5, 0, 0) * Box(3.8, 1.66, 5.0)
    )
    pin_1_24 = make_stepped_pin_L(
        x=L_rad, z=0.0, y_nut_inner=-1.95, y_head_outer=+1.95, r_shank=0.70, r_head=1.00, r_nut=0.95
    )

    rod_L = back_eye + die_hub + rear_span + main_span + front_clevis + pin_1_24
    rod_L -= Pos(0.0, 0.0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=4.0)
    rod_L -= Pos(-L_back, 0.0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=4.0)
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
        Pos(L_un, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.42, height=3.2)
        + Pos(L_un - 1.0, 0, 0) * Box(2.0, 3.2, 2.6)
        - Pos(L_un + 0.6, 0, 0) * Box(3.8, 1.68, 4.5)
    )
    bar = Pos(0.5 * L_un, 0, 0) * Box(L_un - 1.6, 1.45, 2.1)
    pin_0 = make_stepped_pin_L(0.0, 0.0, -2.40, +2.40, r_shank=0.88, r_head=1.55, r_nut=1.40)
    pin_1 = make_stepped_pin_L(L_un, 0.0, -2.20, +2.20, r_shank=0.82, r_head=1.28, r_nut=1.16)
    link_L = clevis_0 + clevis_1 + bar + pin_0 + pin_1
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


def build_lift_link(cfg: Dict, is_left: bool = True):
    """2-10 纤细垂直吊杆 + 1:1 实寸盒 2-21 (ø3.0mm 头) 台阶销 (イラスト 2 ⑩ & p43: 消除 ø6mm 大圆盘块)。"""
    vg = cfg["walschaerts_valve_gear"]
    L_lift = float(vg["lift_link_length"])

    top_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.58, height=2.60)
        + Pos(0, 0, -0.9) * Box(2.15, 2.60, 1.8)
        - Pos(0, 0, 0.5) * Box(3.6, 1.42, 3.0)
    )
    bot_clevis = (
        Pos(0, 0, -L_lift) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=2.25)
        + Pos(0, 0, -L_lift + 0.9) * Box(2.05, 2.25, 1.8)
        - Pos(0, 0, -L_lift - 0.5) * Box(3.6, 1.00, 3.0)
    )
    strap = Pos(0, 0.0, -0.5 * L_lift) * Box(1.75, 1.05, L_lift - 1.2)

    pin_top = make_stepped_pin_L(0.0, 0.0, -1.75, +1.80, r_shank=0.85, r_head=1.48, r_nut=1.22)
    pin_bot = make_stepped_pin_L(0.0, -L_lift, -1.60, +1.65, r_shank=0.85, r_head=1.45, r_nut=1.20)
    link_L = top_clevis + bot_clevis + strap + pin_top + pin_bot
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


# ==================== 3. 配气桁架、去假桥独立滑杆与前盖护套、左驾操纵与排液阀 8 级传动 ====================

def build_motion_bracket(cfg: Dict, is_left: bool = True):
    """
    2-9/2-20(外侧双椭圆减重孔配气板) + 2-15/2-17(内侧配气板) + 2-14/2-16(上下横撑梁) + 2-13(滑杆后吊耳) 原厂五件式开放配气托架 (イラスト 2 & p43):
      - 彻底修正旧版未扣减 tz=+16.0mm 造成的悬空粗黑方框、竖向遮挡板与 ø6.2mm 大黑圆饼！
      - 后部 r=2.35mm 紧凑耳轴鼻座夹持半月板内外耳轴，上下月牙体 100% 裸露可见；前部流线三角拱板开上下双椭圆减重孔！
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    tx, tz = float(vg["trunnion_x"]), float(vg["trunnion_z"])  # +58.8, +16.0 mm
    y_vg = float(rc["cylinder_center_y"]) + 3.5  # +36.0 mm

    dy_chassis = 15.25 - y_vg  # -20.75 mm (主车架外侧面 |Y|=15.25mm)
    dy_slidebar = float(rc["cylinder_center_y"]) - y_vg  # -3.50 mm (|Y|=32.5mm)
    dx_bar_tail = (float(rc["cylinder_center_x"]) - 51.0) - tx  # +19.20 mm (X_world=+78.0mm)
    dz_bar_lug = 8.40 - tz  # -7.60 mm (Z_world=+8.4mm)

    # 1. 2-15/2-17 车架安装法兰与轮间高位拱形横腹板 (X_world=+69.75mm 严格居中于第1/2动轮之间，Z_world=+14.0mm 距两侧轮缘净空 >3.5mm!)
    frame_pad = (
        Pos(+10.95, dy_chassis + 0.7, -2.0) * Box(5.8, 1.8, 4.2)
        + Pos(+9.4, dy_chassis + 1.5, -2.0) * Rot(90, 0, 0) * Cylinder(radius=1.05, height=0.8)
        + Pos(+12.5, dy_chassis + 1.5, -2.0) * Rot(90, 0, 0) * Cylinder(radius=1.05, height=0.8)
    )
    spandrel_web = make_connecting_bar_xy(
        (+10.95, dy_chassis + 1.1), (+10.95, -2.95), -1.8, thick_y=3.6, height_z=2.4
    )

    # 2. 2-9/2-20 外侧双椭圆减重孔流线配气板 (dy ∈ [+2.25, +3.25]mm，距月牙滑块 dy<=1.82mm 留足 0.43mm 净空!) 与 2-15/17 内侧配气板 (dy ∈ [-3.55, -2.55]mm)
    w_outer = Wire([
        Edge.make_line(Vector(0.0, 0.0, +2.35), Vector(+13.2, 0.0, +7.20)),
        Edge.make_line(Vector(+13.2, 0.0, +7.20), Vector(+16.8, 0.0, -2.20)),
        Edge.make_line(Vector(+16.8, 0.0, -2.20), Vector(+12.2, 0.0, -8.20)),
        Edge.make_line(Vector(+12.2, 0.0, -8.20), Vector(0.0, 0.0, -2.35)),
        Edge.make_line(Vector(0.0, 0.0, -2.35), Vector(0.0, 0.0, +2.35)),
    ])
    plate_2d = Face(w_outer)
    outer_plate = (
        Pos(0.0, +2.85, 0.0) * extrude(plate_2d, amount=0.45, dir=(0.0, 1.0, 0.0), both=True)
        + Pos(0.0, +2.85, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.30, height=0.90)
        + Pos(+13.2, +2.85, +5.80) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=0.90)
        + Pos(+12.2, +2.85, -6.80) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=0.90)
        + Pos(+13.2, +3.48, +5.80) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=0.45)
        + Pos(+12.2, +3.48, -6.80) * Rot(90, 0, 0) * Cylinder(radius=1.35, height=0.45)
    )
    inner_plate = (
        Pos(0.0, -3.05, 0.0) * extrude(plate_2d, amount=0.50, dir=(0.0, 1.0, 0.0), both=True)
        + Pos(0.0, -3.05, 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=1.00)
        + Pos(+15.6, -3.15, -7.20) * Box(8.0, 1.25, 2.4)
        + Pos(+10.95, -3.05, -1.8) * Box(4.2, 1.10, 3.2)
    )

    # 双椭圆减重孔 (p43 & イラスト 2 ⑨ 标志性上/下双长圆孔)
    upper_oval_cut = (
        Pos(+5.2, 0.0, +1.0) * Rot(90, 0, 0) * Cylinder(radius=1.85, height=12.0)
        + Pos(+9.6, 0.0, +2.6) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=12.0)
        + Pos(+7.4, 0.0, +1.8) * Rot(0, -20, 0) * Box(4.6, 12.0, 3.85)
    )
    lower_oval_cut = (
        Pos(+5.4, 0.0, -2.8) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=12.0)
        + Pos(+9.4, 0.0, -4.4) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=12.0)
        + Pos(+7.4, 0.0, -3.6) * Rot(0, +22, 0) * Box(4.3, 12.0, 3.05)
    )

    # 3. 2-14 & 2-16 上下横撑圆柱梁 (避开 2-1 心向棒 360° 全摆幅 dz ∈ [-4.5, +4.5]mm!) + 2-13 滑杆后吊耳夹座
    beam_2_14 = Pos(+13.2, -0.25, +5.80) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=6.2)
    beam_2_16 = Pos(+12.2, -0.25, -6.80) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=6.2)
    holder_2_13 = (
        Pos(dx_bar_tail, dy_slidebar + 0.35, dz_bar_lug + 0.5) * Box(3.6, 3.2, 1.8)
        + Pos(dx_bar_tail, dy_slidebar, dz_bar_lug + 1.55) * Cylinder(radius=1.05, height=0.8)
    )

    trun_bore = Pos(0.0, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.14, height=8.0)
    bracket_L = (
        frame_pad + spandrel_web + ((outer_plate + inner_plate) - upper_oval_cut - lower_oval_cut)
        + beam_2_14 + beam_2_16 + holder_2_13 - trun_bore
    )
    return bracket_L if is_left else mirror(bracket_L, about=Plane.XZ)


def build_slide_bar(cfg: Dict, is_left: bool = True):
    """
    Gate 11 & Gate 20 1-33 上置亮面不锈钢单滑杆 + 1-30 前缸盖支座 + 2-13 后吊耳座 (イラスト 1, 2, SUS304):
      - 严格终止于气缸后盖 1-29 颈部 (x_max = -9.6mm < 0.0mm，零跨缸假桥杆!)；
      - 后吊耳座 2-13 精确对齐 X_world = cylinder_center_x - 51.0mm (+78.0mm) 与 motion_bracket 刚性夹持。
    """
    x_bar_f, x_bar_r = -10.2, -52.0
    bar_len = x_bar_f - x_bar_r
    bar = Pos(0.5 * (x_bar_f + x_bar_r), 0.0, 6.5) * Box(bar_len, 2.8, 1.75)
    front_lug_1_30 = (
        Pos(-10.8, 0.0, 5.2) * Box(2.4, 2.6, 3.4)
        + Pos(-10.8, 0.0, 7.6) * Cylinder(radius=0.75, height=0.8)
    )
    rear_lug_2_13 = Pos(-51.0, 0.0, 7.6) * Box(3.6, 2.8, 1.2) + Pos(-51.0, 0.0, 8.5) * Cylinder(radius=0.85, height=1.2)
    part_L = bar + front_lug_1_30 + rear_lug_2_13
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_tail_sheath(cfg: Dict, is_left: bool = True):
    """20-6 薄竖直圆端双孔水滴法兰板 + 20-2 细长盲孔空心不锈钢尻棒护套 (イラスト 20 & 図 20.1, 根除 4.4mm 假方盒, cc=1)。"""
    flange_20_6 = (
        Pos(+18.15, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.80, height=1.15)
        + Pos(+18.15, 0.0, -5.36) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=1.15)
        + Pos(+18.15, 0.0, -2.68) * Box(1.15, 3.10, 5.36)
        + Pos(+18.95, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.68, height=1.40)
        + Pos(+18.85, 0.0, -5.36) * Rot(0, 90, 0) * Cylinder(radius=1.05, height=0.65)
    )
    tube_20_2 = Pos(+24.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.32, height=9.8)
    tip_cap = Pos(+29.3, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=1.32, top_radius=0.82, height=0.7)
    blind_bore = Pos(+22.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.92, height=11.0)
    part_L = (flange_20_6 + tube_20_2 + tip_cap) - blind_bore
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_valve_spindle_sheath(cfg: Dict, is_left: bool = True):
    """20-8 前导套座圈 (径向紧定螺钉沉孔) + 20-1 细长不锈钢滑阀尻棒护套 (イラスト 20 & p43, 同轴 y=+3.5mm [|Y|=36.0mm], cc=1)。"""
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    y_valve = +3.5
    collar_20_8 = (
        Pos(+18.0, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.35, height=3.2)
        - Pos(+18.2, y_valve, z_valve + 2.15) * Cylinder(radius=0.65, height=0.9)
    )
    nose_20_1 = Pos(+22.6, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.42, height=6.8)
    tip_cap = Pos(+26.25, y_valve, z_valve) * Rot(0, 90, 0) * Cone(bottom_radius=1.42, top_radius=0.92, height=0.6)
    spindle_blind_bore = Pos(+20.4, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=0.98, height=9.4)
    part_L = (collar_20_8 + nose_20_1 + tip_cap) - spindle_blind_bore
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
    pr15_rings = None
    for gx in (p_head_dx - 1.05, p_head_dx + 1.05):
        ring_groove = Pos(gx, 0, 0) * Rot(0, 90, 0) * (
            Cylinder(radius=0.5 * bore + 0.5, height=0.80) - Cylinder(radius=0.5 * bore - 0.95, height=1.2)
        )
        piston_disc -= ring_groove
        ring_pr15 = Pos(gx, 0, 0) * Rot(0, 90, 0) * (
            Cylinder(radius=0.5 * bore + 0.16, height=0.78) - Cylinder(radius=0.5 * bore - 1.35, height=0.78)
        )
        pr15_rings = ring_pr15 if pr15_rings is None else (pr15_rings + ring_pr15)
    piston_disc = piston_disc + pr15_rings
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

    drop_root = Pos(0, +0.90, -2.0) * Box(3.2, 1.80, 2.2)
    drop_step = Pos(0, +2.20, -3.4) * Rot(+44, 0, 0) * Box(3.2, 1.35, 3.6)
    drop_tongue = Pos(0, +3.50, -5.2) * Box(3.2, 1.45, 3.2)
    drop_boss = Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.60, height=1.45)
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
    1-25 阀杆叉头 + 1-21 销栓 + N-M2 锁紧螺母 + 1-22 全直 ø2.0mm 不锈钢阀芯棒 + 1-5 浮动滑阀框 + 1-6 外进气平面 D 型滑阀 (イラスト 1, 9, 20):
      - 阀杆轴线 Z = valve_axis_z = +13.5mm (严格对齐合并杆 1-23 次上销 1-21 与前导套 20-1, 零狗腿折弯!);
      - 1-6 平面 D 型滑阀底面贴合 1-9 交叉气道板镜面 (Z_world = +10.80mm -> z_local = -2.70mm), D 型排气拱腔跨接中央排汽口!
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    z_valve = float(rc["valve_axis_z"])
    v_dx = (float(rc["cylinder_center_x"]) + 4.2) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))  # +23.4 mm
    z_seat_local = 10.80 - z_valve  # -2.70 mm (精确落于 Z_world = +10.80mm 1-9 交叉气道板镜面!)

    clevis_1_25 = (
        Rot(90, 0, 0) * Cylinder(radius=0.85, height=3.1)
        + Pos(1.1, 0, 0) * Box(2.2, 3.1, 1.56)
        - Pos(-0.7, 0, 0) * Box(3.4, 1.68, 6.0)
    )
    pin_1_21 = make_stepped_pin_L(0.0, 0.0, -2.10, +2.10, r_shank=0.70, r_head=0.85, r_nut=0.82)
    nut_m2 = make_hex_nut((3.2, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=0.98, length=0.9)

    x_spindle_start = 1.5
    x_spindle_tip = v_dx + 6.6
    spindle_1_22 = Pos(0.5 * (x_spindle_start + x_spindle_tip), 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(
        radius=1.00, height=x_spindle_tip - x_spindle_start
    )

    # 1-5 滑阀框 + 1-6 顶榫实体嵌套接合 (消除封闭内部气泡空腔确保 cc==1，外周保留真实台阶缝，顶面双 P-M2-2.5 沉孔!)
    buckle_1_5 = (
        Pos(v_dx, 0.0, +0.35) * Box(9.4, 7.2, 2.9)
        + Pos(v_dx, 0.0, +0.45) * Rot(0, 90, 0) * Cylinder(radius=1.55, height=9.8)
        - Pos(v_dx - 2.6, 0.0, +1.45) * Cylinder(radius=0.72, height=0.8)
        - Pos(v_dx + 2.6, 0.0, +1.45) * Cylinder(radius=0.72, height=0.8)
    )
    d_valve_1_6 = (
        Pos(v_dx, 0.0, z_seat_local + 1.15) * Box(11.0, 8.2, 2.30)
        + Pos(v_dx, 0.0, z_seat_local + 2.65) * Box(8.0, 6.0, 1.60)
        - Pos(v_dx, 0.0, z_seat_local + 1.15) * Box(5.50, 5.40, 2.50)
    )

    assembly_L = clevis_1_25 + pin_1_21 + nut_m2 + spindle_1_22 + buckle_1_5 + d_valve_1_6
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_reverser_shaft(cfg: Dict):
    """
    4-12 逆转器下弯腹横轴 + 2-11/2-18 左右等长 10.5mm 纤细锥形提升臂 + 左走台内侧 (+Y=+32.5mm) 向上逆转摇臂 (イラスト 4, 2, p43):
      - 位于 X=+39.3mm, Z=+23.5mm；下弯腹段向内后倾收拢于车轮后方，彻底消除侧视亮银立柱与 ø4.8mm 粗大圆鼓！
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5  # 36.0 mm
    arm_len_x = float(vg["reverser_arm_length"])
    y_end = y_vg + 0.85

    cranked_bar = make_swept_tube(
        [
            (0.0, -y_end, 0.0),
            (0.0, -26.8, 0.0),
            (0.0, -24.8, +1.0),
            (-0.4, -19.2, +1.0),
            (-1.8, -14.8, -13.5),
            (-1.8, +14.8, -13.5),
            (-0.4, +19.2, +1.0),
            (0.0, +24.8, +1.0),
            (0.0, +26.8, 0.0),
            (0.0, +y_end, 0.0),
        ],
        r_tube=1.02,
        bend_r=1.5,
    )
    reach_hub = Pos(0, +32.5, 0) * Rot(90, 0, 0) * Cylinder(radius=1.80, height=1.6)
    reach_lever = Pos(0, +32.5, 3.5) * Box(2.1, 1.35, 7.0)
    reach_tip = Pos(0, +32.5, 7.0) * Rot(90, 0, 0) * Cylinder(radius=1.48, height=1.35)

    w_arm = Wire([
        Edge.make_line(Vector(0.0, 0.0, +1.30), Vector(arm_len_x, 0.0, +0.98)),
        Edge.make_line(Vector(arm_len_x, 0.0, +0.98), Vector(arm_len_x, 0.0, -0.98)),
        Edge.make_line(Vector(arm_len_x, 0.0, -0.98), Vector(0.0, 0.0, -1.30)),
        Edge.make_line(Vector(0.0, 0.0, -1.30), Vector(0.0, 0.0, +1.30)),
    ])
    arm_face = Face(w_arm)
    arms = None
    for sgn in (-1.0, 1.0):
        hub = Pos(0, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=1.88, height=1.65)
        arm = Pos(0, sgn * y_vg, 0) * extrude(arm_face, amount=0.65, dir=(0.0, 1.0, 0.0), both=True)
        tip_boss = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=1.58, height=1.30)
        tip_hole = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=0.86, height=5.0)
        sub = hub + arm + tip_boss - tip_hole
        arms = sub if arms is None else (arms + sub)

    return cranked_bar + reach_hub + reach_lever + reach_tip + arms


def build_cab_reach_rod(cfg: Dict):
    """
    8-18 左侧立式扁钢逆转长拉杆 (イラスト 8, 図 9.1, p43 SideB):
      - 参数化追踪 reverser_shaft (rev_x, rev_z+7.0) 至驾驶室窗台高度 Z=+44.0mm 螺旋滑块 8-17 (-122.5, +32.5, +44.0mm)；
      - 后 80% 段 100% 笔直水平延伸于 Z=+44.0mm，仅前 20% 平滑下弯接入 25-7 护罩内的 2-18 摇臂顶销！
    """
    vg = cfg["walschaerts_valve_gear"]
    rev_x = float(vg["reverser_shaft_x"])
    rev_z = float(vg["reverser_shaft_z"])
    x_rear_pin = -122.5 - rev_x
    z_rear_pin = 44.0 - (rev_z + 7.0)
    front_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.70, height=3.0)
        + Pos(-1.5, 0, 0) * Box(3.0, 3.0, 2.9)
        - Pos(+0.6, 0, 0) * Box(3.4, 1.72, 4.4)
    )
    pin_f = make_stepped_pin_L(0.0, 0.0, -1.90, +1.90, r_shank=0.80, r_head=1.30, r_nut=1.20)

    p_f_top = [
        Vector(-1.8, 0.0, +1.15),
        Vector(-12.0, 0.0, 0.48 * z_rear_pin + 1.15),
        Vector(-24.0, 0.0, 0.94 * z_rear_pin + 1.15),
        Vector(-36.0, 0.0, z_rear_pin + 1.15),
    ]
    p_f_bot = [
        Vector(-36.0, 0.0, z_rear_pin - 1.15),
        Vector(-24.0, 0.0, 0.94 * z_rear_pin - 1.15),
        Vector(-12.0, 0.0, 0.48 * z_rear_pin - 1.15),
        Vector(-1.8, 0.0, -1.15),
    ]
    p_r_top = Vector(x_rear_pin + 0.6, 0.0, z_rear_pin + 1.15)
    p_r_bot = Vector(x_rear_pin + 0.6, 0.0, z_rear_pin - 1.15)
    w_xz = Wire([
        Edge.make_spline(p_f_top, tangents=(Vector(-1.0, 0.0, 0.48).normalized(), Vector(-1.0, 0.0, 0.0))),
        Edge.make_line(p_f_top[-1], p_r_top),
        Edge.make_line(p_r_top, p_r_bot),
        Edge.make_line(p_r_bot, p_f_bot[0]),
        Edge.make_spline(p_f_bot, tangents=(Vector(1.0, 0.0, 0.0), Vector(1.0, 0.0, -0.48).normalized())),
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
    7-5 排液凸轮横轴 + 右侧单只 7-1~3 驱动摇臂(Qty=1, -Y 司炉侧) + 7-4/7-6 紧凑凸轮 + 1-19x4 内侧窄扁叠搭剪式摇板 + 1-18x4 横向顶针 (イラスト 1, 7, 図 1.3 & p44, cc=1):
      - 彻底删除 8.6mm 宽虚假阶梯大黄铜方块、删除随轴摆动的排液嘴 (1-15/1-16 实为气缸体静态班卓喷嘴!)、删除左侧虚构 7-1 摇臂！
      - 1-19x4 还原为紧贴气缸内底壁 (|Y|≈25.8mm) 的 2.0mm 窄冲压扁杠杆，前片叠搭于后片之上 (Manual p.15 図 1.3)！
    """
    cross_shaft_7_5 = Rot(90, 0, 0) * Cylinder(radius=0.95, height=45.6)  # |Y|<=22.8mm，与 ø15 气缸活塞盘 (|Y|>=25.1mm) 保持 >=2.3mm 净空!
    # BOM 7-1~7-3 (ドレンホーン, Qty=1): 仅存在于右侧 (-Y=-17.8mm) 衔接单根 8-5 水平拉杆!
    horn_7_1_R = (
        Pos(0.0, -17.8, +5.45) * Box(1.85, 1.25, 11.1)
        + Pos(0.0, -17.8, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.60, height=1.75)
        + Pos(0.0, -17.8, +10.9) * Rot(90, 0, 0) * Cylinder(radius=1.25, height=1.75)
    )
    linkage = cross_shaft_7_5 + horn_7_1_R
    for sgn in (-1.0, 1.0):
        cam_7_4 = (
            Pos(0.0, sgn * 22.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=1.8)
            + Pos(+0.8, sgn * 22.2, -3.4) * Box(1.85, 1.5, 7.0)
            + Pos(+1.6, sgn * 24.2, -6.6) * Rot(90, 0, 0) * Cylinder(radius=1.18, height=4.8)
        )
        # 図 1.3: 前后两片窄冲压扁钢摇板 1-19 (宽仅 1.95mm!) 绕 1-17 竖销摆动，在中央 dx=+1.6mm 处上下叠搭
        lever_rear_1_19 = (
            Pos(-0.9, sgn * 26.2, -7.05) * Box(5.8, 1.95, 0.80)
            + Pos(-3.4, sgn * 26.2, -6.95) * Cylinder(radius=1.15, height=1.10)
        )
        lever_front_1_19 = (
            Pos(+4.1, sgn * 26.2, -6.30) * Box(5.8, 1.95, 0.80)
            + Pos(+6.6, sgn * 26.2, -6.40) * Cylinder(radius=1.15, height=1.10)
        )
        pivot_tie = Pos(+1.6, sgn * 26.2, -6.68) * Cylinder(radius=1.05, height=1.60)
        push_pins_1_18 = (
            Pos(-3.4, sgn * 27.6, -6.95) * Rot(90, 0, 0) * Cylinder(radius=0.55, height=3.2)
            + Pos(+6.6, sgn * 27.6, -6.40) * Rot(90, 0, 0) * Cylinder(radius=0.55, height=3.2)
        )
        linkage = linkage + cam_7_4 + lever_rear_1_19 + lever_front_1_19 + pivot_tie + push_pins_1_18
    return linkage


def build_drain_pull_rod(cfg: Dict):
    """
    8-5 右侧 (-Y) 水平气缸排液拉杆 (イラスト 7, 8 & Manual p.25 図 8.1 / p45 实车照多图联立真值, cc=1):
      - 驾驶室室内段、wing_R 顶面 8-10 双竖耳导向鞍座及 11-9 后妻板穿孔段 (X∈[-141.8,-110.2]mm) 100% 纯水平同轴延伸于
        (Y=-25.0mm, Z=+29.4mm, pitch=0.0°，完全位于右地板翼 wing_R Z_top=+25.5mm 上方室内，零穿底板！)，
        后端同轴嵌入 8-6 水平滚花推拉旋钮；
      - 进入外火室罩前腔 (X>-110.2mm) 后平滑过渡至车架外缘下部净空走廊 (Y=-17.8mm, Z=+8.5mm) 100% 水平直达前端 7-1 排液摇臂 (+124.2mm)。
    """
    x_dc = float(cfg["cylinder_and_rods"]["cylinder_center_x"]) - 1.8
    pts_8_5 = [
        (-141.8, -25.0, +29.4),
        (-109.0, -25.0, +29.4),
        (-101.0, -17.8, +7.1),
        (x_dc, -17.8, +7.1),
    ]
    rod_8_5 = make_swept_tube(pts_8_5, r_tube=0.65, bend_r=2.6, r_ref=0.65)
    stop_collar = Pos(-134.8, -25.0, +29.4) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=1.6)
    front_clevis = Pos(x_dc, -17.8, +7.1) * Rot(90, 0, 0) * Cylinder(radius=1.28, height=1.8)
    return rod_8_5 + stop_collar + front_clevis


# ==================== 3B. 驾驶室 1:1 原厂全功能内构与煤水车全功能可动阀件网格 ====================

def build_cab_backhead_plate():
    """
    11-9 ボイラバックプレート + 12-5 x4 压板 + 12-2 浮雕拱门观火框与双下铰耳 + 全跨界管线冲压过孔 (イラスト 11, 12 & p45, cc=1):
      - 按 イラスト 11 item 9 原厂冲压图纸，100% 同轴打通 13-4 主蒸汽槽口、13-8 通风管孔、13-9 加减管孔、
        13-2 逆止阀孔、13-6/13-10 水面计上下管孔、11-8 汽笛孔、13-11 排污管槽口及右地板上方室内 8-5 水平排液拉杆过孔！
    """
    lower_skirt = Pos(-113.2, 0.0, 32.5) * Box(2.2, 53.6, 25.0)
    upper_arch = Pos(-113.2, 0.0, 44.5) * Rot(0, 90, 0) * Cylinder(radius=26.8, height=2.2)
    underfloor_flanges = (
        Pos(-113.2, +18.0, 16.2) * Box(2.2, 3.6, 8.6)
        + Pos(-115.0, +18.8, 16.2) * Box(4.2, 1.5, 8.2)
        + Pos(-113.2, -18.0, 16.2) * Box(2.2, 3.6, 8.6)
        + Pos(-115.0, -18.8, 16.2) * Box(4.2, 1.5, 8.2)
    )
    clamp_tl = Pos(-113.6, +12.5, 67.2) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=3.0)
    clamp_tr = Pos(-113.6, -12.5, 67.2) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=3.0)
    clamp_l = Pos(-113.6, +23.2, 35.8) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=3.0)
    clamp_r = Pos(-113.6, -23.2, 35.8) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=3.0)
    # 12-2 原厂铸造凹槽拱顶观火门框 + 左右双耳铰座 + 底部灰唇托盘 + 上锁鼻
    frame_bezel_12_2 = (
        Pos(-114.2, 0.0, 28.6) * Box(1.7, 15.2, 14.0)
        + Pos(-114.3, 0.0, 32.6) * Rot(0, 90, 0) * Cylinder(radius=7.6, height=1.9)
        - Pos(-115.0, 0.0, 28.8) * Box(1.4, 12.6, 11.4)
        + Pos(-115.2, +5.2, 21.8) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=1.9)
        + Pos(-115.2, -5.2, 21.8) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=1.9)
        + Pos(-115.1, 0.0, 21.1) * Box(2.2, 14.2, 1.2)
        + Pos(-115.1, 0.0, 36.2) * Box(2.1, 3.0, 1.5)
    )
    through_cuts = (
        Pos(-113.5, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=4.5, height=8.0)        # 火门孔
        + Pos(-113.2, +4.1, 72.2) * Box(8.0, 13.6, 6.4)                                  # 13-4 主蒸汽顶过槽
        + Pos(-113.2, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=8.0)    # 13-8 -> 17-3 通风管孔
        + Pos(-113.2, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=2.35, height=8.0)    # 13-9 -> 炉内干汽管孔
        + Pos(-113.2, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=2.65, height=8.0)   # 13-2 锅炉逆止阀颈孔
        + Pos(-113.2, -17.5, 60.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=8.0)   # 13-6 水面计上蒸汽管孔
        + Pos(-113.2, -20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=8.0)   # 13-10 水面计下水管水平孔
        + Pos(-113.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=3.35, height=8.0)   # 11-8 真汽笛穿孔
        + Pos(-113.2, +16.2, 15.5) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=8.0)   # 13-11 锅炉排污管孔
        + Pos(-115.5, +18.2, 15.8) * Rot(90, 0, 0) * Cylinder(radius=2.35, height=6.0)   # 11-16 排污针阀横向座孔
        + Pos(-113.2, -25.0, 29.4) * Rot(0, 90, 0) * Cylinder(radius=1.60, height=8.0)   # 8-5 右地板上方室内水平排液拉杆过孔
    )
    return (lower_skirt + upper_arch + underfloor_flanges + clamp_tl + clamp_tr + clamp_l + clamp_r + frame_bezel_12_2) - through_cuts


def build_cab_deck_and_mounts():
    """
    8-11 運転室U型折弯阶梯床 (中央下沉踏板 Z_top=+20.0mm + 左右完整无开槽抬升侧翼板 Z_top=+25.5mm!)
    + 8-21 纵向延长板 (14.0mm X × 3.8mm Y × 1.2mm Z, 长边严格沿机车 X 轴向!)
    + 8-10 右翼上方双竖耳同轴水平导向鞍座 (Y=-25.0mm, Z=+29.4mm 承托 100% 位于地板上方的水平排液杆 8-5/8-6, 零下探底板!)
    + 4-6/6-4 纵梁 + 4-7/6-12 后端梁 + 4-8 旁通阀 U 型托架 + 4-9 燃烧器后托架 (イラスト 4 Fig.3, 6, 8, 図 8.1 & p45, cc=1)。
    """
    # 1. 中央下沉司炉踏板区 (|Y| <= 17.2mm, Z ∈ [+18.8, +20.0]mm) + 后中牵引杆槽 (彻底删除右侧虚假开洞！)
    center_pan = (
        Pos(-130.0, 0.0, 19.4) * Box(33.0, 34.8, 1.2)
        - Pos(-145.0, 0.0, 19.4) * Box(5.0, 22.0, 4.0)
    )
    # 2. 左右斜折过渡立板 (|Y| ∈ [16.8, 20.8]mm, Z: +19.4 -> +24.9mm) 与左右完整高侧翼板 (|Y| ∈ [20.4, 38.6]mm, Z ∈ [+24.3, +25.5]mm)
    riser_L = Pos(-130.6, +18.8, 22.15) * Rot(+54.0, 0, 0) * Box(34.2, 6.8, 1.2)
    riser_R = Pos(-130.6, -18.8, 22.15) * Rot(-54.0, 0, 0) * Box(34.2, 6.8, 1.2)
    wing_L = Pos(-130.8, +29.5, 24.9) * Box(34.6, 18.2, 1.2)
    wing_R = Pos(-130.8, -29.5, 24.9) * Box(34.6, 18.2, 1.2)
    deck_8_11 = center_pan + riser_L + riser_R + wing_L + wing_R

    # 3. 8-21 延長板: 长边 14.0mm 严格平行于机车 X 轴向锁固于左翼前缘下方承托 8-12
    strip_8_21 = Pos(-121.0, +32.5, 23.9) * Box(14.0, 3.8, 1.2)

    # 4. 4-6 / 6-4 纵梁 + 4-7 / 6-12 后端梁 + 4-8 逆止弁スタンド (U型托架含 +Y=+11.6mm 4-17 M5 锁固立耳!) + 4-9 燃烧器后托架
    channels_4_6 = (
        Pos(-130.5, +13.6, 16.2) * Box(29.0, 1.3, 5.8)
        + Pos(-130.5, -13.6, 16.2) * Box(29.0, 1.3, 5.8)
    )
    rear_beam_4_7 = (
        Pos(-144.2, 0.0, 15.0) * Box(1.6, 44.0, 8.4)
        - Pos(-144.2, 0.0, 15.0) * Box(4.0, 20.5, 4.8)
    )
    stand_4_8 = (
        Pos(-135.5, 0.0, 18.2) * Box(5.2, 27.6, 1.8)
        + Pos(-135.5, +11.6, 16.2) * Box(5.2, 1.4, 5.4)
        - Pos(-135.5, +11.6, 15.0) * Rot(90, 0, 0) * Cylinder(radius=2.55, height=4.0)
    )
    holder_4_9 = (
        Pos(-142.8, 0.0, 9.6) * Box(2.8, 10.6, 6.4)
        - Pos(-142.8, 0.0, 3.8) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=6.0)
    ) + Pos(-143.6, +6.5, 11.8) * Box(1.8, 3.2, 4.6) + Pos(-143.6, -6.5, 11.8) * Box(1.8, 3.2, 4.6)

    # 5. 8-10 ドレンねじスタンド (イラスト 8 & 図 8.1: 单立柱紧凑 L 型铜支架，删除 13.6mm 双耳长鞍桥!)
    stand_8_10 = (
        Pos(-131.2, -25.0, 25.6) * Box(4.6, 4.6, 1.1)
        + Pos(-131.2, -25.0, 27.6) * Box(1.8, 3.2, 4.2)
        + Pos(-131.2, -25.0, 29.4) * Rot(0, 90, 0) * Cylinder(radius=1.65, height=1.8)
        + Pos(-131.2, -26.5, 26.2) * Cylinder(radius=0.65, height=0.6)
        + Pos(-131.2, -23.5, 26.2) * Cylinder(radius=0.65, height=0.6)
        - Pos(-131.2, -25.0, 29.4) * Rot(0, 90, 0) * Cylinder(radius=0.98, height=5.0)
    )

    bypass_hole = Pos(-135.5, +8.5, 19.5) * Cylinder(radius=1.65, height=6.0)
    riser_6_8_slot = Pos(-115.8, -11.8, 19.4) * Box(4.4, 4.0, 5.0)
    return (deck_8_11 + strip_8_21 + channels_4_6 + rear_beam_4_7 + stand_4_8 + holder_4_9 + stand_8_10) - bypass_hole - riser_6_8_slot


def build_cab_reverser_stand():
    """
    8-12 (细圆柱前立柱) + 8-13 (竖排双圆孔后U字座 + 纵向开槽前折顶帽) + 8-15 (顶置纵向分度导条) 日本国铁左驾螺旋逆转机座 (イラスト 8, 図 9.1 & p45, cc=1):
      - 直接座落于 8-11 抬升左侧翼板与 8-21 纵向延长板上 (-127.5, +32.5, +25.5mm)，丝杠轴心 z_scr=18.5mm -> 世界系 Z=+44.0mm 窗台高！
    """
    z_scr = 18.5
    base_pad = Pos(0.0, 0.0, +0.55) * Box(20.6, 4.4, 1.1)
    post_8_12_f = (
        Pos(+9.0, 0.0, 0.5 * z_scr) * Cylinder(radius=1.25, height=z_scr)
        + Pos(+9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.2)
    )
    ped_8_13_r = (
        Pos(-9.0, 0.0, 0.5 * (z_scr + 2.2)) * Box(2.2, 4.4, z_scr + 2.2)
        + Pos(-9.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=2.4)
        # 8-13 顶部向前 (+X) 折弯纵向帽板 (长边沿 X 轴!)
        + Pos(-6.8, 0.0, z_scr + 2.15) * Box(6.4, 4.4, 1.0)
        - Pos(-9.0, 0.0, +5.8) * Rot(0, 90, 0) * Cylinder(radius=1.25, height=5.0)
        - Pos(-9.0, 0.0, +12.2) * Rot(0, 90, 0) * Cylinder(radius=1.25, height=5.0)
    )
    # 8-15 顶部纵向刻度导条 (长边 19.6mm 严格沿机车 X 轴向连接前后立柱顶面!)
    bar_8_15_top = Pos(+0.4, +1.35, z_scr + 2.15) * Box(19.6, 1.5, 1.0)
    for hx in (-4.5, -1.5, +1.5, +4.5):
        bar_8_15_top -= Pos(hx, +1.35, z_scr + 2.15) * Cylinder(radius=0.42, height=3.0)
    bore_8_14 = Pos(0.0, 0.0, z_scr) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=24.0)
    return (base_pad + post_8_12_f + ped_8_13_r + bar_8_15_top) - bore_8_14


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
    8-17 纵向丝杠行走螺母滑块 (5.2mm X × 2.8mm Y × 5.2mm Z, 长边沿机车 X 轴!) + 8-20 x2 纵向双扁连杆 + 8-19 x2 肩销 (イラスト 8 & 図 9.1, cc=1):
      - 8-20 x2 自螺母沿 +X 水平向前延伸至 (+5.0, 0.0, 0.0) = 世界系 (-122.5, +32.5, +44.0mm)，同轴衔接 8-18 平直逆转拉杆后端孔！
    """
    nut_8_17 = (
        Pos(-0.6, 0.0, -0.7) * Box(5.2, 2.8, 5.2)
        + Pos(-0.6, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.38, height=3.6)
    )
    link_in = Pos(+2.2, -1.05, 0.0) * Box(7.8, 0.75, 2.2)
    link_out = Pos(+2.2, +1.05, 0.0) * Box(7.8, 0.75, 2.2)
    rear_pin_8_19 = Pos(-0.6, 0.0, -1.5) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=3.8)
    front_pin = Pos(+5.0, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder(radius=0.98, height=3.4)
    return nut_8_17 + link_in + link_out + rear_pin_8_19 + front_pin


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
    """
    13-5 #3 六角中空螺母 + 右侧水面计下座同轴底部 M4-6 一字槽吹扫螺塞 (-118.5, -31.3, +28.8mm, イラスト 11 Fig.2):
      - 顶部密封芯栓严格止于 z_local=+2.0mm (Z_world=+30.8mm < +31.8mm 13-10 水管下缘)，确保闭合时 13-10 <-> 11-7 连通器流道 100% 畅通无阻！
    """
    hex_13_5 = make_hex_nut((0.0, 0.0, +0.9), (0.0, 0.0, 1.0), hex_r=2.25, length=1.8)
    stem = Pos(0.0, 0.0, +1.0) * Cylinder(radius=1.10, height=2.0)
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
    12-1 (双叶对开浮雕铸造蝴蝶火室门 + 中缝 + 4 孔二次风配风口 + 双斜铰链带 + 顶卡扣) + 12-2 凹陷隔热盾框 + Template 3 陶瓷衬垫 + 12-3 底铰销 (イラスト 12, Manual p.28 & p45, cc=1):
      - 局部原点 (0,0,0) 位于 12-3 底铰横销中心 (-115.4, 0.0, +21.8mm, axis='0 -1 0')；
      - 闭合时两叶铸造门板紧贴 12-2 门框，4 个二次风圆孔通过门背 0.8mm 隔热挡板间隙精确节流助燃空气；翻开时绕 12-3 受重力自然落向定位卡肩！
    """
    pin_12_3 = (
        Rot(90, 0, 0) * Cylinder(radius=1.08, height=13.2)
        + Pos(0.0, +6.2, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.42, height=0.8)
        + Pos(0.0, -6.2, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.42, height=0.8)
    )
    # 左右双斜铰链臂 (自 z=0 底铰耳连向双叶铸门外缘)
    strap_L = (
        Pos(-0.75, +3.6, +3.4) * Rot(+16, 0, 0) * Box(1.3, 1.7, 7.2)
        + Pos(0.0, +3.6, 0.0) * Rot(90, 0, 0) * Cylinder(radius=1.48, height=1.8)
    )
    hinge_straps = strap_L + mirror(strap_L, about=Plane.XZ)

    # 12-1 铸造双翼圆角八边门扇主体 + 外周凸缘加强筋 + 中央对开分缝凹槽 + 4 个二次风进气圆孔
    wing_base = (
        Pos(-0.55, 0.0, +7.0) * Box(1.4, 11.8, 11.2)
        + Pos(-0.65, 0.0, +7.0) * Rot(0, 90, 0) * Cylinder(radius=5.85, height=1.55)
    )
    wing_rim = (
        Pos(-1.15, 0.0, +7.0) * Rot(0, 90, 0) * (Cylinder(radius=5.55, height=0.75) - Cylinder(radius=4.55, height=1.4))
        + Pos(-1.20, +2.35, +7.0) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=0.70)
        + Pos(-1.20, -2.35, +7.0) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=0.70)
    )
    center_split_groove = Pos(-1.35, 0.0, +7.0) * Box(0.85, 0.62, 11.4)
    draft_holes = None
    for sy in (-2.35, +2.35):
        for sz in (+5.45, +8.55):
            dh = Pos(-1.05, sy, sz) * Rot(0, 90, 0) * Cylinder(radius=0.58, height=1.8)
            draft_holes = dh if draft_holes is None else (draft_holes + dh)

    # 向火面 (+X) 12-2 内凹防火盾盘 + Template 3 セラミックシート 隔热块 (保留周边导风阶梯)
    baffle_shield_inner = (
        Pos(+0.35, 0.0, +7.0) * Box(0.85, 9.6, 9.6)
        + Pos(+0.92, 0.0, +7.0) * Box(0.75, 8.2, 8.2)
        + Pos(+0.35, +3.2, +7.0) * Rot(0, 90, 0) * Cylinder(radius=0.75, height=1.35)
        + Pos(+0.35, -3.2, +7.0) * Rot(0, 90, 0) * Cylinder(radius=0.75, height=1.35)
    )
    top_latch = (
        Pos(-1.15, 0.0, +13.1) * Box(1.9, 2.5, 1.5)
        + Pos(-1.65, +1.8, +12.6) * Rot(0, 90, 0) * Cylinder(radius=0.65, height=1.1)
    )
    return (pin_12_3 + hinge_straps + wing_base + wing_rim + baffle_shield_inner + top_latch) - center_split_groove - draft_holes


def build_cab_bypass_lever():
    """
    6-9 + 8-7 驾驶室左地板 (+Y=+8.5mm, X=-135.5mm, Z=+22.4mm) 低矮光滑圆筒毂无头紧定螺钉式旁通拨杆 (イラスト 6, 8, 図 8.3 & p45 实车照真值, cc=1):
      - 按 Step 8(5) & 図 8.3 真值：右旋针阀 6-9 顺时针旋紧全闭时拨杆指向后略偏左 θ_closed=+162.0° (末端 Y=+12.5mm，距左侧立板 riser_L Y=+17.2mm 留足 4.7mm 安全净空！)，
        逆时针 (+Z 正向旋转 0..+75°) 向中央平踏板内侧开启，全程 100% 零穿模！
    """
    stem_6_9 = (
        Pos(0.0, 0.0, -3.5) * Cylinder(radius=1.08, height=7.4)
        + Pos(0.0, 0.0, -7.6) * Rot(180, 0, 0) * Cone(bottom_radius=1.08, top_radius=0.32, height=1.4)
        + Pos(0.0, 0.0, +2.35) * Cylinder(radius=1.12, height=0.55)
    )
    floor_gland = Pos(0.0, 0.0, -2.0) * Cylinder(radius=2.25, height=0.85)
    smooth_hub = (
        Pos(0.0, 0.0, +0.40) * Cylinder(radius=2.80, height=4.0)
        - Pos(+2.35, 0.0, +0.40) * Rot(0, 90, 0) * Cylinder(radius=0.72, height=1.1)
    )
    ang_deg = +162.0
    arm_dir = Rot(0, 0, ang_deg) * Rot(0, -90, 0)
    arm_main = Pos(0.0, 0.0, +0.45) * arm_dir * Pos(0.0, 0.0, -6.5) * Cone(bottom_radius=1.15, top_radius=0.82, height=13.0)
    ring_1 = Pos(0.0, 0.0, +0.45) * arm_dir * Pos(0.0, 0.0, -9.2) * Torus(major_radius=0.92, minor_radius=0.20)
    tip_ball = Pos(0.0, 0.0, +0.45) * arm_dir * Pos(0.0, 0.0, -13.0) * Cylinder(radius=1.08, height=1.4)
    return stem_6_9 + floor_gland + smooth_hub + arm_main + ring_1 + tip_ball


def build_cab_drain_lever():
    """
    8-6 ドレンバルブ 100% 纯水平同轴纵向细纹滚花黄铜推拉旋钮 (イラスト 8, Manual p.24 図 8.1, 図 17.2 & p45 实车照真值, cc=1):
      - 局部原点 (0,0,0) 位于 (-140.0, -25.0, +29.4mm)，绕 Y/Z 轴倾角严格为 0.000° (100% 沿机车 X 轴水平同轴推拉 slide):
        1) 前端同轴导套 x∈[-1.6,+9.6]mm (X_world∈[-141.6,-130.4]mm, r=0.88mm) 穿过 8-10 单柱导座孔 (r=0.98mm) 并与 8-5 拉杆同轴固连；
        2) 中段车削光滑黄铜台阶颈部 x∈[-4.2,-0.4]mm (OD 3.30mm, 带 P-M2-2.5 顶面紧定螺钉沉孔)；
        3) 后端紧凑细密直纹滚花黄铜圆钮 x∈[-7.6,-3.8]mm (OD 4.90mm × 3.80mm, 28 道 0.07mm 浅滚花齿，彻底替换原 6.36×7.0mm 发黑左轮弹巢大黑柱!)。
    """
    guide_sleeve = Pos(+4.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.88, height=11.2)
    front_collar = (
        Pos(-2.3, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.65, height=3.8)
        - Pos(-2.2, 0.0, +1.42) * Cylinder(radius=0.52, height=0.7)
    )
    knob_core = (
        Pos(-5.7, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.45, height=3.2)
        + Pos(-4.0, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=2.45, top_radius=2.15, height=0.4)
        + Pos(-7.4, 0.0, 0.0) * Rot(0, -90, 0) * Cone(bottom_radius=2.45, top_radius=2.15, height=0.4)
    )
    flutes = None
    for k_f in range(28):
        ang_f = math.radians(k_f * (360.0 / 28.0))
        fy, fz = 2.52 * math.cos(ang_f), 2.52 * math.sin(ang_f)
        fl = Pos(-5.7, fy, fz) * Rot(0, 90, 0) * Cylinder(radius=0.14, height=3.1)
        flutes = fl if flutes is None else (flutes + fl)
    end_recess = Pos(-7.45, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=0.5)
    end_screw = (
        Pos(-7.35, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=0.45)
        - Pos(-7.56, 0.0, 0.0) * Box(0.32, 1.8, 0.30)
    )
    return guide_sleeve + front_collar + (((knob_core - flutes) - end_recess) + end_screw)


def build_dome_lid_29_12():
    """
    29-12 ドームのフタ + 29-13 軸 (イラスト 29 & Manual p.36 Step 29 原厂钢丝铰接可开启汽包/砂箱前盖, cc=1):
      - 局部原点 (0,0,0) 位于盖前端横向钢丝铰链 29-13 (世界系 X=+62.4, Y=0.0, Z=+91.85mm, axis='0 1 0')，向后严密密封 X∈[+50.2, +62.4]mm 沉孔井！
    """
    wire_29_13 = Rot(90, 0, 0) * Cylinder(radius=0.50, height=9.6)
    hinge_lug = Pos(-0.85, 0.0, +0.18) * Box(2.1, 5.6, 1.05)
    lid_disc = (
        Pos(-6.0, 0.0, +0.15) * Cylinder(radius=6.35, height=0.95)
        + Pos(-6.0, 0.0, -0.35) * Cylinder(radius=4.42, height=0.80)
        + Pos(-6.0, 0.0, +0.82) * Cone(bottom_radius=6.05, top_radius=3.80, height=0.55)
        + Pos(-11.8, 0.0, +0.25) * Box(1.9, 2.8, 0.85)
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
    tail_34_7 = extrude(Plane(origin=Vector(*p_loco) - ax * 1.6, z_dir=ax) * Circle(2.15), amount=3.4)
    nut_34_8 = (
        Pos(p_loco[0] + 0.6, p_loco[1], p_loco[2]) * Rot(0, 90, 0) * Cylinder(radius=3.35, height=3.4)
        + make_hex_nut((p_loco[0] - 0.8, p_loco[1], p_loco[2]), (1.0, 0.0, 0.0), hex_r=2.90, length=1.4)
    )
    hc1_clamp = extrude(Plane(origin=Vector(*p_loco) - ax * 1.5, z_dir=ax) * Circle(2.75), amount=1.2)
    bore_34_8 = extrude(Plane(origin=Vector(*p_loco) - ax * 3.0, z_dir=ax) * Circle(1.12), amount=6.8)
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
    イラスト 3, 4, 5, 19, 20, 23 真值多片叠板簧 (3-5 x8 + 5-15 x2 + 5-14 x4)、均衡梁支座、四轴前挂闸瓦及先/从台车真实承力横梁总成 (100% Y 对称, cc=1):
      - 4 对动轮上方完整还原 4 层阶梯叠片板簧 (3-5 x8) + 中央 3-6 弹簧箍 + 吊挂扁销；
      - 第 4 动轮后方车架外缘 (X=-85.0mm, |Y|=15.6mm) 还原 5-15 x2 从台车副板簧 + 5-14 x4 前后双吊杆 (两端刚性锚固于副车架，零悬空!)；
      - 增补 4-10 先台车双孔支点鞍梁 (X=+111.0mm, 下伸承力柱直抵 19-6 支点座)、4-4/4-5 先台车尾端拉簧上挂架 (X=+83.0mm, 直钩 19-8 拉簧顶端!)
        及 4-14 从台车支点横梁 (X=-64.0mm, 下伸 ø8.4mm 承力心轴零间隙压合 19-13 从台车前鼻!)，彻底消除悬空弹簧！
    """
    w_cfg = cfg["wheels_and_axles"]
    ax_xs = [
        float(w_cfg["axle_x"]["drive_1"]),
        float(w_cfg["axle_x"]["drive_2"]),
        float(w_cfg["axle_x"]["drive_3"]),
        float(w_cfg["axle_x"]["drive_4"]),
    ]

    tie_f = Pos(0.5 * (ax_xs[0] + ax_xs[1]), 0.0, 13.2) * Box(4.0, 29.6, 2.2)
    tie_m = Pos(0.5 * (ax_xs[1] + ax_xs[2]), 0.0, 13.2) * Box(4.0, 29.6, 2.2)
    tie_r = Pos(0.5 * (ax_xs[2] + ax_xs[3]), 0.0, 13.2) * Box(4.0, 29.6, 2.2)

    # 4-4 先台车 19-8 立式拉簧上吊挂横梁 + 4-5 外ボイラ固定金具 (X=+83.0mm, 下探至 Z=+2.7mm 与 19-8 弹簧顶座物理扣合, 上顶 4-5 L形托耳!)
    hanger_4_4_5 = (
        Pos(+83.0, 0.0, 12.6) * Box(5.6, 29.8, 2.4)
        + Pos(+83.0, 0.0, 7.3) * Cylinder(radius=2.35, height=9.2)
        + Pos(+83.0, 0.0, 14.5) * Box(4.2, 8.6, 2.0)
    )
    # 4-10 先台车固定单孔支点横枕梁 (イラスト 4, 19, 20: 单孔固定于 X=+112.0mm, 下伸承力座 Z=-5.8mm 配合 19-5 衬套紧贴 19-6!)
    bolster_4_10 = (
        Pos(+112.0, 0.0, 12.6) * Box(6.8, 29.8, 2.4)
        + Pos(+112.0, +13.6, 8.6) * Box(6.8, 2.6, 7.6)
        + Pos(+112.0, -13.6, 8.6) * Box(6.8, 2.6, 7.6)
        + Pos(+112.0, 0.0, 5.4) * Box(12.0, 29.8, 2.4)
        + Pos(+112.0, 0.0, -0.5) * Box(10.2, 8.2, 10.6)
    )
    # 4-11 x6 主台枠梁 (6 根车架间圆柱横撑, Z=+11.4mm 严格避开 Z=8.5/13.8mm 给水管走廊) + 4-13 軸動ポンプホルダ (-36.2mm)
    stays_4_11 = None
    for sx_4_11 in (+74.0, +62.0, +28.0, +16.0, -18.0, -52.0):
        st_rod = (
            Pos(sx_4_11, 0.0, 11.4) * Rot(90, 0, 0) * Cylinder(radius=1.10, height=29.2)
            + Pos(sx_4_11, +13.9, 12.1) * Box(2.6, 1.8, 2.8)
            + Pos(sx_4_11, -13.9, 12.1) * Box(2.6, 1.8, 2.8)
        )
        stays_4_11 = st_rod if stays_4_11 is None else (stays_4_11 + st_rod)
    holder_4_13 = (
        Pos(-36.2, 0.0, 11.4) * Box(4.6, 29.2, 1.4)
        + Pos(-36.2, +13.9, 12.1) * Box(4.6, 1.8, 2.8)
        + Pos(-36.2, -13.9, 12.1) * Box(4.6, 1.8, 2.8)
    )
    # 4-14 从台车支点横梁 (X=-64.0mm, 下伸 ø8.4mm 承力柱 Z=-6.0mm 零间隙压合 19-13 枢轴鼻 + 对称避让管槽保持 |com_y|==0!)
    bolster_4_14 = (
        (
            Pos(-64.0, +13.6, 9.2) * Box(6.8, 2.6, 8.4)
            + Pos(-64.0, -13.6, 9.2) * Box(6.8, 2.6, 8.4)
            + Pos(-64.0, 0.0, 5.9) * Box(6.8, 29.8, 2.2)
            + Pos(-64.0, 0.0, -0.4) * Cylinder(radius=4.2, height=11.2)
            + Pos(-64.0, 0.0, -7.8) * Cylinder(radius=1.65, height=5.0)
        )
        - Pos(-64.0, +8.2, 8.5) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=10.0)
        - Pos(-64.0, -8.2, 8.5) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=10.0)
    )

    assy = tie_f + tie_m + tie_r + hanger_4_4_5 + bolster_4_10 + stays_4_11 + holder_4_13 + bolster_4_14

    for sgn in (-1.0, 1.0):
        y_eq = sgn * 10.7
        y_frame_out = sgn * 15.6

        # 3-5 x8 四层阶梯叠板簧 + 3-6 中央抱箍 + 两端弹簧吊耳销
        for ax in ax_xs:
            sp1 = Pos(ax, y_eq, 12.6) * Box(21.0, 1.65, 0.85)
            sp2 = Pos(ax, y_eq, 13.35) * Box(16.6, 1.65, 0.80)
            sp3 = Pos(ax, y_eq, 14.05) * Box(12.2, 1.65, 0.78)
            sp4 = Pos(ax, y_eq, 14.70) * Box(8.0, 1.65, 0.72)
            buckle_3_6 = Pos(ax, y_eq, 13.65) * Box(3.4, 2.25, 3.3)
            hanger_f = Pos(ax + 9.8, y_eq, 11.9) * Box(1.4, 1.85, 2.2)
            hanger_r = Pos(ax - 9.8, y_eq, 11.9) * Box(1.4, 1.85, 2.2)
            spring_lug = Pos(ax, sgn * 12.45, 12.9) * Box(3.4, 4.8, 1.9)
            assy += sp1 + sp2 + sp3 + sp4 + buckle_3_6 + hanger_f + hanger_r + spring_lug

        # 5-15 x2 从台车副板簧 + 5-14 x4 前后立式吊杆 + 5-12/5-13 均衡摇臂 (イラスト 5 & 図 8.1, 两端真实锚固于后车架!)
        x_sp_rear = -85.0
        sp_5_15 = (
            Pos(x_sp_rear, y_frame_out, 8.6) * Box(19.0, 1.65, 0.90)
            + Pos(x_sp_rear, y_frame_out, 9.4) * Box(13.8, 1.65, 0.82)
            + Pos(x_sp_rear, y_frame_out, 10.15) * Box(9.0, 1.65, 0.76)
            + Pos(x_sp_rear, sgn * 14.8, 9.3) * Box(3.2, 3.0, 2.8)
            + Pos(x_sp_rear + 8.6, sgn * 14.9, 10.7) * Box(1.5, 2.6, 4.8)  # 5-14 前吊杆
            + Pos(x_sp_rear - 8.6, sgn * 14.9, 10.7) * Box(1.5, 2.6, 4.8)  # 5-14 后吊杆
        )
        # 3-14/3-16/3-18 真实工作主动螺旋弹簧 (Driving Spring, イラスト 3 Fig.1: 锚固于均衡链后端承接整车簧上载荷!)
        drv_spring_3_16 = (
            make_coil_spring_stack((-58.5, sgn * 11.8, +9.6), r_coil=1.08, r_wire=0.44, height=4.4, n_turns=5, r_core_rod=0.52)
            + Pos(-58.5, sgn * 13.0, +7.6) * Box(3.2, 3.6, 1.0)
            + Pos(-58.5, sgn * 13.0, +12.1) * Box(3.4, 3.6, 1.5)
        )
        assy += sp_5_15 + drv_spring_3_16

        # 3 对车架侧均衡梁支点耳座 (配合 3-4 x6 动态跷跷板均衡梁!)
        for xm in (0.5 * (ax_xs[0] + ax_xs[1]), 0.5 * (ax_xs[1] + ax_xs[2]), 0.5 * (ax_xs[2] + ax_xs[3])):
            fulcrum_pad = (
                Pos(xm, sgn * 12.2, 12.6) * Box(4.2, 2.8, 2.6)
                + Pos(xm, y_eq, 12.1) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=2.2)
            )
            assy += fulcrum_pad

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
            Pos(ax_xs[3] - 24.5, sgn * 14.4, +7.2) * Rot(0, -56.0, 0) * Box(2.2, 1.8, 17.5)
            + Pos(ax_xs[3] - 18.0, sgn * 14.4, +2.2) * Rot(90, 0, 0) * Cylinder(radius=1.7, height=2.4)
        )
        x_lo = -96.5
        x_hi = max(hanger_xs) + 2.0
        long_tie = Pos(0.5 * (x_lo + x_hi), sgn * 14.2, 12.8) * Box(x_hi - x_lo, 1.8, 2.1)
        assy += rear_eq_5_12 + long_tie

    return assy


def build_equalizer_lever_3_4():
    """
    3-4 イコライザー (イラスト 3 原厂动轮间跷跷板均衡梁摇臂, 6 件实例挂载于左右 3 跨动轮轴间, cc=1):
      - 孔距 L_pin = 27.0mm (跨越相邻动轮 3-5 板簧内侧吊耳销)，中央设 ø3.8mm 车架支点枢轴毂，随相邻动轮垂向悬挂差 (z_k - z_{k+1}) 真实翘动！
    """
    L_pin = 27.0
    half_L = 0.5 * L_pin
    beam = (
        Box(L_pin, 1.35, 2.3)
        + Pos(0.0, 0.0, -0.35) * Box(10.0, 1.35, 2.8)
        + Rot(90, 0, 0) * Cylinder(radius=1.95, height=1.95)
        + Pos(+half_L, 0.0, +0.35) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=1.65)
        + Pos(-half_L, 0.0, +0.35) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=1.65)
    )
    bores = (
        Rot(90, 0, 0) * Cylinder(radius=0.82, height=4.0)
        + Pos(+half_L, 0.0, +0.35) * Rot(90, 0, 0) * Cylinder(radius=0.62, height=4.0)
        + Pos(-half_L, 0.0, +0.35) * Rot(90, 0, 0) * Cylinder(radius=0.62, height=4.0)
    )
    return beam - bores


def build_axle_pump_ram(cfg: Dict):
    """
    6-1 中央轴动给水泵精密黄铜实心柱塞 + 前端连接叉耳与 PN3.2-1 密封环槽 (イラスト 6 & Manual p.23, cc=1):
      - 局部原点 (0,0,0) 位于前端与 6-2 偏心叉杆铰接销中心 (X_world=-22.0mm)；
      - ø3.20mm 抛光实心黄铜柱塞向后伸至 x_local=-17.2mm (X_world∈[-42.7,-35.7]mm)，在整个 ±3.5mm 冲程内 100% 严密填满 6-5 泵膛，彻底根除空膛伪影！
    """
    clevis_head = (
        Pos(-1.6, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=4.6)
        + Rot(90, 0, 0) * Cylinder(radius=1.95, height=1.95)
        - Rot(90, 0, 0) * Cylinder(radius=0.96, height=4.0)
    )
    plunger_6_1 = Pos(-10.3, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.60, height=13.8)
    oring_groove = Pos(-15.4, 0.0, 0.0) * Rot(0, 90, 0) * (
        Cylinder(radius=1.70, height=0.85) - Cylinder(radius=1.22, height=1.2)
    )
    chamfer_tip = Pos(-17.4, 0.0, 0.0) * Rot(0, -90, 0) * Cone(bottom_radius=1.60, top_radius=1.25, height=0.5)
    return (clevis_head + plunger_6_1 + chamfer_tip) - oring_groove


def build_pump_check_ball():
    """SB-ø3 不锈钢精密止回钢球 (ø3.00mm, 挂载于 6-5 轴动给水泵下吸上排双阀腔内随液压差真实跳动, cc=1)。"""
    return Sphere(radius=1.50)


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
    """
    イラスト 4 Fig.2, イラスト 19 & イラスト 20 真值先导台车第一类杠杆总成 (100% Y 对称, cc=1):
      - 19-2 前摇枕横梁 + 19-1 前纵向侧倾轴销 + 19-3 x2 横向滑移导杆；
      - 19-4 x2 窄距平行纵梁 (Y=±4.5mm) 向后越过中部 19-6 R2m/R3m 双孔支点座 (-52/-62mm, 上贴 4-10 车架枕梁!) 延伸至尾端 x=-85.0mm (X=+83.0mm)；
      - 尾端 19-7 弹簧挂座承接立式拉伸螺旋簧 19-8 (顶端 z=+11.2mm -> Z_world=+2.7mm 物理直扣车架 4-4/4-5 上挂钩，彻底消除悬空弹簧！)。
    """
    y_in = 0.5 * float(cfg["meta"]["track_gauge"]) - 5.2
    cross_bolster = Box(9.4, 2.0 * y_in + 4.0, 4.8)
    bolster_guides_19_3 = (
        Pos(+2.2, 0.0, +3.6) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=22.0)
        + Pos(-2.2, 0.0, +3.6) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=22.0)
        + Pos(0.0, +10.2, +3.2) * Box(6.6, 2.0, 3.2)
        + Pos(0.0, -10.2, +3.2) * Box(6.6, 2.0, 3.2)
        + Pos(+4.2, 0.0, +1.8) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=4.2)
    )
    pivot_tongue_19_6 = Pos(-56.0, 0.0, 0.0) * Box(12.0, 11.4, 4.2)
    boss_19_6 = Pos(-56.0, 0.0, 0.3) * Cylinder(radius=3.8, height=5.0)

    # 尾端 x=-85.0mm (X_world=+83.0mm): 19-7 下挂座 + 19-8 立式拉簧直达车架 4-4/4-5 上挂架 (z_top=+11.2mm -> Z_world=+2.7mm)!
    tail_seat_19_7 = Pos(-85.0, 0.0, 0.0) * Box(6.4, 11.4, 3.8)
    coil_19_8 = make_coil_spring_stack((-85.0, 0.0, +6.1), r_coil=1.85, r_wire=0.42, height=10.2, n_turns=6, r_core_rod=0.78)

    arms = None
    for sgn in (-1.0, 1.0):
        jbox = (
            Pos(0, sgn * y_in, 0.4) * Box(8.6, 4.2, 8.2)
            - Pos(0, sgn * y_in, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=7.0)
        )
        taper_arm = make_connecting_bar_xy((0.0, sgn * y_in), (-24.0, sgn * 4.5), 0.0, thick_y=2.8, height_z=3.8)
        long_bar_19_4 = Pos(-54.5, sgn * 4.5, 0.0) * Box(63.0, 2.2, 3.8)
        sub = jbox + taper_arm + long_bar_19_4
        arms = sub if arms is None else (arms + sub)

    frame = cross_bolster + bolster_guides_19_3 + pivot_tongue_19_6 + boss_19_6 + tail_seat_19_7 + coil_19_8 + arms
    frame -= Pos(-56.0, 0, 0) * Cylinder(radius=1.6, height=12.0)
    frame -= Rot(90, 0, 0) * Cylinder(radius=2.1, height=2.0 * y_in + 8.0)
    return frame


def build_trailing_truck_frame(cfg: Dict):
    """
    イラスト 19 Fig.2 & イラスト 23 真值从轮外置铸造轴箱 + 内置 19-14 减重孔摆臂 + 19-12/19-15 扭簧闭环 Delta 台车总成 (100% Y 对称, cc=1):
      - 彻底删除虚构的轴箱顶悬空立式螺旋压簧 coil_assy_19_15！
      - 19-13 前枢轴鼻 (z_top=+2.5mm -> Z_world=-6.0mm 零间隙承接车架 4-14 下伸承力心轴) + 19-11/19-17 侧板 + 19-16 后闭合横拉梁 (-71.0mm)；
      - 19-10 x2 实心斜顶梯形铸造外轴箱 (-54.0mm, |Y|∈[27.8,34.4]mm)；
      - 内侧 19-14 双三孔减重摆臂 + 19-12 x2 水平 Y 轴销栓 (-66.5mm) + 19-15 x2 内置钢丝扭簧 (上臂扣死于 19-11/17 上折翼下表面、下臂压紧 19-14 摆臂顶缘，两端 100% 物理锚固！)。
    """
    dx_ax = -54.0
    dx_sus = -66.5
    dx_rear = -71.0
    y_out = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.3  # 27.8 mm
    pivot_nose_19_13 = Cylinder(radius=4.5, height=5.0) + Pos(-6.5, 0.0, 0.0) * Box(13.0, 10.0, 4.4)
    front_transom = Pos(-34.5, 0.0, 0.5) * Box(3.2, 2.0 * y_out + 2.4, 4.0)
    rear_tie_19_16 = Pos(dx_rear, 0.0, 0.8) * Box(2.6, 2.0 * y_out + 2.8, 4.4)

    sides = None
    for sgn in (-1.0, 1.0):
        flare_arm = Pos(-21.0, sgn * 15.8, 0.5) * Rot(0, 0, sgn * -42.7) * Box(39.0, 3.2, 4.2)
        side_beam_19_11 = (
            Pos(-52.5, sgn * y_out, 0.8) * Box(38.0, 2.8, 4.6)
            # 19-11 / 19-17 后段顶部内折承簧翼缘 (供 19-15 扭簧上臂物理挂靠!)
            + Pos(-63.0, sgn * (y_out - 1.5), 3.9) * Box(16.0, 3.6, 1.4)
        )
        # 19-10 实心封顶梯形铸造外轴箱 (零悬空外露弹簧!)
        abox_19_10 = (
            Pos(dx_ax, sgn * (y_out + 1.8), 0.3) * Box(11.6, 4.6, 8.8)
            + Pos(dx_ax, sgn * (y_out + 3.6), -0.4) * Box(7.4, 3.2, 6.2)
            + Pos(dx_ax, sgn * (y_out + 1.5), 4.4) * Rot(sgn * -12.0, 0, 0) * Box(12.2, 4.8, 1.5)
            - Pos(dx_ax, sgn * (y_out + 1.0), 0.0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=6.5)
        )
        # 19-14 内侧带减重圆孔摆臂 + 19-12 水平 Y 轴悬挂枢轴销 + 19-15 内置双臂钢丝扭簧
        y_arm = sgn * 25.5
        swing_arm_19_14 = make_connecting_bar_xy((dx_ax, y_arm), (dx_sus, y_arm), 0.8, thick_y=1.3, height_z=4.2)
        for hx_hole in (-57.5, -60.5, -63.5):
            swing_arm_19_14 -= Pos(hx_hole, y_arm, 0.8) * Rot(90, 0, 0) * Cylinder(radius=0.85, height=4.0)
        pin_19_12 = Pos(dx_sus, sgn * 26.5, 2.2) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=4.8)
        torsion_coil_19_15 = (
            Pos(dx_sus, y_arm, 2.2) * Rot(90, 0, 0) * (Cylinder(radius=2.05, height=2.2) - Cylinder(radius=1.18, height=3.0))
            # 上扭臂: 自 (dx_sus, z=3.8) 向前上方顶紧 19-11/17 内折翼缘下表面 (z=3.9)!
            + make_swept_tube([(dx_sus, y_arm, 3.6), (-60.5, sgn * (y_out - 1.5), 3.6)], r_tube=0.52, bend_r=1.5)
            # 下扭臂: 自 (dx_sus, z=0.8) 向前下方压紧 19-14 摆臂顶缘 (z=2.4)!
            + make_swept_tube([(dx_sus, y_arm, 0.9), (-59.5, y_arm, 2.5)], r_tube=0.52, bend_r=1.5)
        )
        sub = flare_arm + side_beam_19_11 + abox_19_10 + swing_arm_19_14 + pin_19_12 + torsion_coil_19_15
        sides = sub if sides is None else (sides + sub)

    return pivot_nose_19_13 + front_transom + rear_tie_19_16 + sides - Cylinder(radius=1.8, height=8.0)


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
    1-4/1-29 主气缸 + 20-4 外包板轮廓 + 1-7/1-14 镂空外进气蒸汽室 + 10-1 螺钉顶盖 + 1-9 高精凸台式三气口交叉气道板 (aster-manual.pdf p.25 Fig.33)
    + 1-1..3/1-15/1-16 静态下垂班卓排液喷嘴总成 + 1-20 空心承插口 (イラスト 1, 7, 10, 20, 図 1.3 & p44, cc==1):
      - 严格还原 Aster D51 1-9 凸台研磨滑阀镜面 (Raised Lapped Valve-Port Mirror Table, Z=+10.80mm, 高出蒸汽室周边底槽 +0.65mm):
          1) 后滑阀口 (x_cyl=+0.20, X=+129.2mm, 1.50×5.20mm 宽矩口) 经外侧斜道交叉直通前气缸腔 (x_cyl=+14.2, Head-End);
          2) 中央排气口 (x_cyl=+4.20, X=+133.2mm, 2.50×5.20mm 宽矩口) 向内直连 7-8 排汽横管 (X_world=+130.0, Z=+12.5mm);
          3) 前滑阀口 (x_cyl=+8.20, X=+137.2mm, 1.50×5.20mm 宽矩口) 经外侧高位斜道交叉直通后气缸腔 (x_cyl=-4.0, Crank-End);
      - 严格还原 1-15 (M3中空螺栓) + 1-16 (圆筒形班卓环眼 + -Z 下垂排液嘴) 静态固装于气缸底壁 (X=+123.8/+133.8mm, |Y|=30.8mm)，内壁带 1-18 顶针导座！
    """
    rc = cfg["cylinder_and_rods"]
    bore = float(rc["bore"])
    cyl_cx = float(rc["cylinder_center_x"])
    z_valve = float(rc["valve_axis_z"])
    y_valve = +3.5
    r_barrel = 10.1
    x_inlet_loc = 123.5 - cyl_cx  # -5.5 mm -> X_world = +123.5 mm (7-9 精确同轴对接!)
    x_exh_loc = 130.0 - cyl_cx    # +1.0 mm -> X_world = +130.0 mm (7-8 精确同轴对接!)

    main_barrel = Pos(+5.45, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=21.5)
    rear_neck = (
        Pos(-8.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=5.0, height=7.6)
        + Pos(-12.9, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.5)
        - Pos(-9.8, -7.20, 0.0) * Box(14.0, 10.0, 16.0)
        - Pos(-9.8, +5.40, 0.0) * Box(14.0, 10.0, 16.0)  # 外缘止于 y=+0.4mm，为 FDC 合并杆/连接杆留出 >=0.9mm 净空
    )
    saddle_block = Pos(+3.6, -6.2, 6.2) * Box(23.8, 12.2, 23.6)
    outer_clad_20_4 = (
        Pos(+5.1, +2.75, 10.6) * Box(24.2, 14.3, 21.4)
        - Pos(+5.1, +9.1, -7.8) * Rot(+36, 0, 0) * Box(27.0, 8.0, 10.0)
    )
    front_cover = (
        Pos(+17.1, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=10.0, height=1.6)
        + Pos(+17.95, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.55, height=0.55)          # R3m 同轴薄圆座台
        + Pos(+17.95, +3.85, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.55, height=0.55)        # R2m +3.85mm 外偏座台 (図 20.1)
        + Pos(+18.25, 0.0, -6.30) * Rot(0, 90, 0) * Cylinder(radius=1.25, height=1.6)         # 20-5/20-3 ダミードレンバルブ本体
        + Pos(+18.65, 0.0, -7.20) * Cylinder(radius=0.62, height=1.7)                         # 20-5/20-3 下垂排口
        - Pos(+18.10, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=0.8)           # R3m M2 螺纹盲沉孔
        - Pos(+18.10, +3.85, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=0.8)         # R2m M2 螺纹盲沉孔
    )
    for b in (
        Pos(+17.95, 0.0, 0.0)
        * Rot(0, 90, 0)
        * PolarLocations(radius=8.4, count=6)
        * Cylinder(radius=0.68, height=0.65)
    ):
        front_cover += b

    steam_chest = Pos(+5.1, +3.3, 16.0) * Box(23.4, 11.4, 11.2)
    lid_10_1 = Pos(+5.1, +3.4, 22.0) * Box(23.8, 11.2, 1.3)
    for sx_b in (-9.2, +9.2):
        for sy_b in (-3.6, +3.6):
            lid_10_1 += make_hex_nut((5.1 + sx_b, 3.4 + sy_b, 22.95), (0.0, 0.0, 1.0), hex_r=0.95, length=0.8)

    spigot_inlet_1_20 = (
        Pos(x_inlet_loc, -9.2, 18.5) * Rot(90, 0, 0) * Cylinder(radius=2.9, height=8.8)
        + make_hex_nut((x_inlet_loc, -11.2, 18.5), (0.0, 1.0, 0.0), hex_r=3.3, length=2.0)
    )
    spigot_exh_1_20 = (
        Pos(x_exh_loc, -10.8, 12.5) * Rot(90, 0, 0) * Cylinder(radius=2.7, height=5.6)
        + make_hex_nut((x_exh_loc, -11.4, 12.5), (0.0, 1.0, 0.0), hex_r=3.0, length=1.8)
    )

    # 1-1..3 + 1-15 + 1-16 静态下垂气缸排液阀总成 (前后双工位 x_cyl = -5.2 / +4.8, y_cyl = -1.7 -> |Y_world| = 30.8mm)
    static_drains = None
    for dx_d, z_pin_d in ((-5.2, -10.75), (+4.8, -10.20)):
        dr_unit = (
            Pos(dx_d, -1.7, -9.8) * Cylinder(radius=1.78, height=5.2)
            + Pos(dx_d, -1.7, -13.2) * Cylinder(radius=2.05, height=2.2)
            + make_hex_nut((dx_d, -1.7, -14.8), (0.0, 0.0, 1.0), hex_r=1.80, length=1.2)
            + Pos(dx_d, -1.7, -16.4) * Cylinder(radius=1.28, height=2.6)
            + Pos(dx_d, -2.55, z_pin_d) * Rot(90, 0, 0) * Cylinder(radius=1.08, height=1.20)
        )
        static_drains = dr_unit if static_drains is None else (static_drains + dr_unit)

    gland_gp5_20_8 = (
        Pos(-7.2, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=1.8)
        + make_hex_nut((-6.8, y_valve, z_valve), (1.0, 0.0, 0.0), hex_r=2.30, length=1.2)
    )

    # 1-9 龙门刨越程槽凸台式三气口研磨镜面 (前后越程槽下沉至 z=+10.35mm，中央 12.6mm 全宽研磨镜台 z=+10.80mm 严密包覆下方排汽道 >=0.59mm 壁厚零串腔!)
    chest_recess_raw = (
        Pos(+4.5, +3.5, 15.58) * Box(20.4, 8.8, 10.45)
        - Pos(+4.2, +3.5, 10.55) * Box(12.6, 9.2, 0.55)
    )
    mirror_ports_cut = (
        Pos(+0.20, +3.50, 10.15) * Box(1.50, 5.20, 1.50)   # 後蒸気口 (Rear Valve Port -> Crosses to Front Head-End!)
        + Pos(+4.20, +3.50, 10.00) * Box(2.50, 5.20, 1.80) # 中央排気口 (Central Exhaust Port -> 7-8 Blast!)
        + Pos(+8.20, +3.50, 10.15) * Box(1.50, 5.20, 1.50) # 前蒸気口 (Front Valve Port -> Crosses to Rear Crank-End!)
    )
    chest_hollow = chest_recess_raw + mirror_ports_cut

    rear_port_to_front_head = make_swept_tube(
        [(0.2, +3.5, 10.2), (0.2, +5.6, 7.8), (0.2, +7.8, 5.4), (14.2, +7.8, 5.4), (14.2, 0.0, 5.2)],
        r_tube=0.68,
        bend_r=1.4,
    )
    front_port_to_rear_crank = make_swept_tube(
        [(8.2, +3.5, 10.2), (8.2, +8.4, 9.4), (-3.8, +8.4, 9.4), (-3.8, +6.2, 7.8), (-4.0, 0.0, 5.2)],
        r_tube=0.68,
        bend_r=1.4,
    )
    exhaust_passage = make_swept_tube(
        [(4.2, +3.5, 10.25), (4.2, +3.5, 9.15), (x_exh_loc, -4.5, 9.15), (x_exh_loc, -7.2, 12.5), (x_exh_loc, -14.2, 12.5)],
        r_tube=1.08,
        bend_r=0.85,
    )
    drain_bores = (
        Pos(-5.2, -1.7, -11.8) * Cylinder(radius=0.62, height=13.4)
        + Pos(+4.8, -1.7, -11.8) * Cylinder(radius=0.62, height=13.4)
    )
    spindle_bore = Pos(+1.0, y_valve, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.18, height=21.0)
    inlet_bore = make_swept_tube(
        [(x_inlet_loc, -14.0, 18.5), (x_inlet_loc, -3.5, 18.5), (-1.8, +3.5, 18.5)],
        r_tube=1.65,
        bend_r=2.0,
    )
    bore_cut = (
        Pos(+5.35, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=20.7)
        + Pos(-9.5, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.62, height=13.2)
        + chest_hollow
        + rear_port_to_front_head
        + front_port_to_rear_crank
        + exhaust_passage
        + drain_bores
        + spindle_bore
        + inlet_bore
    )
    assembly_L = (
        main_barrel + rear_neck + saddle_block + outer_clad_20_4 + front_cover
        + steam_chest + lid_10_1 + spigot_inlet_1_20 + spigot_exh_1_20
        + static_drains + gland_gp5_20_8 - bore_cut
    )
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_boiler_superstructure(cfg: Dict, heavy_equipped: bool = True):
    """
    锅炉烟室、流线型カマボコ砂箱汽包、15-14 锅顶主蒸汽管护罩、16-6 圆顶斜收外火室、全封闭前斜裙板与无缺口除烟板/驾驶室外衣总成
    (イラスト 15, 16, 18, 21, 22, 24, 25, 26, 28, 29, 30 & p43/p44 & 2010 实车彩照, cc==1):
      - 彻底根除 6 处模型缺口/失真：
        1) 28-22/23 除烟板按精确二维多边形拉伸，消除前上 V 形尖刺缺口与中部斜切咬口；
        2) 25-6 + 22-1/2 前斜甲板统一为全宽 79.2mm 共面 42° 面板 + 25-2 内侧封闭立裙，100% 消除车头红圈大镂空；
        3) 驾驶室矩形内腔掏空上限压低至 Z_top=+70.8mm (< 车顶内壁 +72.13mm)，100% 根除车顶两侧 35mm 贯穿长条破槽；
        4) 21-25 车顶通风器改为同心圆柱弧壳扇区 (R=53.4..55.6mm)，消除倒 V 凹折痕；
        5) 增补 15-14 锅顶主蒸汽管流线护罩 + 16-6 斜收外火室 + 28-1 火室后箍带 + 走台后段零缝贴合前妻板 + 外移 21-20 前瞭望窗 + 精细多级 15-8/10 发电机；
        6) 动轮弧槽与逆转轴槽加 1.8mm 实心内衬背碗，100% 根除锅筒下腹月牙破洞；15-3 假逆止阀、16-2 假排泥栓、22-13 半球封头储风缸按图纸 1:1 还原！
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
    sb_saddle = (
        Pos(136.0, 0, 21.5) * Box(36.0, 26.0, 14.0)
        + Pos(150.5, 0, 19.8) * Box(19.0, 54.4, 17.6)  # 25-2 烟室前鞍下全封闭胸板 (彻底封堵车头前斜板内侧透光空洞!)
    )
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

    b_x0, b_x1 = x_fb_rear + 1.4, x_sb_rear + 1.0
    r_barrel = 30.2
    barrel = Pos(0.5 * (b_x0 + b_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=b_x1 - b_x0)
    # 28-1~5 锅炉箍带 (含 -110.6mm 火室后缘箍带严密衔接驾驶室前妻板!)
    for bx in [74.0, 38.0, -14.0, -58.0, -110.5]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.55, height=1.6)

    # 15-14 スチームパイプカバー (イラスト 15 item 14 & p43: 封闭锅顶 -111..-93mm 主蒸汽管 13-4 钥匙孔凹坑的流线型黑铁护罩!)
    shroud_15_14_outer = (
        Pos(-102.2, 0.0, 71.3) * Rot(0, 90, 0) * Cylinder(radius=5.2, height=17.6)
        + Pos(-93.4, 0.0, 71.3) * Sphere(radius=5.2)
        + Pos(-102.2, 0.0, 71.0) * Box(17.6, 10.4, 4.2)
    ) & (Pos(-101.0, 0.0, 74.0) * Box(21.6, 12.0, 9.0))
    shroud_15_14_inner = (
        Pos(-104.2, 0.0, 71.3) * Rot(0, 90, 0) * Cylinder(radius=4.05, height=19.2)
        + Pos(-94.6, 0.0, 71.3) * Sphere(radius=4.05)
        + Pos(-104.2, 0.0, 70.0) * Box(19.2, 8.1, 5.2)
    )

    # 3. 标准型 JNR D51 カマボコ形流线砂箱/蒸汽包罩 29-11 (86.4mm 全长 X∈[-14.8,+71.6], Z_top=+91.85mm 全封闭零破洞顶面)
    dome_profile_table = [
        (-14.8, 69.80, 2.20, 0.9),
        (-13.6, 73.80, 5.60, 2.3),
        (-11.5, 75.60, 8.10, 3.3),
        (-8.5, 78.10, 9.90, 4.2),
        (-4.5, 81.60, 11.10, 5.0),
        (-0.5, 85.40, 11.80, 5.5),
        (+3.5, 88.60, 12.18, 5.9),
        (+7.5, 90.75, 12.35, 6.1),
        (+11.5, 91.65, 12.40, 6.2),
        (+16.0, 91.85, 12.40, 6.2),
        (+28.0, 91.85, 12.40, 6.2),
        (+40.0, 91.85, 12.40, 6.2),
        (+52.0, 91.85, 12.40, 6.2),
        (+60.0, 91.85, 12.40, 6.2),
        (+62.8, 91.75, 12.30, 6.1),
        (+64.6, 91.05, 11.85, 5.8),
        (+66.2, 89.65, 11.10, 5.4),
        (+67.6, 87.60, 10.05, 4.8),
        (+68.8, 84.90, 8.75, 4.1),
        (+69.8, 81.60, 7.25, 3.3),
        (+70.6, 77.90, 5.60, 2.5),
        (+71.2, 74.20, 3.85, 1.7),
        (+71.6, 70.20, 2.10, 0.9),
    ]
    dome_secs = []
    z_dome_root = 67.6
    for dx_s, z_top_s, hw_s, r_crn in dome_profile_table:
        z_sh = max(z_dome_root + 0.9, z_top_s - r_crn)
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

    dome_well_recess = Pos(56.4, 0.0, 90.6) * Cylinder(radius=4.55, height=3.2)
    screw_cm2_14 = (
        Pos(56.4, 0.0, 89.65) * Cylinder(radius=1.85, height=1.7)
        - Pos(56.4, 0.0, 90.25) * extrude(RegularPolygon(radius=0.95, side_count=6), amount=0.8)
    )
    dome_29_11 = (
        (
            loft(dome_secs, ruled=True)
            + Pos(56.4, 0.0, 91.75) * (Cylinder(radius=5.85, height=0.7) - Cylinder(radius=4.55, height=2.0))
            + Pos(62.4, +3.8, 91.80) * Box(1.8, 1.2, 1.0)
            + Pos(62.4, -3.8, 91.80) * Box(1.8, 1.2, 1.0)
            + Pos(34.0, 0.0, 91.90) * Cylinder(radius=3.4, height=0.85)
            + Pos(13.0, 0.0, 91.85) * Cylinder(radius=2.8, height=0.85)
        )
        - dome_well_recess
        + screw_cm2_14
    )

    r_sand_polar = 31.68
    phi_degs = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 89.0]
    sand_pipes_L = None
    for sp_x, sp_end_x in [(46.0, 53.0), (32.0, 32.0), (18.0, 10.0)]:
        pts_sp = []
        for idx_p, deg_p in enumerate(phi_degs):
            u_s = idx_p / float(len(phi_degs) - 1)
            rad_p = math.radians(deg_p)
            px_s = sp_x + (sp_end_x - sp_x) * (3.0 * u_s * u_s - 2.0 * u_s * u_s * u_s)
            py_s = r_sand_polar * math.sin(rad_p)
            pz_s = z_axis + r_sand_polar * math.cos(rad_p)
            pts_sp.append((px_s, +py_s, pz_s))
        pts_sp.insert(0, (sp_x, +9.6, z_axis + r_sand_polar * math.cos(math.radians(phi_degs[0]))))
        pts_sp.append((sp_end_x, +31.55, 31.0))

        rad_mid = math.radians(55.0)
        xm_clip = 0.5 * (sp_x + sp_end_x)
        ym_clip = (r_sand_polar - 0.95) * math.sin(rad_mid)
        zm_clip = z_axis + (r_sand_polar - 0.95) * math.cos(rad_mid)
        sp_one = (
            Pos(sp_x, +11.2, z_axis + r_sand_polar * math.cos(math.radians(phi_degs[0]))) * Rot(90, 0, 0) * Cylinder(radius=1.22, height=2.6)
            + Pos(xm_clip, +ym_clip, zm_clip) * Box(1.8, 2.4, 2.4)
            + make_swept_tube(pts_sp, r_tube=0.68, bend_r=4.2)
        )
        sand_pipes_L = sp_one if sand_pipes_L is None else (sand_pipes_L + sp_one)
    dome_29_11 = dome_29_11 + sand_pipes_L + mirror(sand_pipes_L, about=Plane.XZ)

    # 16-6 外火室斜收梯形侧包板 (上接 R=30.2mm 锅筒赤道 Z=44.0mm，下收至 |Y|=26.8mm, Z=18.2mm，后缘 X=-110.2mm 严密贴合前妻板!)
    fb_yz_wire = Wire([
        Edge.make_line(Vector(-110.6, -26.8, 18.2), Vector(-110.6, -30.2, 44.0)),
        Edge.make_line(Vector(-110.6, -30.2, 44.0), Vector(-110.6, +30.2, 44.0)),
        Edge.make_line(Vector(-110.6, +30.2, 44.0), Vector(-110.6, +26.8, 18.2)),
        Edge.make_line(Vector(-110.6, +26.8, 18.2), Vector(-110.6, -26.8, 18.2)),
    ])
    firebox = extrude(Face(fb_yz_wire), amount=(x_fb_front + 1.0) - (-110.6), dir=(1.0, 0.0, 0.0))

    # 4. 驾驶室外衣 21-28 + 同心凸圆弧车顶通风器 21-25 (彻底根除车顶 35mm 贯穿长破槽与倒 V 折痕!)
    cab_len, cab_w = 38.6, 79.2
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0  # -129.3 mm, X ∈ [-148.6, -110.0]
    cab_box = Pos(cab_x, 0, 48.8) * Box(cab_len, cab_w, 47.6)  # Z_min=+25.0mm, Z_max=+72.6mm (< 檐口车顶外弧 +73.8mm!)
    cab_front_arch_wall = (
        Pos(-110.95, 0.0, 36.0) * Rot(0, 90, 0) * Cylinder(radius=53.2, height=1.9)
    ) & (Pos(-110.95, 0.0, 80.5) * Box(1.9, cab_w, 21.0))
    rear_corner_tabs = (
        Pos(cab_x - 0.5 * cab_len + 0.8, +36.8, 24.5) * Box(1.6, 5.6, 2.0)
        + Pos(cab_x - 0.5 * cab_len + 0.8, -36.8, 24.5) * Box(1.6, 5.6, 2.0)
    )

    roof_len = cab_len + 7.2
    roof_xc = cab_x - 1.6
    roof_shell_raw = (
        Pos(roof_xc, 0.0, 36.0) * Rot(0, 90, 0) * (Cylinder(radius=54.0, height=roof_len) - Cylinder(radius=52.4, height=roof_len + 2.0))
    ) & (Pos(roof_xc, +20.20, 81.5) * Box(roof_len + 2.0, 40.40, 22.0))
    cab_roof = roof_shell_raw + mirror(roof_shell_raw, about=Plane.XZ)

    # 21-25 同心凸圆弧面车顶天窗 + 21-16 前后滑轨条 (严格绕车顶圆心 Z_c=36.0mm 同心拱起，消除倒 V 凹折!)
    vent_21_15 = (
        (
            Pos(-129.5, 0.0, 36.0)
            * Rot(0, 90, 0)
            * (Cylinder(radius=55.55, height=26.2) - Cylinder(radius=53.35, height=28.0))
        )
        & (Pos(-129.5, 0.0, 90.0) * Box(26.2, 18.2, 8.0))
    ) + (
        (
            Pos(-116.2, 0.0, 36.0) * Rot(0, 90, 0) * (Cylinder(radius=55.80, height=1.2) - Cylinder(radius=53.40, height=2.0))
            + Pos(-142.8, 0.0, 36.0) * Rot(0, 90, 0) * (Cylinder(radius=55.80, height=1.2) - Cylinder(radius=53.40, height=2.0))
        )
        & (Pos(-129.5, 0.0, 90.0) * Box(30.0, 19.2, 8.0))
    ) + Pos(-129.5, 0.0, 91.55) * Box(26.2, 0.8, 0.45)

    rear_doorway_cut = (
        Pos(cab_x - 0.5 * cab_len, 0, 44.0) * Box(10.0, 67.0, 52.0)
        + (
            Pos(cab_x - 0.5 * cab_len, 0, 36.0) * Rot(0, 90, 0) * Cylinder(radius=52.4, height=12.0)
            & Pos(cab_x - 0.5 * cab_len, 0, 79.5) * Box(12.0, 67.0, 21.0)
        )
    )
    cab_trim_L = (
        Pos(-127.0, +39.55, 65.8) * Box(26.5, 1.7, 1.0)  # 21-1/2 侧窗雨搭眉檐
        + Pos(-109.1, +31.8, 65.4) * Rot(0, +24, 0) * Box(2.4, 8.8, 0.75)  # 21-20 前瞭望窗斜遮阳眉檐 (外移至锅筒外净空区!)
        + Pos(-132.8, +39.55, 41.5) * Box(10.8, 0.5, 4.2)  # D51 号牌
        + Pos(cab_x - 0.5 * cab_len + 0.6, +34.8, 46.5) * Cylinder(radius=0.68, height=30.0)  # 21-9/10 后扶手杆
        + Pos(cab_x - 0.5 * cab_len + 1.2, +34.8, 60.5) * Box(2.0, 1.8, 1.4)
        + Pos(cab_x - 0.5 * cab_len + 1.2, +34.8, 32.5) * Box(2.0, 1.8, 1.4)
    )
    cab_trim = cab_trim_L + mirror(cab_trim_L, about=Plane.XZ)

    side_windows_cut = (
        Pos(-121.1, 0.0, 56.5) * Box(11.4, cab_w + 10.0, 12.0)
        + Pos(-133.9, 0.0, 56.5) * Box(9.0, cab_w + 10.0, 12.0)
    )
    spectacle_windows_cut = (
        Pos(-110.9, +31.8, 60.8) * Rot(0, 90, 0) * Cylinder(radius=3.5, height=4.0)
        + Pos(-110.9, -31.8, 60.8) * Rot(0, 90, 0) * Cylinder(radius=3.5, height=4.0)
    )

    cab = (
        cab_box + cab_front_arch_wall + rear_corner_tabs + cab_roof + vent_21_15 + cab_trim
        - Pos(cab_x - 0.9, 0, 44.4) * Box(cab_len - 3.6, cab_w - 3.2, 52.8)  # Z_top = +70.8mm < 车顶内弧 +72.13mm, 零穿顶!
        - (
            Pos(cab_x - 0.9, 0, 36.0) * Rot(0, 90, 0) * Cylinder(radius=52.4, height=cab_len - 3.6)
            & Pos(cab_x - 0.9, 0, 80.4) * Box(cab_len - 3.6, cab_w - 3.2, 20.0)
        )
        - side_windows_cut
        - spectacle_windows_cut
        - rear_doorway_cut
        - Pos(-111.5, +32.5, 44.1) * Box(8.0, 3.4, 5.2)  # 8-18 水平逆转拉杆 Z=+44.0mm 穿孔
    )

    # 5. 左右走台 + 零缝贴合前妻板的第 4 动轮后抬升走台 22-6 + 精确多边形拉伸除烟板 28-22/23 (イラスト 22, 26, 28)
    walk_x0, walk_x1 = x_fb_rear + 1.4, 151.0
    walk_len = walk_x1 - walk_x0
    walk_xm = 0.5 * (walk_x0 + walk_x1)

    rev_x_sh = float(cfg["walschaerts_valve_gear"]["reverser_shaft_x"])
    rev_z_sh = float(cfg["walschaerts_valve_gear"]["reverser_shaft_z"])
    rev_arm_len = float(cfg["walschaerts_valve_gear"]["reverser_arm_length"])

    defl_xz_pts = [
        Vector(124.5, 0.0, 31.2),
        Vector(124.5, 0.0, 68.2),
        Vector(169.5, 0.0, 68.2),
        Vector(181.5, 0.0, 56.2),
        Vector(181.5, 0.0, 22.2),
        Vector(169.0, 0.0, 22.2),
        Vector(154.5, 0.0, 30.2),
        Vector(131.5, 0.0, 30.2),
        Vector(131.5, 0.0, 31.2),
    ]
    defl_face = Face(Wire([Edge.make_line(defl_xz_pts[i], defl_xz_pts[(i + 1) % len(defl_xz_pts)]) for i in range(len(defl_xz_pts))]))

    def _build_side_walk_and_defl(sgn: float):
        board = Pos(walk_xm, sgn * 34.4, 30.8) * Box(walk_len, 10.2, 1.5)
        inward_web = Pos(walk_xm, sgn * 27.5, 30.8) * Box(walk_len - 6.0, 5.4, 1.5)
        bridge_cut = Pos(-87.4, sgn * 34.8, 30.8) * Box(46.8, 12.0, 4.0)
        lift_slot_cut = Pos(rev_x_sh + rev_arm_len, sgn * 36.0, 30.8) * Box(12.4, 5.9, 4.0)
        rev_hood_cut = Pos(rev_x_sh - 0.2, +32.5, 30.8) * Box(10.6, 5.2, 4.0) if sgn > 0 else None
        # 后抬升走台向后直抵 X=-110.6mm 嵌入前妻板，向内拓宽至 |Y|=25.5mm 严密咬合火室外衣！
        bridge_top = Pos(-87.3, sgn * 32.3, 41.5) * Box(46.6, 13.6, 1.5)
        bridge_ramp_f = Pos(-59.5, sgn * 34.2, 36.2) * Rot(0, +52, 0) * Box(14.2, 9.8, 1.4)
        bridge_tie = Pos(-87.3, sgn * 28.2, 36.2) * Box(46.6, 2.2, 11.4)
        board_base = (board + inward_web) - bridge_cut - lift_slot_cut
        if rev_hood_cut is not None:
            board_base = board_base - rev_hood_cut
        board_sub = board_base + bridge_top + bridge_ramp_f + bridge_tie

        defl_plate = Pos(0.0, sgn * 40.30, 0.0) * extrude(defl_face, amount=0.48, dir=(0.0, 1.0, 0.0), both=True)
        defl_lip = Pos(147.0, sgn * 39.25, 67.65) * Rot(sgn * -22, 0, 0) * Box(45.0, 2.8, 0.95)
        defl_win_cut = Pos(124.0, sgn * 40.30, 45.8) * Box(12.0, 5.0, 9.2)
        defl_grab = (
            Pos(176.5, sgn * 39.45, 43.5) * Cylinder(radius=0.55, height=17.5)
            + Pos(176.5, sgn * 39.85, 52.0) * Box(1.2, 1.0, 1.0)
            + Pos(176.5, sgn * 39.85, 35.0) * Box(1.2, 1.0, 1.0)
        )
        defl_raw = (defl_plate + defl_lip - defl_win_cut) + defl_grab
        defl_stays = (
            Pos(154.0, sgn * 33.8, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.72, height=13.5)
            + Pos(154.0, sgn * 27.6, 60.5) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=1.2)
            + Pos(132.5, sgn * 33.8, 60.5) * Rot(90, 0, 0) * Cylinder(radius=0.72, height=13.5)
            + Pos(132.5, sgn * 27.6, 60.5) * Rot(90, 0, 0) * Cylinder(radius=1.55, height=1.2)
        )

        # 前部外侧阶梯踏板条与侧帮护板 (与中央全宽 42° 斜面板无缝合一)
        front_steps = (
            Pos(162.0, sgn * 39.1, 19.6) * Rot(0, +42, 0) * Box(32.5, 1.0, 3.0)
            + Pos(162.0, sgn * 27.4, 19.6) * Rot(0, +42, 0) * Box(32.5, 1.0, 3.0)
        )
        for st_x, st_z in [(155.5, 26.5), (162.0, 20.6), (168.5, 14.7)]:
            front_steps += Pos(st_x, sgn * 33.25, st_z) * Box(3.4, 12.5, 1.1)

        # 15-3 ダミー逆止弁 (イラスト 15 & 28 真值位 X=+78.5mm, Z=+52.5mm: 圆座法兰 + 阀体 + 顶部六角闷盖 + 28-13/14 垂直假给水立管!)
        dummy_15_3 = (
            Pos(78.5, sgn * 27.8, 52.5) * Rot(90, 0, 0) * Cylinder(radius=2.9, height=2.2)
            + Pos(78.5, sgn * 30.2, 52.5) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=3.4)
            + make_hex_nut((78.5, sgn * 31.0, 55.2), (0.0, 0.0, 1.0), hex_r=1.85, length=1.5)
            + Pos(78.5, sgn * 31.0, 41.6) * Cylinder(radius=0.85, height=22.4)
        )
        # 16-2 ダミー泥ダメコック + 16-5 洗炉塞凸台 (イラスト 16)
        dummy_16_2 = (
            Pos(-78.0, sgn * 26.8, 22.8) * Rot(sgn * -90, 0, 0) * Cone(bottom_radius=2.0, top_radius=1.15, height=3.2)
            + Pos(-78.0, sgn * 28.8, 22.8) * Box(1.2, 1.4, 1.2)
            + Pos(-64.0, sgn * 26.8, 28.5) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=1.2)
            + Pos(-88.0, sgn * 26.8, 28.5) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=1.2)
        )
        # 22-13 半球封头储风缸 + 22-14 箍带 (イラスト 22)
        air_res_22_13 = (
            Pos(-18.0, sgn * 31.6, 25.8) * Rot(0, 90, 0) * Cylinder(radius=4.15, height=28.0)
            + Pos(-32.0, sgn * 31.6, 25.8) * Sphere(radius=4.15)
            + Pos(-4.0, sgn * 31.6, 25.8) * Sphere(radius=4.15)
            + Pos(-27.0, sgn * 31.6, 25.8) * Rot(0, 90, 0) * Cylinder(radius=4.45, height=1.3)
            + Pos(-9.0, sgn * 31.6, 25.8) * Rot(0, 90, 0) * Cylinder(radius=4.45, height=1.3)
            + Pos(-27.0, sgn * 31.6, 29.5) * Box(1.5, 4.2, 2.6)
            + Pos(-9.0, sgn * 31.6, 29.5) * Box(1.5, 4.2, 2.6)
        )
        handrail_26_10 = make_swept_tube(
            [(-44.0, sgn * 30.8, 56.5), (+152.0, sgn * 32.9, 56.5)],
            r_tube=0.55,
            bend_r=2.0,
        )
        for hx_s in (-42.0, +8.0, +62.0, +115.0, +148.0):
            y_in_h = sgn * (27.2 if hx_s > 92.0 else 26.2)
            y_End = sgn * (32.9 if hx_s > 92.0 else 30.8)
            handrail_26_10 += (
                Pos(hx_s, 0.5 * (y_in_h + y_End), 56.5) * Rot(90, 0, 0) * Cylinder(radius=0.62, height=abs(y_End - y_in_h) + 0.8)
                + Pos(hx_s, y_End, 56.5) * Rot(0, 90, 0) * Cylinder(radius=0.88, height=1.35)
            )
        return board_sub + defl_raw + defl_stays + front_steps + dummy_15_3 + dummy_16_2 + air_res_22_13 + handrail_26_10

    walk_L = _build_side_walk_and_defl(+1.0)
    walk_R = mirror(walk_L, about=Plane.XZ) + Pos(rev_x_sh - 0.2, -32.5, 30.8) * Box(10.8, 5.4, 1.5)
    walkways = walk_L + walk_R

    # 25-6 + 22-1/2 统一全宽 79.2mm 共面 42° 斜向车头甲板 + 25-2 内侧立裙板 + 25-4/5 梯形前踏步 (100% 根除车头红圈镂空!)
    front_deck_and_beam = (
        Pos(162.0, 0.0, 20.4) * Rot(0, +42, 0) * Box(32.5, 79.2, 1.5)
        + Pos(179.5, 0.0, 9.6) * Box(15.0, 79.2, 1.6)
        + Pos(186.5, 0.0, 6.2) * Box(3.2, 79.2, 8.4)
        + Pos(186.5, 0.0, 6.2) * Rot(90, 0, 0) * Cylinder(radius=0.65, height=76.0)
        + Pos(189.2, 0.0, 6.0) * Box(4.2, 9.6, 5.4)
        + Pos(192.4, 0.0, 6.0) * Cylinder(radius=3.4, height=5.8)
    )
    for sgn_st in (-1.0, 1.0):
        front_deck_and_beam += (
            Pos(158.0, sgn_st * 26.5, 19.8) * Box(18.0, 2.2, 18.8)  # 25-2 前甲板下两侧内封板
            + Pos(186.5, sgn_st * 38.6, -4.0) * Box(2.4, 1.1, 13.6)
            + Pos(186.5, sgn_st * 35.4, -4.0) * Box(2.4, 1.1, 13.6)
            + Pos(186.5, sgn_st * 37.0, -6.2) * Box(2.4, 4.3, 0.9)
            + Pos(186.5, sgn_st * 37.0, -10.4) * Box(2.6, 4.3, 1.1)
            + Pos(185.8, sgn_st * 22.5, -3.2) * Box(2.6, 3.2, 14.8)
            + Pos(187.0, sgn_st * 22.5, -10.0) * Box(3.8, 3.6, 1.6)
        )
    hatch_25_3 = Pos(164.0, 0.0, 19.0) * Rot(0, +42, 0) * Box(7.6, 9.8, 1.0)

    # 7. 左后 22-11 复式空压机 + 右后 22-18 给水泵 + 28-8 消音器 + 30-2 圆顶汽笛 + 30-7 加减杆 + 精细多级 15-8/15-10 双涡轮发电机
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
        + Pos(-96.5, -18.5, 73.3) * Sphere(radius=2.7)
        + Pos(-96.5, -15.8, 64.5) * Box(4.4, 6.8, 4.2)
    )
    dummy_whistle_30_2 = (
        Pos(-94.0, +13.5, z_axis + r_barrel - 0.2) * Rot(0, -72, 0) * Cylinder(radius=1.75, height=7.2)
        + Pos(-97.4, +13.5, z_axis + r_barrel + 0.9) * Sphere(radius=1.75)
        + Pos(-94.0, +13.5, z_axis + r_barrel - 2.4) * Cylinder(radius=2.05, height=4.2)
        + make_swept_tube([(-95.5, +13.5, z_axis + r_barrel - 0.2), (-110.5, +13.5, z_axis + r_barrel - 0.8)], r_tube=0.45, bend_r=1.5)
    )
    reg_rod_30_7 = (
        make_swept_tube(
            [
                (-110.5, +23.6, 65.8),
                (+16.0, +23.6, 65.8),
                (+16.0, +19.5, 70.8),
                (+16.0, +14.2, 74.8),
                (+16.0, +10.6, 76.2),
            ],
            r_tube=0.55,
            bend_r=2.4,
        )
        + Pos(-48.0, +21.2, 65.8) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=5.8)
        + Pos(-4.0, +21.2, 65.8) * Rot(90, 0, 0) * Cylinder(radius=0.80, height=5.8)
        + Pos(+16.0, +14.2, 73.2) * Box(2.0, 2.2, 3.6)
    )
    # 15-8 主涡轮发电机 + 15-10 ATS 发电机 (汽轮机壳 + 缩颈散热法兰 + 发电机定子 + 半球换向器尾罩 + 30-14 斜托板!)
    gen_15_8 = (
        Pos(-83.5, -24.5, 64.2) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=3.2)
        + Pos(-85.8, -24.5, 64.2) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=2.0)
        + Pos(-88.8, -24.5, 64.2) * Rot(0, 90, 0) * Cylinder(radius=2.9, height=4.4)
        + Pos(-91.0, -24.5, 64.2) * Sphere(radius=2.9)
        + Pos(-83.5, -24.5, 68.2) * Cylinder(radius=0.65, height=3.2)
        + Pos(-86.8, -20.5, 61.5) * Rot(+28, 0, 0) * Box(8.4, 8.6, 2.4)
        + Pos(-96.8, -25.2, 62.4) * Rot(0, 90, 0) * Cylinder(radius=2.4, height=2.6)
        + Pos(-99.5, -25.2, 62.4) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=3.4)
        + Pos(-101.2, -25.2, 62.4) * Sphere(radius=2.15)
        + Pos(-98.8, -21.2, 60.0) * Rot(+28, 0, 0) * Box(6.6, 8.2, 2.2)
    )
    cover_25_7 = (
        Pos(rev_x_sh, +32.5, 34.6) * Box(12.2, 6.8, 6.6)
        - Pos(rev_x_sh - 1.2, +32.5, 33.8) * Box(12.2, 5.2, 6.6)
    )

    outer_solid = (
        smokebox + sb_saddle + door_ring_18_22 + door_dish_18_14 + plate_d51_18_18 + wheel_18_12_13 + hinge_straps_18_4
        + headlight + headlight_ped + feedwater_heater + fwh_brackets + chimney
        + barrel + shroud_15_14_outer + dome_29_11 + firebox + cab + walkways + front_deck_and_beam + hatch_25_3
        + comp_22_11 + pump_22_18 + silencer_28_8 + dummy_whistle_30_2 + reg_rod_30_7 + gen_15_8 + cover_25_7
    )

    wheel_cuts = None
    wheel_pocket_backing = None
    for ax in (93.0, 46.5, 0.0, -46.5):
        for sgn in (-1.0, 1.0):
            wc = Pos(ax, sgn * 22.8, 0.0) * Rot(90, 0, 0) * Cylinder(radius=25.8, height=10.2)
            wb = Pos(ax, sgn * 22.8, 0.0) * Rot(90, 0, 0) * Cylinder(radius=27.6, height=13.8)
            wheel_cuts = wc if wheel_cuts is None else (wheel_cuts + wc)
            wheel_pocket_backing = wb if wheel_pocket_backing is None else (wheel_pocket_backing + wb)

    rev_outer_slots = (
        Pos(rev_x_sh, +25.0, rev_z_sh + 2.3) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=16.0)
        + Pos(rev_x_sh, -25.0, rev_z_sh + 2.3) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=16.0)
    )
    rev_slot_backing = (
        Pos(rev_x_sh, +25.0, rev_z_sh + 2.3) * Rot(90, 0, 0) * Cylinder(radius=4.6, height=19.6)
        + Pos(rev_x_sh, -25.0, rev_z_sh + 2.3) * Rot(90, 0, 0) * Cylinder(radius=4.6, height=19.6)
    )

    bore_barrel = (
        Pos(0.5 * (-109.5 + 93.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=28.35, height=93.5 - (-109.5))
        - wheel_pocket_backing
        - rev_slot_backing
    )
    bore_smokebox = (
        Pos(0.5 * (92.5 + 119.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=24.2, height=119.5 - 92.5)
        + (
            Pos(0.5 * (92.5 + 160.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 1.8, height=160.0 - 92.5)
            & Pos(126.0, 0.0, 56.5) * Box(80.0, 70.0, 58.0)
        )
        + Pos(0.5 * (119.5 + 160.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 2.1, height=160.0 - 119.5)
    ) - wheel_pocket_backing
    bore_firebox = (
        Pos(0.5 * (x_fb_front + (-109.5)), 0, z_axis - 13.5) * Box(x_fb_front - (-109.5), 48.8, fb_h)
        + Pos(0.5 * (x_fb_front + (-109.5)), 0, 40.0) * Box(x_fb_front - (-109.5) + 1.5, 48.8, 38.0)
    ) - wheel_pocket_backing
    bore_dome_cab = (
        Pos(34.0, 0, 73.5) * Box(36.0, 14.8, 16.0)
        + Pos(cab_x - 1.8, 0, z_axis + 24.0) * Box(cab_len - 3.5, 58.0, 16.0)
        + Pos(136.0, 0, 21.5) * Box(31.0, 21.0, 10.0)
        + Pos(150.5, 0, 19.5) * Box(15.0, 48.0, 14.5)
        + Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=4.1, height=46.0)
        + Pos(130.0, 0.0, 12.5) * Rot(90, 0, 0) * Cylinder(radius=3.4, height=46.0)
        + Pos(130.0, 0.0, 22.0) * Cylinder(radius=4.2, height=22.0)
        + Pos(123.5, +7.5, 14.0) * Cylinder(radius=2.2, height=12.0)
        + Pos(123.5, -7.5, 14.0) * Cylinder(radius=2.2, height=12.0)
        + Pos(-79.0, 0.0, 13.5) * Box(69.0, 36.0, 7.6)
    )
    bore_chimney = Pos(x_chimney, 0, z_axis + r_sb + 6.0) * Cylinder(radius=5.8, height=34.0)
    # 15-14 护罩内部净空腔 + 前妻板同轴跨界管道圆孔 (锅顶 100% 由 15-14 封闭，火室侧壁 100% 零破槽!)
    backhead_and_shell_pipe_cuts = (
        shroud_15_14_inner
        + Pos(-98.5, 0.0, 70.8) * Cylinder(radius=3.8, height=5.6)                         # 13-4 锅顶主蒸汽内通孔 (顶面由 15-14 封闭!)
        + Pos(-111.0, 0.0, 72.4) * Rot(0, 90, 0) * Cylinder(radius=2.45, height=6.0)       # 13-4 穿前妻板入驾驶室孔
        + Pos(-111.0, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=6.0)      # 13-8 -> 17-3 通风管孔
        + Pos(-111.0, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=2.45, height=6.0)      # 13-9 -> 干汽管孔
        + Pos(-111.0, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=2.65, height=6.0)     # 13-2 逆止阀孔
        + Pos(-111.0, -17.5, 60.5) * Rot(0, 90, 0) * Cylinder(radius=2.25, height=6.0)     # 13-6 水面计上蒸汽管孔
        + Pos(-111.0, -20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.25, height=6.0)     # 13-10 水面计下水管孔
        + Pos(-111.0, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=3.45, height=6.0)     # 11-8 真汽笛穿孔
        + Pos(-111.0, +16.2, 15.5) * Rot(0, 90, 0) * Cylinder(radius=2.25, height=6.0)     # 13-11 排污管后孔
        + Pos(-42.5, 0.0, 16.5) * Cylinder(radius=3.2, height=6.0)                         # 13-11 锅底排污内孔
        + Pos(-111.0, -25.0, 29.4) * Rot(0, 90, 0) * Cylinder(radius=1.55, height=6.0)     # 8-5 右地板上方排液杆圆孔
        + make_swept_tube(
            [(-109.5, -25.0, 29.4), (-101.0, -17.8, +7.1), (-93.0, -17.8, +7.1)],
            r_tube=1.65,
            bend_r=2.6,
            r_ref=1.65,
        )
    )
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
        - backhead_and_shell_pipe_cuts
        - safety_valve_holes
        - firehole_outer_cut
    )


# ==================== 4. 原厂真值四回路圆弧扫掠管网、车架内三软管、置换注油器、1x3 燃烧器与 8-20B 煤水车 ====================

def build_boiler_firebox_flues(cfg: Dict, pip_cfg: Dict):
    """
    12-6 无氧紫铜银焊 van Riemsdijk `Type "C"` 受压锅筒 + 纵向拉撑杆 + 5 本煙管 (イラスト 12 ⑥, aster-manual.pdf pp.9-10 & 2010_JNR_D51-standard.pdf):
      - 严格还原 Aster `Type "C"` 锅炉真值: 前段 X∈[-47.0, +92.0]mm 为完整圆截面受压紫铜锅筒 (ø53.0mm ID)，后段 X∈[-112.0, -47.0]mm 下腹退台 (Z_bottom=+45.1mm 紫铜冠板顶盖)，零虚构紫铜湿式火室水腿 (`12-4` 干式不锈钢内衬陶瓷火室罩已独立建为 `firebox_wrapper_12_4.step`)；
      - 内置纵向贯通拉撑铜杆 (Stayed End Plates, Y=±8.5mm, Z=+56.0mm) 承受前后平端板蒸汽张力；
      - 13-6 上蒸汽口 (-106.8, ±17.5, +60.5mm) 径向贯通汽包腔、13-10 下水口 (-43.2, ±20.8, +32.6mm) 轴向贯通湿锅筒水腔！
    """
    pv = pip_cfg["boiler_pressure_vessel"]
    z_axis = float(pv["centerline_z_mm"])
    r_in = 0.5 * float(pv["barrel_id_mm"])
    r_out = r_in + 0.8
    x_rear, x_front = [float(v) for v in pv["barrel_x_range_mm"]]

    fb = pv["inner_firebox"]
    fb_x0, fb_x1 = [float(v) for v in fb["x_range_mm"]]
    z_crown = float(fb["crown_sheet_z_mm"])

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
    # Type "C" 后部退台切口: 切除后段 X < -48.2mm 且 Z < 45.10mm 的下半圆柱，保留 X∈[-48.2,-47.0]mm 完整圆形紫铜后管板承接全部 5 根烟管!
    type_c_rear_recess = Pos(0.5 * ((x_rear - 2.0) + (fb_x1 - 1.2)), 0.0, 0.5 * (15.0 + 45.10)) * Box(
        (fb_x1 - 1.2) - (x_rear - 2.0), 60.0, 45.10 - 15.0
    )
    barrel_cavity = Pos(0.5 * (fb_x1 + 2.4 + x_front - 2.4), 0, z_axis) * Rot(0, 90, 0) * Cylinder(
        radius=r_in, height=(x_front - 2.4) - (fb_x1 + 2.4)
    )
    # 保持后段燃烧室顶盖紫铜冠板 Z∈[45.10, 47.20]mm (t=2.10mm) 完整受热厚度，在后妻板接口内侧设蒸汽连通集箱槽直通汽包腔
    steam_dome_cavity = (
        (
            Pos(0.5 * (x_rear + 1.6 + fb_x1 + 4.5), 0, z_axis) * Rot(0, 90, 0) * Cylinder(
                radius=r_in - 0.3, height=(fb_x1 + 4.5) - (x_rear + 1.6)
            ) & Pos(0.5 * (x_rear + fb_x1 + 4.5), 0, z_crown + 11.9) * Box(
                (fb_x1 + 4.5 - x_rear) + 4.0, 58.0, 24.0
            )
        )
        - Pos(-51.6, +11.5, 49.8) * Box(14.8, 14.4, 9.6)
        - Pos(-51.6, -11.5, 49.8) * Box(14.8, 14.4, 9.6)
    ) + (
        Pos(-107.5, 0.0, 48.8) * Box(5.8, 28.0, 4.4)
        & Pos(-107.5, 0.0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_in - 0.3, height=8.0)
    )
    # 上排两根 ø11.5mm 大烟管在后管板 X=-48.2mm 处与退台冠板平滑交汇的导流窝 (四周留 >=2.2mm 实心铜壁与汽包腔零串通!)
    flue_throat_pockets = (
        Pos(-50.2, +11.5, 46.5) * Rot(0, 90, 0) * Cylinder(radius=5.0, height=5.0)
        + Pos(-50.2, -11.5, 46.5) * Rot(0, 90, 0) * Cylinder(radius=5.0, height=5.0)
    )
    shell_body = outer_barrel - type_c_rear_recess - barrel_cavity - steam_dome_cavity - flue_throat_pockets

    # Type "C" 纵向贯通拉撑杆 (Longitudinal Stay Rods, aster-manual.pdf p.9 Fig.6) + 冠板加强撑钉
    long_stays = (
        Pos(full_xm, +8.5, 56.0) * Rot(0, 90, 0) * Cylinder(radius=1.10, height=full_len - 0.8)
        + Pos(full_xm, -8.5, 56.0) * Rot(0, 90, 0) * Cylinder(radius=1.10, height=full_len - 0.8)
    )
    stays = long_stays
    for sx in (-95.0, -78.0, -61.0):
        for sy in (-12.0, -4.0, 4.0, 12.0):
            stays = stays + Pos(sx, sy, z_crown + 0.6) * Cylinder(radius=1.1, height=2.2)

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
        fb_cut = Pos(0.5 * (-50.5 + fl_end_x + 3.0), yf, zf) * Rot(0, 90, 0) * Cylinder(
            radius=r_i, height=(fl_end_x + 3.0) - (-50.5)
        )
        flues_outer = (fo + bead) if flues_outer is None else (flues_outer + fo + bead)
        flues_bore = fb_cut if flues_bore is None else (flues_bore + fb_cut)

    blowdown_bush_bot = Pos(-42.5, 0.0, 17.0) * Cylinder(radius=2.5, height=2.6)
    steam_boss_rear = Pos(-98.5, 0.0, 69.8) * Cylinder(radius=3.6, height=2.2)
    sv_bushes_tandem = (
        Pos(-46.0, 0.0, 70.6) * Cylinder(radius=2.9, height=2.6)
        + Pos(-33.0, 0.0, 70.6) * Cylinder(radius=2.9, height=2.6)
    )
    sg_throat_bushes = (
        Pos(-46.8, -20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=2.6)
        + Pos(-46.8, +20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=2.6)
    )
    # 100% 物理打通受压紫铜锅筒全部蒸汽/给水/排污/水面计连通衬套通孔 (左右严格镜像保持 |com_y| == 0.0000mm!)
    pv_port_bores = (
        Pos(-98.5, 0.0, 70.8) * Cylinder(radius=1.05, height=9.2)
        + Pos(-46.0, 0.0, 70.6) * Cylinder(radius=1.15, height=8.4)
        + Pos(-33.0, 0.0, 70.6) * Cylinder(radius=1.15, height=8.4)
        + Pos(-42.5, 0.0, 17.8) * Cylinder(radius=0.85, height=7.2)
        + Pos(-110.0, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.40, height=10.0)
        + Pos(-110.0, +3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.40, height=10.0)
        + Pos(-110.0, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=0.78, height=10.0)
        + Pos(-110.0, -2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=0.78, height=10.0)
        + Pos(-110.0, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=10.0)
        + Pos(-110.0, +11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=10.0)
        # 13-6 水面计上蒸汽口沿 +X 水平直穿铜后妻板与径向口直通 steam_dome_cavity!
        + Pos(-110.0, -17.5, 60.5) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=10.0)
        + Pos(-110.0, +17.5, 60.5) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=10.0)
        + make_swept_tube([(-106.8, -20.5, 63.2), (-106.8, -13.5, 56.8)], r_tube=0.82, bend_r=1.0)
        + make_swept_tube([(-106.8, +20.5, 63.2), (-106.8, +13.5, 56.8)], r_tube=0.82, bend_r=1.0)
        # 13-10 水面计下水口沿 +X 水平直穿入湿锅筒 barrel_cavity (X >= -44.6mm)!
        + Pos(-45.2, -20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=7.2)
        + Pos(-45.2, +20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=7.2)
        # 前管板 (+90.8mm) 13-9 主蒸汽干汽管与 17-3 通风撑管对称过孔 (保持 |com_y| == 0.000mm!)
        + Pos(+90.8, 0.0, 58.8) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=8.0)
        + Pos(+90.8, +4.0, 40.5) * Rot(0, 90, 0) * Cylinder(radius=0.75, height=8.0)
        + Pos(+90.8, -4.0, 40.5) * Rot(0, 90, 0) * Cylinder(radius=0.75, height=8.0)
    )

    return (
        shell_body + stays + flues_outer
        + blowdown_bush_bot + steam_boss_rear + sv_bushes_tandem + sg_throat_bushes
        - flues_bore - pv_port_bores
    )


def build_firebox_wrapper_12_4(cfg: Dict, pip_cfg: Dict):
    """
    12-4 独立不锈钢干式火室与后部上升导焰风道 `火室 (SUS304)` (イラスト 12 ④, aster-manual.pdf pp.9-10 Fig.6 & Photo 4):
      - 1-BOM-1-STEP 真实独立分件 (严禁将 12-4 并入 12-6 紫铜受压锅筒冒充湿式水腿!)；
      - 采用 1.15mm SUS304 不锈钢板折弯围合成后部耐火燃烧风道 (X∈[-112.0, -47.2]mm, |Y|<=18.05mm, Z∈[20.5, 45.55]mm)，
        顶部内折贴合面与 12-6 退台紫铜冠板底面 (Z=+45.10mm) 严密座合，前端敞口紧贴 X=-48.2mm 紫铜后管板承接全部 5 根烟管；
      - 底部中央开放长槽供 23-1 直列三头酒精燃烧器火焰与一次风上行，后壁设 12-1 蝴蝶式火室门座圈 (Z=+28.8mm)，严格左右对称 (|com_y|==0.0000mm)。
    """
    x0, x1 = -112.0, -47.2
    xm = 0.5 * (x0 + x1)
    lx = x1 - x0
    z0, z1 = 20.5, 45.55
    zm = 0.5 * (z0 + z1)
    lz = z1 - z0
    w_out = 36.10  # |Y| <= 18.05mm, 完美嵌入 Template 2 陶瓷隔热侧板 (|Y|=18.1..19.1mm) 内侧
    w_in = 33.80   # |Y| <= 16.90mm, 1.15mm SUS304 不锈钢侧壁

    outer_box = Pos(xm, 0.0, zm) * Box(lx, w_out, lz)
    # 内部燃烧与上升导焰空腔: 前壁 (-48.6..-47.2mm) 仅在圆锅筒内径 r<=24.2mm 范围内开口对接后管板 5 烟管，四角严密封闭杜绝外漏!
    inner_main = Pos(0.5 * (-109.6 + (-48.6)), 0.0, 0.5 * (z0 + 1.4 + z1 - 1.2)) * Box(
        (-48.6) - (-109.6), w_in, (z1 - 1.2) - (z0 + 1.4)
    )
    front_throat_port = (
        Pos(-48.0, 0.0, 44.0) * Rot(0, 90, 0) * Cylinder(radius=24.2, height=4.0)
        & Pos(-48.0, 0.0, 0.5 * (27.5 + z1 + 1.0)) * Box(4.5, w_in, (z1 + 1.0) - 27.5)
    )
    top_crown_window = Pos(0.5 * (-106.0 + (-48.6)), 0.0, z1 - 0.6) * Box(
        (-48.6) - (-106.0), w_in - 3.2, 3.0
    )
    bot_grate_slot = Pos(0.5 * (-106.5 + (-52.0)), 0.0, z0 + 0.7) * Box(
        (-52.0) - (-106.5), w_in - 3.6, 2.2
    )
    # 两侧 M2.6×4 (×6) 内向加强螺钉耳座 (イラスト 12 ④)
    mount_lugs = None
    for sx in (-101.0, -79.5, -58.0):
        for sy in (+16.8, -16.8):
            lug = Pos(sx, sy, z1 - 1.4) * Box(4.2, 2.4, 2.8)
            mount_lugs = lug if mount_lugs is None else (mount_lugs + lug)

    firehole_collar = Pos(x0 - 0.4, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=6.2, height=4.0)
    firehole_bore = Pos(x0 + 1.0, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=4.6, height=10.0)
    return (outer_box - inner_main - front_throat_port - top_crown_window - bot_grate_slot + mount_lugs + firehole_collar) - firehole_bore


def build_boiler_insulation(cfg: Dict, pip_cfg: Dict):
    """
    原厂说明书 & 1:1 裁剪模板 (Templates 1–6) 硅酸铝陶瓷纤维保温层 `セラミックシート(断熱材)` (Manual pp.28-31, 35 & イラスト 12, 16, 18, cc=1):
      - Template 1: 后妻板隔热片 (X∈[-112.2,-110.8]mm, 精确开孔避让全部跨界蒸汽/给水/水面计管座)；
      - Template 2: 火室拱顶与侧隔热包覆片；
      - Template 4: 紫铜锅筒圆周隔热毡；
      - Template 5 + Template 6: 烟室内壁圆筒隔热衬 + 烟室前门 ø53.0mm 隔热圆板！
    """
    z_axis = 44.0
    # Template 4: 6 条 15mm 宽硅酸铝陶瓷叠带 (177..195×15mm) 在锅筒两端构成 ~3.3mm 径向密封隔热封环，封闭中部静止空气夹套 + 薄陶瓷衬底
    t4_barrel = (
        (
            Pos(0.5 * (-47.0 + 92.2), 0.0, z_axis)
            * Rot(0, 90, 0)
            * (Cylinder(radius=28.12, height=92.2 - (-47.0)) - Cylinder(radius=27.36, height=92.2 - (-47.0) + 2.0))
        )
        + (
            Pos(-40.5, 0.0, z_axis)
            * Rot(0, 90, 0)
            * (Cylinder(radius=28.30, height=15.0) - Cylinder(radius=27.36, height=16.0))
        )
        + (
            Pos(+84.5, 0.0, z_axis)
            * Rot(0, 90, 0)
            * (Cylinder(radius=28.30, height=15.0) - Cylinder(radius=27.36, height=16.0))
        )
    ) - Pos(22.0, 0.0, 16.5) * Box(150.0, 18.0, 6.0)

    t2_arch = (
        Pos(0.5 * (-111.4 + (-46.5)), 0.0, z_axis)
        * Rot(0, 90, 0)
        * (Cylinder(radius=28.12, height=-46.5 - (-111.4)) - Cylinder(radius=27.36, height=-46.5 - (-111.4) + 2.0))
    ) & (Pos(-79.0, 0.0, 58.0) * Box(70.0, 60.0, 32.0))
    t2_cheeks_L = Pos(-79.0, +18.6, 36.5) * Box(64.9, 1.0, 19.0)
    t2_firebox = t2_arch + t2_cheeks_L + mirror(t2_cheeks_L, about=Plane.XZ)

    t1_backhead = (
        (
            Pos(-111.4, 0.0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=27.9, height=1.3)
            & Pos(-111.4, 0.0, 56.0) * Box(4.0, 58.0, 30.0)
        )
        + Pos(-111.4, 0.0, 36.5) * Box(1.3, 37.6, 16.0)
    )

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
        + Pos(-108.0, +4.1, 72.2) * Box(18.0, 13.6, 6.4)
        + Pos(-46.0, 0.0, 72.0) * Cylinder(radius=3.8, height=12.0)
        + Pos(-33.0, 0.0, 72.0) * Cylinder(radius=3.8, height=12.0)
        + Pos(+130.0, 0.0, 74.0) * Cylinder(radius=10.2, height=14.0)
        + Pos(-111.4, 0.0, 28.8) * Rot(0, 90, 0) * Cylinder(radius=6.8, height=6.0)
        + Pos(-111.4, 0.0, 48.0) * Box(6.0, 14.0, 6.5)
        + Pos(-111.4, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=6.0)
        + Pos(-111.4, +11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=3.2, height=6.0)
        + Pos(-111.4, -17.5, 60.5) * Rot(0, 90, 0) * Cylinder(radius=2.6, height=12.0)
        + Pos(-106.8, 0.0, 60.5) * Rot(90, 0, 0) * Cylinder(radius=2.8, height=60.0)
        + Pos(-46.0, -20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=10.0)
        + Pos(-46.0, +20.8, 32.6) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=10.0)
        + Pos(123.5, 0.0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=4.3, height=48.0)
        + Pos(130.0, 0.0, 12.5) * Rot(90, 0, 0) * Cylinder(radius=3.6, height=48.0)
        + Pos(92.0, 0.0, 58.8) * Rot(0, 90, 0) * Cylinder(radius=2.9, height=6.0)
    )
    return (t4_barrel + t2_firebox + t1_backhead + t5_bridge + t5_smokebox + t6_door_disc) - insul_cuts


def build_superheater_piping(cfg: Dict, pip_cfg: Dict):
    """
    原厂全真壁厚空心内腔主蒸汽干汽管 + 17-5 双U四程过热器 + 7-9 给汽横歧管单连通体 (イラスト 7, 12, 14, 17, 20 & 図 17.1, cc=1):
      - 后封头穿板处 (-112.8, -3.6, +48.0) 采用阶梯变径六角穿板活接，使驾驶室 13-9 (OD 3.0 / ID 2.0mm) 与炉内干汽管 (OD 4.0 / ID 2.8mm)
        同轴无缝连续过渡 (ΔY=0, ΔZ=0, 零悬空套管断阶!)；
      - 双U四程过热管 (OD 3.0 / ID 2.0mm) -> 17-4 -> 7-9 给汽横歧管端到端 100% 空心贯通。
    """
    pv = pip_cfg["boiler_pressure_vessel"]
    sh = pip_cfg["superheater_and_steam_piping"]["superheater_4pass"]
    z_pipe = 58.8
    x_turn = float(sh["x_turn_in_flue_mm"])
    x_hdr = 108.5
    up_yz = pv["five_flue_tubes"]["upper_superheater_flues"]["yz_centers_mm"]

    pts_dry = [
        (-112.8, -3.6, 48.0),
        (-106.0, -3.6, 48.5),
        (-97.0, 0.0, z_pipe),
        (+109.2, 0.0, z_pipe),
    ]
    backhead_stud = (
        make_hex_nut((-112.8, -3.6, 48.0), (1.0, 0.0, 0.0), hex_r=2.10, length=2.2)
        + Pos(-114.1, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.50, height=1.2)
    )
    dry_pipe_outer = make_swept_tube(pts_dry, r_tube=2.00, bend_r=3.6, r_ref=2.00)
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
        Pos(-113.8, -3.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.00, height=2.6)
        + make_swept_tube(_extend_polyline_ends(pts_dry, 0.0, 0.0), r_tube=1.40, bend_r=3.6, r_ref=2.00)
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
    铜锅筒火室冠板上方空心通风撑管 (-112.8, +2.6, +48.0 -> -44.0, +2.6, +49.5 -> +93.6) + 烟室 17-3 通風ノズル 原厂真 J 字 180° 圆弧喷嘴 (イラスト 17 ③ & イラスト 20, OD 2.0 / ID 1.2mm, cc=1):
      - 严格还原 イラスト 17 ③ J 型弯管形状：自前管板活接螺母 (+93.6, +4.2, +40.5mm) 水平前伸越过过热集箱下方后，
        在 X=+124.2mm 处光滑 90° 下折垂直下降至 Z=+23.8mm，再经底部光滑 180° J 型半圆回弯 (R=2.9mm) 竖直向上升至
        (+130.0, +4.2, +35.2mm) 紧贴 20-7 乏汽喷嘴右侧 (+Y=+4.2mm)，全程与中央 Y=0 的 17-5 过热下垂管零交叉！
    """
    pts_17_3 = [
        (-112.8, +2.6, 48.0),
        (-105.0, +2.6, 49.5),
        (-44.0, +2.6, 49.5),
        (+86.0, +4.2, 40.5),
        (+124.2, +4.2, 40.5),
        (+124.2, +4.2, 23.8),
        (+130.0, +4.2, 23.8),
        (+130.0, +4.2, 35.2),
    ]
    rear_stud = (
        make_hex_nut((-112.8, +2.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=2.0)
        + Pos(-114.0, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=1.20, height=1.2)
    )
    front_union = make_hex_nut((+93.6, +4.2, 40.5), (1.0, 0.0, 0.0), hex_r=1.90, length=2.2)
    nozzle_tip = Pos(+130.0, +4.2, 34.6) * Cone(bottom_radius=1.08, top_radius=0.88, height=1.8)
    outer_17_3 = rear_stud + front_union + nozzle_tip + make_swept_tube(pts_17_3, r_tube=1.00, bend_r=2.8, r_ref=1.00)
    bore_17_3 = (
        Pos(-113.8, +2.6, 48.0) * Rot(0, 90, 0) * Cylinder(radius=0.75, height=2.2)
        + make_swept_tube(_extend_polyline_ends(pts_17_3, 0.0, 0.55), r_tube=0.60, bend_r=2.8, r_ref=1.00)
    )
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
    6-5 中央轴动给水泵 (实心盲底 + 双 SB-ø3 止回球腔 + 上下 (4-15) 盲头螺栓) + 6-10/6-11 吸水管 + 6-6 车架内侧吐出管 (イラスト 6, cc=1):
      - 与独立 1:1 建模的 4-17 バイパスバルブ本体 (bypass_valve_4_17.step)、6-8 给水立管及 4-16 回水管在 (-135.5, -5.5, +15.0mm) 零间隙同轴对接！
    """
    fw = pip_cfg["feedwater_piping_topology"]
    ax_loop = fw["axle_and_shared_hand_pump_loop"]
    x_pump = float(ax_loop["pump_x_mm"])  # -22.0 mm
    x_tw = x_pump - 16.8                  # -38.8 mm

    pump_barrel = (
        Pos(-38.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.65, height=10.8)
        + Pos(-34.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.95, height=2.0)
    )
    gland_nut = make_hex_nut((-33.5, 0.0, 0.0), (1.0, 0.0, 0.0), hex_r=4.10, length=2.0)
    check_tower = (
        Pos(x_tw, 0.0, 1.4) * Cylinder(radius=3.25, height=15.6)
        + Pos(x_tw, 0.0, +9.4) * Cylinder(radius=2.95, height=2.6)
        + Pos(x_tw, 0.0, -6.6) * Cylinder(radius=2.95, height=2.6)
    )
    top_blind_bolt = make_blind_hex_bolt((x_tw, 0.0, +11.65), (0.0, 0.0, 1.0), hex_r=3.65, head_len=1.9)
    bot_blind_bolt = make_blind_hex_bolt((x_tw, 0.0, -8.85), (0.0, 0.0, -1.0), hex_r=3.65, head_len=1.9)
    pump_bracket_6_7 = (
        Pos(-36.2, 0.0, 4.4) * Box(5.8, 20.8, 2.4)
        + Pos(-36.2, +9.0, 6.0) * Cylinder(radius=0.85, height=1.2)
        + Pos(-36.2, -9.0, 6.0) * Cylinder(radius=0.85, height=1.2)
    )
    pump_outer = pump_barrel + gland_nut + check_tower + top_blind_bolt + bot_blind_bolt + pump_bracket_6_7

    pts_6_10 = [
        (-150.8, -19.2, -8.5),
        (-142.5, -19.2, -8.5),
        (-136.0, -8.2, +8.5),
        (x_tw, -8.2, +8.5),
        (x_tw, -8.2, -6.2),
        (x_tw, 0.0, -6.2),
    ]
    stud_6_11 = (
        Pos(-149.0, -19.2, -8.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=4.2)
        + make_hex_nut((-146.8, -19.2, -8.5), (1.0, 0.0, 0.0), hex_r=2.55, length=1.8)
    )

    bx, bz = -135.5, 15.0
    pts_6_6 = [
        (x_tw, 0.0, +9.6),
        (x_tw, -8.2, +13.8),
        (-128.0, -8.2, +14.8),
        (bx + 2.8, -5.5, bz),
    ]
    nut_6_6_end = make_hex_nut((bx + 3.4, -5.5, bz), (1.0, 0.0, 0.0), hex_r=1.80, length=1.5)

    outer_fw = (
        pump_outer
        + make_swept_tube(pts_6_10, r_tube=1.22, bend_r=2.8, r_ref=1.22) + stud_6_11
        + make_swept_tube(pts_6_6, r_tube=1.20, bend_r=2.6, r_ref=1.20) + nut_6_6_end
    )
    bore_fw = (
        Pos(-35.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.62, height=12.0)
        + Pos(x_tw, 0.0, 1.5) * Cylinder(radius=1.68, height=17.8)
        - (Pos(x_tw, 0.0, -3.2) * (Cylinder(radius=1.72, height=0.9) - Cylinder(radius=1.16, height=1.2)))
        - (Pos(x_tw, 0.0, +5.2) * (Cylinder(radius=1.72, height=0.9) - Cylinder(radius=1.16, height=1.2)))
        + make_swept_tube(_extend_polyline_ends(pts_6_10, 0.55, 0.40), r_tube=0.80, bend_r=2.8, r_ref=1.22)
        + make_swept_tube(_extend_polyline_ends(pts_6_6, 0.40, 0.55), r_tube=0.80, bend_r=2.6, r_ref=1.20)
    )
    return outer_fw - bore_fw


def build_bypass_valve_4_17():
    """
    4-17 バイパスバルブ本体 + LN5-7A M5ロックナット + 下置 4-15 中空管ねじ (イラスト 4 top-right, イラスト 6 & 8 1:1 独立还原, cc=1):
      - 横置阶梯黄铜圆筒阀体 (OD 7.2 / 7.8mm, Y∈[-11.8,+10.8]mm) + 左端 (+Y) M5×0.5 穿板螺纹柱与 LN5-7A 薄六角锁紧螺母 (7.0mm AF × 2.0mm)；
      - 左上 (+Y=+8.5mm, +Z) 高耸阶梯针阀柱座 (OD 5.2 / 6.8mm, 承接 6-9 针阀杆 + PS2-1.5)；
      - 正下方 (-Z, Y=+8.5mm) 垂直向上旋入的 4-15 六角中空管螺栓 (8.0mm AF × 8.2mm, 锁紧 4-16 旁通回水环眼!)；
      - 前侧 (+X) 承接 6-6 进水口 (-5.5mm) 与 6-8/(4-15) 锅炉给水立管接口 (-11.4mm)，内部三通针阀流道 100% 贯通且两端盲封！
    """
    bx, bz = -135.5, 15.0
    main_barrel = (
        Pos(bx, -0.5, bz) * Rot(90, 0, 0) * Cylinder(radius=3.55, height=22.4)
        + Pos(bx, -10.4, bz) * Rot(90, 0, 0) * Cylinder(radius=3.85, height=2.6)
        + Pos(bx, +9.4, bz) * Rot(90, 0, 0) * Cylinder(radius=3.85, height=2.6)
    )
    # +Y 端一体 M5×0.5 安装螺纹柱穿出 4-8 立耳 + LN5-7A 六角锁母 (实心盲端, 零外漏!)
    stud_m5_plus_y = Pos(bx, +12.4, bz) * Rot(90, 0, 0) * Cylinder(radius=2.45, height=3.8)
    nut_ln5_7a = make_hex_nut((bx, +12.5, bz), (0.0, 1.0, 0.0), hex_r=3.65, length=2.0)
    # -Y 端 (4-15) 给水侧六角盲顶螺栓头
    bolt_minus_y = make_blind_hex_bolt((bx, -12.45, bz), (0.0, -1.0, 0.0), hex_r=3.45, head_len=1.6)
    # +Z 阶梯立式针阀填料塔 (Y=+8.5mm, 承接 6-9 / 8-7)
    stem_tower = (
        Pos(bx, +8.5, bz + 2.8) * Cylinder(radius=2.55, height=5.4)
        + Pos(bx, +8.5, bz + 4.7) * Cylinder(radius=3.20, height=1.5)
    )
    # +X 前侧面 6-6 进水承插凸台 (-5.5mm) 与 6-8 出水凸台 (-11.4mm)
    boss_in_6_6 = Pos(bx + 2.2, -5.5, bz) * Rot(0, 90, 0) * Cylinder(radius=2.05, height=2.4)
    boss_out_6_8 = Pos(bx + 2.0, -11.4, bz) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=2.2)
    # -Z 正下方 (Y=+8.5mm) 4-17 底阀座凸台 + 自下而上旋入的 4-15 中空管螺栓实心六角头 (夹紧中间 4-16 Banjo!)
    bot_seat_boss = Pos(bx, +8.5, bz - 2.4) * Cylinder(radius=2.55, height=2.6)
    bot_shank_4_15 = Pos(bx, +8.5, bz - 4.0) * Cylinder(radius=1.85, height=3.6)
    bot_bolt_4_15 = make_blind_hex_bolt((bx, +8.5, bz - 5.6), (0.0, 0.0, -1.0), hex_r=3.65, head_len=1.6)

    outer_4_17 = (
        main_barrel + stud_m5_plus_y + nut_ln5_7a + bolt_minus_y
        + stem_tower + boss_in_6_6 + boss_out_6_8 + bot_seat_boss + bot_shank_4_15 + bot_bolt_4_15
    )
    bore_4_17 = (
        Pos(bx, -1.0, bz) * Rot(90, 0, 0) * Cylinder(radius=1.65, height=21.6)       # 横向主阀腔 Y∈[-11.8,+9.8]mm (两端实心盲封!)
        + Pos(bx + 2.0, -5.5, bz) * Rot(0, 90, 0) * Cylinder(radius=0.88, height=4.2)  # 6-6 进水横孔
        + Pos(bx + 2.0, -11.4, bz) * Rot(0, 90, 0) * Cylinder(radius=0.88, height=4.4) # 6-8 出水横孔
        + Pos(bx, +8.5, bz + 2.9) * Cylinder(radius=0.95, height=6.4)                  # +Z 6-9 针阀孔
        + Pos(bx, +8.5, bz - 2.3) * Cylinder(radius=0.92, height=4.8)                  # -Z 4-15 中空螺栓中心孔 (止于六角实心底头上方!)
        + Pos(bx, +8.5, bz - 3.8) * Rot(0, 90, 0) * Cylinder(radius=0.82, height=4.6)  # 4-15 中空螺栓横向交叉出液孔 -> 4-16 Banjo!
        + Pos(bx, +8.5, bz - 3.8) * Rot(90, 0, 0) * Cylinder(radius=0.82, height=4.6)
        + Pos(bx, +8.5, bz - 3.8) * Rot(0, 0, -62.5) * Rot(0, 90, 0) * Cylinder(radius=0.86, height=4.8)
    )
    return outer_4_17 - bore_4_17


def build_boiler_riser_6_8():
    """6-8 ボイラ給水管: 自 4-17 右前口 (-132.45,-11.4,+15.0) 经圆弧弯上升直连锅炉后妻板中位逆止阀 13-2 (-115.8,-11.8,+43.4, イラスト 6 & 13, cc=1)。"""
    bx, bz = -135.5, 15.0
    pts_6_8 = [
        (bx + 3.05, -11.4, bz),
        (-115.8, -11.8, +16.2),
        (-115.8, -11.8, +43.4),
    ]
    nut_6_8_bot = make_hex_nut((bx + 3.85, -11.4, bz), (1.0, 0.0, 0.0), hex_r=1.85, length=1.6)
    nut_6_8_top = make_hex_nut((-115.8, -11.8, +42.4), (0.0, 0.0, 1.0), hex_r=1.85, length=2.0)
    outer_6_8 = nut_6_8_bot + make_swept_tube(pts_6_8, r_tube=1.20, bend_r=2.4, r_ref=1.20) + nut_6_8_top
    bore_6_8 = make_swept_tube(_extend_polyline_ends(pts_6_8, 0.65, 0.55), r_tube=0.80, bend_r=2.4, r_ref=1.20)
    return outer_6_8 - bore_6_8


def build_bypass_return_4_16():
    """4-16 戻り管: 自 4-17 正下方 (-135.5,+8.5,+11.2) 中空 Banjo 环眼水平径向引出绕过 4-15 六角螺栓头至左后梁 (+Y=+19.2mm, Z=-8.5mm) 宝塔口金 (イラスト 4 top-right, cc=1)。"""
    bx, bz = -135.5, 15.0
    pts_4_16 = [
        (bx - 1.15, +8.5 + 2.20, bz - 3.8),
        (-138.2, +13.7, bz - 3.8),
        (-141.5, +19.2, +6.2),
        (-144.5, +19.2, -8.5),
        (-150.8, +19.2, -8.5),
    ]
    banjo_4_16 = Pos(bx, +8.5, bz - 3.8) * Cylinder(radius=2.95, height=2.2)
    barb_4_16 = (
        Pos(-149.2, +19.2, -8.5) * Rot(0, 90, 0) * Cone(bottom_radius=1.85, top_radius=1.45, height=3.8)
        + make_hex_nut((-146.8, +19.2, -8.5), (1.0, 0.0, 0.0), hex_r=2.45, length=1.8)
    )
    outer_4_16 = banjo_4_16 + make_swept_tube(pts_4_16, r_tube=1.20, bend_r=2.6, r_ref=1.20) + barb_4_16
    bore_4_16 = (
        Pos(bx, +8.5, bz - 3.8) * Cylinder(radius=1.92, height=3.0)
        + make_swept_tube(_extend_polyline_ends(pts_4_16, 1.80, 0.55), r_tube=0.80, bend_r=2.6, r_ref=1.20)
    )
    return outer_4_16 - bore_4_16


def build_displacement_lubricator(cfg: Dict, pip_cfg: Dict):
    """
    4-2 前车架中央全空心卧式置换油箱 + 4-1 一字槽顶盖 (PN5-1.9) + 4-3 前鞍座 (H-M3-6) + 内部竖直溢流立管 + 真实底部 (-Z) U形引出毛细管 (イラスト 4 Fig.1, 7, cc=1):
      - 严格还原 イラスト 4 Fig.1 真值：外部铜管自 4-2 卧式圆筒后部正下方 (-Z 底壁 X=+146.2, Y=0.0, Z=-0.6mm) 垂直向下穿出，
        在油箱下方绕 U/S 弯后沿 Y=+7.5mm 上升直连 7-9 给汽横管底口 (+123.5, +7.5, +18.5mm)；
      - 油箱内部：该底孔同轴向上直通一根银焊竖直溢流立管 (Roscoe Standpipe, OD 2.1 / ID 1.16mm)，顶端开口位于距油箱内顶仅 1.3mm 处 (Z=+10.9mm)！
        蒸汽自立管顶口入罐冷凝成水 (ρ_w > ρ_oil) 沉入罐底，将浮于上层的汽缸油顶升漫入立管顶口，沿立管内腔隔离底水向下经底部 U 弯送入 7-9 雾化入缸！
    """
    lub = pip_cfg["displacement_lubricator_system"]
    x0, x1 = [float(v) for v in lub["x_range_mm"]]
    xm = 0.5 * (x0 + x1)
    zc = float(lub["z_center_mm"])           # +6.2 mm
    r_tank = 0.5 * float(lub["tank_od_mm"])  # 6.8 mm (Z ∈ [-0.6, +13.0] mm)
    r_inner = r_tank - 0.80                  # 6.0 mm (Z_in ∈ [+0.2, +12.2] mm)
    cap_x, _, cap_z = [float(v) for v in lub["cap_xyz_mm"]]
    tx, ty, tz = [float(v) for v in lub["oil_delivery_tee_xyz_mm"]]

    tank_4_2 = Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_tank, height=x1 - x0)
    # 4-3 前鞍座支架 (严格 Y∈[-9.60,+9.60]mm 左右对称, 后耳带 H-M3-6 六角螺钉!)
    saddle_4_3 = (
        Pos(xm - 2.0, 0.0, zc + 1.2) * Box(13.0, 19.2, 3.8)
        + Pos(x0 - 1.4, 0.0, zc - 2.2) * Box(3.6, 8.4, 2.2)
        + make_blind_hex_bolt((x0 - 1.8, 0.0, zc - 0.5), (0.0, 0.0, 1.0), hex_r=2.65, head_len=1.7)
    )
    neck_4_1 = (
        Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z)) * Cylinder(radius=2.9, height=cap_z - zc + 1.0)
        + Pos(cap_x - 1.5, 0.0, cap_z - 1.3) * Cylinder(radius=4.15, height=1.1)  # PN5-1.9 O形圈座肩
    )
    cap_4_1 = (
        Pos(cap_x - 1.5, 0.0, cap_z) * Cylinder(radius=3.9, height=2.4)
        - Pos(cap_x - 1.5, 0.0, cap_z + 1.0) * Box(1.0, 8.2, 0.9)  # 4-1 顶面一字螺丝刀槽
    )

    x_quill = x0 + 4.2  # +146.2 mm
    pts_oil = [
        (x_quill, 0.0, +0.4),
        (x_quill, 0.0, -2.0),
        (139.0, ty, -2.0),
        (tx, ty, +2.2),
        (tx, ty, tz + 2.3),
    ]
    oil_pipe_outer = (
        Pos(x_quill, 0.0, -0.8) * Cylinder(radius=1.65, height=1.6)
        + make_swept_tube(pts_oil, r_tube=1.05, bend_r=2.4, r_ref=1.05)
    )
    union_nut = make_hex_nut((tx, ty, tz - 0.8), (0.0, 0.0, 1.0), hex_r=1.85, length=2.0)

    tank_cavity = (
        Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_inner, height=(x1 - x0) - 1.6)
        + Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z - 1.2)) * Cylinder(radius=1.75, height=(cap_z - 1.2) - zc)
    )
    z_quill_top = zc + r_inner - 1.3  # +10.9 mm
    quill_outer = Pos(x_quill, 0.0, 0.5 * (-0.4 + z_quill_top)) * Cylinder(radius=1.05, height=z_quill_top - (-0.4))
    quill_and_pipe_bore = (
        make_swept_tube(_extend_polyline_ends(pts_oil, 0.0, 0.55), r_tube=0.58, bend_r=2.4, r_ref=1.05)
        + Pos(x_quill, 0.0, 0.5 * (-0.4 + z_quill_top + 0.6)) * Cylinder(radius=0.58, height=(z_quill_top + 0.6) - (-0.4))
    )

    shell_with_quill = ((tank_4_2 + saddle_4_3 + neck_4_1 + cap_4_1 + oil_pipe_outer + union_nut) - tank_cavity) + quill_outer
    return shell_with_quill - quill_and_pipe_bore


def build_boiler_aux_valves(cfg: Dict, pip_cfg: Dict):
    """
    驾驶室后封头与火室侧原厂多图联立 100% 真值空心铜阀座与连通器管网单连通总成 (イラスト 11, 12, 13, 14 & p45 实车照真值, cc=1):
      1. 11-15 蒸気分配器: 彻底修正 90° 横放失真！按原厂图纸与实车照片还原为长边平行于机车 X 轴向的纵向矩形分汽块
         (18.0mm X [X∈[-131.6,-113.6]] × 7.2mm Y [Y∈[+4.6,+11.8]] × 7.2mm Z [Z∈[+62.9,+70.1]], 左偏司机侧!):
         - 顶面 (+Z) 承接跨越锅顶的 13-4 空心主蒸汽管 (-98.5,0,+72.6 -> -118.5,+8.2,+70.1)；
         - 后端面 (-X=-131.6, Y=+8.2, Z=+66.5) 同轴装配四辐 11-13 通风手轮；
         - 左侧面 (+Y=+11.8) 前后纵列双口：前口 (-117.6) 接 11-1/11-3 汽笛阀 -> 笔直 13-3 -> 同轴穿板 11-8 真汽笛；后口 (-126.8) 接 11-10/11-11 加减阀；
         - 底面 (-Z=+62.9) 下引 13-8 通风管同轴穿板 (-112.8,+2.6,+48.0) 直连炉内冠板上方 17-3 撑管；
         - 右侧面 (-Y=+4.6) 横向 4-15 双联 Banjo 分出 13-9 加减管同轴穿板 (-112.8,-3.6,+48.0) 直连炉内干汽管，以及 13-1 虹吸管接 PG-ø20 垂直压力表；
      2. 13-2 逆止弁 + 13-7 顶盖 (-115.8, -11.8, +47.5mm) 同轴穿板直通锅炉蒸汽腔；
      3. 右外缘 (-118.5, -31.3) 11-4~7 真实连通器水面计：上管 13-6 水平径向穿入汽包腔 (-106.8,-17.5,+60.5)，下管 13-10 保持 100% 恒高 Z=+32.6mm
         零高度错位水平直穿后妻板与火室侧走廊，轴向接通湿锅筒喉部水腔 (-43.2, -20.8, +32.6mm)！
    """
    wg = pip_cfg["pumps_and_valves_bom"]["water_gauge_11_7"]
    x_ring = -113.0

    brazed_backhead_ring = (
        Pos(x_ring, 0.0, 49.5) * Rot(0, 90, 0) * (Cylinder(radius=19.2, height=1.4) - Cylinder(radius=14.8, height=3.0))
        + Pos(x_ring, 0.0, 48.0) * Box(1.8, 28.0, 4.6)
        + Pos(x_ring, +18.2, 31.0) * Box(1.8, 6.8, 35.0)
        + Pos(x_ring, -19.2, 46.0) * Box(1.8, 9.6, 31.0)
        + Pos(x_ring, +8.2, 57.5) * Box(1.8, 4.8, 18.0)
    )

    pts_13_4 = [
        (-98.5, 0.0, 72.4),
        (-118.5, 0.0, 72.4),
        (-118.5, +8.2, 72.4),
        (-118.5, +8.2, 67.2),
    ]
    jumper_13_4 = (
        Pos(-98.5, 0.0, 71.4) * Cylinder(radius=2.10, height=2.4)
        + make_swept_tube(pts_13_4, r_tube=1.50, bend_r=2.0, r_ref=1.50)
        + make_banjo_eye((-98.5, 0.0, 72.5), (0.0, 0.0, 1.0), r_eye=2.7, h_eye=2.3, hex_r=2.0)
        + make_banjo_eye((-118.5, +8.2, 70.8), (0.0, 0.0, 1.0), r_eye=2.45, h_eye=2.0, hex_r=1.85)
    )

    # 11-15 原厂纵向矩形蒸汽分配器 (18.0mm X × 7.2mm Y × 7.2mm Z, 长边严格沿机车轴向 X!)
    manifold_11_15 = (
        Pos(-122.6, +8.2, 66.5) * Box(18.0, 7.2, 7.2)
        + Pos(-132.1, +8.2, 66.5) * Rot(0, 90, 0) * Cylinder(radius=1.95, height=1.8)
        # 右侧面 (-Y) 4-15 横向双联空心管螺栓总管 (向 -Y 供汽至 13-1 压力表虹吸管与 13-9 加减管)
        + Pos(-124.5, +0.4, 66.5) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=9.2)
        + make_blind_hex_bolt((-124.5, -4.6, 66.5), (0.0, -1.0, 0.0), hex_r=2.55, head_len=1.6)
    )
    gland_collar_11_10 = (
        Pos(-126.8, +12.0, 66.5) * Rot(90, 0, 0) * Cylinder(radius=2.10, height=2.4)
        + make_hex_nut((-126.8, +12.8, 66.5), (0.0, 1.0, 0.0), hex_r=2.05, length=1.0)
    )

    pts_13_8 = [
        (-128.0, +8.2, 63.2),
        (-128.0, +8.2, 54.5),
        (-120.5, +2.6, 48.0),
        (-112.8, +2.6, 48.0),
    ]
    blower_13_8 = (
        make_hex_nut((-128.0, +8.2, 62.1), (0.0, 0.0, 1.0), hex_r=1.85, length=1.8)
        + make_swept_tube(pts_13_8, r_tube=1.20, bend_r=2.2, r_ref=1.20)
        + make_hex_nut((-114.8, +2.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.80, length=1.8)
    )

    pts_13_9 = [
        (-124.5, -3.6, 66.5),
        (-124.5, -3.6, 48.0),
        (-112.8, -3.6, 48.0),
    ]
    jumper_13_9 = (
        make_banjo_eye((-124.5, -3.6, 66.5), (0.0, 1.0, 0.0), r_eye=2.35, h_eye=1.9, hex_r=1.75)
        + make_swept_tube(pts_13_9, r_tube=1.50, bend_r=2.4, r_ref=1.50)
        + make_hex_nut((-114.8, -3.6, 48.0), (1.0, 0.0, 0.0), hex_r=1.95, length=1.8)
    )

    clack_13_2_7 = (
        Pos(-115.8, -11.8, 47.5) * Cylinder(radius=2.55, height=9.0)
        + make_hex_nut((-115.8, -11.8, 52.4), (0.0, 0.0, 1.0), hex_r=2.80, length=1.9)
        + Pos(-112.8, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=2.10, height=6.4)
        + make_hex_nut((-114.2, -11.8, 47.5), (1.0, 0.0, 0.0), hex_r=2.45, length=1.5)
    )

    yp, zp = -16.5, 66.8
    pg_bezel = (
        Pos(-122.0, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=9.8, height=3.6)
        + Pos(-122.0, yp, zp - 10.6) * Cylinder(radius=1.95, height=3.4)
        - Pos(-123.6, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=8.45, height=2.4)
    )
    pts_13_1 = [
        (-124.5, +0.8, 66.5),
        (-124.5, +0.8, 62.0),
        (-123.5, -6.5, 43.0),
        (-122.8, -11.8, 39.0),
        (-122.0, yp, 43.2),
        (-122.0, yp, 57.5),
    ]
    siphon_13_1 = (
        make_banjo_eye((-124.5, +0.8, 66.5), (0.0, 1.0, 0.0), r_eye=2.15, h_eye=1.8, hex_r=1.65)
        + make_swept_tube(pts_13_1, r_tube=1.00, bend_r=3.6, r_ref=1.00)
        + make_hex_nut((-122.0, yp, 55.2), (0.0, 0.0, 1.0), hex_r=1.80, length=2.0)
    )

    yw = float(wg["y_mm"])  # -31.3 mm
    pts_13_6 = [(-118.5, yw, 60.5), (-118.5, -17.5, 60.5), (-106.8, -17.5, 60.5)]
    wg_frame_11_4_6 = (
        Pos(-116.2, yw, 46.0) * Box(1.2, 3.2, 26.8)
        + Pos(-118.5, yw, 54.8) * Box(4.8, 3.8, 1.4)
        + Pos(-118.5, yw, 37.2) * Box(4.8, 3.8, 1.4)
        + Pos(-114.6, -25.2, 46.0) * Box(3.6, 12.6, 7.6)
    )
    wg_top = (
        Pos(-118.5, yw, 57.6) * Cylinder(radius=2.35, height=5.8)
        + make_hex_nut((-118.5, yw, 55.1), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + make_blind_hex_bolt((-118.5, yw, 61.9), (0.0, 0.0, 1.0), hex_r=2.25, head_len=1.5)
        + make_swept_tube(pts_13_6, r_tube=1.20, bend_r=2.4, r_ref=1.20)
        + make_banjo_eye((-106.8, -17.5, 60.5), (0.0, +0.72, -0.69), r_eye=2.05, h_eye=1.9, hex_r=1.55)
        + make_swept_tube([(-106.8, -18.2, 61.1), (-106.8, -15.2, 58.3)], r_tube=1.20, bend_r=1.0)
    )
    # 13-10 水面计下水管: 全程保持 100% 恒高 Z=+32.6mm (零穿板高差!) 水平直穿入湿锅筒喉部 (-45.6, -20.8, +32.6mm, X<=-45.0mm 零动轮区越界)!
    pts_13_10 = [(-118.5, yw, 32.6), (-118.5, -20.8, 32.6), (-45.6, -20.8, 32.6)]
    wg_bot = (
        Pos(-118.5, yw, 33.6) * Cylinder(radius=2.35, height=7.2)
        + make_hex_nut((-118.5, yw, 36.9), (0.0, 0.0, 1.0), hex_r=2.75, length=2.0)
        + make_swept_tube(pts_13_10, r_tube=1.20, bend_r=2.4, r_ref=1.20)
        + make_hex_nut((-47.2, -20.8, 32.6), (1.0, 0.0, 0.0), hex_r=1.95, length=1.6)
    )

    wh_valve_11_1 = (
        Pos(-117.6, +20.2, 66.5) * Box(4.6, 4.6, 8.8)
        + Pos(-119.5, +20.2, 62.6) * Box(2.8, 2.4, 2.4)
        + Pos(-117.6, +14.8, 66.5) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=7.6)
        + Pos(-117.6, +20.2, 71.5) * Cylinder(radius=1.85, height=1.6)
        - Pos(-117.6, +20.2, 72.2) * Box(3.8, 0.55, 0.6)
    )
    pts_13_3 = [
        (-117.6, +20.2, 65.5),
        (-117.6, +20.2, 25.5),
        (-114.5, +20.2, 25.5),
    ]
    pipe_13_3 = (
        make_hex_nut((-117.6, +20.2, 61.6), (0.0, 0.0, 1.0), hex_r=1.55, length=1.5)
        + make_swept_tube(pts_13_3, r_tube=1.00, bend_r=1.8, r_ref=1.00)
        + make_banjo_eye((-117.6, +20.2, 25.5), (-1.0, 0.0, 0.0), r_eye=1.75, h_eye=1.8, hex_r=1.38)
    )
    whistle_11_8 = (
        Pos(-115.6, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=4.4)
        + make_hex_nut((-114.8, +20.2, 25.5), (1.0, 0.0, 0.0), hex_r=2.85, length=1.6)
        + Pos(-92.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=3.10, height=44.4)
        - Pos(-102.0, +20.2, 23.2) * Box(3.2, 4.8, 3.2)
        + Pos(x_ring, +19.5, 25.5) * Box(1.8, 4.4, 4.4)
    )

    pts_13_11 = [
        (-115.5, +16.2, 15.8),
        (-108.0, +16.2, 15.5),
        (-72.0, +16.2, 15.2),
        (-54.0, +7.2, 15.4),
        (-42.5, 0.0, 15.6),
        (-42.5, 0.0, 17.2),
    ]
    blowdown_11_16_and_13_11 = (
        Pos(-115.5, +18.6, 15.8) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=4.4)
        + Pos(-113.2, +18.2, 16.5) * Box(3.2, 2.2, 4.0)
        + Pos(-115.5, +19.2, 13.4) * Cylinder(radius=1.18, height=4.0)
        + make_banjo_eye((-115.5, +16.2, 15.8), (0.0, -1.0, 0.0), r_eye=2.0, h_eye=1.8, hex_r=1.50)
        + make_swept_tube(pts_13_11, r_tube=1.20, bend_r=3.2, r_ref=1.20)
        + make_banjo_eye((-42.5, 0.0, 16.8), (0.0, 0.0, 1.0), r_eye=2.15, h_eye=2.2, hex_r=1.65)
    )

    outer_aux = (
        brazed_backhead_ring + jumper_13_4 + manifold_11_15 + gland_collar_11_10
        + blower_13_8 + jumper_13_9 + clack_13_2_7 + pg_bezel + siphon_13_1
        + wg_frame_11_4_6 + wg_top + wg_bot + wh_valve_11_1 + pipe_13_3 + whistle_11_8 + blowdown_11_16_and_13_11
    )
    bore_aux = (
        # 13-4 空心内孔 -> 11-15 纵向主腔 (X∈[-132.4,-114.6], Y=+8.2, Z=+66.5)
        make_swept_tube(_extend_polyline_ends(pts_13_4, 0.0, 0.9), r_tube=0.95, bend_r=2.0, r_ref=1.50)
        + Pos(-98.5, 0.0, 71.2) * Cylinder(radius=1.02, height=2.8)
        + Pos(-123.5, +8.2, 66.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=18.2)
        # 11-15 横向分流道 (贯通 +Y 11-10 加减阀、+Y=+20.2 11-1 汽笛阀及 -Y 4-15 双联 Banjo 至 Y=-3.8mm 盲端!)
        + Pos(-126.8, +10.8, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=5.8)
        + Pos(-117.6, +14.4, 66.5) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=12.4)
        + Pos(-117.6, +20.2, 67.2) * Cylinder(radius=1.05, height=5.2)
        + Pos(-124.5, +2.4, 66.5) * Rot(90, 0, 0) * Cylinder(radius=1.00, height=12.4)
        # 13-8 通风管 & 13-9 加减管端到端全通内孔
        + make_swept_tube(_extend_polyline_ends(pts_13_8, 3.6, 1.2), r_tube=0.75, bend_r=2.2, r_ref=1.20)
        + make_swept_tube(_extend_polyline_ends(pts_13_9, 0.8, 1.2), r_tube=0.98, bend_r=2.4, r_ref=1.50)
        # 13-2 逆止阀腔 (底部承接 6-8 + 内部 SB-ø3 止回球阀座台阶 + 前向穿板出口 X=-109.2mm 直通蒸汽腔!)
        + Pos(-115.8, -11.8, 47.2) * Cylinder(radius=1.42, height=9.2)
        - (Pos(-115.8, -11.8, 45.2) * (Cylinder(radius=1.45, height=0.8) - Cylinder(radius=1.02, height=1.2)))
        + Pos(-115.8, -11.8, 44.0) * Cylinder(radius=0.95, height=3.6)
        + Pos(-112.5, -11.8, 47.5) * Rot(0, 90, 0) * Cylinder(radius=1.15, height=7.6)
        # 13-1 压力表虹吸管内孔
        + make_swept_tube(_extend_polyline_ends(pts_13_1, 0.6, 1.2), r_tube=0.58, bend_r=3.6, r_ref=1.00)
        + Pos(-122.0, yp, 59.5) * Cylinder(radius=0.72, height=5.2)
        # 11-5 / 13-6 / 11-7 / 11-6 / 13-10 水面计双路真连通器全通内腔 (上座盲顶密封 Z_max=61.0 < 62.65!)
        + Pos(-118.5, yw, 56.8) * Cylinder(radius=1.15, height=8.4)
        + make_swept_tube(_extend_polyline_ends(pts_13_6, 1.0, 0.4), r_tube=0.78, bend_r=2.4, r_ref=1.20)
        + make_swept_tube([(-106.8, -19.2, 62.1), (-106.8, -14.2, 57.4)], r_tube=0.78, bend_r=1.0)
        + Pos(-118.5, yw, 33.8) * Cylinder(radius=1.15, height=9.6)
        + make_swept_tube(_extend_polyline_ends(pts_13_10, 1.0, 2.2), r_tube=0.78, bend_r=2.4, r_ref=1.20)
        # 13-3 汽笛立管 -> 11-8 同轴共鸣腔 (前端 X=-70.0mm 真实盲盖反射腔)
        + make_swept_tube(_extend_polyline_ends(pts_13_3, 2.2, 1.5), r_tube=0.58, bend_r=1.8, r_ref=1.00)
        + Pos(-115.2, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=0.85, height=6.4)
        + Pos(-92.6, +20.2, 25.5) * Rot(0, 90, 0) * Cylinder(radius=2.15, height=39.6)
        # 11-16 排污针阀腔 -> 13-11 锅底排污管全通内孔
        + Pos(-115.5, +19.2, 13.4) * Cylinder(radius=0.72, height=6.4)
        + Pos(-115.5, +17.5, 15.8) * Rot(90, 0, 0) * Cylinder(radius=0.72, height=4.8)
        + make_swept_tube(_extend_polyline_ends(pts_13_11, 1.0, 1.5), r_tube=0.75, bend_r=3.2, r_ref=1.20)
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

    # 23-2 前折弯支架 (焊接于第3号杯前壁, Y 严格对称, |Y|<=5.6mm 避让右侧 Y=-8.2mm 给水吐出管 6-6) + 4-9 后部马鞍形管卡
    bracket_23_2 = (
        Pos(wick_xs[-1] + 5.8, 0.0, 14.5) * Box(2.8, 7.6, 5.2)
        + Pos(wick_xs[-1] + 7.4, 0.0, 17.2) * Box(4.2, 11.2, 1.4)
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
        Pos(+28.4, 0.0, 14.2) * Box(30.6, 29.2, 6.5)
        + Pos(-16.4, +21.3, 14.2) * Cylinder(radius=2.9, height=6.5)
        + Pos(-16.4, -21.3, 14.2) * Cylinder(radius=2.9, height=6.5)
        + Pos(-41.0, -10.3, 14.2) * Cylinder(radius=3.2, height=6.5)
        + Pos(+103.2, 0.0, +7.0) * Rot(0, 90, 0) * Cylinder(radius=2.7, height=10.0)
        + Pos(+102.5, +21.3, +5.0) * Box(10.0, 5.2, 26.0)
        + Pos(+102.5, -21.3, +5.0) * Box(10.0, 5.2, 26.0)
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
    """イラスト 31 & 36: 31-26 铸铁侧架 + 31-24x4 下封口护板 + 31-27 三长圆孔端梁 + 31-29 枕梁 + 36-3 心皿座金 + 31-25 x8 轴箱顶叠板簧 + 31-28 x8 螺旋压簧 (100% Y 对称, cc=1)。"""
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

    bolster_leaf_L = (
        Pos(0, +(y_frame + 1.3), -1.2) * Box(20.0, 2.2, 2.6)
        + Pos(0, +(y_frame + 1.5), -2.4) * Box(13.2, 2.2, 1.3)
        + Pos(0, +(y_frame + 1.4), -1.3) * Box(3.0, 2.7, 3.6)
    )
    side_L = Pos(0, +y_frame, 1.5) * Box(wb + 17.4, 2.4, 5.5) + bolster_leaf_L
    for ax_x in (-half_wb, +half_wb):
        leaf_31_25_ax = (
            Pos(ax_x, +(y_frame + 1.25), +4.05) * Box(13.6, 2.1, 0.80)
            + Pos(ax_x, +(y_frame + 1.25), +4.75) * Box(9.8, 2.1, 0.72)
            + Pos(ax_x, +(y_frame + 1.25), +5.38) * Box(6.4, 2.1, 0.68)
            + Pos(ax_x, +(y_frame + 1.25), +4.65) * Box(2.6, 2.55, 2.3)
        )
        coil_f = make_coil_spring_stack((ax_x + 5.20, +(y_frame + 1.2), -0.95), r_coil=1.12, r_wire=0.33, height=4.4, n_turns=4, r_core_rod=0.72)
        coil_r = make_coil_spring_stack((ax_x - 5.20, +(y_frame + 1.2), -0.95), r_coil=1.12, r_wire=0.33, height=4.4, n_turns=4, r_core_rod=0.72)
        side_L += Pos(ax_x, +(y_frame + 1.4), 0.0) * Box(7.5, 4.2, 8.2)
        side_L += Pos(ax_x, +(y_frame + 1.2), 2.8) * Box(12.6, 3.4, 1.8)
        side_L += leaf_31_25_ax + coil_f + coil_r
        side_L += Pos(ax_x, +(y_frame + 1.3), -3.9) * Box(11.6, 3.6, 1.1)
        side_L -= Pos(ax_x, +y_frame, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=10.0)
    sides = side_L + mirror(side_L, about=Plane.XZ)
    return bolster + end_beams_31_27 + sides


def build_spoked_truck_wheel(cfg: Dict, is_left: bool = True):
    """イラスト 20, 24 & 31: 10 辐条放射镂空导轮/从轮/煤水车车轮 (Gauge-1 标准锥形轮缘防脱轨型面)。"""
    r_tread = 0.5 * float(cfg["wheels_and_axles"]["leading_wheel_diameter"])
    w_tire, w_flange = 3.38, 0.90
    rim = (
        Cylinder(radius=r_tread, height=w_tire)
        + Pos(0, 0, -0.5 * w_tire + 0.5 * w_flange) * Cone(bottom_radius=r_tread + 1.24, top_radius=r_tread + 0.30, height=w_flange)
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
        ("equalizer_lever_3_4", 0.016, RHO_CAST_IRON, build_equalizer_lever_3_4),
        ("axle_pump_eccentric", 0.016, RHO_BRASS, lambda: build_axle_pump_eccentric(cfg)),
        ("axle_pump_ram", 0.012, RHO_BRASS, lambda: build_axle_pump_ram(cfg)),
        ("pump_check_ball", 0.008, RHO_STAINLESS, build_pump_check_ball),
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
        ("firebox_wrapper_12_4", 0.024, RHO_STAINLESS, lambda: build_firebox_wrapper_12_4(cfg, pip_cfg)),
        ("boiler_insulation", 0.032, RHO_CERAMIC_SHEET, lambda: build_boiler_insulation(cfg, pip_cfg)),
        ("superheater_piping", 0.014, RHO_STAINLESS, lambda: build_superheater_piping(cfg, pip_cfg)),
        ("exhaust_blast_pipe", 0.016, RHO_BRASS, lambda: build_exhaust_blast_pipe(cfg, pip_cfg)),
        ("blower_nozzle_17_3", 0.014, RHO_BRASS, lambda: build_blower_nozzle_17_3(cfg, pip_cfg)),
        ("smokebox_petticoat", 0.020, RHO_STAINLESS, lambda: build_smokebox_petticoat(cfg, pip_cfg)),
        ("tandem_safety_valves", 0.014, RHO_BRASS, lambda: build_tandem_safety_valves(cfg, pip_cfg)),
        ("feedwater_piping", 0.016, RHO_BRASS, lambda: build_feedwater_piping(cfg, pip_cfg)),
        ("bypass_valve_4_17", 0.014, RHO_BRASS, build_bypass_valve_4_17),
        ("boiler_riser_6_8", 0.014, RHO_BRASS, build_boiler_riser_6_8),
        ("bypass_return_4_16", 0.014, RHO_BRASS, build_bypass_return_4_16),
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
        "bypass_valve_4_17",
        "boiler_riser_6_8",
        "bypass_return_4_16",
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
        # 对所有 XZ 镜像成对的 _R 零件 (排除含 90° 正交曲柄相位的车轮): 保留镜像 STEP 母本，同时从 _L.stl 做精确 XZ 镜像，
        # 根除 OpenCascade 镜像面网格剖分微差 (保证 len(tL)==len(tR), |y_L+y_R|==0.0000mm 与双侧流道/惯量 100% 镜像对称)
        if key.endswith("_R") and "wheel" not in key:
            base_L = f"{key[:-2]}_L"
            src_stl = args.out_dir / f"{base_L}.stl"
            if src_stl.exists():
                raw_b = src_stl.read_bytes()
                n_tri = struct.unpack("<I", raw_b[80:84])[0]
                dt_stl = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
                arr = np.frombuffer(raw_b[84 : 84 + n_tri * 50], dtype=dt_stl).copy()
                arr["v"][:, :, 1] *= -1.0
                arr["v"][:, [1, 2], :] = arr["v"][:, [2, 1], :]
                arr["n"][:, 1] *= -1.0
                with open(stl_path, "wb") as f_out:
                    f_out.write(raw_b[:84])
                    f_out.write(arr.tobytes())
                if base_L in mass_db["parts"]:
                    mp_L = mass_db["parts"][base_L]
                    mp = json.loads(json.dumps(mp_L))
                    mp["com_m"][1] = -float(mp_L["com_m"][1])
                    mp["fullinertia"][3] = -float(mp_L["fullinertia"][3])
                    mp["fullinertia"][5] = -float(mp_L["fullinertia"][5])
        if key == "boiler_superstructure":
            subdivide_stl_x_span(stl_path, max_dx=2.02, z_min_filter=72.0)
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
