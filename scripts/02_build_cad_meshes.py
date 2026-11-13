#!/usr/bin/env python3
"""
scripts/02_build_cad_meshes.py
Aster D51 全套 18 件参数化 CAD 建模 + OpenCascade 精确物理质量/偏心质心/3x3惯量张量积分器。
严格遵循 GEMINI.md 第 5 节契约：
  - 黄铜 (BS: 气缸、滑阀、锅筒配件、动轮轴箱、月牙滑块): 8500 kg/m³
  - 不锈钢 (SUS: 主/耦合连杆、华氏连杆、逆转横轴、阀芯棒): 7930 kg/m³
  - 铸铁/钢 (FC/S45C: Boxpok 动轮、导从轮、煤水车转向架与轴箱): 7350 kg/m³
输出:
  - models/meshes/*.stl (18 个高精度网格)
  - configs/d51_mass_props.json (SI 单位制: kg, m, kg·m²)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from build123d import (
    Box,
    Cylinder,
    Plane,
    PolarLocations,
    Pos,
    Rot,
    SlotOverall,
    Sphere,
    export_stl,
    extrude,
    mirror,
)

# 原厂金属材质密度表 (kg/m³)
RHO_BRASS = 8500.0        # 黄铜 BS (1-6 滑阀, 5-9 轴箱, 2-3 滑块等)
RHO_STAINLESS = 7930.0    # 不锈钢 SUS (连杆系, 4-12 逆转横轴等)
RHO_CAST_IRON = 7350.0    # 铸铁/碳钢 (Boxpok 动轮, 台车轮, 转向架等)
RHO_COMPOSITE_VALVE = 8250.0   # 1-6 黄铜滑阀 + 1-22 不锈钢阀芯棒复合体
RHO_COMPOSITE_PISTON = 8200.0  # 1-28 黄铜活塞 + SUS 活塞杆 + 1-31 十字头复合体


def compute_occ_mass_props(solid: Any, density_kg_m3: float) -> Dict[str, Any]:
    """
    调用底层 OpenCascade BRepGProp 对三维实体做精确体积积分 (建模单位 mm -> 物理单位 SI m/kg)。
      - volume_mm3 * 1e-9 * density -> mass (kg)
      - MatrixOfInertia (关于质心 CoM, 单位 mm^5) * 1e-15 * density -> I_com (kg·m²)
    返回 MuJoCo fullinertia 所需的 6 分量 [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] 及 3x3 矩阵。
    """
    props = GProp_GProps()
    wrapped = solid.wrapped if hasattr(solid, "wrapped") else solid
    BRepGProp.VolumeProperties_s(wrapped, props)

    vol_mm3 = float(props.Mass())
    com = props.CentreOfMass()
    mat = props.MatrixOfInertia()  # OpenCascade 返回关于 CentreOfMass 的惯量张量 (ρ=1, mm^5)

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


# ==================== 1. 轮系与连杆建模 ====================

def build_boxpok_wheel(cfg: Dict, phase_deg: float, is_left: bool):
    """JNR D51 Boxpok 箱型动轮 (イラスト 5): 含多孔箱型轮心与 180° 反侧月牙配重块 (产生真实偏心质心)。"""
    d_tread = float(cfg["wheels_and_axles"]["driving_wheel_diameter"])
    r_tread = 0.5 * d_tread
    r_flange = r_tread + 1.4
    r_rim_inner = r_tread - 2.4
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    w_tire, w_flange, w_web = 3.8, 1.0, 2.4

    tire = Cylinder(radius=r_tread, height=w_tire)
    flange = Pos(0, 0, -0.5 * w_tire + 0.5 * w_flange) * Cylinder(radius=r_flange, height=w_flange)
    wheel = tire + flange - Cylinder(radius=r_rim_inner, height=w_tire + 2.0)

    web = Cylinder(radius=r_rim_inner + 0.2, height=w_web)
    hole_angles = [42, 75, 108, 252, 285, 318]
    r_hole_ring = 0.54 * (r_rim_inner + 4.5)
    for ang_deg in hole_angles:
        ang_rad = math.radians(ang_deg)
        hx, hy = r_hole_ring * math.cos(ang_rad), r_hole_ring * math.sin(ang_rad)
        web -= Pos(hx, hy, 0) * Cylinder(radius=2.3, height=w_web + 2.0)
        hx2, hy2 = (r_rim_inner - 2.6) * math.cos(ang_rad), (r_rim_inner - 2.6) * math.sin(ang_rad)
        web -= Pos(hx2, hy2, 0) * Cylinder(radius=1.15, height=w_web + 2.0)
    wheel += web

    # 180° 反侧月牙形平衡配重块 (实心加厚铸铁块，使动轮质心向曲柄反侧偏移)
    cw_outer = Cylinder(radius=r_rim_inner + 0.1, height=w_tire - 0.2)
    cw_cutter_arc = Pos(7.5, 0, 0) * Cylinder(radius=r_rim_inner - 1.5, height=w_tire + 2.0)
    cw_half_mask = Pos(r_flange, 0, 0) * Box(2.0 * r_flange + 5.0, 2.0 * r_flange + 5.0, w_tire + 2.0)
    wheel += cw_outer - cw_cutter_arc - cw_half_mask

    boss_z = 0.5 * w_tire
    center_hub = Cylinder(radius=4.6, height=w_tire + 0.6)
    crank_pad = Pos(0, 0, -0.5 * w_web) * extrude(
        Pos(0.5 * r_crank, 0) * SlotOverall(width=r_crank + 7.4, height=7.0),
        amount=boss_z + 0.5 * w_web + 0.5,
    )
    crank_pin = Pos(r_crank, 0, boss_z + 1.8) * Cylinder(radius=1.8, height=3.6)
    wheel = wheel + center_hub + crank_pad + crank_pin - Cylinder(radius=2.8, height=w_tire + 10.0)

    wheel_mj = Rot(-90, 0, 0) * wheel
    if not is_left:
        wheel_mj = mirror(wheel_mj, about=Plane.XZ)
    if abs(phase_deg) > 1e-6:
        wheel_mj = Rot(0, -phase_deg, 0) * wheel_mj
    return wheel_mj


def build_main_rod(cfg: Dict):
    """7-11 / 7-16 主连杆 (SUS 不锈钢)，大头原点 (0,0,0)，小头 (+L_main,0,0)。"""
    L_main = float(cfg["cylinder_and_rods"]["main_rod_length"])
    d_small = float(cfg["cylinder_and_rods"]["crosshead_pin_dia"])
    thickness_y, boss_thick_y = 2.6, 3.6
    h_big, h_small, h_shank = 6.8, 5.0, 3.6

    big_end = Box(8.4, boss_thick_y, h_big)
    big_oil_cup = Pos(0, 0, 0.5 * h_big + 0.7) * Cylinder(radius=1.3, height=1.4)
    small_end = Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * h_small, height=boss_thick_y)
    shank = Pos(0.5 * L_main, 0, 0) * Box(L_main - 2.0, thickness_y, h_shank)
    rod = big_end + big_oil_cup + small_end + shank

    flute_len = L_main - 14.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * L_main, sgn * (0.5 * thickness_y - 0.35), 0) * Box(flute_len, 0.8, h_shank - 1.3)

    rod -= Rot(90, 0, 0) * Cylinder(radius=1.8, height=boss_thick_y + 4.0)
    rod -= Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * d_small, height=boss_thick_y + 4.0)
    return rod


def build_coupling_rod(cfg: Dict):
    """7-12 / 7-15 四轴耦合连杆 (SUS 不锈钢)，原点对齐第 3 主动轮曲柄销 (0,0,0)。"""
    ax = cfg["wheels_and_axles"]["axle_x"]
    xs = [float(ax["drive_4"]), float(ax["drive_3"]), float(ax["drive_2"]), float(ax["drive_1"])]
    x_min, x_max = min(xs), max(xs)
    shank_t, boss_t, shank_h, boss_r = 2.4, 3.4, 3.4, 3.4

    rod = Pos(0.5 * (x_min + x_max), 0, 0) * Box(x_max - x_min, shank_t, shank_h)
    for idx_i, x_pos in enumerate(xs):
        r_b = 4.1 if idx_i == 1 else boss_r
        rod += Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=r_b, height=boss_t)
        rod += Pos(x_pos, 0, r_b + 0.6) * Cylinder(radius=1.1, height=1.3)

    hinge_knuckle_x = float(ax["drive_2"]) - 7.5
    rod += Pos(hinge_knuckle_x, 0, 0) * Box(4.2, boss_t - 0.2, shank_h + 1.4)

    for i in range(3):
        seg_mid = 0.5 * (xs[i] + xs[i + 1])
        seg_len = (xs[i + 1] - xs[i]) - 12.0
        if seg_len > 6.0:
            for sgn in (-1.0, 1.0):
                rod -= Pos(seg_mid, sgn * (0.5 * shank_t - 0.35), 0) * Box(seg_len, 0.8, shank_h - 1.2)

    for x_pos in xs:
        rod -= Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.7, height=boss_t + 4.0)
    rod -= Pos(hinge_knuckle_x, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.9, height=boss_t + 4.0)
    return rod


# ==================== 2. 补齐的 6 类细分运动部件建模 (イラスト 1, 2, 4, 5, 31) ====================

def build_piston_crosshead(cfg: Dict, is_left: bool = True):
    """
    1-28 黄铜活塞 + 活塞杆 + 1-31/1-35 单导轨十字头总成 (イラスト 1)。
    局部原点 (0, 0, 0) 严格对齐十字头主销孔 (7-13, ø3mm):
      - 活塞盘中心位于 (+p_head_dx, 0, 0)
      - 顶部抱轨滑靴包裹 Z = +6.2mm 处的单导轨滑杆 (1-33)
      - 底部下垂合并杆耳轴孔位于 (0, +y_drop, -crosshead_drop)
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    bore = float(rc["bore"])                       # 15.0 mm
    L_main = float(rc["main_rod_length"])          # 92.0 mm
    cyl_cx = float(rc["cylinder_center_x"])        # 126.0 mm
    drop_z = float(vg["crosshead_drop"])           # 8.5 mm

    p_head_dx = cyl_cx - L_main                    # 34.0 mm (十字头销至活塞盘中心距离)
    sgn = 1.0 if is_left else -1.0

    # 1. ø15mm 黄铜活塞体 (含 PR-15 活塞环槽特征) + ø3mm 不锈钢活塞杆
    piston_disc = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore - 0.1, height=4.6)
    ring_groove = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * (
        Cylinder(radius=0.5 * bore + 0.5, height=1.6) - Cylinder(radius=0.5 * bore - 0.9, height=2.0)
    )
    piston_disc -= ring_groove
    piston_rod = Pos(0.5 * p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.5, height=p_head_dx)

    # 2. 1-31 十字头叉形主体 + 上方抱合单导轨的滑靴 (1-32 クロスヘッドシュー)
    ch_hub = Box(9.0, 5.8, 7.2)
    top_shoe = Pos(0, 0, 6.0) * Box(13.5, 6.2, 5.2) - Pos(0, 0, 6.2) * Box(15.0, 4.4, 3.4)

    # 3. 下垂先导短连杆耳轴臂 (连接 1-24 ユニオンリンク)
    drop_arm = Pos(0, sgn * 1.8, -0.5 * drop_z) * Box(4.2, 2.4, drop_z + 2.5)
    drop_boss = Pos(0, sgn * 3.5, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)

    part = piston_disc + piston_rod + ch_hub + top_shoe + drop_arm + drop_boss
    # 钻出十字头主销孔 (ø3.0mm) 与下垂耳孔 (ø2.0mm)
    part -= Rot(90, 0, 0) * Cylinder(radius=1.5, height=10.0)
    part -= Pos(0, sgn * 3.5, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.0, height=6.0)
    return part


def build_slide_valve_assembly(cfg: Dict, is_left: bool = True):
    """
    1-6 黄铜平面 D 型滑阀 + 1-5 阀框 + 1-22 贯穿阀芯棒 + 1-25 阀杆十字头叉 (イラスト 1 & 20-1)。
    局部原点 (0, 0, 0) 位于后端阀杆叉头销孔 (s_valve_clevis)，左侧滑阀中心向内偏移 (+v_dx, -3.5mm, 0)，
    右侧通过 Plane.XZ 镜像使底部排气内腔凹槽 (EP) 保持朝下贴合气口镜面，并精确计算镜像惯量积。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    # 估算滑阀中心至后部合并杆上铰点 (s_valve_clevis) 的轴向距离 (~15.5mm)
    v_dx = float(rc["cylinder_center_x"]) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))

    # 1-25 后端合并杆连接叉头 (原点 0,0,0)
    clevis = Box(6.0, 3.8, 4.6) - Box(3.6, 1.8, 6.0)
    clevis -= Rot(90, 0, 0) * Cylinder(radius=1.0, height=6.0)

    # 1-22 贯穿不锈钢阀芯棒 (ø2.0mm) + 20-1 前导杆延伸
    rod_len = v_dx + 20.0
    stem = Pos(0.5 * rod_len - 2.0, -3.5, 0) * Rot(0, 90, 0) * Cylinder(radius=1.0, height=rod_len)
    cross_arm = Pos(1.5, -1.75, 0) * Box(2.6, 5.5, 3.2)

    # 1-6 黄铜平面滑阀块 + 1-5 驱动外框 (掏出底部排气内腔凹槽 EP)
    valve_block = Pos(v_dx, -3.5, -0.5) * Box(10.2, 9.6, 5.2)
    exhaust_cavity = Pos(v_dx, -3.5, -2.2) * Box(4.8, 6.6, 2.4)
    valve_block -= exhaust_cavity

    assembly = clevis + cross_arm + stem + valve_block
    return assembly if is_left else mirror(assembly, about=Plane.XZ)


def build_die_block(cfg: Dict):
    """2-3 月牙槽矩形黄铜滑块 + 2-2 销钉 (イラスト 2)，尺寸约 4.6 x 2.2 x 3.6 mm。"""
    block = Box(4.6, 2.2, 3.6)
    pin_boss = Rot(90, 0, 0) * Cylinder(radius=1.4, height=4.2)
    hole = Rot(90, 0, 0) * Cylinder(radius=0.8, height=6.0)
    return block + pin_boss - hole


def build_reverser_shaft(cfg: Dict):
    """4-12 逆转器横轴 + 2-11/2-18 左右双提升曲柄臂 (イラスト 2 & 4)。"""
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5   # 37.5 mm
    arm_len_x = (float(vg["trunnion_x"]) - float(vg["radius_rod_back_ext"])) - float(vg["reverser_shaft_x"])

    # 横跨主车架的不锈钢主轴 (ø3.2mm) + 右侧逆转连杆立臂
    main_bar = Rot(90, 0, 0) * Cylinder(radius=1.6, height=2.0 * y_vg + 4.0)
    reach_lever = Pos(0, -18.0, 5.5) * Box(3.6, 2.2, 12.0)

    arms = None
    for sgn in (-1.0, 1.0):
        hub = Pos(0, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=3.0, height=2.8)
        arm = Pos(0.5 * arm_len_x, sgn * y_vg, 0) * Box(abs(arm_len_x) + 3.0, 2.0, 3.2)
        tip_hole = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=0.9, height=5.0)
        sub = hub + arm - tip_hole
        arms = sub if arms is None else (arms + sub)

    return main_bar + reach_lever + arms


def build_driving_axlebox(cfg: Dict):
    """5-9 动轮左右一对黄铜轴箱体与下托板组 (イラスト 5)，局部原点位于车轴中心 (0,0,0)。"""
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) - 6.5  # 车架侧板处轴箱横向中心 ±16.0mm
    pair = None
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0.5) * Box(10.5, 5.2, 11.5)
        flange = Pos(0, sgn * y_box, 0.5) * Box(12.5, 2.2, 13.0)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=8.0)
        one = blk + flange - bore
        pair = one if pair is None else (pair + one)
    return pair


def build_tender_axlebox(cfg: Dict):
    """31-22 / 31-23 煤水车单轴左右一对轴箱导框滑块组 (イラスト 31)。"""
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2
    pair = None
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0) * Box(7.0, 3.8, 7.6)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=1.9, height=6.0)
        one = blk - bore
        pair = one if pair is None else (pair + one)
    return pair


# ==================== 3. 华氏月牙链、气缸、锅炉与煤水车 ====================

def build_expansion_link(cfg: Dict):
    vg = cfg["walschaerts_valve_gear"]
    R_arc = float(vg["radius_rod_length"])
    r_tail = float(vg["exp_link_tail_radius"])
    slot_half = float(vg["die_slot_half_travel"])
    thick_y, slot_w, body_w = 3.0, 2.4, 5.8
    z_top, z_bot = slot_half + 2.8, -(slot_half + 2.8)

    outer_ring = Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc + 0.5 * body_w, height=thick_y
    ) - Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc - 0.5 * body_w, height=thick_y + 2.0
    )
    crescent_body = outer_ring & (Pos(0, 0, 0.5 * (z_top + z_bot)) * Box(18.0, thick_y + 2.0, z_top - z_bot))

    slot_ring = Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc + 0.5 * slot_w, height=thick_y + 2.0
    ) - Pos(R_arc, 0, 0) * Rot(90, 0, 0) * Cylinder(
        radius=R_arc - 0.5 * slot_w, height=thick_y + 4.0
    )
    crescent_body -= slot_ring & Box(18.0, thick_y + 4.0, 2.0 * slot_half + 1.6)

    top_cap = Pos(R_arc - math.hypot(R_arc, z_top), 0, z_top) * Rot(90, 0, 0) * Cylinder(radius=0.5 * body_w, height=thick_y)
    bot_cap = Pos(R_arc - math.hypot(R_arc, z_bot), 0, z_bot) * Rot(90, 0, 0) * Cylinder(radius=0.5 * body_w, height=thick_y)
    tail_arm = Pos(0.4, 0, 0.5 * (z_bot - r_tail)) * Box(4.2, thick_y - 0.6, abs(r_tail - z_bot) + 3.0)
    tail_boss = Pos(0, 0, -r_tail) * Rot(90, 0, 0) * Cylinder(radius=2.3, height=thick_y)
    trunnion_bridge = Pos(-0.5 * body_w - 0.6, 0, 0) * Box(2.2, thick_y + 2.6, 4.6)
    trunnion_pins = Rot(90, 0, 0) * Cylinder(radius=1.8, height=thick_y + 3.6)

    link = (
        crescent_body + top_cap + bot_cap + tail_arm + tail_boss + trunnion_bridge
        + (trunnion_pins - Box(slot_w + 0.4, thick_y - 0.8, 6.0))
    )
    link -= Pos(0, 0, -r_tail) * Rot(90, 0, 0) * Cylinder(radius=1.0, height=thick_y + 4.0)
    return link


def build_cylinder_block(cfg: Dict, is_left: bool = True):
    bore = float(cfg["cylinder_and_rods"]["bore"])
    stroke = 2.0 * float(cfg["cylinder_and_rods"]["crank_radius"])
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    cyl_len, cyl_od = stroke + 12.0, bore + 7.0

    main_barrel = Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od, height=cyl_len)
    saddle_block = Pos(0, -5.5, 4.0) * Box(cyl_len - 2.0, 14.0, cyl_od + 4.0)
    front_cover = Pos(0.5 * cyl_len + 1.0, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od + 0.8, height=2.0)
    rear_cover = Pos(-0.5 * cyl_len - 1.0, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od + 0.8, height=2.0)
    gland_gp7 = Pos(-0.5 * cyl_len - 3.5, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=3.8, height=4.0)

    for b in Pos(0.5 * cyl_len + 2.2, 0, 0) * Rot(0, 90, 0) * PolarLocations(radius=0.5 * bore + 2.2, count=8) * Cylinder(radius=0.9, height=1.2):
        front_cover += b

    chest_len = cyl_len - 4.0
    steam_chest = Pos(0, 1.5, z_valve) * Box(chest_len, 17.0, 11.0)
    gland_gp5 = Pos(-0.5 * chest_len - 2.5, 3.5, z_valve) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=4.0)
    bar_len = stroke + 26.0
    slide_bar = Pos(-0.5 * cyl_len - 0.5 * bar_len, 0, 6.2) * Box(bar_len, 4.2, 3.2)
    drain_f = Pos(0.32 * cyl_len, 2.0, -0.5 * cyl_od - 1.5) * Cylinder(radius=1.8, height=4.0)
    drain_r = Pos(-0.32 * cyl_len, 2.0, -0.5 * cyl_od - 1.5) * Cylinder(radius=1.8, height=4.0)

    assembly = (
        main_barrel + saddle_block + front_cover + rear_cover + gland_gp7
        + steam_chest + gland_gp5 + slide_bar + drain_f + drain_r
        - Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=cyl_len - 2.0)
    )
    return assembly if is_left else mirror(assembly, about=Plane.XZ)


def build_boiler_superstructure(cfg: Dict, heavy_equipped: bool = True):
    b_cfg = cfg["boiler_and_aux"]
    od_sb = float(b_cfg["smokebox_od"])
    r_sb, L_sb = 0.5 * od_sb, float(b_cfg["smokebox_length"])
    fb_l, fb_w, fb_h = [float(v) for v in b_cfg["firebox_lwh"]]
    z_axis, x_sb_front = 44.0, 162.0
    x_sb_rear = x_sb_front - L_sb
    x_fb_rear = -112.0
    x_fb_front = x_fb_rear + fb_l

    smokebox = Pos(0.5 * (x_sb_front + x_sb_rear), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb, height=L_sb)
    door_ring = Pos(x_sb_front + 1.2, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 1.2, height=2.4)
    door_dish = (Pos(x_sb_front - 38.0, 0, z_axis) * Sphere(radius=44.5)) & (
        Pos(x_sb_front + 3.5, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_sb - 4.8, height=6.0)
    )
    door_handle = Pos(x_sb_front + 6.8, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=2.6, height=2.2)
    hinge_u = Pos(x_sb_front + 3.6, -11.0, z_axis + 9.0) * Box(2.5, 22.0, 2.2)
    hinge_l = Pos(x_sb_front + 3.6, -11.0, z_axis - 9.0) * Box(2.5, 22.0, 2.2)
    headlight = Pos(x_sb_front + 2.0, 0, z_axis + r_sb + 4.8) * Rot(0, 90, 0) * Cylinder(radius=4.5, height=9.5)
    feedwater_heater = Pos(x_sb_front - 14.0, 0, z_axis + r_sb + 2.5) * Rot(90, 0, 0) * Cylinder(radius=5.2, height=44.0)

    x_chimney = x_sb_front - 32.0
    chimney = Pos(x_chimney, 0, z_axis + r_sb + 6.5) * Cylinder(radius=7.8, height=15.0)
    if heavy_equipped:
        chimney += Pos(x_chimney - 2.0, 0, z_axis + r_sb + 11.5) * Box(22.0, 19.0, 13.0)
        chimney += Pos(x_chimney - 10.0, 0, z_axis + r_sb + 15.0) * Box(12.0, 17.0, 6.0)

    L_barrel, r_barrel = x_sb_rear - x_fb_rear, 30.6
    barrel = Pos(0.5 * (x_sb_rear + x_fb_rear), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=L_barrel)
    for bx in [72.0, 28.0, -16.0, -60.0]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.6, height=2.2)

    dome_base = Pos(26.0, 0, z_axis + r_barrel - 2.0) * extrude(SlotOverall(width=66.0, height=28.0), amount=13.0)
    sv_1 = Pos(-36.0, 3.8, z_axis + r_barrel + 4.5) * Cylinder(radius=2.2, height=9.0)
    sv_2 = Pos(-36.0, -3.8, z_axis + r_barrel + 4.5) * Cylinder(radius=2.2, height=9.0)
    firebox = Pos(0.5 * (x_fb_front + x_fb_rear), 0, z_axis - 12.0) * Box(fb_l, fb_w, fb_h)

    cab_len, cab_w = 54.0, 74.0
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0
    cab_box = Pos(cab_x, 0, z_axis + 4.0) * Box(cab_len, cab_w, 54.0)
    cab_roof = (Pos(cab_x - 2.0, 0, z_axis + 8.0) * Rot(0, 90, 0) * Cylinder(radius=42.0, height=cab_len + 8.0)) & (
        Pos(cab_x - 2.0, 0, z_axis + 35.0) * Box(cab_len + 8.0, cab_w + 4.0, 22.0)
    )
    cab = (
        cab_box + cab_roof
        - Pos(cab_x - 2.0, 0, z_axis + 3.0) * Box(cab_len - 3.0, cab_w - 4.0, 52.0)
        - Pos(cab_x + 2.0, 0, z_axis + 14.0) * Box(28.0, cab_w + 10.0, 18.0)
        - Pos(cab_x - 0.5 * cab_len, 0, z_axis + 2.0) * Box(8.0, 36.0, 48.0)
    )

    walkway_len, walkway_x = x_sb_front - x_fb_rear + 14.0, 0.5 * (x_sb_front + x_fb_rear) + 4.0
    walkways = None
    for sgn in (-1.0, 1.0):
        part = Pos(walkway_x, sgn * 34.5, 22.5) * Box(walkway_len, 9.5, 1.8) + Pos(
            x_sb_front - 26.0, sgn * 39.2, z_axis + 4.0
        ) * Box(68.0, 1.2, 46.0)
        walkways = part if walkways is None else (walkways + part)

    return (
        smokebox + door_ring + door_dish + door_handle + hinge_u + hinge_l + headlight
        + feedwater_heater + chimney + barrel + dome_base + sv_1 + sv_2 + firebox + cab + walkways
        # 按 Template 6 (ø53.0mm 锅筒内衬直径) 与火室燃烧腔掏空内部，避免实心黄铜块超重至 6.4kg
        - Pos(0.5 * (x_sb_front - 3.0 + x_fb_rear + 3.0), 0, z_axis)
        * Rot(0, 90, 0)
        * Cylinder(radius=0.5 * float(b_cfg["boiler_front_id"]) + 1.5, height=(x_sb_front - x_fb_rear) - 6.0)
        - Pos(0.5 * (x_fb_front + x_fb_rear), 0, z_axis - 12.0) * Box(fb_l - 4.0, fb_w - 4.0, fb_h - 4.0)
    )


def build_tender_body(cfg: Dict):
    """8-20B 煤水车薄壁金属板壳车体 (掏空水箱内腔以获得真实的 ~1.1kg 空车质量)。"""
    t_cfg = cfg["tender_and_track"]
    L, W, H = float(t_cfg["tender_body_length"]), float(t_cfg["tender_body_width"]), float(t_cfg["tender_body_height"])

    base_plate = Pos(0, 0, 1.2) * Box(L + 4.0, W + 2.0, 2.4)
    front_beam = Pos(0.5 * L + 1.5, 0, 0.5) * Box(3.5, W + 2.0, 6.5)
    rear_beam = Pos(-0.5 * L - 1.5, 0, 0.5) * Box(3.5, W + 2.0, 6.5)

    # 外壳板厚 1.6mm (掏空内部水柜空腔，避免实心金属块算成十几公斤!)
    tank_outer = Pos(-4.0, 0, 2.4 + 0.5 * H) * Box(L - 8.0, W, H)
    tank_inner_hollow = Pos(-4.0, 0, 2.4 + 0.5 * H) * Box(L - 11.2, W - 3.2, H - 3.2)
    coal_bunker_cut = Pos(32.0, 0, 2.4 + H - 7.0) * Box(118.0, W - 14.0, 16.0)
    tank_shell = tank_outer - tank_inner_hollow - coal_bunker_cut

    bunker_walls = None
    for sgn in (-1.0, 1.0):
        wall = Pos(26.0, sgn * (0.5 * W - 6.5), 2.4 + H + 3.5) * Box(126.0, 1.5, 8.0)
        bunker_walls = wall if bunker_walls is None else (bunker_walls + wall)

    alc_filler_cap = Pos(48.0, 8.0, 2.4 + H - 6.5) * Cylinder(radius=4.5, height=4.0)
    water_hatch = Pos(-78.0, 0, 2.4 + H + 1.5) * Cylinder(radius=9.5, height=3.0)
    sump_box = Pos(42.0, 0, -3.5) * Box(22.0, 18.0, 7.0) - Pos(42.0, 0, -3.0) * Box(19.0, 15.0, 6.0)

    return base_plate + front_beam + rear_beam + tank_shell + bunker_walls + alc_filler_cap + water_hatch + sump_box


def build_tender_bogie_frame(cfg: Dict):
    wb = float(cfg["tender_and_track"]["bogie_wheelbase"])
    half_wb = 0.5 * wb
    y_frame = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2

    bolster = Pos(0, 0, 2.0) * Box(14.0, 2.0 * y_frame, 5.5)
    sides = None
    for sgn in (-1.0, 1.0):
        yf = sgn * y_frame
        side_part = Pos(0, yf, 1.5) * Box(wb + 18.0, 2.4, 5.5) + Pos(0, yf + sgn * 1.2, -1.5) * Box(22.0, 2.2, 4.2)
        for ax_x in (-half_wb, +half_wb):
            side_part += Pos(ax_x, yf + sgn * 1.4, 0.0) * Box(7.5, 4.2, 8.2)
            side_part += Pos(ax_x - 4.8, yf + sgn * 1.0, 2.5) * Cylinder(radius=1.6, height=5.5)
            side_part += Pos(ax_x + 4.8, yf + sgn * 1.0, 2.5) * Cylinder(radius=1.6, height=5.5)
            side_part -= Pos(ax_x, yf, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=10.0)
        sides = side_part if sides is None else (sides + side_part)
    return bolster + sides


def build_spoked_truck_wheel(cfg: Dict, is_left: bool = True):
    r_tread = 0.5 * float(cfg["wheels_and_axles"]["leading_wheel_diameter"])
    w_tire = 3.4
    rim = (
        Cylinder(radius=r_tread, height=w_tire)
        + Pos(0, 0, -0.5 * w_tire + 0.45) * Cylinder(radius=r_tread + 1.3, height=0.9)
        - Cylinder(radius=r_tread - 2.0, height=w_tire + 2.0)
    )
    disc = Cylinder(radius=r_tread - 1.8, height=1.8)
    for ang_deg in range(0, 360, 45):
        ang = math.radians(ang_deg)
        disc -= Pos(7.2 * math.cos(ang), 7.2 * math.sin(ang), 0) * Cylinder(radius=1.45, height=4.0)

    wheel = rim + disc + Cylinder(radius=3.5, height=w_tire + 0.4) - Cylinder(radius=1.8, height=w_tire + 6.0)
    wheel_mj = Rot(-90, 0, 0) * wheel
    return wheel_mj if is_left else mirror(wheel_mj, about=Plane.XZ)


def main():
    parser = argparse.ArgumentParser(description="Aster D51 全零件 CAD 建模与材料密度惯量积分器")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/meshes"))
    parser.add_argument("--mass-out", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--standard-stack", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    # (part_key, stl_tolerance, density_kg_m3, builder_fn)
    tasks: List[Tuple[str, float, float, Callable[[], Any]]] = [
        ("driving_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True)),
        ("driving_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False)),
        ("main_rod", 0.02, RHO_STAINLESS, lambda: build_main_rod(cfg)),
        ("coupling_rod", 0.02, RHO_STAINLESS, lambda: build_coupling_rod(cfg)),
        ("piston_crosshead_L", 0.03, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, True)),
        ("piston_crosshead_R", 0.03, RHO_COMPOSITE_PISTON, lambda: build_piston_crosshead(cfg, False)),
        ("slide_valve_L", 0.02, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, True)),
        ("slide_valve_R", 0.02, RHO_COMPOSITE_VALVE, lambda: build_slide_valve_assembly(cfg, False)),
        ("die_block", 0.015, RHO_BRASS, lambda: build_die_block(cfg)),
        ("reverser_shaft", 0.025, RHO_STAINLESS, lambda: build_reverser_shaft(cfg)),
        ("driving_axlebox", 0.03, RHO_BRASS, lambda: build_driving_axlebox(cfg)),
        ("tender_axlebox", 0.03, RHO_CAST_IRON, lambda: build_tender_axlebox(cfg)),
        ("expansion_link", 0.02, RHO_STAINLESS, lambda: build_expansion_link(cfg)),
        ("cylinder_block_L", 0.04, RHO_BRASS, lambda: build_cylinder_block(cfg, True)),
        ("cylinder_block_R", 0.04, RHO_BRASS, lambda: build_cylinder_block(cfg, False)),
        ("boiler_superstructure", 0.08, RHO_BRASS, lambda: build_boiler_superstructure(cfg, not args.standard_stack)),
        ("tender_body", 0.08, RHO_BRASS, lambda: build_tender_body(cfg)),
        ("tender_bogie_frame", 0.04, RHO_CAST_IRON, lambda: build_tender_bogie_frame(cfg)),
        ("truck_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, True)),
        ("truck_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_spoked_truck_wheel(cfg, False)),
    ]

    mass_db: Dict[str, Any] = {
        "meta": {
            "units": "SI (kg, m, kg*m^2)",
            "densities_kg_m3": {
                "brass_BS": RHO_BRASS,
                "stainless_SUS": RHO_STAINLESS,
                "cast_iron_FC": RHO_CAST_IRON,
            },
        },
        "parts": {},
    }

    for idx, (key, tol, rho, builder) in enumerate(tasks, start=1):
        stl_path = args.out_dir / f"{key}.stl"
        print(f"[CAD+MASS] {idx:02d}/{len(tasks)} 构建并积分 {key}.stl (ρ={rho:.0f} kg/m³) ...")
        solid = builder()
        export_stl(solid, str(stl_path), tolerance=tol)
        mp = compute_occ_mass_props(solid, rho)
        mass_db["parts"][key] = mp
        print(f"           -> 质量 = {mp['mass_kg']*1000:6.2f} g | 质心 = [{mp['com_m'][0]*1000:+.1f}, {mp['com_m'][1]*1000:+.1f}, {mp['com_m'][2]*1000:+.1f}] mm")

    args.mass_out.parent.mkdir(parents=True, exist_ok=True)
    args.mass_out.write_text(json.dumps(mass_db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] 已导出 {len(tasks)} 个 STL 网格，并将精确材料惯量库写入: {args.mass_out}")


if __name__ == "__main__":
    main()