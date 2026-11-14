#!/usr/bin/env python3
"""
scripts/02_build_cad_meshes.py
Aster D51 全套 32 件参数化 CAD 建模 + OpenCascade 精确物理质量/偏心质心/3x3惯量张量积分器 (V2-GEAR 精密走行配气版)。
严格遵循 GEMINI.md 第 5 节契约：
  - 黄铜 (BS: 气缸、滑阀、锅筒配件、动轮轴箱、月牙滑块、给水泵偏心箍): 8500 kg/m³
  - 不锈钢 (SUS: 主连杆、三段铰接耦合连杆、7-10 回动曲柄、华氏五杆系、逆转横轴): 7930 kg/m³
  - 铸铁/钢 (FC/S45C: 第3主动轮/从动轮、2-5 配气导轨托架、先导/从轮台车架、煤水车转向架): 7350 kg/m³
输出:
  - models/meshes/*.stl (32 个高精度水密网格)
  - configs/d51_mass_props.json (SI 单位制: kg, m, kg·m²)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
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
RHO_BRASS = 8500.0              # 黄铜 BS (1-6 滑阀, 5-9 轴箱, 2-3 滑块, 6-3 偏心环等)
RHO_STAINLESS = 7930.0          # 不锈钢 SUS (连杆系, 7-10 回动曲柄, 华氏五杆, 4-12 逆转横轴等)
RHO_CAST_IRON = 7350.0          # 铸铁/碳钢 (Boxpok 动轮, 2-5 配气托架, 台车架, 转向架等)
RHO_COMPOSITE_VALVE = 8250.0    # 1-6 黄铜滑阀 + 1-22 不锈钢阀芯棒复合体
RHO_COMPOSITE_PISTON = 8200.0   # 1-28 黄铜活塞 + SUS 活塞杆 + 1-31 十字头复合体


def compute_occ_mass_props(solid: Any, density_kg_m3: float) -> Dict[str, Any]:
    """
    调用底层 OpenCascade BRepGProp 对三维实体做精确体积积分 (建模单位 mm -> 物理单位 SI m/kg)。
      - volume_mm3 * 1e-9 * density -> mass (kg)
      - MatrixOfInertia (关于质心 CoM, 单位 mm^5) * 1e-15 * density -> I_com (kg·m²)
    返回 MuJoCo fullinertia 所需的 6 分量 [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]。
    """
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


# ==================== 1. 动轮系、主连杆、三段铰接耦合连杆与 7-10 回动曲柄 ====================

def build_boxpok_wheel(cfg: Dict, phase_deg: float, is_left: bool, is_main_driver: bool = False):
    """
    JNR D51 Boxpok 箱型动轮 (イラスト 5):
      - is_main_driver=False (5-5, 5-6, 5-8 第 1/2/4 动轮): 标准月牙配重 + 短曲柄销 (承接耦合连杆);
      - is_main_driver=True  (5-7 第 3 主动轮): 加大扇形平衡配重块 + 三阶加长主曲柄销 (依次串接耦合连杆、主连杆大头与 7-10 回动曲柄抱箍)。
    """
    d_tread = float(cfg["wheels_and_axles"]["driving_wheel_diameter"])
    r_tread = 0.5 * d_tread
    r_flange = r_tread + 1.4
    r_rim_inner = r_tread - 2.4
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    w_tire, w_flange, w_web = 3.8, 1.0, (2.8 if is_main_driver else 2.4)

    tire = Cylinder(radius=r_tread, height=w_tire)
    flange = Pos(0, 0, -0.5 * w_tire + 0.5 * w_flange) * Cylinder(radius=r_flange, height=w_flange)
    wheel = tire + flange - Cylinder(radius=r_rim_inner, height=w_tire + 2.0)

    web = Cylinder(radius=r_rim_inner + 0.2, height=w_web)
    hole_angles = [42, 75, 108, 252, 285, 318]
    r_hole_ring = 0.54 * (r_rim_inner + 4.5)
    for ang_deg in hole_angles:
        ang_rad = math.radians(ang_deg)
        hx, hy = r_hole_ring * math.cos(ang_rad), r_hole_ring * math.sin(ang_rad)
        web -= Pos(hx, hy, 0) * Cylinder(radius=2.1 if is_main_driver else 2.3, height=w_web + 2.0)
        hx2, hy2 = (r_rim_inner - 2.6) * math.cos(ang_rad), (r_rim_inner - 2.6) * math.sin(ang_rad)
        web -= Pos(hx2, hy2, 0) * Cylinder(radius=1.05 if is_main_driver else 1.15, height=w_web + 2.0)
    wheel += web

    # 180° 反侧月牙形平衡配重块: 第 3 主动轮需额外平衡往复主连杆质量，配重扇面更大更厚
    cw_thick = w_tire if is_main_driver else (w_tire - 0.3)
    cw_cut_x = 9.2 if is_main_driver else 7.5
    cw_outer = Cylinder(radius=r_rim_inner + 0.1, height=cw_thick)
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
        # 三阶主曲柄销: 耦合连杆段 (z=1.9..6.6) -> 主连杆大头段 (z=6.6..10.7) -> 7-10 回动曲柄抱箍颈 (z=10.7..12.1)
        pin_coup = Pos(r_crank, 0, 4.25) * Cylinder(radius=1.75, height=4.7)
        pin_main = Pos(r_crank, 0, 8.65) * Cylinder(radius=1.70, height=4.1)
        pin_ret = Pos(r_crank, 0, 11.40) * Cylinder(radius=1.45, height=1.4)
        crank_pin = pin_coup + pin_main + pin_ret
    else:
        # 从动轮短曲柄销 + 外端止退垫圈帽
        crank_pin = Pos(r_crank, 0, boss_z + 2.3) * Cylinder(radius=1.70, height=4.6)
        crank_pin += Pos(r_crank, 0, boss_z + 4.8) * Cylinder(radius=2.35, height=0.7)

    wheel = wheel + center_hub + crank_pad + crank_pin - Cylinder(radius=2.8, height=w_tire + 10.0)

    wheel_mj = Rot(-90, 0, 0) * wheel
    if not is_left:
        wheel_mj = mirror(wheel_mj, about=Plane.XZ)
    if abs(phase_deg) > 1e-6:
        wheel_mj = Rot(0, -phase_deg, 0) * wheel_mj
    return wheel_mj


def build_return_crank(cfg: Dict, is_left: bool = True):
    """
    7-10 回动曲柄臂 (リターンクランク, SUS 不锈钢, イラスト 7 & 图 9.2):
      - 局部原点 (0, 0, 0) 位于第 3 主动轮主曲柄销 (cx, ±y_vg, cz);
      - 曲柄臂抱箍面处于内侧层 (y_local = -1.45mm, 对应 |Y| = 34.55mm)，避让外侧偏心棒全周摆动包络;
      - 偏心销位于 (rx - cx, 0, rz - cz) (严格距轮心 7.0mm, 相位 -90°)，向外侧 (+Y/-Y) 伸出承接偏心棒。
    """
    r_crank = float(cfg["cylinder_and_rods"]["crank_radius"])
    r_ret = float(cfg["walschaerts_valve_gear"]["return_crank_radius"])
    ph_ret = math.radians(float(cfg["walschaerts_valve_gear"]["return_crank_phase_deg"]))
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    dx = r_ret * math.cos(ph_ret) - r_crank
    dz = r_ret * math.sin(ph_ret) - 0.0
    arm_len = math.hypot(dx, dz)
    pitch_deg = math.degrees(-math.atan2(dz, dx))

    # 在沿 +X 轴标准姿态建模后旋转至 (dx, dz)
    y_web = -1.45
    clamp_hub = Pos(0, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=1.5)
    pinch_lug = Pos(-2.6, y_web, 1.4) * Box(3.2, 1.5, 2.8)
    pinch_slit = Pos(-3.2, y_web, 0.0) * Box(3.2, 2.2, 0.5)
    web = Pos(0.5 * arm_len, y_web, 0) * Box(arm_len, 1.2, 3.6)
    ecc_hub = Pos(arm_len, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=2.3, height=1.5)
    # 向外侧伸出的偏心销颈 (穿过 y_local = 0 处的偏心棒后孔) + 外端螺母帽
    ecc_pin = Pos(arm_len, 0.05, 0) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=2.2)
    ecc_cap = Pos(arm_len, 1.25, 0) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=0.55)

    crank = clamp_hub + pinch_lug + web + ecc_hub + ecc_pin + ecc_cap - pinch_slit
    crank -= Pos(0, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=1.45, height=4.0)
    crank_L = Rot(0, pitch_deg, 0) * crank

    if is_left:
        return crank_L
    return Rot(0, -q_deg, 0) * mirror(crank_L, about=Plane.XZ)


def build_main_rod(cfg: Dict):
    """7-11 / 7-16 主连杆 (SUS 不锈钢)，大头原点 (0,0,0)，小头 (+L_main,0,0)。"""
    L_main = float(cfg["cylinder_and_rods"]["main_rod_length"])
    d_small = float(cfg["cylinder_and_rods"]["crosshead_pin_dia"])
    thickness_y, boss_thick_y = 2.4, 3.0
    h_big, h_small, h_shank = 6.8, 5.0, 3.6

    big_end = Box(8.4, boss_thick_y, h_big)
    big_oil_cup = Pos(0, 0, 0.5 * h_big + 0.7) * Cylinder(radius=1.3, height=1.4)
    small_end = Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * h_small, height=boss_thick_y)
    shank = Pos(0.5 * L_main, 0, 0) * Box(L_main - 2.0, thickness_y, h_shank)
    rod = big_end + big_oil_cup + small_end + shank

    flute_len = L_main - 14.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * L_main, sgn * (0.5 * thickness_y - 0.35), 0) * Box(flute_len, 0.75, h_shank - 1.3)

    rod -= Rot(90, 0, 0) * Cylinder(radius=1.75, height=boss_thick_y + 4.0)
    rod -= Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * d_small, height=boss_thick_y + 4.0)
    return rod


def build_coupling_rod_main(cfg: Dict):
    """
    7-12 / 7-15 分段铰接式四轴耦合连杆 — 主中段 (第 3 主动轮 0.0mm <-> 第 2 动轮 +46.5mm):
      - 原点 (0,0,0) 对齐第 3 主动轮曲柄销;
      - 后端带后铰接单耳舌片 (-k_off = -8.5mm)，前端带前铰接单耳舌片 (+46.5 + 8.5 = +55.0mm)。
    """
    w_cfg = cfg["wheels_and_axles"]
    x_d2 = float(w_cfg["axle_x"]["drive_2"])
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    x_kr, x_kf = -k_off, x_d2 + k_off
    shank_t, boss_t, shank_h = 2.4, 3.2, 3.4

    rod = Pos(0.5 * x_d2, 0, 0) * Box(x_d2, shank_t, shank_h)
    for x_pos, r_b in [(0.0, 4.1), (x_d2, 3.4)]:
        rod += Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=r_b, height=boss_t)
        rod += Pos(x_pos, 0, r_b + 0.6) * Cylinder(radius=1.1, height=1.3)

    # 前/后铰接单耳中心舌片 (厚度 1.3mm, 插入前后副杆叉头内)
    tongue_t = 1.3
    rod += Pos(0.5 * x_kr, 0, 0) * Box(abs(x_kr), shank_t, shank_h - 0.2)
    rod += Pos(x_kr, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=tongue_t)
    rod += Pos(x_d2 + 0.5 * k_off, 0, 0) * Box(k_off, shank_t, shank_h - 0.2)
    rod += Pos(x_kf, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=tongue_t)

    flute_len = x_d2 - 12.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * x_d2, sgn * (0.5 * shank_t - 0.35), 0) * Box(flute_len, 0.75, shank_h - 1.2)

    for x_pos in (0.0, x_d2):
        rod -= Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.78, height=boss_t + 4.0)
    for x_k in (x_kr, x_kf):
        rod -= Pos(x_k, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=boss_t + 4.0)
    return rod


def build_coupling_rod_end(cfg: Dict, is_front: bool = True):
    """
    7-12 / 7-15 分段铰接式四轴耦合连杆 — 前段 (接第1动轮) / 后段 (接第4动轮):
      - 原点 (0,0,0) 位于铰接节双叉耳孔 (Knuckle Clevis)，远端位于 dx = +38.0mm (前) 或 -38.0mm (后) 曲柄销孔。
    """
    w_cfg = cfg["wheels_and_axles"]
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    dx = (float(w_cfg["axle_x"]["drive_1"]) - (float(w_cfg["axle_x"]["drive_2"]) + k_off)) if is_front else (
        float(w_cfg["axle_x"]["drive_4"]) - (-k_off)
    )
    L_seg = abs(dx)
    sgn_x = 1.0 if dx >= 0 else -1.0
    shank_t, boss_t, shank_h, boss_r = 2.4, 3.2, 3.3, 3.4

    # 铰接双叉头 (内槽宽 1.5mm 包容主中段 1.3mm 舌片) + 贯通铰接销帽
    clevis = Rot(90, 0, 0) * Cylinder(radius=2.3, height=boss_t) + Pos(sgn_x * 1.8, 0, 0) * Box(3.6, boss_t, 4.2)
    clevis -= Box(5.2, 1.5, 6.0)
    knuckle_pin = Rot(90, 0, 0) * Cylinder(radius=0.92, height=boss_t + 0.6)

    shank = Pos(0.5 * dx, 0, 0) * Box(L_seg, shank_t, shank_h)
    wheel_boss = Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=boss_r, height=boss_t)
    oil_cup = Pos(dx, 0, boss_r + 0.6) * Cylinder(radius=1.05, height=1.2)
    rod = clevis + knuckle_pin + shank + wheel_boss + oil_cup

    flute_len = L_seg - 10.5
    if flute_len > 6.0:
        for sy in (-1.0, 1.0):
            rod -= Pos(0.5 * dx, sy * (0.5 * shank_t - 0.35), 0) * Box(flute_len, 0.75, shank_h - 1.2)

    rod -= Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=boss_t + 4.0)
    return rod


# ==================== 2. 华氏五杆精密不锈钢 CAD 实体 (替代胶囊近似杆) ====================

def build_eccentric_rod(cfg: Dict):
    """2-12 / 2-19 偏心棒 (SUS 不锈钢): 原点 (0,0,0) 后偏心销孔，前端 (+L_ecc, 0, 0) 双叉头抱合月牙板尾端。"""
    L_ecc = float(cfg["walschaerts_valve_gear"]["eccentric_rod_length"])
    rear_boss = Rot(90, 0, 0) * Cylinder(radius=2.3, height=1.9) + Pos(0, 0, 2.6) * Cylinder(radius=0.85, height=0.9)
    shank = Pos(0.5 * L_ecc, 0, 0) * Box(L_ecc, 1.4, 2.5)
    front_clevis = (
        Pos(L_ecc, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=3.8)
        + Pos(L_ecc - 1.6, 0, 0) * Box(3.2, 3.8, 3.8)
        - Pos(L_ecc + 0.6, 0, 0) * Box(4.6, 2.55, 5.0)
    )
    rod = rear_boss + shank + front_clevis
    for sy in (-1.0, 1.0):
        rod -= Pos(0.5 * L_ecc, sy * 0.55, 0) * Box(L_ecc - 11.0, 0.45, 1.3)
    rod -= Rot(90, 0, 0) * Cylinder(radius=1.18, height=5.0)
    rod -= Pos(L_ecc, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=6.0)
    return rod


def build_radius_rod(cfg: Dict):
    """
    2-1 心向棒 (SUS 不锈钢):
      - 原点 (0,0,0) 对齐月牙槽滑块销孔 (s_die);
      - 后端延伸至 (-L_back, 0, 0) 连接逆转吊杆; 前端延伸至 (+L_rad, 0, 0) 双叉头抱合合并杆。
    """
    vg = cfg["walschaerts_valve_gear"]
    L_rad = float(vg["radius_rod_length"])
    L_back = float(vg["radius_rod_back_ext"])

    # 跨越月牙板本体 (厚 3.0mm) 的前后双侧夹板框 (-5mm .. +8mm)
    yoke_frame = Box(13.0, 4.6, 3.4) - Box(9.6, 3.2, 5.0)
    die_boss = Rot(90, 0, 0) * Cylinder(radius=1.9, height=4.8)
    back_bar = Pos(-0.5 * (L_back + 5.0), 0, 0) * Box(L_back - 3.0, 1.5, 2.3)
    back_eye = Pos(-L_back, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=1.6)

    front_len = L_rad - 6.5
    front_shank = Pos(6.5 + 0.5 * front_len, 0, 0) * Box(front_len, 1.5, 2.6)
    front_clevis = (
        Pos(L_rad, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=3.4)
        + Pos(L_rad - 1.5, 0, 0) * Box(3.0, 3.4, 3.6)
        - Pos(L_rad + 0.5, 0, 0) * Box(4.2, 1.75, 5.0)
    )
    rod = yoke_frame + die_boss + back_bar + back_eye + front_shank + front_clevis
    for sy in (-1.0, 1.0):
        rod -= Pos(6.5 + 0.5 * front_len, sy * 0.58, 0) * Box(front_len - 8.0, 0.45, 1.3)
    for xp, r_h in [(0.0, 0.85), (-L_back, 0.85), (L_rad, 0.90)]:
        rod -= Pos(xp, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=r_h, height=6.5)
    return rod


def build_combination_lever(cfg: Dict):
    """
    1-23 合併てこ (SUS 不锈钢):
      - 原点 (0,0,0) 下端先导短连杆孔;
      - 次上孔 (0, 0, L_comb - L_mv = +24.5mm) 接心向棒前端;
      - 最上孔 (0, 0, L_comb = +28.0mm) 接平面滑阀叉头。
    """
    vg = cfg["walschaerts_valve_gear"]
    L_comb = float(vg["comb_lever_total_length"])
    L_mv = float(vg["comb_lever_lap_lead_offset"])
    z_mid = L_comb - L_mv

    bot_eye = Rot(90, 0, 0) * Cylinder(radius=1.85, height=1.5)
    lower_shank = Pos(0, 0, 0.5 * z_mid) * Box(2.4, 1.5, z_mid)
    mid_boss = Pos(0, 0, z_mid) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=1.6)
    upper_neck = Pos(0, 0, z_mid + 0.5 * L_mv) * Box(2.6, 1.6, L_mv)
    top_boss = Pos(0, 0, L_comb) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=1.6)

    lever = bot_eye + lower_shank + mid_boss + upper_neck + top_boss
    for zp in (0.0, z_mid, L_comb):
        lever -= Pos(0, 0, zp) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=5.0)
    return lever


def build_union_link(cfg: Dict):
    """1-27 先导短连杆 (ユニオンリンク, SUS 不锈钢): 原点 (0,0,0) 至 (+L_union, 0, 0)。"""
    L_un = float(cfg["walschaerts_valve_gear"]["union_link_length"])
    clevis_0 = Rot(90, 0, 0) * Cylinder(radius=1.85, height=3.4) - Pos(-0.6, 0, 0) * Box(3.6, 2.3, 4.5)
    clevis_1 = Pos(L_un, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.85, height=3.2) - Pos(L_un + 0.6, 0, 0) * Box(3.6, 1.65, 4.5)
    bar = Pos(0.5 * L_un, 0, 0) * Box(L_un - 1.0, 1.5, 2.2)
    link = clevis_0 + clevis_1 + bar
    for xp in (0.0, L_un):
        link -= Pos(xp, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.88, height=5.0)
    return link


def build_lift_link(cfg: Dict):
    """2-10 逆转吊杆 (SUS 不锈钢): 上孔原点 (0,0,0) 接逆转摇臂，下孔 (0, 0, -L_lift) 接心向棒尾端。"""
    L_lift = float(cfg["walschaerts_valve_gear"]["lift_link_length"])
    top_clevis = Rot(90, 0, 0) * Cylinder(radius=1.8, height=3.4) - Pos(0, 0, 0.6) * Box(4.0, 2.15, 3.4)
    bot_clevis = Pos(0, 0, -L_lift) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=3.2) - Pos(0, 0, -L_lift - 0.6) * Box(4.0, 1.75, 3.4)
    strap = Pos(0, 0, -0.5 * L_lift) * Box(2.2, 1.5, L_lift - 0.8)
    link = top_clevis + bot_clevis + strap
    for zp in (0.0, -L_lift):
        link -= Pos(0, 0, zp) * Rot(90, 0, 0) * Cylinder(radius=0.85, height=5.0)
    return link


# ==================== 3. 2-5 配气导轨托架、轴动泵偏心环与先导/从轮台车架 ====================

def build_motion_bracket(cfg: Dict, is_left: bool = True):
    """
    2-5 加減リンク受 / 配气与十字头单导轨支撑托架 (イラスト 2, 铸铁/钢):
      - 局部原点 (0, 0, 0) 对齐月牙板定轴中心 (trunnion_x=46.5, +y_vg=+36.0, trunnion_z=15.5);
      - 向内侧固定于主车架侧板 (Y=13.5mm)，向上拱起跨过心向棒通道抱持月牙板内外耳轴;
      - 向前伸出托臂承托气缸单导轨滑杆 (1-33, X=66.5mm, Y=32.5mm, Z=7.8mm) 悬臂尾端。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    tx, tz = float(vg["trunnion_x"]), float(vg["trunnion_z"])
    y_vg = float(rc["cylinder_center_y"]) + 3.5

    dy_chassis = 14.5 - y_vg          # -21.5 mm (主车架外侧面)
    dy_slidebar = float(rc["cylinder_center_y"]) - y_vg  # -3.5 mm (单导轨中心 Y=32.5mm)
    dx_bar_tail = 66.5 - tx           # +20.0 mm (避开十字头最后死点 75.2mm)
    dz_bar_top = 8.4 - tz             # -7.1 mm  (贴合单导轨上表面 Z=7.8mm)

    # 1. 车架安装底座与横向悬臂肋板 (Z=18.5mm, 完全高于主连杆摆动包络)
    frame_pad = Pos(0, dy_chassis, 1.0) * Box(11.0, 2.2, 12.0)
    trans_beam = Pos(0, 0.5 * (dy_chassis - 3.6), 3.2) * Box(5.6, abs(dy_chassis - 3.6), 3.4)
    inner_boss = Pos(0, -3.7, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=2.0)

    # 2. 顶部跨桥与外侧耳轴座 (净空 y ∈ [-2.7, +2.7], z ∈ [-8.5, +7.2] 供月牙板与心向棒无干涉摆动)
    arch_in = Pos(0, -3.7, 4.4) * Box(4.8, 2.0, 8.8)
    arch_top = Pos(0, 0.0, 8.5) * Box(4.8, 9.4, 2.0)
    arch_out = Pos(0, +3.7, 4.4) * Box(4.8, 2.0, 8.8)
    outer_boss = Pos(0, +3.7, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=2.0)

    # 3. 前伸单导轨尾端托臂 (沿内侧 y_local=-5.2mm 延伸至 X=66.5mm 后向外抱合导轨顶面)
    guide_arm = Pos(0.5 * dx_bar_tail, -5.2, 0.5 * dz_bar_top + 1.5) * Rot(0, 22, 0) * Box(dx_bar_tail + 2.0, 2.2, 3.2)
    guide_clamp = Pos(dx_bar_tail, dy_slidebar - 0.8, dz_bar_top) * Box(5.2, 6.2, 2.2)

    bracket = (
        frame_pad + trans_beam + inner_boss + arch_in + arch_top + arch_out
        + outer_boss + guide_arm + guide_clamp
        - Rot(90, 0, 0) * Cylinder(radius=1.85, height=12.0)
    )
    return bracket if is_left else mirror(bracket, about=Plane.XZ)


def build_axle_pump_eccentric(cfg: Dict):
    """6-3 / 6-4 第 3 轴中央给水泵黄铜偏心轮盘与偏心环连杆: 原点 (0,0,0) 位于偏心轮心 (+r_pump,0,0)。"""
    b_cfg = cfg["boiler_and_aux"]
    L_rod = abs(float(b_cfg["axle_pump_x"]) - float(b_cfg["axle_pump_eccentric"]))

    sheave_strap = Rot(90, 0, 0) * Cylinder(radius=6.4, height=3.4) - Rot(90, 0, 0) * Cylinder(radius=2.9, height=5.0)
    flange_ring = Rot(90, 0, 0) * Cylinder(radius=5.2, height=4.2) - Rot(90, 0, 0) * Cylinder(radius=2.9, height=6.0)
    neck_rod = Pos(-0.5 * (L_rod + 4.5), 0, 0) * Box(L_rod - 4.5, 2.2, 3.0)
    ram_clevis = (
        Pos(-L_rod, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.4, height=4.2)
        - Pos(-L_rod - 0.8, 0, 0) * Box(4.2, 2.2, 5.5)
        - Pos(-L_rod, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.0, height=6.0)
    )
    return sheave_strap + flange_ring + neck_rod + ram_clevis


def build_leading_truck_frame(cfg: Dict):
    """
    19-2 先导径向摆臂铸铁台车架 + 20-6 R2m/R3m 双孔调节槽座 (イラスト 19 & 20):
      - 局部原点 (0, 0, 0) 对齐先导轮轴中心 (leading_wheelset)，向后延伸至 -52mm (R2m孔) 与 -62mm (R3m孔)。
    """
    y_in = 0.5 * float(cfg["meta"]["track_gauge"]) - 5.2
    cross_bolster = Box(9.0, 2.0 * y_in + 4.0, 4.6)
    pivot_tongue = Pos(-57.0, 0, 0) * Box(20.0, 11.0, 4.2)
    # 20-6 标志性的前后 10mm 双孔定位凸台 (-52mm R2m 与 -62mm R3m)
    boss_r2m = Pos(-52.0, 0, 0) * Cylinder(radius=3.6, height=5.2)
    boss_r3m = Pos(-62.0, 0, 0) * Cylinder(radius=3.6, height=5.2)

    arms = None
    for sgn in (-1.0, 1.0):
        jbox = Pos(0, sgn * y_in, 0.4) * Box(8.6, 4.2, 8.2) - Pos(0, sgn * y_in, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=7.0)
        beam = Pos(-26.0, sgn * 0.5 * (y_in + 4.0), 0) * Rot(0, 0, sgn * -14.5) * Box(52.0, 3.2, 3.8)
        sub = jbox + beam
        arms = sub if arms is None else (arms + sub)

    frame = cross_bolster + pivot_tongue + boss_r2m + boss_r3m + arms
    frame -= Pos(-52.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Pos(-62.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Rot(90, 0, 0) * Cylinder(radius=2.1, height=2.0 * y_in + 8.0)
    return frame


def build_trailing_truck_frame(cfg: Dict):
    """
    19-11 从轮 Delta 式铸铁台车架 (イラスト 19):
      - 局部原点 (0, 0, 0) 对齐从轮台车前摆动支点 (trailing_truck_arm)，向后延伸 -54.0mm 至从轮轴箱。
    """
    dx_ax = -54.0
    y_out = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.0
    pivot_nose = Cylinder(radius=4.5, height=5.0) + Pos(-6.0, 0, 0) * Box(12.0, 10.0, 4.4)
    rear_tie = Pos(dx_ax - 6.5, 0, 1.2) * Box(4.2, 2.0 * y_out + 2.0, 3.6)

    sides = None
    for sgn in (-1.0, 1.0):
        beam = Pos(0.5 * dx_ax, sgn * 0.52 * y_out, 0.5) * Rot(0, 0, sgn * 26.5) * Box(58.0, 3.2, 4.2)
        abox = (
            Pos(dx_ax, sgn * y_out, 0.4) * Box(10.0, 4.4, 9.2)
            + Pos(dx_ax, sgn * (y_out + sgn * 1.2), 3.6) * Box(12.0, 3.6, 3.0)
            - Pos(dx_ax, sgn * y_out, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=7.0)
        )
        sub = beam + abox
        sides = sub if sides is None else (sides + sub)

    return pivot_nose + rear_tie + sides - Cylinder(radius=1.8, height=8.0)


# ==================== 4. 原有细分运动部件与锅炉/煤水车总成 ====================

def build_piston_crosshead(cfg: Dict, is_left: bool = True):
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    bore = float(rc["bore"])
    L_main = float(rc["main_rod_length"])
    cyl_cx = float(rc["cylinder_center_x"])
    drop_z = float(vg["crosshead_drop"])

    p_head_dx = cyl_cx - L_main
    sgn = 1.0 if is_left else -1.0

    piston_disc = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore - 0.1, height=4.6)
    ring_groove = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * (
        Cylinder(radius=0.5 * bore + 0.5, height=1.6) - Cylinder(radius=0.5 * bore - 0.9, height=2.0)
    )
    piston_disc -= ring_groove
    piston_rod = Pos(0.5 * p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.5, height=p_head_dx)

    ch_hub = Box(9.0, 5.8, 7.2)
    top_shoe = Pos(0, 0, 6.0) * Box(13.5, 6.2, 5.2) - Pos(0, 0, 6.2) * Box(15.0, 4.4, 3.4)

    drop_arm = Pos(0, sgn * 1.8, -0.5 * drop_z) * Box(4.2, 2.4, drop_z + 2.5)
    drop_boss = Pos(0, sgn * 3.5, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.2)

    part = piston_disc + piston_rod + ch_hub + top_shoe + drop_arm + drop_boss
    part -= Rot(90, 0, 0) * Cylinder(radius=1.5, height=10.0)
    part -= Pos(0, sgn * 3.5, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.0, height=6.0)
    return part


def build_slide_valve_assembly(cfg: Dict, is_left: bool = True):
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    v_dx = float(rc["cylinder_center_x"]) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))

    clevis = Box(6.0, 3.8, 4.6) - Box(3.6, 1.8, 6.0)
    clevis -= Rot(90, 0, 0) * Cylinder(radius=1.0, height=6.0)

    rod_len = v_dx + 20.0
    stem = Pos(0.5 * rod_len - 2.0, -3.5, 0) * Rot(0, 90, 0) * Cylinder(radius=1.0, height=rod_len)
    cross_arm = Pos(1.5, -1.75, 0) * Box(2.6, 5.5, 3.2)

    valve_block = Pos(v_dx, -3.5, -0.5) * Box(10.2, 9.6, 5.2)
    exhaust_cavity = Pos(v_dx, -3.5, -2.2) * Box(4.8, 6.6, 2.4)
    valve_block -= exhaust_cavity

    assembly = clevis + cross_arm + stem + valve_block
    return assembly if is_left else mirror(assembly, about=Plane.XZ)


def build_die_block(cfg: Dict):
    block = Box(4.6, 2.2, 3.6)
    pin_boss = Rot(90, 0, 0) * Cylinder(radius=1.4, height=4.2)
    hole = Rot(90, 0, 0) * Cylinder(radius=0.8, height=6.0)
    return block + pin_boss - hole


def build_reverser_shaft(cfg: Dict):
    """4-12 逆转器横轴 + 2-11/2-18 左右等长 11.0mm 双提升曲柄臂 (イラスト 2 & 4)。"""
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5
    arm_len_x = float(vg["reverser_arm_length"])

    main_bar = Rot(90, 0, 0) * Cylinder(radius=1.6, height=2.0 * y_vg + 4.0)
    reach_lever = Pos(0, -18.0, 5.5) * Box(3.6, 2.2, 12.0)

    arms = None
    for sgn in (-1.0, 1.0):
        hub = Pos(0, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=3.0, height=2.6)
        arm = Pos(0.5 * arm_len_x, sgn * y_vg, 0) * Box(arm_len_x, 2.0, 3.2)
        tip_boss = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.0)
        tip_hole = Pos(arm_len_x, sgn * y_vg, 0) * Rot(90, 0, 0) * Cylinder(radius=0.85, height=5.0)
        sub = hub + arm + tip_boss - tip_hole
        arms = sub if arms is None else (arms + sub)

    return main_bar + reach_lever + arms


def build_driving_axlebox(cfg: Dict):
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) - 6.5
    pair = None
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0.5) * Box(10.5, 5.2, 11.5)
        flange = Pos(0, sgn * y_box, 0.5) * Box(12.5, 2.2, 13.0)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=8.0)
        one = blk + flange - bore
        pair = one if pair is None else (pair + one)
    return pair


def build_tender_axlebox(cfg: Dict):
    y_box = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2
    pair = None
    for sgn in (-1.0, 1.0):
        blk = Pos(0, sgn * y_box, 0) * Box(7.0, 3.8, 7.6)
        bore = Pos(0, sgn * y_box, 0) * Rot(90, 0, 0) * Cylinder(radius=1.9, height=6.0)
        one = blk - bore
        pair = one if pair is None else (pair + one)
    return pair


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
    tail_boss = Pos(0, 0, -r_tail) * Rot(90, 0, 0) * Cylinder(radius=2.2, height=2.4)
    trunnion_bridge = Pos(-0.5 * body_w - 0.6, 0, 0) * Box(2.2, thick_y + 2.6, 4.6)
    trunnion_pins = Rot(90, 0, 0) * Cylinder(radius=1.8, height=thick_y + 5.6)

    link = (
        crescent_body + top_cap + bot_cap + tail_arm + tail_boss + trunnion_bridge
        + (trunnion_pins - Box(slot_w + 0.4, thick_y - 0.8, 6.0))
    )
    link -= Pos(0, 0, -r_tail) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=thick_y + 4.0)
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
        - Pos(0.5 * (x_sb_front - 3.0 + x_fb_rear + 3.0), 0, z_axis)
        * Rot(0, 90, 0)
        * Cylinder(radius=0.5 * float(b_cfg["boiler_front_id"]) + 1.5, height=(x_sb_front - x_fb_rear) - 6.0)
        - Pos(0.5 * (x_fb_front + x_fb_rear), 0, z_axis - 12.0) * Box(fb_l - 4.0, fb_w - 4.0, fb_h - 4.0)
    )


def build_tender_body(cfg: Dict):
    t_cfg = cfg["tender_and_track"]
    L, W, H = float(t_cfg["tender_body_length"]), float(t_cfg["tender_body_width"]), float(t_cfg["tender_body_height"])

    base_plate = Pos(0, 0, 1.2) * Box(L + 4.0, W + 2.0, 2.4)
    front_beam = Pos(0.5 * L + 1.5, 0, 0.5) * Box(3.5, W + 2.0, 6.5)
    rear_beam = Pos(-0.5 * L - 1.5, 0, 0.5) * Box(3.5, W + 2.0, 6.5)

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
        ("driving_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, False)),
        ("driving_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, False)),
        ("main_driving_wheel_L", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, 0.0, True, True)),
        ("main_driving_wheel_R", 0.03, RHO_CAST_IRON, lambda: build_boxpok_wheel(cfg, q_deg, False, True)),
        ("return_crank_L", 0.015, RHO_STAINLESS, lambda: build_return_crank(cfg, True)),
        ("return_crank_R", 0.015, RHO_STAINLESS, lambda: build_return_crank(cfg, False)),
        ("main_rod", 0.02, RHO_STAINLESS, lambda: build_main_rod(cfg)),
        ("coupling_rod", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_main(cfg)),
        ("coupling_rod_front", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, True)),
        ("coupling_rod_rear", 0.02, RHO_STAINLESS, lambda: build_coupling_rod_end(cfg, False)),
        ("eccentric_rod", 0.015, RHO_STAINLESS, lambda: build_eccentric_rod(cfg)),
        ("radius_rod", 0.015, RHO_STAINLESS, lambda: build_radius_rod(cfg)),
        ("combination_lever", 0.015, RHO_STAINLESS, lambda: build_combination_lever(cfg)),
        ("union_link", 0.015, RHO_STAINLESS, lambda: build_union_link(cfg)),
        ("lift_link", 0.015, RHO_STAINLESS, lambda: build_lift_link(cfg)),
        ("motion_bracket_L", 0.03, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, True)),
        ("motion_bracket_R", 0.03, RHO_CAST_IRON, lambda: build_motion_bracket(cfg, False)),
        ("axle_pump_eccentric", 0.02, RHO_BRASS, lambda: build_axle_pump_eccentric(cfg)),
        ("leading_truck_frame", 0.03, RHO_CAST_IRON, lambda: build_leading_truck_frame(cfg)),
        ("trailing_truck_frame", 0.03, RHO_CAST_IRON, lambda: build_trailing_truck_frame(cfg)),
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
        print(
            f"           -> 质量 = {mp['mass_kg']*1000:6.2f} g | "
            f"质心 = [{mp['com_m'][0]*1000:+.1f}, {mp['com_m'][1]*1000:+.1f}, {mp['com_m'][2]*1000:+.1f}] mm"
        )

    args.mass_out.parent.mkdir(parents=True, exist_ok=True)
    args.mass_out.write_text(json.dumps(mass_db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] 已导出 {len(tasks)} 个 STL 网格，并将精确材料惯量库写入: {args.mass_out}")


if __name__ == "__main__":
    main()
