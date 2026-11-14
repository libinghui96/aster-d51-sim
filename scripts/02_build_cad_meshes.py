#!/usr/bin/env python3
"""
scripts/02_build_cad_meshes.py
Aster Hobby JNR D51 (1/32, Gauge 1) 全套参数化水密 CAD 建模 + OpenCascade 材料密度积分惯量器
(P0-C V2-MECH-TRUE + P0-D V2-PIPING-LUBE-TRUE 版)。

核心升级：
  1. 修复 Gate 2 布尔共面吞噬与对称缺陷：锅筒—烟室—火室保留 >=0.5mm 轴向重叠，烟室前门采用无退化双锥碟盖，
     确保 x ∈ [92, 162] mm 烟室壳体连续无断层；修复 trailing_truck_frame 镜像符号笔误 (|min_y+max_y| < 1e-4mm)。
  2. P0-C 走行与华氏配气原厂级还原：
     - 2-4 半月板窄弧体 (3.8mm 宽，消除底部肥大圆帽) + 精确 exp_link_tail_cant_deg (~4.3527°) 后折下尾耳，实现滑槽 0.00° 对齐；
     - 2-19(L)/2-12(R) 偏心杆前端改为内侧单剪扁眼叠搭 (Single-Shear Lap)，同轴并入 2-2 一字槽台阶销栓与内侧螺母；
     - 全车华氏与走行铰接轴线一体合并 2-2, 2-21, 1-21/24/26, 7-13, 7-14/17 台阶销栓与螺母，彻底消除空洞黑孔；
     - 重建 イラスト 2 双层 4 孔镂空配气桁架梁 (2-9/14~17/20 モーションプレート) + 耳轴轴承盖 (2-5~8)；
     - 将上置亮面不锈钢单滑杆 1-33 (+ 1-30/2-13 支座 + 20-1/2/8 前盖活塞尻棒护套) 独立拆分为 slide_bar_L/R.stl；
     - 新增 cab_reach_rod.stl (8-18/8-20)、drain_cock_linkage.stl (8-5/1-16/17)、chassis_equalizer_brakes.stl (3 均衡梁板簧 + 5-1~4 闸瓦)；
     - 锅炉外衣补齐 15-3 假逆止阀 (无下垂假水管!)、28-11/12 除烟板、22-11 空压机、22-13 储风缸、28-8 消音器、30-2 假汽笛、15-8 发电机、25-3 前甲板注油活门、25-7 逆转罩。
  3. P0-D 四回路真实内脏分材质水密建模 (车轮外侧零水管!):
     - boiler_firebox_flues.stl (C1100 紫铜: 内火室 + 顶板撑 + 上2粗ø11.5过热烟管 + 下3细ø8.0火管 = 5 本煙管)；
     - superheater_piping.stl (SUS 不锈钢: 后壁 11-10 加减阀 -> 13-9 炉内管 -> 17-6 单空心螺栓 + 17-5 双U四程过热器 -> 13-4/7-9 -> 7-8/20-7 喷嘴 + 18-28 裙罩 + 11-12/13-8/17-3 通风嘴)；
     - feedwater_piping.stl (黄铜/紫铜: 100% 车架内 |Y|<=11.0mm 向后走线: 6-10 -> 6-5 轴动泵 -> 6-6 向后送水 -> 6-8 -> 后上部 13-7 逆止阀 + 4-16/4-17/4-15 旁通回水 + 13-11 -> 火室底 13-2 逆止阀)；
     - displacement_lubricator.stl (BS 黄铜: 前车架中央卧式 4-2 油箱 |Y|<=10.5mm + 4-1 顶盖 + 毛细油管汇入 13-4/7-9 三通)；
     - boiler_aux_valves.stl (BS 黄铜: 28-6x2 安全阀 + 11-4~7 水面计 + 13-1 虹吸压力表 + 11-1~3/13-3/11-8 车底真汽笛 + 11-16/17 后壁排污阀)；
     - alcohol_burner.stl (BS 黄铜: 2x2 四头立式 10mm 燃烧器 23-1 + 23-2 托架)；
     - tender_internals.stl (BS 黄铜: 31-5 手摇泵 + 36-4 130ml 酒精箱 + 36-6 针阀 + 34-2 恒液位サンプ + 31-12/13/14 立管接头)。
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from build123d import (
    Box,
    Compound,
    Cone,
    Cylinder,
    Plane,
    PolarLocations,
    Pos,
    Rot,
    SlotOverall,
    export_stl,
    extrude,
    mirror,
)

# 原厂金属材质密度表 (kg/m³)
RHO_COPPER = 8940.0             # 无氧紫铜 C1100 (内火室、顶板、5根烟管、管板)
RHO_BRASS = 8500.0              # 黄铜 C3604/BS (气缸、滑阀、6-5 轴动泵、4-2 注油箱、阀件、轴箱、滑块、偏心环)
RHO_STAINLESS = 7930.0          # 不锈钢 SUS304 (1-33 滑杆、17-5 双U过热器、连杆系、华氏五杆、台阶销栓、逆转轴/拉杆)
RHO_CAST_IRON = 7350.0          # 铸铁/碳钢 FC/S45C (Boxpok 动轮、镂空配气桁架、均衡梁闸瓦、台车架、转向架)
RHO_COMPOSITE_VALVE = 8250.0    # 1-6 黄铜滑阀 + 1-22 不锈钢阀芯叉杆复合体
RHO_COMPOSITE_PISTON = 8200.0   # 1-28 黄铜活塞 + SUS 活塞杆 + 1-31 十字头 + 1-32 滑块夹板复合体
RHO_BURNER_ASSY = 8200.0        # 23-1 四头黄铜燃烧器 + 陶瓷纤维灯芯总成


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
    """
    构建左侧标准姿态 (+Y 为外侧、-Y 为内侧) 的原厂台阶销栓 (段付ねじ + 内侧锁紧螺母):
      - 外侧带一字起子槽圆帽 (y_head_outer 处向外凸起)；
      - 实心圆柱销轴贯通孔心 (消除黑洞穿透)；
      - 内侧锁紧螺母帽 (y_nut_inner 处向内收尾)。
    """
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
    """在水平面内连接 (x0, y0) 与 (x1, y1) 的平滑斜置矩形杆段 (含两端圆柱过渡防缺口)。"""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    yaw_deg = math.degrees(math.atan2(dy, dx))
    bar = Pos(0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), z_c) * Rot(0, 0, yaw_deg) * Box(L, thick_y, height_z)
    c0 = Pos(p0[0], p0[1], z_c) * Cylinder(radius=0.5 * thick_y, height=height_z)
    c1 = Pos(p1[0], p1[1], z_c) * Cylinder(radius=0.5 * thick_y, height=height_z)
    return bar + c0 + c1


# ==================== 1. 动轮系、主连杆、三段铰接耦合连杆与 7-10 回动曲柄 ====================

def build_boxpok_wheel(cfg: Dict, phase_deg: float, is_left: bool, is_main_driver: bool = False):
    """JNR D51 Boxpok 箱型动轮 (イラスト 5, 含实心阶梯曲柄销与外端止退挡圈 7-14/17)。"""
    d_tread = float(cfg["wheels_and_axles"]["driving_wheel_diameter"])
    r_tread = 0.5 * d_tread
    r_flange = r_tread + 1.4
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
        # 三阶主曲柄销: 耦合连杆段 -> 主连杆大头段 -> 7-10 回动曲柄抱箍颈 (全程重叠 >=0.5mm)
        pin_coup = Pos(r_crank, 0, 4.1) * Cylinder(radius=1.75, height=5.2)
        pin_main = Pos(r_crank, 0, 8.5) * Cylinder(radius=1.70, height=4.6)
        pin_ret = Pos(r_crank, 0, 11.4) * Cylinder(radius=1.45, height=2.2)
        crank_pin = pin_coup + pin_main + pin_ret
    else:
        # 从动轮曲柄销 + 7-14/17 外端一字槽止退压盖 (贴合耦合连杆外侧面 |Y|<=29.7mm，留足滑杆横向净空)
        crank_pin = Pos(r_crank, 0, boss_z + 2.0) * Cylinder(radius=1.72, height=4.4)
        cap = Pos(r_crank, 0, boss_z + 4.25) * Cylinder(radius=2.25, height=0.65)
        cap -= Pos(r_crank, 0, boss_z + 4.50) * Box(4.8, 0.42, 0.40)
        crank_pin += cap

    wheel = wheel + center_hub + crank_pad + crank_pin - Cylinder(radius=2.75, height=w_tire + 10.0)

    wheel_mj = Rot(-90, 0, 0) * wheel
    if not is_left:
        wheel_mj = mirror(wheel_mj, about=Plane.XZ)
    if abs(phase_deg) > 1e-6:
        wheel_mj = Rot(0, -phase_deg, 0) * wheel_mj
    return wheel_mj


def build_return_crank(cfg: Dict, is_left: bool = True):
    """7-10 回动曲柄臂 (SUS 不锈钢): 内叠层抱箍 (dy=-1.45mm) + 主销端盖 + 外伸偏心台阶销 (承接偏心棒后眼)。"""
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
    main_pin_cap = Pos(0, y_web + 0.85, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=0.6)
    pinch_lug = Pos(-2.4, y_web, 1.35) * Box(3.0, 1.4, 2.7)
    pinch_bolt = Pos(-2.8, y_web, 2.7) * Cylinder(radius=0.65, height=1.2)
    pinch_slit = Pos(-3.1, y_web, 0.0) * Box(3.0, 2.2, 0.45)
    web = Pos(0.5 * arm_len, y_web, 0) * Box(arm_len, 1.2, 3.4)
    ecc_hub = Pos(arm_len, y_web, 0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=1.4)

    # 向外侧伸出的台阶偏心销 + 一字槽圆帽 (锁紧偏心杆后圆眼 y_local ∈ [-0.65, +0.95])
    ecc_pin = Pos(arm_len, 0.15, 0) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=2.2)
    ecc_cap = Pos(arm_len, 1.30, 0) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=0.65)
    ecc_cap -= Pos(arm_len, 1.55, 0) * Box(4.0, 0.40, 0.42)

    crank = clamp_hub + main_pin_cap + pinch_lug + pinch_bolt + web + ecc_hub + ecc_pin + ecc_cap - pinch_slit
    crank -= Pos(0, y_web - 0.6, 0) * Rot(90, 0, 0) * Cylinder(radius=1.42, height=1.8)
    crank_L = Rot(0, pitch_deg, 0) * crank

    if is_left:
        return crank_L
    return Rot(0, -q_deg, 0) * mirror(crank_L, about=Plane.XZ)


def build_main_rod(cfg: Dict):
    """7-11 / 7-16 主连杆 (SUS 不锈钢, 双面对称工字槽杆, 小头嵌入十字头双颊并由 7-13 台阶销贯通)。"""
    L_main = float(cfg["cylinder_and_rods"]["main_rod_length"])
    d_small = float(cfg["cylinder_and_rods"]["crosshead_pin_dia"])
    thickness_y, big_boss_y, small_boss_y = 2.2, 2.7, 2.7
    h_big, h_small, h_shank = 6.6, 4.8, 3.5

    big_end = Box(8.2, big_boss_y, h_big)
    big_oil_cup = Pos(0, 0, 0.5 * h_big + 0.5) * Cylinder(radius=1.25, height=1.5)
    small_end = Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * h_small, height=small_boss_y)
    shank = Pos(0.5 * L_main, 0, 0) * Box(L_main - 1.5, thickness_y, h_shank)
    rod = big_end + big_oil_cup + small_end + shank

    flute_len = L_main - 14.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * L_main, sgn * (0.5 * thickness_y - 0.32), 0) * Box(flute_len, 0.70, h_shank - 1.3)

    rod -= Rot(90, 0, 0) * Cylinder(radius=1.72, height=big_boss_y + 4.0)
    rod -= Pos(L_main, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.5 * d_small + 0.02, height=small_boss_y + 4.0)
    return rod


def build_coupling_rod_main(cfg: Dict):
    """7-12 / 7-15 分段铰接式四轴耦合连杆 — 主中段 (第3轴 <-> 第2轴, 两侧对称)。"""
    w_cfg = cfg["wheels_and_axles"]
    x_d2 = float(w_cfg["axle_x"]["drive_2"])
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    x_kr, x_kf = -k_off, x_d2 + k_off
    shank_t, boss_t, shank_h = 2.3, 3.1, 3.4

    rod = Pos(0.5 * x_d2, 0, 0) * Box(x_d2, shank_t, shank_h)
    for x_pos, r_b in [(0.0, 4.0), (x_d2, 3.4)]:
        rod += Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=r_b, height=boss_t)
        rod += Pos(x_pos, 0, r_b + 0.4) * Cylinder(radius=1.1, height=1.4)

    tongue_t = 1.3
    rod += Pos(0.5 * x_kr, 0, 0) * Box(abs(x_kr) + 1.0, shank_t, shank_h - 0.2)
    rod += Pos(x_kr, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=tongue_t)
    rod += Pos(x_d2 + 0.5 * k_off, 0, 0) * Box(k_off + 1.0, shank_t, shank_h - 0.2)
    rod += Pos(x_kf, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=2.1, height=tongue_t)

    flute_len = x_d2 - 12.0
    for sgn in (-1.0, 1.0):
        rod -= Pos(0.5 * x_d2, sgn * (0.5 * shank_t - 0.32), 0) * Box(flute_len, 0.70, shank_h - 1.2)

    for x_pos in (0.0, x_d2):
        rod -= Pos(x_pos, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.76, height=boss_t + 4.0)
    for x_k in (x_kr, x_kf):
        rod -= Pos(x_k, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=0.95, height=boss_t + 4.0)
    return rod


def build_coupling_rod_end(cfg: Dict, is_front: bool = True):
    """7-12 / 7-15 耦合连杆前/后铰接段 (含贯通实心铰接销 7-14/17 双端圆帽，消除铰接黑洞)。"""
    w_cfg = cfg["wheels_and_axles"]
    k_off = float(w_cfg.get("coupling_knuckle_offset", 8.5))
    dx = (float(w_cfg["axle_x"]["drive_1"]) - (float(w_cfg["axle_x"]["drive_2"]) + k_off)) if is_front else (
        float(w_cfg["axle_x"]["drive_4"]) - (-k_off)
    )
    L_seg = abs(dx)
    sgn_x = 1.0 if dx >= 0 else -1.0
    shank_t, boss_t, shank_h, boss_r = 2.3, 3.1, 3.3, 3.4

    clevis = Rot(90, 0, 0) * Cylinder(radius=2.3, height=boss_t) + Pos(sgn_x * 1.8, 0, 0) * Box(3.6, boss_t, 4.1)
    clevis -= Pos(-sgn_x * 0.8, 0, 0) * Box(4.6, 1.5, 6.0)
    # 贯通实心铰接销 + 对称双端铆帽 (控制外沿 |Y| <= 29.7mm)
    knuckle_pin = Rot(90, 0, 0) * Cylinder(radius=0.93, height=boss_t + 0.3) + (
        Rot(90, 0, 0) * Cylinder(radius=1.40, height=boss_t + 0.3)
        - Rot(90, 0, 0) * Cylinder(radius=1.45, height=boss_t - 0.2)
    )

    shank = Pos(0.5 * dx, 0, 0) * Box(L_seg, shank_t, shank_h)
    wheel_boss = Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=boss_r, height=boss_t)
    oil_cup = Pos(dx, 0, boss_r + 0.4) * Cylinder(radius=1.05, height=1.3)
    rod = clevis + knuckle_pin + shank + wheel_boss + oil_cup

    flute_len = L_seg - 10.5
    if flute_len > 6.0:
        for sy in (-1.0, 1.0):
            rod -= Pos(0.5 * dx, sy * (0.5 * shank_t - 0.32), 0) * Box(flute_len, 0.70, shank_h - 1.2)

    rod -= Pos(dx, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.74, height=boss_t + 4.0)
    return rod


# ==================== 2. 华氏五杆、单剪叠搭与全套台阶销栓 (P0-C V2-MECH-TRUE) ====================

def build_eccentric_rod(cfg: Dict, is_left: bool = True):
    """
    2-19 (左) / 2-12 (右) 偏心棒 (イラスト 2, SUS 不锈钢):
      - 后端 (0, +0.15, 0) 圆眼套入 7-10 回动曲柄偏心销；
      - 前端 (+L_ecc, -1.40, 0) 采用原厂 **内侧单剪扁眼 (Single-Shear Flat Eye, y ∈ [-2.05, -0.75])**,
        严丝合缝叠搭在 2-4 半月板下尾耳内侧面，彻底淘汰错误的宽双叉头！
    """
    vg = cfg["walschaerts_valve_gear"]
    L_ecc = float(vg["eccentric_rod_length"])
    y_rear = float(vg["y_layer_offsets"]["eccentric_rod_rear_dy"])          # +0.15 mm
    y_front = float(vg["y_layer_offsets"]["eccentric_rod_front_lap_dy"])    # -1.40 mm

    rear_boss = (
        Pos(0, y_rear, 0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=1.55)
        + Pos(0, y_rear, 2.35) * Cylinder(radius=0.80, height=0.9)
    )
    front_flat_eye = Pos(L_ecc, y_front, 0) * Rot(90, 0, 0) * Cylinder(radius=2.05, height=1.30)
    shank = make_connecting_bar_xy((1.2, y_rear), (L_ecc - 1.0, y_front), 0.0, thick_y=1.20, height_z=2.35)

    rod_L = rear_boss + shank + front_flat_eye
    rod_L -= Pos(0, y_rear, 0) * Rot(90, 0, 0) * Cylinder(radius=1.18, height=5.0)
    rod_L -= Pos(L_ecc, y_front, 0) * Rot(90, 0, 0) * Cylinder(radius=1.02, height=5.0)
    return rod_L if is_left else mirror(rod_L, about=Plane.XZ)


def build_expansion_link(cfg: Dict, is_left: bool = True):
    """
    2-4 月牙加减链 (加減リンク, イラスト 2 & 9, SUS 不锈钢):
      - 原点 (0,0,0) 为定轴耳轴中心，圆弧滑槽中心位于 (+R_arc, 0, 0) (R=64.0mm)；
      - 窄弧流线本体 (body_w = 3.8mm, slot_w = 1.8mm)，彻底消除底部肥大 r=2.9mm 圆帽凸起；
      - 下尾耳精确按 exp_link_tail_cant_deg (~4.352738°) 斜置至 (x_tail, +0.25, z_tail)，与 solve_side_pose 零角度偏差；
      - 同轴合并 2-2 段付ねじ (外侧一字槽圆头 + ø2.0mm 销颈 + 内侧 M1.7 螺母)，锁紧内侧偏心棒扁眼；
      - 外侧拱桥跨座 (y ∈ [+2.05, +3.00]) + 外耳轴颈 (y ∈ [+2.80, +5.05]) 供 2-5~8 镂空桁架轴承盖抱持。
    """
    vg = cfg["walschaerts_valve_gear"]
    R_arc = float(vg["radius_rod_length"])
    r_tail = float(vg["exp_link_tail_radius"])
    cant_rad = math.radians(float(vg["exp_link_tail_cant_deg"]))
    slot_half = float(vg["die_slot_half_travel"])

    thick_y = 2.2
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

    # 平滑渐缩斜置下尾耳 (y ∈ [-0.65, +1.15], 贴合外叠层) + 同轴 2-2 段付ねじ台阶销栓
    x_tail = r_tail * math.sin(cant_rad)
    z_tail = -r_tail * math.cos(cant_rad)
    y_ear = 0.25
    ear_t = 1.80
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
        x=x_tail, z=z_tail, y_nut_inner=-2.75, y_head_outer=+1.95, r_shank=1.00, r_head=1.80, r_nut=1.55
    )

    # 外侧拱形耳轴跨座 (避开滑槽内 2-3 滑块与外端螺钉帽 y <= +1.80mm) + 外耳轴颈
    standoff_f = Pos(+1.42, 1.55, 0) * Box(0.95, 1.3, 4.4)
    standoff_r = Pos(-1.42, 1.55, 0) * Box(0.95, 1.3, 4.4)
    trun_bridge = Pos(0.0, 2.50, 0) * Box(3.8, 0.95, 4.6)
    trun_pin = Pos(0.0, 3.95, 0) * Rot(90, 0, 0) * Cylinder(radius=1.72, height=2.2)
    trun_cap = Pos(0.0, 5.10, 0) * Rot(90, 0, 0) * Cylinder(radius=2.25, height=0.55)

    link_L = crescent_body + tail_neck + tail_boss + pin_2_2 + standoff_f + standoff_r + trun_bridge + trun_pin + trun_cap
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


def build_die_block(cfg: Dict, is_left: bool = True):
    """2-3 月牙滑块 (BS 黄铜滑块本体 + 同轴 2-21 薄头台阶销与内侧螺母，避让外侧耳轴跨座)。"""
    block = Box(3.6, 2.1, 3.1)
    pin_2_21 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-2.85, y_head_outer=+1.78, r_shank=0.85, r_head=1.42, r_nut=1.35, slot_w=0.36
    )
    part_L = block + pin_2_21
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_radius_rod(cfg: Dict, is_left: bool = True):
    """
    2-1 心向棒 (SUS 不锈钢):
      - 后段与中段位于内侧单剪层 (y_bar = -1.70mm, 厚度 1.15mm)，在半月板内侧面顺畅往复无干涉；
      - 前端平滑过渡至 y=0 双叉头抱合 1-23 合并杆次上孔，并一体合并 1-21 台阶销栓与内侧螺母。
    """
    vg = cfg["walschaerts_valve_gear"]
    L_rad = float(vg["radius_rod_length"])
    L_back = float(vg["radius_rod_back_ext"])
    y_bar = float(vg["y_layer_offsets"]["radius_rod_bar_dy"])  # -1.70 mm
    t_bar = 1.15

    back_eye = Pos(-L_back, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=1.75, height=t_bar)
    die_hub = Pos(0.0, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=t_bar)
    rear_span = Pos(-0.5 * L_back, y_bar, 0) * Box(L_back, t_bar, 2.2)
    mid_span = Pos(22.0, y_bar, 0) * Box(44.0, t_bar, 2.5)
    trans_span = make_connecting_bar_xy((43.5, y_bar), (L_rad - 2.2, 0.0), 0.0, thick_y=1.25, height_z=2.5)

    front_clevis = (
        Pos(L_rad, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=3.4)
        + Pos(L_rad - 1.5, 0, 0) * Box(3.0, 3.4, 3.5)
        - Pos(L_rad + 0.6, 0, 0) * Box(4.2, 1.70, 5.0)
    )
    pin_1_21 = make_stepped_pin_L(
        x=L_rad, z=0.0, y_nut_inner=-2.35, y_head_outer=+2.35, r_shank=0.88, r_head=1.55, r_nut=1.40
    )

    rod_L = back_eye + die_hub + rear_span + mid_span + trans_span + front_clevis + pin_1_21
    rod_L -= Pos(0.0, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=0.87, height=4.0)
    rod_L -= Pos(-L_back, y_bar, 0) * Rot(90, 0, 0) * Cylinder(radius=0.87, height=4.0)
    return rod_L if is_left else mirror(rod_L, about=Plane.XZ)


def build_combination_lever(cfg: Dict):
    """1-23 合併てこ (SUS 不锈钢, 中心对称单板三眼杠杆, 厚度 1.50mm 嵌入上下中三组双叉口内)。"""
    vg = cfg["walschaerts_valve_gear"]
    L_comb = float(vg["comb_lever_total_length"])
    L_mv = float(vg["comb_lever_lap_lead_offset"])
    z_mid = L_comb - L_mv
    t_web = 1.50

    bot_eye = Rot(90, 0, 0) * Cylinder(radius=1.80, height=t_web)
    lower_shank = Pos(0, 0, 0.5 * z_mid) * Box(2.35, t_web, z_mid)
    mid_boss = Pos(0, 0, z_mid) * Rot(90, 0, 0) * Cylinder(radius=2.00, height=t_web)
    upper_neck = Pos(0, 0, z_mid + 0.5 * L_mv) * Box(2.50, t_web, L_mv)
    top_boss = Pos(0, 0, L_comb) * Rot(90, 0, 0) * Cylinder(radius=1.90, height=t_web)

    lever = bot_eye + lower_shank + mid_boss + upper_neck + top_boss
    for zp in (0.0, z_mid, L_comb):
        lever -= Pos(0, 0, zp) * Rot(90, 0, 0) * Cylinder(radius=0.90, height=5.0)
    return lever


def build_union_link(cfg: Dict, is_left: bool = True):
    """1-27 先导短连杆 (SUS 不锈钢, 两端双叉抱合 + 1-26 台阶销栓与内侧螺母)。"""
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
    """2-10 逆转吊杆 (SUS 不锈钢: 上叉口 y=0 抱合逆转摇臂, 下叉口 y=-1.70mm 抱合心向棒尾眼 + 双端 2-21 台阶销)。"""
    vg = cfg["walschaerts_valve_gear"]
    L_lift = float(vg["lift_link_length"])
    y_bot = float(vg["y_layer_offsets"]["lift_link_bot_clevis_dy"])  # -1.70 mm

    top_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.80, height=3.3)
        + Pos(0, 0, -1.1) * Box(3.0, 3.3, 2.2)
        - Pos(0, 0, 0.6) * Box(4.0, 1.95, 3.4)
    )
    bot_clevis = (
        Pos(0, y_bot, -L_lift) * Rot(90, 0, 0) * Cylinder(radius=1.80, height=3.1)
        + Pos(0, y_bot, -L_lift + 1.1) * Box(3.0, 3.1, 2.2)
        - Pos(0, y_bot, -L_lift - 0.6) * Box(4.0, 1.50, 3.4)
    )
    # 斜向过渡竖片连接 y=0 与 y=-1.70mm
    ang_x = math.degrees(math.atan2(y_bot, L_lift - 2.2))
    strap = Pos(0, 0.5 * y_bot, -0.5 * L_lift) * Rot(ang_x, 0, 0) * Box(2.1, 1.35, math.hypot(L_lift - 1.6, y_bot))

    pin_top = make_stepped_pin_L(0.0, 0.0, -2.30, +2.30, r_shank=0.85, r_head=1.50, r_nut=1.35)
    pin_bot = make_stepped_pin_L(0.0, -L_lift, y_bot - 2.20, y_bot + 2.20, r_shank=0.85, r_head=1.50, r_nut=1.35)
    link_L = top_clevis + bot_clevis + strap + pin_top + pin_bot
    return link_L if is_left else mirror(link_L, about=Plane.XZ)


# ==================== 3. 双层 4 孔镂空配气桁架梁、独立亮面单滑杆、操纵机构与均衡梁闸瓦 ====================

def build_motion_bracket(cfg: Dict, is_left: bool = True):
    """
    イラスト 2 原厂双层 4 孔镂空配气桁架梁 (2-9, 2-14~17, 2-20 モーションプレート) + 耳轴轴承盖 (2-5~8):
      - 局部原点 (0,0,0) 对齐半月板定轴中心 (trunnion_x=46.5, ±y_vg=±36.0, trunnion_z=15.5)；
      - 前后双层平行钢桁架板横跨主车架侧板 (dy=-21.5mm) 至外侧耳轴支座 (dy=+4.0mm)，加工 4 颗减重圆孔；
      - 外端垂向轴承座 + 2-5~8 螺栓压盖抱持 2-4 半月板外耳轴颈 (y_local ∈ [+3.05, +4.95])；
      - 前伸支臂 (2-13) 在 (dx=+19.0mm, dy=-3.5mm, dz=-7.3mm) 锁紧独立亮面单滑杆 1-33 尾端。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    tx, tz = float(vg["trunnion_x"]), float(vg["trunnion_z"])
    y_vg = float(rc["cylinder_center_y"]) + 3.5

    dy_chassis = 14.5 - y_vg                                # -21.5 mm (主车架外侧面 Y=14.5mm)
    dy_slidebar = float(rc["cylinder_center_y"]) - y_vg     # -3.5 mm  (单导轨中心 Y=32.5mm)
    dx_bar_tail = 65.5 - tx                                 # +19.0 mm (X=65.5mm, 远在十字头后死点 77.5mm 之后)
    dz_bar_top = 8.4 - tz                                   # -7.1 mm  (贴合单滑杆顶面 Z=7.6mm)

    # 1. 车架安装法兰底座 (Y=14.5mm, 远在车轮内侧 >4.5mm)
    frame_pad = Pos(0, dy_chassis + 0.8, 11.2) * Box(9.5, 2.0, 8.2)

    # 2. 双层 4 孔镂空横梁桁架 (Z_local ∈ [+9.8, +15.0] -> Z_world ∈ [25.3, 30.5], 高于第2动轮轮缘与月牙滑块!)
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

    # 贯穿前后双层钢板的 4 颗标志性圆孔减重窗 (4-hole lightened truss motion plate)
    for yh in (-17.2, -12.0, -6.8, -1.6):
        truss -= Pos(0.0, yh, truss_zm) * Rot(0, 90, 0) * Cylinder(radius=1.35, height=9.0)

    # 3. 外侧耳轴轴承立柱 + 2-5~8 分体螺栓轴承压盖 (y_local ∈ [+3.05, +4.90], 与半月板外跨座无干涉)
    outer_leg = Pos(0.0, +3.95, 6.2) * Box(5.0, 1.85, 12.4)
    bearing_housing = Pos(0.0, +3.95, 0.0) * Rot(90, 0, 0) * Cylinder(radius=3.1, height=1.85)
    bearing_cap_lug = Pos(0.0, +3.95, -2.2) * Box(6.6, 1.85, 1.8)
    cap_bolt_f = Pos(+2.5, +3.95, -2.5) * Cylinder(radius=0.55, height=2.2)
    cap_bolt_r = Pos(-2.5, +3.95, -2.5) * Cylinder(radius=0.55, height=2.2)

    # 4. 2-13 单滑杆尾端 L 型高架前伸支臂 (内移 0.55mm 保证对心向棒内侧净空 >= 0.95mm，底部抱紧 1-33 滑杆尾端)
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
    1-33 上置亮面不锈钢单滑杆 + 1-30/2-13 前后安装座 + 20-1/2/8 前气缸盖活塞尻棒护套 (SUS304):
      - 局部原点 (0,0,0) 与 cylinder_block 一致 (cylinder_center_x=126.0, ±32.5, 0.0)；
      - 独立于 cylinder_block.stl，使滑杆呈真实不锈钢金属亮面，且开启 X-Ray 气缸透视时常驻不透明！
    """
    bore = float(cfg["cylinder_and_rods"]["bore"])
    stroke = 2.0 * float(cfg["cylinder_and_rods"]["crank_radius"])
    cyl_len = stroke + 12.0  # 32.0 mm -> 后盖位于 x_local = -16.5mm, 前盖位于 x_local = +16.8mm

    # 1. 1-33 Laird 单上置精密矩形滑杆 (x_local ∈ [-61.5, -16.8] -> X ∈ [64.5, 109.2] mm, Z = +6.4 mm)
    x_bar_f, x_bar_r = -16.8, -61.5
    bar_len = x_bar_f - x_bar_r
    bar = Pos(0.5 * (x_bar_f + x_bar_r), 0.0, 6.4) * Box(bar_len, 2.8, 1.9)
    # 1-30 前支座 (X=109.2mm, 位于十字头前死点 107.2mm 之前)
    front_lug = Pos(-16.8, 0.0, 5.4) * Box(2.2, 3.2, 4.0)
    # 2-13 后吊耳螺栓台 (X = 65.5mm -> x_local = -60.5mm)
    rear_lug = Pos(-60.5, 0.0, 7.6) * Box(3.6, 2.8, 1.2) + Pos(-60.5, 0.0, 8.5) * Cylinder(radius=0.85, height=1.2)
    bar_assy = bar + front_lug + rear_lug

    # 2. 20-1 / 20-2 / 20-8 气缸前盖活塞先导尻棒护套 (Tail-Rod Sheath, x_local ∈ [+17.2, +29.5] -> X ∈ [143.2, 155.5])
    flange_base = Pos(0.5 * cyl_len + 1.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.1, height=1.8)
    sheath_tube = Pos(0.5 * cyl_len + 7.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.85, height=10.2)
    tip_cone = Pos(0.5 * cyl_len + 12.8, 0.0, 0.0) * Rot(0, 90, 0) * Cone(bottom_radius=1.85, top_radius=1.1, height=1.8)
    sheath_assy = flange_base + sheath_tube + tip_cone

    # 沿气缸肩部安全走廊 (|Y|=28.7mm, Z=9.2mm) 纵向贯通拉紧螺柱，避开第1动轮、合并杆与十字头
    tie_rod = (
        Pos(-16.8, -1.9, 6.4) * Box(1.4, 4.8, 1.4)
        + Pos(-16.8, -3.8, 7.8) * Box(1.4, 1.4, 4.2)
        + Pos(0.5 * (-16.8 + 17.8), -3.8, 9.2) * Box(36.0, 1.3, 1.3)
        + Pos(17.8, -3.8, 4.6) * Box(1.4, 1.4, 10.5)
        + Pos(17.8, -1.9, 0.0) * Box(1.4, 4.8, 1.4)
    )

    part_L = bar_assy + sheath_assy + tie_rod
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_piston_crosshead(cfg: Dict, is_left: bool = True):
    """
    1-28 活塞 + 活塞杆 + 1-31 双颊十字头 + 1-32 上导轨滑块抱夹片 + 7-13 一字槽十字头台阶销:
      - 滑块槽留出充足净空包容 1-33 滑杆 (2.8 x 1.9 mm @ z=6.4mm)；
      - 十字头双颊抱合主连杆小头，同轴合并 7-13 台阶销帽与内侧螺母，彻底消除十字头黑孔。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    bore = float(rc["bore"])
    L_main = float(rc["main_rod_length"])
    cyl_cx = float(rc["cylinder_center_x"])
    drop_z = float(vg["crosshead_drop"])
    p_head_dx = cyl_cx - L_main

    piston_disc = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore - 0.1, height=4.6)
    ring_groove = Pos(p_head_dx, 0, 0) * Rot(0, 90, 0) * (
        Cylinder(radius=0.5 * bore + 0.5, height=1.5) - Cylinder(radius=0.5 * bore - 0.9, height=2.0)
    )
    piston_disc -= ring_groove
    piston_rod = Pos(0.5 * p_head_dx + 1.5, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.5, height=p_head_dx - 1.0)

    # 双颊十字头毂 (内槽宽 2.95mm 包容 2.70mm 主连杆小头, 前沿 +1.8mm 确保前死点距气缸后盖 >0.8mm)
    ch_hub = Pos(-1.4, 0, 0) * Box(6.4, 5.8, 6.8) - Pos(-2.2, 0, 0) * Box(6.4, 2.95, 7.6)
    # 1-31 上滑靴 + 1-32 下抱夹唇口 (内净空 y ∈ [-2.2, +2.2], z ∈ [4.8, 8.0] 包容 1-33 滑杆 2.8x1.9 @ z=6.4)
    shoe_outer = Pos(-2.6, 0, 6.45) * Box(8.6, 5.8, 5.1)
    bar_slot = Pos(-2.6, 0, 6.40) * Box(11.0, 4.4, 3.2)
    under_open = Pos(-2.6, 0, 4.30) * Box(11.0, 2.2, 1.6)
    top_shoe = shoe_outer - bar_slot - under_open
    neck = Pos(-1.4, 0, 3.6) * Box(5.6, 4.4, 1.8)

    # 下垂先导耳臂 + 7-13 同轴台阶销栓
    drop_arm = Pos(0, +1.85, -0.5 * drop_z) * Box(3.8, 2.1, drop_z + 2.0)
    drop_boss = Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=1.95, height=1.80)
    pin_7_13 = make_stepped_pin_L(
        x=0.0, z=0.0, y_nut_inner=-3.55, y_head_outer=+3.65, r_shank=1.48, r_head=2.35, r_nut=2.10, slot_w=0.50
    )

    part_L = piston_disc + piston_rod + ch_hub + neck + top_shoe + drop_arm + drop_boss + pin_7_13
    part_L -= Pos(0, +3.50, -drop_z) * Rot(90, 0, 0) * Cylinder(radius=0.90, height=4.0)
    return part_L if is_left else mirror(part_L, about=Plane.XZ)


def build_slide_valve_assembly(cfg: Dict, is_left: bool = True):
    """1-6 外进气平面 D 型滑阀 + 1-22 阀芯棒叉头 + 同轴 1-24 台阶销栓与内侧螺母。"""
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    v_dx = float(rc["cylinder_center_x"]) - (float(vg["trunnion_x"]) + float(vg["radius_rod_length"]))

    clevis = Box(5.8, 3.4, 4.4) - Pos(-1.2, 0, 0) * Box(4.2, 1.70, 6.0)
    pin_1_24 = make_stepped_pin_L(0.0, 0.0, -2.35, +2.35, r_shank=0.88, r_head=1.55, r_nut=1.40)

    rod_len = v_dx + 20.0
    stem = Pos(0.5 * rod_len - 2.0, -3.5, 0) * Rot(0, 90, 0) * Cylinder(radius=1.0, height=rod_len)
    cross_arm = Pos(1.6, -1.75, 0) * Box(2.6, 5.5, 3.2)

    valve_block = Pos(v_dx, -3.5, -0.5) * Box(10.2, 9.6, 5.2)
    exhaust_cavity = Pos(v_dx, -3.5, -2.2) * Box(4.8, 6.6, 2.4)
    valve_block -= exhaust_cavity

    assembly_L = clevis + pin_1_24 + cross_arm + stem + valve_block
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_reverser_shaft(cfg: Dict):
    """
    4-12 逆转器横轴 + 2-11/2-18 左右等长 11.0mm 提升臂 + 右走台外侧向上逆转摇臂 (接 8-18 逆转长拉杆):
      - 右侧操纵臂置于 y=-33.5mm, 向上高 7.0mm (顶点 z_local=+7.0 -> Z_world=33.5mm, 位于锅筒外侧、25-7 护罩内)。
    """
    rc = cfg["cylinder_and_rods"]
    vg = cfg["walschaerts_valve_gear"]
    y_vg = float(rc["cylinder_center_y"]) + 3.5
    arm_len_x = float(vg["reverser_arm_length"])

    main_bar = Rot(90, 0, 0) * Cylinder(radius=1.4, height=2.0 * y_vg + 3.0)
    reach_hub = Pos(0, -33.5, 0) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=2.2)
    reach_lever = Pos(0, -33.5, 3.5) * Box(3.0, 1.8, 7.0)
    reach_tip = Pos(0, -33.5, 7.0) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=1.8)

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
    8-18 右侧逆转长拉杆 + 8-20 前叉接头 (SUS304, イラスト 8):
      - 局部原点 (0,0,0) 位于 reverser_shaft 右操纵臂顶点 (X=25.5, Y=-33.5, Z=33.5 mm)；
      - 沿右走台外侧向后直通驾驶室 (-144.5mm -> X=-119.0mm)，通过 <equality><joint> 与 j_reverser 同步联动。
    """
    L_rod = 144.5
    front_clevis = (
        Rot(90, 0, 0) * Cylinder(radius=1.9, height=3.6)
        + Pos(-1.6, 0, 0) * Box(3.2, 3.6, 3.4)
        - Pos(+0.6, 0, 0) * Box(3.8, 1.95, 4.6)
    )
    pin = make_stepped_pin_L(0.0, 0.0, -2.35, +2.35, r_shank=0.85, r_head=1.45, r_nut=1.35)
    # 右侧向外为 -Y，镜像销帽使圆槽头朝外 (-Y)
    pin_R = mirror(pin, about=Plane.XZ)
    shaft = Pos(-0.5 * L_rod, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=0.95, height=L_rod)
    guide_collar = Pos(-72.0, 0, 0) * Rot(0, 90, 0) * Cylinder(radius=1.45, height=2.6)
    rear_clevis = Pos(-L_rod, 0, 0) * Box(4.4, 2.2, 2.8)
    return front_clevis + pin_R + shaft + guide_collar + rear_clevis


def build_drain_cock_linkage(cfg: Dict):
    """
    8-5 排液阀联动横轴 + 1-16/1-17 左右气缸底部齿轮排液旋塞对 (BS 黄铜, イラスト 1 & 8):
      - 局部原点 (0,0,0) 置于气缸正下方安全净空区 (X=124.0, Y=0.0, Z=-14.5 mm)；
      - 横轴跨度 Y ∈ [-29.0, +29.0] mm，远在先导架下方 (>4mm)、十字头/短连杆前方 (>6.5mm)，两侧完全对称！
    """
    y_cock = 29.0
    cross_shaft = Rot(90, 0, 0) * Cylinder(radius=1.1, height=2.0 * y_cock + 2.0)
    sides = None
    for sgn in (-1.0, 1.0):
        yc = sgn * y_cock
        arm = Pos(0.0, yc, 1.4) * Box(2.4, 1.8, 3.6)
        tie_bar = Pos(5.0, yc, 2.6) * Box(13.6, 1.4, 1.8)
        cock_rear = (
            Pos(0.0, yc, 2.8) * Cylinder(radius=1.8, height=3.2)
            + Pos(0.0, yc, 1.5) * Cylinder(radius=2.4, height=1.1)
            + Pos(0.0, yc, -0.8) * Cylinder(radius=0.85, height=2.6)
        )
        cock_front = (
            Pos(10.0, yc, 2.8) * Cylinder(radius=1.8, height=3.2)
            + Pos(10.0, yc, 1.5) * Cylinder(radius=2.4, height=1.1)
            + Pos(10.0, yc, -0.8) * Cylinder(radius=0.85, height=2.6)
        )
        sub = arm + tie_bar + cock_rear + cock_front
        sides = sub if sides is None else (sides + sub)
    return cross_shaft + sides


def build_chassis_equalizer_brakes(cfg: Dict):
    """
    イラスト 3 车架内侧板簧/跷跷板均衡梁 (3) + イラスト 5 四轴静态铸铁制动闸瓦与吊杆 (5-1~5-4):
      - 局部原点 (0,0,0) 对齐 loco_frame；中心严格左右对称 (|min_y + max_y| = 0)；
      - 板簧与均衡梁位于车架侧板内缘 |Y| ∈ [10.0, 11.5] mm, Z ∈ [9.2, 14.5] mm (高于车轴且避开内侧水路)；
      - 制动闸瓦置于四对动轮前下象限 (-46°, 距轮心 26.5mm > 轮缘 23.3mm + 悬挂行程)，确保全工况净空 >= 2.0mm！
    """
    w_cfg = cfg["wheels_and_axles"]
    ax_xs = [
        float(w_cfg["axle_x"]["drive_1"]),
        float(w_cfg["axle_x"]["drive_2"]),
        float(w_cfg["axle_x"]["drive_3"]),
        float(w_cfg["axle_x"]["drive_4"]),
    ]

    assy = None
    # 1. 横梁连接枕 (确保单一体连通且避开所有轮轴与水路: X=+69.75mm 与 X=-23.25mm, Z=+13.2mm)
    tie_f = Pos(0.5 * (ax_xs[0] + ax_xs[1]), 0.0, 13.2) * Box(4.0, 29.0, 2.2)
    tie_r = Pos(0.5 * (ax_xs[2] + ax_xs[3]), 0.0, 13.2) * Box(4.0, 29.0, 2.2)
    assy = tie_f + tie_r

    for sgn in (-1.0, 1.0):
        y_eq = sgn * 10.7
        y_frame_out = sgn * 15.6
        y_shoe = sgn * 23.4

        # 四组叠片板簧 (高于车轴最大跳动高度 Z=+6.3mm)
        for ax in ax_xs:
            sp1 = Pos(ax, y_eq, 12.8) * Box(18.0, 1.5, 1.6)
            sp2 = Pos(ax, y_eq, 14.0) * Box(12.5, 1.5, 1.4)
            buckle = Pos(ax, y_eq, 13.3) * Box(3.2, 2.0, 3.0)
            assy += sp1 + sp2 + buckle

        # 前后两组跷跷板均衡摇梁
        for xm in (0.5 * (ax_xs[0] + ax_xs[1]), 0.5 * (ax_xs[2] + ax_xs[3])):
            rocker = Pos(xm, y_eq, 11.4) * Box(26.0, 1.4, 2.4)
            fulcrum = Pos(xm, y_eq, 12.6) * Rot(90, 0, 0) * Cylinder(radius=1.8, height=2.2)
            assy += rocker + fulcrum

        # 四对动轮前下象限铸铁制动闸瓦 (5-1~5-4): 沿轮缘切线 +44° 布置，距轮缘外径净空 > 1.8mm！
        for ax in ax_xs:
            sx, sz = ax + 18.8, -19.2
            shoe = Pos(sx, y_shoe, sz) * Rot(0, +44, 0) * Box(2.4, 2.4, 7.2)
            # 横向制动梁向内固接于主车架下缘 (Z=-3.2mm, X=ax+19.5mm 避开前后两轮轮缘!)
            hanger_v = Pos(ax + 19.5, y_frame_out, -11.4) * Box(2.2, 1.8, 17.2)
            cross_pin = Pos(ax + 19.5, 0.5 * (y_frame_out + y_shoe), sz - 0.4) * Rot(90, 0, 0) * Cylinder(
                radius=1.10, height=abs(y_shoe - y_frame_out) + 1.6
            )
            pad_f = Pos(ax + 19.5, sgn * 14.2, -3.2) * Box(3.8, 2.6, 3.0)
            assy += shoe + hanger_v + cross_pin + pad_f

        # 车架外侧纵向联系杆连接前后两组闸瓦支座以构成单连通闭合水密网格
        long_tie = Pos(0.5 * (ax_xs[0] + ax_xs[3]) + 19.5, sgn * 14.2, 12.8) * Box(
            (ax_xs[0] - ax_xs[3]) + 4.0, 1.6, 1.8
        )
        riser_f = Pos(ax_xs[0] + 19.5, sgn * 14.2, 4.8) * Box(2.4, 1.6, 17.5)
        riser_r = Pos(ax_xs[3] + 19.5, sgn * 14.2, 4.8) * Box(2.4, 1.6, 17.5)
        assy += long_tie + riser_f + riser_r

    return assy


def build_axle_pump_eccentric(cfg: Dict):
    """6-3 / 6-4 第 3 轴中央给水泵黄铜偏心轮盘与偏心环连杆 (含尾端实心叉销，消除黑孔)。"""
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
    """19-2 先导径向摆臂铸铁台车架 + 20-6 R2m/R3m 双孔调节槽座 (100% 左右对称)。"""
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
        beam = Pos(-26.0, sgn * 0.5 * (y_in + 4.0), 0) * Rot(0, 0, sgn * -14.5) * Box(52.0, 3.2, 3.8)
        sub = jbox + beam
        arms = sub if arms is None else (arms + sub)

    frame = cross_bolster + pivot_tongue + boss_r2m + boss_r3m + arms
    frame -= Pos(-52.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Pos(-62.0, 0, 0) * Cylinder(radius=1.6, height=8.0)
    frame -= Rot(90, 0, 0) * Cylinder(radius=2.1, height=2.0 * y_in + 8.0)
    return frame


def build_trailing_truck_frame(cfg: Dict):
    """19-11 从轮 Delta 式铸铁台车架 (修复 sgn*(y_out + 1.2) 镜像对称 bug，|min_y+max_y| < 1e-4mm)。"""
    dx_ax = -54.0
    y_out = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.0
    pivot_nose = Cylinder(radius=4.5, height=5.0) + Pos(-6.0, 0, 0) * Box(12.0, 10.0, 4.4)
    rear_tie = Pos(dx_ax - 6.5, 0, 1.2) * Box(4.2, 2.0 * y_out + 2.0, 3.6)

    sides = None
    for sgn in (-1.0, 1.0):
        beam = Pos(0.5 * dx_ax, sgn * 0.52 * y_out, 0.5) * Rot(0, 0, sgn * 26.5) * Box(58.0, 3.2, 4.2)
        abox = (
            Pos(dx_ax, sgn * y_out, 0.4) * Box(10.0, 4.4, 9.2)
            + Pos(dx_ax, sgn * (y_out + 1.2), 3.6) * Box(12.0, 3.6, 3.0)
            - Pos(dx_ax, sgn * y_out, 0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=7.0)
        )
        sub = beam + abox
        sides = sub if sides is None else (sides + sub)

    return pivot_nose + rear_tie + sides - Cylinder(radius=1.8, height=8.0)


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
    1-4 气缸体与外进气蒸汽室壳体 (已将 1-33 亮面滑杆与 20-1 前盖尻棒护套分离至 slide_bar_L/R.stl！):
      - 按 イラスト 1 还原前部主保温套筒 + 后段精车气缸颈 (r=7.05mm)，全行程避开第1动轮轮缘、合并杆与先导短连杆 (>=0.85mm)；
      - 蒸汽室填料函 gland_gp5 严格对准 y_local=0.0 阀芯杆轴线；X-Ray 透视时本壳体半透明，内部活塞与亮面单滑杆清晰可见。
    """
    bore = float(cfg["cylinder_and_rods"]["bore"])
    stroke = 2.0 * float(cfg["cylinder_and_rods"]["crank_radius"])
    z_valve = float(cfg["cylinder_and_rods"]["valve_axis_z"])
    cyl_od = bore + 7.0  # 22.0 mm

    main_barrel = Pos(+4.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od, height=21.2)
    rear_neck = (
        Pos(-9.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=6.15, height=8.8)
        + Pos(-14.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=1.6)
        - Pos(-10.2, +6.85, 0.0) * Box(14.0, 10.0, 16.0)
    )
    saddle_block = Pos(+4.8, -5.5, 4.5) * Box(20.0, 14.0, cyl_od + 3.0)
    front_cover = Pos(+15.8, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * cyl_od + 0.8, height=2.2)

    for b in (
        Pos(+17.0, 0.0, 0.0)
        * Rot(0, 90, 0)
        * PolarLocations(radius=0.5 * bore + 2.2, count=8)
        * Cylinder(radius=0.85, height=1.2)
    ):
        front_cover += b

    steam_chest = Pos(+4.8, 0.5, z_valve) * Box(21.2, 15.0, 11.0)
    gland_gp5 = Pos(-6.2, 0.0, z_valve) * Rot(0, 90, 0) * Cylinder(radius=1.9, height=1.8)

    bore_cut = (
        Pos(+5.2, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=0.5 * bore, height=19.2)
        + Pos(-8.8, -1.4, 0.0) * Rot(0, 90, 0) * Cylinder(radius=2.5, height=9.6)
    )
    assembly_L = main_barrel + rear_neck + saddle_block + front_cover + steam_chest + gland_gp5 - bore_cut
    return assembly_L if is_left else mirror(assembly_L, about=Plane.XZ)


def build_boiler_superstructure(cfg: Dict, heavy_equipped: bool = True):
    """
    锅炉烟室驾驶室外衣总成 (修复 Gate 2 布尔共面吞噬 + 补齐全部 Tier 3 静态外观件):
      - smokebox (x ∈ [91.5, 162.0]) 与 barrel (x ∈ [-112.0, 93.0]) 保留 1.5mm 实体重叠，烟室门采用无退化双锥碟盖；
      - 走台提升至真实 JNR D51 轮顶高度 Z=30.8mm，并留出逆转摇臂避让槽 + 25-7 右走台护罩；
      - 补齐 15-3 假逆止阀 (无下垂水管!)、16-2 假泥溜栓、28-11/12 除烟板、22-11 双联空压机、22-13 储风缸、
        28-8 消音器、30-2 假汽笛、15-8 涡轮发电机、25-3 前甲板注油器活门盖。
    """
    b_cfg = cfg["boiler_and_aux"]
    od_sb = float(b_cfg["smokebox_od"])
    r_sb, L_sb = 0.5 * od_sb, float(b_cfg["smokebox_length"])
    fb_l, fb_w, fb_h = [float(v) for v in b_cfg["firebox_lwh"]]
    z_axis, x_sb_front = 44.0, 162.0
    x_sb_rear = x_sb_front - L_sb        # +92.0 mm
    x_fb_rear = -112.0
    x_fb_front = x_fb_rear + fb_l        # -47.0 mm

    # 1. 烟室圆筒 (x ∈ [91.5, 162.0], 与锅筒保留 1.5mm 重叠) + 18-4~9 无退化双锥碟形烟室门
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
        # JNR D51 标志性半流线型长给水暖器砂箱烟囱罩 (ナメクジ/重装集烟罩风格)
        chimney += Pos(x_chimney - 2.0, 0, z_axis + r_sb + 10.5) * Box(22.0, 19.0, 13.0)
        chimney += Pos(x_chimney - 10.0, 0, z_axis + r_sb + 14.0) * Box(12.0, 17.0, 6.0)

    # 2. 锅筒 (x ∈ [-112.0, +93.0], 与烟室重叠 1.5mm) + 汽包砂箱 + 外火室
    b_x0, b_x1 = x_fb_rear, x_sb_rear + 1.0
    r_barrel = 28.4
    barrel = Pos(0.5 * (b_x0 + b_x1), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel, height=b_x1 - b_x0)
    for bx in [72.0, 38.0, -16.0, -60.0]:
        barrel += Pos(bx, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_barrel + 0.6, height=2.2)

    dome_base = Pos(22.0, 0, z_axis + r_barrel - 3.0) * extrude(SlotOverall(width=72.0, height=28.0), amount=14.0)
    firebox = Pos(0.5 * ((x_fb_front + 1.0) + x_fb_rear), 0, z_axis - 11.0) * Box(fb_l + 1.0, fb_w, fb_h)

    # 3. 驾驶室 (x ∈ [-164.0, -110.0], 与火室后壁重叠 2.0mm, 右前壁留出 8-18 逆转拉杆穿墙孔)
    cab_len, cab_w = 54.0, 74.0
    cab_x = x_fb_rear - 0.5 * cab_len + 2.0
    cab_box = Pos(cab_x, 0, z_axis + 4.5) * Box(cab_len, cab_w, 53.0)
    cab_roof = (Pos(cab_x - 2.0, 0, z_axis + 8.0) * Rot(0, 90, 0) * Cylinder(radius=42.0, height=cab_len + 8.0)) & (
        Pos(cab_x - 2.0, 0, z_axis + 35.0) * Box(cab_len + 8.0, cab_w + 4.0, 22.0)
    )
    cab = (
        cab_box + cab_roof
        - Pos(cab_x - 2.5, 0, z_axis + 3.5) * Box(cab_len - 3.0, cab_w - 4.0, 51.0)
        - Pos(cab_x + 2.0, 0, z_axis + 14.5) * Box(28.0, cab_w + 10.0, 18.0)
        - Pos(cab_x - 0.5 * cab_len, 0, z_axis + 2.5) * Box(8.0, 36.0, 47.0)
        - Pos(-116.0, -33.5, 33.5) * Box(18.0, 5.0, 5.6)
    )

    # 4. 真实高度左右走台 (Z=30.8mm > 动轮顶 + 蒸汽室顶) + 前端缓冲甲板与 25-3 置换注油器活门盖 + 28-11/12 除烟板
    walk_x0, walk_x1 = x_fb_rear + 1.0, x_sb_front + 6.0
    walk_len = walk_x1 - walk_x0
    walk_xm = 0.5 * (walk_x0 + walk_x1)
    walkways = None
    for sgn in (-1.0, 1.0):
        board = Pos(walk_xm, sgn * 34.4, 30.8) * Box(walk_len, 10.2, 1.5)
        inward_web = Pos(walk_xm, sgn * 27.5, 30.8) * Box(walk_len - 8.0, 5.0, 1.5)
        # 避让逆转摇臂、8-18 拉杆前叉与吊杆的走台宽裕开口窗 (X ∈ [17.5, 43.5], |Y| ∈ [29.4, 40.8])
        rev_cut = Pos(30.5, sgn * 35.1, 30.8) * Box(26.0, 11.4, 5.0)
        board_sub = board + inward_web - rev_cut

        # 28-11 / 28-12 JNR D51 标志性前部除烟板 (Smoke Deflectors, 外边界精确锁定 |Y| = 40.8mm 保证整体对称)
        defl = Pos(131.5, sgn * 40.2, 49.2) * Box(59.0, 1.2, 38.0)
        defl -= Pos(161.0, sgn * 40.2, 68.2) * Rot(0, -38, 0) * Box(18.0, 4.0, 18.0)
        defl_stay_u = Pos(144.0, sgn * 35.0, 61.0) * Box(1.8, 11.0, 1.8)
        defl_stay_l = Pos(112.0, sgn * 35.0, 58.0) * Box(1.8, 11.0, 1.8)

        # 15-3 ダミー逆止弁 (前锅筒两侧纯外观法兰座，严禁向下接水管到车轮！)
        dummy_15_3 = (
            Pos(58.0, sgn * 28.2, 44.0) * Rot(90, 0, 0) * Cylinder(radius=3.0, height=3.6)
            + Pos(58.0, sgn * 29.8, 45.2) * Cylinder(radius=2.1, height=4.8)
        )
        # 16-2 ダミー泥溜栓 (外火室下部装饰螺塞)
        dummy_16_2 = Pos(-78.0, sgn * 24.4, 24.0) * Rot(90, 0, 0) * Cylinder(radius=2.0, height=2.4)
        # 22-13 元空気溜 (两侧紧贴锅筒腰部 |Y|=27.6mm, Z=36.2mm，远离右侧 8-18 逆转拉杆 >2mm！)
        air_res = Pos(-18.0, sgn * 27.6, 36.2) * Rot(0, 90, 0) * Cylinder(radius=2.3, height=34.0)

        part = board_sub + defl + defl_stay_u + defl_stay_l + dummy_15_3 + dummy_16_2 + air_res
        walkways = part if walkways is None else (walkways + part)

    # 前甲板台阶 + 25-3 前デッキ蓋 (位于 4-1 注油箱顶盖 X=164, Y=0 正上方，远高于先导轮 Z<=6.2mm)
    front_apron = (
        Pos(x_sb_front + 3.0, 0.0, 24.0) * Box(2.2, 76.0, 14.5)
        + Pos(x_sb_front + 6.5, 0.0, 17.2) * Box(8.5, 76.0, 1.8)
    )
    hatch_25_3 = (
        Pos(165.0, 0.0, 18.4) * Box(6.4, 9.2, 1.2)
        + Pos(162.0, 0.0, 19.0) * Rot(90, 0, 0) * Cylinder(radius=0.7, height=8.0)
    )

    # Tier 3 左右专用外观细节 (均严格处于 |Y| <= 39.4mm 内，不改变外包络 [-40.8, +40.8]mm 对称性！)
    # 22-11 左走台复式空气压缩机 + 28-8 左前走台消音器
    comp_22_11 = (
        Pos(-46.0, +33.2, 37.5) * Cylinder(radius=3.4, height=12.5)
        + Pos(-46.0, +33.2, 41.5) * Cylinder(radius=3.9, height=3.2)
        + Pos(-46.0, +28.0, 37.5) * Box(5.0, 10.0, 4.0)
    )
    silencer_28_8 = (
        Pos(118.0, +34.2, 34.2) * Rot(0, 90, 0) * Cylinder(radius=2.5, height=11.0)
        + Pos(118.0, +31.0, 32.2) * Box(4.0, 6.5, 3.0)
    )
    # 30-2 锅筒顶假汽笛 + 15-8~10 右后侧涡轮发电机 + 25-7 右走台逆转摇臂桥式护罩 (前后贯通避让 8-18 拉杆)
    dummy_whistle_30_2 = Pos(-48.0, -7.5, z_axis + r_barrel + 3.0) * Cylinder(radius=1.8, height=7.5)
    gen_15_8 = (
        Pos(-84.0, -25.5, 62.5) * Rot(0, 90, 0) * Cylinder(radius=3.0, height=9.5)
        + Pos(-84.0, -21.0, 59.5) * Box(5.0, 8.5, 5.0)
    )
    cover_25_7 = (
        Pos(22.5, -34.2, 35.4) * Box(14.0, 10.6, 9.8)
        - Pos(22.5, -34.2, 34.0) * Box(17.0, 8.2, 9.4)
    )

    outer_solid = (
        smokebox + sb_saddle + door_ring + door_cone1 + door_cone2 + door_handle + hinge_u + hinge_l
        + headlight + headlight_ped + feedwater_heater + fwh_brackets + chimney
        + barrel + dome_base + firebox + cab + walkways + front_apron + hatch_25_3
        + comp_22_11 + silencer_28_8 + dummy_whistle_30_2 + gen_15_8 + cover_25_7
    )

    # 四对动轮上部轮拱避让槽 (对称保证 >= 1.9mm 全行程动轮轮缘净空且不穿透内腔!)
    wheel_cuts = None
    for ax in (93.0, 46.5, 0.0, -46.5):
        for sgn in (-1.0, 1.0):
            wc = Pos(ax, sgn * 22.5, 0.0) * Rot(90, 0, 0) * Cylinder(radius=25.8, height=9.2)
            wheel_cuts = wc if wheel_cuts is None else (wheel_cuts + wc)

    # 内部净空掏腔 (真实薄壁黄铜蒙皮质量 ~1.65kg，预留 4-12 逆转轴套管且与轮拱槽严格隔离 >=1.7mm)
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

    return outer_solid - wheel_cuts - bore_barrel - bore_smokebox - bore_firebox - bore_dome_cab - bore_chimney - rev_shaft_tunnel


# ==================== 4. P0-D 四回路真实炉内管网、车架内双水路、置换注油器、四头燃烧器与煤水车内脏 ====================

def build_boiler_firebox_flues(cfg: Dict, pip_cfg: Dict):
    """
    紫铜受压内胆 + 真实 5 本煙管总成 (C1100 银焊无氧紫铜, イラスト 12 & 17):
      - 上排 2 根粗过热烟管 (ø11.5mm OD / ø10.0mm ID, y=±11.5mm, z=46.5mm)；
      - 下排弧形 3 根细火管 (ø8.0mm OD / ø6.7mm ID, (y=±13.0, z=35.0) 与 (y=0.0, z=33.2))；
      - 100% 中心对称水密正体积网格。
    """
    pv = pip_cfg["boiler_pressure_vessel"]
    z_axis = float(pv["centerline_z_mm"])
    r_in = 0.5 * float(pv["barrel_id_mm"])        # 26.5 mm
    r_out = r_in + 0.8                            # 27.3 mm (< 28.4mm 外衣)
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

    liner_len = x_front - fb_x1 + 1.0
    liner = (
        Pos(0.5 * (x_front + fb_x1 - 1.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_out, height=liner_len)
        - Pos(0.5 * (x_front + fb_x1 - 1.0), 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_in, height=liner_len + 2.0)
    )
    front_tubeplate = Pos(x_front - 1.2, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_out, height=2.6)
    throat_plate = Pos(fb_x1 + 0.8, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_out, height=2.4)
    back_ring = (
        Pos(fb_x0 + 1.0, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_out, height=2.2)
        - Pos(fb_x0 + 1.0, 0, z_axis) * Rot(0, 90, 0) * Cylinder(radius=r_in - 3.2, height=4.0)
    )

    fb_outer = Pos(fb_xm, 0, fb_z0 + 0.5 * fb_h) * Box(fb_len + 1.0, fb_w, fb_h)
    fb_cavity = Pos(fb_xm, 0, fb_z0 + 0.5 * (fb_h - t_cu) - 0.5) * Box(
        fb_len - 2.0 * t_cu, fb_w - 2.0 * t_cu, fb_h - t_cu + 1.0
    )
    mud_ring = Pos(fb_xm, 0, fb_z0 + 1.0) * Box(fb_len + 2.4, fb_w + 3.2, 2.2) - Pos(
        fb_xm, 0, fb_z0 + 1.0
    ) * Box(fb_len - 2.0 * t_cu, fb_w - 2.0 * t_cu, 4.0)
    firehole_ring = Pos(fb_x0 - 1.2, 0, z_crown - 9.5) * Rot(0, 90, 0) * Cylinder(radius=6.6, height=3.8)

    stays = None
    for sx in (-95.0, -78.0, -61.0):
        for sy in (-12.0, -4.0, 4.0, 12.0):
            b = Pos(sx, sy, z_crown + 0.6) * Cylinder(radius=1.1, height=1.8)
            stays = b if stays is None else (stays + b)

    # 5 根紫铜烟管 (上 2 粗过热烟管 + 下 3 细火管)
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

    throat_steam_pass = Pos(fb_x1 + 0.8, 0, z_axis + 16.5) * Rot(0, 90, 0) * Cylinder(radius=6.5, height=5.0)
    firehole_bore = Pos(fb_x0 - 1.0, 0, z_crown - 9.5) * Rot(0, 90, 0) * Cylinder(radius=4.8, height=8.0)
    rev_liner_cut = Pos(25.5, 0, 26.5) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=58.0)

    return (
        liner + front_tubeplate + throat_plate + back_ring
        + (fb_outer - fb_cavity) + mud_ring + firehole_ring + stays + flues_outer
        - throat_steam_pass - flues_bore - firehole_bore - rev_liner_cut
    )


def build_superheater_piping(cfg: Dict, pip_cfg: Dict):
    """
    真实后壁加减阀 + 双 U 四程过热器 + 主蒸汽/乏汽/通风回路 (SUS304, イラスト 11, 13, 17, 18, 20):
      - 驾驶室后壁 11-10 加减阀 -> 13-9 炉内上部加减管 -> 17-6 单空心螺栓集箱；
      - 17-5 双 U 四程过热器插入上排 2 根 ø11.5mm 过热烟管 (y=±11.5, z=46.5±1.7mm)；
      - 17-4 接头 -> 13-4 主蒸汽下垂管 -> 注油三通 (122,0,18.5) -> 7-9/1-20 左右进汽管 -> 蒸汽室；
      - 7-8 排汽管 -> 20-7 乏汽喷嘴 (ø4.2mm) + 18-28 烟囱裙罩 + 11-12->13-8->17-3 通风喷嘴。
    """
    sp = pip_cfg["superheater_and_steam_piping"]
    pv = pip_cfg["boiler_pressure_vessel"]
    sh = sp["superheater_4pass"]
    br = sp["branch_steam_pipes"]
    ej = sp["smokebox_draft_ejector"]
    reg = sp["cab_regulator_11_10"]

    x_reg, _, z_reg = [float(v) for v in reg["xyz_mm"]]
    z_pipe = float(reg["internal_pipe_z_mm"])       # 60.5 mm (上部蒸汽空间)
    r_dry = 0.5 * float(reg["internal_pipe_od_mm"])
    x_turn = float(sh["x_turn_in_flue_mm"])         # -24.0 mm
    x_hdr = float(sh["x_smokebox_header_mm"])       # +102.0 mm
    r_sh = 0.5 * float(sh["tube_od_mm"])            # 1.5 mm

    up_yz = pv["five_flue_tubes"]["upper_superheater_flues"]["yz_centers_mm"]

    # 1. 驾驶室后壁 11-10/11 加減弁阀座、扇形柄 + 炉内 13-9 加減管直通前管板
    reg_body = (
        Pos(x_reg, 0.0, z_reg) * Rot(0, 90, 0) * Cylinder(radius=3.4, height=4.8)
        + Pos(x_reg - 2.6, 2.2, z_reg + 2.8) * Rot(28, 0, 0) * Box(1.6, 2.0, 9.5)
        + Pos(x_reg + 1.5, 0.0, 0.5 * (z_reg + z_pipe)) * Cylinder(radius=r_dry, height=z_pipe - z_reg + 2.0)
    )
    pipe_13_9 = Pos(0.5 * (x_reg + 1.5 + x_hdr), 0.0, z_pipe) * Rot(0, 90, 0) * Cylinder(
        radius=r_dry, height=x_hdr - (x_reg + 1.5)
    )

    # 2. 烟室 17-6 单空心螺栓过热集箱 + 17-5 双 U 四程过热器 (深入左右上排粗烟管 y=±11.5, z=46.5)
    bolt_17_6 = (
        Pos(x_hdr, 0.0, 56.5) * Cylinder(radius=3.2, height=10.5)
        + Pos(x_hdr, 0.0, 61.8) * Cylinder(radius=4.1, height=2.4)
        + Pos(x_hdr, 0.0, 53.5) * Box(6.4, 28.0, 4.6)
    )
    sh_loops = None
    L_leg = (x_hdr - 1.5) - x_turn
    xm_leg = 0.5 * ((x_hdr - 1.5) + x_turn)
    for yz in up_yz:
        yf, zf = float(yz[0]), float(yz[1])
        sgn = 1.0 if yf > 0 else -1.0
        y_a, z_a = yf - sgn * 1.6, zf + 1.6
        y_b, z_b = yf + sgn * 1.6, zf - 1.6
        leg1 = Pos(xm_leg, y_a, z_a) * Rot(0, 90, 0) * Cylinder(radius=r_sh, height=L_leg)
        leg2 = Pos(xm_leg, y_b, z_b) * Rot(0, 90, 0) * Cylinder(radius=r_sh, height=L_leg)
        u_bend = Pos(x_turn, yf, zf) * Box(3.4, 4.8, 4.8)
        riser1 = Pos(x_hdr - 1.6, y_a, 0.5 * (z_a + 53.5)) * Cylinder(radius=r_sh, height=53.5 - z_a + 1.5)
        riser2 = Pos(x_hdr + 1.4, y_b, 0.5 * (z_b + 53.5)) * Cylinder(radius=r_sh, height=53.5 - z_b + 1.5)
        sub = leg1 + leg2 + u_bend + riser1 + riser2
        sh_loops = sub if sh_loops is None else (sh_loops + sub)

    # 3. 17-4 接头 -> 13-4 中央主蒸汽管 -> 注油三通 (122, 0, 18.5) -> 7-9 左右给汽管入蒸汽室 (x=126, y=±27.5, z=18.5)
    tx_lub, _, tz_lub = [float(v) for v in br["lubricator_tee_xyz_mm"]]
    pipe_13_4 = (
        Pos(0.5 * (x_hdr + tx_lub), 0.0, 52.0) * Rot(0, 90, 0) * Cylinder(radius=1.8, height=tx_lub - x_hdr + 2.0)
        + Pos(tx_lub, 0.0, 0.5 * (tz_lub + 52.0)) * Cylinder(radius=1.8, height=52.0 - tz_lub + 2.0)
        + Pos(tx_lub, 0.0, tz_lub) * Box(5.0, 5.4, 4.6)
    )
    x_chest = float(br["x_chest_entry_mm"])
    r_br = 0.5 * float(br["tube_od_mm"])
    pipes_7_9 = (
        Pos(tx_lub, 0.0, tz_lub) * Rot(90, 0, 0) * Cylinder(radius=r_br, height=50.4)
        + Pos(0.5 * (tx_lub + x_chest), +24.5, tz_lub) * Rot(0, 90, 0) * Cylinder(radius=r_br, height=x_chest - tx_lub + 2.0)
        + Pos(0.5 * (tx_lub + x_chest), -24.5, tz_lub) * Rot(0, 90, 0) * Cylinder(radius=r_br, height=x_chest - tx_lub + 2.0)
    )

    # 4. 7-8 排汽汇流三通 + 20-7 乏汽喷嘴 (ø4.2mm) + 11-12/13-8/17-3 通风嘴 + 18-28 喇叭口烟囱裙罩
    x_ch = float(ej["chimney_axis_x_mm"])
    z_tip = float(ej["blast_nozzle_tip_z_mm"])
    r_orifice = 0.5 * float(ej["blast_nozzle_orifice_dia_mm"])

    exh_7_8 = Pos(x_ch, 0.0, 15.2) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=48.0)
    conn_exh = Pos(0.5 * (tx_lub + x_ch), 0.0, 16.8) * Box(x_ch - tx_lub + 2.0, 3.6, 3.6)
    blast_col = Pos(x_ch, 0.0, 0.5 * (14.5 + z_tip)) * Cone(bottom_radius=4.3, top_radius=3.1, height=z_tip - 14.5)
    blower_ring = Pos(x_ch, 0.0, z_tip - 1.2) * Cylinder(radius=4.7, height=1.8)
    blast_bore = Pos(x_ch, 0.0, z_tip - 4.0) * Cylinder(radius=r_orifice, height=10.0)

    # 11-12 通風弁 -> 13-8 炉内通风细管 -> 17-3 弯头通风喷嘴
    blower_11_12 = Pos(x_reg, +6.2, 57.5) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=4.2)
    pipe_13_8 = Pos(0.5 * (x_reg + x_ch - 3.5), +6.2, 57.5) * Rot(0, 90, 0) * Cylinder(
        radius=1.0, height=(x_ch - 3.5) - x_reg
    )
    drop_17_3 = (
        Pos(x_ch - 3.5, +6.2, 0.5 * (z_tip - 1.2 + 57.5)) * Cylinder(radius=1.0, height=57.5 - (z_tip - 1.2) + 1.5)
        + Pos(x_ch - 1.8, +3.1, z_tip - 1.2) * Box(4.2, 7.0, 1.8)
    )

    # 18-28 烟囱文丘里裙罩 (通过烟室侧壁托架与集箱相连构成单连通实体)
    z_pet = float(ej["petticoat_entry_z_mm"])
    r_pet_th = 0.5 * float(ej["petticoat_throat_dia_mm"])
    r_pet_fl = 0.5 * float(ej["petticoat_flare_dia_mm"])
    petticoat_outer = (
        Pos(x_ch, 0, z_pet + 5.5) * Cone(bottom_radius=r_pet_fl, top_radius=r_pet_th + 1.0, height=11.0)
        + Pos(x_ch, 0, z_pet + 14.0) * Cylinder(radius=r_pet_th + 1.0, height=7.0)
    )
    petticoat_inner = (
        Pos(x_ch, 0, z_pet + 5.5) * Cone(bottom_radius=r_pet_fl - 1.1, top_radius=r_pet_th, height=11.6)
        + Pos(x_ch, 0, z_pet + 14.0) * Cylinder(radius=r_pet_th, height=9.0)
    )
    pet_stay = Pos(0.5 * (x_hdr + x_ch - r_pet_fl + 1.0), 0.0, z_pet + 2.0) * Box(
        (x_ch - r_pet_fl) - x_hdr + 4.0, 2.4, 2.2
    )

    return (
        reg_body + pipe_13_9 + bolt_17_6 + sh_loops + pipe_13_4 + pipes_7_9
        + exh_7_8 + conn_exh + (blast_col + blower_ring - blast_bore)
        + blower_11_12 + pipe_13_8 + drop_17_3 + pet_stay + (petticoat_outer - petticoat_inner)
    )


def build_feedwater_piping(cfg: Dict, pip_cfg: Dict):
    """
    真实双水路给水管网总成 (100% 位于主车架内侧 |Y| <= 11.0mm，向后走线，车轮外侧零管线！):
      - 轴动泵回路: 煤水车 31-12 -> 6-10 吸水管 (y=+6.8, z=+9.2) -> 6-5 中央轴动泵阀体
        -> 6-6 向后吐出管 (y=-6.5, z=+10.2) -> 车尾三通 (-114,-6.5,10.2) -> 6-8 上升管至锅炉后上部 13-7 逆止阀，
        分流 4-16 -> 驾驶室右下 4-17 旁通阀 -> 4-15 回水管；
      - 手摇泵回路: 煤水车 31-13 -> 13-11 配水管 (y=+3.8, z=+9.2) -> 火室底部独立 13-2 逆止阀。
    """
    fw = pip_cfg["feedwater_piping_topology"]
    ax_loop = fw["axle_pump_loop"]
    hp_loop = fw["hand_pump_loop"]
    x_pump = float(ax_loop["pump_x_mm"])  # -22.0 mm (柱塞中心基准)

    # 1. 6-5 中央轴动泵固定黄铜阀体 (置于 -42.4..-32.5mm, 前距偏心环叉头后死点 -31.3mm >= 1.2mm, 后距第4轴 -43.75mm >= 1.35mm!)
    x_tw = x_pump - 16.8  # -38.8 mm
    pump_barrel = Pos(x_pump - 15.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=3.6, height=7.2)
    gland_nut = Pos(x_pump - 11.4, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=4.0, height=1.8)
    check_tower = Pos(x_tw, 0.0, 1.5) * Cylinder(radius=3.0, height=16.5)
    top_hex_cap = Pos(x_tw, 0.0, 10.4) * Cylinder(radius=3.6, height=2.2)
    bot_hex_cap = Pos(x_tw, 0.0, -7.4) * Cylinder(radius=3.6, height=2.0)
    # 车架内安装横担严格控制 |Y| <= 10.6 mm (< 11.0 mm!), 预留嵌入重叠避免零厚度相切
    pump_bracket = Pos(x_pump - 15.0, 0.0, 4.2) * Box(5.8, 21.2, 2.6)
    ram_bore = Pos(x_pump - 14.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(radius=1.90, height=8.4)
    pump_6_5 = pump_barrel + gland_nut + check_tower + top_hex_cap + bot_hex_cap + pump_bracket - ram_bore

    # 2. 6-10 吸水管 (车架内 y=+6.8mm, z=+9.2mm，完全高于第4轴 z<=+6.3mm 与从轮轴 z=-8.5mm)
    x_rear_end = -142.0
    y_suc, z_suc = float(ax_loop["suction_y_mm"]), float(ax_loop["suction_z_mm"])
    suc_elbow = (
        Pos(x_tw, 0.5 * y_suc, -5.2) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=y_suc + 1.6)
        + Pos(x_tw, y_suc, -5.2) * Box(3.0, 3.0, 3.0)
        + Pos(x_tw, y_suc, 0.5 * (-5.2 + z_suc)) * Cylinder(radius=1.25, height=z_suc - (-5.2))
        + Pos(x_tw, y_suc, z_suc) * Box(3.0, 3.0, 3.0)
    )
    pipe_6_10 = Pos(0.5 * (x_rear_end + x_tw), y_suc, z_suc) * Rot(0, 90, 0) * Cylinder(
        radius=1.25, height=x_tw - x_rear_end
    )

    # 3. 6-6 向后送水管 (y=-6.5mm, z=+10.2mm) -> 车尾三通 (-114.0) -> 6-8 上升至后上部 13-7 逆止阀
    y_del, z_del = float(ax_loop["delivery_y_mm"]), float(ax_loop["delivery_z_mm"])
    x_tee = float(ax_loop["rear_tee_xyz_mm"][0])
    cx_13_7, cy_13_7, cz_13_7 = [float(v) for v in ax_loop["boiler_check_xyz_mm"]]

    del_out = (
        Pos(x_tw, 0.5 * y_del, z_del) * Rot(90, 0, 0) * Cylinder(radius=1.32, height=abs(y_del) + 1.6)
        + Pos(x_tw, y_del, z_del) * Box(3.0, 3.0, 3.0)
    )
    pipe_6_6 = Pos(0.5 * (x_tee + x_tw), y_del, z_del) * Rot(0, 90, 0) * Cylinder(
        radius=1.25, height=x_tw - x_tee
    )
    rear_tee = Pos(x_tee, 0.5 * (y_del + cy_13_7), z_del) * Box(4.2, abs(y_del - cy_13_7) + 3.6, 3.6)
    pipe_6_8 = Pos(x_tee, cy_13_7, 0.5 * (z_del + cz_13_7)) * Cylinder(
        radius=1.25, height=cz_13_7 - z_del
    )
    clack_13_7 = (
        Pos(x_tee, cy_13_7, cz_13_7) * Box(3.2, 3.2, 3.2)
        + Pos(0.5 * (x_tee + cx_13_7), cy_13_7, cz_13_7) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=abs(cx_13_7 - x_tee) + 2.5)
        + Pos(x_tee, cy_13_7, cz_13_7 + 0.5) * Cylinder(radius=2.9, height=6.4)
    )

    # 4. 4-16 分支 -> 4-17 旁通针阀 (x=-114.5, y=-8.4, z=25.5, 手轮半径 2.1mm -> |Y|<=10.5mm!) -> 4-15 回水管
    bx, _, bz = [float(v) for v in ax_loop["bypass_valve_xyz_mm"]]
    by = -8.4
    pipe_4_16 = (
        Pos(x_tee, 0.5 * (y_del + by), z_del) * Rot(90, 0, 0) * Cylinder(radius=1.15, height=abs(by - y_del) + 1.6)
        + Pos(0.5 * (x_tee + bx), by, z_del) * Box(abs(x_tee - bx) + 2.8, 2.8, 2.8)
        + Pos(bx, by, 0.5 * (z_del + bz)) * Cylinder(radius=1.15, height=bz - z_del)
    )
    valve_4_17 = (
        Pos(bx, by, bz) * Box(4.4, 3.8, 4.4)
        + Pos(bx - 2.6, by, bz) * Rot(0, 90, 0) * Cylinder(radius=2.1, height=2.4)
    )
    pipe_4_15 = Pos(0.5 * (x_rear_end + bx), by, z_del - 1.0) * Rot(0, 90, 0) * Cylinder(
        radius=1.15, height=bx - x_rear_end
    ) + Pos(bx, by, z_del - 0.5) * Box(2.8, 2.8, 3.2)

    # 5. 手摇泵独立补水线: 13-11 配水管 (y=+3.8, z=+9.2) -> 火室底部独立 13-2 逆止阀 (x=-104.0, y=+3.8, z=+17.5)
    y_hp, z_hp = float(hp_loop["delivery_y_mm"]), float(hp_loop["delivery_z_mm"])
    hx, hy, hz = [float(v) for v in hp_loop["firebox_check_xyz_mm"]]
    pipe_13_11 = Pos(0.5 * (x_rear_end + hx), y_hp, z_hp) * Rot(0, 90, 0) * Cylinder(
        radius=1.25, height=hx - x_rear_end
    )
    riser_13_2 = (
        Pos(hx, hy, z_hp) * Box(3.0, 3.0, 3.0)
        + Pos(hx, hy, 0.5 * (z_hp + hz)) * Cylinder(radius=1.35, height=hz - z_hp)
    )
    clack_13_2 = Pos(hx, hy, hz) * Cylinder(radius=2.9, height=5.8)
    # 车尾内侧管夹横梁连接左右水路构成单连通实体 (|Y| <= 9.6mm)
    rear_clip = Pos(-136.0, -0.8, z_hp) * Box(3.2, 17.6, 3.2)

    return (
        pump_6_5 + suc_elbow + pipe_6_10 + del_out + pipe_6_6 + rear_tee
        + pipe_6_8 + clack_13_7 + pipe_4_16 + valve_4_17 + pipe_4_15
        + pipe_13_11 + riser_13_2 + clack_13_2 + rear_clip
    )


def build_displacement_lubricator(cfg: Dict, pip_cfg: Dict):
    """
    4-1 / 4-2 前车架中央卧式置换式注油器 (BS 黄铜, イラスト 4 & 25):
      - 卧式圆柱油箱 4-2 置于前主车架中央 (x ∈ [142.0, 166.0], y=0, z=+6.2mm, |Y| <= 8.2mm <= 10.5mm)，
        位于先导台车支点上方、烟室鞍座下方，底部光滑无排油塞 (has_bottom_drain=False)；
      - 顶盖 4-1 (PN5-1.9) 位于 (164.0, 0, 14.0) 正对前甲板活门 25-3；
      - 后部蒸汽油细管接入中央 13-4/7-9 主蒸汽三通 (122.0, 0, 18.5)；100% 左右中心对称！
    """
    lub = pip_cfg["displacement_lubricator_system"]
    x0, x1 = [float(v) for v in lub["x_range_mm"]]
    xm = 0.5 * (x0 + x1)
    zc = float(lub["z_center_mm"])
    r_tank = 0.5 * float(lub["tank_od_mm"])  # 6.8 mm
    cap_x, _, cap_z = [float(v) for v in lub["cap_xyz_mm"]]
    tx, _, tz = [float(v) for v in lub["oil_delivery_tee_xyz_mm"]]

    tank_4_2 = Pos(xm, 0.0, zc) * Rot(0, 90, 0) * Cylinder(radius=r_tank, height=x1 - x0)
    saddle_cradle = Pos(xm, 0.0, zc + 1.5) * Box(12.0, 16.4, 4.2)
    neck_4_1 = Pos(cap_x - 1.5, 0.0, 0.5 * (zc + cap_z)) * Cylinder(radius=2.8, height=cap_z - zc + 1.0)
    cap_4_1 = Pos(cap_x - 1.5, 0.0, cap_z) * Cylinder(radius=3.8, height=2.2)

    oil_pipe_h = Pos(0.5 * (tx + x0), 0.0, zc + 3.2) * Rot(0, 90, 0) * Cylinder(radius=1.1, height=x0 - tx + 2.0)
    oil_pipe_v = Pos(tx, 0.0, 0.5 * (zc + 3.2 + tz)) * Cylinder(radius=1.1, height=tz - (zc + 3.2) + 2.0)
    union_nut = Pos(tx, 0.0, tz - 2.5) * Cylinder(radius=2.1, height=2.4)

    return tank_4_2 + saddle_cradle + neck_4_1 + cap_4_1 + oil_pipe_h + oil_pipe_v + union_nut


def build_boiler_aux_valves(cfg: Dict, pip_cfg: Dict):
    """
    锅炉安全阀、驾驶室仪表、车底真汽笛与后壁排污阀总成 (BS 黄铜, 全顶点 |Y| <= 16.5mm < 24.0mm):
      - 28-6 x2 双安全阀；
      - 11-4~7 水面计；
      - 13-1 环形虹吸盘管 + PG-ø20 压力表；
      - 11-1~3 汽笛阀 -> 13-3 汽笛管 -> 11-8 车底真汽笛；
      - 11-16/17 后壁下部排污阀。
    """
    bom = pip_cfg["pumps_and_valves_bom"]
    aux = pip_cfg["superheater_and_steam_piping"]["auxiliary_steam_lines"]
    sv = bom["twin_safety_valves_28_6"]
    wg = bom["water_gauge_11_7"]
    pg = bom["pressure_gauge_13_1"]
    wh = aux["functional_whistle"]
    bd = aux["backhead_blowdown"]

    x_back = float(wg["x_cab_backhead_mm"])  # -114.5 mm

    # 1. 28-6 x2 双安全阀 + 内部连通汇流管连接至后壁仪表盘
    sv_x = float(sv["x_mm"])
    sv_1 = Pos(sv_x, +3.8, 76.5) * Cylinder(radius=2.3, height=9.0)
    sv_2 = Pos(sv_x, -3.8, 76.5) * Cylinder(radius=2.3, height=9.0)
    sv_base = Pos(sv_x, 0.0, 72.5) * Box(5.0, 12.0, 3.0)
    top_tie = Pos(0.5 * (x_back + sv_x), 0.0, 62.0) * Rot(0, 90, 0) * Cylinder(radius=1.1, height=sv_x - x_back + 2.0)
    sv_riser = Pos(sv_x, 0.0, 67.2) * Cylinder(radius=1.2, height=11.5)

    # 2. 11-4~7 水面计 (y = -9.5mm)
    yw = float(wg["y_mm"])
    wg_top = Pos(x_back, yw, 57.0) * Box(4.6, 3.8, 3.4)
    wg_bot = Pos(x_back, yw, 39.0) * Box(4.6, 3.8, 3.4)
    wg_glass = Pos(x_back - 1.0, yw, 48.0) * Cylinder(radius=1.7, height=19.0)

    # 3. 13-1 虹吸弯管 + PG-ø20 压力表 (y = +9.5mm, r=6.2 -> max |Y| = 15.7mm < 24.0mm)
    yp, zp = float(pg["y_mm"]), float(pg["z_mm"])
    pg_case = Pos(x_back - 1.5, yp, zp) * Rot(0, 90, 0) * Cylinder(radius=6.2, height=3.2)
    siphon_loop = (
        Pos(x_back - 1.2, yp, zp - 8.2) * Rot(0, 90, 0) * Cylinder(radius=2.8, height=1.8)
        - Pos(x_back - 1.2, yp, zp - 8.2) * Rot(0, 90, 0) * Cylinder(radius=1.2, height=4.0)
    )
    pg_stem = Pos(x_back - 1.2, yp, zp - 5.2) * Cylinder(radius=1.1, height=11.0)

    # 4. 11-1~3 汽笛阀 -> 13-3 汽笛管 -> 11-8 车底真汽笛 (-126.0, -6.5, 11.5)
    wx, wy, wz = [float(v) for v in wh["whistle_xyz_mm"]]
    wh_valve = Pos(x_back, wy, 61.0) * Box(4.2, 3.6, 3.6)
    pipe_13_3 = Pos(x_back - 0.8, wy, 0.5 * (wz + 61.0)) * Cylinder(radius=1.05, height=61.0 - wz + 2.0)
    whistle_11_8 = Pos(0.5 * (wx - 8.0 + x_back), wy, wz) * Rot(0, 90, 0) * Cylinder(
        radius=2.4, height=(x_back - (wx - 8.0)) + 2.0
    )

    # 5. 11-16/17 后壁下部排污阀 (-114.5, +6.5, 24.0)
    bdx, bdy, bdz = [float(v) for v in bd["xyz_mm"]]
    blowdown = (
        Pos(bdx, bdy, bdz) * Box(4.6, 3.8, 4.2)
        + Pos(bdx - 2.4, bdy, bdz) * Rot(0, 90, 0) * Cylinder(radius=2.2, height=2.2)
        + Pos(bdx, bdy, 0.5 * (14.5 + bdz)) * Cylinder(radius=1.1, height=bdz - 14.5 + 1.5)
        + Pos(bdx, bdy, 36.5) * Cylinder(radius=1.0, height=26.0)
    )

    back_manifold = (
        Pos(x_back + 0.5, 0.0, 60.5) * Box(2.4, 22.0, 3.2)
        + Pos(x_back + 0.5, yw, 58.5) * Box(2.4, 2.8, 5.0)
        + Pos(x_back + 0.5, bdy, 50.0) * Box(2.4, 2.6, 22.0)
    )

    return (
        sv_1 + sv_2 + sv_base + top_tie + sv_riser
        + wg_top + wg_bot + wg_glass + pg_case + siphon_loop + pg_stem
        + wh_valve + pipe_13_3 + whistle_11_8 + blowdown + back_manifold
    )


def build_alcohol_burner(cfg: Dict, pip_cfg: Dict):
    """
    23-1 火室底 2x2 四头立式 10mm 燃烧器 + 23-2 バーナーホルダ托架 (イラスト 23, 100% 中心对称):
      - 2x2 矩形阵列: X = [-89.0, -65.0] mm, Y = [-5.8, +5.8] mm (总宽 22.4mm 嵌于车架侧板内侧)；
      - 中央恒液位供醇管沿 Y=0 向后接煤水车 34-2 サンプ。
    """
    ab = pip_cfg["alcohol_burner_system"]
    wick_xs = [float(x) for x in ab["wick_x_mm"]]
    wick_ys = [float(y) for y in ab["wick_y_mm"]]
    r_cup_out = 0.5 * float(ab["wick_cup_od_mm"])   # 5.4 mm
    r_wick = 0.5 * float(ab["wick_dia_mm"])         # 5.0 mm
    z_top = float(ab["wick_top_z_mm"])              # 18.8 mm
    r_tube = 0.5 * float(ab["supply_tube_od_mm"])

    xm = 0.5 * (wick_xs[0] + wick_xs[-1])
    # 23-2 底部供醇歧管座与安装耳板 (|Y| <= 11.2mm, 左右完全对称)
    holder_23_2 = Pos(xm, 0.0, 12.0) * Box((wick_xs[-1] - wick_xs[0]) + 13.0, 22.0, 2.8)
    rear_lug = Pos(wick_xs[0] - 9.0, 0.0, 13.2) * Box(6.0, 16.0, 4.2)

    cups = None
    for wx in wick_xs:
        for wy in wick_ys:
            cup_shell = Pos(wx, wy, 0.5 * (11.5 + z_top)) * Cylinder(radius=r_cup_out, height=z_top - 11.5)
            cup_well = Pos(wx, wy, z_top - 2.0) * Cylinder(radius=r_wick, height=4.6)
            wick_core = Pos(wx, wy, z_top - 1.2) * Cylinder(radius=r_wick - 0.8, height=4.2)
            one = (cup_shell - cup_well) + wick_core
            cups = one if cups is None else (cups + one)

    feed_pipe = Pos(0.5 * (-142.0 + wick_xs[0] - 8.0), 0.0, 11.6) * Rot(0, 90, 0) * Cylinder(
        radius=r_tube, height=(wick_xs[0] - 8.0) - (-142.0) + 2.0
    )
    return holder_23_2 + rear_lug + cups + feed_pipe


def build_tender_body(cfg: Dict):
    """8-20B 四轴煤水车外衣壳体 (100% 中心对称，X-Ray 模式下变为半透明以展示内部手摇泵与酒精箱)。"""
    t_cfg = cfg["tender_and_track"]
    L, W, H = float(t_cfg["tender_body_length"]), float(t_cfg["tender_body_width"]), float(t_cfg["tender_body_height"])

    base_plate = Pos(0, 0, 1.2) * Box(L + 4.0, W + 2.0, 2.4)
    front_beam = Pos(0.5 * L + 1.2, 0, 0.5) * Box(3.6, W + 2.0, 6.5)
    rear_beam = Pos(-0.5 * L - 1.2, 0, 0.5) * Box(3.6, W + 2.0, 6.5)

    tank_outer = Pos(-4.0, 0, 2.4 + 0.5 * H) * Box(L - 8.0, W, H)
    tank_inner_hollow = Pos(-4.0, 0, 2.4 + 0.5 * H) * Box(L - 12.0, W - 3.6, H - 3.6)
    coal_bunker_cut = Pos(32.0, 0, 2.4 + H - 7.0) * Box(118.0, W - 14.0, 16.0)
    tank_shell = tank_outer - tank_inner_hollow - coal_bunker_cut

    bunker_walls = None
    for sgn in (-1.0, 1.0):
        wall = Pos(26.0, sgn * (0.5 * W - 6.5), 2.4 + H + 3.2) * Box(126.0, 1.6, 8.2)
        bunker_walls = wall if bunker_walls is None else (bunker_walls + wall)

    water_hatch = Pos(-78.0, 0, 2.4 + H + 1.2) * Cylinder(radius=9.5, height=3.2)
    return base_plate + front_beam + rear_beam + tank_shell + bunker_walls + water_hatch


def build_tender_internals(cfg: Dict):
    """
    P0-D 煤水车内部功能总成 (BS 黄铜, イラスト 31, 34, 36, 100% 左右中心对称):
      - 31-5 手摇给水泵 (ø7.0mm, PN7-1.9, 位于后部水箱盖正下方 x=-78mm)；
      - 36-4 130ml 燃料酒精箱 (x ∈ [+18, +86]mm) + 顶部注醇口 + 36-6 燃料调节针阀；
      - 34-2 车底前缘恒液位槽 (サンプ, x=+98mm, 避开第1转向架 x<=+89mm)；
      - 31-12 / 31-13 / 31-14 前部给水/回水/供醇立管接头组。
    """
    # 1. 底部贯通导板与前部 34-2 恒液位サンプ (x=+98.0mm, 完全避开转向架 1)
    floor_spine = Pos(12.0, 0.0, 4.0) * Box(196.0, 16.0, 2.4)
    sump_34_2 = (
        Pos(98.0, 0.0, -2.2) * Box(14.0, 16.0, 6.8)
        + Pos(98.0, 0.0, 2.2) * Cylinder(radius=3.5, height=5.0)
    )

    # 2. 36-4 130ml 酒精燃料箱 (68 x 50 x 38.5 mm ≈ 130.9 ml, 壁厚抽壳保证真实黄铜空箱质量) + 36-6 燃料针阀
    alc_outer = Pos(52.0, 0.0, 23.5) * Box(68.0, 50.0, 38.0)
    alc_hollow = Pos(52.0, 0.0, 23.5) * Box(65.2, 47.2, 35.2)
    alc_tank_36_4 = alc_outer - alc_hollow
    filler_cap = Pos(42.0, 0.0, 44.0) * Cylinder(radius=4.8, height=4.2)
    needle_36_6 = (
        Pos(74.0, 0.0, 45.5) * Cylinder(radius=2.4, height=7.5)
        + Pos(74.0, 0.0, 49.5) * Box(2.2, 9.0, 1.8)
    )
    sump_drop = Pos(89.0, 0.0, 12.0) * Box(20.0, 4.0, 18.0)

    # 3. 31-5 手摇给水泵 (ø7.0mm 泵筒 + 底座逆止球室 + 摇柄插口架, x=-78.0, y=0)
    hand_pump_31_5 = (
        Pos(-78.0, 0.0, 18.0) * Cylinder(radius=5.8, height=28.0)
        + Pos(-78.0, 0.0, 6.5) * Box(14.0, 18.0, 4.5)
        + Pos(-78.0, 0.0, 35.0) * Box(4.5, 9.0, 10.0)
        + Pos(-72.0, 0.0, 41.0) * Rot(0, -18, 0) * Box(3.0, 2.6, 12.0)
    )

    # 4. 31-12 / 31-13 / 31-14 前缘对称四联立管快速接头 (y = [-8.8, -3.8, +3.8, +8.8] mm)
    standpipes = Pos(106.0, 0.0, 5.5) * Box(5.0, 22.0, 3.2)
    for sy in (-8.8, -3.8, +3.8, +8.8):
        standpipes += Pos(106.0, sy, 11.5) * Cylinder(radius=1.5, height=13.0)
        standpipes += Pos(109.5, sy, 5.5) * Rot(0, 90, 0) * Cylinder(radius=1.6, height=5.0)

    return floor_spine + sump_34_2 + alc_tank_36_4 + filler_cap + needle_36_6 + sump_drop + hand_pump_31_5 + standpipes


def build_tender_bogie_frame(cfg: Dict):
    wb = float(cfg["tender_and_track"]["bogie_wheelbase"])
    half_wb = 0.5 * wb
    y_frame = 0.5 * float(cfg["meta"]["track_gauge"]) + 5.2

    bolster = Pos(0, 0, 2.0) * Box(14.0, 2.0 * y_frame + 1.0, 5.5)
    sides = None
    for sgn in (-1.0, 1.0):
        yf = sgn * y_frame
        side_part = Pos(0, yf, 1.5) * Box(wb + 18.0, 2.4, 5.5) + Pos(0, sgn * (y_frame + 1.2), -1.5) * Box(22.0, 2.2, 4.2)
        for ax_x in (-half_wb, +half_wb):
            side_part += Pos(ax_x, sgn * (y_frame + 1.4), 0.0) * Box(7.5, 4.2, 8.2)
            side_part += Pos(ax_x - 4.8, sgn * (y_frame + 1.0), 2.5) * Cylinder(radius=1.6, height=5.5)
            side_part += Pos(ax_x + 4.8, sgn * (y_frame + 1.0), 2.5) * Cylinder(radius=1.6, height=5.5)
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
    disc = Cylinder(radius=r_tread - 1.7, height=1.8)
    for ang_deg in range(0, 360, 45):
        ang = math.radians(ang_deg)
        disc -= Pos(7.2 * math.cos(ang), 7.2 * math.sin(ang), 0) * Cylinder(radius=1.45, height=4.0)

    wheel = rim + disc + Cylinder(radius=3.5, height=w_tire + 0.4) - Cylinder(radius=1.8, height=w_tire + 6.0)
    wheel_mj = Rot(-90, 0, 0) * wheel
    return wheel_mj if is_left else mirror(wheel_mj, about=Plane.XZ)


def subdivide_stl_x_span(stl_path: Path, max_dx: float = 1.85, z_min_filter: float = 52.0) -> None:
    """
    对二进制 STL 的上层锅筒/烟室壳体长轴向边同步二分加密 (Synchronous Manifold Edge Bisection)：
      - 判定准则仅依赖无向边端点 {u, v}，确保共享该边的相邻三角形同步插入完全一致的中点 m_uv；
      - 100% 保持水密流形 (boundary_edges == 0)、体积不变、Y 对称不变，同时保证每 2mm X 分箱均含真实网格顶点。
    """
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
    parser = argparse.ArgumentParser(description="Aster D51 P0-C + P0-D 水密 CAD 建模与材料惯量积分器")
    parser.add_argument("--config", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--piping-config", type=Path, default=Path("configs/d51_boiler_piping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/meshes"))
    parser.add_argument("--mass-out", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--standard-stack", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pip_cfg = json.loads(args.piping_config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 清理废弃的旧单件无后缀网格 (如旧 eccentric_rod.stl / expansion_link.stl 等)，确保目录干净无残留
    q_deg = float(cfg["wheels_and_axles"]["quartering_deg_R_minus_L"])

    # (part_key, stl_tolerance, density_kg_m3, builder_fn)
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
        ("cab_reach_rod", 0.025, RHO_STAINLESS, lambda: build_cab_reach_rod(cfg)),
        ("drain_cock_linkage", 0.025, RHO_BRASS, lambda: build_drain_cock_linkage(cfg)),
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
        ("superheater_piping", 0.035, RHO_STAINLESS, lambda: build_superheater_piping(cfg, pip_cfg)),
        ("feedwater_piping", 0.03, RHO_BRASS, lambda: build_feedwater_piping(cfg, pip_cfg)),
        ("displacement_lubricator", 0.03, RHO_BRASS, lambda: build_displacement_lubricator(cfg, pip_cfg)),
        ("boiler_aux_valves", 0.03, RHO_BRASS, lambda: build_boiler_aux_valves(cfg, pip_cfg)),
        ("alcohol_burner", 0.03, RHO_BURNER_ASSY, lambda: build_alcohol_burner(cfg, pip_cfg)),
        ("tender_body", 0.08, RHO_BRASS, lambda: build_tender_body(cfg)),
        ("tender_internals", 0.05, RHO_BRASS, lambda: build_tender_internals(cfg)),
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
            "stage": "P0-C + P0-D (V2-MECH-TRUE + V2-PIPING-LUBE-TRUE)",
            "units": "SI (kg, m, kg*m^2)",
            "densities_kg_m3": {
                "copper_C1100": RHO_COPPER,
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
