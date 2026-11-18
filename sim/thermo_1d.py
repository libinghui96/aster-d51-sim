#!/usr/bin/env python3
"""
sim/thermo_1d.py
Aster Hobby JNR D51 (1/32, Gauge 1) 一维集总参数热力学—流体—配气—极限损伤耦合求解器。
涵盖特性:
  - 酒精恒液位サンプ供油、烟室文丘里引风 (优先插值 RTX 5080 smokebox_lut.json 3D CFD 查找表) 与过热器
  - 分层紫铜火室顶板壁温 T_crown_c、空焚き (干烧退火软化/银焊微渗漏) 与高温冷水闪蒸压力尖峰
  - 纵坡俯仰角与加减速引起的表观假水位 (Apparent Water Level) 与顶板裸露判定
  - 外进气平面 D 型滑阀 (1-6) 配气孔口可压缩阻塞流 + 蒸汽室背压压紧滑动摩擦反力
  - 冷气缸分腔液态冷凝水积聚与未开排液阀时的不可压缩水锤 (Hydrolock) 冲击锁死及密封退化
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim.iapws_thermo import (
    sat_liquid_props,
    sat_pressure_pa as iapws_sat_pressure_pa,
    sat_temp_c as iapws_sat_temp_c,
    sat_vapor_props,
    steam_ph_state,
)

PA_PER_KGF_CM2 = 98066.5
P_ATM = 101325.0
R_STEAM = 461.5       # 水蒸气气体常数 J/(kg·K)
GAMMA_STEAM = 1.30    # 过热/干饱和蒸汽绝热指数
CP_WATER = 4186.0     # 液态水比热容 J/(kg·K)
CP_COPPER = 385.0     # 紫铜比热容 J/(kg·K)
RHO_LIQ_WATER = 980.0 # 热态液态冷凝水密度 kg/m^3


@dataclass
class CabControls:
    """驾驶室与煤水车整备操作控制量"""
    fuel_valve: float = 0.0       # 36-6 燃料弁开度 [0, 1]
    burner_lit: bool = False      # 23-1 酒精燃烧器点火状态
    chimney_fan: bool = False     # 烟囱外置电动抽风机 (煙突通風機) ON/OFF
    blower_valve: float = 0.0     # 11-13 通風弁开度 [0, 1] (30° 约对应 0.35)
    regulator: float = 0.0        # 11-11 加減弁开度 [0, 1] (起步 30° 约对应 0.35)
    reverser: float = 0.0         # 8-16 螺旋逆転機归一化位置 [-1.0 后退满档, 0.0 中立 N, +1.0 前进满档]
    drain_cock: bool = True       # 8-6 -> 8-5 -> 1-16 ドレン弁开关 (True=打开排冷凝水, False=关闭正常行驶)
    bypass_valve: float = 0.40    # 6-9 / 4-17 バイパス弁开度 [0.0 全关强制给水, 1.0 全开回流煤水车]
    whistle_valve: float = 0.0    # 11-3 -> 11-8 车底真汽笛压柄开度 [0, 1]
    blowdown_valve: float = 0.0   # 11-16/17 驾驶室后封头排污阀开度 [0, 1]
    firebox_door: float = 0.0     # 12-2 铰接火室门开启度 [0.0 闭合, 1.0 敞开进二次冷风]
    fuel_cap_open: bool = False   # 36-5 煤水车马氏瓶酒精箱气密盖是否拧开 (供油中拧开将破坏真空导致漫油!)
    water_lid_open: bool = False  # 35-6 煤水车后部开放水箱注水铰接盖开启状态
    water_fill_valve: float = 0.0 # 35-1 后部敞口注水槽外部加水流量开度 [0, 1] (1.0 ≈ 18 ml/s 常温补水)
    tender_drain_open: bool = False  # 31-11 -> 31-9 煤水车水箱底部提拉排水栓开关
    sight_drain_valve: bool = False  # 1-2 右侧透明玻璃水面计下座冲洗吹扫针阀开关
    cab_roof_open: bool = False   # 兼容保留字段 (21-15 实为螺钉固定通风器)
    dome_lid_open: bool = False   # 29-12 カマボコ砂箱前部钢丝铰接顶盖开启状态
    coal_cover_open: bool = False # 36-1/36-2 煤水车前部炭库盖翻开整备酒精箱状态
    safety_test_pin: bool = False # 28-6 安全阀顶端「上部の小さな突起」镊子提拉排汽测试开关 (Manual p.30)
    thermal_warp: float = 1.0     # 热力学时间倍率 (1.0=真实物理时间, >1.0=加速烧水升温)
    grade_deg: float = 0.0        # 附加模拟线路坡度 (度, >0 上坡, <0 下坡, 用于滚轮台假水位测试)


@dataclass
class CylinderCalibration:
    """启动时从 MuJoCo 模型自动测定的活塞止点与滑阀气口中心"""
    q_fdc_L: float = 0.0          # 左缸前死点 (Head-End 最小容积处) 关节坐标 (m)
    q_fdc_R: float = 0.010545     # 右缸前死点关节坐标 (m)
    stroke: float = 0.020         # 活塞行程 (m)
    q_valve_center_L: float = 0.0 # 左滑阀气口对称中心关节坐标 (m)
    q_valve_center_R: float = 0.0 # 右滑阀气口对称中心关节坐标 (m)
    steam_lap: float = 0.00136    # 外余面 Steam Lap (m)
    exhaust_lap: float = 0.00005  # 内余面 Exhaust Lap (m)
    port_width: float = 0.00080   # 蒸汽口 SP 最大有效开度宽 (m)


@dataclass
class ThermoTelemetry:
    """用于终端仪表盘与曲线记录的实时遥测快照"""
    sim_time: float = 0.0
    boiler_temp_c: float = 25.0
    boiler_pressure_kgf: float = 0.0
    steam_chest_kgf: float = 0.0
    superheat_temp_c: float = 25.0
    water_level_pct: float = 50.0          # 水面计表观水位 (含坡道/加减速液面倾斜效应)
    true_water_level_pct: float = 50.0     # 锅筒真实静水位百分比
    crown_temp_c: float = 25.0             # 紫铜火室顶板实时壁温 (°C)
    crown_submerged_mm: float = 11.5       # 火室顶板真实浸没水深 (mm, <=0 表示裸露干烧)
    copper_anneal_damage: float = 0.0      # 紫铜干烧退火/银焊热损伤度 [0, 1]
    seal_health: float = 1.0               # PR-15/PTFE 密封完好度 [0.25, 1.0]
    alcohol_ml: float = 130.0
    tender_water_ml: float = 320.0
    lubricator_oil_pct: float = 80.0
    cyl_wall_temp_c: float = 25.0
    condensate_g: float = 0.0
    safety_valve_open: bool = False
    dry_fire_alarm: bool = False
    emergency_overpressure: bool = False
    flash_steam_spike: bool = False        # 干烧高温补水闪蒸压力尖峰标志
    hydrolock_L: bool = False              # 左气缸不可压缩水锤冲击锁死标志
    hydrolock_R: bool = False              # 右气缸不可压缩水锤冲击锁死标志
    valve_friction_L: float = 0.0          # 左平面滑阀背压摩擦反力 (N)
    valve_friction_R: float = 0.0          # 右平面滑阀背压摩擦反力 (N)
    smokebox_vacuum_pa: float = 0.0        # 烟室抽吸负压 (Pa)
    lut_active: bool = False               # 是否激活 RTX 5080 CFD 查找表 / 3D PINO 算子
    pino_active: bool = False              # 是否由 L2 3D PINO 物理神经算子实时驱动热流边界
    steam_dryness_dome: float = 0.96       # 汽包出口初始湿蒸汽干度 x_dome ∈ [0, 1]
    steam_dryness_chest: float = 1.00      # 经 17-5 四程过热器干化后的蒸汽室干度 x_chest ∈ [0, 1]
    heat_input_w: float = 0.0
    feedwater_g_s: float = 0.0
    return_water_ml_s: float = 0.0         # 经 4-16 -> 31-14 鹅颈立管回流煤水车水箱流量 (ml/s)
    tender_water_temp_c: float = 25.0      # 煤水车水箱水温 (°C, 随旁通回水微幅预热)
    boiling_progress_pct: float = 0.0      # 冷炉点火升温沸腾进度 (25°C -> 100°C 沸腾为 0~100%)
    mariotte_bubbling: bool = False        # 36-4 -> 34-2 马氏瓶恒液位补气冒泡标志
    mariotte_flood_alarm: bool = False     # 36-5 旋盖在 36-6 开启时打开引发的酒精漫溢告警
    mariotte_headspace_kpa: float = 0.0    # 36-4 马氏瓶密闭顶空负压真空度 (kPa, 36-5 拧开时归零)
    sight_glass_level_pct: float = 50.0    # 11-7 左侧透明玻璃水面计真实弯月面高度 (%, 1-2 冲洗时吹落)
    pg_bourdon_kgf: float = 0.0            # PG-ø20 经右侧 13-1 下部 U 弯冷凝水虹吸缓冲后的压力表读数 (kgf/cm²)
    turret_pressure_kgf: float = 0.0       # 11-15 驾驶室后壁干汽分汽包总管表压 (kgf/cm²)
    water_fill_ml_s: float = 0.0           # 35-1 煤水车敞口水槽外部加水流量 (ml/s)
    tender_drain_ml_s: float = 0.0         # 31-11 -> 31-9 煤水车底部排水栓放水流量 (ml/s)
    tender_overflow_ml_s: float = 0.0      # 31-10 煤水车水箱满容 (340 ml) 溢流流量 (ml/s)
    blowdown_g_s: float = 0.0              # 11-16/17 左床下排水针阀排液率 (g/s)
    sight_purge_g_s: float = 0.0           # 1-2 右侧水面计下座冲洗蒸汽吹扫率 (g/s)
    whistle_steam_g_s: float = 0.0         # 11-8 左火室颊下真汽笛耗汽率 (g/s)
    exhaust_g_s: float = 0.0               # 20-7 烟室乏汽喷嘴瞬时排汽质量流率 (g/s)
    blower_g_s: float = 0.0                # 17-3 烟室通风喷嘴瞬时蒸汽流率 (g/s)
    safety_g_s: float = 0.0                # 28-6x2 双安全阀超压泄放蒸汽流率 (g/s)
    safety_valve_lift_mm: float = 0.0      # 28-6x2 Gauge-1 蓄压环腔突跳安全阀实时阀瓣升程 (mm, 0..0.58mm)
    safety_huddling_kgf: float = 0.0       # 28-6 蓄压反冲环腔实时表压 (kgf/cm²)
    siphon_seal_water_pct: float = 96.0    # 13-1 压力表虹吸弯管底部液封水塞充盈度 (%)
    roscoe_quill_oil_rate_mg_min: float = 0.0  # 4-2 置换注油器中心立管蒸汽油顶出速率 (mg/min)
    pino_5op_active: bool = False          # 五算子 PINO 套件在线标志
    pino_6op_active: bool = False          # 六算子 PINO (含陶瓷灯芯毛细+多层保温共轭传热算子) 在线标志
    burner_lumen_connected: bool = True    # 23-1 供醇管->3x立孔->3x灯芯杯三维流道漫水连通标志
    wick_capillary_ml_s: float = 0.0       # セラミックヤーン 三芯毛细吸醇汽化总流量 (ml/s)
    wick_saturation: float = 0.90          # 陶瓷纤维灯芯含醇饱和度 S_l ∈ [0, 1]
    wick_tip_temp_c: float = 25.0          # 陶瓷纤维灯芯顶端温度 (°C, 湿态~82°C, 断醇赤热>520°C)
    casing_temp_c: float = 25.0            # 锅炉烤漆黄铜外皮温度 (°C)
    shell_heat_loss_w: float = 0.0         # 经紫铜+Templates 1-6 陶瓷保温层的总散热损失 (对流+辐射, W)
    shell_rad_loss_w: float = 0.0          # 外表面 Stefan-Boltzmann 热辐射散失功率 (W)
    boiler_efficiency_pct: float = 0.0     # 锅炉第一性原理实时热效率 (%)
    boiler_water_ml: float = 210.0         # 锅筒实时残余液态水体积 (ml, 干烧可直至 0.0 ml)
    water_free_z_mm: float = 55.5          # 锅筒自由液面真实物理高度 Z (mm, 18.2..69.5 mm)
    steam_gen_g_s: float = 0.0             # 锅水真实相变蒸发产汽率 (g/s, 0ml 干锅时严格归零!)
    steam_out_g_s: float = 0.0             # 锅筒顶空实际排汽总质量流率 (g/s)
    cutoff_pct: float = 0.0                # 华氏配气机构实时进汽截止率 (% 行程, 随螺旋逆转机变化)
    cyl_expansion_work_w: float = 0.0      # 双缸蒸汽多变膨胀指示机械功率 (W)
    specific_steam_g_kj: float = 0.0       # 比耗汽率 (g/kJ 指示功, 验证收逆转提膨胀效率)
    pump_chamber_kgf: float = 0.0          # 6-5 中央轴动给水泵腔实时表压 (kgf/cm², 吸程为负真空!)
    pump_suction_vac_kpa: float = 0.0      # 6-5 轴动泵吸程抽水真空度 (kPa)
    pump_shaft_power_w: float = 0.0        # 6-5 轴动泵吸收第3动轮轴机械功率 (W)
    lube_oil_temp_c: float = 25.0          # 4-2 置换注油箱内蒸汽汽缸油温度 (°C)
    lube_oil_viscosity_mpas: float = 1850.0  # 蒸汽汽缸油 Walther 动态黏度 (mPa·s)
    alcohol_temp_c: float = 25.0           # 供醇管路燃料乙醇温度 (°C)
    alcohol_viscosity_mpas: float = 1.08   # 燃料乙醇动态黏度 (mPa·s)
    drain_blowdown_g_s: float = 0.0        # 1-16x4 气缸排液阀气液两相闪蒸吹除总流率 (g/s)
    flue_deflection_mm: List[float] = field(default_factory=lambda: [0.0] * 5)
    flue_ovalization_mm: List[float] = field(default_factory=lambda: [0.0] * 5)
    crown_bulge_mm: float = 0.0            # 紫铜火室顶板热-压耦合鼓曲形变 (mm)
    max_boiler_stress_mpa: float = 0.0     # 受压紫铜烟管/顶板最大 Von Mises 等效应力 (MPa)
    copper_yield_mpa: float = 185.0        # 当前壁温下 C1100 紫铜屈服强度 (MPa, >340°C 急剧退火软化)
    spring_deflect_mm: List[float] = field(default_factory=lambda: [0.0] * 10)
    derailed: bool = False                 # 轮轨脱线落枕告警标志
    derail_reason: str = ""                # 脱轨车轴与物理原因描述
    pino_7op_active: bool = False          # 七算子 PINO (含热弹性形变算子) 在线标志
    sight_glass_lumen_connected: bool = True   # 13-6 蒸汽管 & 13-10 下水管 -> 11-7 Pyrex 水面计三维全通标志
    sight_glass_meniscus_z_mm: float = 55.5    # 11-7 Pyrex 水面计连通器真实物理液柱高度 Z (mm)
    sight_glass_vel_mm_s: float = 0.0          # 11-7 水面计连通器液柱实时升降速度 (mm/s)
    gpu_monitor: Dict[str, Any] = field(default_factory=dict)
    force_L: float = 0.0
    force_R: float = 0.0
    wheel_rpm: float = 0.0
    scale_speed_kmh: float = 0.0


def water_volume_to_free_z_mm(water_ml: float) -> float:
    """
    ø51.6mm 紫铜锅筒 (Z_axis=+44.0mm, 内底 +18.2mm, 内顶 +69.8mm, 扣减火室与 5 烟管排体积) 精确容积->自由液面高度反解:
      - 350 ml -> +68.2 mm (满水上限)
      - 280 ml -> +62.5 mm (11-7 水面计 100%)
      - 210 ml -> +55.5 mm (11-7 水面计 50% 标准水位)
      - 140 ml -> +47.0 mm (11-7 水面计 0%, 低于 +47.5mm 紫铜火室顶板 -> 空焚き开始!)
      -  70 ml -> +36.2 mm (上排双过热烟管完全裸露，下排 3 火管半露)
      -  20 ml -> +25.0 mm (5 根烟管全部裸露，仅剩锅底残水与火室水腿)
      -   0 ml -> +18.2 mm (锅筒彻底蒸干!)
    """
    v = max(0.0, min(360.0, float(water_ml)))
    knots_v = (0.0, 20.0, 70.0, 140.0, 210.0, 280.0, 360.0)
    knots_z = (18.2, 25.0, 36.2, 47.0, 55.5, 62.5, 68.8)
    for i in range(len(knots_v) - 1):
        v0, v1 = knots_v[i], knots_v[i + 1]
        if v <= v1:
            t = (v - v0) / max(1e-9, v1 - v0)
            return knots_z[i] + t * (knots_z[i + 1] - knots_z[i])
    return knots_z[-1]


def free_z_mm_to_sight_pct(z_mm: float) -> float:
    """11-7 Pyrex 透明硼硅玻璃水面计物理弯月面高度 Z (mm) -> 可视水位百分比 (47.0mm=0%, 55.5mm=50%, 62.5mm=100%)。"""
    z = float(z_mm)
    if z <= 47.0:
        return 0.0
    if z <= 55.5:
        return 50.0 * (z - 47.0) / 8.5
    if z <= 62.5:
        return 50.0 + 50.0 * (z - 55.5) / 7.0
    return 100.0


def sat_pressure_pa(temp_c: float) -> float:
    """IAPWS-95 饱和水蒸气压高精度解析式"""
    return iapws_sat_pressure_pa(temp_c)


def sat_temp_c(p_pa: float) -> float:
    """IAPWS-95 牛顿反解精确饱和温度 (°C, 误差 < 1e-6 °C)"""
    return iapws_sat_temp_c(p_pa)


def latent_heat_j_kg(temp_c: float) -> float:
    """IAPWS 汽化潜热 h_fg (J/kg)"""
    _, h_fg, _, _ = sat_vapor_props(temp_c)
    return h_fg


def orifice_mass_flow(p_up: float, p_down: float, temp_k: float, area: float, cd: float = 0.68) -> float:
    """带零压差平滑正则化的可压缩蒸汽孔口质量流量公式 (kg/s)"""
    if area <= 1e-12 or p_up <= p_down:
        return 0.0
    dp = p_up - p_down
    r_crit = (2.0 / (GAMMA_STEAM + 1.0)) ** (GAMMA_STEAM / (GAMMA_STEAM - 1.0))
    p_eff = max(p_down, p_up * r_crit)
    ratio = p_eff / p_up
    term = max(0.0, (ratio ** (2.0 / GAMMA_STEAM)) - (ratio ** ((GAMMA_STEAM + 1.0) / GAMMA_STEAM)))
    psi = math.sqrt((2.0 * GAMMA_STEAM / (GAMMA_STEAM - 1.0)) * term)
    smooth = dp / ((dp * dp + 250.0 * 250.0) ** 0.25) * ((dp) ** -0.5 if dp > 1e-6 else 0.0)
    mdot_ideal = cd * area * p_up * psi / math.sqrt(R_STEAM * max(290.0, temp_k))
    return mdot_ideal * min(1.0, smooth) if dp < 1500.0 else mdot_ideal


def _bilinear_interp(
    x_grid: np.ndarray, y_grid: np.ndarray, table: np.ndarray, x: float, y: float
) -> float:
    """二维网格双线性插值"""
    xc = float(np.clip(x, x_grid[0], x_grid[-1]))
    yc = float(np.clip(y, y_grid[0], y_grid[-1]))
    ix = min(len(x_grid) - 2, max(0, int(np.searchsorted(x_grid, xc) - 1)))
    iy = min(len(y_grid) - 2, max(0, int(np.searchsorted(y_grid, yc) - 1)))
    tx = (xc - x_grid[ix]) / max(1e-12, x_grid[ix + 1] - x_grid[ix])
    ty = (yc - y_grid[iy]) / max(1e-12, y_grid[iy + 1] - y_grid[iy])
    v00, v10 = table[ix, iy], table[ix + 1, iy]
    v01, v11 = table[ix, iy + 1], table[ix + 1, iy + 1]
    return float((1.0 - tx) * ((1.0 - ty) * v00 + ty * v01) + tx * ((1.0 - ty) * v10 + ty * v11))


class AsterD51Thermo1D:
    """Aster D51 热力学、气液两相极限损伤与摩擦耦合状态机"""

    def __init__(self, calib: CylinderCalibration, lut_path: Optional[Path] = None):
        self.calib = calib

        # 1. 严格对应图纸标准件的几何常数 (SI 单位)
        self.A_head = math.pi * (0.015 / 2.0) ** 2                      # PR-15: 176.71 mm^2
        self.A_crank = math.pi * ((0.015**2 - 0.003**2) / 4.0)          # PF3-1.9: 169.65 mm^2
        self.V_clear = 0.12 * self.A_head * calib.stroke                # 12% 单腔余隙容积 (~0.424 ml)
        self.port_breadth = 0.0055                                      # 平面滑阀气口展宽 5.5mm
        self.A_valve_top = 4.6e-5                                       # 1-6 平面滑阀受蒸汽室下压净背面积 46 mm^2
        self.A_valve_port_relief = 1.2e-5                               # 底部气口向上卸荷投影面积 12 mm^2

        self.A_axle_pump = math.pi * (0.0032 / 2.0) ** 2                # PN3.2-1: 8.04 mm^2
        self.A_hand_pump = math.pi * (0.0070 / 2.0) ** 2                # PN7-1.9: 38.48 mm^2
        self.hand_pump_stroke = 0.016                                   # 手摇泵有效行程 16mm

        # 2. 锅筒、紫铜火室顶板与煤水车容积基准
        self.V_boiler_total = 3.80e-4                                   # 锅筒内总容积 380 ml
        self.M_copper_shell = 0.395                                     # 锅筒外筒与管板紫铜质量 kg
        self.M_copper_crown = 0.065                                     # 火室顶板及烟管前端受火紫铜质量 kg
        self.water_min_ml = 140.0                                       # 水位计 0% 对应最低安全水量
        self.water_max_ml = 280.0                                       # 水位计 100% 对应水量 (50% = 210 ml)

        # 3. 自动探测并挂载 RTX 5080 烟室 3D CFD 查找表 (O5)
        self.lut: Optional[Dict[str, np.ndarray]] = None
        default_lut = lut_path or (Path(__file__).resolve().parent.parent / "configs" / "smokebox_lut.json")
        if default_lut.exists():
            self.load_smokebox_lut(default_lut)

        # 4. 初始物理状态
        self.reset_cold_prepared()

    def load_smokebox_lut(self, path: Path) -> None:
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self.lut = {
            "exh_g_s": np.asarray(raw["mdot_exhaust_grid_g_s"], dtype=np.float64),
            "blw_g_s": np.asarray(raw["mdot_blower_grid_g_s"], dtype=np.float64),
            "draft_eff": np.asarray(raw["draft_eff_table"], dtype=np.float64),
            "eta_boil": np.asarray(raw["eta_boil_table"], dtype=np.float64),
            "sh_delta_c": np.asarray(raw["superheat_delta_c_table"], dtype=np.float64),
            "vacuum_pa": np.asarray(raw["smokebox_vacuum_pa_table"], dtype=np.float64),
        }

    def reset_cold_prepared(self) -> None:
        """重置为冷炉整备完毕状态：水位 50%、酒精 130ml、煤水车水 80%、机油 80%"""
        self.T_boiler_c = 25.0
        self.T_crown_c = 25.0
        self.M_water_kg = 0.210                                         # 210g -> 水位计 50%
        self.water_free_z_mm = water_volume_to_free_z_mm(210.0)
        self.P_boiler = P_ATM
        self.P_steam_chest = P_ATM
        self.T_superheat_c = 25.0
        self.T_sh_wall_c = 25.0                                         # 17-5 双 U 四程过热器 SUS304 管壁温度 (°C)
        self.safety_valve_open = False
        self.safety_valve_lift_mm = 0.0
        self.safety_valve_vel_mm_s = 0.0
        self.safety_huddling_kgf = 0.0
        self.siphon_seal_water_pct = 92.0
        self.roscoe_quill_oil_rate_mg_min = 0.0

        self.alcohol_ml = 130.0
        self.sump_fuel_ml = 0.0
        self.tender_water_cap_ml = 340.0                                # 31-10 水箱满容积 340 ml
        self.tender_water_ml = 272.0                                    # 说明书 p.39: 标准 8 分满 (80% = 272 ml)，预留回水余量
        self.T_feedwater_c = 25.0
        self.mariotte_flood_active = False
        self._mariotte_bubble_acc_ml = 0.0
        self._mariotte_bubble_timer = 0.0
        self.lube_tank_cap_ml = 3.2                                     # 4-2 前车架卧式置换注油器容积 (3.2 ml)
        self.lube_condensate_ml = 0.20 * self.lube_tank_cap_ml          # 沉底凝结水体积 (初始 80% 蒸汽油)
        self.lubricator_oil_pct = 80.0

        self.P_cyl_L = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.P_cyl_R = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.T_gas_L = np.array([298.15, 298.15], dtype=np.float64)     # 左缸前/后腔蒸汽实时绝对温度 (K, 截止后绝热膨胀降温!)
        self.T_gas_R = np.array([298.15, 298.15], dtype=np.float64)     # 右缸前/后腔蒸汽实时绝对温度 (K)
        self.T_cyl_wall_c = 25.0
        # 左右气缸 [前腔 Head, 后腔 Crank] 液态冷凝水质量 (kg)
        self.M_cond_L = np.array([8.0e-5, 8.0e-5], dtype=np.float64)
        self.M_cond_R = np.array([8.0e-5, 8.0e-5], dtype=np.float64)
        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

        # 6-5 轴动给水泵真实可压缩液腔状态与抽吸真空/止回球状态
        self.P_pump_chamber = P_ATM
        self.P_fw_manifold = P_ATM
        self.pump_suction_vac_kpa = 0.0
        self.pump_shaft_power_w = 0.0
        self.pump_ball_lifts_mm = [0.0, 0.0, 0.0]  # [6-5 下吸球, 6-5 上排球, 13-2 入炉球]

        # 4-2 蒸汽油与 36-4->23-1 酒精真实温度-黏度-压力-重力状态
        self.T_lube_oil_c = 25.0
        self.lube_oil_visc_mpas = 1850.0
        self.T_alcohol_c = 25.0
        self.alcohol_visc_mpas = 1.08

        # 锅筒顶空真实水蒸气质量守恒 (消除 0ml 干锅时分支跳变与虚假蒸発中读数!)
        v_gas_0 = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)
        self.M_vap_kg = (P_ATM * v_gas_0) / (R_STEAM * 298.15)
        self.T_vap_c = 25.0
        self.flue_fsi_mult = 1.0

        # 华氏配气实时截止率与仪表平滑器状态
        self.cutoff_pct = 0.0
        self._prev_adm_open_L = False
        self._prev_adm_open_R = False
        self._measured_cutoff_pct = 0.0
        self._ema_exhaust_gs = 0.0
        self._ema_evap_gs = 0.0
        self._ema_heat_w = 0.0
        self._ema_eta_pct = 0.0
        self._ema_vac_pa = 0.0
        self._ema_ind_work_w = 0.0
        self._ema_spec_steam = 0.0

        self.copper_anneal_damage = 0.0
        self.seal_health = 1.0
        self._flash_overpressure_pa = 0.0
        self._hand_pump_pending_kg = 0.0
        self._hand_pump_pino_ml = 0.0
        self.pg_bourdon_kgf = 0.0
        self.sight_glass_level_pct = 50.0
        self.sg_meniscus_z_mm = 55.5
        self.sg_vel_m_s = 0.0
        self.P_sg_top = P_ATM
        self._sg_trapped_pa_mm = P_ATM * (61.5 - 55.5)
        self.sg_steam_lumen_cond: float = 1.0
        self.sg_water_lumen_cond: float = 1.0
        self.turret_pressure_kgf = 0.0
        self.mariotte_headspace_kpa = 2.50
        self.steam_dryness_dome = 0.96
        self.steam_dryness_chest = 0.96
        # L2 7-Operator PINO 实时物理神经算子注入槽 (为 None 时使用 1D + 离线 CFD LUT 兜底)
        self.ext_pino_burner: Optional[Dict[str, float]] = None
        self.ext_pino_cyl_L: Optional[Tuple[float, float, float, float, float, float]] = None
        self.ext_pino_cyl_R: Optional[Tuple[float, float, float, float, float, float]] = None
        self.ext_pino_cab: Optional[Tuple[float, ...]] = None
        self.ext_pino_tender: Optional[Tuple[float, ...]] = None
        self.ext_pino_safety: Optional[Tuple[float, ...]] = None
        self.ext_pino_wick_cht: Optional[Tuple[float, ...]] = None
        self.ext_pino_elastic: Optional[Dict[str, Any]] = None
        self.burner_lumen_cond: float = 1.0
        self.insulation_coverage: float = 0.86
        self.wick_saturation: float = 0.90
        self.wick_tip_temp_c: float = 25.0
        self.casing_temp_c: float = 25.0
        self.boiler_efficiency_pct: float = 0.0

    def reset_hot_ready(self) -> None:
        """一键热炉预设：气压 3.8 kgf/cm²、水位 50%、暖缸就绪，可直接开阀发车"""
        self.reset_cold_prepared()
        self.T_boiler_c = 149.6
        self.T_vap_c = 149.6
        self.T_crown_c = 168.5
        self.P_boiler = sat_pressure_pa(self.T_boiler_c)
        v_gas_h = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)
        self.M_vap_kg = (self.P_boiler * v_gas_h) / (R_STEAM * (self.T_boiler_c + 273.15))
        self.P_steam_chest = P_ATM
        self.P_sg_top = self.P_boiler
        self._sg_trapped_pa_mm = self.P_boiler * (61.5 - self.sg_meniscus_z_mm)
        self.pg_bourdon_kgf = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        self.turret_pressure_kgf = self.pg_bourdon_kgf
        self.T_superheat_c = 165.0
        self.T_sh_wall_c = 172.0
        self.sump_fuel_ml = 3.5
        self.T_cyl_wall_c = 132.0
        self.T_gas_L[:] = 132.0 + 273.15
        self.T_gas_R[:] = 132.0 + 273.15
        self.T_lube_oil_c = 78.0
        self.lube_oil_visc_mpas = 92.0
        self.T_alcohol_c = 31.5
        self.alcohol_visc_mpas = 0.96
        self.wick_saturation = 0.72
        self.wick_tip_temp_c = 82.8
        self.casing_temp_c = 71.2
        self.steam_dryness_dome = 0.965
        self.steam_dryness_chest = 1.00
        self.M_cond_L[:] = 1.5e-5
        self.M_cond_R[:] = 1.5e-5
        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

    def trigger_hand_pump(self, strokes: int = 5) -> None:
        """摇动煤水车手动给水泵 (31-5): 约 30 次可补满半锅水位"""
        vol_ml = strokes * (self.A_hand_pump * self.hand_pump_stroke * 1e6) * 1.15
        if self.ext_pino_tender is not None:
            # 挂载 TenderHydraulicPINO 时不预扣水箱水量，直接排队真实柱塞冲程交由 PINO 解算五止回球竞争与旁通回流!
            self._hand_pump_pino_ml += min(vol_ml, self.tender_water_ml)
        else:
            avail_ml = min(vol_ml, self.tender_water_ml)
            self.tender_water_ml -= avail_ml
            self._hand_pump_pending_kg += avail_ml * 1e-3

    def _valve_port_areas(self, q_valve: float, q_center: float) -> Tuple[float, float, float, float]:
        xv = q_valve - q_center
        es = self.calib.steam_lap
        ee = self.calib.exhaust_lap
        wp = self.calib.port_width
        b = self.port_breadth

        a_adm_head = b * min(wp, max(0.0, -xv - es))
        a_exh_head = b * min(wp, max(0.0, +xv - ee))
        a_adm_crank = b * min(wp, max(0.0, +xv - es))
        a_exh_crank = b * min(wp, max(0.0, -xv - ee))
        return a_adm_head, a_exh_head, a_adm_crank, a_exh_crank

    def _compute_slide_valve_friction(self, p_pair: np.ndarray, v_valve: float) -> float:
        """O3-2: 外进气平面 D 型滑阀受蒸汽室表压压紧在镜面上的动态库仑+黏性摩擦反力 (N)"""
        dp_chest = max(0.0, self.P_steam_chest - P_ATM)
        dp_ports = 0.5 * (max(0.0, p_pair[0] - P_ATM) + max(0.0, p_pair[1] - P_ATM))
        normal_load = max(0.0, dp_chest * self.A_valve_top - dp_ports * self.A_valve_port_relief)
        mu_film = 0.042 if self.lubricator_oil_pct > 2.0 else 0.135
        mu_eff = mu_film * (2.0 - self.seal_health)
        return -mu_eff * normal_load * math.tanh(v_valve / 0.015) - 0.10 * v_valve

    def _compute_walschaerts_cutoff_pct(self, reverser: float) -> float:
        """
        根据原厂华氏配气机构几何参数 (外余面 e_s=1.36mm, 恒定先导量 lead=0.28mm, 满档半行程 R_max=2.65mm)
        解析计算给定螺旋逆转机档位 reverser ∈ [-1, +1] 下的理论进汽截止率 (% 行程)。
        """
        u = min(1.0, abs(float(reverser)))
        es = self.calib.steam_lap * 1000.0
        a_lead = es + 0.28
        b_quad = u * math.sqrt(max(0.0, 2.65 ** 2 - a_lead ** 2))
        r_amp = math.hypot(a_lead, b_quad)
        if r_amp <= es + 1e-6:
            return 0.0
        phi = math.atan2(a_lead, max(1e-9, b_quad))
        theta_cut = math.pi - phi - math.asin(min(1.0, es / r_amp))
        theta_cut = max(0.0, min(math.pi, theta_cut))
        return 50.0 * (1.0 - math.cos(theta_cut))

    def _step_cylinder_side(
        self,
        dt_sub: float,
        p_pair: np.ndarray,
        t_gas_pair: np.ndarray,
        m_cond_pair: np.ndarray,
        q_piston: float,
        v_piston: float,
        q_fdc: float,
        areas: Tuple[float, float, float, float],
        drain_open: bool,
        pino_cyl: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> Tuple[float, float, float, float, float, float, bool]:
        """
        单侧双作用气缸开口系第一定律真实气体多变膨胀做功 (dU = dm_in*h_in - dm_out*h - P*dV - Q_wall)、
        1-16x2 气液两相均相平衡闪蒸排液吹除与不可压缩水锤 (Hydrolock) 子步求解。
        返回: (进汽 kg/s, 乏汽排汽 kg/s, 排液阀总吹除 kg/s, 冷凝生成 kg/s, 指示膨胀功率 W, 水锤反力 N, 水锤标志)
        """
        a_adm_h, a_exh_h, a_adm_c, a_exh_c = areas
        t_sh_k = max(295.0, self.T_superheat_c + 273.15)
        t_wall_k = max(290.0, self.T_cyl_wall_c + 273.15)
        cp_steam = (GAMMA_STEAM / (GAMMA_STEAM - 1.0)) * R_STEAM
        cv_steam = (1.0 / (GAMMA_STEAM - 1.0)) * R_STEAM

        # 1-16 齿轮旋塞排液孔 (ø1.1mm, A=0.95mm²; 开启时通畅吹除，关闭但超压 > P_chest+28kPa 时阀芯微量顶起泄压)
        a_drain_base = 3.6e-7 if drain_open else 0.0
        a_blowby = 2.2e-7 * (1.0 - self.seal_health)

        s_p = min(self.calib.stroke, max(0.0, q_fdc - q_piston))
        ds_dt = -v_piston

        v_h_tot = self.V_clear + self.A_head * s_p
        dv_h = +self.A_head * ds_dt
        v_c_tot = self.V_clear + self.A_crank * (self.calib.stroke - s_p)
        dv_c = -self.A_crank * ds_dt

        # 扣除腔内液态冷凝水占据的不可压缩体积，得到有效气相空间
        vol_w_h = m_cond_pair[0] / RHO_LIQ_WATER
        vol_w_c = m_cond_pair[1] / RHO_LIQ_WATER
        v_h_gas = max(0.08 * self.V_clear, v_h_tot - vol_w_h)
        v_c_gas = max(0.08 * self.V_clear, v_c_tot - vol_w_c)

        t_h_k = float(np.clip(t_gas_pair[0], 280.0, 680.0))
        t_c_k = float(np.clip(t_gas_pair[1], 280.0, 680.0))

        # 未开排液阀且腔内积水超限 (>0.48 * V_clear) 时，气液两相乳化阻塞滑阀排气口产生高背压憋死
        exh_choke_h = 0.14 if (not drain_open and vol_w_h > 0.48 * self.V_clear) else 1.0
        exh_choke_c = 0.14 if (not drain_open and vol_w_c > 0.48 * self.V_clear) else 1.0

        if pino_cyl is not None:
            cd_ah, cd_eh, cd_ac, cd_ec, _, cond_nu_mult = pino_cyl
        else:
            cd_ah = cd_eh = cd_ac = cd_ec = 0.68
            cond_nu_mult = 1.0

        # 超压自动顶阀排液保护面积
        over_p_h = max(0.0, p_pair[0] - max(self.P_steam_chest + 28000.0, P_ATM + 45000.0))
        over_p_c = max(0.0, p_pair[1] - max(self.P_steam_chest + 28000.0, P_ATM + 45000.0))
        a_drn_h = a_drain_base + min(2.4e-7, 1.2e-11 * over_p_h)
        a_drn_c = a_drain_base + min(2.4e-7, 1.2e-11 * over_p_c)

        # 1-16 排液阀气液两相均相平衡流 (Homogeneous-Equilibrium Two-Phase Blowdown):
        # 底部积水占比 alpha_w 越高，吹除射流中液相夹带密度越高，积水排净后自动过渡为纯蒸汽射流
        alpha_w_h = min(0.94, vol_w_h / max(1e-12, vol_w_h + 0.10 * self.V_clear)) if m_cond_pair[0] > 1e-9 else 0.0
        alpha_w_c = min(0.94, vol_w_c / max(1e-12, vol_w_c + 0.10 * self.V_clear)) if m_cond_pair[1] > 1e-9 else 0.0
        dp_drn_h = max(0.0, p_pair[0] - P_ATM + RHO_LIQ_WATER * 9.81 * 0.015 * alpha_w_h)
        dp_drn_c = max(0.0, p_pair[1] - P_ATM + RHO_LIQ_WATER * 9.81 * 0.015 * alpha_w_c)

        m_drn_liq_h = 0.64 * (a_drn_h * alpha_w_h) * math.sqrt(2.0 * RHO_LIQ_WATER * dp_drn_h) if a_drn_h > 0.0 else 0.0
        m_drn_liq_c = 0.64 * (a_drn_c * alpha_w_c) * math.sqrt(2.0 * RHO_LIQ_WATER * dp_drn_c) if a_drn_c > 0.0 else 0.0
        m_drn_vap_h = orifice_mass_flow(p_pair[0], P_ATM, t_h_k, a_drn_h * (1.0 - 0.82 * alpha_w_h), cd=0.66)
        m_drn_vap_c = orifice_mass_flow(p_pair[1], P_ATM, t_c_k, a_drn_c * (1.0 - 0.82 * alpha_w_c), cd=0.66)

        m_in_h = orifice_mass_flow(self.P_steam_chest, p_pair[0], t_sh_k, a_adm_h, cd=cd_ah)
        m_out_h = orifice_mass_flow(p_pair[0], P_ATM, t_h_k, a_exh_h * exh_choke_h, cd=cd_eh)

        m_in_c = orifice_mass_flow(self.P_steam_chest, p_pair[1], t_sh_k, a_adm_c, cd=cd_ac)
        m_out_c = orifice_mass_flow(p_pair[1], P_ATM, t_c_k, a_exh_c * exh_choke_c, cd=cd_ec)

        # 前后腔活塞环窜汽
        if p_pair[0] >= p_pair[1]:
            m_blow = orifice_mass_flow(p_pair[0], p_pair[1], t_h_k, a_blowby)
            m_blow_in_h, m_blow_out_h = 0.0, m_blow
            m_blow_in_c, m_blow_out_c = m_blow, 0.0
            t_blow_in_h, t_blow_in_c = t_c_k, t_h_k
        else:
            m_blow_mag = orifice_mass_flow(p_pair[1], p_pair[0], t_c_k, a_blowby)
            m_blow = -m_blow_mag
            m_blow_in_h, m_blow_out_h = m_blow_mag, 0.0
            m_blow_in_c, m_blow_out_c = 0.0, m_blow_mag
            t_blow_in_h, t_blow_in_c = t_c_k, t_h_k

        # 湿蒸汽夹带液滴 + 冷气缸壁湍流冲击凝结积水 vs 1-16 两相闪蒸排液吹除
        t_sat_chest = sat_temp_c(self.P_steam_chest)
        subcool_c = max(0.0, t_sat_chest - self.T_cyl_wall_c - 6.0)
        wet_carry = max(0.0, 1.0 - self.steam_dryness_chest)
        cond_frac = min(0.88, 0.024 * subcool_c * cond_nu_mult + wet_carry)
        mdot_cond_gen = 3.2 * cond_frac * (m_in_h + m_in_c)
        dm_w_h = 3.2 * cond_frac * m_in_h - m_drn_liq_h
        dm_w_c = 3.2 * cond_frac * m_in_c - m_drn_liq_c
        if not drain_open:
            if a_exh_h > 1e-8 and vol_w_h > 0.85 * self.V_clear:
                dm_w_h -= 0.000015 * (m_out_h / max(1e-6, m_out_h + 2e-4))
            if a_exh_c > 1e-8 and vol_w_c > 0.85 * self.V_clear:
                dm_w_c -= 0.000015 * (m_out_c / max(1e-6, m_out_c + 2e-4))

        max_cond_chamber = 3.20 * self.V_clear * RHO_LIQ_WATER
        w_prev_h, w_prev_c = m_cond_pair[0], m_cond_pair[1]
        m_cond_pair[0] = float(np.clip(w_prev_h + dm_w_h * dt_sub, 0.0, max_cond_chamber))
        m_cond_pair[1] = float(np.clip(w_prev_c + dm_w_c * dt_sub, 0.0, max_cond_chamber))
        actual_drn_liq_h = max(0.0, (w_prev_h + 3.2 * cond_frac * m_in_h * dt_sub - m_cond_pair[0]) / max(1e-9, dt_sub))
        actual_drn_liq_c = max(0.0, (w_prev_c + 3.2 * cond_frac * m_in_c * dt_sub - m_cond_pair[1]) / max(1e-9, dt_sub))

        # 气缸壁面对流传热 (Woschni-like 活塞速度增强换热系数)
        h_conv_cyl = 58.0 + 340.0 * abs(v_piston)
        a_wall_h = 2.0 * self.A_head + math.pi * 0.015 * (s_p + 0.002)
        a_wall_c = 2.0 * self.A_crank + math.pi * 0.015 * ((self.calib.stroke - s_p) + 0.002)
        q_wall_h = h_conv_cyl * a_wall_h * (t_h_k - t_wall_k)
        q_wall_c = h_conv_cyl * a_wall_c * (t_c_k - t_wall_k)

        # 开口系热力学第一定律精确解算腔压 P 与绝热膨胀降温 T_gas:
        # 进汽截止后 (m_in=0, m_out=0) 气缸内封闭蒸汽按 P*V^gamma 绝热膨胀继续推动活塞做功且零额外耗汽!
        m_in_eff_h = (1.0 - 0.5 * cond_frac) * m_in_h
        m_in_eff_c = (1.0 - 0.5 * cond_frac) * m_in_c
        m_leave_h = m_out_h + m_drn_vap_h + m_blow_out_h
        m_leave_c = m_out_c + m_drn_vap_c + m_blow_out_c

        dp_h = ((GAMMA_STEAM - 1.0) / v_h_gas) * (
            m_in_eff_h * cp_steam * t_sh_k
            + m_blow_in_h * cp_steam * t_blow_in_h
            - m_leave_h * cp_steam * t_h_k
            - q_wall_h
        ) - (GAMMA_STEAM * p_pair[0] / v_h_gas) * dv_h

        dp_c = ((GAMMA_STEAM - 1.0) / v_c_gas) * (
            m_in_eff_c * cp_steam * t_sh_k
            + m_blow_in_c * cp_steam * t_blow_in_c
            - m_leave_c * cp_steam * t_c_k
            - q_wall_c
        ) - (GAMMA_STEAM * p_pair[1] / v_c_gas) * dv_c

        m_gas_h = max(1.2e-9, (p_pair[0] * v_h_gas) / (R_STEAM * t_h_k))
        m_gas_c = max(1.2e-9, (p_pair[1] * v_c_gas) / (R_STEAM * t_c_k))
        dT_h = (
            m_in_eff_h * (cp_steam * t_sh_k - cv_steam * t_h_k)
            + m_blow_in_h * (cp_steam * t_blow_in_h - cv_steam * t_h_k)
            - m_leave_h * R_STEAM * t_h_k
            - p_pair[0] * dv_h
            - q_wall_h
        ) / (m_gas_h * cv_steam)
        dT_c = (
            m_in_eff_c * (cp_steam * t_sh_k - cv_steam * t_c_k)
            + m_blow_in_c * (cp_steam * t_blow_in_c - cv_steam * t_c_k)
            - m_leave_c * R_STEAM * t_c_k
            - p_pair[1] * dv_c
            - q_wall_c
        ) / (m_gas_c * cv_steam)

        p_relief = max(self.P_steam_chest + 32000.0, P_ATM + 48000.0)
        p_pair[0] = min(p_relief, max(0.82 * P_ATM, p_pair[0] + dp_h * dt_sub))
        p_pair[1] = min(p_relief, max(0.82 * P_ATM, p_pair[1] + dp_c * dt_sub))
        t_gas_pair[0] = float(np.clip(t_h_k + dT_h * dt_sub, 280.0, max(360.0, t_sh_k + 25.0)))
        t_gas_pair[1] = float(np.clip(t_c_k + dT_c * dt_sub, 280.0, max(360.0, t_sh_k + 25.0)))

        # 双腔瞬时指示膨胀机械功率 P_ind = (P_h - P_atm)*dV_h/dt + (P_c - P_atm)*dV_c/dt (W)
        p_ind_w = (p_pair[0] - P_ATM) * dv_h + (p_pair[1] - P_ATM) * dv_c

        # 不可压缩液态水锤 (Hydrolock) 单边接触冲击模型:
        f_hydro_x = 0.0
        hydrolock = False
        if not drain_open and vol_w_h > 0.48 * self.V_clear:
            over_h = vol_w_h - 0.34 * v_h_tot
            if over_h > 0.0:
                hydrolock = True
                if v_piston > -0.002:
                    f_hydro_x -= min(420.0, 3.5e8 * over_h * (1.0 if v_piston >= 0.0 else 0.15) + 3200.0 * max(0.0, v_piston))

        if not drain_open and vol_w_c > 0.48 * self.V_clear:
            over_c = vol_w_c - 0.34 * v_c_tot
            if over_c > 0.0:
                hydrolock = True
                if v_piston < 0.002:
                    f_hydro_x += min(420.0, 3.5e8 * over_c * (1.0 if v_piston <= 0.0 else 0.15) + 3200.0 * max(0.0, -v_piston))

        mdot_drain_tot = actual_drn_liq_h + actual_drn_liq_c + m_drn_vap_h + m_drn_vap_c
        return (m_in_h + m_in_c), (m_out_h + m_out_c), mdot_drain_tot, mdot_cond_gen, p_ind_w, f_hydro_x, hydrolock

    def step(
        self,
        dt: float,
        q_pL: float,
        v_pL: float,
        q_vL: float,
        q_pR: float,
        v_pR: float,
        q_vR: float,
        omega_3: float,
        v_pump_ram: float,
        ctrl: CabControls,
        sim_time: float = 0.0,
        v_vL: float = 0.0,
        v_vR: float = 0.0,
        pitch_rad: float = 0.0,
        accel_x: float = 0.0,
        v_hand_ram: float = 0.0,
    ) -> Tuple[float, float, float, ThermoTelemetry]:
        """
        推进单个联合仿真步 dt (如 0.5ms)，返回:
        (左活塞推力 F_L, 右活塞推力 F_R, 给水泵柱塞反力 F_pump, 遥测数据包)
        """
        dt_th = dt * max(1.0, ctrl.thermal_warp)

        # --- 1. 煤水车 36-4 马氏瓶恒液位酒精箱 -> 34-2 サンプ -> ST-ø7 -> 23-1 真实温度/黏度/蒸气压/重力/Poiseuille 流 ---
        theta_grade_only = max(-0.30, min(0.30, pitch_rad + math.radians(ctrl.grade_deg)))
        # 供醇管路乙醇温度随火室辐射与车底迎风/煤水车常温来流醇动态热平衡 (常态 ~36-44°C，严重干烧辐射时升至 ~55-64°C):
        q_alc_rad_w = (0.028 if ctrl.burner_lit else 0.005) * max(0.0, 0.22 * self.T_crown_c + 0.35 * self.T_boiler_c - self.T_alcohol_c)
        q_alc_cool_w = (0.19 + 0.015 * min(35.0, abs(omega_3))) * (self.T_alcohol_c - 25.0)
        self.T_alcohol_c = float(np.clip(
            self.T_alcohol_c + ((q_alc_rad_w - q_alc_cool_w) / 18.0) * dt_th,
            18.0, 74.0
        ))
        # 燃料乙醇真实物理物性: Andrade 动态黏度 mu(T)、热膨胀密度 rho(T) 与 Antoine 饱和蒸气压 P_vap(T)
        t_alc_k = self.T_alcohol_c + 273.15
        self.alcohol_visc_mpas = 1.08 * math.exp(1680.0 * (1.0 / t_alc_k - 1.0 / 298.15))
        rho_ethanol = 789.0 * (1.0 - 0.00108 * (self.T_alcohol_c - 25.0))
        p_vap_eth_kpa = 0.133322 * (10.0 ** (8.20417 - 1642.89 / (230.30 + self.T_alcohol_c)))
        vap_head_boost_kpa = max(0.0, p_vap_eth_kpa - 7.87) * 0.08  # 升温蒸气压削弱马氏瓶顶空净负压

        cap_open_flag = bool(getattr(ctrl, "fuel_cap_open", False))
        # 34-2 サンプ (X=-268mm) 至 23-1 燃烧杯 (X=-92mm, ΔX=+176mm) 重力液位差 Δh_grav (含坡道俯仰角!)
        dh_grade_m = -0.176 * math.sin(theta_grade_only)
        visc_fluidity_eth = 1.08 / max(0.35, self.alcohol_visc_mpas)

        if self.ext_pino_tender is not None:
            # 由 L2 TenderHydraulicPINO 解算马氏瓶密闭顶空真空度 (kPa) 与重力补醇流量 (ml/s)，叠加温度蒸气压与黏度修正
            self.mariotte_headspace_kpa = max(0.0, float(self.ext_pino_tender[9]) - vap_head_boost_kpa)
            q_fuel_pino = max(0.0, float(self.ext_pino_tender[10])) * (0.75 + 0.25 * visc_fluidity_eth)
            sump_equil_ml = max(1.6, 3.5 + 3.8 * math.exp(-self.mariotte_headspace_kpa / 0.08) + 85.0 * dh_grade_m)
            refill_ml = min(self.alcohol_ml, q_fuel_pino * dt_th, max(0.0, sump_equil_ml - self.sump_fuel_ml))
            self.alcohol_ml -= refill_ml
            self.sump_fuel_ml += refill_ml
            self.mariotte_flood_active = bool(self.sump_fuel_ml > 4.25 or (self.mariotte_headspace_kpa < 0.05 and q_fuel_pino > 0.65))
            if self.mariotte_headspace_kpa > 0.15 and refill_ml > 0.0:
                self._mariotte_bubble_acc_ml += refill_ml
                if self._mariotte_bubble_acc_ml >= 0.22:
                    self._mariotte_bubble_acc_ml = 0.0
                    self._mariotte_bubble_timer = 0.45
        else:
            self.mariotte_flood_active = bool(
                cap_open_flag and ctrl.fuel_valve > 0.05 and self.alcohol_ml > 0.5
            )
            alc_norm_1d = max(0.0, min(1.0, self.alcohol_ml / 130.0))
            raw_vac_kpa = 0.0 if cap_open_flag else (0.35 + 2.15 * alc_norm_1d) * (1.0 if self.alcohol_ml > 0.5 else 0.0)
            self.mariotte_headspace_kpa = max(0.0, raw_vac_kpa - vap_head_boost_kpa)
            sump_target_ml = max(1.6, (7.2 if self.mariotte_flood_active else 3.5) + 85.0 * dh_grade_m)
            # Hagen-Poiseuille 管路流导 G ∝ rho * g * Δh / mu(T)
            grav_drive = max(0.25, 1.0 + dh_grade_m / 0.018)
            refill_rate_coeff = (1.35 if self.mariotte_flood_active else 0.45) * visc_fluidity_eth * grav_drive
            if ctrl.fuel_valve > 0.01 and self.alcohol_ml > 0.0 and self.sump_fuel_ml < sump_target_ml:
                refill_ml = min(self.alcohol_ml, refill_rate_coeff * ctrl.fuel_valve * dt_th, sump_target_ml - self.sump_fuel_ml)
                self.alcohol_ml -= refill_ml
                self.sump_fuel_ml += refill_ml
                if not cap_open_flag:
                    self._mariotte_bubble_acc_ml += refill_ml
                    if self._mariotte_bubble_acc_ml >= 0.22:
                        self._mariotte_bubble_acc_ml = 0.0
                        self._mariotte_bubble_timer = 0.45
        if self._mariotte_bubble_timer > 0.0:
            self._mariotte_bubble_timer = max(0.0, self._mariotte_bubble_timer - dt_th)
        mariotte_bubbling = self._mariotte_bubble_timer > 0.0

        if self.sump_fuel_ml <= 1e-4:
            ctrl.burner_lit = False

        # --- 2. 蒸汽室 (Steam Chest)、华氏配气截止率检测与双缸 4 子步开口系真实气体膨胀做功 + 两相排液/水锤 ---
        a_regulator = 3.2e-6 * (ctrl.regulator ** 1.35)
        t_boil_k = self.T_boiler_c + 273.15
        has_steam = self.P_boiler > P_ATM + 400.0

        if self.ext_pino_cab is not None and has_steam and ctrl.regulator > 1e-4:
            mdot_reg = max(0.0, float(self.ext_pino_cab[1])) * 1e-3
        else:
            mdot_reg = orifice_mass_flow(self.P_boiler, self.P_steam_chest, t_boil_k, a_regulator) if has_steam else 0.0

        areas_L = self._valve_port_areas(q_vL, self.calib.q_valve_center_L)
        areas_R = self._valve_port_areas(q_vR, self.calib.q_valve_center_R)

        # 实时捕捉华氏滑阀进汽口关闭瞬间 (Admission Cutoff) 的真实活塞行程百分比，并与解析华氏椭圆融合
        theory_cutoff_pct = self._compute_walschaerts_cutoff_pct(ctrl.reverser)
        s_L_now = min(self.calib.stroke, max(0.0, self.calib.q_fdc_L - q_pL))
        s_R_now = min(self.calib.stroke, max(0.0, self.calib.q_fdc_R - q_pR))
        adm_open_L = (areas_L[0] > 1e-9) or (areas_L[2] > 1e-9)
        adm_open_R = (areas_R[0] > 1e-9) or (areas_R[2] > 1e-9)
        if abs(omega_3) > 0.8 and abs(ctrl.reverser) > 0.05:
            if self._prev_adm_open_L and not adm_open_L:
                frac_L = (s_L_now / self.calib.stroke) if v_pL < 0.0 else (1.0 - s_L_now / self.calib.stroke)
                if 0.06 <= frac_L <= 0.92:
                    self._measured_cutoff_pct = 100.0 * frac_L
            if self._prev_adm_open_R and not adm_open_R:
                frac_R = (s_R_now / self.calib.stroke) if v_pR < 0.0 else (1.0 - s_R_now / self.calib.stroke)
                if 0.06 <= frac_R <= 0.92:
                    self._measured_cutoff_pct = 100.0 * frac_R
            if self._measured_cutoff_pct > 1.0:
                self.cutoff_pct = 0.55 * theory_cutoff_pct + 0.45 * self._measured_cutoff_pct
            else:
                self.cutoff_pct = theory_cutoff_pct
        else:
            self._measured_cutoff_pct = theory_cutoff_pct
            self.cutoff_pct = theory_cutoff_pct
        self._prev_adm_open_L = adm_open_L
        self._prev_adm_open_R = adm_open_R

        n_sub = 4
        dt_sub = dt / n_sub
        mdot_cyl_in = 0.0
        mdot_cyl_exh = 0.0
        mdot_drain_blow = 0.0
        mdot_cond_tot = 0.0
        ind_work_w_acc = 0.0
        f_hydro_L_acc = 0.0
        f_hydro_R_acc = 0.0
        hydrolock_L = False
        hydrolock_R = False
        v_sc_vol = 2.8e-5

        for _ in range(n_sub):
            in_L, exh_L, drn_L, cgen_L, pind_L, fh_L, hl_L = self._step_cylinder_side(
                dt_sub, self.P_cyl_L, self.T_gas_L, self.M_cond_L, q_pL, v_pL, self.calib.q_fdc_L, areas_L, ctrl.drain_cock, self.ext_pino_cyl_L
            )
            in_R, exh_R, drn_R, cgen_R, pind_R, fh_R, hl_R = self._step_cylinder_side(
                dt_sub, self.P_cyl_R, self.T_gas_R, self.M_cond_R, q_pR, v_pR, self.calib.q_fdc_R, areas_R, ctrl.drain_cock, self.ext_pino_cyl_R
            )
            in_tot = in_L + in_R
            # 填料函老化外漏损失
            mdot_gland_leak = orifice_mass_flow(
                self.P_steam_chest, P_ATM, self.T_superheat_c + 273.15, 1.5e-7 * (1.0 - self.seal_health)
            )
            dp_sc = (R_STEAM * (self.T_superheat_c + 273.15) / v_sc_vol) * (mdot_reg - in_tot - mdot_gland_leak)
            self.P_steam_chest = min(self.P_boiler, max(P_ATM, self.P_steam_chest + dp_sc * dt_sub))
            mdot_cyl_in += in_tot / n_sub
            mdot_cyl_exh += (exh_L + exh_R) / n_sub
            mdot_drain_blow += (drn_L + drn_R) / n_sub
            mdot_cond_tot += (cgen_L + cgen_R) / n_sub
            ind_work_w_acc += (pind_L + pind_R) / n_sub
            f_hydro_L_acc += fh_L / n_sub
            f_hydro_R_acc += fh_R / n_sub
            hydrolock_L = hydrolock_L or hl_L
            hydrolock_R = hydrolock_R or hl_R

        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

        # --- 3. 气缸内衬凝结潜热 + 对流暖缸换热、4-2 真实温度/Walther 黏度/重力浮力/压脉置换注油器与密封退化 ---
        if mdot_cyl_in > 1e-6 or mdot_cond_tot > 1e-7:
            q_lat_cyl = min(920.0, mdot_cond_tot * 2.10e6)
            q_conv_cyl = max(0.0, (self.T_superheat_c - self.T_cyl_wall_c) * 2.2)
            self.T_cyl_wall_c = min(
                self.T_superheat_c,
                self.T_cyl_wall_c + ((q_lat_cyl + q_conv_cyl) / 48.0) * dt_th,
            )
        else:
            self.T_cyl_wall_c += (25.0 - self.T_cyl_wall_c) * 0.002 * dt_th

        # 4-2 前车架卧式置换式注油器 (ISO VG 460 蒸汽汽缸油真实 Walther 黏度 mu(T) + 水油密度差浮力压头 + 蒸汽脉动毛细管 Poiseuille 流):
        air_cool_factor = 1.0 + 0.025 * min(40.0, abs(omega_3))
        if self.P_steam_chest > P_ATM + 1500.0:
            t_sat_lube_c = sat_temp_c(self.P_steam_chest)
            q_oil_heat = 0.085 * max(0.0, t_sat_lube_c - self.T_lube_oil_c)
        else:
            t_sat_lube_c = 25.0
            q_oil_heat = 0.015 * max(0.0, self.T_cyl_wall_c - self.T_lube_oil_c)
        q_oil_cool = 0.024 * air_cool_factor * (self.T_lube_oil_c - 25.0)
        self.T_lube_oil_c = float(np.clip(self.T_lube_oil_c + ((q_oil_heat - q_oil_cool) / 6.2) * dt_th, 20.0, 165.0))
        # ISO VG 460 复配蒸汽汽缸油黏温方程 (25°C ~1850 mPa·s -> 95°C ~82 mPa·s -> 135°C ~38 mPa·s)
        self.lube_oil_visc_mpas = 0.42 * math.exp(1180.0 / (self.T_lube_oil_c + 115.0))
        rho_oil = 902.0 * (1.0 - 0.00065 * (self.T_lube_oil_c - 25.0))
        rho_cond_w = 997.0 - 0.18 * max(0.0, self.T_lube_oil_c - 25.0)

        if self.P_steam_chest > P_ATM + 2500.0 and self.lube_condensate_ml < self.lube_tank_cap_ml:
            steam_pulse_pa = abs(self.P_boiler - self.P_steam_chest) * min(1.0, mdot_cyl_in / 1.2e-4)
            # 水沉底产生的阿基米德浮力静压差 ΔP_buoy = (rho_w - rho_oil) * g * h_oil * cos(theta)
            h_oil_m = 0.012 * (self.lubricator_oil_pct / 100.0) * math.cos(theta_grade_only)
            dp_buoy_pa = max(0.0, (rho_cond_w - rho_oil) * 9.81 * h_oil_m)
            # 前车架 4-2 裸露黄铜油箱 (A≈16.5cm²) 蒸汽凝结散热 ~1.35 W -> 0.039 ml/min (2.4 ml / 60 min 原厂续航真值!)
            q_lube_cond_w = 0.0115 * air_cool_factor * (0.35 + 0.65 * min(1.0, mdot_cyl_in / 1.5e-4)) * max(0.0, t_sat_lube_c - 28.0)
            mdot_lube_cond_kg_s = q_lube_cond_w / latent_heat_j_kg(t_sat_lube_c)
            dv_cond_ml_s = (mdot_lube_cond_kg_s / rho_cond_w) * 1e6
            # ø0.30mm x 24mm 节流计量毛细孔 Hagen-Poiseuille 黏性通流上限 (24°C ~2000mPa·s 冷态节流 ~7mg/min，暖机后 ~35mg/min 满额置换!)
            mu_pa_s = self.lube_oil_visc_mpas * 1e-3
            dp_quill_pa = 260.0 + dp_buoy_pa + 0.012 * steam_pulse_pa
            q_poiseuille_max_ml_s = ((math.pi * (0.00030 ** 4) * dp_quill_pa) / (128.0 * max(1e-4, mu_pa_s) * 0.024)) * 1e6
            dv_disp_ml_s = min(dv_cond_ml_s, q_poiseuille_max_ml_s)
            dv_disp_ml = dv_disp_ml_s * dt_th
            self.lube_condensate_ml = min(self.lube_tank_cap_ml, self.lube_condensate_ml + dv_disp_ml)
            self.lubricator_oil_pct = max(
                0.0, 100.0 * (1.0 - self.lube_condensate_ml / self.lube_tank_cap_ml)
            )
            self.roscoe_quill_oil_rate_mg_min = round(dv_disp_ml_s * (rho_oil * 1e-3) * 1000.0 * 60.0, 3)
        else:
            self.roscoe_quill_oil_rate_mg_min = 0.0

        # O4-3: 过热 (>225°C)、干摩缺油或水锤冲击累积损伤密封圈
        seal_wear_rate = 0.0
        if self.T_superheat_c > 225.0:
            seal_wear_rate += 0.0008 * (self.T_superheat_c - 225.0)
        if self.lubricator_oil_pct < 1.0 and (abs(v_pL) + abs(v_pR)) > 0.08:
            seal_wear_rate += 0.0025
        if hydrolock_L or hydrolock_R:
            seal_wear_rate += 0.015
        self.seal_health = max(0.25, self.seal_health - seal_wear_rate * dt_th)

        lube_factor = 1.0 if self.lubricator_oil_pct > 2.0 else 2.8
        water_drag = (
            115.0 * min(1.0, max(0.0, self.condensate_kg - 0.00045) / 0.00085)
            if not ctrl.drain_cock
            else 0.8
        )

        def net_piston_force(p_pair: np.ndarray, vel: float, f_hydro: float) -> float:
            f_gas = (p_pair[1] - P_ATM) * self.A_crank - (p_pair[0] - P_ATM) * self.A_head
            f_seal = lube_factor * (1.35 * math.tanh(vel * 80.0) + water_drag * vel)
            return f_gas - f_seal + f_hydro

        F_L = net_piston_force(self.P_cyl_L, v_pL, f_hydro_L_acc)
        F_R = net_piston_force(self.P_cyl_R, v_pR, f_hydro_R_acc)
        F_vfric_L = self._compute_slide_valve_friction(self.P_cyl_L, v_vL)
        F_vfric_R = self._compute_slide_valve_friction(self.P_cyl_R, v_vR)

        # --- 4. 通风阀 (11-13)、银焊热损伤渗漏与 28-6x2 Gauge-1 蓄压环腔突跳安全阀 (零 boolean 迟滞开关!) ---
        p_gauge_kgf = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        # 求解 28-6 纵列双突跳安全阀真实二阶牛顿—蓄压环腔反冲裙边动力学 (零 boolean 迟滞开关!):
        #   阀座 ø2.20mm (A_seat = 3.8013 mm²), 蓄压环腔外径 ø2.60mm (ΔA_lip = 1.5080 mm²), 反冲裙边特征间隙 c_skirt = 0.035mm,
        #   dF_hud/dy|_{y=0+} = +6.1 N/mm >> k_s = 0.028 N/mm 满足流体静力正反馈失稳突跳条件 (4.000 kgf/cm² 瞬时全开 -> 3.56 kgf/cm² 干脆回座!)
        d_seat_sv = 2.20
        a_seat_mm2 = 3.801327
        da_lip_mm2 = 1.507964
        f_preload_n = 1.49113
        k_spring_n_mm = 0.095
        f_pin_pull_n = 1.68 if bool(getattr(ctrl, "safety_test_pin", False)) else 0.0
        hud_gain_eff = 0.365
        if self.ext_pino_safety is not None and self.safety_valve_lift_mm > 0.08 and p_gauge_kgf > 0.5:
            pino_phud = float(self.ext_pino_safety[0])
            hb_ref = self.safety_valve_lift_mm / (self.safety_valve_lift_mm + 0.035)
            hud_gain_eff = max(0.33, min(0.41, pino_phud / max(1e-4, p_gauge_kgf * hb_ref)))

        dt_sv = min(0.00020, max(1e-5, dt / 4.0))
        n_sv_steps = max(4, min(24, int(math.ceil(dt / dt_sv))))
        dt_sv_use = dt / n_sv_steps
        y_mm = float(self.safety_valve_lift_mm)
        vy_mm_s = float(self.safety_valve_vel_mm_s)
        p_hud_now = 0.0
        for _ in range(n_sv_steps):
            hb = y_mm / (y_mm + 0.035) if y_mm > 0.0 else 0.0
            p_hud_now = p_gauge_kgf * hb * hud_gain_eff
            f_up_n = 0.0980665 * (p_gauge_kgf * a_seat_mm2 + p_hud_now * da_lip_mm2) + f_pin_pull_n
            f_spr_n = f_preload_n + k_spring_n_mm * y_mm
            acc_mm_s2 = (f_up_n - f_spr_n - 0.00075 * vy_mm_s) / 1.5e-5
            vy_mm_s += acc_mm_s2 * dt_sv_use
            y_mm += vy_mm_s * dt_sv_use
            if y_mm <= 0.0:
                y_mm = 0.0
                if vy_mm_s < 0.0:
                    vy_mm_s = 0.0
            elif y_mm >= 0.56:
                y_mm = 0.56
                if vy_mm_s > 0.0:
                    vy_mm_s = 0.0

        self.safety_valve_lift_mm = y_mm
        self.safety_valve_vel_mm_s = vy_mm_s
        self.safety_huddling_kgf = p_hud_now
        self.safety_valve_open = bool(y_mm > 0.025)

        a_blower = 0.16e-6 * ctrl.blower_valve
        a_whistle = 0.28e-6 * max(0.0, min(1.0, float(getattr(ctrl, "whistle_valve", 0.0))))
        sg_drain_open = 1.0 if bool(getattr(ctrl, "sight_drain_valve", False)) else 0.0
        a_sg_purge = 0.14e-6 * sg_drain_open
        # 2 只纵列 28-6 突跳安全阀总有效泄放喉口面积 A_safety = 2 * min(A_seat, pi * d_seat * y)
        a_single_sv_m2 = min(a_seat_mm2, math.pi * d_seat_sv * y_mm) * 1e-6
        a_safety = 2.0 * a_single_sv_m2
        a_braze_damage_leak = 0.9e-6 * max(0.0, self.copper_anneal_damage - 0.55)
        blowdown_open = max(0.0, min(1.0, float(getattr(ctrl, "blowdown_valve", 0.0))))

        if self.ext_pino_cab is not None:
            # 由 L2 CabManifoldSightGlassPINO 解算 11-15 分汽包压力、通风/汽笛/水面计冲洗/排污质量流率!
            self.turret_pressure_kgf = max(0.0, float(self.ext_pino_cab[0])) if has_steam else 0.0
            mdot_blw = (max(0.0, float(self.ext_pino_cab[2])) * 1e-3) if (has_steam and ctrl.blower_valve > 1e-4) else 0.0
            mdot_whistle = (max(0.0, float(self.ext_pino_cab[3])) * 1e-3) if (has_steam and a_whistle > 0.0) else 0.0
            mdot_blowdown_liq = (max(0.0, float(self.ext_pino_cab[4])) * 1e-3) if (blowdown_open > 0.01 and self.M_water_kg > 2.0e-4) else 0.0
            mdot_sg_purge = (max(0.0, float(self.ext_pino_cab[5])) * 1e-3) if (has_steam and sg_drain_open > 0.5) else 0.0
        else:
            mdot_blw = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_blower) if has_steam else 0.0
            mdot_whistle = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_whistle) if has_steam else 0.0
            mdot_sg_purge = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_sg_purge) if has_steam else 0.0
            if blowdown_open > 0.01 and self.M_water_kg > 2.0e-4:
                dp_bd = max(1500.0, self.P_boiler - P_ATM + 1500.0)
                mdot_blowdown_liq = 0.62 * (0.45e-6 * blowdown_open) * math.sqrt(2.0 * RHO_LIQ_WATER * dp_bd)
            else:
                mdot_blowdown_liq = 0.0
            mdot_turret_gs = (mdot_reg + mdot_blw + mdot_whistle) * 1000.0
            dp_turret = min(0.22 * p_gauge_kgf, 0.018 * (mdot_turret_gs ** 2) / (0.55 + 0.48 * p_gauge_kgf))
            self.turret_pressure_kgf = max(0.0, p_gauge_kgf - dp_turret)

        mdot_sv = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_safety) if has_steam else 0.0
        mdot_braze = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_braze_damage_leak) if has_steam else 0.0

        # --- 5. 烟室负压引风与酒精火管传热 (优先采用 L2 3D PINO 实时三维表面积分通量，次选 3D CFD 查找表) ---
        pino_on = self.ext_pino_burner is not None
        if pino_on and self.ext_pino_burner is not None:
            pb = self.ext_pino_burner
            smokebox_vac_pa = float(pb["smokebox_vac_pa"])
            grate_air_gs = float(pb["grate_air_gs"])
            draft_eff = max(0.05, min(1.15, grate_air_gs / 1.35))
            eta_boil_eff = 0.34 * draft_eff
            sh_delta_c = 28.0 * draft_eff
        elif self.lut is not None:
            exh_gs = mdot_cyl_exh * 1000.0
            blw_gs = mdot_blw * 1000.0
            lut_draft = _bilinear_interp(self.lut["exh_g_s"], self.lut["blw_g_s"], self.lut["draft_eff"], exh_gs, blw_gs)
            lut_eta = _bilinear_interp(self.lut["exh_g_s"], self.lut["blw_g_s"], self.lut["eta_boil"], exh_gs, blw_gs)
            lut_sh_dt = _bilinear_interp(self.lut["exh_g_s"], self.lut["blw_g_s"], self.lut["sh_delta_c"], exh_gs, blw_gs)
            smokebox_vac_pa = _bilinear_interp(self.lut["exh_g_s"], self.lut["blw_g_s"], self.lut["vacuum_pa"], exh_gs, blw_gs)
            if ctrl.chimney_fan:
                draft_eff = min(1.32, max(lut_draft, 0.88))
                eta_boil_eff = max(lut_eta, 0.30)
                sh_delta_c = max(lut_sh_dt, 24.0)
                smokebox_vac_pa = max(smokebox_vac_pa, 95.0)
            else:
                draft_eff = lut_draft
                eta_boil_eff = lut_eta
                sh_delta_c = lut_sh_dt
        else:
            draft_index = (
                (1.00 if ctrl.chimney_fan else 0.0)
                + 18000.0 * mdot_blw
                + 2600.0 * mdot_cyl_exh
            )
            draft_eff = min(1.30, 0.12 + 0.88 * math.tanh(draft_index))
            eta_boil_eff = 0.34 * draft_eff
            sh_delta_c = 28.0 * draft_eff
            smokebox_vac_pa = 140.0 * (draft_eff - 0.12)

        # 耦合 5 根紫铜烟管热弹塑性弓曲/椭圆扁化流固耦合阻尼因子 flue_fsi_mult 与 12-1 炉门 4 孔二次风旁通效应
        fsi_m = float(np.clip(self.flue_fsi_mult, 0.25, 1.05))
        door_open = max(0.0, min(1.0, float(getattr(ctrl, "firebox_door", 0.0))))
        draft_eff = draft_eff * fsi_m * (1.0 - 0.14 * door_open)
        smokebox_vac_pa = max(0.0, smokebox_vac_pa * (0.65 + 0.35 * fsi_m))

        wick_head_factor = max(0.72, min(1.25, 1.0 - 0.34 * math.tan(theta_grade_only)))
        door_chill = 1.0 - (0.06 if pino_on else 0.26) * door_open
        lhv_ethanol = 26.8e6  # 无水/燃料乙醇真实低位发热量 26.8 MJ/kg

        # --- 5b. 多孔硅酸铝陶瓷纤维灯芯 (`セラミックヤーン`) Darcy-Washburn 毛细吸液汽化 (受流道导通度与乙醇黏温特性 mu(T) 物理约束!) ---
        lumen_c = max(0.0, min(1.0, float(self.burner_lumen_cond)))
        sump_h_norm = max(0.0, min(1.8, self.sump_fuel_ml / 3.5))
        supply_gate_1d = lumen_c * (sump_h_norm / (sump_h_norm + 0.08))
        # Darcy-Washburn 毛细上吸速率随升温黏度下降提升，但在接近沸点 (>70°C) 时出现微气阻衰减
        vap_lock_atten = 1.0 - 0.35 * max(0.0, (self.T_alcohol_c - 66.0) / 10.0)
        cap_visc_gain = max(0.55, min(1.42, (0.72 + 0.28 * visc_fluidity_eth) * vap_lock_atten))
        if self.ext_pino_wick_cht is not None:
            wick_cap_raw = max(0.0, 0.57 * float(self.ext_pino_wick_cht[0]) * cap_visc_gain)
            self.wick_saturation = max(0.0, min(1.0, float(self.ext_pino_wick_cht[1])))
            self.wick_tip_temp_c = max(20.0, float(self.ext_pino_wick_cht[2]))
        else:
            flood_mult_1d = 1.0 + 0.55 * max(0.0, sump_h_norm - 1.05)
            evap_drv_1d = (0.52 + 0.48 * draft_eff) if ctrl.burner_lit else (0.008 * max(0.0, self.T_boiler_c - 60.0) / 100.0)
            wick_cap_raw = supply_gate_1d * evap_drv_1d * 0.076 * flood_mult_1d * cap_visc_gain
            sat_tgt = supply_gate_1d * max(0.0, min(0.96, 0.92 - (0.24 * (0.4 + 0.6 * draft_eff) if ctrl.burner_lit else 0.0)))
            self.wick_saturation += min(1.0, 4.0 * dt_th) * (sat_tgt - self.wick_saturation)
            wet_pin_1d = min(1.0, max(0.0, self.wick_saturation / 0.22))
            t_gas_est = (760.0 + 180.0 * draft_eff) if (ctrl.burner_lit and self.wick_saturation > 0.05) else (self.T_crown_c if self.T_crown_c > 120.0 else self.T_boiler_c)
            tip_tgt = 25.0 + wet_pin_1d * (58.8 if ctrl.burner_lit else 0.0) + (1.0 - wet_pin_1d) * 0.65 * max(0.0, t_gas_est - 25.0)
            self.wick_tip_temp_c += min(1.0, 3.5 * dt_th) * (tip_tgt - self.wick_tip_temp_c)

        q_chem_total_w = 0.0
        wick_capillary_ml_s = 0.0
        if ctrl.burner_lit and self.sump_fuel_ml > 0.0 and wick_cap_raw > 1e-7:
            burn_ml_s = wick_cap_raw * wick_head_factor
            used_ml = min(self.sump_fuel_ml, burn_ml_s * dt_th)
            self.sump_fuel_ml -= used_ml
            wick_capillary_ml_s = (used_ml / dt_th) if dt_th > 0 else 0.0
            mdot_alc_kg_s = wick_capillary_ml_s * (rho_ethanol * 1e-6)
            q_chem_total_w = lhv_ethanol * mdot_alc_kg_s
            if pino_on and self.ext_pino_burner is not None:
                # 严格受化学释放总能量上限与 (PINO 三维烟气对流换热积分 + 火室高温气体/灯芯 Stefan-Boltzmann 热辐射并联通量) 共同约束
                q_chem_max = 0.58 * q_chem_total_w
                a_fb_m2 = 8.20e-3  # 65x36x29mm 紫铜火室五面受火内表面积
                t_flame_rad_k = 980.0 + 140.0 * min(1.2, draft_eff)
                q_rad_fb_w = 0.34 * 5.670374e-8 * a_fb_m2 * max(0.0, (t_flame_rad_k ** 4) - (t_boil_k ** 4))
                fuel_scale = min(1.0, wick_capillary_ml_s / 0.055)
                q_conv_3d = float(self.ext_pino_burner["q_boiler_w"]) * fuel_scale * fsi_m
                q_rad_3d = q_rad_fb_w * fuel_scale
                # 5根细火管发展段 Nusselt 近壁热边界层解析修正 + 火室高温燃气/陶瓷芯 Stefan-Boltzmann 热辐射并联换热
                q_comb_w = min(q_chem_max, 0.86 * q_conv_3d + 0.78 * q_rad_3d) * door_chill
            else:
                # 漫油富燃时大量酒精在缺氧下不完全燃烧，热效率下降但耗醇激增
                flood_eff_pen = 0.72 if self.mariotte_flood_active else 1.0
                eta_with_rad = max(0.24, 1.42 * eta_boil_eff)
                q_comb_w = min(0.60 * q_chem_total_w, eta_with_rad * flood_eff_pen * door_chill * q_chem_total_w)
        else:
            q_comb_w = 0.0

        # L1 汽包出口初始湿蒸汽干度 x_dome (随高水位汽空间压缩与剧烈沸腾夹带微小水滴)
        fill_ratio = (self.M_water_kg / 960.0) / max(1e-6, self.V_boiler_total)
        boil_int = min(1.0, max(0.0, q_comb_w / 520.0)) if self.T_boiler_c >= 99.5 else 0.0
        self.steam_dryness_dome = max(
            0.88, min(0.99, 0.985 - 0.055 * max(0.0, (fill_ratio - 0.45) / 0.35) ** 1.5 - 0.025 * boil_int)
        )

        # 17-5 双 U 四程过热器: 采用 IAPWS 统一比焓—压力 (p, h) 方程解算「湿蒸汽液滴蒸干 (x->1.0) -> 单相过热升温」
        if pino_on and self.ext_pino_burner is not None and ctrl.burner_lit:
            q_gas_to_sh_wall = float(self.ext_pino_burner["q_sh_w"])
            t_flue_gas_c = max(self.T_boiler_c, float(self.ext_pino_burner.get("t_flue_upper_c", self.T_boiler_c + 40.0)))
            g_gas_to_sh = 4.2 * (0.25 + 0.75 * draft_eff)
            q_sh_in = 0.55 * q_gas_to_sh_wall + 0.45 * g_gas_to_sh * (t_flue_gas_c - self.T_sh_wall_c)
        else:
            t_flue_gas_c = self.T_boiler_c + (1.32 * sh_delta_c if ctrl.burner_lit else 0.0)
            g_gas_to_sh = 4.2 * (0.25 + 0.75 * draft_eff)
            q_sh_in = g_gas_to_sh * (t_flue_gas_c - self.T_sh_wall_c)

        h_f_b, _, _ = sat_liquid_props(self.T_boiler_c)
        _, h_fg_b, _, _ = sat_vapor_props(self.T_boiler_c)
        h_dome_in = h_f_b + self.steam_dryness_dome * h_fg_b

        if mdot_reg > 5e-7 and has_steam:
            u_sh_in = 2.2 + 4600.0 * mdot_reg
            q_sh_to_steam = u_sh_in * max(0.0, self.T_sh_wall_c - self.T_boiler_c)
            h_sh_out = h_dome_in + q_sh_to_steam / mdot_reg
            ph_out = steam_ph_state(max(P_ATM, self.P_steam_chest), h_sh_out)
            target_sh_c = min(self.T_sh_wall_c, ph_out.temp_c)
            target_x_chest = ph_out.dryness_x
        else:
            q_sh_to_steam = 0.25 * (self.T_sh_wall_c - self.T_boiler_c)
            target_sh_c = 0.35 * self.T_boiler_c + 0.65 * self.T_sh_wall_c
            target_x_chest = 1.0 if target_sh_c > sat_temp_c(max(P_ATM, self.P_steam_chest)) + 0.5 else self.steam_dryness_dome

        dT_sh_wall = (q_sh_in - q_sh_to_steam) / 7.0
        self.T_sh_wall_c = max(25.0, min(360.0, self.T_sh_wall_c + dT_sh_wall * dt_th))
        self.T_superheat_c += min(1.0, 2.5 * dt_th) * (target_sh_c - self.T_superheat_c)
        self.steam_dryness_chest += min(1.0, 4.0 * dt_th) * (target_x_chest - self.steam_dryness_chest)

        # --- 6. 6-5 中央轴动给水泵真实可压缩液腔 (抽吸真空 + 双 SB-ø3 止回球 + 4-17/6-9 旁通针阀分流 + 13-2 进锅逆止阀 + 曲轴机械轴功) ---
        water_fill_open = max(0.0, min(1.0, float(getattr(ctrl, "water_fill_valve", 0.0))))
        water_fill_ml_s = 18.0 * water_fill_open
        if water_fill_ml_s > 1e-5:
            # 外部向 35-1 后部敞口水槽倾注 25°C 常温清水: 质量守恒 + 焓混降温
            fill_ml = water_fill_ml_s * dt_th
            w_prev = max(1e-3, self.tender_water_ml)
            self.T_feedwater_c = (w_prev * self.T_feedwater_c + fill_ml * 25.0) / (w_prev + fill_ml)
            self.tender_water_ml += fill_ml

        rho_fw = 999.0 - 0.16 * max(0.0, self.T_feedwater_c - 20.0)
        p_vap_fw = sat_pressure_pa(max(15.0, min(98.0, self.T_feedwater_c)))
        h_tdr_m = 0.038 * max(0.0, min(1.0, self.tender_water_ml / self.tender_water_cap_ml))
        # 煤水车 34-6 出水口至机车 6-5 泵体 (ΔX=+236mm) 静水重力压头 (含坡道俯仰!)
        p_suction_supply = P_ATM + rho_fw * 9.81 * (h_tdr_m - 0.236 * math.sin(theta_grade_only))
        p_return_standpipe = P_ATM + rho_fw * 9.81 * 0.042
        p_clack_boiler = self.P_boiler + rho_fw * 9.81 * 0.033 + 4500.0
        a_byp_orf = 5.2e-6 * (max(0.0, min(1.0, ctrl.bypass_valve)) ** 1.85) + 2.5e-10
        a_clack_max = 1.15e-6  # 13-2 锅炉中位逆止阀 SB-ø3 球座最大通流面积

        # 求解 6-5 泵腔 8 子步弹性液柱容积动力学 (含 BT-ø5 氟橡胶软管柔度 K_eff / V_ch):
        #   - 回程吸液 (v_pump_ram > 0): 腔压跌破大气压形成真实抽吸负压真空，推开下部 SB-ø3 止回球从煤水车吸液，同时在柱塞上产生向前的真空拉拽阻力!
        #   - 前推压水 (v_pump_ram < 0): 腔压飙升推开上部 SB-ø3 止回球进入 6-6->4-17 总管，按旁通阀流阻与锅炉表压竞争分流，同时对第3动轮偏心轮施加真实液压反力!
        n_pmp = max(8, min(320, int(math.ceil(dt / 2.5e-5))))
        dt_pmp = dt / n_pmp
        k_over_v = 2.4e14  # Pa/m³ 6-5 黄铜泵腔液柱体积弹性刚度 (~0.24mm 冲程即可建压至 3.5 kgf/cm²)
        q_suc_m3_s_acc = 0.0
        q_del_m3_s_acc = 0.0
        q_byp_m3_s_acc = 0.0
        q_boil_m3_s_acc = 0.0
        lift_suc_mm = 0.0
        lift_del_mm = 0.0
        lift_clack_mm = 0.0
        has_prime = self.tender_water_ml > 0.25

        # 手摇泵 31-5 注入共用总管的体积流率 (m³/s)
        q_hand_m3_s = 0.0
        if v_hand_ram < -1e-4 and has_prime:
            q_hand_m3_s += 0.90 * self.A_hand_pump * (-v_hand_ram)
        if self._hand_pump_pending_kg > 1e-7:
            inj_rate_kg_s = min(self._hand_pump_pending_kg / max(1e-6, dt_th), 5.6e-3)
            q_hand_m3_s += inj_rate_kg_s / rho_fw

        for _ in range(n_pmp):
            dv_ram_dt = self.A_axle_pump * v_pump_ram  # >0 吸程膨胀, <0 压程压缩
            if has_prime and p_suction_supply > self.P_pump_chamber + 2200.0:
                dp_s = p_suction_supply - self.P_pump_chamber - 2200.0
                lift_suc_mm = min(0.65, 0.065 * (dp_s / 1000.0))
                a_s = min(9.5e-7, math.pi * 0.0022 * (lift_suc_mm * 1e-3))
                q_in = 0.64 * a_s * math.sqrt(2.0 * dp_s / rho_fw)
            else:
                lift_suc_mm = 0.0
                q_in = 0.0

            if has_prime and self.P_pump_chamber > self.P_fw_manifold + 1600.0:
                dp_d = self.P_pump_chamber - self.P_fw_manifold - 1600.0
                lift_del_mm = min(0.65, 0.055 * (dp_d / 1000.0))
                a_d = min(9.5e-7, math.pi * 0.0022 * (lift_del_mm * 1e-3))
                q_out = 0.64 * a_d * math.sqrt(2.0 * dp_d / rho_fw)
            else:
                lift_del_mm = 0.0
                q_out = 0.0

            k_eff = k_over_v if has_prime else 2.8e11  # 缺水进气空转时可压缩空气刚度骤降无法建压!
            self.P_pump_chamber = float(np.clip(
                self.P_pump_chamber + k_eff * (q_in - q_out - dv_ram_dt) * dt_pmp,
                p_vap_fw,
                max(self.P_boiler + 3.5e5, P_ATM + 6.0e5),
            ))

            # 4-17 旁通阀总管容积节点压力平衡: Q_out + Q_hand = Q_bypass(P_man) + Q_boiler_clack(P_man)
            dp_byp = max(0.0, self.P_fw_manifold - p_return_standpipe)
            q_byp_sub = 0.65 * a_byp_orf * math.sqrt(2.0 * dp_byp / rho_fw)
            if self.P_fw_manifold > p_clack_boiler:
                dp_clk = self.P_fw_manifold - p_clack_boiler
                lift_clack_mm = min(0.68, 0.048 * (dp_clk / 1000.0))
                a_clk = min(a_clack_max, math.pi * 0.0024 * (lift_clack_mm * 1e-3))
                q_boil_sub = 0.64 * a_clk * math.sqrt(2.0 * dp_clk / rho_fw)
            else:
                lift_clack_mm = 0.0
                q_boil_sub = 0.0

            self.P_fw_manifold = float(np.clip(
                self.P_fw_manifold + 3.4e13 * (q_out + q_hand_m3_s - q_byp_sub - q_boil_sub) * dt_pmp,
                P_ATM,
                max(self.P_boiler + 3.2e5, P_ATM + 5.5e5),
            ))
            q_suc_m3_s_acc += q_in / n_pmp
            q_del_m3_s_acc += (q_out + q_hand_m3_s) / n_pmp
            q_byp_m3_s_acc += q_byp_sub / n_pmp
            q_boil_m3_s_acc += q_boil_sub / n_pmp

        self.pump_ball_lifts_mm = [round(lift_suc_mm, 3), round(lift_del_mm, 3), round(lift_clack_mm, 3)]
        self.pump_suction_vac_kpa = max(0.0, (P_ATM - self.P_pump_chamber) * 1e-3)

        # 柱塞真实双向流体压力载荷 + PN3.2-1 O 型圈摩擦:
        #   - 压水冲程 (v_pump_ram < 0, P_ch > P_atm): F_pump > 0 (+X 顶推抗拒前压)
        #   - 吸水冲程 (v_pump_ram > 0, P_ch < P_atm): F_pump < 0 (-X 真空抽拉抗拒后退)
        f_pump_seal = 0.085 * math.tanh(v_pump_ram / 0.006) + 0.35 * v_pump_ram
        F_pump = (self.P_pump_chamber - P_ATM) * self.A_axle_pump - f_pump_seal
        self.pump_shaft_power_w = max(0.0, -F_pump * v_pump_ram)

        # 严格采用 8 子步积分所得真实旁通回流与 13-2 逆止阀进锅体积流量 (杜绝单步末瞬时压力阈值丢流!)
        q_byp_1d_mls = q_byp_m3_s_acc * 1e6
        q_boil_1d_mls = q_boil_m3_s_acc * 1e6

        if self.ext_pino_tender is not None:
            clack_open_gate = 1.0 if self.P_fw_manifold > p_clack_boiler else 0.0
            q_boil_pino = clack_open_gate * max(0.0, float(self.ext_pino_tender[0]))
            q_ret_pino = max(0.0, float(self.ext_pino_tender[1]))
            q_drain_mls = max(0.0, float(self.ext_pino_tender[2])) if bool(getattr(ctrl, "tender_drain_open", False)) else 0.0
            q_boil_mls = 0.55 * q_boil_pino + 0.45 * q_boil_1d_mls
            q_ret_mls = 0.55 * q_ret_pino + 0.45 * q_byp_1d_mls

            net_out_ml = (q_boil_mls + q_drain_mls) * dt_th
            if net_out_ml > self.tender_water_ml and net_out_ml > 1e-9:
                scale_avail = self.tender_water_ml / net_out_ml
                q_boil_mls *= scale_avail
                q_drain_mls *= scale_avail
                net_out_ml = self.tender_water_ml
            self.tender_water_ml = max(0.0, self.tender_water_ml - net_out_ml)
            if self._hand_pump_pino_ml > 0.0:
                self._hand_pump_pino_ml = max(0.0, self._hand_pump_pino_ml - (q_boil_mls + q_ret_mls) * dt_th)
            if self._hand_pump_pending_kg > 0.0:
                self._hand_pump_pending_kg = max(0.0, self._hand_pump_pending_kg - (q_boil_mls + q_ret_mls) * 1e-3 * dt_th)

            return_water_ml_s = q_ret_mls if self.tender_water_ml > 0.2 else 0.0
            tender_drain_ml_s = q_drain_mls
            mdot_fw_total = q_boil_mls * (rho_fw * 1e-6)
        else:
            if self._hand_pump_pending_kg > 1e-7:
                used_hp_kg = min(self._hand_pump_pending_kg, q_hand_m3_s * rho_fw * dt_th)
                self._hand_pump_pending_kg = max(0.0, self._hand_pump_pending_kg - used_hp_kg)
                # trigger_hand_pump 已从水箱预扣体积，故旁通回流部分归还煤水车水箱
                hp_frac = (q_hand_m3_s * 1e6) / max(1e-9, q_del_m3_s_acc * 1e6) if q_del_m3_s_acc > 1e-12 else 0.0
                self.tender_water_ml += (q_byp_1d_mls * hp_frac) * dt_th

            axle_boil_ml = max(0.0, (q_boil_1d_mls - q_hand_m3_s * 1e6) * dt_th)
            if axle_boil_ml > self.tender_water_ml:
                scale_ax = self.tender_water_ml / max(1e-9, axle_boil_ml)
                q_boil_1d_mls *= scale_ax
                axle_boil_ml = self.tender_water_ml
            self.tender_water_ml = max(0.0, self.tender_water_ml - axle_boil_ml)

            if bool(getattr(ctrl, "tender_drain_open", False)) and self.tender_water_ml > 0.05:
                w_norm_1d = max(0.0, min(1.0, self.tender_water_ml / self.tender_water_cap_ml))
                tender_drain_ml_s = 4.0 * math.sqrt(w_norm_1d)
                drn_ml = min(self.tender_water_ml, tender_drain_ml_s * dt_th)
                self.tender_water_ml -= drn_ml
                tender_drain_ml_s = drn_ml / dt_th if dt_th > 0 else 0.0
            else:
                tender_drain_ml_s = 0.0

            return_water_ml_s = q_byp_1d_mls if has_prime else 0.0
            mdot_fw_total = q_boil_1d_mls * (rho_fw * 1e-6)

        # 超过 31-10 水箱满容积 340.0 ml 时从 35-1 敞口挡水围板自然溢流
        if self.tender_water_ml > self.tender_water_cap_ml:
            overflow_ml = self.tender_water_ml - self.tender_water_cap_ml
            self.tender_water_ml = self.tender_water_cap_ml
            tender_overflow_ml_s = overflow_ml / dt_th if dt_th > 0 else 0.0
        else:
            tender_overflow_ml_s = 0.0

        if return_water_ml_s > 1e-3 and self.tender_water_ml > 5.0:
            # 经机车车架内热态 6-5/4-17 黄铜阀体回流至 31-14 高立管的暖水微幅预热煤水车水箱
            t_ret_c = min(68.0, self.T_feedwater_c + 0.06 * max(0.0, self.T_boiler_c - self.T_feedwater_c))
            mix_frac = min(0.05, (return_water_ml_s * dt_th) / max(10.0, self.tender_water_ml))
            self.T_feedwater_c = min(55.0, self.T_feedwater_c + mix_frac * (t_ret_c - self.T_feedwater_c))
        else:
            self.T_feedwater_c += (25.0 - self.T_feedwater_c) * 0.0005 * dt_th

        # --- 7. O4-1/O4-2: 坡道假水位、分层紫铜火室顶板传热、空焚き退火与冷水闪蒸尖峰 ---
        theta_eff = pitch_rad + math.radians(ctrl.grade_deg) + math.atan2(accel_x, 9.81)
        theta_eff = max(-0.30, min(0.30, theta_eff))
        tan_th = math.tan(theta_eff)

        water_ml = self.M_water_kg * 1000.0
        z_free_static_mm = water_volume_to_free_z_mm(water_ml)
        true_level_pct = 100.0 * (water_ml - self.water_min_ml) / (self.water_max_ml - self.water_min_ml)
        dh_gauge_mm = +68.0 * tan_th
        apparent_level_pct = true_level_pct + (dh_gauge_mm / 15.5) * 100.0

        h_crown_static_mm = z_free_static_mm - 47.5
        dh_crown_worst_mm = -24.0 * abs(tan_th)
        crown_submerged_mm = h_crown_static_mm + dh_crown_worst_mm
        f_wet = max(0.0, min(1.0, crown_submerged_mm / 4.0))
        f_wet_tubes = max(0.0, min(1.0, (z_free_static_mm - 18.2) / 26.5))

        q_fire_crown = 0.58 * q_comb_w
        q_fire_tubes = 0.42 * q_comb_w
        g_dry_cu_cond = 0.68 * (min(1.0, max(0.0, water_ml) / 18.0) ** 0.67)
        g_crown_to_water = 22.0 * f_wet + g_dry_cu_cond * (1.0 - f_wet)
        q_crown_to_water = g_crown_to_water * (self.T_crown_c - self.T_boiler_c)
        q_waterleg_boil = 0.28 * q_fire_crown * (1.0 - f_wet) * f_wet_tubes
        q_crown_rad_loss = 0.08 * max(0.0, self.T_crown_c - 25.0)

        # 高温干烧状态下突然打入冷水 -> 灼热顶板表面急剧膜态闪蒸直接产生气相超压尖峰 (Flash Steam Spike)
        flash_spike = False
        q_flash_w = 0.0
        mdot_flash_vap = 0.0
        if self.T_crown_c > 215.0 and f_wet < 0.45 and mdot_fw_total > 1.5e-4:
            flash_spike = True
            q_flash_w = min(
                2800.0,
                26.0 * (self.T_crown_c - self.T_boiler_c) * min(4.0, mdot_fw_total / 0.0012),
            )
            mdot_flash_vap = q_flash_w / latent_heat_j_kg(self.T_boiler_c)
            v_vap_now = max(4e-5, self.V_boiler_total - self.M_water_kg / 960.0)
            dp_flash = (R_STEAM * (self.T_boiler_c + 273.15) / v_vap_now) * (mdot_flash_vap * dt_th)
            self._flash_overpressure_pa = min(1.8 * PA_PER_KGF_CM2, self._flash_overpressure_pa + dp_flash)
        else:
            self._flash_overpressure_pa *= math.exp(-1.2 * dt_th)

        c_crown = self.M_copper_crown * CP_COPPER
        dT_crown = (q_fire_crown - q_waterleg_boil - q_crown_to_water - q_crown_rad_loss - q_flash_w) / c_crown
        self.T_crown_c = max(25.0, min(780.0, self.T_crown_c + dT_crown * dt_th))

        if self.T_crown_c > 340.0:
            dmg_rate = 0.0018 * ((self.T_crown_c - 340.0) / 80.0) ** 1.4
            self.copper_anneal_damage = min(1.0, self.copper_anneal_damage + dmg_rate * dt_th)

        # --- 8. 多层复合壁散热与液态水 M_water_kg + 顶空蒸汽 M_vap_kg 连续统一守恒 (0ml 干锅严格零蒸发、零分支跳变!) ---
        v_air_ext = abs(omega_3) * (0.5 * 0.0438)
        if self.ext_pino_wick_cht is not None:
            self.casing_temp_c = max(20.0, float(self.ext_pino_wick_cht[3]))
            q_loss_w = max(0.0, float(self.ext_pino_wick_cht[4]))
            q_rad_shell_w = max(0.0, float(self.ext_pino_wick_cht[6]))
        else:
            a_ext_m2 = 0.0365
            ins_cov = max(0.0, min(1.0, float(self.insulation_coverage)))
            k_cer = 0.068 + 1.4e-4 * max(0.0, 0.5 * (self.T_boiler_c + 25.0) - 25.0)
            wet_f_shell = max(0.15, 0.32 + 0.68 * f_wet_tubes)
            r_ins_z = ((0.0020 / k_cer) + 0.088) / wet_f_shell
            r_bare_z = 0.0048 / wet_f_shell

            def _zone_loss_1d(r_z: float, init_f: float) -> Tuple[float, float, float]:
                tz = 25.0 + init_f * max(0.0, self.T_boiler_c - 25.0)
                for _ in range(6):
                    dte = max(0.0, tz - 25.0)
                    h_c = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * math.sqrt(v_air_ext + 0.04)
                    tk = tz + 273.15
                    q_r = 0.88 * 5.670374e-8 * max(0.0, (tk ** 4) - (298.15 ** 4))
                    q_i = max(0.0, self.T_boiler_c - tz) / r_z
                    dq = -(1.0 / r_z) - h_c - 4.0 * 0.88 * 5.670374e-8 * (tk ** 3)
                    tz = min(self.T_boiler_c, max(25.0, tz - (q_i - (h_c * dte + q_r)) / min(-1e-3, dq)))
                dte = max(0.0, tz - 25.0)
                h_c = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * math.sqrt(v_air_ext + 0.04)
                q_r = 0.88 * 5.670374e-8 * max(0.0, ((tz + 273.15) ** 4) - (298.15 ** 4))
                return tz, h_c * dte, q_r

            t_zi, qcv_i, qrd_i = _zone_loss_1d(r_ins_z, 0.32)
            t_zb, qcv_b, qrd_b = _zone_loss_1d(r_bare_z, 0.88)
            self.casing_temp_c = ins_cov * t_zi + (1.0 - ins_cov) * t_zb
            q_rad_shell_w = a_ext_m2 * (ins_cov * qrd_i + (1.0 - ins_cov) * qrd_b)
            q_loss_w = a_ext_m2 * (ins_cov * qcv_i + (1.0 - ins_cov) * qcv_b) + q_rad_shell_w

        q_net_water_w = (q_fire_tubes * (0.16 + 0.84 * f_wet_tubes)) + q_waterleg_boil + q_crown_to_water + q_flash_w - q_loss_w
        if q_chem_total_w > 5.0 and ctrl.burner_lit:
            q_useful_tot = max(
                0.0,
                q_comb_w * (0.58 + 0.42 * (0.16 + 0.84 * f_wet_tubes))
                + max(0.0, q_sh_in)
                - q_loss_w
                - q_crown_rad_loss,
            )
            boiler_eff_pct = max(0.0, min(92.0, 100.0 * q_useful_tot / q_chem_total_w))
        else:
            boiler_eff_pct = 0.0
        self.boiler_efficiency_pct = boiler_eff_pct

        mdot_steam_out = mdot_reg + mdot_blw + mdot_whistle + mdot_sg_purge + mdot_sv + mdot_braze
        h_fg = latent_heat_j_kg(self.T_boiler_c)
        v_gas = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)

        # 连续相变与气相质量积分 (彻底消除 M_water_kg == 0.1ml 处的硬分支跳变与 0ml 虚假蒸发读数!):
        p_s1 = sat_pressure_pa(self.T_boiler_c)
        p_s2 = sat_pressure_pa(self.T_boiler_c + 0.1)
        drho_dT = ((p_s2 / (R_STEAM * (t_boil_k + 0.1))) - (p_s1 / (R_STEAM * t_boil_k))) / 0.1
        boil_gate = min(1.0, max(0.0, (self.T_boiler_c - 99.5) / 0.7))
        wet_phase_gate = min(1.0, self.M_water_kg / 0.0012)
        phase_couple = boil_gate * wet_phase_gate

        c_eff = self.M_water_kg * CP_WATER + self.M_copper_shell * CP_COPPER + phase_couple * v_gas * h_fg * drho_dT
        dT_dt = (
            q_net_water_w
            - phase_couple * mdot_steam_out * h_fg
            - mdot_fw_total * CP_WATER * (self.T_boiler_c - self.T_feedwater_c)
        ) / max(80.0, c_eff)
        self.T_boiler_c = max(25.0, min(215.0, self.T_boiler_c + dT_dt * dt_th))

        max_liq_avail_rate = max(0.0, self.M_water_kg / max(1e-9, dt_th) + mdot_fw_total - mdot_blowdown_liq)
        if self.M_water_kg > 1.0e-7 and (boil_gate > 0.01 or flash_spike):
            rho_sat_now = sat_pressure_pa(self.T_boiler_c) / (R_STEAM * (self.T_boiler_c + 273.15))
            vap_deficit_rate = max(0.0, (rho_sat_now * v_gas - self.M_vap_kg) / 0.18)
            q_boil_direct = boil_gate * max(0.0, q_net_water_w) / max(1e5, h_fg)
            evap_eq = max(
                wet_phase_gate * max(0.0, mdot_steam_out + v_gas * drho_dT * dT_dt + vap_deficit_rate),
                q_boil_direct,
            ) + mdot_flash_vap
            evap_rate_kg_s = min(max_liq_avail_rate, evap_eq)
        else:
            evap_rate_kg_s = 0.0

        self.M_water_kg = float(np.clip(
            self.M_water_kg + (mdot_fw_total - mdot_blowdown_liq - evap_rate_kg_s) * dt_th,
            0.0, 0.350
        ))
        if self.M_water_kg <= 1.0e-7:
            self.M_water_kg = 0.0
            evap_rate_kg_s = 0.0

        v_gas_new = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)
        if self.M_water_kg > 5.0e-4:
            self.T_vap_c = self.T_boiler_c
        else:
            t_dry_eq = 0.65 * self.T_boiler_c + 0.35 * min(320.0, self.T_crown_c)
            self.T_vap_c += min(1.0, 1.8 * dt_th) * (t_dry_eq - self.T_vap_c)

        t_vap_k = max(295.0, self.T_vap_c + 273.15)
        m_vap_atm = (P_ATM * v_gas_new) / (R_STEAM * t_vap_k)
        self.M_vap_kg = max(m_vap_atm, self.M_vap_kg + (evap_rate_kg_s - mdot_steam_out) * dt_th)

        p_sat_now = max(P_ATM, sat_pressure_pa(self.T_boiler_c))
        if phase_couple > 0.02:
            m_vap_sat = (p_sat_now * v_gas_new) / (R_STEAM * (self.T_boiler_c + 273.15))
            self.M_vap_kg += min(1.0, 12.0 * phase_couple * dt_th) * (m_vap_sat - self.M_vap_kg)

        p_gas_real = (self.M_vap_kg * R_STEAM * t_vap_k) / v_gas_new
        self.P_boiler = max(
            P_ATM,
            phase_couple * p_sat_now + (1.0 - phase_couple) * p_gas_real + self._flash_overpressure_pa
        )

        self.water_free_z_mm = water_volume_to_free_z_mm(self.M_water_kg * 1000.0)

        # --- 9. 13-6/11-7/13-10 真实二阶连通器流体力学微分方程、波登管阻尼与物理仪表防跳变平滑 ---
        p_gauge_now = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        p_sc_now = max(0.0, (self.P_steam_chest - P_ATM) / PA_PER_KGF_CM2)
        app_clamped = max(0.0, min(100.0, apparent_level_pct))

        # 连通器双侧物理物性：锅筒沸腾气液混合密度 vs 舱外自然对流微冷 Pyrex 玻璃管液柱密度 + ID 2.30mm 毛细弯月面升力
        z_port_bot = 32.6  # 13-10 水平下水管轴线 Z=+32.6mm (穿入湿锅筒喉部 X=-43.2mm)
        z_boil_at_sg = max(18.2, z_free_static_mm + dh_gauge_mm)
        raw_evap_now_gs = evap_rate_kg_s * 1000.0
        void_alpha = min(0.042, 0.016 * raw_evap_now_gs)
        rho_boil = max(840.0, 1000.0 - 0.085 * (self.T_boiler_c - 20.0) - 0.0028 * ((self.T_boiler_c - 20.0) ** 2)) * (1.0 - void_alpha)
        t_sg_c = 25.0 + 0.68 * max(0.0, self.T_boiler_c - 25.0)
        rho_sg = max(860.0, 1000.0 - 0.085 * (t_sg_c - 20.0) - 0.0028 * ((t_sg_c - 20.0) ** 2))
        sigma_sg = max(0.042, 0.0728 - 0.000165 * max(0.0, t_sg_c - 20.0))
        dp_cap = (2.0 * sigma_sg * 0.208) / 1.15e-3  # 热态硼硅玻璃真实接触角 (θ≈78°) 毛细升压 (~18..24 Pa ≈ +1.9mm, 恰与管外温降增密平衡!)

        g_top = max(0.0, min(1.0, float(self.sg_steam_lumen_cond)))
        g_bot = max(0.0, min(1.0, float(self.sg_water_lumen_cond)))
        mu_sg = 2.414e-5 * (10.0 ** (247.8 / (t_sg_c + 133.15)))
        # ID 1.56mm, L=86mm 13-10 水平下水管 Hagen-Poiseuille 黏性流阻折算至 ID 2.30mm 玻璃液柱速度 + 弯头局部阻尼
        r_poiseuille = (32.0 * mu_sg * 0.086 / ((1.56e-3) ** 2)) * 2.174
        r_lin = (r_poiseuille + 2150.0) / max(0.04, g_bot)
        dt_sg = dt_th / 4.0
        for _ in range(4):
            if g_top > 1.0e-4:
                p_sg_eq = P_ATM + (self.P_boiler - P_ATM) * (g_top / (g_top + 0.08 * sg_drain_open))
                # 13-6 (ID 1.56mm, L=25.5mm) 蒸汽导入 0.06ml 玻璃顶腔的真实 Poiseuille 气相均压速率 (~3200 s^-1, 严防升温 dP/dt 滞后推翻 9.4Pa/mm 静水压头!)
                self.P_sg_top += min(1.0, 3200.0 * g_top * dt_sg) * (p_sg_eq - self.P_sg_top)
                self._sg_trapped_pa_mm = self.P_sg_top * max(0.8, 61.5 - self.sg_meniscus_z_mm)
            else:
                # 若上蒸汽管 13-6 三维流道堵塞 (g_top == 0)，11-7 盲顶封闭气体等温压缩背压阻止液面自由升降!
                self.P_sg_top = self._sg_trapped_pa_mm / max(0.8, 61.5 - self.sg_meniscus_z_mm)

            if g_bot <= 1.0e-4:
                self.sg_vel_m_s = 0.0
            else:
                p_boil_port = self.P_boiler + rho_boil * 9.81 * max(0.0, z_boil_at_sg - z_port_bot) * 1e-3
                dp_drain_vent = sg_drain_open * (
                    0.82 * max(0.0, self.P_boiler - P_ATM)
                    + rho_boil * 9.81 * max(0.0, z_boil_at_sg - 34.5) * 1e-3
                )
                p_sg_col_base = self.P_sg_top + rho_sg * 9.81 * max(0.0, self.sg_meniscus_z_mm - z_port_bot) * 1e-3 - dp_cap
                dp_drive = (p_boil_port - dp_drain_vent) - p_sg_col_base
                accel_sg = (dp_drive - r_lin * self.sg_vel_m_s - 2600.0 * abs(self.sg_vel_m_s) * self.sg_vel_m_s) / (0.21 * rho_sg)
                self.sg_vel_m_s = float(np.clip(self.sg_vel_m_s + accel_sg * dt_sg, -1.5, 1.5))
                new_z = self.sg_meniscus_z_mm + self.sg_vel_m_s * 1000.0 * dt_sg
                if new_z <= z_port_bot:
                    new_z = z_port_bot
                    if self.sg_vel_m_s < 0.0:
                        self.sg_vel_m_s = 0.0
                elif new_z >= 63.2:
                    new_z = 63.2
                    if self.sg_vel_m_s > 0.0:
                        self.sg_vel_m_s = 0.0
                self.sg_meniscus_z_mm = float(new_z)

        self.sight_glass_level_pct = free_z_mm_to_sight_pct(self.sg_meniscus_z_mm)
        if self.ext_pino_cab is not None:
            pg_target = max(0.0, float(self.ext_pino_cab[7]))
            self.pg_bourdon_kgf += min(1.0, 12.0 * dt_th) * (pg_target - self.pg_bourdon_kgf)
        else:
            self.pg_bourdon_kgf += min(1.0, 10.0 * dt_th) * (p_gauge_now - self.pg_bourdon_kgf)

        if has_steam and p_gauge_now > 0.08:
            self.siphon_seal_water_pct = min(100.0, self.siphon_seal_water_pct + 0.65 * dt_th)
        else:
            self.siphon_seal_water_pct = max(86.0, self.siphon_seal_water_pct - 0.02 * dt_th)

        # 物理仪表盘阻尼平滑 (tau ≈ 0.16s 滤除四冲排气脉冲引起的高频数字跳字，同时保留真实加减档瞬态，首帧非零直接对齐避免冷启动滞后!)
        alpha_inst = min(1.0, dt / 0.16)
        raw_evap_gs = evap_rate_kg_s * 1000.0
        if self.M_water_kg <= 1.0e-7:
            self._ema_evap_gs = 0.0
        elif self._ema_evap_gs < 1e-4 and raw_evap_gs > 1e-3:
            self._ema_evap_gs = raw_evap_gs
        else:
            self._ema_evap_gs += alpha_inst * (raw_evap_gs - self._ema_evap_gs)
        self._ema_exhaust_gs += alpha_inst * (mdot_cyl_exh * 1000.0 - self._ema_exhaust_gs)
        q_net_pos = max(0.0, q_net_water_w)
        if self._ema_heat_w < 1e-3 and q_net_pos > 1.0:
            self._ema_heat_w = q_net_pos
        else:
            self._ema_heat_w += alpha_inst * (q_net_pos - self._ema_heat_w)
        if self._ema_eta_pct < 1e-3 and boiler_eff_pct > 0.5:
            self._ema_eta_pct = boiler_eff_pct
        else:
            self._ema_eta_pct += alpha_inst * (boiler_eff_pct - self._ema_eta_pct)
        if self._ema_vac_pa < 1e-3 and smokebox_vac_pa > 0.5:
            self._ema_vac_pa = smokebox_vac_pa
        else:
            self._ema_vac_pa += min(1.0, dt / 0.004) * (smokebox_vac_pa - self._ema_vac_pa)
        self._ema_ind_work_w += min(1.0, dt / 0.22) * (max(0.0, ind_work_w_acc) - self._ema_ind_work_w)
        if self._ema_ind_work_w > 0.08 and mdot_cyl_in > 1e-6:
            inst_spec = min(999.0, (mdot_cyl_in * 1000.0) / (self._ema_ind_work_w * 1e-3))
            self._ema_spec_steam += min(1.0, dt / 0.30) * (inst_spec - self._ema_spec_steam)
        elif mdot_cyl_in <= 1e-6:
            self._ema_spec_steam *= math.exp(-2.0 * dt)

        rpm = (omega_3 / (2.0 * math.pi)) * 60.0
        v_real_ms = abs(omega_3) * (0.5 * 0.0438) * 32.0
        dry_alarm = (crown_submerged_mm <= 0.5 and ctrl.burner_lit) or (self.T_crown_c >= 240.0)
        boil_prog = min(100.0, max(0.0, (self.T_boiler_c - 25.0) / 75.0 * 100.0))

        el_info = self.ext_pino_elastic or {}
        flue_defl = list(el_info.get("flue_deflection_mm", [0.0] * 5))
        flue_oval = list(el_info.get("flue_ovalization_mm", [0.0] * 5))
        crown_blg = float(el_info.get("crown_bulge_mm", 0.0))
        max_stress = float(el_info.get("max_boiler_stress_mpa", 0.0))
        cu_yield = float(el_info.get("copper_yield_mpa", 185.0))
        spr_defl = list(el_info.get("spring_deflect_mm", [0.0] * 10))

        telemetry = ThermoTelemetry(
            sim_time=sim_time,
            boiler_temp_c=self.T_boiler_c,
            boiler_pressure_kgf=p_gauge_now,
            steam_chest_kgf=p_sc_now,
            superheat_temp_c=self.T_superheat_c,
            water_level_pct=app_clamped,
            true_water_level_pct=max(0.0, min(100.0, true_level_pct)),
            crown_temp_c=self.T_crown_c,
            crown_submerged_mm=crown_submerged_mm,
            copper_anneal_damage=self.copper_anneal_damage,
            seal_health=self.seal_health,
            alcohol_ml=self.alcohol_ml + self.sump_fuel_ml,
            tender_water_ml=self.tender_water_ml,
            lubricator_oil_pct=self.lubricator_oil_pct,
            cyl_wall_temp_c=self.T_cyl_wall_c,
            condensate_g=self.condensate_kg * 1000.0,
            safety_valve_open=self.safety_valve_open,
            dry_fire_alarm=dry_alarm,
            emergency_overpressure=(p_gauge_now >= 5.0),
            flash_steam_spike=flash_spike,
            hydrolock_L=hydrolock_L,
            hydrolock_R=hydrolock_R,
            valve_friction_L=F_vfric_L,
            valve_friction_R=F_vfric_R,
            smokebox_vacuum_pa=self._ema_vac_pa,
            lut_active=(self.lut is not None or pino_on),
            pino_active=pino_on,
            steam_dryness_dome=self.steam_dryness_dome,
            steam_dryness_chest=self.steam_dryness_chest,
            heat_input_w=self._ema_heat_w,
            feedwater_g_s=mdot_fw_total * 1000.0,
            return_water_ml_s=return_water_ml_s,
            tender_water_temp_c=self.T_feedwater_c,
            boiling_progress_pct=boil_prog,
            mariotte_bubbling=mariotte_bubbling,
            mariotte_flood_alarm=self.mariotte_flood_active,
            mariotte_headspace_kpa=self.mariotte_headspace_kpa,
            sight_glass_level_pct=self.sight_glass_level_pct,
            pg_bourdon_kgf=self.pg_bourdon_kgf,
            turret_pressure_kgf=self.turret_pressure_kgf,
            water_fill_ml_s=water_fill_ml_s,
            tender_drain_ml_s=tender_drain_ml_s,
            tender_overflow_ml_s=tender_overflow_ml_s,
            blowdown_g_s=mdot_blowdown_liq * 1000.0,
            sight_purge_g_s=mdot_sg_purge * 1000.0,
            whistle_steam_g_s=mdot_whistle * 1000.0,
            exhaust_g_s=self._ema_exhaust_gs,
            blower_g_s=mdot_blw * 1000.0,
            safety_g_s=mdot_sv * 1000.0,
            safety_valve_lift_mm=round(self.safety_valve_lift_mm, 4),
            safety_huddling_kgf=round(self.safety_huddling_kgf, 3),
            siphon_seal_water_pct=round(self.siphon_seal_water_pct, 2),
            roscoe_quill_oil_rate_mg_min=self.roscoe_quill_oil_rate_mg_min,
            pino_5op_active=(self.ext_pino_safety is not None),
            pino_6op_active=(self.ext_pino_wick_cht is not None),
            burner_lumen_connected=(self.burner_lumen_cond > 0.15),
            wick_capillary_ml_s=round(wick_capillary_ml_s, 4),
            wick_saturation=round(self.wick_saturation, 3),
            wick_tip_temp_c=round(self.wick_tip_temp_c, 1),
            casing_temp_c=round(self.casing_temp_c, 1),
            shell_heat_loss_w=round(q_loss_w, 2),
            shell_rad_loss_w=round(q_rad_shell_w, 2),
            boiler_efficiency_pct=round(self._ema_eta_pct, 2),
            boiler_water_ml=self.M_water_kg * 1000.0,
            water_free_z_mm=self.water_free_z_mm,
            steam_gen_g_s=round(self._ema_evap_gs, 4),
            steam_out_g_s=round(mdot_steam_out * 1000.0, 4),
            cutoff_pct=round(self.cutoff_pct, 1),
            cyl_expansion_work_w=round(self._ema_ind_work_w, 3),
            specific_steam_g_kj=round(self._ema_spec_steam, 2),
            pump_chamber_kgf=round((self.P_pump_chamber - P_ATM) / PA_PER_KGF_CM2, 3),
            pump_suction_vac_kpa=round(self.pump_suction_vac_kpa, 2),
            pump_shaft_power_w=round(self.pump_shaft_power_w, 4),
            lube_oil_temp_c=round(self.T_lube_oil_c, 1),
            lube_oil_viscosity_mpas=round(self.lube_oil_visc_mpas, 1),
            alcohol_temp_c=round(self.T_alcohol_c, 1),
            alcohol_viscosity_mpas=round(self.alcohol_visc_mpas, 3),
            drain_blowdown_g_s=round(mdot_drain_blow * 1000.0, 3),
            flue_deflection_mm=flue_defl,
            flue_ovalization_mm=flue_oval,
            crown_bulge_mm=round(crown_blg, 4),
            max_boiler_stress_mpa=round(max_stress, 2),
            copper_yield_mpa=round(cu_yield, 1),
            spring_deflect_mm=spr_defl,
            pino_7op_active=bool(el_info.get("pino_7op_active", False)),
            sight_glass_lumen_connected=bool(self.sg_steam_lumen_cond > 0.25 and self.sg_water_lumen_cond > 0.25),
            sight_glass_meniscus_z_mm=round(self.sg_meniscus_z_mm, 2),
            sight_glass_vel_mm_s=round(self.sg_vel_m_s * 1000.0, 2),
            force_L=F_L,
            force_R=F_R,
            wheel_rpm=rpm,
            scale_speed_kmh=v_real_ms * 3.6,
        )
        return F_L, F_R, F_pump, telemetry
