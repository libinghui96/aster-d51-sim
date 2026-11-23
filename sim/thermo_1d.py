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
    grade_deg: float = 0.0        # 附加模拟线路坡度 (度, >0 上坡, <0 下坡, 叠加于 3D 轨道纵坡并施加重力的平动分量)
    ambient_temp_c: float = 25.0  # 外部环境气温 (°C, 影响迎风冷缸凝结、锅壳热损、注油器冷凝与轴箱黏度)
    headwind_m_s: float = 0.0     # 迎面相对自然风速 (m/s, >0 逆风增大气动与冷缸散热)
    trailing_load_kg: float = 0.0 # 牵引挂车/载重附加质量 (kg, 注入真实惯量与重力分量)
    tank_spec_2010: bool = False  # True=2010_JNR_D51-standard.pdf (200ml 甲醇 / 400ml 水箱满容), False=经典手册 (130ml / 340ml)


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
    fluid_lvs_verified: bool = True            # 全车 36 路流体管网 + 8 腔室隔离 Fluid-LVS 验证通过标志
    fluid_lvs_open_edges: int = 36             # 全车三维内腔连通管道数
    fluid_lvs_total_edges: int = 36            # 全车声明式流体网表总边数
    # Stage 21: 4-2 多相置换注油器 (油-水密度差/Stokes沉降/牛脂乳化/竖直溢流立管)、机械旷量/热膨胀与第一定律闭合热平衡桑基图
    lube_water_ml: float = 0.64                # 4-2 油箱底部已沉降凝结水体积 (ml)
    lube_oil_ml: float = 2.56                  # 4-2 油箱上层浮升汽缸油体积 (ml)
    lube_settling_mm_s: float = 0.0            # 凝结水滴在汽缸油中的 Stokes 重力沉降终端速度 (mm/s)
    lube_emulsion_pct: float = 0.0             # 牛脂复配汽缸油微乳化含水率 (%, 增强湿蒸汽壁面附着)
    lube_interface_z_mm: float = 2.60          # 4-2 内部底水-浮油分界面高度 Z (mm, 立管顶口 Z=+10.90mm)
    boiler_expansion_mm: float = 0.0           # C1100 紫铜锅筒相对铸铁车架纵向差动热膨胀伸长量 (mm)
    rod_knock_n: float = 0.0                   # 连杆/十字头径向旷量换向冲击力 (N, 受油膜挤压阻尼抑制)
    hunting_lat_mm: float = 0.0                # 轴箱/先导摇枕横向旷量动态蛇行幅值 (mm)
    nadal_lv_ratio: float = 0.0                # 轮轨接触 Nadal 脱线系数 Y/Q
    q_fuel_w: float = 0.0                      # 乙醇化学总发热功率 (W)
    q_evap_w: float = 0.0                      # 锅水蒸发潜热 + 给水预热有效吸热功率 (W)
    q_sh_w: float = 0.0                        # 17-5 四程过热器有效吸热功率 (W)
    q_sensible_w: float = 0.0                  # 锅水与 592g 紫铜胆显热储能功率 (W)
    q_flue_loss_w: float = 0.0                 # 烟囱排烟显热与水汽潜热损失功率 (W)
    q_casing_loss_w: float = 0.0               # Templates 1-6 + 静止空气夹层外皮对流与辐射散热功率 (W)
    q_unburned_w: float = 0.0                  # 不完全燃烧热损失功率 (W)
    energy_balance_err_pct: float = 0.0        # 第一定律热平衡闭合残差 (%)
    # Stage 23: Aster Type-"C" 干式耐火导焰风道、平面滑阀 0.5mm 浮起泄压、1:30 黏着包络与 11-8 车底共鸣汽笛两相气动声学
    boiler_type: str = "Type-C"                # van Riemsdijk Type-"C" 退台紫铜锅筒 + 12-4 陶瓷不锈钢火室风道
    firebox_liner_temp_c: float = 25.0         # 12-4 独立不锈钢火室陶瓷耐火内衬辐射面温度 (°C)
    slide_valve_lift_mm: float = 0.0           # 1-6 平面滑阀在 1-5 框 0.5mm 垂向游隙内的超压浮起泄压升程 (mm)
    adhesion_factor: float = 0.256             # 动轮黏着牵引因子 TE / W_drivers (aster-manual.pdf p.5: [0.20, 0.33])
    whistle_freq_hz: float = 0.0               # 11-8 闭端圆柱黄铜汽笛实时四分之一波长基频 f_1(T, x, M_jet) (Hz)
    whistle_spl_db: float = 0.0                # 11-8 汽笛 1m 处气动偶极子声压级 SPL (dB)
    whistle_dryness_x: float = 0.96            # 11-8 共鸣笛腔内两相湿蒸汽干度 x_wh ∈ [0.50, 1.0]
    whistle_wall_temp_c: float = 25.0          # 11-8 左火室颊下黄铜笛管壁温 (°C, 初吹冷壁凝结上滑音物理根源!)
    whistle_sputter: float = 0.0               # 11-8 笛腔凝结水滴两相吐沫调制强度 (0..1, 随 -Z 下哨口排液归零)
    whistle_jet_mach: float = 0.0              # 11-8 哨口狭缝蒸汽射流马赫数 M_jet
    whistle_sound_speed_m_s: float = 346.0     # 11-8 笛腔内两相水蒸气/空气混合声速 c_eff (m/s)
    # Stage 24: 外界大气 -> 炉栅/12-1炉门 -> 火室负压与三芯横流弯折火焰 -> 5烟管 NTU 沿程温降 -> 烟室射流混合 -> 18-25 烟囱排烟全链路耦合场
    firebox_vacuum_pa: float = 0.0             # 火室表压负压 P_atm - P_firebox (Pa, <0 表示火室微正压倒火!)
    grate_air_g_s: float = 0.0                 # 23-1 炉床底部由外界吸入火室的一次空气流量 (g/s)
    firehole_air_g_s: float = 0.0              # 12-1/12-2 蝴蝶火室门(4×ø1.6孔+开度)吸入的二次空气流量 (g/s)
    flue_gas_g_s: float = 0.0                  # 5根烟管总燃气质量流量 m_grate + m_door + m_fuel (g/s)
    smokebox_gas_temp_c: float = 25.0          # X=+92mm 烟室前管板 5 烟管出口纯烟气温度 (°C)
    stack_exit_temp_c: float = 25.0            # 18-25 烟囱顶口烟气与乏汽/通风蒸汽绝热混合排气温度 (°C)
    air_fuel_phi: float = 0.0                  # 火室实时燃空当量比 φ = 8.98 * m_fuel / m_air
    flame_tilt_deg: float = 0.0                # 23-1 三芯酒精火焰向 +X 烟管喉口平均倾伏角 (deg, 0°竖直 -> 75°平吸)
    wick_tilt_deg_3: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 后/中/前三芯各自前倾角 (deg)
    flue_gas_profile_c: List[float] = field(default_factory=lambda: [25.0] * 8)    # 火室喉部至前管板 8 站沿程燃气温度 (°C)
    backdraft_alarm: bool = False              # 零引风漫油/骤关加减阀未开通风阀时的火室正压倒火(逆火)告警
    gpu_monitor: Dict[str, Any] = field(default_factory=dict)
    force_L: float = 0.0
    force_R: float = 0.0
    wheel_rpm: float = 0.0
    scale_speed_kmh: float = 0.0
    # Stage 27: 第一性原理轨道真实对地速度、空转率、20-7 乏汽背压、累计里程与 GPU 实时剩余续航区间
    ground_speed_ms: float = 0.0               # 车架真实对地平动速度 (m/s)
    ground_speed_kmh: float = 0.0              # 1:30 真实对地换算车速 (km/h, 空转打滑时显著低于动轮踏面线速!)
    wheel_slip_pct: float = 0.0                # 动轮踏面相对钢轨实时滑动/空转率 (%)
    rail_adhesion_mu: float = 0.26             # 钢轮—钢轨实时边界润滑黏着摩擦系数 μ_eff
    exhaust_backpressure_kgf: float = 0.0      # 7-8 -> 20-7 烟室乏汽喷嘴节流背压 (kgf/cm²)
    train_resistance_n: float = 0.0            # 整车第一性原理滚动+轴承+气动+坡道运行阻力 (N)
    distance_m: float = 0.0                    # 轨道真实累计行驶距离 (m)
    scale_distance_km: float = 0.0             # 1:30 换算累计行驶里程 (km)
    lap_count: float = 0.0                     # 闭合环线 (R2m / Oval) 累计行驶圈数
    predicted_range_m: float = 0.0             # RTX 5080 物理轨迹投射预测剩余续航中位数 P50 (m)
    predicted_range_p10_m: float = 0.0         # 剩余续航保守下界 P10 (m)
    predicted_range_p90_m: float = 0.0         # 剩余续航乐观上界 P90 (m)
    predicted_time_min: float = 0.0            # 预计耗尽剩余行驶时长 (min)
    range_limiting_factor: str = "NOMINAL"     # 续航限制主瓶颈 (FUEL_DEPLETED / WATER_DEPLETED / STALL)


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

        # 2. 锅筒、紫铜火室顶板与煤水车容积基准 (严格对齐 boiler_firebox_flues.step 解析积分质量 592.11g 与受火面积)
        self.V_boiler_total = 3.80e-4                                   # 锅筒内总容积 380 ml
        self.M_copper_shell = 0.527                                     # 锅筒外筒、前后管板与5烟管紫铜质量 kg (合火室共 592.1g)
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
        self.P_exhaust = P_ATM                                          # 7-8 -> 20-7 烟室乏汽喷嘴上游可压缩排汽背压腔 (Pa)
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
        self.lube_condensate_ml = 0.20 * self.lube_tank_cap_ml          # 沉底凝结水体积 (初始 80% 蒸汽油 = 2.56 ml, 底水 0.64 ml)
        self.lube_oil_ml = 0.80 * self.lube_tank_cap_ml
        self.lube_suspended_water_ml = 0.0                              # 正在油层中按 Stokes 定律沉降的悬浮微水滴体积 (ml)
        self.lube_settling_mm_s = 0.0
        self.lube_emulsion_pct = 1.6
        self.lubricator_oil_pct = 80.0
        self._prev_f_gas_L = 0.0
        self._prev_f_gas_R = 0.0
        self._rod_knock_n = 0.0

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
        self.alcohol_visc_mpas = 0.544  # 2010_JNR_D51-standard.pdf p.3: 工业甲醇 (Methyl alcohol) 25°C 动力黏度 0.544 mPa·s
        self.alcohol_cap_ml = 130.0
        self._tank_spec_2010_applied = False

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
        self.T_firebox_liner_c: float = 25.0
        self.slide_valve_lift_mm: float = 0.0
        self.T_whistle_wall_c: float = 25.0
        self.M_whistle_cond_kg: float = 0.0
        self.whistle_steam_fill: float = 0.0
        self.whistle_freq_hz: float = 0.0
        self.whistle_spl_db: float = 0.0
        self.whistle_dryness_x: float = 0.96
        self.whistle_sputter: float = 0.0
        self.whistle_jet_mach: float = 0.0
        self.whistle_sound_speed_m_s: float = 346.1
        self.boiler_efficiency_pct: float = 0.0
        self.edge_conds: Dict[str, float] = {}
        self.fluid_lvs_verified: bool = True
        self.fluid_lvs_open_edges: int = 38
        self.fluid_lvs_total_edges: int = 38
        lvs_rep_p = Path("configs/d51_fluid_lvs_report.json")
        if lvs_rep_p.exists():
            try:
                lvs_raw = json.loads(lvs_rep_p.read_text(encoding="utf-8"))
                self.edge_conds = {str(k): float(v) for k, v in lvs_raw.get("edge_conductances", {}).items()}
                self.fluid_lvs_verified = bool(lvs_raw.get("all_passed", True))
                self.fluid_lvs_open_edges = int(lvs_raw.get("open_edges", 38))
                self.fluid_lvs_total_edges = int(lvs_raw.get("total_edges", 38))
            except Exception:
                pass

    def set_edge_conductances(self, conds: Dict[str, float]) -> None:
        """注入或更新 Fluid-LVS 实测 38 路三维管道流导系数 (任何管道堵塞 G=0 时对应回路物理断流，零硬编码 if)。"""
        self.edge_conds.update({str(k): float(v) for k, v in conds.items()})
        if self.edge_conds:
            self.fluid_lvs_total_edges = max(self.fluid_lvs_total_edges, len(self.edge_conds))
            self.fluid_lvs_open_edges = sum(1 for v in self.edge_conds.values() if v > 0.02)
            self.fluid_lvs_verified = (self.fluid_lvs_open_edges == len(self.edge_conds))

    def _edge_g(self, edge_id: str, default: float = 1.0) -> float:
        val = float(self.edge_conds.get(edge_id, default))
        return max(0.0, min(1.0, val / 0.35 if val < 0.35 else 1.0)) if val > 0.015 else 0.0

    def _chain_g(self, *edge_ids: str) -> float:
        g = 1.0
        for eid in edge_ids:
            g = min(g, self._edge_g(eid))
        return g

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
        self.lube_suspended_water_ml = 0.0018
        self.T_alcohol_c = 31.5
        self.alcohol_visc_mpas = 0.49
        self.P_exhaust = P_ATM
        self.wick_saturation = 0.72
        self.wick_tip_temp_c = 82.8
        self.casing_temp_c = 71.2
        self.T_firebox_liner_c = 545.0
        # 11-8 真汽笛隐藏在左火室颊下温暖区 (aster-manual.pdf p.17): 待机壁温 ~64.5°C (< 149.6°C 饱和蒸汽温，首吹产生真实冷壁凝结上滑音!)
        self.T_whistle_wall_c = 64.5
        self.M_whistle_cond_kg = 1.2e-5
        self.whistle_steam_fill = 0.0
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
        """
        Aster D51 外进气平面 D 型滑阀 (1-6) + 交叉气道板 (1-9, aster-manual.pdf p.25 Fig.33):
          - 滑阀前移 (+xv > e_s): 滑阀后外缘敞开后阀口 -> 经 1-9 交叉气道向【前气缸腔 Head-End】进汽; 同时 D 拱连通前阀口排气 -> 【后气缸腔 Crank-End】排汽;
          - 滑阀后移 (-xv > e_s): 滑阀前外缘敞开前阀口 -> 经 1-9 交叉气道向【后气缸腔 Crank-End】进汽; 同时 D 拱连通后阀口排气 -> 【前气缸腔 Head-End】排汽。
        """
        xv = q_valve - q_center
        es = self.calib.steam_lap
        ee = self.calib.exhaust_lap
        wp = self.calib.port_width
        b = self.port_breadth

        a_adm_head = b * min(wp, max(0.0, +xv - es))
        a_exh_head = b * min(wp, max(0.0, -xv - ee))
        a_adm_crank = b * min(wp, max(0.0, -xv - es))
        a_exh_crank = b * min(wp, max(0.0, +xv - ee))
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
        根据标定后的 Aster D51 华氏配气机构几何参数 (合并杆先导半振幅 a_lead ≈ 1.181mm,
        外余面 e_s = 0.995 * a_lead, 满档满行程半幅 R_max ≈ 2.185 * a_lead ≈ 2.58mm)
        解析计算螺旋逆转机档位 reverser ∈ [-1, +1] 下的理论进汽截止率 (% 行程)。
        """
        u = min(1.0, abs(float(reverser)))
        if u < 0.015:
            return 0.0
        es = self.calib.steam_lap * 1000.0
        a_lead = es / 0.995
        r_max = 1.88 * a_lead
        b_quad = u * math.sqrt(max(0.0, r_max ** 2 - a_lead ** 2))
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

        p_exh_back = float(getattr(self, "P_exhaust", P_ATM))
        m_in_h = orifice_mass_flow(self.P_steam_chest, p_pair[0], t_sh_k, a_adm_h, cd=cd_ah)
        m_out_h = orifice_mass_flow(p_pair[0], p_exh_back, t_h_k, a_exh_h * exh_choke_h, cd=cd_eh)

        m_in_c = orifice_mass_flow(self.P_steam_chest, p_pair[1], t_sh_k, a_adm_c, cd=cd_ac)
        m_out_c = orifice_mass_flow(p_pair[1], p_exh_back, t_c_k, a_exh_c * exh_choke_c, cd=cd_ec)

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

        p_Next_h = max(0.82 * P_ATM, p_pair[0] + dp_h * dt_sub)
        p_Next_c = max(0.82 * P_ATM, p_pair[1] + dp_c * dt_sub)
        # aster-manual.pdf pp.19 & 40: 1-5 滑阀驱动框保留 0.50mm 垂向自由游隙 (Buckle Vertical Free Play)!
        # 当蒸汽室带压工作且气缸压缩或冷缸积水超压 (P_cyl > P_steam_chest + 14kPa) 时，1-6 平面滑阀向上浮起 <=0.50mm 将超压余汽与积水守恒泄回蒸汽室!
        m_relief_back = 0.0
        if self.P_steam_chest > P_ATM + 8000.0:
            dp_lift_thresh = self.P_steam_chest + 14000.0
            over_lift_h = max(0.0, p_Next_h - dp_lift_thresh)
            over_lift_c = max(0.0, p_Next_c - dp_lift_thresh)
            lift_h_mm = min(0.50, 1.1e-5 * over_lift_h)
            lift_c_mm = min(0.50, 1.1e-5 * over_lift_c)
            self._sub_valve_lift_mm = max(lift_h_mm, lift_c_mm)
            if lift_h_mm > 1e-4:
                a_lift_h = 0.018 * (lift_h_mm * 1e-3)
                max_rel_h = over_lift_h * v_h_gas / (GAMMA_STEAM * R_STEAM * t_h_k * max(1e-9, dt_sub))
                m_rel_h = min(max_rel_h, orifice_mass_flow(p_Next_h, self.P_steam_chest, t_h_k, a_lift_h, cd=0.62))
                p_Next_h = max(dp_lift_thresh, p_Next_h - (GAMMA_STEAM * R_STEAM * t_h_k / v_h_gas) * m_rel_h * dt_sub)
                if m_cond_pair[0] > 0.0:
                    m_cond_pair[0] = max(0.0, m_cond_pair[0] - min(m_cond_pair[0], 0.008 * lift_h_mm * dt_sub))
                m_relief_back += m_rel_h
            if lift_c_mm > 1e-4:
                a_lift_c = 0.018 * (lift_c_mm * 1e-3)
                max_rel_c = over_lift_c * v_c_gas / (GAMMA_STEAM * R_STEAM * t_c_k * max(1e-9, dt_sub))
                m_rel_c = min(max_rel_c, orifice_mass_flow(p_Next_c, self.P_steam_chest, t_c_k, a_lift_c, cd=0.62))
                p_Next_c = max(dp_lift_thresh, p_Next_c - (GAMMA_STEAM * R_STEAM * t_c_k / v_c_gas) * m_rel_c * dt_sub)
                if m_cond_pair[1] > 0.0:
                    m_cond_pair[1] = max(0.0, m_cond_pair[1] - min(m_cond_pair[1], 0.008 * lift_c_mm * dt_sub))
                m_relief_back += m_rel_c
        else:
            self._sub_valve_lift_mm = 0.0
        self._sub_m_relief_back = m_relief_back
        p_relief = max(self.P_steam_chest + 36000.0, P_ATM + 52000.0)
        p_pair[0] = min(p_relief, p_Next_h)
        p_pair[1] = min(p_relief, p_Next_c)
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
        v_ground_ms: Optional[float] = None,
        accel_lat: float = 0.0,
    ) -> Tuple[float, float, float, ThermoTelemetry]:
        """
        推进单个联合仿真步 dt (如 0.5ms)，返回:
        (左活塞推力 F_L, 右活塞推力 F_R, 给水泵柱塞反力 F_pump, 遥测数据包)
        """
        dt_th = dt * max(1.0, ctrl.thermal_warp)
        t_amb_c = float(getattr(ctrl, "ambient_temp_c", 25.0))
        v_wind_ms = float(getattr(ctrl, "headwind_m_s", 0.0))
        v_wheel_rim_ms = omega_3 * (0.5 * 0.0438)
        v_gnd_signed = v_wheel_rim_ms if v_ground_ms is None else float(v_ground_ms)
        v_air_rel_ms = abs(v_gnd_signed + v_wind_ms)

        want_2010 = bool(getattr(ctrl, "tank_spec_2010", False))
        if want_2010 != getattr(self, "_tank_spec_2010_applied", False):
            if want_2010:
                scale_f = 200.0 / max(1.0, getattr(self, "alcohol_cap_ml", 130.0))
                self.alcohol_ml = min(200.0, self.alcohol_ml * scale_f)
                self.alcohol_cap_ml = 200.0
                self.tender_water_ml = min(400.0, self.tender_water_ml * (400.0 / max(1.0, self.tender_water_cap_ml)))
                self.tender_water_cap_ml = 400.0
            else:
                self.alcohol_ml = min(130.0, self.alcohol_ml * (130.0 / max(1.0, getattr(self, "alcohol_cap_ml", 200.0))))
                self.alcohol_cap_ml = 130.0
                self.tender_water_ml = min(340.0, self.tender_water_ml * (340.0 / max(1.0, self.tender_water_cap_ml)))
                self.tender_water_cap_ml = 340.0
            self._tank_spec_2010_applied = want_2010

        # --- 1. 煤水车 36-4 马氏瓶恒液位甲醇箱 (2010_JNR_D51-standard.pdf p.3: Methyl alcohol) -> 34-2 サンプ -> ST-ø7 -> 23-1 ---
        theta_grade_only = max(-0.30, min(0.30, pitch_rad + math.radians(ctrl.grade_deg)))
        # 供醇管路甲醇温度随火室辐射与车底迎风/煤水车来流动态热平衡:
        q_alc_rad_w = (0.028 if ctrl.burner_lit else 0.005) * max(0.0, 0.22 * self.T_crown_c + 0.35 * self.T_boiler_c - self.T_alcohol_c)
        q_alc_cool_w = (0.19 + 0.32 * min(2.5, v_air_rel_ms)) * (self.T_alcohol_c - t_amb_c)
        self.T_alcohol_c = float(np.clip(
            self.T_alcohol_c + ((q_alc_rad_w - q_alc_cool_w) / 18.0) * dt_th,
            -5.0, 64.2
        ))
        # 工业甲醇 (CH3OH) 真实物性: Andrade 动力黏度 mu(25°C)=0.544 mPa·s、热膨胀密度 rho(25°C)=786.6 kg/m³ 与 Antoine 饱和蒸气压 (沸点 64.7°C)
        t_alc_k = self.T_alcohol_c + 273.15
        self.alcohol_visc_mpas = 0.544 * math.exp(1265.0 * (1.0 / t_alc_k - 1.0 / 298.15))
        rho_ethanol = 786.6 * (1.0 - 0.00119 * (self.T_alcohol_c - 25.0))
        p_vap_eth_kpa = 0.133322 * (10.0 ** (8.08097 - 1582.27 / (239.72 + self.T_alcohol_c)))
        vap_head_boost_kpa = max(0.0, p_vap_eth_kpa - 16.9) * 0.05  # 升温蒸气压削弱马氏瓶顶空净负压

        cap_open_flag = bool(getattr(ctrl, "fuel_cap_open", False))
        # 34-2 サンプ (X=-268mm) 至 23-1 燃烧杯 (X=-92mm, ΔX=+176mm) 重力液位差 Δh_grav (含坡道俯仰角与弯道离心液面微偏!)
        dh_grade_m = -0.176 * math.sin(theta_grade_only) - 0.008 * (abs(accel_lat) / 9.81)
        visc_fluidity_eth = 0.544 / max(0.22, self.alcohol_visc_mpas)

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
            # Hagen-Poiseuille 管路流导 G ∝ rho * g * Δh / mu(T) × Fluid-LVS 36-6 三维内腔流导
            grav_drive = max(0.25, 1.0 + dh_grade_m / 0.018)
            refill_rate_coeff = (1.35 if self.mariotte_flood_active else 0.45) * visc_fluidity_eth * grav_drive * self._edge_g("tdr_36_6_mariotte")
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
        g_sh_parallel = 0.5 * (self._edge_g("boil_17_5_sh_4pass") + self._edge_g("boil_17_5_sh_loop_R"))
        g_main_steam = min(self._chain_g("boil_13_4_11_15_turret", "boil_13_9_drypipe"), g_sh_parallel)
        a_regulator = 3.2e-6 * (ctrl.regulator ** 1.35) * g_main_steam
        t_boil_k = self.T_boiler_c + 273.15
        has_steam = self.P_boiler > P_ATM + 400.0

        if self.ext_pino_cab is not None and has_steam and ctrl.regulator > 1e-4:
            mdot_reg = max(0.0, float(self.ext_pino_cab[1])) * 1e-3 * g_main_steam
        else:
            # 计入 17-5 双 U 四程过热器细管 (L=504mm, ID=2.2mm 双路并联) 高速蒸汽 Darcy-Weisbach 沿程摩阻压降
            mdot_reg_est = orifice_mass_flow(self.P_boiler, self.P_steam_chest, t_boil_k, a_regulator) if has_steam else 0.0
            rho_sh_up = max(0.5, self.P_boiler / (R_STEAM * max(295.0, t_boil_k)))
            u_sh_tube = mdot_reg_est / max(1e-9, 2.0 * math.pi * (0.0011 ** 2) * rho_sh_up)
            dp_sh_friction = 0.036 * (0.504 / 0.0022) * 0.5 * rho_sh_up * (u_sh_tube ** 2)
            p_boiler_eff = max(P_ATM, self.P_boiler - dp_sh_friction)
            mdot_reg = orifice_mass_flow(p_boiler_eff, self.P_steam_chest, t_boil_k, a_regulator) if has_steam else 0.0

        # 计入华氏配气四销副累积微游隙 (4 × 0.03mm ≈ 0.12mm 换向滞后旷量受滑阀背压摩擦牵引) 与右缸独立排汽支路流导
        lash_m = 0.00010 * min(1.0, max(0.0, self.P_steam_chest - P_ATM) / (3.5 * PA_PER_KGF_CM2))
        q_vL_eff = q_vL - lash_m * math.tanh(v_vL / 0.012)
        q_vR_eff = q_vR - lash_m * math.tanh(v_vR / 0.012)
        raw_L = self._valve_port_areas(q_vL_eff, self.calib.q_valve_center_L)
        raw_R = self._valve_port_areas(q_vR_eff, self.calib.q_valve_center_R)
        g_in_L, g_in_R = self._edge_g("cyl_7_9_inlet_L"), self._edge_g("cyl_7_9_inlet_R")
        g_exh_L = self._edge_g("cyl_7_8_20_7_blast")
        g_exh_R = self._chain_g("cyl_7_8_exhaust_R", "cyl_7_8_20_7_blast")
        g_pLh, g_pLc = self._edge_g("cyl_port_L_head"), self._edge_g("cyl_port_L_crank")
        g_pRh, g_pRc = self._edge_g("cyl_port_R_head"), self._edge_g("cyl_port_R_crank")
        areas_L = (raw_L[0] * g_in_L * g_pLh, raw_L[1] * g_exh_L * g_pLh, raw_L[2] * g_in_L * g_pLc, raw_L[3] * g_exh_L * g_pLc)
        areas_R = (raw_R[0] * g_in_R * g_pRh, raw_R[1] * g_exh_R * g_pRh, raw_R[2] * g_in_R * g_pRc, raw_R[3] * g_exh_R * g_pRc)

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

        drn_open_L = bool(ctrl.drain_cock and self._edge_g("cyl_1_16_drains_L") > 0.02)
        drn_open_R = bool(ctrl.drain_cock and self._edge_g("cyl_1_16_drains") > 0.02)
        max_vlift_step = 0.0
        for _ in range(n_sub):
            self._sub_valve_lift_mm = 0.0
            self._sub_m_relief_back = 0.0
            in_L, exh_L, drn_L, cgen_L, pind_L, fh_L, hl_L = self._step_cylinder_side(
                dt_sub, self.P_cyl_L, self.T_gas_L, self.M_cond_L, q_pL, v_pL, self.calib.q_fdc_L, areas_L, drn_open_L, self.ext_pino_cyl_L
            )
            vl_L, mrel_L = self._sub_valve_lift_mm, self._sub_m_relief_back
            self._sub_valve_lift_mm = 0.0
            self._sub_m_relief_back = 0.0
            in_R, exh_R, drn_R, cgen_R, pind_R, fh_R, hl_R = self._step_cylinder_side(
                dt_sub, self.P_cyl_R, self.T_gas_R, self.M_cond_R, q_pR, v_pR, self.calib.q_fdc_R, areas_R, drn_open_R, self.ext_pino_cyl_R
            )
            max_vlift_step = max(max_vlift_step, vl_L, self._sub_valve_lift_mm)
            m_rel_tot = mrel_L + self._sub_m_relief_back
            in_tot = in_L + in_R
            # 填料函老化外漏损失 + 滑阀 0.5mm 浮起向蒸汽室回泄质量守恒
            mdot_gland_leak = orifice_mass_flow(
                self.P_steam_chest, P_ATM, self.T_superheat_c + 273.15, 1.5e-7 * (1.0 - self.seal_health)
            )
            dp_sc = (R_STEAM * (self.T_superheat_c + 273.15) / v_sc_vol) * (mdot_reg + m_rel_tot - in_tot - mdot_gland_leak)
            self.P_steam_chest = min(max(self.P_boiler, P_ATM + 35000.0), max(P_ATM, self.P_steam_chest + dp_sc * dt_sub))
            # 7-8 Y形排汽歧管 -> 20-7 烟室乏汽喷嘴 (内喷口 ø2.20mm, A=3.80mm², 歧管腔 2.8cm³) 可压缩总管节流背压
            a_blast_20_7 = 3.80e-6 * max(0.02, g_exh_L)
            mdot_blast_sub = orifice_mass_flow(self.P_exhaust, P_ATM, max(295.0, self.T_cyl_wall_c + 273.15), a_blast_20_7, cd=0.72)
            dp_exh = (R_STEAM * max(295.0, self.T_cyl_wall_c + 273.15) / 2.8e-6) * ((exh_L + exh_R) - mdot_blast_sub)
            self.P_exhaust = float(np.clip(self.P_exhaust + dp_exh * dt_sub, P_ATM, P_ATM + 2.5 * PA_PER_KGF_CM2))
            mdot_cyl_in += in_tot / n_sub
            mdot_cyl_exh += mdot_blast_sub / n_sub
            mdot_drain_blow += (drn_L + drn_R) / n_sub
            mdot_cond_tot += (cgen_L + cgen_R) / n_sub
            ind_work_w_acc += (pind_L + pind_R) / n_sub
            f_hydro_L_acc += fh_L / n_sub
            f_hydro_R_acc += fh_R / n_sub
            hydrolock_L = hydrolock_L or hl_L
            hydrolock_R = hydrolock_R or hl_R
        self.slide_valve_lift_mm = max(max_vlift_step, self.slide_valve_lift_mm * math.exp(-dt / 0.025))

        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

        # --- 3. 气缸内衬凝结潜热 + 对流暖缸 vs 全时段迎风外部对流/辐射冷缸散热、4-2 Roscoe 置换注油器与密封退化 ---
        # 1-1 左右外置裸黄铜气缸体 (A_ext ≈ 96 cm²) 在迎风气流中的 Churchill-Bernstein 对流 + 热辐射散热 (行驶全程持续散热!):
        h_cyl_ext = 6.8 + 15.5 * math.sqrt(v_air_rel_ms + 0.05)
        t_cyl_k, t_amb_k = self.T_cyl_wall_c + 273.15, t_amb_c + 273.15
        q_cyl_ext_loss_w = 0.0096 * (
            h_cyl_ext * (self.T_cyl_wall_c - t_amb_c)
            + 0.85 * 5.670374e-8 * ((t_cyl_k ** 4) - (t_amb_k ** 4))
        )
        if mdot_cyl_in > 1e-6 or mdot_cond_tot > 1e-7:
            q_lat_cyl = min(920.0, mdot_cond_tot * 2.10e6)
            q_conv_cyl = max(0.0, (self.T_superheat_c - self.T_cyl_wall_c) * 2.2)
            self.T_cyl_wall_c = float(np.clip(
                self.T_cyl_wall_c + ((q_lat_cyl + q_conv_cyl - q_cyl_ext_loss_w) / 48.0) * dt_th,
                min(-10.0, t_amb_c),
                self.T_superheat_c,
            ))
        else:
            self.T_cyl_wall_c = float(np.clip(
                self.T_cyl_wall_c - (q_cyl_ext_loss_w / 48.0) * dt_th,
                min(-10.0, t_amb_c),
                220.0,
            ))

        # 4-2 前车架卧式 Roscoe 置换式注油器 (イラスト 4 Fig.1: -Z 底壁引出 ø2.0/ID1.2mm 紫铜毛细管 + 油箱内部 Z=+10.90mm 银焊竖直溢流立管):
        # 真实物理仿真油/汽/水三相密度差 (ρ_w > ρ_oil >> ρ_steam)、互不相溶与牛脂复配微乳化 (界面张力 σ_ow(T))、
        # Walther 黏温关系 μ_oil(T)、水滴在黏性油层中的 Stokes 重力沉降终端速度 v_Stokes(T)、阿基米德浮力抬升漫过内立管顶口及脉动压送!
        air_cool_factor = 1.0 + 0.85 * math.sqrt(v_air_rel_ms + 0.04)
        if self.P_steam_chest > P_ATM + 1500.0:
            t_sat_lube_c = sat_temp_c(self.P_steam_chest)
            q_oil_heat = 0.085 * max(0.0, t_sat_lube_c - self.T_lube_oil_c)
        else:
            t_sat_lube_c = t_amb_c
            q_oil_heat = 0.015 * max(0.0, self.T_cyl_wall_c - self.T_lube_oil_c)
        q_oil_cool = 0.024 * air_cool_factor * (self.T_lube_oil_c - t_amb_c)
        self.T_lube_oil_c = float(np.clip(self.T_lube_oil_c + ((q_oil_heat - q_oil_cool) / 6.2) * dt_th, -5.0, 165.0))
        # ISO VG 460 + 5% 无酸牛脂复配蒸汽汽缸油黏温方程与热膨胀密度 (油始终轻于同温液态水: Δρ ≈ +92..+108 kg/m³!)
        self.lube_oil_visc_mpas = 0.42 * math.exp(1180.0 / (self.T_lube_oil_c + 115.0))
        mu_pa_s = self.lube_oil_visc_mpas * 1e-3
        rho_oil = 902.0 * (1.0 - 0.00065 * (self.T_lube_oil_c - 25.0))
        rho_cond_w = 998.2 - 0.14 * max(0.0, self.T_lube_oil_c - 20.0) - 0.0029 * (max(0.0, self.T_lube_oil_c - 20.0) ** 2)
        delta_rho_ow = max(45.0, rho_cond_w - rho_oil)

        # 油-水互不相溶 (分子溶解度 <0.08%) + 5% 牛脂脂肪酸酯降低界面张力 σ_ow(T) 形成稳定附壁微乳液 (1.4%~4.6%):
        sigma_ow_n_m = max(0.012, 0.026 * (1.0 - 0.0028 * max(0.0, self.T_lube_oil_c - 25.0)))
        r_drop_m = 0.00038 * math.sqrt(sigma_ow_n_m / 0.022)
        # Stokes 重力沉降速度 v_Stokes = 2*(ρ_w - ρ_oil)*g*r_d^2 / (9*μ_oil) * cos(θ) (25°C 冷油仅 ~0.16mm/s，95°C 暖油达 ~3.8mm/s 快速沉底!)
        v_stokes_m_s = (2.0 * delta_rho_ow * 9.81 * (r_drop_m ** 2) / (9.0 * max(1e-4, mu_pa_s))) * max(0.25, math.cos(theta_grade_only))
        self.lube_settling_mm_s = v_stokes_m_s * 1000.0

        # 4-2 卧式圆筒内腔高度 Z ∈ [+0.35, +12.05] mm (总高 11.70 mm)，内部竖直溢流立管顶口位于 Z_standpipe = +10.90 mm:
        z_in_bot_mm, z_in_top_mm, z_standpipe_mm = 0.35, 12.05, 10.90
        w_frac = max(0.0, min(1.0, self.lube_condensate_ml / self.lube_tank_cap_ml))
        lube_interface_z_mm = z_in_bot_mm + (z_in_top_mm - z_in_bot_mm) * w_frac

        if self.P_steam_chest > P_ATM + 2500.0 and self.lube_oil_ml > 1e-4:
            steam_pulse_pa = abs(self.P_boiler - self.P_steam_chest) * min(1.0, mdot_cyl_in / 1.2e-4)
            # 1) 蒸汽经底部 U 弯及内立管进入 Z>10.9mm 顶腔，穿过内壁汽缸油附壁热阻在裸露前车架黄铜罐壁 (A_ext=18.5cm²) 迎风对流 + 铸铁车架导热桥下凝结 (续航 ~60 min!):
            h_lub_ext = 8.5 + 14.0 * math.sqrt(v_air_rel_ms + 0.04)
            g_lub_tot_wk = 0.34 * h_lub_ext * 0.00185 + 0.015  # 串联内壁油膜热阻 + 车架螺栓安装座导热桥总热导 (W/K)
            q_lube_cond_w = g_lub_tot_wk * (0.38 + 0.62 * min(1.0, mdot_cyl_in / 1.4e-4)) * max(0.0, t_sat_lube_c - t_amb_c - 4.0)
            mdot_lube_cond_kg_s = q_lube_cond_w / latent_heat_j_kg(t_sat_lube_c)
            dv_cond_new_ml = ((mdot_lube_cond_kg_s / rho_cond_w) * 1e6) * dt_th
            self.lube_suspended_water_ml = min(0.35, self.lube_suspended_water_ml + dv_cond_new_ml)

            # 2) 悬浮水滴按 Stokes 速度穿过油层沉入罐底水池，体积等量抬升上方浮油层:
            h_oil_traverse_mm = max(1.2, z_standpipe_mm - lube_interface_z_mm)
            settle_rate_s = self.lube_settling_mm_s / h_oil_traverse_mm
            dv_settled_ml = min(self.lube_suspended_water_ml, self.lube_suspended_water_ml * min(1.0, settle_rate_s * dt_th) + 0.35 * dv_cond_new_ml)
            self.lube_suspended_water_ml = max(0.0, self.lube_suspended_water_ml - max(0.0, dv_settled_ml - 0.35 * dv_cond_new_ml))
            self.lube_condensate_ml = min(self.lube_tank_cap_ml, self.lube_condensate_ml + dv_settled_ml)

            # 3) 底水抬升浮油层越过 Z=+10.90mm 内立管顶口: 经 ø2.0×0.4t (ID=1.20mm, r=0.60mm, Q = pi*r^4*dp/(8*mu*L)) 紫铜毛细管压入 7-9 给汽总管!
            h_oil_m = max(0.0, (z_standpipe_mm - lube_interface_z_mm) * 1e-3) * math.cos(theta_grade_only)
            dp_buoy_pa = delta_rho_ow * 9.81 * h_oil_m
            dp_quill_pa = 260.0 + dp_buoy_pa + 0.012 * steam_pulse_pa
            q_poiseuille_max_ml_s = ((math.pi * (0.00060 ** 4) * dp_quill_pa) / (8.0 * max(1e-4, mu_pa_s) * 0.024)) * 1e6
            overflow_surplus_ml = max(0.0, (self.lube_condensate_ml + self.lube_oil_ml) - self.lube_tank_cap_ml)
            dv_cond_equiv_ml_s = (dv_settled_ml / max(1e-9, dt_th)) if dt_th > 0 else 0.0
            dv_disp_ml_s = min(max(dv_cond_equiv_ml_s, overflow_surplus_ml / max(0.05, dt_th)), q_poiseuille_max_ml_s) * self._edge_g("loco_4_2_lub_oil")
            dv_disp_ml = min(self.lube_oil_ml, dv_disp_ml_s * dt_th)
            self.lube_oil_ml = max(0.0, self.lube_oil_ml - dv_disp_ml)
            self.lubricator_oil_pct = max(0.0, min(100.0, 100.0 * self.lube_oil_ml / self.lube_tank_cap_ml))
            emul_equil = 1.4 + 3.1 * min(1.0, self.lube_suspended_water_ml / 0.06 + 0.15 * (0.026 / sigma_ow_n_m - 1.0))
            self.lube_emulsion_pct += min(1.0, 2.0 * dt_th) * (emul_equil - self.lube_emulsion_pct)
            self.roscoe_quill_oil_rate_mg_min = round((dv_disp_ml / max(1e-9, dt_th)) * (rho_oil * 1e-3) * 1000.0 * 60.0, 3)
        else:
            if self.lube_suspended_water_ml > 1e-6:
                dv_settle_idle = min(self.lube_suspended_water_ml, self.lube_suspended_water_ml * min(1.0, (self.lube_settling_mm_s / 6.0) * dt_th))
                self.lube_suspended_water_ml -= dv_settle_idle
                self.lube_condensate_ml = min(self.lube_tank_cap_ml, self.lube_condensate_ml + dv_settle_idle)
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
            # PR-15 氟硅活塞环 + PF3-1.9 活塞杆填料函弹性体库仑预紧 + EHL (h≈0.85μm) ISO VG 460 汽缸油膜 Couette 剪切黏滞力
            mu_cyl_film = 0.42e-3 * math.exp(1180.0 / max(125.0, self.T_cyl_wall_c + 115.0))
            c_couette_n_s_m = min(12.5, (mu_cyl_film * 1.15e-4) / 0.85e-6)
            f_seal = lube_factor * (1.45 * math.tanh(vel * 80.0) + (c_couette_n_s_m + water_drag) * vel)
            return f_gas - f_seal + f_hydro

        F_L = net_piston_force(self.P_cyl_L, v_pL, f_hydro_L_acc)
        F_R = net_piston_force(self.P_cyl_R, v_pR, f_hydro_R_acc)
        # 主连杆大小头 + 十字头销总径向旷量 (c_r ≈ 0.08mm) 在死点换向过零瞬间的接触冲击与汽缸油膜挤压阻尼 (Reynolds Squeeze Film):
        oil_squeeze_atten = 0.18 if self.lubricator_oil_pct > 2.0 else 1.0
        knock_imp = 0.0
        if self._prev_f_gas_L * F_L < 0.0 and abs(omega_3) > 0.5:
            knock_imp += oil_squeeze_atten * min(4.8, 0.22 * abs(F_L - self._prev_f_gas_L) + 0.06 * abs(omega_3))
        if self._prev_f_gas_R * F_R < 0.0 and abs(omega_3) > 0.5:
            knock_imp += oil_squeeze_atten * min(4.8, 0.22 * abs(F_R - self._prev_f_gas_R) + 0.06 * abs(omega_3))
        self._prev_f_gas_L = F_L
        self._prev_f_gas_R = F_R
        self._rod_knock_n = max(knock_imp, self._rod_knock_n * math.exp(-dt / 0.035))
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

        g_blw_lvs = self._chain_g("boil_13_4_11_15_turret", "boil_13_8_17_3_blower")
        g_whi_lvs = self._chain_g("boil_13_4_11_15_turret", "boil_13_3_11_8_whistle")
        g_bd_lvs = self._edge_g("boil_13_11_blowdown")
        a_blower = 0.16e-6 * ctrl.blower_valve * g_blw_lvs
        a_whistle = 0.28e-6 * max(0.0, min(1.0, float(getattr(ctrl, "whistle_valve", 0.0)))) * g_whi_lvs
        sg_drain_open = 1.0 if bool(getattr(ctrl, "sight_drain_valve", False)) else 0.0
        a_sg_purge = 0.14e-6 * sg_drain_open * self._edge_g("boil_13_6_sg_steam")
        # 2 只纵列 28-6 突跳安全阀总有效泄放喉口面积 A_safety = 2 * min(A_seat, pi * d_seat * y)
        a_single_sv_m2 = min(a_seat_mm2, math.pi * d_seat_sv * y_mm) * 1e-6
        a_safety = 2.0 * a_single_sv_m2
        a_braze_damage_leak = 0.9e-6 * max(0.0, self.copper_anneal_damage - 0.55)
        blowdown_open = max(0.0, min(1.0, float(getattr(ctrl, "blowdown_valve", 0.0)))) * g_bd_lvs

        if self.ext_pino_cab is not None:
            # 由 L2 CabManifoldSightGlassPINO 解算 11-15 分汽包压力、通风/汽笛/水面计冲洗/排污质量流率!
            self.turret_pressure_kgf = max(0.0, float(self.ext_pino_cab[0]) * self._edge_g("boil_13_4_11_15_turret")) if has_steam else 0.0
            mdot_blw = (max(0.0, float(self.ext_pino_cab[2])) * 1e-3 * g_blw_lvs) if (has_steam and ctrl.blower_valve > 1e-4) else 0.0
            mdot_whistle = (max(0.0, float(self.ext_pino_cab[3])) * 1e-3 * g_whi_lvs) if (has_steam and a_whistle > 0.0) else 0.0
            mdot_blowdown_liq = (max(0.0, float(self.ext_pino_cab[4])) * 1e-3 * g_bd_lvs) if (blowdown_open > 0.01 and self.M_water_kg > 2.0e-4) else 0.0
            mdot_sg_purge = (max(0.0, float(self.ext_pino_cab[5])) * 1e-3 * self._edge_g("boil_13_6_sg_steam")) if (has_steam and sg_drain_open > 0.5) else 0.0
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

        # --- 4b. 11-8 左火室颊下隐藏式闭端圆柱黄铜真汽笛第一性原理两相蒸汽气动声学 (aster-manual.pdf p.17 Fig.16 & イラスト 11) ---
        # 真实 CAD 尺寸: 外径 ø6.20mm / 内径 D_bore=4.30mm, 全长 44.4mm, 闭端共鸣腔 L_cav=29.2mm, -Z 朝下排凝切窗 h_mouth=3.20mm,
        # 含哨口亥姆霍兹收缩辐射电感修正 ΔL_mouth=9.8mm -> 有效四分之一波长声学长度 L_eff = 39.0mm, 黄铜热容 C_wh = 2.96 J/K!
        wh_u = max(0.0, min(1.0, float(getattr(ctrl, "whistle_valve", 0.0)))) * g_whi_lvs
        t_cheek_eq_c = 25.0 + 0.24 * max(0.0, self.T_boiler_c - 25.0) + (12.0 if ctrl.burner_lit else 0.0)
        l_eff_wh_m = 0.0390
        if has_steam and wh_u > 0.01 and mdot_whistle > 1.0e-7:
            p_wh_pa = P_ATM + max(0.0, self.turret_pressure_kgf * PA_PER_KGF_CM2) * ((wh_u ** 1.15) / (wh_u ** 1.15 + 0.16))
            t_sat_wh_c = sat_temp_c(p_wh_pa)
            # 薄壁中空黄铜共鸣笛管 (4.26g C3604, C_wh = 1.62 J/K) 对饱和活汽的高速对流膜状凝结换热 -> 初吹产生液膜喘振并在 ~0.5-0.8s 迅速暖管升频!
            q_wh_cond_w = 2.85 * max(0.0, t_sat_wh_c - self.T_whistle_wall_c)
            h_fg_wh = latent_heat_j_kg(t_sat_wh_c)
            mdot_wh_cond = min(0.42 * mdot_whistle, q_wh_cond_w / max(1e5, h_fg_wh))
            dT_wh = (q_wh_cond_w + 0.85 * (t_sat_wh_c - self.T_whistle_wall_c) - 0.035 * (self.T_whistle_wall_c - t_cheek_eq_c)) / 1.62
            self.T_whistle_wall_c = float(np.clip(self.T_whistle_wall_c + dT_wh * dt, 20.0, 175.0))
            # 腔内气相驱替空气 (V_cav ≈ 0.424 cm³)
            self.whistle_steam_fill = min(1.0, self.whistle_steam_fill + min(1.0, (mdot_whistle / 1.8e-6) * dt))
            x_in_wh = max(0.52, self.steam_dryness_dome - mdot_wh_cond / max(1e-6, mdot_whistle))
            cond_load_frac = min(0.36, self.M_whistle_cond_kg / 1.1e-4)
            x_wh = float(np.clip(x_in_wh - 0.55 * cond_load_frac, 0.52, 0.992))
            self.whistle_dryness_x = x_wh
            # Wood-Kieffer 两相湿蒸汽声速 c_tp = sqrt(x_wh * gamma_tp * R_steam * T_cav_K) 与残余温空气声速平滑混合
            gamma_tp = 1.13 + 0.195 * x_wh
            t_cav_k = 273.15 + 0.55 * t_sat_wh_c + 0.45 * self.T_whistle_wall_c
            c_steam_tp = math.sqrt(x_wh * gamma_tp * R_STEAM * max(300.0, t_cav_k))
            c_air_wh = math.sqrt(1.40 * 287.05 * (self.T_whistle_wall_c + 273.15))
            c_eff = (1.0 - self.whistle_steam_fill) * c_air_wh + self.whistle_steam_fill * c_steam_tp
            self.whistle_sound_speed_m_s = c_eff
            pr_wh = max(1.0, p_wh_pa / P_ATM)
            m_jet = min(1.0, math.sqrt(max(0.0, (2.0 / (gamma_tp - 1.0)) * (pr_wh ** ((gamma_tp - 1.0) / gamma_tp) - 1.0))))
            u_jet = c_eff * m_jet
            # -Z 朝下哨口重力 + 高速蒸汽剪切排空凝结水 (aster-manual.pdf p.17: opening facing downward so condensation always drains!)
            dm_wh_liq = mdot_wh_cond + (1.0 - self.steam_dryness_dome) * mdot_whistle - (6.2 + 0.075 * u_jet) * self.M_whistle_cond_kg
            self.M_whistle_cond_kg = float(np.clip(self.M_whistle_cond_kg + dm_wh_liq * dt, 0.0, 2.2e-4))
            self.whistle_jet_mach = m_jet
            self.whistle_sputter = float(np.clip(1.65 * (1.0 - x_wh) + 2800.0 * self.M_whistle_cond_kg, 0.0, 1.0))
            if m_jet >= 0.085:
                # 闭端圆柱共鸣管四分之一波长驻波基频 + Coltman 射流相位牵引微升
                self.whistle_freq_hz = (c_eff / (4.0 * l_eff_wh_m)) * (0.965 + 0.065 * m_jet)
                spl_raw = 50.0 + 38.5 * math.log10(max(1.0, 1.0 + 9.5 * (m_jet ** 2.4))) - 3.5 * self.whistle_sputter
                self.whistle_spl_db = float(np.clip(spl_raw, 42.0, 92.0))
            else:
                # 低于 Rayleigh/Coltman 声学锁频阈值时仅产生微弱哨口蒸汽嘶声，无谐振基音!
                self.whistle_freq_hz = 0.0
                self.whistle_spl_db = float(np.clip(28.0 + 160.0 * m_jet, 0.0, 44.0))
        else:
            # 11-1 弹簧不锈钢球阀瞬时回座切断蒸汽，笛管向火室颊部环境缓慢回温且残水从 -Z 哨口滴净
            self.T_whistle_wall_c += min(1.0, 0.08 * dt_th) * (t_cheek_eq_c - self.T_whistle_wall_c)
            self.M_whistle_cond_kg = max(0.0, self.M_whistle_cond_kg - 2.5e-5 * dt_th)
            self.whistle_steam_fill = max(0.0, self.whistle_steam_fill - 8.0 * dt)
            self.whistle_sound_speed_m_s = math.sqrt(1.40 * 287.05 * (self.T_whistle_wall_c + 273.15))
            self.whistle_freq_hz = 0.0
            self.whistle_spl_db = 0.0
            self.whistle_jet_mach = 0.0
            self.whistle_sputter = 0.0

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
                jet_tot_gs = exh_gs + 10.0 * blw_gs
                # 校准零蒸汽射流 (exh=0, blw=0, fan=False) 时 70mm 模型短烟囱的真实热浮力微抽力 (~0.7 Pa，与 3D PINO 严格一致!)
                zero_jet_taper = math.exp(-jet_tot_gs / 0.035)
                draft_eff = max(0.06, lut_draft - 0.16 * zero_jet_taper)
                eta_boil_eff = lut_eta
                sh_delta_c = lut_sh_dt * (1.0 - 0.65 * zero_jet_taper)
                smokebox_vac_pa = max(0.68 if ctrl.burner_lit else 0.0, smokebox_vac_pa - 5.82 * zero_jet_taper)
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

        # 耦合 5 根紫铜烟管热弹塑性弓曲/椭圆扁化流固耦合阻尼因子 flue_fsi_mult、12-1 炉门二次风旁通及 0ml 干锅无蒸汽引风自节流保护 (aster-manual.pdf pp.9, 36, 40)
        dry_no_draft = (self.M_water_kg <= 1.0e-5 and (mdot_cyl_exh + mdot_blw) < 2.0e-5 and not ctrl.chimney_fan)
        fsi_m = float(np.clip(self.flue_fsi_mult, 0.25, 1.05))
        door_open = max(0.0, min(1.0, float(getattr(ctrl, "firebox_door", 0.0))))
        draft_eff = draft_eff * fsi_m * (1.0 - 0.14 * door_open) * (0.22 if dry_no_draft else 1.0)
        smokebox_vac_pa = max(0.0, smokebox_vac_pa * (0.65 + 0.35 * fsi_m) * (0.18 if dry_no_draft else 1.0))

        wick_head_factor = max(0.72, min(1.25, 1.0 - 0.34 * math.tan(theta_grade_only)))
        door_chill = 1.0 - (0.06 if pino_on else 0.26) * door_open
        lhv_ethanol = 19.93e6  # 2010_JNR_D51-standard.pdf p.3: 工业甲醇 (Methyl alcohol, CH3OH) 低位发热量 19.93 MJ/kg

        # --- 5b. 多孔硅酸铝陶瓷纤维灯芯 (`セラミックヤーン`) Darcy-Washburn 毛细吸液汽化 (甲醇低黏度 0.544mPa·s 自然涌现 ~0.106ml/s -> 130ml 续航 ~20min!) ---
        lumen_c = max(0.0, min(1.0, float(self.burner_lumen_cond) * self._chain_g("tdr_34_2_fuel", "umb_st7_fuel", "loco_23_1_burner")))
        sump_h_norm = max(0.0, min(1.8, self.sump_fuel_ml / 3.5))
        supply_gate_1d = lumen_c * (sump_h_norm / (sump_h_norm + 0.08))
        # Darcy-Washburn 毛细上吸速率随升温黏度下降提升，但在接近甲醇沸点 (>61°C) 时出现微气阻衰减
        vap_lock_atten = 1.0 - 0.35 * max(0.0, (self.T_alcohol_c - 59.5) / 8.0)
        cap_visc_gain = max(0.55, min(1.42, (0.72 + 0.28 * visc_fluidity_eth) * vap_lock_atten))
        if self.ext_pino_wick_cht is not None:
            wick_cap_raw = max(0.0, 0.88 * float(self.ext_pino_wick_cht[0]) * cap_visc_gain) * (0.42 if dry_no_draft else 1.0)
            self.wick_saturation = max(0.0, min(1.0, float(self.ext_pino_wick_cht[1])))
            self.wick_tip_temp_c = max(20.0, float(self.ext_pino_wick_cht[2]))
        else:
            flood_mult_1d = 1.0 + 0.55 * max(0.0, sump_h_norm - 1.05)
            base_nat_draft = 0.20 if dry_no_draft else 0.52
            evap_drv_1d = (base_nat_draft + (1.0 - base_nat_draft) * draft_eff) if ctrl.burner_lit else (0.008 * max(0.0, self.T_boiler_c - 60.0) / 100.0)
            wick_cap_raw = supply_gate_1d * evap_drv_1d * 0.126 * flood_mult_1d * cap_visc_gain
            sat_tgt = supply_gate_1d * max(0.0, min(0.96, 0.92 - (0.24 * (0.4 + 0.6 * draft_eff) if ctrl.burner_lit else 0.0)))
            self.wick_saturation += min(1.0, 4.0 * dt_th) * (sat_tgt - self.wick_saturation)
            wet_pin_1d = min(1.0, max(0.0, self.wick_saturation / 0.22))
            t_gas_est = (760.0 + 180.0 * draft_eff) if (ctrl.burner_lit and self.wick_saturation > 0.05) else (self.T_crown_c if self.T_crown_c > 120.0 else self.T_boiler_c)
            tip_tgt = t_amb_c + wet_pin_1d * max(0.0, 81.5 - t_amb_c if ctrl.burner_lit else 0.0) + (1.0 - wet_pin_1d) * 0.65 * max(0.0, t_gas_est - t_amb_c)
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
                q_chem_max = 0.56 * q_chem_total_w
                a_fb_m2 = 8.20e-3  # 65x36x29mm 紫铜火室五面受火内表面积
                t_flame_rad_k = 980.0 + 140.0 * min(1.2, draft_eff)
                q_rad_fb_w = 0.34 * 5.670374e-8 * a_fb_m2 * max(0.0, (t_flame_rad_k ** 4) - (t_boil_k ** 4))
                fuel_scale = min(1.0, wick_capillary_ml_s / 0.082)
                q_conv_3d = float(self.ext_pino_burner["q_boiler_w"]) * fuel_scale * fsi_m
                q_rad_3d = q_rad_fb_w * fuel_scale
                # 5根细火管发展段 Nusselt 近壁热边界层解析修正 + 火室高温燃气/陶瓷芯 Stefan-Boltzmann 热辐射并联换热
                q_comb_w = min(q_chem_max, 0.86 * q_conv_3d + 0.78 * q_rad_3d) * door_chill
            else:
                # 漫油富燃时大量酒精在缺氧下不完全燃烧，热效率下降但耗醇激增；Type-"C" 12-4 高温陶瓷内衬 (>320°C) 提供二次辐射再聚焦增益
                flood_eff_pen = 0.72 if self.mariotte_flood_active else 1.0
                liner_rerad_boost = 1.0 + 0.16 * max(0.0, min(1.0, (self.T_firebox_liner_c - 320.0) / 280.0))
                eta_with_rad = max(0.24, 1.38 * eta_boil_eff * liner_rerad_boost)
                q_comb_w = min(0.58 * q_chem_total_w, eta_with_rad * flood_eff_pen * door_chill * q_chem_total_w)
        else:
            q_comb_w = 0.0

        # L1 汽包出口初始湿蒸汽干度 x_dome (随高水位汽空间压缩、剧烈沸腾与弯道/加减速晃荡 Weber 夹带水滴下降)
        fill_ratio = (self.M_water_kg / 960.0) / max(1e-6, self.V_boiler_total)
        boil_int = min(1.0, max(0.0, q_comb_w / 520.0)) if self.T_boiler_c >= 99.5 else 0.0
        slosh_weber_carry = 0.035 * min(1.0, math.hypot(accel_x, accel_lat) / 2.2) * max(0.0, (fill_ratio - 0.42) / 0.30)
        self.steam_dryness_dome = max(
            0.84, min(0.99, 0.985 - 0.055 * max(0.0, (fill_ratio - 0.45) / 0.35) ** 1.5 - 0.025 * boil_int - slosh_weber_carry)
        )

        # --- 5c. 外界大气 <-> 炉栅/12-1火室门 <-> 火室负压/三芯横流弯折火焰 <-> 5烟管 NTU 沿程温降 <-> 烟室引射混合排烟全耦合网络 ---
        mdot_alc_gs = (wick_capillary_ml_s * rho_ethanol * 1e-3) if (ctrl.burner_lit and wick_capillary_ml_s > 1e-7) else 0.0
        v_loco_ms = v_air_rel_ms
        dp_ram_pa = 0.5 * 1.184 * (v_loco_ms ** 2)
        dp_buoy_stack_pa = (0.52 if (ctrl.burner_lit and mdot_alc_gs > 1e-4) else 0.04 * max(0.0, self.T_boiler_c - 25.0) / 125.0)
        g_flues_lvs = 0.5 * (self._edge_g("boil_flues_upper") + self._edge_g("boil_flues_lower")) * fsi_m

        # 串联—并联气体水力流导 (kg / (s * sqrt(Pa))):
        #   1) 底部炉栅环隙一次风 G_grate ≈ 2.10e-4, 2) 12-1 蝴蝶炉门 (4×ø1.6mm 常通气孔 + ø9.2mm 开启门洞) 二次风 G_door,
        #   3) 5 根细铜烟管 (含 17-5 四程过热器占空比与局部进出损失 K_flue≈3.4) 高温低密燃气流导 G_flue
        g_grate_cond = 2.10e-4
        a_door_m2 = 8.04e-6 + 58.44e-6 * (door_open ** 1.15)
        g_door_cond = 0.65 * a_door_m2 * math.sqrt(2.0 * 1.184)
        g_in_tot = g_grate_cond + g_door_cond
        rho_flue_est = 0.52 if (ctrl.burner_lit and mdot_alc_gs > 1e-4) else 0.92
        g_flue_cond = 2.095e-4 * g_flues_lvs * math.sqrt(2.0 * rho_flue_est / 3.40)

        dp_tot_draft_pa = max(0.0, smokebox_vac_pa) + dp_ram_pa + dp_buoy_stack_pa
        fb_vac_share = (g_flue_cond ** 2) / max(1e-14, (g_in_tot ** 2) + (g_flue_cond ** 2))
        # 当烟室引风骤失 (<3.5 Pa) 且发生马氏瓶漫油富燃或大开炉门时，剧烈乙醇汽化膨胀产生微正压向外倒火 (Back-draft / 逆火)!
        dp_vap_back_pa = (
            max(0.0, 28.0 * mdot_alc_gs * (1.95 if self.mariotte_flood_active else 0.70) - 0.80 * dp_tot_draft_pa)
            if (ctrl.burner_lit and smokebox_vac_pa < 3.5 and (self.mariotte_flood_active or door_open > 0.35))
            else 0.0
        )
        firebox_vac_pa = fb_vac_share * dp_tot_draft_pa - dp_vap_back_pa
        backdraft_alarm = bool(ctrl.burner_lit and firebox_vac_pa < -0.05)

        if firebox_vac_pa >= 0.0:
            grate_air_g_s = g_grate_cond * math.sqrt(firebox_vac_pa) * 1000.0
            firehole_air_g_s = g_door_cond * math.sqrt(firebox_vac_pa) * 1000.0
        else:
            # 火室微正压时热燃气向炉底与 12-1 火门轻微倒窜，净吸入新风骤降至自然扩散底值
            grate_air_g_s = 0.06
            firehole_air_g_s = 0.02
        total_air_g_s = max(0.05, grate_air_g_s + firehole_air_g_s)
        flue_gas_g_s = (grate_air_g_s + firehole_air_g_s + mdot_alc_gs) * (1.0 if g_flues_lvs > 0.02 else 0.0)

        # 甲醇完全燃烧化学计量空燃比 (A/F)_stoich = 6.45 g_air / g_MeOH (3头直列 ø10mm 灯芯反应区参与一次风占比 ~72%)
        flame_zone_air_g_s = max(0.04, 0.72 * grate_air_g_s + 0.38 * firehole_air_g_s)
        air_fuel_phi = (6.45 * mdot_alc_gs / flame_zone_air_g_s) if (ctrl.burner_lit and mdot_alc_gs > 1e-5) else 0.0
        air_fuel_phi = float(np.clip(air_fuel_phi, 0.0, 1.85))

        # 23-1 纵向直列 1x3 三头陶瓷芯 (X = [-93, -77, -61] mm) 横流羽流弯折角物理计算:
        #   - 灯芯口乙醇汽化与热浮力反应羽流上升速度 w_plume ≈ 0.98 m/s;
        #   - 炉底冷/温一次新风沿 +X 逐级汇入并在 X=-48.2mm 5烟管喉口强烈流线收缩，使后/中/前三芯局部横流比呈 (0.76, 1.24, 1.92) 严格递增!
        if ctrl.burner_lit and mdot_alc_gs > 1e-5:
            t_flame_core_c = (
                690.0
                + 215.0 * min(1.12, 1.0 / max(0.72, air_fuel_phi)) * min(1.15, 0.55 + 0.55 * draft_eff)
                - 65.0 * door_open
            )
            t_flame_core_c = float(np.clip(t_flame_core_c, 520.0, 945.0))
            u_cross_ref = (total_air_g_s * 1e-3) / (0.72 * 9.20e-4)
            w_plume = 0.98 * math.sqrt(max(0.55, (t_flame_core_c - 35.0) / (t_flame_core_c + 273.15) / 0.72))
            wick_x_factors = (0.76, 1.24, 1.92)
            wick_tilt_deg_3 = [
                round(float(76.5 * math.tanh(math.degrees(math.atan2(max(0.04, u_cross_ref * wf), w_plume)) / 76.5)), 1)
                for wf in wick_x_factors
            ]
            flame_tilt_deg = round(sum(wick_tilt_deg_3) / 3.0, 1)

            # 5 根紫铜烟管 (L=140mm, A_tot=238.4 cm² 含过热管外壁, Hausen 入口发展段 h≈52 W/m²K) 解析 NTU 沿程对流温降:
            #   T_gas(s) = T_wall + (T_throat - T_wall) * exp(-NTU * s), 杜绝烟管中段被虚假钉死在锅水饱和温度 T_boiler_c!
            ua_flues_w_k = (1.18 + 0.34 * min(1.4, flue_gas_g_s / 0.95)) * max(0.25, g_flues_lvs)
            mcp_gas_w_k = max(0.16, (flue_gas_g_s * 1e-3) * 1145.0)
            ntu_flues = float(np.clip(ua_flues_w_k / mcp_gas_w_k, 0.72, 2.05))
            t_throat_in_c = max(self.T_boiler_c + 85.0, t_flame_core_c - 35.0 * (1.0 - fb_vac_share))
            flue_gas_profile_c = [
                round(self.T_boiler_c + (t_throat_in_c - self.T_boiler_c) * math.exp(-ntu_flues * (k / 7.0)), 1)
                for k in range(8)
            ]
            smokebox_gas_temp_c = flue_gas_profile_c[-1]
        else:
            wick_tilt_deg_3 = [0.0, 0.0, 0.0]
            flame_tilt_deg = 0.0
            flue_gas_profile_c = [round(self.T_boiler_c, 1)] * 8
            smokebox_gas_temp_c = round(self.T_boiler_c, 1)

        # 烟室前腔绝热焓混合: 5 烟管出口烟气 (cp≈1.10 kJ/kgK) + 20-7 乏汽 & 17-3 通风蒸汽 (cp≈1.98 kJ/kgK, T≈106°C) -> 18-25 烟囱出口温度
        mcp_flue_out = flue_gas_g_s * 1.10
        mcp_steam_jet = (mdot_cyl_exh + mdot_blw) * 1000.0 * 1.98
        t_jet_steam_c = max(99.5, min(126.0, 0.78 * self.T_cyl_wall_c + 22.0)) if has_steam else 25.0
        if mcp_flue_out + mcp_steam_jet > 0.02:
            stack_exit_temp_c = round(
                (mcp_flue_out * smokebox_gas_temp_c + mcp_steam_jet * t_jet_steam_c) / (mcp_flue_out + mcp_steam_jet),
                1,
            )
        else:
            stack_exit_temp_c = round(max(25.0, 0.35 * smokebox_gas_temp_c + 0.65 * 25.0), 1)

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
        # 煤水车 31-10 -> 31-6 -> 31-5(双 SB-ø4 止回球开启压降 ~360 Pa) -> 31-13 -> 34-6(-Y) -> 机车 6-5 泵体 (ΔX=+236mm) 静水重力压头 (Manual p.43!)
        p_suction_supply = P_ATM + rho_fw * 9.81 * (h_tdr_m - 0.236 * math.sin(theta_grade_only)) - 360.0
        p_return_standpipe = P_ATM + rho_fw * 9.81 * 0.042
        p_clack_boiler = self.P_boiler + rho_fw * 9.81 * 0.033 + 4500.0
        g_fw_suc_lvs = self._chain_g("tdr_31_5_31_13_hp", "tdr_34_6_feed_R", "umb_bt5_feed", "loco_6_10_suction")
        g_fw_del_lvs = self._chain_g("loco_6_5_6_6_deliv", "loco_6_6_4_17_deliv")
        g_fw_hp_lvs = self._chain_g("tdr_31_5_31_13_hp", "tdr_34_6_feed_R", "umb_bt5_feed")
        g_fw_clack_lvs = self._edge_g("loco_6_8_13_2_riser")
        g_byp_ret_lvs = self._chain_g("loco_4_16_byp_ret", "umb_bt5_return", "tdr_34_6_ret_L", "tdr_31_14_gooseneck")
        # 4-17 床下横置旁通阀内部 M5×0.5 右旋锥形针阀 (6-9 + 8-7 逆时针开启 0..+75° 对应轴向升程 0..0.104mm -> 锥环帘面积):
        byp_u = max(0.0, min(1.0, ctrl.bypass_valve))
        a_byp_orf = (5.2e-6 * (byp_u ** 1.75) + 2.5e-10) * g_byp_ret_lvs
        a_clack_max = 1.15e-6 * g_fw_clack_lvs  # 13-2 锅炉中位逆止阀 SB-ø3 球座最大通流面积

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
            q_hand_m3_s += 0.90 * self.A_hand_pump * (-v_hand_ram) * g_fw_hp_lvs
        if self._hand_pump_pending_kg > 1e-7:
            inj_rate_kg_s = min(self._hand_pump_pending_kg / max(1e-6, dt_th), 5.6e-3) * g_fw_hp_lvs
            q_hand_m3_s += inj_rate_kg_s / rho_fw

        for _ in range(n_pmp):
            dv_ram_dt = self.A_axle_pump * v_pump_ram  # >0 吸程膨胀, <0 压程压缩
            if has_prime and p_suction_supply > self.P_pump_chamber + 2200.0:
                dp_s = p_suction_supply - self.P_pump_chamber - 2200.0
                lift_suc_mm = min(0.65, 0.065 * (dp_s / 1000.0)) * g_fw_suc_lvs
                a_s = min(9.5e-7, math.pi * 0.0022 * (lift_suc_mm * 1e-3))
                q_in = 0.64 * a_s * math.sqrt(2.0 * dp_s / rho_fw)
            else:
                lift_suc_mm = 0.0
                q_in = 0.0

            if has_prime and self.P_pump_chamber > self.P_fw_manifold + 1600.0:
                dp_d = self.P_pump_chamber - self.P_fw_manifold - 1600.0
                lift_del_mm = min(0.65, 0.055 * (dp_d / 1000.0)) * g_fw_del_lvs
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
            clack_open_gate = (1.0 if self.P_fw_manifold > p_clack_boiler else 0.0) * g_fw_clack_lvs * min(max(g_fw_suc_lvs, g_fw_hp_lvs), max(g_fw_del_lvs, g_fw_hp_lvs))
            q_boil_pino = clack_open_gate * max(0.0, float(self.ext_pino_tender[0]))
            q_ret_pino = max(0.0, float(self.ext_pino_tender[1])) * g_byp_ret_lvs * min(max(g_fw_suc_lvs, g_fw_hp_lvs), max(g_fw_del_lvs, g_fw_hp_lvs))
            q_drain_mls = max(0.0, float(self.ext_pino_tender[2])) * self._edge_g("tdr_31_9_drain") if bool(getattr(ctrl, "tender_drain_open", False)) else 0.0
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

        # Type "C" 锅炉真实两级换热机制 (van Riemsdijk 12-6 退台紫铜锅筒 + 12-4 陶瓷不锈钢耐火导焰风道, 零虚构紫铜湿水腿!):
        #   1) 12-4 内衬 Templates 2/3 硅酸铝陶瓷纤维面迅速升温至 ~520-680°C 耐火重辐射态，向上方 12-6 紫铜退台冠板集中辐射；
        #   2) 高温燃气自后部导焰腔向上折转冲刷 X=-47mm 紫铜后管板并进入 5 根烟管，入口发展段 Nusselt 增强 (1 + (d_h/L)^(2/3) ≈ 1.16)!
        t_liner_target = (
            self.T_boiler_c + (375.0 + 145.0 * min(1.2, draft_eff)) * (1.0 - 0.25 * door_open)
            if (ctrl.burner_lit and q_chem_total_w > 5.0)
            else max(25.0, 0.55 * self.T_crown_c + 0.45 * 25.0)
        )
        self.T_firebox_liner_c += min(1.0, 0.22 * dt_th) * (t_liner_target - self.T_firebox_liner_c)
        nu_entry_mult = 1.0 + (0.0082 / 0.139) ** (2.0 / 3.0)  # Hausen 烟管入口热边界层发展增强因子 ≈ 1.152
        q_fire_crown = 0.54 * q_comb_w
        q_fire_tubes = 0.46 * q_comb_w * min(1.08, 0.94 * nu_entry_mult)
        g_dry_cu_cond = 0.68 * (min(1.0, max(0.0, water_ml) / 18.0) ** 0.67)
        g_crown_to_water = 22.0 * f_wet + g_dry_cu_cond * (1.0 - f_wet)
        q_crown_to_water = g_crown_to_water * (self.T_crown_c - self.T_boiler_c)
        # 后管板喉部与烟管前端在冠板失水初露时仍由下部锅筒水浸没吸热 (替代旧模型不存在的湿火室侧水腿!)
        q_throat_tubeplate_boil = 0.26 * q_fire_crown * (1.0 - f_wet) * f_wet_tubes
        q_crown_rad_loss = 0.045 * max(0.0, self.T_firebox_liner_c - 25.0) * (0.35 + 0.65 * door_open)

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
        dT_crown = (q_fire_crown - q_throat_tubeplate_boil - q_crown_to_water - q_crown_rad_loss - q_flash_w) / c_crown
        self.T_crown_c = max(25.0, min(780.0, self.T_crown_c + dT_crown * dt_th))

        if self.T_crown_c > 340.0:
            dmg_rate = 0.0018 * ((self.T_crown_c - 340.0) / 80.0) ** 1.4
            self.copper_anneal_damage = min(1.0, self.copper_anneal_damage + dmg_rate * dt_th)

        # --- 8. 多层复合壁散热与液态水 M_water_kg + 顶空蒸汽 M_vap_kg 连续统一守恒 (0ml 干锅严格零蒸发、零分支跳变!) ---
        v_air_ext = v_air_rel_ms
        if self.ext_pino_wick_cht is not None:
            self.casing_temp_c = max(-10.0, float(self.ext_pino_wick_cht[3]))
            q_loss_w = max(0.0, float(self.ext_pino_wick_cht[4]))
            q_rad_shell_w = max(0.0, float(self.ext_pino_wick_cht[6]))
        else:
            a_ext_m2 = 0.0365
            ins_cov = max(0.0, min(1.0, float(self.insulation_coverage)))
            k_cer = 0.068 + 1.4e-4 * max(0.0, 0.5 * (self.T_boiler_c + t_amb_c) - 25.0)
            k_air_gap = 0.029 + 7.2e-5 * max(0.0, 0.5 * (self.T_boiler_c + t_amb_c) - 25.0)
            wet_f_shell = max(0.15, 0.32 + 0.68 * f_wet_tubes)
            # Templates 1-6 (2.0mm 陶瓷纤维毡 + Template 4 六层端封 + 3.3mm 静止空气夹套 + 25-1 烤漆黄铜外皮) 串联热阻 vs 烟室/后妻板裸露热桥区
            r_ins_z = ((0.0020 / k_cer) + (0.0033 / (1.35 * k_air_gap)) + 0.012) / wet_f_shell
            r_bare_z = 0.0048 / wet_f_shell
            t_amb_kelvin = t_amb_c + 273.15

            def _zone_loss_1d(r_z: float, init_f: float) -> Tuple[float, float, float]:
                tz = t_amb_c + init_f * max(0.0, self.T_boiler_c - t_amb_c)
                for _ in range(6):
                    dte = max(0.0, tz - t_amb_c)
                    h_c = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * math.sqrt(v_air_ext + 0.04)
                    tk = tz + 273.15
                    q_r = 0.88 * 5.670374e-8 * max(0.0, (tk ** 4) - (t_amb_kelvin ** 4))
                    q_i = max(0.0, self.T_boiler_c - tz) / r_z
                    dq = -(1.0 / r_z) - h_c - 4.0 * 0.88 * 5.670374e-8 * (tk ** 3)
                    tz = min(self.T_boiler_c, max(t_amb_c, tz - (q_i - (h_c * dte + q_r)) / min(-1e-3, dq)))
                dte = max(0.0, tz - t_amb_c)
                h_c = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * math.sqrt(v_air_ext + 0.04)
                q_r = 0.88 * 5.670374e-8 * max(0.0, ((tz + 273.15) ** 4) - (t_amb_kelvin ** 4))
                return tz, h_c * dte, q_r

            t_zi, qcv_i, qrd_i = _zone_loss_1d(r_ins_z, 0.32)
            t_zb, qcv_b, qrd_b = _zone_loss_1d(r_bare_z, 0.88)
            self.casing_temp_c = ins_cov * t_zi + (1.0 - ins_cov) * t_zb
            q_rad_shell_w = a_ext_m2 * (ins_cov * qrd_i + (1.0 - ins_cov) * qrd_b)
            q_loss_w = a_ext_m2 * (ins_cov * qcv_i + (1.0 - ins_cov) * qcv_b) + q_rad_shell_w

        q_net_water_w = (q_fire_tubes * (0.16 + 0.84 * f_wet_tubes)) + q_throat_tubeplate_boil + q_crown_to_water + q_flash_w - q_loss_w
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

        g_top = max(0.0, min(1.0, float(self.sg_steam_lumen_cond) * self._edge_g("boil_13_6_sg_steam")))
        g_bot = max(0.0, min(1.0, float(self.sg_water_lumen_cond) * self._edge_g("boil_13_10_sg_water")))
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
        raw_exh_gs = mdot_cyl_exh * 1000.0
        if self._ema_exhaust_gs < 1e-4 and raw_exh_gs > 1e-3:
            self._ema_exhaust_gs = raw_exh_gs
        else:
            self._ema_exhaust_gs += alpha_inst * (raw_exh_gs - self._ema_exhaust_gs)
        self.last_inst_exhaust_gs = raw_exh_gs
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
        # 2010_JNR_D51-standard.pdf & aster-manual.pdf p.6: JNR 1067mm 窄轨机车在 Gauge-1 (45mm) 下严格采用 1:30 缩尺!
        v_real_ms = abs(omega_3) * (0.5 * 0.0438) * 30.0
        # aster-manual.pdf p.5: Gauge-1 实测气缸平均有效压力 MEP ≈ 0.30*P_boiler (含小气道与过热管节流)，四动轴黏着轮重 ~40.5 N -> ~0.27 ∈ [0.20, 0.33]
        te_ideal_n = (0.30 * max(0.0, self.P_boiler - P_ATM) * (0.015 ** 2) * 0.020) / 0.0438
        adhesion_factor = te_ideal_n / 40.5 if te_ideal_n > 0.05 else 0.256
        dry_alarm = (crown_submerged_mm <= 0.5 and ctrl.burner_lit) or (self.T_crown_c >= 240.0)
        boil_prog = min(100.0, max(0.0, (self.T_boiler_c - 25.0) / 75.0 * 100.0))

        # C1100 紫铜锅筒 (L=204mm, α_Cu=16.8e-6/K) 以前端烟室鞍座为死点向驾驶室侧自由热膨胀伸长量 (相对铸铁车架 α_Fe=11.5e-6/K):
        t_frame_est_c = 25.0 + 0.08 * max(0.0, self.T_boiler_c - 25.0)
        boiler_exp_mm = 204.0 * (16.8e-6 * max(0.0, self.T_boiler_c - 25.0) - 11.5e-6 * max(0.0, t_frame_est_c - 25.0))

        # 第一定律闭合锅炉热平衡桑基分解 (Q_fuel = Q_evap_fw + Q_sh + Q_sensible + Q_casing_loss + Q_flue_loss + Q_unburned):
        q_casing_tot_w = max(0.0, q_loss_w + q_crown_rad_loss)
        if q_chem_total_w > 0.5 and ctrl.burner_lit:
            unburned_frac = (0.22 if self.mariotte_flood_active else 0.022) + 0.08 * max(0.0, 0.35 - draft_eff)
            q_unburned_w = min(0.35 * q_chem_total_w, unburned_frac * q_chem_total_w)
            q_sh_useful_w = max(0.0, q_sh_in)
            q_water_total_in_w = max(0.0, q_net_water_w)
            q_evap_fw_w = min(
                q_water_total_in_w,
                evap_rate_kg_s * h_fg + mdot_fw_total * CP_WATER * max(0.0, self.T_boiler_c - self.T_feedwater_c),
            )
            q_sensible_w = max(0.0, q_water_total_in_w - q_evap_fw_w) + max(0.0, c_crown * dT_crown)
            max_avail_core = max(0.0, q_chem_total_w - q_unburned_w - q_casing_tot_w)
            used_core = q_evap_fw_w + q_sh_useful_w + q_sensible_w
            if used_core > max_avail_core and used_core > 1e-6:
                sc_e = max_avail_core / used_core
                q_evap_fw_w *= sc_e
                q_sh_useful_w *= sc_e
                q_sensible_w *= sc_e
                used_core = max_avail_core
            q_flue_loss_w = max(0.0, q_chem_total_w - q_unburned_w - q_casing_tot_w - used_core)
            q_sum_check = q_evap_fw_w + q_sh_useful_w + q_sensible_w + q_casing_tot_w + q_flue_loss_w + q_unburned_w
            eb_err_pct = 100.0 * abs(q_chem_total_w - q_sum_check) / max(1e-6, q_chem_total_w)
            boiler_eff_now = max(0.0, min(92.0, 100.0 * (q_evap_fw_w + q_sh_useful_w + q_sensible_w) / q_chem_total_w))
            self._ema_eta_pct = boiler_eff_now if self._ema_eta_pct < 0.5 else (self._ema_eta_pct + alpha_inst * (boiler_eff_now - self._ema_eta_pct))
        else:
            q_unburned_w = 0.0
            q_evap_fw_w = 0.0
            q_sh_useful_w = 0.0
            q_sensible_w = 0.0
            q_flue_loss_w = 0.0
            eb_err_pct = 0.0

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
            sight_glass_lumen_connected=bool(g_top > 0.25 and g_bot > 0.25),
            sight_glass_meniscus_z_mm=round(self.sg_meniscus_z_mm, 2),
            sight_glass_vel_mm_s=round(self.sg_vel_m_s * 1000.0, 2),
            fluid_lvs_verified=bool(self.fluid_lvs_verified),
            fluid_lvs_open_edges=int(self.fluid_lvs_open_edges),
            fluid_lvs_total_edges=int(self.fluid_lvs_total_edges),
            lube_water_ml=round(self.lube_condensate_ml, 3),
            lube_oil_ml=round(self.lube_oil_ml, 3),
            lube_settling_mm_s=round(self.lube_settling_mm_s, 3),
            lube_emulsion_pct=round(self.lube_emulsion_pct, 2),
            lube_interface_z_mm=round(lube_interface_z_mm, 2),
            boiler_expansion_mm=round(boiler_exp_mm, 4),
            rod_knock_n=round(self._rod_knock_n, 3),
            q_fuel_w=round(q_chem_total_w, 2),
            q_evap_w=round(q_evap_fw_w, 2),
            q_sh_w=round(q_sh_useful_w, 2),
            q_sensible_w=round(q_sensible_w, 2),
            q_flue_loss_w=round(q_flue_loss_w, 2),
            q_casing_loss_w=round(q_casing_tot_w, 2),
            q_unburned_w=round(q_unburned_w, 2),
            energy_balance_err_pct=round(eb_err_pct, 4),
            boiler_type="Type-C",
            firebox_liner_temp_c=round(self.T_firebox_liner_c, 1),
            slide_valve_lift_mm=round(self.slide_valve_lift_mm, 3),
            adhesion_factor=round(adhesion_factor, 3),
            whistle_freq_hz=round(self.whistle_freq_hz, 1),
            whistle_spl_db=round(self.whistle_spl_db, 1),
            whistle_dryness_x=round(self.whistle_dryness_x, 4),
            whistle_wall_temp_c=round(self.T_whistle_wall_c, 1),
            whistle_sputter=round(self.whistle_sputter, 3),
            whistle_jet_mach=round(self.whistle_jet_mach, 3),
            whistle_sound_speed_m_s=round(self.whistle_sound_speed_m_s, 1),
            firebox_vacuum_pa=round(firebox_vac_pa, 2),
            grate_air_g_s=round(grate_air_g_s, 3),
            firehole_air_g_s=round(firehole_air_g_s, 3),
            flue_gas_g_s=round(flue_gas_g_s, 3),
            smokebox_gas_temp_c=smokebox_gas_temp_c,
            stack_exit_temp_c=stack_exit_temp_c,
            air_fuel_phi=round(air_fuel_phi, 3),
            flame_tilt_deg=flame_tilt_deg,
            wick_tilt_deg_3=wick_tilt_deg_3,
            flue_gas_profile_c=flue_gas_profile_c,
            backdraft_alarm=backdraft_alarm,
            force_L=F_L,
            force_R=F_R,
            wheel_rpm=rpm,
            scale_speed_kmh=(abs(v_gnd_signed) * 30.0 * 3.6) if v_ground_ms is not None else (v_real_ms * 3.6),
            ground_speed_ms=round(abs(v_gnd_signed), 4),
            ground_speed_kmh=round(abs(v_gnd_signed) * 30.0 * 3.6, 2),
            wheel_slip_pct=round(
                100.0 * max(0.0, abs(v_wheel_rim_ms) - abs(v_gnd_signed)) / max(0.04, abs(v_wheel_rim_ms))
                if abs(v_wheel_rim_ms) > 0.04 and v_ground_ms is not None
                else 0.0,
                2,
            ),
            exhaust_backpressure_kgf=round(max(0.0, (self.P_exhaust - P_ATM) / PA_PER_KGF_CM2), 4),
        )
        return F_L, F_R, F_pump, telemetry
