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

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

PA_PER_KGF_CM2 = 98066.5
P_ATM = 101325.0
R_STEAM = 461.5       # 水蒸气气体常数 J/(kg·K)
GAMMA_STEAM = 1.30    # 过热/干饱和蒸汽绝热指数
CP_WATER = 4186.0     # 液态水比热容 J/(kg·K)
CP_COPPER = 385.0     # 紫铜比热容 J/(kg·K)
RHO_LIQ_WATER = 980.0 # 热态液态冷凝水密度 kg/m^3


@dataclass
class CabControls:
    """驾驶室与整备操作控制量"""
    fuel_valve: float = 0.0       # 燃料弁开度 [0, 1]
    burner_lit: bool = False      # 酒精燃烧器点火状态
    chimney_fan: bool = False     # 烟囱外置电动抽风机 (通風機) ON/OFF
    blower_valve: float = 0.0     # 通風弁开度 [0, 1] (30° 约对应 0.35)
    regulator: float = 0.0        # 加減弁开度 [0, 1] (起步 30° 约对应 0.35)
    reverser: float = 0.0         # 逆転機归一化位置 [-1.0 后退满档, 0.0 中立 N, +1.0 前进满档]
    drain_cock: bool = True       # ドレン弁开关 (True=打开排冷凝水, False=关闭正常行驶)
    bypass_valve: float = 0.40    # バイパス弁开度 [0.0 全关强制给水, 1.0 全开回流煤水车]
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
    lut_active: bool = False               # 是否激活 RTX 5080 CFD 查找表
    heat_input_w: float = 0.0
    feedwater_g_s: float = 0.0
    steam_gen_g_s: float = 0.0
    force_L: float = 0.0
    force_R: float = 0.0
    wheel_rpm: float = 0.0
    scale_speed_kmh: float = 0.0


def sat_pressure_pa(temp_c: float) -> float:
    """IAPWS-95 饱和水蒸气压高精度解析式 (适用 15°C ~ 220°C)"""
    tk = max(273.16, min(temp_c + 273.15, 640.0))
    theta = tk / 647.096
    tau = 1.0 - theta
    ln_pi = (647.096 / tk) * (
        -7.85951783 * tau
        + 1.84408259 * (tau ** 1.5)
        - 11.7866497 * (tau ** 3.0)
        + 22.6807411 * (tau ** 3.5)
        - 15.9618719 * (tau ** 4.0)
        + 1.80122502 * (tau ** 7.5)
    )
    return 22.064e6 * math.exp(ln_pi)


def sat_temp_c(p_pa: float) -> float:
    """由绝对压力反查近似饱和温度 (°C)"""
    p_bar = max(0.2, p_pa * 1e-5)
    # Antoine 近似式 (1~10 bar 误差 < 0.4°C)
    return (1730.63 / (8.07131 - math.log10(p_bar * 750.062))) - 233.426


def latent_heat_j_kg(temp_c: float) -> float:
    """水的汽化潜热 h_fg (J/kg)"""
    return max(1.5e6, (2501.0 - 2.36 * temp_c) * 1e3)


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
        self.P_boiler = P_ATM
        self.P_steam_chest = P_ATM
        self.T_superheat_c = 25.0
        self.safety_valve_open = False

        self.alcohol_ml = 130.0
        self.sump_fuel_ml = 0.0
        self.tender_water_ml = 320.0
        self.T_feedwater_c = 25.0
        self.lubricator_oil_pct = 80.0

        self.P_cyl_L = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.P_cyl_R = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.T_cyl_wall_c = 25.0
        # 左右气缸 [前腔 Head, 后腔 Crank] 液态冷凝水质量 (kg)
        self.M_cond_L = np.array([8.0e-5, 8.0e-5], dtype=np.float64)
        self.M_cond_R = np.array([8.0e-5, 8.0e-5], dtype=np.float64)
        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

        self.copper_anneal_damage = 0.0
        self.seal_health = 1.0
        self._flash_overpressure_pa = 0.0
        self._hand_pump_pending_kg = 0.0

    def reset_hot_ready(self) -> None:
        """一键热炉预设：气压 3.8 kgf/cm²、水位 50%、暖缸就绪，可直接开阀发车"""
        self.reset_cold_prepared()
        self.T_boiler_c = 149.6
        self.T_crown_c = 156.0
        self.P_boiler = sat_pressure_pa(self.T_boiler_c)
        self.P_steam_chest = P_ATM
        self.T_superheat_c = 165.0
        self.sump_fuel_ml = 3.5
        self.T_cyl_wall_c = 132.0
        self.M_cond_L[:] = 1.5e-5
        self.M_cond_R[:] = 1.5e-5
        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

    def trigger_hand_pump(self, strokes: int = 5) -> None:
        """摇动煤水车手动给水泵 (31-5): 约 30 次可补满半锅水位"""
        vol_ml = strokes * (self.A_hand_pump * self.hand_pump_stroke * 1e6) * 1.15
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

    def _step_cylinder_side(
        self,
        dt_sub: float,
        p_pair: np.ndarray,
        m_cond_pair: np.ndarray,
        q_piston: float,
        v_piston: float,
        q_fdc: float,
        areas: Tuple[float, float, float, float],
        drain_open: bool,
    ) -> Tuple[float, float, float, float, bool]:
        """
        单侧双作用气缸充放气、壁面冷凝积水与不可压缩水锤 (Hydrolock) 子步求解。
        返回: (进汽流量 kg/s, 乏汽流量 kg/s, 壁面凝结生成率 kg/s, 水锤附加冲击反力 N, 是否触发水锤)
        """
        a_adm_h, a_exh_h, a_adm_c, a_exh_c = areas
        t_sh_k = self.T_superheat_c + 273.15
        a_drain = 3.2e-7 if drain_open else 0.0
        # O4-3: 活塞环热退化窜汽面积
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

        # 未开排液阀且腔内积水超限 (>0.48 * V_clear) 时，气液两相乳化阻塞滑阀排气口产生高背压憋死
        exh_choke_h = 0.14 if (not drain_open and vol_w_h > 0.48 * self.V_clear) else 1.0
        exh_choke_c = 0.14 if (not drain_open and vol_w_c > 0.48 * self.V_clear) else 1.0

        m_in_h = orifice_mass_flow(self.P_steam_chest, p_pair[0], t_sh_k, a_adm_h)
        m_out_h = orifice_mass_flow(p_pair[0], P_ATM, t_sh_k, a_exh_h * exh_choke_h)
        m_drn_h = orifice_mass_flow(p_pair[0], P_ATM, t_sh_k, a_drain)

        m_in_c = orifice_mass_flow(self.P_steam_chest, p_pair[1], t_sh_k, a_adm_c)
        m_out_c = orifice_mass_flow(p_pair[1], P_ATM, t_sh_k, a_exh_c * exh_choke_c)
        m_drn_c = orifice_mass_flow(p_pair[1], P_ATM, t_sh_k, a_drain)

        # 前后腔活塞环窜汽
        if p_pair[0] >= p_pair[1]:
            m_blow = orifice_mass_flow(p_pair[0], p_pair[1], t_sh_k, a_blowby)
        else:
            m_blow = -orifice_mass_flow(p_pair[1], p_pair[0], t_sh_k, a_blowby)

        # O4-4: 冷气缸壁进汽凝结积水 vs 排液阀吹除
        t_sat_chest = sat_temp_c(self.P_steam_chest)
        subcool_c = max(0.0, t_sat_chest - self.T_cyl_wall_c - 6.0)
        cond_frac = min(0.88, 0.024 * subcool_c)
        mdot_cond_gen = 3.2 * cond_frac * (m_in_h + m_in_c)
        dm_w_h = 3.2 * cond_frac * m_in_h
        dm_w_c = 3.2 * cond_frac * m_in_c
        if drain_open:
            dp_h_bar = max(0.25, (p_pair[0] - P_ATM) * 1e-5)
            dp_c_bar = max(0.25, (p_pair[1] - P_ATM) * 1e-5)
            dm_w_h -= 0.0060 * dp_h_bar * (1.0 if m_cond_pair[0] > 0.0 else 0.0)
            dm_w_c -= 0.0060 * dp_c_bar * (1.0 if m_cond_pair[1] > 0.0 else 0.0)
        else:
            if a_exh_h > 1e-8 and vol_w_h > 0.85 * self.V_clear:
                dm_w_h -= 0.000015 * (m_out_h / max(1e-6, m_out_h + 2e-4))
            if a_exh_c > 1e-8 and vol_w_c > 0.85 * self.V_clear:
                dm_w_c -= 0.000015 * (m_out_c / max(1e-6, m_out_c + 2e-4))

        max_cond_chamber = 3.20 * self.V_clear * RHO_LIQ_WATER
        m_cond_pair[0] = float(np.clip(m_cond_pair[0] + dm_w_h * dt_sub, 0.0, max_cond_chamber))
        m_cond_pair[1] = float(np.clip(m_cond_pair[1] + dm_w_c * dt_sub, 0.0, max_cond_chamber))

        dp_h = (GAMMA_STEAM * R_STEAM * t_sh_k / v_h_gas) * (
            (1.0 - 0.5 * cond_frac) * m_in_h - m_out_h - m_drn_h - m_blow
        ) - (GAMMA_STEAM * p_pair[0] / v_h_gas) * dv_h
        dp_c = (GAMMA_STEAM * R_STEAM * t_sh_k / v_c_gas) * (
            (1.0 - 0.5 * cond_frac) * m_in_c - m_out_c - m_drn_c + m_blow
        ) - (GAMMA_STEAM * p_pair[1] / v_c_gas) * dv_c

        p_relief = max(self.P_steam_chest + 25000.0, P_ATM + 35000.0)
        p_pair[0] = min(p_relief, max(0.85 * P_ATM, p_pair[0] + dp_h * dt_sub))
        p_pair[1] = min(p_relief, max(0.85 * P_ATM, p_pair[1] + dp_c * dt_sub))

        # 不可压缩液态水锤 (Hydrolock) 单边接触冲击模型:
        # 仅在活塞挤压液柱时产生刚性液压阻力与强非弹性阻尼，回程液柱气穴分离不产生虚假弹性推进功
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

        return (m_in_h + m_in_c), (m_out_h + m_out_c), mdot_cond_gen, f_hydro_x, hydrolock

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
    ) -> Tuple[float, float, float, ThermoTelemetry]:
        """
        推进单个联合仿真步 dt (如 0.5ms)，返回:
        (左活塞推力 F_L, 右活塞推力 F_R, 给水泵柱塞反力 F_pump, 遥测数据包)
        """
        dt_th = dt * max(1.0, ctrl.thermal_warp)

        # --- 1. 煤水车真空恒液位酒精箱 -> サンプ -> 三头毛细燃烧器 ---
        if ctrl.fuel_valve > 0.01 and self.alcohol_ml > 0.0 and self.sump_fuel_ml < 3.5:
            refill_ml = min(self.alcohol_ml, 0.45 * ctrl.fuel_valve * dt_th, 3.5 - self.sump_fuel_ml)
            self.alcohol_ml -= refill_ml
            self.sump_fuel_ml += refill_ml

        if self.sump_fuel_ml <= 1e-4:
            ctrl.burner_lit = False

        # --- 2. 蒸汽室 (Steam Chest)、过热器与双缸 4 子步充放气 + 水锤检测 ---
        a_regulator = 3.2e-6 * (ctrl.regulator ** 1.35)
        t_boil_k = self.T_boiler_c + 273.15
        has_steam = self.M_water_kg > 0.012 and self.P_boiler > P_ATM

        mdot_reg = orifice_mass_flow(self.P_boiler, self.P_steam_chest, t_boil_k, a_regulator) if has_steam else 0.0

        areas_L = self._valve_port_areas(q_vL, self.calib.q_valve_center_L)
        areas_R = self._valve_port_areas(q_vR, self.calib.q_valve_center_R)

        n_sub = 4
        dt_sub = dt / n_sub
        mdot_cyl_in = 0.0
        mdot_cyl_exh = 0.0
        mdot_cond_tot = 0.0
        f_hydro_L_acc = 0.0
        f_hydro_R_acc = 0.0
        hydrolock_L = False
        hydrolock_R = False
        v_sc_vol = 2.8e-5

        for _ in range(n_sub):
            in_L, exh_L, cgen_L, fh_L, hl_L = self._step_cylinder_side(
                dt_sub, self.P_cyl_L, self.M_cond_L, q_pL, v_pL, self.calib.q_fdc_L, areas_L, ctrl.drain_cock
            )
            in_R, exh_R, cgen_R, fh_R, hl_R = self._step_cylinder_side(
                dt_sub, self.P_cyl_R, self.M_cond_R, q_pR, v_pR, self.calib.q_fdc_R, areas_R, ctrl.drain_cock
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
            mdot_cond_tot += (cgen_L + cgen_R) / n_sub
            f_hydro_L_acc += fh_L / n_sub
            f_hydro_R_acc += fh_R / n_sub
            hydrolock_L = hydrolock_L or hl_L
            hydrolock_R = hydrolock_R or hl_R

        self.condensate_kg = float(np.sum(self.M_cond_L) + np.sum(self.M_cond_R))

        # --- 3. 气缸内衬凝结潜热 + 对流暖缸换热、置换式注油器与密封退化 ---
        if mdot_cyl_in > 1e-6 or mdot_cond_tot > 1e-7:
            q_lat_cyl = min(920.0, mdot_cond_tot * 2.10e6)
            q_conv_cyl = max(0.0, (self.T_superheat_c - self.T_cyl_wall_c) * 2.2)
            self.T_cyl_wall_c = min(
                self.T_superheat_c,
                self.T_cyl_wall_c + ((q_lat_cyl + q_conv_cyl) / 48.0) * dt_th,
            )
            self.lubricator_oil_pct = max(0.0, self.lubricator_oil_pct - (80.0 / 3600.0) * dt_th)
        else:
            self.T_cyl_wall_c += (25.0 - self.T_cyl_wall_c) * 0.002 * dt_th

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

        # --- 4. 通风阀 (11-13)、银焊热损伤渗漏与双安全阀 (28-6: 4.0 kgf/cm² 起跳, 3.6 kgf/cm² 回座) ---
        p_gauge_kgf = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        if p_gauge_kgf >= 4.00:
            self.safety_valve_open = True
        elif p_gauge_kgf <= 3.60:
            self.safety_valve_open = False

        a_blower = 0.16e-6 * ctrl.blower_valve
        a_safety = 3.6e-6 if self.safety_valve_open else 0.0
        a_braze_damage_leak = 0.9e-6 * max(0.0, self.copper_anneal_damage - 0.55)
        mdot_blw = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_blower) if has_steam else 0.0
        mdot_sv = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_safety) if has_steam else 0.0
        mdot_braze = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_braze_damage_leak) if has_steam else 0.0

        # --- 5. 烟室负压引风与酒精火管传热 (优先双线性插值 RTX 5080 3D CFD 查找表) ---
        if self.lut is not None:
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

        if ctrl.burner_lit and self.sump_fuel_ml > 0.0:
            burn_ml_s = 0.108 * (0.55 + 0.45 * draft_eff)
            used_ml = min(self.sump_fuel_ml, burn_ml_s * dt_th)
            self.sump_fuel_ml -= used_ml
            mdot_alc_kg_s = (used_ml / dt_th) * 789e-6 if dt_th > 0 else 0.0
            q_comb_w = eta_boil_eff * 22.5e6 * mdot_alc_kg_s
        else:
            q_comb_w = 0.0

        target_sh_c = self.T_boiler_c + (sh_delta_c if ctrl.burner_lit else 0.0)
        self.T_superheat_c += min(1.0, 1.5 * dt_th) * (target_sh_c - self.T_superheat_c)

        # --- 6. 轴动给水泵 (6-5) + 旁通阀 (4-17) 分流 + 手动泵补水 ---
        v_pump_m3_s = 0.85 * self.A_axle_pump * max(0.0, -v_pump_ram) if self.tender_water_ml > 0.5 else 0.0
        dp_boil_check = max(8000.0, self.P_boiler - P_ATM)
        a_bypass = 1.6e-7 * (ctrl.bypass_valve ** 1.4)
        v_byp_cap = 0.65 * a_bypass * math.sqrt(2.0 * dp_boil_check / 1000.0)

        v_to_boiler = max(0.0, v_pump_m3_s - v_byp_cap)
        used_tender_ml = v_to_boiler * 1e6 * dt_th
        if used_tender_ml > self.tender_water_ml:
            v_to_boiler = (self.tender_water_ml * 1e-6) / dt_th
            used_tender_ml = self.tender_water_ml
        self.tender_water_ml -= used_tender_ml

        if self._hand_pump_pending_kg > 1e-7:
            inj_kg = min(self._hand_pump_pending_kg, max(2.5e-5, self._hand_pump_pending_kg * min(1.0, dt_th / 0.35)))
            self._hand_pump_pending_kg -= inj_kg
            mdot_fw_hand = inj_kg / dt_th
        else:
            mdot_fw_hand = 0.0
            self._hand_pump_pending_kg = 0.0
        mdot_fw_total = v_to_boiler * 1000.0 + mdot_fw_hand

        if v_pump_ram < -1e-4 and self.tender_water_ml > 0.5:
            p_line = dp_boil_check if v_to_boiler > 0.0 else dp_boil_check * ((v_pump_m3_s / max(1e-12, v_byp_cap)) ** 2)
            F_pump = +p_line * self.A_axle_pump
        else:
            F_pump = 0.0

        # --- 7. O4-1/O4-2: 坡道假水位、分层紫铜火室顶板传热、空焚き退火与冷水闪蒸尖峰 ---
        # 等效液面倾角: 上坡 (pitch_rad > 0 或 ctrl.grade_deg > 0) 或向前加速 (accel_x > 0) 使锅水后涌向驾驶室水面计
        theta_eff = pitch_rad + math.radians(ctrl.grade_deg) + math.atan2(accel_x, 9.81)
        theta_eff = max(-0.30, min(0.30, theta_eff))
        tan_th = math.tan(theta_eff)

        water_ml = self.M_water_kg * 1000.0
        true_level_pct = 100.0 * (water_ml - self.water_min_ml) / (self.water_max_ml - self.water_min_ml)
        # 水面计距液面几何中心后偏 68mm; 液位总高程差约 14.7mm 对应 0~100%
        dh_gauge_mm = +68.0 * tan_th
        apparent_level_pct = true_level_pct + (dh_gauge_mm / 14.7) * 100.0

        # 火室顶板最高受火前缘在静水位 50% (210ml) 时浸没深度 11.5mm，在 ~145ml (3.5%) 时裸露为 0mm
        h_crown_static_mm = (water_ml - 145.0) * (11.5 / 65.0)
        # 火室顶板前缘位于锅筒中心偏前 24mm (上坡时前缘水浅，下坡急刹时后缘水浅 -> 取前/后最不利浅点!)
        dh_crown_worst_mm = -24.0 * abs(tan_th)
        crown_submerged_mm = h_crown_static_mm + dh_crown_worst_mm
        f_wet = max(0.0, min(1.0, crown_submerged_mm / 4.0))

        # 火焰 58% 辐射/强对流打在紫铜火室顶板，42% 由火管传向锅水
        q_fire_crown = 0.58 * q_comb_w
        q_fire_tubes = 0.42 * q_comb_w
        # 顶板—锅水传热导率: 浸没核态沸腾 22 W/K vs 干蒸汽弱对流 0.36 W/K
        g_crown_to_water = 22.0 * f_wet + 0.36 * (1.0 - f_wet)
        q_crown_to_water = g_crown_to_water * (self.T_crown_c - self.T_boiler_c)
        q_crown_rad_loss = 0.08 * max(0.0, self.T_crown_c - 25.0)

        # 高温干烧状态下突然打入冷水 -> 灼热顶板表面急剧膜态闪蒸直接产生气相超压尖峰 (Flash Steam Spike)
        flash_spike = False
        q_flash_w = 0.0
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
        dT_crown = (q_fire_crown - q_crown_to_water - q_crown_rad_loss - q_flash_w) / c_crown
        self.T_crown_c = max(25.0, min(780.0, self.T_crown_c + dT_crown * dt_th))

        # 紫铜火室 >340°C 退火软化与 >520°C 银钎焊损伤累积
        if self.T_crown_c > 340.0:
            dmg_rate = 0.0018 * ((self.T_crown_c - 340.0) / 80.0) ** 1.4
            self.copper_anneal_damage = min(1.0, self.copper_anneal_damage + dmg_rate * dt_th)

        # --- 8. 两相热平衡锅筒能量与质量微分方程积分 ---
        q_loss_w = 0.42 * (self.T_boiler_c - 25.0)
        q_net_water_w = (q_fire_tubes * (0.25 + 0.75 * f_wet)) + q_crown_to_water + q_flash_w - q_loss_w

        mdot_steam_out = mdot_reg + mdot_blw + mdot_sv + mdot_braze
        h_fg = latent_heat_j_kg(self.T_boiler_c)
        v_gas = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)

        if self.T_boiler_c < 99.9 and not flash_spike:
            c_tot = self.M_water_kg * CP_WATER + self.M_copper_shell * CP_COPPER
            dT_dt = (q_net_water_w - mdot_fw_total * CP_WATER * (self.T_boiler_c - self.T_feedwater_c)) / max(60.0, c_tot)
            self.T_boiler_c += dT_dt * dt_th
            self.M_water_kg = max(0.0, min(0.350, self.M_water_kg + mdot_fw_total * dt_th))
            self.P_boiler = P_ATM
        else:
            p_s1 = sat_pressure_pa(self.T_boiler_c)
            p_s2 = sat_pressure_pa(self.T_boiler_c + 0.1)
            drho_dT = ((p_s2 / (R_STEAM * (t_boil_k + 0.1))) - (p_s1 / (R_STEAM * t_boil_k))) / 0.1

            c_eff = self.M_water_kg * CP_WATER + self.M_copper_shell * CP_COPPER + v_gas * h_fg * drho_dT
            dT_dt = (
                q_net_water_w
                - mdot_steam_out * h_fg
                - mdot_fw_total * CP_WATER * (self.T_boiler_c - self.T_feedwater_c)
            ) / max(85.0, c_eff)

            self.T_boiler_c = max(25.0, min(198.0, self.T_boiler_c + dT_dt * dt_th))
            dM_dt = mdot_fw_total - mdot_steam_out - v_gas * drho_dT * dT_dt
            self.M_water_kg = max(0.0, min(0.350, self.M_water_kg + dM_dt * dt_th))
            self.P_boiler = (
                max(P_ATM, sat_pressure_pa(self.T_boiler_c) + self._flash_overpressure_pa)
                if self.M_water_kg > 0.012
                else P_ATM
            )

        # --- 9. 打包实时遥测指标 ---
        p_gauge_now = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        p_sc_now = max(0.0, (self.P_steam_chest - P_ATM) / PA_PER_KGF_CM2)
        rpm = (omega_3 / (2.0 * math.pi)) * 60.0
        v_real_ms = abs(omega_3) * (0.5 * 0.0438) * 32.0
        dry_alarm = (crown_submerged_mm <= 0.5 and ctrl.burner_lit) or (self.T_crown_c >= 240.0)

        telemetry = ThermoTelemetry(
            sim_time=sim_time,
            boiler_temp_c=self.T_boiler_c,
            boiler_pressure_kgf=p_gauge_now,
            steam_chest_kgf=p_sc_now,
            superheat_temp_c=self.T_superheat_c,
            water_level_pct=max(0.0, min(100.0, apparent_level_pct)),
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
            smokebox_vacuum_pa=smokebox_vac_pa,
            lut_active=(self.lut is not None),
            heat_input_w=max(0.0, q_net_water_w),
            feedwater_g_s=mdot_fw_total * 1000.0,
            steam_gen_g_s=mdot_steam_out * 1000.0,
            force_L=F_L,
            force_R=F_R,
            wheel_rpm=rpm,
            scale_speed_kmh=v_real_ms * 3.6,
        )
        return F_L, F_R, F_pump, telemetry
