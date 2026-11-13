#!/usr/bin/env python3
"""
sim/thermo_1d.py
Aster Hobby JNR D51 (1/32, Gauge 1) 一维集总参数热力学—流体—配气耦合求解器。
"""

from dataclasses import dataclass, field
import math
from typing import Tuple

import numpy as np

PA_PER_KGF_CM2 = 98066.5
P_ATM = 101325.0
R_STEAM = 461.5       # 水蒸气气体常数 J/(kg·K)
GAMMA_STEAM = 1.30    # 过热/干饱和蒸汽绝热指数
CP_WATER = 4186.0     # 液态水比热容 J/(kg·K)
CP_COPPER = 385.0     # 铜锅筒比热容 J/(kg·K)


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
    water_level_pct: float = 50.0
    alcohol_ml: float = 130.0
    tender_water_ml: float = 320.0
    lubricator_oil_pct: float = 80.0
    cyl_wall_temp_c: float = 25.0
    condensate_g: float = 0.0
    safety_valve_open: bool = False
    dry_fire_alarm: bool = False
    emergency_overpressure: bool = False
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


def latent_heat_j_kg(temp_c: float) -> float:
    """水的汽化潜热 h_fg (J/kg)"""
    return max(1.5e6, (2501.0 - 2.36 * temp_c) * 1e3)


def orifice_mass_flow(p_up: float, p_down: float, temp_k: float, area: float, cd: float = 0.68) -> float:
    """带零压差平滑正则化的可压缩蒸汽孔口质量流量公式 (kg/s)"""
    if area <= 1e-12 or p_up <= p_down:
        return 0.0
    dp = p_up - p_down
    # 临界压比判断阻塞流 (Choked Flow)
    r_crit = (2.0 / (GAMMA_STEAM + 1.0)) ** (GAMMA_STEAM / (GAMMA_STEAM - 1.0))
    p_eff = max(p_down, p_up * r_crit)
    ratio = p_eff / p_up
    term = max(0.0, (ratio ** (2.0 / GAMMA_STEAM)) - (ratio ** ((GAMMA_STEAM + 1.0) / GAMMA_STEAM)))
    psi = math.sqrt((2.0 * GAMMA_STEAM / (GAMMA_STEAM - 1.0)) * term)
    # 低压差光滑过渡防止刚性抖振
    smooth = dp / ((dp * dp + 250.0 * 250.0) ** 0.25) * ((dp) ** -0.5 if dp > 1e-6 else 0.0)
    mdot_ideal = cd * area * p_up * psi / math.sqrt(R_STEAM * max(290.0, temp_k))
    return mdot_ideal * min(1.0, smooth) if dp < 1500.0 else mdot_ideal


class AsterD51Thermo1D:
    """Aster D51 热力学与流体耦合状态机"""

    def __init__(self, calib: CylinderCalibration):
        self.calib = calib

        # 1. 严格对应图纸标准件的几何常数 (SI 单位)
        self.A_head = math.pi * (0.015 / 2.0) ** 2                      # PR-15: 176.71 mm^2
        self.A_crank = math.pi * ((0.015**2 - 0.003**2) / 4.0)          # PF3-1.9: 169.65 mm^2
        self.V_clear = 0.12 * self.A_head * calib.stroke                # 12% 气缸余隙容积
        self.port_breadth = 0.0055                                      # 平面滑阀气口展宽 5.5mm

        self.A_axle_pump = math.pi * (0.0032 / 2.0) ** 2                # PN3.2-1: 8.04 mm^2
        self.A_hand_pump = math.pi * (0.0070 / 2.0) ** 2                # PN7-1.9: 38.48 mm^2
        self.hand_pump_stroke = 0.016                                   # 手摇泵有效行程 16mm

        # 2. 锅筒与煤水车容积基准
        self.V_boiler_total = 3.80e-4                                   # 锅筒内总容积 380 ml
        self.M_copper_shell = 0.46                                      # 铜锅筒与火室金属质量 kg
        self.water_min_ml = 140.0                                       # 水位计 0% 对应最低安全水量
        self.water_max_ml = 280.0                                       # 水位计 100% 对应水量 (50% = 210 ml)

        # 3. 初始物理状态 (默认对标试运转手順 ①~④ 整备完毕状态)
        self.reset_cold_prepared()

    def reset_cold_prepared(self) -> None:
        """重置为冷炉整备完毕状态：水位 50%、酒精 130ml、煤水车水 80%、机油 80%"""
        self.T_boiler_c = 25.0
        self.M_water_kg = 0.210                                         # 210g -> 水位计 50%
        self.P_boiler = P_ATM
        self.P_steam_chest = P_ATM
        self.T_superheat_c = 25.0
        self.safety_valve_open = False

        self.alcohol_ml = 130.0                                         # 燃料箱满载 130 ml
        self.sump_fuel_ml = 0.0                                         # 车底サンプ恒液位油池 (上限 3.5 ml)
        self.tender_water_ml = 320.0                                    # 煤水车水箱 80% 水量
        self.T_feedwater_c = 25.0
        self.lubricator_oil_pct = 80.0                                  # 前车架置换注油器 80% 油位

        # 左右气缸腔压 [Head-End 前腔, Crank-End 后腔] 与缸体热状态
        self.P_cyl_L = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.P_cyl_R = np.array([P_ATM, P_ATM], dtype=np.float64)
        self.T_cyl_wall_c = 25.0
        self.condensate_kg = 0.0008                                     # 初始冷缸微量残水

        self._hand_pump_pending_kg = 0.0

    def reset_hot_ready(self) -> None:
        """一键热炉预设：气压 3.8 kgf/cm²、水位 50%、暖缸就绪，可直接开阀发车"""
        self.reset_cold_prepared()
        self.T_boiler_c = 149.6                                         # 对应约 3.8 kgf/cm² 表压
        self.P_boiler = sat_pressure_pa(self.T_boiler_c)
        self.P_steam_chest = P_ATM
        self.T_superheat_c = 165.0
        self.sump_fuel_ml = 3.5
        self.T_cyl_wall_c = 115.0
        self.condensate_kg = 0.0002

    def trigger_hand_pump(self, strokes: int = 5) -> None:
        """摇动煤水车手动给水泵 (31-5): 约 30 次可补满半锅水位"""
        vol_ml = strokes * (self.A_hand_pump * self.hand_pump_stroke * 1e6) * 1.15
        avail_ml = min(vol_ml, self.tender_water_ml)
        self.tender_water_ml -= avail_ml
        self._hand_pump_pending_kg += avail_ml * 1e-3

    def _valve_port_areas(self, q_valve: float, q_center: float) -> Tuple[float, float, float, float]:
        """
        外进气平面 D 型滑阀 (1-6) 气口开度映射:
        外进气平面 D 型滑阀 (1-6) 气口开度映射 (已用强制转轮实测: 前进档活塞离开前死点时 xv < 0):
        xv < -steam_lap -> 前腔(Head)进气，后腔(Crank)排气
        xv > +steam_lap -> 后腔(Crank)进气，前腔(Head)排气
        """
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

    def _step_cylinder_side(
        self,
        dt_sub: float,
        p_pair: np.ndarray,
        q_piston: float,
        v_piston: float,
        q_fdc: float,
        areas: Tuple[float, float, float, float],
        drain_open: bool,
    ) -> Tuple[float, float]:
        """单侧双作用气缸充放气子步积分，返回消耗蒸汽质量流量与排出乏汽质量流量 (kg/s)"""
        a_adm_h, a_exh_h, a_adm_c, a_exh_c = areas
        t_sh_k = self.T_superheat_c + 273.15
        a_drain = 1.1e-6 if drain_open else 0.0

        # 自前死点向后死点方向测量的位移 s_p in [0, stroke]
        s_p = min(self.calib.stroke, max(0.0, q_fdc - q_piston))
        ds_dt = -v_piston

        v_h = self.V_clear + self.A_head * s_p
        dv_h = +self.A_head * ds_dt
        v_c = self.V_clear + self.A_crank * (self.calib.stroke - s_p)
        dv_c = -self.A_crank * ds_dt

        # 前腔 (Head-End) 质量流量与压力演化
        m_in_h = orifice_mass_flow(self.P_steam_chest, p_pair[0], t_sh_k, a_adm_h)
        m_out_h = orifice_mass_flow(p_pair[0], P_ATM, t_sh_k, a_exh_h)
        m_drn_h = orifice_mass_flow(p_pair[0], P_ATM, t_sh_k, a_drain)
        dp_h = (GAMMA_STEAM * R_STEAM * t_sh_k / v_h) * (m_in_h - m_out_h - m_drn_h) - (
            GAMMA_STEAM * p_pair[0] / v_h
        ) * dv_h

        # 后腔 (Crank-End) 质量流量与压力演化
        m_in_c = orifice_mass_flow(self.P_steam_chest, p_pair[1], t_sh_k, a_adm_c)
        m_out_c = orifice_mass_flow(p_pair[1], P_ATM, t_sh_k, a_exh_c)
        m_drn_c = orifice_mass_flow(p_pair[1], P_ATM, t_sh_k, a_drain)
        dp_c = (GAMMA_STEAM * R_STEAM * t_sh_k / v_c) * (m_in_c - m_out_c - m_drn_c) - (
            GAMMA_STEAM * p_pair[1] / v_c
        ) * dv_c

        # 平面滑阀超压自动浮起保护 (滑り弁の浮上り): 腔压超过蒸汽室压力 0.25 bar 时顶开滑阀回流卸荷
        p_relief = max(self.P_steam_chest + 25000.0, P_ATM + 35000.0)
        p_pair[0] = min(p_relief, max(0.85 * P_ATM, p_pair[0] + dp_h * dt_sub))
        p_pair[1] = min(p_relief, max(0.85 * P_ATM, p_pair[1] + dp_c * dt_sub))

        return (m_in_h + m_in_c), (m_out_h + m_out_c)

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

        # --- 2. 蒸汽室 (Steam Chest)、过热器与双缸 4 子步充放气 ---
        a_regulator = 3.2e-6 * (ctrl.regulator ** 1.35)
        t_boil_k = self.T_boiler_c + 273.15
        has_steam = self.M_water_kg > 0.015 and self.P_boiler > P_ATM

        mdot_reg = orifice_mass_flow(self.P_boiler, self.P_steam_chest, t_boil_k, a_regulator) if has_steam else 0.0

        areas_L = self._valve_port_areas(q_vL, self.calib.q_valve_center_L)
        areas_R = self._valve_port_areas(q_vR, self.calib.q_valve_center_R)

        n_sub = 4
        dt_sub = dt / n_sub
        mdot_cyl_in = 0.0
        mdot_cyl_exh = 0.0
        v_sc_vol = 2.8e-5  # 过热管 + 左右蒸汽室总容积 28 cm^3

        for _ in range(n_sub):
            in_L, exh_L = self._step_cylinder_side(
                dt_sub, self.P_cyl_L, q_pL, v_pL, self.calib.q_fdc_L, areas_L, ctrl.drain_cock
            )
            in_R, exh_R = self._step_cylinder_side(
                dt_sub, self.P_cyl_R, q_pR, v_pR, self.calib.q_fdc_R, areas_R, ctrl.drain_cock
            )
            in_tot = in_L + in_R
            dp_sc = (R_STEAM * (self.T_superheat_c + 273.15) / v_sc_vol) * (mdot_reg - in_tot)
            self.P_steam_chest = min(self.P_boiler, max(P_ATM, self.P_steam_chest + dp_sc * dt_sub))
            mdot_cyl_in += in_tot / n_sub
            mdot_cyl_exh += (exh_L + exh_R) / n_sub

        # --- 3. 冷缸蒸汽凝结、疏水排液与置换式注油器 ---
        if mdot_cyl_in > 1e-6:
            # 蒸汽加热冷气缸金属壁
            q_cond = max(0.0, (self.T_superheat_c - self.T_cyl_wall_c) * 1.8)
            self.T_cyl_wall_c += (q_cond / 42.0) * dt_th
            if self.T_cyl_wall_c < 102.0:
                self.condensate_kg = min(0.004, self.condensate_kg + 0.25 * mdot_cyl_in * dt)
            # 置换式注油器 (4-2): 满油续航约 60 分钟 (3600s)
            self.lubricator_oil_pct = max(0.0, self.lubricator_oil_pct - (80.0 / 3600.0) * dt_th)
        else:
            self.T_cyl_wall_c += (25.0 - self.T_cyl_wall_c) * 0.002 * dt_th

        if ctrl.drain_cock and self.condensate_kg > 0.0:
            self.condensate_kg = max(0.0, self.condensate_kg - 0.0012 * dt_th)

        # 计算沿 MuJoCo +X 滑动轴的净活塞推力: F_+X = (P_crank - P_atm)*A_crank - (P_head - P_atm)*A_head
        lube_factor = 1.0 if self.lubricator_oil_pct > 2.0 else 2.8
        water_drag = 18.0 * (self.condensate_kg / 0.004) if not ctrl.drain_cock else 1.5

        def net_piston_force(p_pair: np.ndarray, vel: float) -> float:
            f_gas = (p_pair[1] - P_ATM) * self.A_crank - (p_pair[0] - P_ATM) * self.A_head
            f_seal = lube_factor * (1.35 * math.tanh(vel * 80.0) + water_drag * vel)
            return f_gas - f_seal

        F_L = net_piston_force(self.P_cyl_L, v_pL)
        F_R = net_piston_force(self.P_cyl_R, v_pR)

        # --- 4. 通风阀 (11-13) 与双安全阀 (28-6: 4.0 kgf/cm² 起跳, 3.6 kgf/cm² 回座) ---
        p_gauge_kgf = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        if p_gauge_kgf >= 4.00:
            self.safety_valve_open = True
        elif p_gauge_kgf <= 3.60:
            self.safety_valve_open = False

        a_blower = 0.16e-6 * ctrl.blower_valve  # 真实 ø0.45mm 针孔喷嘴截面
        a_safety = 3.6e-6 if self.safety_valve_open else 0.0
        mdot_blw = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_blower) if has_steam else 0.0
        mdot_sv = orifice_mass_flow(self.P_boiler, P_ATM, t_boil_k, a_safety) if has_steam else 0.0

        # --- 5. 烟室负压引风与酒精火管传热 ---
        # 引风系数归一化: 外置风机=1.0, 通风阀30°(2~4kg/cm²)=0.95, 走行气缸乏汽喷射最高可达 1.35
        draft_index = (
            (1.00 if ctrl.chimney_fan else 0.0)
            + 18000.0 * mdot_blw  # 微孔高速射流卷吸系数同步放大
            + 2600.0 * mdot_cyl_exh
        )
        draft_eff = min(1.30, 0.12 + 0.88 * math.tanh(draft_index))

        if ctrl.burner_lit and self.sump_fuel_ml > 0.0:
            # 130 ml 额定燃烧约 20 分钟 (0.108 ml/s)，引风增强时毛细挥发略增
            burn_ml_s = 0.108 * (0.55 + 0.45 * draft_eff)
            used_ml = min(self.sump_fuel_ml, burn_ml_s * dt_th)
            self.sump_fuel_ml -= used_ml
            mdot_alc_kg_s = (used_ml / dt_th) * 789e-6 if dt_th > 0 else 0.0
            # 燃料酒精低热值 ~22.5 MJ/kg，火管换热效率随烟室负压引风显著提升 (点火3分钟升至1.8 kgf/cm²)
            eta_boil = 0.34 * draft_eff
            q_comb_w = eta_boil * 22.5e6 * mdot_alc_kg_s
        else:
            q_comb_w = 0.0

        q_loss_w = 0.42 * (self.T_boiler_c - 25.0)  # 陶瓷隔热垫 (セラミックシート) 包覆下的壳体散热
        q_net_w = q_comb_w - q_loss_w

        # 4折火管过热器 (17-5) 出口蒸汽温度
        target_sh_c = self.T_boiler_c + (28.0 * draft_eff if ctrl.burner_lit else 0.0)
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

        mdot_fw_hand = self._hand_pump_pending_kg / dt_th if self._hand_pump_pending_kg > 0 else 0.0
        self._hand_pump_pending_kg = 0.0
        mdot_fw_total = v_to_boiler * 1000.0 + mdot_fw_hand

        # 给水泵柱塞液压负载反力 (旁通全关时顶开逆止阀需承受全锅压)
        if v_pump_ram < -1e-4 and self.tender_water_ml > 0.5:
            p_line = dp_boil_check if v_to_boiler > 0.0 else dp_boil_check * ((v_pump_m3_s / max(1e-12, v_byp_cap)) ** 2)
            F_pump = +p_line * self.A_axle_pump
        else:
            F_pump = 0.0

        # --- 7. 两相热平衡锅筒能量与质量微分方程积分 ---
        mdot_steam_out = mdot_reg + mdot_blw + mdot_sv
        h_fg = latent_heat_j_kg(self.T_boiler_c)
        v_gas = max(2e-5, self.V_boiler_total - self.M_water_kg / 960.0)

        if self.T_boiler_c < 99.9:
            c_tot = self.M_water_kg * CP_WATER + self.M_copper_shell * CP_COPPER
            dT_dt = (q_net_w - mdot_fw_total * CP_WATER * (self.T_boiler_c - self.T_feedwater_c)) / max(80.0, c_tot)
            self.T_boiler_c += dT_dt * dt_th
            self.M_water_kg += mdot_fw_total * dt_th
            self.P_boiler = P_ATM
        else:
            # 计算饱和蒸汽密度随温度导数 d(rho_g)/dT
            p_s1 = sat_pressure_pa(self.T_boiler_c)
            p_s2 = sat_pressure_pa(self.T_boiler_c + 0.1)
            drho_dT = ((p_s2 / (R_STEAM * (t_boil_k + 0.1))) - (p_s1 / (R_STEAM * t_boil_k))) / 0.1

            c_eff = self.M_water_kg * CP_WATER + self.M_copper_shell * CP_COPPER + v_gas * h_fg * drho_dT
            dT_dt = (
                q_net_w
                - mdot_steam_out * h_fg
                - mdot_fw_total * CP_WATER * (self.T_boiler_c - self.T_feedwater_c)
            ) / max(100.0, c_eff)

            self.T_boiler_c = max(25.0, min(195.0, self.T_boiler_c + dT_dt * dt_th))
            dM_dt = mdot_fw_total - mdot_steam_out - v_gas * drho_dT * dT_dt
            self.M_water_kg = max(0.0, min(0.350, self.M_water_kg + dM_dt * dt_th))
            self.P_boiler = max(P_ATM, sat_pressure_pa(self.T_boiler_c)) if self.M_water_kg > 0.015 else P_ATM

        # --- 8. 打包实时遥测指标 ---
        water_ml = self.M_water_kg * 1000.0
        level_pct = 100.0 * (water_ml - self.water_min_ml) / (self.water_max_ml - self.water_min_ml)
        p_gauge_now = max(0.0, (self.P_boiler - P_ATM) / PA_PER_KGF_CM2)
        p_sc_now = max(0.0, (self.P_steam_chest - P_ATM) / PA_PER_KGF_CM2)
        rpm = (omega_3 / (2.0 * math.pi)) * 60.0
        # 动轮直径 43.8mm，按 1:32 折算实车等效时速 km/h
        v_real_ms = abs(omega_3) * (0.5 * 0.0438) * 32.0

        telemetry = ThermoTelemetry(
            sim_time=sim_time,
            boiler_temp_c=self.T_boiler_c,
            boiler_pressure_kgf=p_gauge_now,
            steam_chest_kgf=p_sc_now,
            superheat_temp_c=self.T_superheat_c,
            water_level_pct=max(0.0, min(100.0, level_pct)),
            alcohol_ml=self.alcohol_ml + self.sump_fuel_ml,
            tender_water_ml=self.tender_water_ml,
            lubricator_oil_pct=self.lubricator_oil_pct,
            cyl_wall_temp_c=self.T_cyl_wall_c,
            condensate_g=self.condensate_kg * 1000.0,
            safety_valve_open=self.safety_valve_open,
            dry_fire_alarm=(self.M_water_kg <= 0.145 and ctrl.burner_lit),
            emergency_overpressure=(p_gauge_kgf >= 5.0),
            heat_input_w=max(0.0, q_net_w),
            feedwater_g_s=mdot_fw_total * 1000.0,
            steam_gen_g_s=mdot_steam_out * 1000.0,
            force_L=F_L,
            force_R=F_R,
            wheel_rpm=rpm,
            scale_speed_kmh=v_real_ms * 3.6,
        )
        return F_L, F_R, F_pump, telemetry