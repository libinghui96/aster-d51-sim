#!/usr/bin/env python3
"""
sim/gpu_track_endurance.py
Stage 27: 第一性原理轮轨牵引/运行阻力解析法则、3D 起伏椭圆环线几何与
RTX 5080 原生 Triton 寄存器驻留融合核全行程续航仿真引擎 (GPUConsistEnduranceEngine)。

核心设计原则 (严禁经验多项式硬编码限速，严禁超大计算图触发 TorchScript NNC 符号化死锁!):
  1. TrackProfile3D: 统一解析 3D 线路中心线几何 (直线 240m / R2m 最小曲线环线 / 3D 坡道大椭圆环线 oval_grade)，
     供 MJCF 轨道生成、CoSimRunner 实时脱线/里程/圈数解算与 GPU 批量轨迹积分共同调用；
  2. ConsistMechanicsLaw: 第一性原理列车纵向与弯道阻力 + 无条件 A-稳定半隐式 Kalker/Polach 轮轨蠕滑—水油边界膜黏着律；
  3. GPUConsistEnduranceEngine:
     - GPU 路径: 采用原生 @triton.jit CUDA 寄存器驻留融合核 (_rollout_chunk_triton_kernel)，将每辆机车的 42 个连续物理状态
       100% 驻留在 RTX 5080 SM 寄存器内连续推进 N_CHUNK=8000 步 (慢速积分累计量采用 FP64 杜绝 ULP 吞没，零中间显存读写、
       零每步 Kernel Launch 开销、零 TorchScript NNC 编译死锁!)；
     - CPU 路径: 纯向量化 NumPy FP64 批量推进回退核，保证无 GPU 的 GitHub Actions CI 100% 快速通过且与 GPU 严格同构。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as tl_libdevice

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

PA_PER_KGF_CM2 = 98066.5
P_ATM = 101325.0
R_STEAM = 461.5
GAMMA_STEAM = 1.30
CP_WATER = 4186.0
CP_COPPER = 385.0
RHO_LIQ_WATER = 980.0

# 2010_JNR_D51-standard.pdf p.3 真值燃料: 工业无水甲醇 (Methyl alcohol, CH3OH)
LHV_METHANOL_J_KG = 19.93e6        # 甲醇低位发热量 19.93 MJ/kg (15.68 MJ/L @ 25°C)
RHO_METHANOL_25 = 786.6            # 25°C 液态甲醇密度 kg/m^3
BETA_METHANOL_K = 0.00119          # 甲醇体膨胀系数 1/K
MU_METHANOL_25_MPAS = 0.544        # 25°C 甲醇动力黏度 mPa·s
STOICH_AF_METHANOL = 6.45          # 甲醇完全燃烧化学计量空燃比 6.45 g_air / g_MeOH

N_STATE = 42
N_CFG = 16


# ==============================================================================
# 1. 统一 3D 轨道几何真值模型 (TrackProfile3D)
# ==============================================================================

@dataclass(frozen=True)
class TrackProjection:
    s_m: float
    lat_err_m: float
    z_rail_top_m: float
    slope_rad: float
    yaw_rad: float
    curvature_inv_m: float


class TrackProfile3D:
    """
    解析 3D 轨道中心线参数方程:
      - "track"     : 平直长测试轨道 X ∈ [-8.0, +240.0] m, Z_top = 0.004 m
      - "curve_r2m" : R = 2.000 m 闭合圆环轨道 (圆心 (0, -2.0)), 周长 12.56637 m, Z_top = 0.004 m
      - "oval_grade": 标准花园铁路 3D 坡道闭合大椭圆环线:
                      2 × 6.00 m 直线段 + 2 × R2.50 m 半圆弯道 (单圈周长 L = 27.70796 m),
                      启动区 s_0 = 3.00 m (X=0, Y=0) 处 Z=Z'=Z''=Z'''=0 绝对平直零残差,
                      后直道最高点 Z_crest = +0.102 m (最大物理纵坡 ±1.50% / ±0.86°)!
    """

    def __init__(
        self,
        mode: str = "oval_grade",
        r_curve_m: float = 2.000,
        oval_straight_m: float = 6.000,
        oval_radius_m: float = 2.500,
        oval_crest_m: float = 0.102,
        rail_base_top_z: float = 0.004,
    ) -> None:
        self.mode = mode
        self.r_curve_m = float(r_curve_m)
        self.L_s = float(oval_straight_m)
        self.R_ov = float(oval_radius_m)
        self.H_crest = float(oval_crest_m)
        self.rail_top_z0 = float(rail_base_top_z)
        self.s0_launch = 0.5 * self.L_s

        if mode == "curve_r2m":
            self.lap_length_m = 2.0 * math.pi * self.r_curve_m
            self.is_closed = True
        elif mode == "oval_grade":
            self.lap_length_m = 2.0 * self.L_s + 2.0 * math.pi * self.R_ov
            self.is_closed = True
        else:
            self.lap_length_m = 240.0
            self.is_closed = False

    def eval_s(self, s_raw: float) -> Tuple[float, float, float, float, float, float]:
        if self.mode == "track":
            return float(s_raw), 0.0, self.rail_top_z0, 0.0, 0.0, 0.0

        if self.mode == "curve_r2m":
            R = self.r_curve_m
            phi = 0.5 * math.pi - (s_raw / R)
            x = R * math.cos(phi)
            y = -R + R * math.sin(phi)
            yaw = phi - 0.5 * math.pi
            return x, y, self.rail_top_z0, yaw, 0.0, -1.0 / R

        L = self.lap_length_m
        s = s_raw % L
        Ls = self.L_s
        R = self.R_ov
        half_x = 0.5 * Ls
        arc = math.pi * R

        if s < Ls:
            x = -half_x + s
            y = 0.0
            yaw = 0.0
            curv = 0.0
        elif s < Ls + arc:
            alpha = (s - Ls) / R
            x = half_x + R * math.sin(alpha)
            y = -R + R * math.cos(alpha)
            yaw = -alpha
            curv = -1.0 / R
        elif s < 2.0 * Ls + arc:
            u2 = s - (Ls + arc)
            x = half_x - u2
            y = -2.0 * R
            yaw = -math.pi
            curv = 0.0
        else:
            beta = (s - (2.0 * Ls + arc)) / R
            x = -half_x - R * math.sin(beta)
            y = -R - R * math.cos(beta)
            yaw = -math.pi - beta
            curv = -1.0 / R

        u = math.pi * (s - self.s0_launch) / L
        su = math.sin(u)
        cu = math.cos(u)
        z_elev = self.H_crest * (su ** 4)
        dz_ds = self.H_crest * (4.0 * math.pi / L) * (su ** 3) * cu
        slope_rad = math.atan(dz_ds)
        return x, y, self.rail_top_z0 + z_elev, yaw, slope_rad, curv

    def project_xy(self, x: float, y: float) -> TrackProjection:
        if self.mode == "track":
            return TrackProjection(
                s_m=float(x),
                lat_err_m=float(y),
                z_rail_top_m=self.rail_top_z0,
                slope_rad=0.0,
                yaw_rad=0.0,
                curvature_inv_m=0.0,
            )

        if self.mode == "curve_r2m":
            R = self.r_curve_m
            dy = y + R
            r = math.hypot(x, dy)
            phi = math.atan2(dy, x)
            s_m = ((0.5 * math.pi - phi) % (2.0 * math.pi)) * R
            return TrackProjection(
                s_m=s_m,
                lat_err_m=r - R,
                z_rail_top_m=self.rail_top_z0,
                slope_rad=0.0,
                yaw_rad=phi - 0.5 * math.pi,
                curvature_inv_m=-1.0 / R,
            )

        Ls = self.L_s
        R = self.R_ov
        half_x = 0.5 * Ls
        arc = math.pi * R

        if x > half_x:
            dx = x - half_x
            dy = y + R
            r = math.hypot(dx, dy)
            alpha = max(0.0, min(math.pi, math.atan2(dx, dy)))
            s_m = Ls + R * alpha
            lat_err = r - R
        elif x < -half_x:
            dx = x + half_x
            dy = y + R
            r = math.hypot(dx, dy)
            beta = max(0.0, min(math.pi, math.atan2(-dx, -dy)))
            s_m = 2.0 * Ls + arc + R * beta
            lat_err = r - R
        elif y >= -R:
            s_m = x + half_x
            lat_err = y
        else:
            s_m = Ls + arc + (half_x - x)
            lat_err = -2.0 * R - y

        _, _, z_top, yaw, slope, curv = self.eval_s(s_m)
        return TrackProjection(
            s_m=s_m,
            lat_err_m=lat_err,
            z_rail_top_m=z_top,
            slope_rad=slope,
            yaw_rad=yaw,
            curvature_inv_m=curv,
        )


# ==============================================================================
# 2. 第一性原理轮轨黏着与列车运行阻力解析法则 (零 8.80*v^2 假阻尼!)
# ==============================================================================

def compute_rail_adhesion_mu(
    v_slip_ms: float,
    drain_cock_open: bool,
    exhaust_gs: float,
    lubricator_oil_pct: float,
    dry_mu_base: float = 0.33,
) -> float:
    wet_film = (0.50 if drain_cock_open else 0.0) + min(0.08, 0.03 * exhaust_gs) * (1.0 if lubricator_oil_pct > 2.0 else 0.5)
    mu_static = dry_mu_base * (1.0 - min(0.54, wet_film))
    slip_decay = 0.72 + 0.28 * math.exp(-abs(v_slip_ms) / 0.32)
    return float(max(0.11, min(0.34, mu_static * slip_decay)))


def compute_physical_train_resistance_n(
    v_ground_ms: float,
    total_mass_kg: float,
    ambient_temp_c: float = 25.0,
    headwind_m_s: float = 0.0,
    n_trailing_cars: int = 0,
    include_curve_flange_1d: float = 0.0,
) -> float:
    w_norm_n = max(1.0, total_mass_kg) * 9.81
    brg_visc_mult = math.exp(920.0 * (1.0 / max(255.0, ambient_temp_c + 273.15) - 1.0 / 298.15))
    c_rr = 0.0028 + 0.00009 * max(0, n_trailing_cars)
    f_roll = c_rr * w_norm_n * math.tanh(abs(v_ground_ms) / 0.006)
    f_journal = (0.048 * (total_mass_kg / 6.0) * brg_visc_mult) * abs(v_ground_ms)

    rho_air = 101325.0 / (287.05 * max(250.0, ambient_temp_c + 273.15))
    cd_tot = 0.82 + 0.16 * max(0, n_trailing_cars)
    a_front_m2 = 0.0112
    v_rel = v_ground_ms + headwind_m_s
    f_aero_signed = 0.5 * rho_air * cd_tot * a_front_m2 * v_rel * abs(v_rel)
    f_aero_opposing = f_aero_signed if v_ground_ms >= 0.0 else -f_aero_signed
    f_curve = (0.0096 * abs(include_curve_flange_1d)) * w_norm_n * math.tanh(abs(v_ground_ms) / 0.008)
    return max(0.0, f_roll + f_journal + f_curve + max(-0.4 * f_roll, f_aero_opposing))


# ==============================================================================
# 3. RTX 5080 原生 @triton.jit 寄存器驻留多步融合核 (零 NNC 死锁、零逐步 Launch 开销!)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _sat_p_tl(T_c):
        tk = tl.minimum(tl.maximum(T_c + 273.15, 273.16), 640.0)
        theta = tk / 647.096
        tau = 1.0 - theta
        tau_sq = tau * tau
        tau_sqrt = tl.sqrt(tl.maximum(tau, 0.0))
        tau_15 = tau * tau_sqrt
        tau_30 = tau_sq * tau
        tau_35 = tau_30 * tau_sqrt
        tau_40 = tau_sq * tau_sq
        tau_75 = tau_40 * tau_35
        ln_pi = (647.096 / tk) * (
            -7.85951783 * tau
            + 1.84408259 * tau_15
            - 11.7866497 * tau_30
            + 22.6807411 * tau_35
            - 15.9618719 * tau_40
            + 1.80122502 * tau_75
        )
        return 22.064e6 * tl.exp(ln_pi)

    @triton.jit
    def _sat_t_tl(P_pa):
        p_bar = tl.minimum(tl.maximum(P_pa * 1.0e-5, 0.20), 25.0)
        log10_mmhg = tl.log(p_bar * 750.062) * 0.4342944819032518
        tk_c = 1730.63 / (8.07131 - log10_mmhg) - 233.426
        return tl.minimum(tl.maximum(tk_c, 20.0), 225.0)

    @triton.jit
    def _orifice_tl(p_up, p_dn, temp_k, area, cd):
        dp = tl.maximum(p_up - p_dn, 0.0)
        p_eff = tl.maximum(p_dn, p_up * 0.545728)
        ratio = tl.minimum(tl.maximum(p_eff / tl.maximum(p_up, 1000.0), 0.01), 1.0)
        r_a = tl_libdevice.pow(ratio, 1.5384615)
        r_b = tl_libdevice.pow(ratio, 1.7692308)
        psi = tl.sqrt(8.6666667 * tl.maximum(r_a - r_b, 0.0))
        smooth = tl.minimum(dp / 1500.0, 1.0)
        mdot = cd * area * p_up * psi / tl.sqrt(461.5 * tl.maximum(temp_k, 290.0))
        return tl.where((area > 1.0e-12) & (dp > 0.0), mdot * smooth, 0.0)

    @triton.jit
    def _rollout_chunk_triton_kernel(
        state_ptr,
        cfg_ptr,
        B,
        n_chunk,
        t_start,
        dt,
        BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        idx = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask = idx < B

        sb = state_ptr + idx * 42
        cb = cfg_ptr + idx * 16

        # 一次性加载状态至 SM 寄存器 (慢速积分累计量采用 FP64 杜绝 ULP 吞没与相位漂移)
        T_boil = tl.load(sb + 0, mask=mask, other=25.0)
        T_crown = tl.load(sb + 1, mask=mask, other=25.0)
        T_liner = tl.load(sb + 2, mask=mask, other=25.0)
        M_water_d = tl.load(sb + 3, mask=mask, other=0.21).to(tl.float64)
        P_boil = tl.load(sb + 4, mask=mask, other=101325.0)
        M_vap = tl.load(sb + 5, mask=mask, other=1e-4)
        P_chest = tl.load(sb + 6, mask=mask, other=101325.0)
        P_exh = tl.load(sb + 7, mask=mask, other=101325.0)
        T_sh = tl.load(sb + 8, mask=mask, other=25.0)
        T_sh_wall = tl.load(sb + 9, mask=mask, other=25.0)
        T_cyl = tl.load(sb + 10, mask=mask, other=25.0)
        T_alc = tl.load(sb + 11, mask=mask, other=25.0)
        alc_ml_d = tl.load(sb + 12, mask=mask, other=130.0).to(tl.float64)
        sump_ml_d = tl.load(sb + 13, mask=mask, other=3.5).to(tl.float64)
        tdr_w_d = tl.load(sb + 14, mask=mask, other=272.0).to(tl.float64)
        lube_oil_ml_d = tl.load(sb + 15, mask=mask, other=2.56).to(tl.float64)
        sv_open_f = tl.load(sb + 16, mask=mask, other=0.0)

        P_Lh = tl.load(sb + 17, mask=mask, other=101325.0)
        P_Lc = tl.load(sb + 18, mask=mask, other=101325.0)
        P_Rh = tl.load(sb + 19, mask=mask, other=101325.0)
        P_Rc = tl.load(sb + 20, mask=mask, other=101325.0)
        Mc_Lh = tl.load(sb + 21, mask=mask, other=1.5e-5)
        Mc_Lc = tl.load(sb + 22, mask=mask, other=1.5e-5)
        Mc_Rh = tl.load(sb + 23, mask=mask, other=1.5e-5)
        Mc_Rc = tl.load(sb + 24, mask=mask, other=1.5e-5)

        theta_w = tl.load(sb + 25, mask=mask, other=0.0)
        omega_w = tl.load(sb + 26, mask=mask, other=0.0)
        v_ground = tl.load(sb + 27, mask=mask, other=0.0)
        dist_m_d = tl.load(sb + 28, mask=mask, other=0.0).to(tl.float64)
        wheel_dist_m_d = tl.load(sb + 29, mask=mask, other=0.0).to(tl.float64)
        s_track_d = tl.load(sb + 30, mask=mask, other=3.0).to(tl.float64)
        spd_int = tl.load(sb + 31, mask=mask, other=0.0)
        has_moved_f = tl.load(sb + 32, mask=mask, other=0.0)
        active_f = tl.load(sb + 33, mask=mask, other=0.0)
        stop_time_s = tl.load(sb + 34, mask=mask, other=2400.0)
        stop_reason_f = tl.load(sb + 35, mask=mask, other=0.0)
        stall_timer = tl.load(sb + 36, mask=mask, other=0.0)
        steam_up_time_s = tl.load(sb + 37, mask=mask, other=0.0)
        peak_slip_pct = tl.load(sb + 38, mask=mask, other=0.0)
        max_p_exh_kgf = tl.load(sb + 39, mask=mask, other=0.0)
        evap_total_g_d = tl.load(sb + 40, mask=mask, other=0.0).to(tl.float64)
        fuel_burned_ml_d = tl.load(sb + 41, mask=mask, other=0.0).to(tl.float64)

        # 一次性加载工况常量至寄存器
        is_hot = tl.load(cb + 0, mask=mask, other=1.0)
        ai_on = tl.load(cb + 1, mask=mask, other=1.0) > 0.5
        reg_manual = tl.load(cb + 2, mask=mask, other=0.48)
        rev_manual = tl.load(cb + 3, mask=mask, other=0.65)
        byp_manual = tl.load(cb + 4, mask=mask, other=0.35)
        blw_manual = tl.load(cb + 5, mask=mask, other=0.22)
        n_cars = tl.load(cb + 6, mask=mask, other=0.0)
        extra_load_kg = tl.load(cb + 7, mask=mask, other=0.0)
        T_amb = tl.load(cb + 8, mask=mask, other=25.0)
        v_wind = tl.load(cb + 9, mask=mask, other=0.0)
        extra_grade_rad = tl.load(cb + 10, mask=mask, other=0.0)
        target_kmh = tl.load(cb + 11, mask=mask, other=50.0)
        mode_code = tl.load(cb + 12, mask=mask, other=2.0)
        seal_drag_mult = tl.load(cb + 13, mask=mask, other=1.0)
        wick_mult = tl.load(cb + 14, mask=mask, other=1.0)
        adh_mult = tl.load(cb + 15, mask=mask, other=1.0)

        p_atm = 101325.0
        pa_kgf = 98066.5
        r_steam = 461.5
        gamma_s = 1.30
        a_head = 1.767146e-4
        a_crank = 1.696460e-4
        v_clear = 4.24115e-7
        es = 0.001175
        ee = 0.000024
        wp = 0.00112
        b_p = 0.0055
        a_lead = 0.0011809
        b_max = 0.001872
        L_oval = 27.707963

        for k in range(n_chunk):
            t_cur = t_start + (k + 1) * dt
            act = (active_f > 0.5) & mask
            act_f = tl.where(act, 1.0, 0.0)

            M_water = M_water_d.to(tl.float32)
            alc_ml = alc_ml_d.to(tl.float32)
            sump_ml = sump_ml_d.to(tl.float32)
            tdr_w = tdr_w_d.to(tl.float32)
            lube_oil_ml = lube_oil_ml_d.to(tl.float32)

            p_kgf = tl.maximum((P_boil - p_atm) / pa_kgf, 0.0)
            steaming_ready = (steam_up_time_s < 0.0) & (p_kgf >= 2.20)
            steam_up_time_s = tl.where(steaming_ready, t_cur, steam_up_time_s)

            water_ml = M_water * 1000.0
            water_pct = 100.0 * (water_ml - 140.0) / 140.0
            cond_tot_g = (Mc_Lh + Mc_Lc + Mc_Rh + Mc_Rc) * 1000.0
            scale_kmh = tl.abs(v_ground) * 108.0

            fuel_empty = (alc_ml + sump_ml) <= 0.08
            low_water_cut = (water_ml <= 115.0) & (tdr_w <= 0.5)
            burner_lit = (~fuel_empty) & (~low_water_cut)
            chimney_fan = (p_kgf < 1.25) & (tl.abs(omega_w) < 4.0) & burner_lit
            need_drain_ai = (T_cyl < 100.0) | (cond_tot_g > 0.28)
            drain_open = tl.where(ai_on, need_drain_ai, (T_cyl < 88.0) & (t_cur < 8.0))

            can_open_reg = tl.where(is_hot > 0.5, True, steam_up_time_s >= 0.0) & (~fuel_empty) & (~low_water_cut)
            spd_ratio = tl.minimum(tl.maximum(scale_kmh / 42.0, 0.0), 1.0)
            rev_ai = 0.84 - 0.30 * spd_ratio
            v_err = target_kmh - scale_kmh
            spd_int = tl.where(can_open_reg & act, tl.minimum(tl.maximum(spd_int + v_err * dt, -68.0), 38.0), spd_int)
            reg_ai = tl.minimum(tl.maximum(0.24 + 0.018 * v_err + 0.0048 * spd_int, 0.0), 0.92)
            reg_ai = tl.where(need_drain_ai, tl.minimum(reg_ai, 0.42), reg_ai)
            reg_cmd = tl.where(can_open_reg, tl.where(ai_on, reg_ai, reg_manual), 0.0)
            rev_cmd = tl.where(ai_on, rev_ai, rev_manual)
            blw_ai = tl.minimum(tl.maximum(0.95 * (3.72 - p_kgf) - tl.minimum(scale_kmh * 0.006, 0.35) + 0.12, 0.0), 1.0)
            blw_cmd = tl.where(burner_lit, tl.where(ai_on, blw_ai, blw_manual), 0.0)
            byp_ai = tl.minimum(tl.maximum(0.42 + 0.065 * (water_pct - 50.0) - 0.50 * tl.minimum(tl.maximum(reg_cmd * 0.35, 0.0), 0.4), 0.0), 1.0)
            byp_cmd = tl.where(ai_on, byp_ai, byp_manual)
            fuel_valve = tl.where(p_kgf > 3.92, tl.maximum(1.0 - 2.4 * (p_kgf - 3.92), 0.35), 1.0)

            # 3D 椭圆坡道与曲率 (FP64 整数圈扣减避免长距离取模漂移)
            n_laps_i = (s_track_d / 27.707963267948966).to(tl.int32)
            s_mod_raw = (s_track_d - n_laps_i.to(tl.float64) * 27.707963267948966).to(tl.float32)
            s_mod = tl.where(s_mod_raw < 0.0, s_mod_raw + L_oval, s_mod_raw)
            u_ov = (3.14159265 / L_oval) * (s_mod - 3.0)
            su_ov = tl.sin(u_ov)
            dz_ds = 0.102 * (12.5663706 / L_oval) * (su_ov * su_ov * su_ov) * tl.cos(u_ov)
            slope_3d = tl.where(mode_code > 1.5, tl_libdevice.atan(dz_ds), 0.0)
            in_ov_turn = ((s_mod >= 6.0) & (s_mod < 13.85398)) | (s_mod >= 19.85398)
            curv_ov = tl.where(in_ov_turn, 0.40, 0.0)
            curv_abs = tl.where(mode_code > 1.5, curv_ov, tl.where(mode_code > 0.5, 0.50, 0.0))
            theta_grade = tl.minimum(tl.maximum(slope_3d + extra_grade_rad, -0.25), 0.25)

            # 甲醇物性 (CH3OH)
            v_air_rel = tl.abs(v_ground + v_wind)
            q_alc_rad = tl.where(burner_lit, 0.028, 0.005) * tl.maximum(0.22 * T_crown + 0.35 * T_boil - T_alc, 0.0)
            q_alc_cool = (0.19 + 0.32 * tl.minimum(v_air_rel, 2.5)) * (T_alc - T_amb)
            T_alc = tl.minimum(tl.maximum(T_alc + ((q_alc_rad - q_alc_cool) / 18.0) * dt * act_f, -5.0), 64.2)
            mu_meoh = 0.544 * tl.exp(1265.0 * (1.0 / (T_alc + 273.15) - 0.003354016))
            rho_meoh = 786.6 * (1.0 - 0.00119 * (T_alc - 25.0))
            visc_gain = tl.minimum(tl.maximum(0.544 / tl.maximum(mu_meoh, 0.22), 0.65), 1.42)

            refill_ml = tl.minimum(tl.minimum(alc_ml, tl.maximum(3.5 - sump_ml, 0.0)), 0.55 * fuel_valve * visc_gain * dt) * act_f
            alc_next_d = alc_ml_d - refill_ml.to(tl.float64)
            alc_ml_d = tl.where(alc_next_d > 0.0, alc_next_d, 0.0)
            sump_ml_d = sump_ml_d + refill_ml.to(tl.float64)
            sump_ml = sump_ml_d.to(tl.float32)

            # 加减阀 + 17-5 过热器摩阻压降 (受锅筒—蒸汽室等温压差物理上界保护，杜绝大步长显式过冲)
            has_st_f = tl.where(P_boil > (p_atm + 400.0), 1.0, 0.0)
            a_reg = 3.2e-6 * tl_libdevice.pow(tl.maximum(reg_cmd, 1.0e-6), 1.35) * tl.where(reg_cmd > 1.0e-4, 1.0, 0.0)
            mdot_reg_raw = _orifice_tl(P_boil, P_chest, T_boil + 273.15, a_reg, 0.68) * has_st_f
            rho_sh_in = tl.maximum(P_boil / (r_steam * (T_boil + 273.15)), 0.5)
            u_sh_pipe = mdot_reg_raw / (7.60265e-6 * rho_sh_in + 1.0e-9)
            dp_sh_pipe = 8.24727 * 0.5 * rho_sh_in * (u_sh_pipe * u_sh_pipe)
            p_boil_eff = tl.maximum(P_boil - dp_sh_pipe, p_atm)
            mdot_reg_unc = _orifice_tl(p_boil_eff, P_chest, T_boil + 273.15, a_reg, 0.68) * has_st_f
            mdot_reg_cap = tl.maximum(p_boil_eff - P_chest, 0.0) * 2.8e-5 / (r_steam * (T_sh + 273.15) * dt)
            mdot_reg = tl.minimum(mdot_reg_unc, mdot_reg_cap * 1.35)

            # 1:1 华氏配气 + 1-9 交叉气道双缸四腔展开寄存器计算 (解析无过冲孔口弛豫: 关阀时 P_chest=p_atm 严禁产生虚假正压!)
            th_L = theta_w
            th_R = theta_w - 1.5707963
            sin_L = tl.sin(th_L)
            cos_L = tl.cos(th_L)
            sin_R = tl.sin(th_R)
            cos_R = tl.cos(th_R)

            s_L = 0.010 * (1.0 - cos_L)
            s_R = 0.010 * (1.0 - cos_R)
            ds_L = 0.010 * sin_L * omega_w
            ds_R = 0.010 * sin_R * omega_w

            xv_L = a_lead * cos_L + rev_cmd * b_max * sin_L
            xv_R = a_lead * cos_R + rev_cmd * b_max * sin_R

            a_adm_Lh = b_p * tl.minimum(tl.maximum(xv_L - es, 0.0), wp)
            a_exh_Lh = b_p * tl.minimum(tl.maximum(-xv_L - ee, 0.0), wp)
            a_adm_Lc = b_p * tl.minimum(tl.maximum(-xv_L - es, 0.0), wp)
            a_exh_Lc = b_p * tl.minimum(tl.maximum(xv_L - ee, 0.0), wp)

            a_adm_Rh = b_p * tl.minimum(tl.maximum(xv_R - es, 0.0), wp)
            a_exh_Rh = b_p * tl.minimum(tl.maximum(-xv_R - ee, 0.0), wp)
            a_adm_Rc = b_p * tl.minimum(tl.maximum(-xv_R - es, 0.0), wp)
            a_exh_Rc = b_p * tl.minimum(tl.maximum(xv_R - ee, 0.0), wp)

            t_sh_k = T_sh + 273.15
            grt_dt = gamma_s * r_steam * t_sh_k * dt
            a_drn = tl.where(drain_open, 3.6e-7, 0.0)
            subcool = tl.maximum(_sat_t_tl(P_chest) - T_cyl - 6.0, 0.0)
            cond_frac = tl.minimum(0.024 * subcool, 0.85)
            in_gas_fac = 1.0 - 0.5 * cond_frac
            p_lift_thr = P_chest + 14000.0
            p_cap_max = tl.maximum(P_chest + 38000.0, p_atm + 52000.0)

            # L_head
            vg_Lh = tl.maximum(v_clear + a_head * s_L - Mc_Lh / 980.0, 0.14 * v_clear)
            p_ad_Lh = P_Lh * tl.minimum(tl.maximum(1.0 - (gamma_s * a_head * ds_L * dt) / vg_Lh, 0.35), 2.8)
            m_in_Lh = tl.minimum(_orifice_tl(P_chest, p_ad_Lh, t_sh_k, a_adm_Lh, 0.68), tl.maximum(P_chest - p_ad_Lh, 0.0) * vg_Lh / (grt_dt * in_gas_fac))
            m_out_Lh = tl.minimum(_orifice_tl(p_ad_Lh, P_exh, t_sh_k, a_exh_Lh, 0.68), tl.maximum(p_ad_Lh - P_exh, 0.0) * vg_Lh / grt_dt)
            m_drn_Lh = tl.minimum(_orifice_tl(p_ad_Lh, p_atm, t_sh_k, a_drn, 0.66), tl.maximum(p_ad_Lh - p_atm, 0.0) * vg_Lh / grt_dt)
            pn_Lh = tl.maximum(p_ad_Lh + (grt_dt / vg_Lh) * (in_gas_fac * m_in_Lh - m_out_Lh - m_drn_Lh), 0.52 * p_atm)
            lf_Lh = tl.minimum(1.1e-5 * tl.maximum(pn_Lh - p_lift_thr, 0.0), 0.50)
            mr_Lh = tl.minimum(_orifice_tl(pn_Lh, P_chest, t_sh_k, 1.8e-5 * lf_Lh, 0.62), tl.maximum(pn_Lh - p_lift_thr, 0.0) * vg_Lh / grt_dt)
            P_Lh = tl.minimum(p_cap_max, pn_Lh - (grt_dt / vg_Lh) * mr_Lh)

            # L_crank
            vg_Lc = tl.maximum(v_clear + a_crank * (0.020 - s_L) - Mc_Lc / 980.0, 0.14 * v_clear)
            p_ad_Lc = P_Lc * tl.minimum(tl.maximum(1.0 - (gamma_s * (-a_crank * ds_L) * dt) / vg_Lc, 0.35), 2.8)
            m_in_Lc = tl.minimum(_orifice_tl(P_chest, p_ad_Lc, t_sh_k, a_adm_Lc, 0.68), tl.maximum(P_chest - p_ad_Lc, 0.0) * vg_Lc / (grt_dt * in_gas_fac))
            m_out_Lc = tl.minimum(_orifice_tl(p_ad_Lc, P_exh, t_sh_k, a_exh_Lc, 0.68), tl.maximum(p_ad_Lc - P_exh, 0.0) * vg_Lc / grt_dt)
            m_drn_Lc = tl.minimum(_orifice_tl(p_ad_Lc, p_atm, t_sh_k, a_drn, 0.66), tl.maximum(p_ad_Lc - p_atm, 0.0) * vg_Lc / grt_dt)
            pn_Lc = tl.maximum(p_ad_Lc + (grt_dt / vg_Lc) * (in_gas_fac * m_in_Lc - m_out_Lc - m_drn_Lc), 0.52 * p_atm)
            lf_Lc = tl.minimum(1.1e-5 * tl.maximum(pn_Lc - p_lift_thr, 0.0), 0.50)
            mr_Lc = tl.minimum(_orifice_tl(pn_Lc, P_chest, t_sh_k, 1.8e-5 * lf_Lc, 0.62), tl.maximum(pn_Lc - p_lift_thr, 0.0) * vg_Lc / grt_dt)
            P_Lc = tl.minimum(p_cap_max, pn_Lc - (grt_dt / vg_Lc) * mr_Lc)

            # R_head
            vg_Rh = tl.maximum(v_clear + a_head * s_R - Mc_Rh / 980.0, 0.14 * v_clear)
            p_ad_Rh = P_Rh * tl.minimum(tl.maximum(1.0 - (gamma_s * a_head * ds_R * dt) / vg_Rh, 0.35), 2.8)
            m_in_Rh = tl.minimum(_orifice_tl(P_chest, p_ad_Rh, t_sh_k, a_adm_Rh, 0.68), tl.maximum(P_chest - p_ad_Rh, 0.0) * vg_Rh / (grt_dt * in_gas_fac))
            m_out_Rh = tl.minimum(_orifice_tl(p_ad_Rh, P_exh, t_sh_k, a_exh_Rh, 0.68), tl.maximum(p_ad_Rh - P_exh, 0.0) * vg_Rh / grt_dt)
            m_drn_Rh = tl.minimum(_orifice_tl(p_ad_Rh, p_atm, t_sh_k, a_drn, 0.66), tl.maximum(p_ad_Rh - p_atm, 0.0) * vg_Rh / grt_dt)
            pn_Rh = tl.maximum(p_ad_Rh + (grt_dt / vg_Rh) * (in_gas_fac * m_in_Rh - m_out_Rh - m_drn_Rh), 0.52 * p_atm)
            lf_Rh = tl.minimum(1.1e-5 * tl.maximum(pn_Rh - p_lift_thr, 0.0), 0.50)
            mr_Rh = tl.minimum(_orifice_tl(pn_Rh, P_chest, t_sh_k, 1.8e-5 * lf_Rh, 0.62), tl.maximum(pn_Rh - p_lift_thr, 0.0) * vg_Rh / grt_dt)
            P_Rh = tl.minimum(p_cap_max, pn_Rh - (grt_dt / vg_Rh) * mr_Rh)

            # R_crank
            vg_Rc = tl.maximum(v_clear + a_crank * (0.020 - s_R) - Mc_Rc / 980.0, 0.14 * v_clear)
            p_ad_Rc = P_Rc * tl.minimum(tl.maximum(1.0 - (gamma_s * (-a_crank * ds_R) * dt) / vg_Rc, 0.35), 2.8)
            m_in_Rc = tl.minimum(_orifice_tl(P_chest, p_ad_Rc, t_sh_k, a_adm_Rc, 0.68), tl.maximum(P_chest - p_ad_Rc, 0.0) * vg_Rc / (grt_dt * in_gas_fac))
            m_out_Rc = tl.minimum(_orifice_tl(p_ad_Rc, P_exh, t_sh_k, a_exh_Rc, 0.68), tl.maximum(p_ad_Rc - P_exh, 0.0) * vg_Rc / grt_dt)
            m_drn_Rc = tl.minimum(_orifice_tl(p_ad_Rc, p_atm, t_sh_k, a_drn, 0.66), tl.maximum(p_ad_Rc - p_atm, 0.0) * vg_Rc / grt_dt)
            pn_Rc = tl.maximum(p_ad_Rc + (grt_dt / vg_Rc) * (in_gas_fac * m_in_Rc - m_out_Rc - m_drn_Rc), 0.52 * p_atm)
            lf_Rc = tl.minimum(1.1e-5 * tl.maximum(pn_Rc - p_lift_thr, 0.0), 0.50)
            mr_Rc = tl.minimum(_orifice_tl(pn_Rc, P_chest, t_sh_k, 1.8e-5 * lf_Rc, 0.62), tl.maximum(pn_Rc - p_lift_thr, 0.0) * vg_Rc / grt_dt)
            P_Rc = tl.minimum(p_cap_max, pn_Rc - (grt_dt / vg_Rc) * mr_Rc)

            in_sum = m_in_Lh + m_in_Lc + m_in_Rh + m_in_Rc
            exh_sum = m_out_Lh + m_out_Lc + m_out_Rh + m_out_Rc
            rel_sum = mr_Lh + mr_Lc + mr_Rh + mr_Rc
            dp_sc = (r_steam * t_sh_k / 2.8e-5) * (mdot_reg + rel_sum - in_sum)
            P_chest = tl.minimum(tl.maximum(P_chest + dp_sc * dt, p_atm), tl.maximum(P_boil, p_atm + 35000.0))

            # 20-7 烟室乏汽喷嘴 (内喷口 ø2.20mm, A=3.80mm²) 可压缩背压
            t_exh_k = tl.maximum(T_cyl + 273.15, 295.0)
            m_blast_unc = _orifice_tl(P_exh, p_atm, t_exh_k, 3.80e-6, 0.72)
            m_blast_cap = exh_sum + tl.maximum(P_exh - p_atm, 0.0) * 2.8e-6 / (r_steam * t_exh_k * dt)
            m_blast_out = tl.minimum(m_blast_unc, m_blast_cap)
            dp_exh = (r_steam * t_exh_k / 2.8e-6) * (exh_sum - m_blast_out)
            P_exh = tl.minimum(tl.maximum(P_exh + dp_exh * dt, p_atm), p_atm + 2.5 * pa_kgf)
            max_p_exh_kgf = tl.maximum(max_p_exh_kgf, (P_exh - p_atm) / pa_kgf)

            drn_sink = tl.where(drain_open, 0.004, 0.0)
            c_max = 3.0 * v_clear * 980.0
            Mc_Lh = tl.minimum(tl.maximum(Mc_Lh + (3.2 * cond_frac * m_in_Lh - drn_sink - 0.0001 * m_out_Lh) * dt, 0.0), c_max)
            Mc_Lc = tl.minimum(tl.maximum(Mc_Lc + (3.2 * cond_frac * m_in_Lc - drn_sink - 0.0001 * m_out_Lc) * dt, 0.0), c_max)
            Mc_Rh = tl.minimum(tl.maximum(Mc_Rh + (3.2 * cond_frac * m_in_Rh - drn_sink - 0.0001 * m_out_Rh) * dt, 0.0), c_max)
            Mc_Rc = tl.minimum(tl.maximum(Mc_Rc + (3.2 * cond_frac * m_in_Rc - drn_sink - 0.0001 * m_out_Rc) * dt, 0.0), c_max)
            mdot_cond_tot = 3.2 * cond_frac * in_sum

            # 气缸 + 蒸汽室 + 7-9 前置裸铜给汽管 (A_ext ≈ 148 cm²) 迎风对流+辐射全时段冷缸散热
            h_cyl_ext = 7.5 + 18.5 * tl.sqrt(v_air_rel + 0.05)
            tk_c = T_cyl + 273.15
            tk_a = T_amb + 273.15
            q_cyl_ext_w = 0.0148 * (h_cyl_ext * (T_cyl - T_amb) + 4.8198e-8 * (tk_c * tk_c * tk_c * tk_c - tk_a * tk_a * tk_a * tk_a))
            q_cyl_in_w = tl.minimum(mdot_cond_tot * 2.10e6, 920.0) + 2.2 * tl.maximum(T_sh - T_cyl, 0.0) * tl.where(in_sum > 1.0e-6, 1.0, 0.0)
            T_cyl = tl.minimum(tl.maximum(T_cyl + ((q_cyl_in_w - q_cyl_ext_w) / 48.0) * dt * act_f, -10.0), 210.0)

            # 4-2 Roscoe 置换注油器真实冷凝沉水排油 (~60 min / 3.2ml, FP64 累加防 ULP 吞没)
            t_sat_lub = _sat_t_tl(P_chest)
            h_lub_ext = 8.5 + 14.0 * tl.sqrt(v_air_rel + 0.04)
            q_lub_cond_w = (0.34 * h_lub_ext * 0.00185 + 0.015) * (0.38 + 0.62 * tl.minimum(in_sum / 1.4e-4, 1.0)) * tl.maximum(t_sat_lub - T_amb - 4.0, 0.0)
            dv_oil_ml = tl.where(
                (P_chest > p_atm + 2500.0) & (lube_oil_ml > 1.0e-5),
                ((q_lub_cond_w / 2.12e6) / 965.0) * 1.0e6 * dt,
                0.0,
            ) * act_f
            lube_next_d = lube_oil_ml_d - dv_oil_ml.to(tl.float64)
            lube_oil_ml_d = tl.where(lube_next_d > 0.0, lube_next_d, 0.0)
            lube_pct = 100.0 * lube_oil_ml_d.to(tl.float32) / 3.20

            # 活塞推力 + Couette 油膜黏滞剪切 + 无条件 A-稳定半隐式轮轨蠕滑黏着动力学
            lube_mult = tl.where(lube_pct > 2.0, 1.0, 2.8) * seal_drag_mult
            mu_cyl_film = 0.42e-3 * tl.exp(1180.0 / tl.maximum(T_cyl + 115.0, 125.0))
            c_couette = tl.minimum((mu_cyl_film * 1.15e-4) / 0.85e-6, 12.5)
            f_gas_L = (P_Lh - p_atm) * a_head - (P_Lc - p_atm) * a_crank
            f_gas_R = (P_Rh - p_atm) * a_head - (P_Rc - p_atm) * a_crank
            f_seal_L = lube_mult * (1.57 * tl_libdevice.tanh(ds_L * 80.0) + (c_couette + 0.35) * ds_L)
            f_seal_R = lube_mult * (1.57 * tl_libdevice.tanh(ds_R * 80.0) + (c_couette + 0.35) * ds_R)
            tau_steam = ((f_gas_L - f_seal_L) * sin_L + (f_gas_R - f_seal_R) * sin_R) * 0.010

            dp_sc_pos = tl.maximum(P_chest - p_atm, 0.0)
            mu_val = tl.where(lube_pct > 2.0, 0.042, 0.135)
            tau_valve = 2.0 * mu_val * (3.4e-5 * dp_sc_pos) * 0.0024 * tl_libdevice.tanh(omega_w / 0.5)

            q_pump_ideal = 8.04248e-6 * 0.0065 * (tl.abs(omega_w) / 6.2831853) * 0.86
            byp_divert = tl.minimum(tl_libdevice.pow(tl.maximum(byp_cmd, 1.0e-6), 1.30) * tl.where(byp_cmd > 1.0e-4, 1.0, 0.0), 1.0)
            q_fw_m3s = tl.where(tdr_w > 0.2, q_pump_ideal * (1.0 - byp_divert), 0.0)
            used_tdr_ml = tl.minimum(tdr_w, q_fw_m3s * 1.0e6 * dt) * act_f
            tdr_next_d = tdr_w_d - used_tdr_ml.to(tl.float64)
            tdr_w_d = tl.where(tdr_next_d > 0.0, tdr_next_d, 0.0)
            mdot_fw = (used_tdr_ml / dt) * 0.997e-3
            tau_pump = tl.where(used_tdr_ml > 1.0e-8, (tl.maximum(P_boil - p_atm, 8000.0) * 8.04248e-6) * 0.0022 * (1.0 - 0.75 * byp_divert), 0.0)

            m_loco_now = 3.88 + M_water
            m_tdr_now = 1.48 + (tdr_w * 0.997 + (alc_ml + sump_ml) * 0.787) * 1.0e-3
            m_total = m_loco_now + m_tdr_now + 2.15 + n_cars * 0.85 + extra_load_kg
            cos_gr = tl.cos(theta_grade)
            sin_gr = tl.sin(theta_grade)
            w_drivers_n = (0.76 * (m_loco_now + 1.35)) * 9.81 * cos_gr

            i_wheel = 0.0036
            r_drv = 0.0219
            sgn_om = tl.where(omega_w >= 0.0, 1.0, -1.0)
            # 含 3D 多体树 24 处华氏/耦合连杆销套黏性耗散与 28 处轮轨滚动微阻尼等效折算轴矩
            tau_drive = tau_steam - tau_valve - tau_pump * sgn_om - 0.0038 * omega_w - 0.00025 * omega_w * tl.abs(omega_w)
            omega_free = omega_w + (tau_drive / i_wheel) * dt
            v_slip_prev = omega_w * r_drv - v_ground
            wet_pen = tl.where(drain_open, 0.50, 0.0) + tl.minimum(0.03 * m_blast_out * 1000.0, 0.08)
            mu_peak = 0.33 * adh_mult * (1.0 - tl.minimum(wet_pen, 0.54)) * (0.72 + 0.28 * tl.exp(-tl.abs(v_slip_prev) / 0.32))
            f_adh_max = mu_peak * w_drivers_n
            v_creep_scale = 0.014 * tl.maximum(tl.abs(v_ground), 0.06)
            k_creep = f_adh_max / v_creep_scale
            v_slip_free = omega_free * r_drv - v_ground
            denom_imp = 1.0 + k_creep * dt * ((r_drv * r_drv) / i_wheel + 1.0 / m_total)
            v_slip_imp = v_slip_free / denom_imp
            f_rail_tract = f_adh_max * tl_libdevice.tanh(v_slip_imp / v_creep_scale)
            omega_w = tl.where(act, omega_free - (f_rail_tract * r_drv / i_wheel) * dt, 0.0)

            brg_visc = tl.exp(920.0 * (1.0 / tl.maximum(T_amb + 273.15, 255.0) - 0.003354016))
            f_roll = (0.0028 + 0.00009 * n_cars) * (m_total * 9.81 * cos_gr) * tl_libdevice.tanh(v_ground / 0.006)
            f_brg = 0.048 * (m_total / 6.0) * brg_visc * v_ground
            rho_air = 101325.0 / (287.05 * tl.maximum(T_amb + 273.15, 250.0))
            v_rel_a = v_ground + v_wind
            f_aero = 0.5 * rho_air * (0.82 + 0.16 * n_cars) * 0.0112 * v_rel_a * tl.abs(v_rel_a)
            f_curve = (0.0096 * curv_abs) * (m_total * 9.81) * tl_libdevice.tanh(v_ground / 0.008)
            f_grade = m_total * 9.81 * sin_gr

            accel_x = (f_rail_tract - f_roll - f_brg - f_aero - f_curve - f_grade) / m_total
            v_ground = tl.where(act, v_ground + accel_x * dt, 0.0)
            static_hold = (tl.abs(v_ground) < 0.0008) & (tl.abs(f_rail_tract - f_grade) < 0.0028 * m_total * 9.81) & (reg_cmd < 0.01)
            v_ground = tl.where(static_hold, 0.0, v_ground)
            omega_w = tl.where(static_hold & (tl.abs(omega_w) < 0.04), 0.0, omega_w)

            theta_next = theta_w + omega_w * dt
            theta_w = theta_next - 6.2831853 * tl_libdevice.floor(theta_next * 0.159154943)
            ds_f32 = v_ground * dt * act_f
            dist_m_d = dist_m_d + tl.abs(ds_f32).to(tl.float64)
            wheel_dist_m_d = wheel_dist_m_d + (tl.abs(omega_w * r_drv * dt) * act_f).to(tl.float64)
            s_track_d = s_track_d + ds_f32.to(tl.float64)
            has_moved_f = tl.where((has_moved_f > 0.5) | (v_ground > 0.06), 1.0, 0.0)

            v_rim_new = tl.abs(omega_w * r_drv)
            slip_now = tl.where(v_rim_new > 0.08, 100.0 * tl.minimum(tl.maximum((v_rim_new - tl.abs(v_ground)) / tl.maximum(v_rim_new, 0.22), 0.0), 1.0), 0.0)
            peak_slip_pct = tl.maximum(peak_slip_pct, slip_now * act_f)

            # 通风/安全阀、甲醇灯芯燃烧与 Type-"C" 锅炉相变热平衡 (对标 smokebox_lut.json 与 20min/130ml 真值)
            sv_open_f = tl.where(p_kgf >= 4.00, 1.0, sv_open_f)
            sv_open_f = tl.where(p_kgf <= 3.56, 0.0, sv_open_f)
            mdot_blw = _orifice_tl(P_boil, p_atm, T_boil + 273.15, 0.16e-6 * blw_cmd, 0.68) * has_st_f
            mdot_sv = _orifice_tl(P_boil, p_atm, T_boil + 273.15, tl.where(sv_open_f > 0.5, 3.6e-6, 0.0), 0.68) * has_st_f

            draft_idx = tl.where(chimney_fan, 1.15, 0.0) + 18000.0 * mdot_blw + 2600.0 * m_blast_out + 0.06 * v_air_rel
            draft_eff = tl.minimum(tl.maximum(0.20 + 0.92 * tl_libdevice.tanh(draft_idx), 0.20), 1.32)

            sump_norm = tl.minimum(tl.maximum(sump_ml / 3.5, 0.0), 1.6)
            supply_gate = sump_norm / (sump_norm + 0.04)
            wick_head = tl.minimum(tl.maximum(1.0 - 0.34 * (sin_gr / tl.maximum(cos_gr, 0.2)), 0.72), 1.25)
            vap_atten = 1.0 - 0.35 * tl.maximum((T_alc - 59.5) / 8.0, 0.0)
            oxy_dens_gain = 298.15 / tl.maximum(T_amb + 273.15, 255.0)
            cap_v_gain = tl.minimum(tl.maximum((0.84 + 0.16 * visc_gain) * oxy_dens_gain * vap_atten, 0.65), 1.42)
            burn_ml_s = tl.where(
                burner_lit & (sump_ml > 1.0e-5),
                supply_gate * (0.66 + 0.34 * draft_eff) * 0.148 * cap_v_gain * wick_head * wick_mult,
                0.0,
            )
            used_alc = tl.minimum(sump_ml, burn_ml_s * dt) * act_f
            sump_next_d = sump_ml_d - used_alc.to(tl.float64)
            sump_ml_d = tl.where(sump_next_d > 0.0, sump_next_d, 0.0)
            fuel_burned_ml_d = fuel_burned_ml_d + used_alc.to(tl.float64)
            q_chem_w = 19.93e6 * ((used_alc / dt) * rho_meoh * 1.0e-6)

            t_liner_tgt = tl.where(burner_lit, T_boil + 375.0 + 145.0 * tl.minimum(draft_eff, 1.2), 0.55 * T_crown + 0.45 * T_amb)
            T_liner = T_liner + tl.minimum(0.22 * dt, 1.0) * (t_liner_tgt - T_liner) * act_f
            eta_boil = tl.minimum(tl.maximum((0.225 + 0.115 * tl_libdevice.tanh(draft_idx)) * (1.0 + 0.16 * tl.minimum(tl.maximum((T_liner - 320.0) / 280.0, 0.0), 1.0)), 0.18), 0.46)
            q_comb_w = q_chem_w * eta_boil

            t_flue_sh = T_boil + tl.where(burner_lit, 36.0 * draft_eff, 0.0)
            g_sh = 4.2 * (0.25 + 0.75 * draft_eff)
            mcp_st = mdot_reg * 2080.0
            eps_sh = 1.0 - tl.exp(-tl.minimum((2.2 + 4600.0 * mdot_reg) / tl.maximum(mcp_st, 1.0e-5), 12.0))
            q_sh_st = tl.where(mcp_st > 1.0e-5, eps_sh * mcp_st * (T_sh_wall - T_boil), 0.22 * (T_sh_wall - T_boil))
            T_sh_wall = tl.minimum(tl.maximum(T_sh_wall + ((g_sh * (t_flue_sh - T_sh_wall) - q_sh_st) / 7.0) * dt * act_f, -10.0), 360.0)
            T_sh_tgt = tl.where(mcp_st > 1.0e-5, T_boil + eps_sh * (T_sh_wall - T_boil), 0.35 * T_boil + 0.65 * T_sh_wall)
            T_sh = T_sh + tl.minimum(2.5 * dt, 1.0) * (T_sh_tgt - T_sh) * act_f

            h_cas = 6.2 + 9.5 * tl.sqrt(v_air_rel + 0.04)
            q_cold_air_w = tl.where(burner_lit, 1.45 * draft_eff * tl.maximum(25.0 - T_amb, 0.0), 0.0)
            q_shell_loss_w = 0.048 * (0.76 * (T_boil - T_amb) / (0.068 + 1.0 / h_cas) + 0.24 * (T_boil - T_amb) / (0.0042 + 1.0 / h_cas)) + q_cold_air_w
            q_net_water_w = q_comb_w - tl.maximum(q_sh_st, 0.0) - q_shell_loss_w

            mdot_steam_out = mdot_reg + mdot_blw + mdot_sv
            h_fg = tl.maximum(2.501e6 - 2370.0 * T_boil, 1.8e6)
            boil_gate = tl.minimum(tl.maximum((T_boil - 99.5) / 0.7, 0.0), 1.0)
            wet_gate = tl.minimum(tl.maximum(M_water / 0.0012, 0.0), 1.0)
            phase_c = boil_gate * wet_gate
            v_gas = tl.maximum(3.80e-4 - M_water / 960.0, 2.0e-5)

            p_s1 = _sat_p_tl(T_boil)
            p_s2 = _sat_p_tl(T_boil + 0.1)
            drho_dT = ((p_s2 / (r_steam * (T_boil + 273.25))) - (p_s1 / (r_steam * (T_boil + 273.15)))) / 0.1
            c_eff = M_water * 4186.0 + 227.92 + phase_c * v_gas * h_fg * drho_dT
            dT_dt = (q_net_water_w - phase_c * mdot_steam_out * h_fg - mdot_fw * 4186.0 * (T_boil - T_amb)) / tl.maximum(c_eff, 80.0)
            T_boil = tl.minimum(tl.maximum(T_boil + dT_dt * dt * act_f, -5.0), 215.0)
            T_crown = T_boil + tl.where(water_ml > 135.0, 16.0 * draft_eff, 180.0 * (1.0 - tl.minimum(tl.maximum(water_ml / 135.0, 0.0), 1.0)))

            rho_sat = _sat_p_tl(T_boil) / (r_steam * (T_boil + 273.15))
            vap_def = tl.maximum((rho_sat * v_gas - M_vap) / 0.18, 0.0)
            evap_eq = tl.maximum(wet_gate * tl.maximum(mdot_steam_out + v_gas * drho_dT * dT_dt + vap_def, 0.0), boil_gate * tl.maximum(q_net_water_w, 0.0) / h_fg)
            evap_kg_s = tl.where((M_water > 1.0e-6) & (boil_gate > 0.01), tl.minimum(M_water / dt + mdot_fw, evap_eq), 0.0)
            mw_next_d = M_water_d + ((mdot_fw - evap_kg_s) * dt * act_f).to(tl.float64)
            M_water_d = tl.where(mw_next_d > 0.350, 0.350, tl.where(mw_next_d > 0.0, mw_next_d, 0.0))
            evap_total_g_d = evap_total_g_d + (evap_kg_s * 1000.0 * dt * act_f).to(tl.float64)

            v_gas_n = tl.maximum(3.80e-4 - M_water_d.to(tl.float32) / 960.0, 2.0e-5)
            M_vap = tl.maximum((p_atm * v_gas_n) / (r_steam * (T_boil + 273.15)), M_vap + (evap_kg_s - mdot_steam_out) * dt * act_f)
            p_sat_n = tl.maximum(p_atm, _sat_p_tl(T_boil))
            m_vap_sat = (p_sat_n * v_gas_n) / (r_steam * (T_boil + 273.15))
            M_vap = tl.where(phase_c > 0.02, M_vap + tl.minimum(12.0 * phase_c * dt, 1.0) * (m_vap_sat - M_vap), M_vap)
            p_gas_r = (M_vap * r_steam * (T_boil + 273.15)) / v_gas_n
            P_boil = tl.maximum(p_atm, phase_c * p_sat_n + (1.0 - phase_c) * p_gas_r)

            stopped = tl.abs(v_ground) < 0.012
            stall_cond = (has_moved_f > 0.5) & can_open_reg & (tl.abs(v_ground) < 0.015) & (~fuel_empty) & (~low_water_cut)
            stall_timer = tl.where(stall_cond & act, stall_timer + dt, 0.0)

            done_fuel = act & fuel_empty & stopped
            done_water = act & low_water_cut & stopped
            done_stall = act & (stall_timer >= 8.0)
            newly_done = done_fuel | done_water | done_stall
            stop_time_s = tl.where(newly_done, t_cur, stop_time_s)
            r_code = tl.where(done_fuel, 1.0, tl.where(done_water, 2.0, 4.0))
            stop_reason_f = tl.where(newly_done, r_code, stop_reason_f)
            active_f = tl.where(newly_done, 0.0, active_f)

        # 一次性写回全局显存
        tl.store(sb + 0, T_boil, mask=mask)
        tl.store(sb + 1, T_crown, mask=mask)
        tl.store(sb + 2, T_liner, mask=mask)
        tl.store(sb + 3, M_water_d.to(tl.float32), mask=mask)
        tl.store(sb + 4, P_boil, mask=mask)
        tl.store(sb + 5, M_vap, mask=mask)
        tl.store(sb + 6, P_chest, mask=mask)
        tl.store(sb + 7, P_exh, mask=mask)
        tl.store(sb + 8, T_sh, mask=mask)
        tl.store(sb + 9, T_sh_wall, mask=mask)
        tl.store(sb + 10, T_cyl, mask=mask)
        tl.store(sb + 11, T_alc, mask=mask)
        tl.store(sb + 12, alc_ml_d.to(tl.float32), mask=mask)
        tl.store(sb + 13, sump_ml_d.to(tl.float32), mask=mask)
        tl.store(sb + 14, tdr_w_d.to(tl.float32), mask=mask)
        tl.store(sb + 15, lube_oil_ml_d.to(tl.float32), mask=mask)
        tl.store(sb + 16, sv_open_f, mask=mask)
        tl.store(sb + 17, P_Lh, mask=mask)
        tl.store(sb + 18, P_Lc, mask=mask)
        tl.store(sb + 19, P_Rh, mask=mask)
        tl.store(sb + 20, P_Rc, mask=mask)
        tl.store(sb + 21, Mc_Lh, mask=mask)
        tl.store(sb + 22, Mc_Lc, mask=mask)
        tl.store(sb + 23, Mc_Rh, mask=mask)
        tl.store(sb + 24, Mc_Rc, mask=mask)
        tl.store(sb + 25, theta_w, mask=mask)
        tl.store(sb + 26, omega_w, mask=mask)
        tl.store(sb + 27, v_ground, mask=mask)
        tl.store(sb + 28, dist_m_d.to(tl.float32), mask=mask)
        tl.store(sb + 29, wheel_dist_m_d.to(tl.float32), mask=mask)
        tl.store(sb + 30, s_track_d.to(tl.float32), mask=mask)
        tl.store(sb + 31, spd_int, mask=mask)
        tl.store(sb + 32, has_moved_f, mask=mask)
        tl.store(sb + 33, active_f, mask=mask)
        tl.store(sb + 34, stop_time_s, mask=mask)
        tl.store(sb + 35, stop_reason_f, mask=mask)
        tl.store(sb + 36, stall_timer, mask=mask)
        tl.store(sb + 37, steam_up_time_s, mask=mask)
        tl.store(sb + 38, peak_slip_pct, mask=mask)
        tl.store(sb + 39, max_p_exh_kgf, mask=mask)
        tl.store(sb + 40, evap_total_g_d.to(tl.float32), mask=mask)
        tl.store(sb + 41, fuel_burned_ml_d.to(tl.float32), mask=mask)


# ==============================================================================
# 4. 纯 NumPy 向量化 FP64 CPU 回退核 (供无 CUDA 的 GitHub Actions CI 快速运行)
# ==============================================================================

def _sat_p_np(T_c: np.ndarray) -> np.ndarray:
    tk = np.clip(T_c + 273.15, 273.16, 640.0)
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
    return 22.064e6 * np.exp(ln_pi)


def _sat_t_np(P_pa: np.ndarray) -> np.ndarray:
    p_bar = np.clip(P_pa * 1e-5, 0.20, 25.0)
    return np.clip(1730.63 / (8.07131 - np.log10(p_bar * 750.062)) - 233.426, 20.0, 225.0)


def _orifice_np(p_up: np.ndarray, p_dn: np.ndarray, temp_k: np.ndarray, area: np.ndarray, cd: float) -> np.ndarray:
    dp = np.maximum(p_up - p_dn, 0.0)
    p_eff = np.maximum(p_dn, p_up * 0.545728)
    ratio = np.clip(p_eff / np.maximum(p_up, 1000.0), 0.01, 1.0)
    psi = np.sqrt(8.6666667 * np.maximum((ratio ** 1.5384615) - (ratio ** 1.7692308), 0.0))
    smooth = np.minimum(dp / 1500.0, 1.0)
    mdot = cd * area * p_up * psi / np.sqrt(461.5 * np.maximum(temp_k, 290.0))
    return np.where((area > 1e-12) & (dp > 0.0), mdot * smooth, 0.0)


def _rollout_cpu_numpy(st: np.ndarray, cfg: np.ndarray, n_steps: int, dt: float) -> None:
    T_boil, T_crown, T_liner, M_water, P_boil, M_vap, P_chest, P_exh = [st[:, i].copy() for i in range(8)]
    T_sh, T_sh_wall, T_cyl, T_alc, alc_ml, sump_ml, tdr_w, lube_oil_ml, sv_open_f = [st[:, i].copy() for i in range(8, 17)]
    P_cyl = st[:, 17:21].copy()
    M_cond = st[:, 21:25].copy()
    theta_w, omega_w, v_ground, dist_m, wheel_dist_m, s_track, spd_int = [st[:, i].copy() for i in range(25, 32)]
    has_moved_f, active_f, stop_time_s, stop_reason_f, stall_timer, steam_up_time_s = [st[:, i].copy() for i in range(32, 38)]
    peak_slip_pct, max_p_exh_kgf, evap_total_g, fuel_burned_ml = [st[:, i].copy() for i in range(38, 42)]

    is_hot = cfg[:, 0]
    ai_on = cfg[:, 1] > 0.5
    reg_manual, rev_manual, byp_manual, blw_manual = cfg[:, 2], cfg[:, 3], cfg[:, 4], cfg[:, 5]
    n_cars, extra_load_kg, T_amb, v_wind, extra_grade_rad, target_kmh, mode_code = [cfg[:, i] for i in range(6, 13)]
    seal_drag_mult, wick_mult, adh_mult = cfg[:, 13], cfg[:, 14], cfg[:, 15]

    p_atm, pa_kgf, r_steam, gamma_s = 101325.0, 98066.5, 461.5, 1.30
    a_head, a_crank, v_clear = 1.767146e-4, 1.696460e-4, 4.24115e-7
    es, ee, wp, b_p, a_lead, b_max = 0.001175, 0.000024, 0.00112, 0.0055, 0.0011809, 0.001872
    L_oval = 27.707963

    for k in range(n_steps):
        if k % 500 == 0 and not np.any(active_f > 0.5):
            break
        t_cur = (k + 1) * dt
        act = active_f > 0.5
        act_f = act.astype(np.float64)

        p_kgf = np.maximum((P_boil - p_atm) / pa_kgf, 0.0)
        steam_up_time_s = np.where((steam_up_time_s < 0.0) & (p_kgf >= 2.20), t_cur, steam_up_time_s)
        water_ml = M_water * 1000.0
        water_pct = 100.0 * (water_ml - 140.0) / 140.0
        cond_tot_g = M_cond.sum(axis=1) * 1000.0
        scale_kmh = np.abs(v_ground) * 108.0

        fuel_empty = (alc_ml + sump_ml) <= 0.08
        low_water_cut = (water_ml <= 115.0) & (tdr_w <= 0.5)
        burner_lit = (~fuel_empty) & (~low_water_cut)
        chimney_fan = (p_kgf < 1.25) & (np.abs(omega_w) < 4.0) & burner_lit
        need_drain_ai = (T_cyl < 100.0) | (cond_tot_g > 0.28)
        drain_open = np.where(ai_on, need_drain_ai, (T_cyl < 88.0) & (t_cur < 8.0))

        can_open_reg = np.where(is_hot > 0.5, True, steam_up_time_s >= 0.0) & (~fuel_empty) & (~low_water_cut)
        spd_ratio = np.clip(scale_kmh / 42.0, 0.0, 1.0)
        rev_ai = 0.84 - 0.30 * spd_ratio
        v_err = target_kmh - scale_kmh
        spd_int = np.where(can_open_reg & act, np.clip(spd_int + v_err * dt, -68.0, 38.0), spd_int)
        reg_ai = np.clip(0.24 + 0.018 * v_err + 0.0048 * spd_int, 0.0, 0.92)
        reg_ai = np.where(need_drain_ai, np.minimum(reg_ai, 0.42), reg_ai)
        reg_cmd = np.where(can_open_reg, np.where(ai_on, reg_ai, reg_manual), 0.0)
        rev_cmd = np.where(ai_on, rev_ai, rev_manual)
        blw_ai = np.clip(0.95 * (3.72 - p_kgf) - np.minimum(scale_kmh * 0.006, 0.35) + 0.12, 0.0, 1.0)
        blw_cmd = np.where(burner_lit, np.where(ai_on, blw_ai, blw_manual), 0.0)
        byp_ai = np.clip(0.42 + 0.065 * (water_pct - 50.0) - 0.50 * np.clip(reg_cmd * 0.35, 0.0, 0.4), 0.0, 1.0)
        byp_cmd = np.where(ai_on, byp_ai, byp_manual)
        fuel_valve = np.where(p_kgf > 3.92, np.maximum(1.0 - 2.4 * (p_kgf - 3.92), 0.35), 1.0)

        s_mod = np.mod(s_track, L_oval)
        u_ov = (math.pi / L_oval) * (s_mod - 3.0)
        dz_ds = 0.102 * (4.0 * math.pi / L_oval) * (np.sin(u_ov) ** 3) * np.cos(u_ov)
        slope_3d = np.where(mode_code > 1.5, np.arctan(dz_ds), 0.0)
        in_ov_turn = ((s_mod >= 6.0) & (s_mod < 13.85398)) | (s_mod >= 19.85398)
        curv_abs = np.where(mode_code > 1.5, np.where(in_ov_turn, 0.40, 0.0), np.where(mode_code > 0.5, 0.50, 0.0))
        theta_grade = np.clip(slope_3d + extra_grade_rad, -0.25, 0.25)

        v_air_rel = np.abs(v_ground + v_wind)
        q_alc_rad = np.where(burner_lit, 0.028, 0.005) * np.maximum(0.22 * T_crown + 0.35 * T_boil - T_alc, 0.0)
        q_alc_cool = (0.19 + 0.32 * np.minimum(v_air_rel, 2.5)) * (T_alc - T_amb)
        T_alc = np.clip(T_alc + ((q_alc_rad - q_alc_cool) / 18.0) * dt * act_f, -5.0, 64.2)
        mu_meoh = 0.544 * np.exp(1265.0 * (1.0 / (T_alc + 273.15) - 1.0 / 298.15))
        rho_meoh = 786.6 * (1.0 - 0.00119 * (T_alc - 25.0))
        visc_gain = np.clip(0.544 / np.maximum(mu_meoh, 0.22), 0.65, 1.42)

        refill_ml = np.minimum(np.minimum(alc_ml, np.maximum(3.5 - sump_ml, 0.0)), 0.55 * fuel_valve * visc_gain * dt) * act_f
        alc_ml = np.maximum(alc_ml - refill_ml, 0.0)
        sump_ml += refill_ml

        has_st_f = (P_boil > (p_atm + 400.0)).astype(np.float64)
        a_reg = 3.2e-6 * (np.clip(reg_cmd, 0.0, 1.0) ** 1.35)
        mdot_reg_raw = _orifice_np(P_boil, P_chest, T_boil + 273.15, a_reg, 0.68) * has_st_f
        rho_sh_in = np.maximum(P_boil / (r_steam * (T_boil + 273.15)), 0.5)
        u_sh_pipe = mdot_reg_raw / (7.60265e-6 * rho_sh_in + 1e-9)
        dp_sh_pipe = 8.24727 * 0.5 * rho_sh_in * (u_sh_pipe ** 2)
        p_boil_eff = np.maximum(P_boil - dp_sh_pipe, p_atm)
        mdot_reg_unc = _orifice_np(p_boil_eff, P_chest, T_boil + 273.15, a_reg, 0.68) * has_st_f
        mdot_reg_cap = np.maximum(p_boil_eff - P_chest, 0.0) * 2.8e-5 / (r_steam * (T_sh + 273.15) * dt)
        mdot_reg = np.minimum(mdot_reg_unc, mdot_reg_cap * 1.35)

        th_L, th_R = theta_w, theta_w - 0.5 * math.pi
        sin_L, cos_L, sin_R, cos_R = np.sin(th_L), np.cos(th_L), np.sin(th_R), np.cos(th_R)
        s_L, s_R = 0.010 * (1.0 - cos_L), 0.010 * (1.0 - cos_R)
        ds_L, ds_R = 0.010 * sin_L * omega_w, 0.010 * sin_R * omega_w
        xv_L = a_lead * cos_L + rev_cmd * b_max * sin_L
        xv_R = a_lead * cos_R + rev_cmd * b_max * sin_R

        a_adm = np.stack([b_p * np.clip(xv_L - es, 0.0, wp), b_p * np.clip(-xv_L - es, 0.0, wp), b_p * np.clip(xv_R - es, 0.0, wp), b_p * np.clip(-xv_R - es, 0.0, wp)], axis=1)
        a_exh = np.stack([b_p * np.clip(-xv_L - ee, 0.0, wp), b_p * np.clip(xv_L - ee, 0.0, wp), b_p * np.clip(-xv_R - ee, 0.0, wp), b_p * np.clip(xv_R - ee, 0.0, wp)], axis=1)
        v_ch = np.stack([v_clear + a_head * s_L, v_clear + a_crank * (0.020 - s_L), v_clear + a_head * s_R, v_clear + a_crank * (0.020 - s_R)], axis=1)
        dv_ch = np.stack([a_head * ds_L, -a_crank * ds_L, a_head * ds_R, -a_crank * ds_R], axis=1)

        t_sh_k = (T_sh + 273.15)[:, None]
        grt_dt = gamma_s * r_steam * t_sh_k * dt
        p_sc_4, p_ex_4 = P_chest[:, None], P_exh[:, None]
        a_drn = np.where(drain_open, 3.6e-7, 0.0)[:, None]
        cond_frac = np.clip(0.024 * np.maximum(_sat_t_np(P_chest) - T_cyl - 6.0, 0.0), 0.0, 0.85)[:, None]
        in_gas_fac = 1.0 - 0.5 * cond_frac

        v_gas_c = np.maximum(v_ch - M_cond / 980.0, 0.14 * v_clear)
        p_ad = P_cyl * np.clip(1.0 - (gamma_s * dv_ch * dt) / v_gas_c, 0.35, 2.8)
        m_in = np.minimum(_orifice_np(p_sc_4, p_ad, t_sh_k, a_adm, 0.68), np.maximum(p_sc_4 - p_ad, 0.0) * v_gas_c / (grt_dt * in_gas_fac))
        m_out = np.minimum(_orifice_np(p_ad, p_ex_4, t_sh_k, a_exh, 0.68), np.maximum(p_ad - p_ex_4, 0.0) * v_gas_c / grt_dt)
        m_drn = np.minimum(_orifice_np(p_ad, p_atm, t_sh_k, a_drn, 0.66), np.maximum(p_ad - p_atm, 0.0) * v_gas_c / grt_dt)
        p_next = np.maximum(p_ad + (grt_dt / v_gas_c) * (in_gas_fac * m_in - m_out - m_drn), 0.52 * p_atm)
        p_lift_thr = p_sc_4 + 14000.0
        lift_mm = np.clip(1.1e-5 * np.maximum(p_next - p_lift_thr, 0.0), 0.0, 0.50)
        m_rel = np.minimum(_orifice_np(p_next, p_sc_4, t_sh_k, 1.8e-5 * lift_mm, 0.62), np.maximum(p_next - p_lift_thr, 0.0) * v_gas_c / grt_dt)
        P_cyl = np.minimum(np.maximum(p_sc_4 + 38000.0, p_atm + 52000.0), p_next - (grt_dt / v_gas_c) * m_rel)

        in_sum, exh_sum, rel_sum = m_in.sum(axis=1), m_out.sum(axis=1), m_rel.sum(axis=1)
        P_chest = np.clip(P_chest + ((r_steam * (T_sh + 273.15) / 2.8e-5) * (mdot_reg + rel_sum - in_sum)) * dt, p_atm, np.maximum(P_boil, p_atm + 35000.0))
        t_exh_k = np.maximum(T_cyl + 273.15, 295.0)
        m_blast_unc = _orifice_np(P_exh, p_atm, t_exh_k, np.full_like(P_exh, 3.80e-6), 0.72)
        m_blast_cap = exh_sum + np.maximum(P_exh - p_atm, 0.0) * 2.8e-6 / (r_steam * t_exh_k * dt)
        m_blast_out = np.minimum(m_blast_unc, m_blast_cap)
        P_exh = np.clip(P_exh + ((r_steam * t_exh_k / 2.8e-6) * (exh_sum - m_blast_out)) * dt, p_atm, p_atm + 2.5 * pa_kgf)
        max_p_exh_kgf = np.maximum(max_p_exh_kgf, (P_exh - p_atm) / pa_kgf)

        M_cond = np.clip(M_cond + (3.2 * cond_frac * m_in - np.where(drain_open[:, None], 0.004, 0.0001 * m_out)) * dt, 0.0, 3.0 * v_clear * 980.0)
        mdot_cond_tot = (3.2 * cond_frac * m_in).sum(axis=1)

        h_cyl_ext = 7.5 + 18.5 * np.sqrt(v_air_rel + 0.05)
        q_cyl_ext_w = 0.0148 * (h_cyl_ext * (T_cyl - T_amb) + 4.8198e-8 * (((T_cyl + 273.15) ** 4) - ((T_amb + 273.15) ** 4)))
        q_cyl_in_w = np.minimum(mdot_cond_tot * 2.10e6, 920.0) + 2.2 * np.maximum(T_sh - T_cyl, 0.0) * (in_sum > 1e-6)
        T_cyl = np.clip(T_cyl + ((q_cyl_in_w - q_cyl_ext_w) / 48.0) * dt * act_f, -10.0, 210.0)

        t_sat_lub = _sat_t_np(P_chest)
        q_lub_cond_w = (0.34 * (8.5 + 14.0 * np.sqrt(v_air_rel + 0.04)) * 0.00185 + 0.015) * (0.38 + 0.62 * np.minimum(in_sum / 1.4e-4, 1.0)) * np.maximum(t_sat_lub - T_amb - 4.0, 0.0)
        dv_oil_ml = np.where((P_chest > p_atm + 2500.0) & (lube_oil_ml > 1e-5), ((q_lub_cond_w / 2.12e6) / 965.0) * 1e6 * dt, 0.0) * act_f
        lube_oil_ml = np.maximum(lube_oil_ml - dv_oil_ml, 0.0)
        lube_pct = 100.0 * lube_oil_ml / 3.20

        lube_mult = np.where(lube_pct > 2.0, 1.0, 2.8) * seal_drag_mult
        mu_cyl_film = 0.42e-3 * np.exp(1180.0 / np.maximum(T_cyl + 115.0, 125.0))
        c_couette = np.minimum((mu_cyl_film * 1.15e-4) / 0.85e-6, 12.5)
        f_gas_L = (P_cyl[:, 0] - p_atm) * a_head - (P_cyl[:, 1] - p_atm) * a_crank
        f_gas_R = (P_cyl[:, 2] - p_atm) * a_head - (P_cyl[:, 3] - p_atm) * a_crank
        f_seal_L = lube_mult * (1.57 * np.tanh(ds_L * 80.0) + (c_couette + 0.35) * ds_L)
        f_seal_R = lube_mult * (1.57 * np.tanh(ds_R * 80.0) + (c_couette + 0.35) * ds_R)
        tau_steam = ((f_gas_L - f_seal_L) * sin_L + (f_gas_R - f_seal_R) * sin_R) * 0.010

        tau_valve = 2.0 * np.where(lube_pct > 2.0, 0.042, 0.135) * (3.4e-5 * np.maximum(P_chest - p_atm, 0.0)) * 0.0024 * np.tanh(omega_w / 0.5)
        q_pump_ideal = 8.04248e-6 * 0.0065 * (np.abs(omega_w) / (2.0 * math.pi)) * 0.86
        byp_divert = np.clip(byp_cmd ** 1.30, 0.0, 1.0)
        used_tdr_ml = np.minimum(tdr_w, np.where(tdr_w > 0.2, q_pump_ideal * (1.0 - byp_divert), 0.0) * 1e6 * dt) * act_f
        tdr_w = np.maximum(tdr_w - used_tdr_ml, 0.0)
        mdot_fw = (used_tdr_ml / dt) * 0.997e-3
        tau_pump = np.where(used_tdr_ml > 1e-8, (np.maximum(P_boil - p_atm, 8000.0) * 8.04248e-6) * 0.0022 * (1.0 - 0.75 * byp_divert), 0.0)

        m_loco_now = 3.88 + M_water
        m_total = m_loco_now + 1.48 + 2.15 + (tdr_w * 0.997 + (alc_ml + sump_ml) * 0.787) * 1e-3 + n_cars * 0.85 + extra_load_kg
        cos_gr, sin_gr = np.cos(theta_grade), np.sin(theta_grade)
        w_drivers_n = 0.76 * (m_loco_now + 1.35) * 9.81 * cos_gr
        i_wheel, r_drv = 0.0036, 0.0219
        sgn_om = np.where(omega_w >= 0.0, 1.0, -1.0)
        tau_drive = tau_steam - tau_valve - tau_pump * sgn_om - 0.0038 * omega_w - 0.00025 * omega_w * np.abs(omega_w)
        omega_free = omega_w + (tau_drive / i_wheel) * dt
        v_slip_prev = omega_w * r_drv - v_ground
        wet_pen = np.where(drain_open, 0.50, 0.0) + np.minimum(0.03 * m_blast_out * 1000.0, 0.08)
        mu_peak = 0.33 * adh_mult * (1.0 - np.minimum(wet_pen, 0.54)) * (0.72 + 0.28 * np.exp(-np.abs(v_slip_prev) / 0.32))
        f_adh_max = mu_peak * w_drivers_n
        v_creep_scale = 0.014 * np.maximum(np.abs(v_ground), 0.06)
        k_creep = f_adh_max / v_creep_scale
        v_slip_free = omega_free * r_drv - v_ground
        v_slip_imp = v_slip_free / (1.0 + k_creep * dt * ((r_drv * r_drv) / i_wheel + 1.0 / m_total))
        f_rail_tract = f_adh_max * np.tanh(v_slip_imp / v_creep_scale)
        omega_w = np.where(act, omega_free - (f_rail_tract * r_drv / i_wheel) * dt, 0.0)

        brg_visc = np.exp(920.0 * (1.0 / np.maximum(T_amb + 273.15, 255.0) - 1.0 / 298.15))
        f_roll = (0.0028 + 0.00009 * n_cars) * (m_total * 9.81 * cos_gr) * np.tanh(v_ground / 0.006)
        f_brg = 0.048 * (m_total / 6.0) * brg_visc * v_ground
        v_rel_a = v_ground + v_wind
        f_aero = 0.5 * (101325.0 / (287.05 * np.maximum(T_amb + 273.15, 250.0))) * (0.82 + 0.16 * n_cars) * 0.0112 * v_rel_a * np.abs(v_rel_a)
        f_curve = (0.0096 * curv_abs) * (m_total * 9.81) * np.tanh(v_ground / 0.008)
        f_grade = m_total * 9.81 * sin_gr

        v_ground = np.where(act, v_ground + ((f_rail_tract - f_roll - f_brg - f_aero - f_curve - f_grade) / m_total) * dt, 0.0)
        static_hold = (np.abs(v_ground) < 0.0008) & (np.abs(f_rail_tract - f_grade) < 0.0028 * m_total * 9.81) & (reg_cmd < 0.01)
        v_ground = np.where(static_hold, 0.0, v_ground)
        omega_w = np.where(static_hold & (np.abs(omega_w) < 0.04), 0.0, omega_w)

        theta_w = np.mod(theta_w + omega_w * dt, 2.0 * math.pi)
        ds_step = v_ground * dt * act_f
        dist_m += np.abs(ds_step)
        wheel_dist_m += np.abs(omega_w * r_drv * dt) * act_f
        s_track += ds_step
        has_moved_f = np.where((has_moved_f > 0.5) | (v_ground > 0.06), 1.0, 0.0)
        v_rim_new = np.abs(omega_w * r_drv)
        peak_slip_pct = np.maximum(peak_slip_pct, np.where(v_rim_new > 0.08, 100.0 * np.clip((v_rim_new - np.abs(v_ground)) / np.maximum(v_rim_new, 0.22), 0.0, 1.0), 0.0) * act_f)

        sv_open_f = np.where(p_kgf >= 4.00, 1.0, np.where(p_kgf <= 3.56, 0.0, sv_open_f))
        mdot_blw = _orifice_np(P_boil, p_atm, T_boil + 273.15, 0.16e-6 * blw_cmd, 0.68) * has_st_f
        mdot_sv = _orifice_np(P_boil, p_atm, T_boil + 273.15, np.where(sv_open_f > 0.5, 3.6e-6, 0.0), 0.68) * has_st_f
        draft_idx = np.where(chimney_fan, 1.15, 0.0) + 18000.0 * mdot_blw + 2600.0 * m_blast_out + 0.06 * v_air_rel
        draft_eff = np.clip(0.20 + 0.92 * np.tanh(draft_idx), 0.20, 1.32)

        sump_norm = np.clip(sump_ml / 3.5, 0.0, 1.6)
        oxy_dens_gain = 298.15 / np.maximum(T_amb + 273.15, 255.0)
        cap_v_gain = np.clip((0.84 + 0.16 * visc_gain) * oxy_dens_gain * (1.0 - 0.35 * np.maximum((T_alc - 59.5) / 8.0, 0.0)), 0.65, 1.42)
        burn_ml_s = np.where(burner_lit & (sump_ml > 1e-5), (sump_norm / (sump_norm + 0.04)) * (0.66 + 0.34 * draft_eff) * 0.148 * cap_v_gain * np.clip(1.0 - 0.34 * np.tan(theta_grade), 0.72, 1.25) * wick_mult, 0.0)
        used_alc = np.minimum(sump_ml, burn_ml_s * dt) * act_f
        sump_ml = np.maximum(sump_ml - used_alc, 0.0)
        fuel_burned_ml += used_alc
        q_chem_w = 19.93e6 * ((used_alc / dt) * rho_meoh * 1e-6)

        T_liner += min(0.22 * dt, 1.0) * (np.where(burner_lit, T_boil + 375.0 + 145.0 * np.minimum(draft_eff, 1.2), 0.55 * T_crown + 0.45 * T_amb) - T_liner) * act_f
        eta_boil = np.clip((0.225 + 0.115 * np.tanh(draft_idx)) * (1.0 + 0.16 * np.clip((T_liner - 320.0) / 280.0, 0.0, 1.0)), 0.18, 0.46)
        q_comb_w = q_chem_w * eta_boil

        mcp_st = mdot_reg * 2080.0
        eps_sh = 1.0 - np.exp(-np.minimum((2.2 + 4600.0 * mdot_reg) / np.maximum(mcp_st, 1e-5), 12.0))
        q_sh_st = np.where(mcp_st > 1e-5, eps_sh * mcp_st * (T_sh_wall - T_boil), 0.22 * (T_sh_wall - T_boil))
        T_sh_wall = np.clip(T_sh_wall + ((4.2 * (0.25 + 0.75 * draft_eff) * (T_boil + np.where(burner_lit, 36.0 * draft_eff, 0.0) - T_sh_wall) - q_sh_st) / 7.0) * dt * act_f, -10.0, 360.0)
        T_sh += min(2.5 * dt, 1.0) * (np.where(mcp_st > 1e-5, T_boil + eps_sh * (T_sh_wall - T_boil), 0.35 * T_boil + 0.65 * T_sh_wall) - T_sh) * act_f

        h_cas = 6.2 + 9.5 * np.sqrt(v_air_rel + 0.04)
        q_cold_air_w = np.where(burner_lit, 1.45 * draft_eff * np.maximum(25.0 - T_amb, 0.0), 0.0)
        q_shell_loss_w = 0.048 * (0.76 * (T_boil - T_amb) / (0.068 + 1.0 / h_cas) + 0.24 * (T_boil - T_amb) / (0.0042 + 1.0 / h_cas)) + q_cold_air_w
        q_net_water_w = q_comb_w - np.maximum(q_sh_st, 0.0) - q_shell_loss_w

        mdot_steam_out = mdot_reg + mdot_blw + mdot_sv
        h_fg = np.maximum(2.501e6 - 2370.0 * T_boil, 1.8e6)
        boil_gate = np.clip((T_boil - 99.5) / 0.7, 0.0, 1.0)
        wet_gate = np.clip(M_water / 0.0012, 0.0, 1.0)
        phase_c = boil_gate * wet_gate
        v_gas = np.maximum(3.80e-4 - M_water / 960.0, 2e-5)

        drho_dT = ((_sat_p_np(T_boil + 0.1) / (r_steam * (T_boil + 273.25))) - (_sat_p_np(T_boil) / (r_steam * (T_boil + 273.15)))) / 0.1
        dT_dt = (q_net_water_w - phase_c * mdot_steam_out * h_fg - mdot_fw * 4186.0 * (T_boil - T_amb)) / np.maximum(M_water * 4186.0 + 227.92 + phase_c * v_gas * h_fg * drho_dT, 80.0)
        T_boil = np.clip(T_boil + dT_dt * dt * act_f, -5.0, 215.0)
        T_crown = T_boil + np.where(water_ml > 135.0, 16.0 * draft_eff, 180.0 * (1.0 - np.clip(water_ml / 135.0, 0.0, 1.0)))

        rho_sat = _sat_p_np(T_boil) / (r_steam * (T_boil + 273.15))
        evap_eq = np.maximum(wet_gate * np.maximum(mdot_steam_out + v_gas * drho_dT * dT_dt + np.maximum((rho_sat * v_gas - M_vap) / 0.18, 0.0), 0.0), boil_gate * np.maximum(q_net_water_w, 0.0) / h_fg)
        evap_kg_s = np.where((M_water > 1e-6) & (boil_gate > 0.01), np.minimum(M_water / dt + mdot_fw, evap_eq), 0.0)
        M_water = np.clip(M_water + (mdot_fw - evap_kg_s) * dt * act_f, 0.0, 0.350)
        evap_total_g += evap_kg_s * 1000.0 * dt * act_f

        v_gas_n = np.maximum(3.80e-4 - M_water / 960.0, 2e-5)
        M_vap = np.maximum((p_atm * v_gas_n) / (r_steam * (T_boil + 273.15)), M_vap + (evap_kg_s - mdot_steam_out) * dt * act_f)
        p_sat_n = np.maximum(p_atm, _sat_p_np(T_boil))
        M_vap = np.where(phase_c > 0.02, M_vap + np.minimum(12.0 * phase_c * dt, 1.0) * ((p_sat_n * v_gas_n) / (r_steam * (T_boil + 273.15)) - M_vap), M_vap)
        P_boil = np.maximum(p_atm, phase_c * p_sat_n + (1.0 - phase_c) * ((M_vap * r_steam * (T_boil + 273.15)) / v_gas_n))

        stopped = np.abs(v_ground) < 0.012
        stall_timer = np.where((has_moved_f > 0.5) & can_open_reg & (np.abs(v_ground) < 0.015) & (~fuel_empty) & (~low_water_cut) & act, stall_timer + dt, 0.0)
        done_fuel = act & fuel_empty & stopped
        done_water = act & low_water_cut & stopped
        done_stall = act & (stall_timer >= 8.0)
        newly_done = done_fuel | done_water | done_stall
        stop_time_s = np.where(newly_done, t_cur, stop_time_s)
        stop_reason_f = np.where(newly_done, np.where(done_fuel, 1.0, np.where(done_water, 2.0, 4.0)), stop_reason_f)
        active_f = np.where(newly_done, 0.0, active_f)

    st[:, 0:17] = np.column_stack([T_boil, T_crown, T_liner, M_water, P_boil, M_vap, P_chest, P_exh, T_sh, T_sh_wall, T_cyl, T_alc, alc_ml, sump_ml, tdr_w, lube_oil_ml, sv_open_f])
    st[:, 17:21] = P_cyl
    st[:, 21:25] = M_cond
    st[:, 25:42] = np.column_stack([theta_w, omega_w, v_ground, dist_m, wheel_dist_m, s_track, spd_int, has_moved_f, active_f, stop_time_s, stop_reason_f, stall_timer, steam_up_time_s, peak_slip_pct, max_p_exh_kgf, evap_total_g, fuel_burned_ml])


# ==============================================================================
# 5. RTX 5080 CUDA 批量曲柄角级全行程续航物理求解器 (GPUConsistEnduranceEngine)
# ==============================================================================

@dataclass
class EnduranceScenario:
    name: str
    track_mode: str = "oval_grade"          # "track" | "curve_r2m" | "oval_grade"
    preset: str = "hot"                     # "hot" | "cold"
    tank_spec_2010: bool = False            # False=130ml/272ml, True=200ml/320ml (2010 spec)
    ai_driver: bool = True                  # True=省汽闭环司炉, False=固定手操开度
    regulator: float = 0.48
    reverser: float = 0.65
    bypass_valve: float = 0.35
    blower_valve: float = 0.22
    trailing_cars: int = 0                  # 挂车节数 (每节干重 0.85kg)
    trailing_load_kg: float = 0.0           # 额外载重 kg
    ambient_temp_c: float = 25.0
    headwind_m_s: float = 0.0
    extra_grade_deg: float = 0.0
    target_speed_kmh: float = 52.0


class GPUConsistEnduranceEngine:
    """
    在本地 NVIDIA RTX 5080 上采用原生 @triton.jit 寄存器驻留核并行推进 B 列 D51 真蒸汽机车直至耗尽停稳。
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r_drv = 0.0219
        self.V_boiler_total = 3.80e-4

    @torch.no_grad()
    def simulate_batch(
        self,
        scenarios: List[EnduranceScenario],
        max_sim_s: float = 2400.0,
        dt: float = 0.0025,
        init_override: Optional[Dict[str, float]] = None,
        monte_carlo_spread: float = 0.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        dev = self.device
        B = len(scenarios)
        t_wall_0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        prof_oval = TrackProfile3D("oval_grade")
        prof_r2m = TrackProfile3D("curve_r2m")

        cfg_np = np.zeros((B, N_CFG), dtype=np.float32)
        st_np = np.zeros((B, N_STATE), dtype=np.float32)

        if monte_carlo_spread > 1e-6:
            seal_m = np.clip(1.0 + monte_carlo_spread * rng.standard_normal(B), 0.78, 1.28)
            wick_m = np.clip(1.0 + 0.65 * monte_carlo_spread * rng.standard_normal(B), 0.85, 1.16)
            adh_m = np.clip(1.0 + 0.70 * monte_carlo_spread * rng.standard_normal(B), 0.80, 1.18)
        else:
            seal_m = np.ones(B, dtype=np.float64)
            wick_m = np.ones(B, dtype=np.float64)
            adh_m = np.ones(B, dtype=np.float64)

        for i, sc in enumerate(scenarios):
            hot = sc.preset == "hot"
            t_amb = float(sc.ambient_temp_c)
            cfg_np[i, 0] = 1.0 if hot else 0.0
            cfg_np[i, 1] = 1.0 if sc.ai_driver else 0.0
            cfg_np[i, 2] = sc.regulator
            cfg_np[i, 3] = sc.reverser
            cfg_np[i, 4] = sc.bypass_valve
            cfg_np[i, 5] = sc.blower_valve
            cfg_np[i, 6] = float(sc.trailing_cars)
            cfg_np[i, 7] = sc.trailing_load_kg
            cfg_np[i, 8] = t_amb
            cfg_np[i, 9] = sc.headwind_m_s
            cfg_np[i, 10] = math.radians(sc.extra_grade_deg)
            cfg_np[i, 11] = sc.target_speed_kmh
            cfg_np[i, 12] = 0.0 if sc.track_mode == "track" else (1.0 if sc.track_mode == "curve_r2m" else 2.0)
            cfg_np[i, 13] = float(seal_m[i])
            cfg_np[i, 14] = float(wick_m[i])
            cfg_np[i, 15] = float(adh_m[i])

            tb0 = 149.6 if hot else t_amb
            pb0 = float(_sat_p_np(np.array([tb0]))[0]) if hot else P_ATM
            mw0 = 0.210
            vg0 = max(2e-5, self.V_boiler_total - mw0 / 960.0)
            alc_tot = 200.0 if sc.tank_spec_2010 else 130.0
            sump0 = 3.5 if hot else 3.2
            tdr_cap = 400.0 if sc.tank_spec_2010 else 340.0

            st_np[i, 0] = tb0
            st_np[i, 1] = 168.5 if hot else t_amb
            st_np[i, 2] = 545.0 if hot else t_amb
            st_np[i, 3] = mw0
            st_np[i, 4] = pb0
            st_np[i, 5] = (pb0 * vg0) / (R_STEAM * (tb0 + 273.15))
            st_np[i, 6] = P_ATM
            st_np[i, 7] = P_ATM
            st_np[i, 8] = 165.0 if hot else t_amb
            st_np[i, 9] = 172.0 if hot else t_amb
            st_np[i, 10] = 132.0 if hot else t_amb
            st_np[i, 11] = 31.5 if hot else t_amb
            st_np[i, 12] = alc_tot - sump0
            st_np[i, 13] = sump0
            st_np[i, 14] = 0.80 * tdr_cap
            st_np[i, 15] = 2.56
            st_np[i, 16] = 0.0
            st_np[i, 17:21] = P_ATM
            st_np[i, 21:25] = 1.5e-5 if hot else 8.0e-5
            st_np[i, 30] = prof_oval.s0_launch
            st_np[i, 33] = 1.0  # active
            st_np[i, 34] = max_sim_s
            st_np[i, 37] = 0.0 if hot else -1.0

        if init_override:
            if "T_boiler_c" in init_override:
                st_np[:, 0] = float(init_override["T_boiler_c"])
            if "P_boiler_pa" in init_override:
                st_np[:, 4] = float(init_override["P_boiler_pa"])
            if "M_water_kg" in init_override:
                st_np[:, 3] = float(init_override["M_water_kg"])
            if "tender_water_ml" in init_override:
                st_np[:, 14] = float(init_override["tender_water_ml"])
            if "alcohol_ml" in init_override:
                tot_alc = max(0.0, float(init_override["alcohol_ml"]))
                sump_v = min(3.5, tot_alc)
                st_np[:, 13] = sump_v
                st_np[:, 12] = max(0.0, tot_alc - sump_v)
            if "lube_oil_ml" in init_override:
                st_np[:, 15] = float(init_override["lube_oil_ml"])
            if "T_cyl_wall_c" in init_override:
                st_np[:, 10] = float(init_override["T_cyl_wall_c"])
            if "v_ground_ms" in init_override:
                vg_i = float(init_override["v_ground_ms"])
                st_np[:, 27] = vg_i
                st_np[:, 26] = vg_i / self.r_drv
                st_np[:, 32] = 1.0 if vg_i > 0.05 else 0.0
            if "s_track_m" in init_override:
                st_np[:, 30] = float(init_override["s_track_m"])

        n_total_steps = int(round(max_sim_s / dt))
        use_triton_gpu = HAS_TRITON and dev.type == "cuda"

        if use_triton_gpu:
            st_gpu = torch.from_numpy(st_np).to(dev, non_blocking=True).contiguous()
            cfg_gpu = torch.from_numpy(cfg_np).to(dev, non_blocking=True).contiguous()
            block_b = 64 if B <= 64 else 128
            grid = (triton.cdiv(B, block_b),)
            chunk_steps = min(8000, max(1, n_total_steps))
            step_done = 0
            while step_done < n_total_steps:
                cur_chunk = min(chunk_steps, n_total_steps - step_done)
                _rollout_chunk_triton_kernel[grid](
                    st_gpu,
                    cfg_gpu,
                    B,
                    cur_chunk,
                    float(step_done * dt),
                    float(dt),
                    BLOCK_B=block_b,
                )
                step_done += cur_chunk
                if step_done < n_total_steps and not bool(torch.any(st_gpu[:, 33] > 0.5).item()):
                    break
            torch.cuda.synchronize(dev)
            st_out = st_gpu.cpu().numpy()
        else:
            st_f64 = st_np.astype(np.float64)
            _rollout_cpu_numpy(st_f64, cfg_np.astype(np.float64), n_total_steps, float(dt))
            st_out = st_f64

        wall_s = time.perf_counter() - t_wall_0
        reason_names = {
            0: "MAX_TIME_REACHED",
            1: "FUEL_DEPLETED",
            2: "WATER_DEPLETED_CUTOFF",
            3: "BOILER_DRY",
            4: "GRADE_OR_PRESSURE_STALL",
        }

        results: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenarios):
            lap_len = (
                prof_oval.lap_length_m
                if sc.track_mode == "oval_grade"
                else (prof_r2m.lap_length_m if sc.track_mode == "curve_r2m" else 240.0)
            )
            d_m = float(st_out[i, 28])
            t_stop = float(st_out[i, 34])
            sup_s = max(0.0, float(st_out[i, 37]))
            results.append(
                {
                    "name": sc.name,
                    "track_mode": sc.track_mode,
                    "preset": sc.preset,
                    "tank_spec": "2010_200ml_400ml" if sc.tank_spec_2010 else "classic_130ml_340ml",
                    "trailing_cars": sc.trailing_cars,
                    "trailing_load_kg": sc.trailing_load_kg,
                    "ambient_temp_c": sc.ambient_temp_c,
                    "distance_m": round(d_m, 2),
                    "wheel_roll_m": round(float(st_out[i, 29]), 2),
                    "scale_distance_km": round(d_m * 30.0 / 1000.0, 3),
                    "laps": round(d_m / lap_len, 2),
                    "duration_min": round(t_stop / 60.0, 2),
                    "steam_up_min": round(sup_s / 60.0, 2),
                    "mean_running_speed_ms": round(d_m / max(1.0, t_stop - sup_s), 3),
                    "limiting_factor": reason_names.get(int(round(float(st_out[i, 35]))), "UNKNOWN"),
                    "fuel_burned_ml": round(float(st_out[i, 41]), 2),
                    "fuel_remaining_ml": round(float(st_out[i, 12] + st_out[i, 13]), 2),
                    "tender_water_remaining_ml": round(float(st_out[i, 14]), 2),
                    "boiler_water_remaining_ml": round(float(st_out[i, 3] * 1000.0), 2),
                    "lube_oil_remaining_pct": round(float(100.0 * st_out[i, 15] / 3.20), 2),
                    "total_steam_evap_g": round(float(st_out[i, 40]), 1),
                    "peak_wheel_slip_pct": round(float(st_out[i, 38]), 2),
                    "peak_exhaust_backpressure_kgf": round(float(st_out[i, 39]), 3),
                }
            )

        total_sim_s = float(np.sum(st_out[:, 34]))
        return {
            "device": f"{torch.cuda.get_device_name(0)} (Triton Register Kernel)" if use_triton_gpu else "CPU (Vectorized NumPy)",
            "batch_size": B,
            "dt_s": dt,
            "wall_time_s": round(wall_s, 4),
            "total_simulated_seconds": round(total_sim_s, 1),
            "effective_speedup_vs_realtime": round(total_sim_s / max(1e-5, wall_s), 1),
            "results": results,
        }

    @torch.no_grad()
    def predict_live_remaining_range(
        self,
        track_mode: str,
        live_state: Dict[str, float],
        ctrl_state: Dict[str, Any],
        n_ensemble: int = 32,
    ) -> Dict[str, Any]:
        sc_base = EnduranceScenario(
            name="live_rollout",
            track_mode=track_mode,
            preset="hot",
            tank_spec_2010=bool(ctrl_state.get("tank_spec_2010", False)),
            ai_driver=bool(ctrl_state.get("ai_driver", False)),
            regulator=max(0.35, float(ctrl_state.get("regulator", 0.48))) if float(ctrl_state.get("regulator", 0.0)) > 0.02 else 0.48,
            reverser=float(ctrl_state.get("reverser", 0.65)) if abs(float(ctrl_state.get("reverser", 0.65))) > 0.08 else 0.65,
            bypass_valve=float(ctrl_state.get("bypass_valve", 0.35)),
            blower_valve=float(ctrl_state.get("blower_valve", 0.22)),
            trailing_cars=int(ctrl_state.get("trailing_cars", 2 if track_mode == "oval_grade" else 0)),
            trailing_load_kg=float(ctrl_state.get("trailing_load_kg", 0.0)),
            ambient_temp_c=float(ctrl_state.get("ambient_temp_c", 25.0)),
            headwind_m_s=float(ctrl_state.get("headwind_m_s", 0.0)),
            extra_grade_deg=float(ctrl_state.get("grade_deg", 0.0)),
        )
        batch_scs = [sc_base for _ in range(max(4, n_ensemble))]
        out = self.simulate_batch(
            batch_scs,
            max_sim_s=2400.0,
            dt=0.004,
            init_override=live_state,
            monte_carlo_spread=0.07,
            seed=107,
        )
        dists = np.array([r["distance_m"] for r in out["results"]], dtype=np.float64)
        mins = np.array([r["duration_min"] for r in out["results"]], dtype=np.float64)
        factors = [r["limiting_factor"] for r in out["results"]]
        top_factor = max(set(factors), key=factors.count) if factors else "FUEL_DEPLETED"
        return {
            "p10_m": round(float(np.percentile(dists, 10)), 1),
            "p50_m": round(float(np.percentile(dists, 50)), 1),
            "p90_m": round(float(np.percentile(dists, 90)), 1),
            "p50_min": round(float(np.percentile(mins, 50)), 1),
            "limiting_factor": top_factor,
            "gpu_eval_ms": round(out["wall_time_s"] * 1000.0, 2),
        }
