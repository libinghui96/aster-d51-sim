#!/usr/bin/env python3
"""
sim/ideal_oracle_controller.py
P2-G 模式 2：理想全知极限多变量最优控制器 (IdealOracleController)。

突破人类生理与单手串行瓶颈的理论上限基准：
  1. 零延迟全状态感知 (Full-State Omniscience)：
     - 直接读取 L1+L2 内部真实静水位 true_water_level_pct、紫铜顶板浸没水深 crown_submerged_mm、
       气缸实时液态水克数 condensate_g、蒸汽干度 steam_dryness_chest 与实时蒸发率 steam_gen_g_s；
  2. 200 Hz 多输入多输出 (MIMO) 无级连续协同：
     - 同步无级联调加減弁、逆転機连续膨胀比、通風弁、36-6 燃料针阀、4-17 旁通阀与 8-6 排液阀；
     - 贴着安全阀起跳阈值下方 (3.93 kgf/cm²) 巡航实现最高朗肯循环热效率且 0.00s 安全阀泄压浪费；
     - 坡道水位前馈预补偿，彻底免疫上下坡假水位；
     - 仅在气缸冷凝水逼近水锤临界阈值 (>0.34g) 的微秒窗口脉冲排液，暖缸零活汽浪费。
"""

import math
from typing import Callable, Optional

from sim.thermo_1d import CabControls, ThermoTelemetry


class IdealOracleController:
    """零延迟全状态 200Hz MIMO 理论最优控制器"""

    def __init__(self, target_speed_kmh: float = 48.0, p_cruise_kgf: float = 3.92):
        self.target_speed_kmh = float(target_speed_kmh)
        self.p_cruise_kgf = float(p_cruise_kgf)
        self.reset()

    def reset(self, direction: float = 1.0) -> None:
        self._direction = 1.0 if direction >= 0.0 else -1.0
        self._v_int = 0.0
        self._pump_accum = 0.0
        self.status_summary = "理想最適制御 (Oracle): 全状態 200Hz MIMO 待機中"

    def update(
        self,
        dt: float,
        tel: ThermoTelemetry,
        ctrl: CabControls,
        hand_pump_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        if abs(ctrl.reverser) > 0.05:
            self._direction = 1.0 if ctrl.reverser >= 0.0 else -1.0

        # 严格锁定气密旋盖与火室门零漏风损耗
        ctrl.fuel_cap_open = False
        ctrl.firebox_door = 0.0
        ctrl.whistle_valve = 0.0

        # 1. 燃烧器与精确化学计量比控压 (紧贴 3.92~3.96 kgf/cm² 零安全阀起跳!)
        ctrl.burner_lit = tel.alcohol_ml > 0.1
        p_err = self.p_cruise_kgf - tel.boiler_pressure_kgf
        if tel.boiler_pressure_kgf > 3.94:
            ctrl.fuel_valve = max(0.28, 1.0 - 5.2 * (tel.boiler_pressure_kgf - 3.94))
        else:
            ctrl.fuel_valve = min(1.0, max(0.45, 0.85 + 0.35 * p_err))

        ctrl.chimney_fan = tel.boiler_pressure_kgf < 0.95 and abs(tel.wheel_rpm) < 15.0
        exh_draft_ff = min(0.55, 0.32 * tel.steam_gen_g_s)
        ctrl.blower_valve = max(0.0, min(1.0, 1.15 * p_err - exh_draft_ff + 0.08))
        if tel.boiler_pressure_kgf >= 3.95:
            ctrl.blower_valve = 0.0

        # 2. 精确基于内部液态水克数 condensate_g 的微脉冲防 Hydrolock 控制 (省去人类盲目长时间开排液阀的白白跑汽!)
        ctrl.drain_cock = bool(
            tel.hydrolock_L
            or tel.hydrolock_R
            or tel.condensate_g > 0.34
            or (tel.cyl_wall_temp_c < 72.0 and tel.condensate_g > 0.22)
        )

        # 3. 真实静水位 + 坡道顶板浸没深度前馈补偿 + 蒸发量精确质量守恒前馈
        grade_rad = math.radians(abs(ctrl.grade_deg))
        # 坡度越大自动提高目标真实静水位，确保火室紫铜顶板前/后最浅点永远 >= 9.5mm 覆水！
        true_lvl_target = min(64.0, 50.0 + 38.0 * math.tan(grade_rad))
        w_err_true = tel.true_water_level_pct - true_lvl_target
        byp_cmd = 0.40 + 0.095 * w_err_true - 0.24 * tel.steam_gen_g_s
        if tel.crown_temp_c > 210.0 and tel.crown_submerged_mm < 1.2:
            byp_cmd = max(0.40, byp_cmd)

        if w_err_true < -4.5 and abs(tel.wheel_rpm) < 28.0 and hand_pump_cb is not None:
            ctrl.bypass_valve = 0.0
            self._pump_accum += dt
            if self._pump_accum >= 0.22:
                self._pump_accum = 0.0
                hand_pump_cb(1)
        else:
            self._pump_accum = 0.0
            ctrl.bypass_valve = max(0.0, min(1.0, byp_cmd))

        # 4. 连续无级最优华氏膨胀截止比 (Stepless Cutoff Optimization) + 高带宽速度跟踪
        if tel.boiler_pressure_kgf < 1.35:
            ctrl.regulator = 0.0
            ctrl.reverser = self._direction * 0.88
            self.status_summary = f"理想最適制御 (Oracle): 最速昇圧中 ({tel.boiler_temp_c:.1f}°C | {tel.boiler_pressure_kgf:.2f} kgf/cm²)"
            return

        spd_norm = max(0.0, min(1.0, tel.scale_speed_kmh / max(12.0, self.target_speed_kmh)))
        optimal_cutoff = 0.88 - 0.36 * (spd_norm ** 0.82)
        ctrl.reverser = self._direction * optimal_cutoff

        v_err = self.target_speed_kmh - tel.scale_speed_kmh
        self._v_int = max(-15.0, min(30.0, self._v_int + v_err * dt))
        p_comp = 3.60 / max(1.35, tel.boiler_pressure_kgf)
        reg_cmd = (0.25 + 0.024 * v_err + 0.0062 * self._v_int) * p_comp
        if ctrl.drain_cock:
            reg_cmd = min(reg_cmd, 0.48)
        ctrl.regulator = max(0.0, min(0.96, reg_cmd))

        # 若锅水过高 (>78%) 挤压汽空间导致干度下降，自动微开后封头排污阀 11-16 降水位保干度
        ctrl.blowdown_valve = 0.35 if tel.true_water_level_pct > 78.0 else 0.0

        self.status_summary = (
            f"理想最適制御 (Oracle 200Hz): P={tel.boiler_pressure_kgf:.2f}kgf | "
            f"連続カットオフ {optimal_cutoff*100:.1f}% | 実水位 {tel.true_water_level_pct:.1f}%"
        )
