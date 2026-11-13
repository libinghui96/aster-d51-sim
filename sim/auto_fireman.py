#!/usr/bin/env python3
"""
sim/auto_fireman.py
O6: Aster D51 自动司炉与驾驶控制器 (Auto-Fireman Agent)。
自动加载由本地 RTX 5080 大规模并行轨迹寻优标定的 configs/auto_fireman_policy.json，实现:
  1. 冷缸自动开排液阀暖缸防 Hydrolock 水锤，壁温达标后自动关阀省汽；
  2. 随车速上升自动收短华氏逆转机截止比 (Notching Up / 省汽膨胀巡航)；
  3. 通风阀与燃料阀协同控压，稳定维持 3.65~3.85 kgf/cm² 并抑制安全阀白白泄压；
  4. 轴动给水泵旁通阀前馈+PI 闭环控制，精准维持 50% 水位并防止高温干烧急冷闪蒸。
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from sim.thermo_1d import CabControls, ThermoTelemetry


@dataclass
class AutoFiremanPolicy:
    target_speed_kmh: float = 50.0
    p_target_kgf: float = 3.72
    cutoff_start: float = 0.85
    cutoff_cruise: float = 0.58
    cutoff_speed_span_kmh: float = 42.0
    kp_reg: float = 0.018
    ki_reg: float = 0.0045
    kp_blower: float = 0.95
    kp_bypass: float = 0.065
    ff_bypass_base: float = 0.42
    warmup_wall_temp_c: float = 102.0


class AutoFiremanAgent:
    """Aster D51 智能司炉状态机"""

    def __init__(self, policy_path: Optional[Path] = None, target_speed_kmh: Optional[float] = None):
        default_path = policy_path or (
            Path(__file__).resolve().parent.parent / "configs" / "auto_fireman_policy.json"
        )
        self.policy = AutoFiremanPolicy()
        if default_path.exists():
            raw = json.loads(default_path.read_text(encoding="utf-8"))
            p = raw.get("optimal_policy", raw)
            for k in self.policy.__dataclass_fields__:
                if k in p:
                    setattr(self.policy, k, float(p[k]))
        if target_speed_kmh is not None:
            self.policy.target_speed_kmh = target_speed_kmh

        self._speed_int = 0.0
        self._direction = 1.0

    def reset(self, direction: float = 1.0) -> None:
        self._speed_int = 0.0
        self._direction = 1.0 if direction >= 0.0 else -1.0

    def update(self, dt: float, tel: ThermoTelemetry, ctrl: CabControls) -> None:
        p = self.policy
        if abs(ctrl.reverser) > 0.05:
            self._direction = 1.0 if ctrl.reverser >= 0.0 else -1.0

        # 1. 点火与防超压燃料微调
        ctrl.burner_lit = tel.alcohol_ml > 0.2
        if tel.boiler_pressure_kgf > 3.90:
            ctrl.fuel_valve = max(0.35, 1.0 - 2.5 * (tel.boiler_pressure_kgf - 3.90))
        else:
            ctrl.fuel_valve = 1.0

        # 2. 烟室通风阀 (Blower) 稳压闭环: 低压助燃，高压及高速乏汽充足时关小防安全阀起跳
        p_err = p.p_target_kgf - tel.boiler_pressure_kgf
        exh_relief = min(0.35, tel.scale_speed_kmh * 0.006)
        ctrl.blower_valve = max(0.0, min(1.0, p.kp_blower * p_err - exh_relief + 0.12))
        ctrl.chimney_fan = tel.boiler_pressure_kgf < 0.8 and tel.wheel_rpm < 5.0

        # 3. 冷缸防 Hydrolock 水锤自动排液阀 (ドレン弁)
        need_drain = (
            tel.hydrolock_L
            or tel.hydrolock_R
            or tel.cyl_wall_temp_c < p.warmup_wall_temp_c
            or tel.condensate_g > 0.28
        )
        ctrl.drain_cock = bool(need_drain)

        # 4. 起压门限判断与省汽膨胀截止比 (Walschaerts Notching-Up) + 加减弁 PI 调速
        if tel.boiler_pressure_kgf < 1.6:
            ctrl.regulator = 0.0
            ctrl.reverser = self._direction * p.cutoff_start
            return

        spd_ratio = max(0.0, min(1.0, tel.scale_speed_kmh / max(10.0, p.cutoff_speed_span_kmh)))
        mag_rev = p.cutoff_start - (p.cutoff_start - p.cutoff_cruise) * spd_ratio
        ctrl.reverser = self._direction * mag_rev

        v_err = p.target_speed_kmh - tel.scale_speed_kmh
        self._speed_int = max(-18.0, min(35.0, self._speed_int + v_err * dt))
        reg_cmd = 0.28 + p.kp_reg * v_err + p.ki_reg * self._speed_int
        if need_drain:
            reg_cmd = min(reg_cmd, 0.42)
        ctrl.regulator = max(0.0, min(0.92, reg_cmd))

        # 5. 轴动给水泵旁通阀闭环稳 50% 水位 + 干烧防闪蒸限流保护
        # 优先采用补偿坡道后的真实静水位估计 (水面计平滑滤波)
        w_err = tel.true_water_level_pct - 50.0
        # w_err > 0 (水多) -> 开大旁通阀回流; w_err < 0 (水少) -> 关小旁通阀强迫进锅
        byp_cmd = p.ff_bypass_base + p.kp_bypass * w_err - 0.18 * tel.steam_gen_g_s
        if tel.crown_temp_c > 215.0 and tel.crown_submerged_mm < 1.5:
            # 高温干烧时渐进补水，避免冷急注触发闪蒸超压尖峰
            byp_cmd = max(0.42, byp_cmd)
        ctrl.bypass_valve = max(0.0, min(1.0, byp_cmd))

        # 若驻车或低速缺水 (<42%)，自动点按手摇给水泵补水
        if tel.true_water_level_pct < 42.0 and abs(tel.wheel_rpm) < 25.0 and tel.crown_temp_c < 210.0:
            ctrl.bypass_valve = 0.0
