#!/usr/bin/env python3
"""
sim/human_engineer_agent.py
P2-G 模式 1：拟人实操蒸汽机车司机智能体 (HumanEngineerAgent)。

严格遵守人类操作 Aster Gauge-1 真蒸汽机车的生理与信息边界：
  1. 仅可观测表观量 (Partial Observability)：
     - 只能读取 PG-20 表盘读数 (0.05 kgf/cm² 刻度量化)、11-7 水面计【表观】水位 (受上下坡假水位欺骗！)、
       动轮转速/排气声节奏以及煤水车 31-14 回水喷流；绝不偷看 true_water_level_pct、T_crown_c、condensate_g 等内部隐状态！
  2. 感知—运动延迟 (0.40 s FIFO 延迟缓冲)：所有决策基于 0.40 秒前的视觉/听觉观测。
  3. 单手/双手注意力切换冷却与有限拨动速率 (Slew-Rate & Reach Cooldown)：
     - 在驾驶室后壁与煤水车手摇泵/燃料阀之间换位需 0.75 秒伸手冷却时间；
     - 8-16 螺旋逆转机摇柄最高 32%/s，且停靠离散档位 (±85%, ±72%, ±60%, 0%)；
     - 驻车或低速缺水时严格执行说明书 p.11 实操规程：「先关 4-17 旁通阀 -> 换手至煤水车以 1.8 Hz 逐次压动 31-5 手摇泵 -> 换回驾驶室重开旁通阀」。
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Deque, Optional, Tuple

from sim.thermo_1d import CabControls, ThermoTelemetry


@dataclass
class DelayedHumanSense:
    sim_time: float
    gauge_pressure_kgf: float
    apparent_water_pct: float
    scale_speed_kmh: float
    wheel_rpm: float
    safety_popping: bool
    chuff_stalled: bool
    alcohol_empty: bool


class HumanEngineerAgent:
    """受限于人类视觉观测、0.40s 反应延迟、换手冷却与螺旋手轮转速的拟人司机"""

    REACTION_DELAY_S = 0.40
    HAND_SWITCH_COOLDOWN_S = 0.75
    REVERSER_SLEW_PER_S = 0.32
    REGULATOR_SLEW_PER_S = 0.45
    BLOWER_SLEW_PER_S = 0.55
    BYPASS_SLEW_PER_S = 0.85
    HANDPUMP_PERIOD_S = 0.56  # ~1.8 Hz 人类单手压水节奏

    def __init__(self, target_speed_kmh: float = 48.0):
        self.target_speed_kmh = float(target_speed_kmh)
        self._sense_buf: Deque[DelayedHumanSense] = deque()
        self.reset()

    def reset(self, direction: float = 1.0) -> None:
        self._sense_buf.clear()
        self._direction = 1.0 if direction >= 0.0 else -1.0
        self._station = "cab"  # "cab" | "tender"
        self._switch_timer = 0.0
        self._warmup_steaming_s = 0.0
        self._handpump_phase = "idle"  # "idle" | "closing_bypass" | "pumping" | "reopening_bypass"
        self._pump_stroke_timer = 0.0
        self._speed_err_int = 0.0
        self.handpump_anim_rad = 0.0
        self.status_summary = "擬人機関士: 運転室計器確認中"

    @staticmethod
    def _slew(curr: float, target: float, max_rate: float, dt: float) -> float:
        diff = target - curr
        step = max_rate * dt
        if abs(diff) <= step:
            return target
        return curr + math.copysign(step, diff)

    def _switch_station(self, target_station: str) -> bool:
        if self._station == target_station:
            return self._switch_timer <= 0.0
        self._station = target_station
        self._switch_timer = self.HAND_SWITCH_COOLDOWN_S
        return False

    def _push_and_read_sense(self, tel: ThermoTelemetry, ctrl: CabControls) -> DelayedHumanSense:
        # 仅采集人类感官可见信号 (水面计为含坡道误差的 apparent water_level_pct!)
        quant_p = round(tel.boiler_pressure_kgf * 20.0) / 20.0
        stalled = (ctrl.regulator > 0.25 and tel.boiler_pressure_kgf > 1.2 and abs(tel.wheel_rpm) < 4.0) or (
            tel.hydrolock_L or tel.hydrolock_R
        )
        now_sense = DelayedHumanSense(
            sim_time=tel.sim_time,
            gauge_pressure_kgf=quant_p,
            apparent_water_pct=round(tel.water_level_pct, 1),
            scale_speed_kmh=tel.scale_speed_kmh,
            wheel_rpm=tel.wheel_rpm,
            safety_popping=tel.safety_valve_open,
            chuff_stalled=stalled,
            alcohol_empty=(tel.alcohol_ml <= 0.2),
        )
        self._sense_buf.append(now_sense)
        while len(self._sense_buf) > 1 and (tel.sim_time - self._sense_buf[0].sim_time) >= self.REACTION_DELAY_S:
            self._sense_buf.popleft()
        return self._sense_buf[0]

    def update(
        self,
        dt: float,
        tel: ThermoTelemetry,
        ctrl: CabControls,
        hand_pump_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        if abs(ctrl.reverser) > 0.08:
            self._direction = 1.0 if ctrl.reverser >= 0.0 else -1.0

        obs = self._push_and_read_sense(tel, ctrl)
        if self._switch_timer > 0.0:
            self._switch_timer = max(0.0, self._switch_timer - dt)

        # 确保煤水车 36-5 气密盖与 12-2 火室门闭合
        ctrl.fuel_cap_open = False
        ctrl.firebox_door = self._slew(ctrl.firebox_door, 0.0, 1.5, dt)

        # 0. 点火整备阶段 (若尚未点火且有酒精)
        if not ctrl.burner_lit and not obs.alcohol_empty:
            if not self._switch_station("tender"):
                self.status_summary = "擬人機関士: 炭水車 36-6 燃料弁へ手を移動中..."
                return
            ctrl.fuel_valve = self._slew(ctrl.fuel_valve, 0.88, 0.75, dt)
            if ctrl.fuel_valve >= 0.65:
                ctrl.burner_lit = True
                ctrl.chimney_fan = obs.gauge_pressure_kgf < 0.9
                self.status_summary = "擬人機関士: 23-1 バーナー点火・煙突通風機 ON"
            return

        # 1. 驻车/低速低水位时按说明书 p.11 执行「关旁通 -> 换手摇 31-5 手摇泵 -> 重开旁通」状态机
        if self._handpump_phase == "idle" and obs.apparent_water_pct < 38.0 and abs(obs.wheel_rpm) < 35.0:
            self._handpump_phase = "closing_bypass"

        if self._handpump_phase == "closing_bypass":
            if not self._switch_station("cab"):
                self.status_summary = "擬人機関士: 運転室バイパス弁へ手を戻し中..."
                return
            ctrl.bypass_valve = self._slew(ctrl.bypass_valve, 0.0, self.BYPASS_SLEW_PER_S, dt)
            self.status_summary = "擬人機関士: 手動給水前に 4-17 バイパス弁を全閉中"
            if ctrl.bypass_valve <= 0.02:
                self._handpump_phase = "pumping"
                self._pump_stroke_timer = 0.0
            return

        if self._handpump_phase == "pumping":
            if not self._switch_station("tender"):
                self.status_summary = "擬人機関士: 炭水車 37-12 手動ポンプ柄へ手を移動中..."
                return
            self._pump_stroke_timer += dt
            self.handpump_anim_rad = 0.42 * math.sin(2.0 * math.pi * self._pump_stroke_timer / self.HANDPUMP_PERIOD_S)
            if self._pump_stroke_timer >= self.HANDPUMP_PERIOD_S:
                self._pump_stroke_timer -= self.HANDPUMP_PERIOD_S
                if hand_pump_cb is not None:
                    hand_pump_cb(1)
            self.status_summary = f"擬人機関士: 31-5 手動ポンプ圧送中 (水面計 {obs.apparent_water_pct:.0f}%)"
            if obs.apparent_water_pct >= 53.0 or tel.tender_water_ml <= 2.0:
                self.handpump_anim_rad = 0.0
                self._handpump_phase = "reopening_bypass"
            return

        if self._handpump_phase == "reopening_bypass":
            if not self._switch_station("cab"):
                self.status_summary = "擬人機関士: 炭水車から運転室操作盤へ手を戻し中..."
                return
            ctrl.bypass_valve = self._slew(ctrl.bypass_valve, 0.42, self.BYPASS_SLEW_PER_S, dt)
            self.status_summary = "擬人機関士: 手動給水完了、4-17 バイパス弁を復帰中"
            if ctrl.bypass_valve >= 0.40:
                self._handpump_phase = "idle"
            return

        self.handpump_anim_rad = 0.0
        if not self._switch_station("cab"):
            self.status_summary = "擬人機関士: 運転室へ視線・手を移動中..."
            return

        # 2. 冷炉升压待命阶段 (PG-20 < 1.8 kgf/cm²)
        if obs.gauge_pressure_kgf < 1.0:
            ctrl.chimney_fan = True
            blw_tgt = 0.05
        else:
            ctrl.chimney_fan = False
            blw_tgt = max(0.0, min(0.85, 0.72 * (3.65 - obs.gauge_pressure_kgf)))

        if obs.safety_popping or obs.gauge_pressure_kgf >= 3.90:
            blw_tgt = 0.0

        ctrl.blower_valve = self._slew(ctrl.blower_valve, blw_tgt, self.BLOWER_SLEW_PER_S, dt)

        if obs.gauge_pressure_kgf < 1.75 and abs(obs.wheel_rpm) < 8.0:
            ctrl.regulator = self._slew(ctrl.regulator, 0.0, self.REGULATOR_SLEW_PER_S, dt)
            ctrl.reverser = self._slew(ctrl.reverser, self._direction * 0.85, self.REVERSER_SLEW_PER_S, dt)
            ctrl.drain_cock = True
            self._warmup_steaming_s = 0.0
            self.status_summary = f"擬人機関士: 昇圧待機中 (PG-20 {obs.gauge_pressure_kgf:.2f} kgf/cm²)"
            return

        # 3. 凭经验计时与排气声操作 8-6 排液阀拉杆 (不知精确缸壁温度与克数，至少暖缸运转 10 秒才推回排液杆)
        if abs(obs.wheel_rpm) > 12.0:
            self._warmup_steaming_s += dt
        if obs.chuff_stalled:
            self._warmup_steaming_s = 0.0
            ctrl.drain_cock = True
        elif self._warmup_steaming_s >= 10.5:
            ctrl.drain_cock = False
        else:
            ctrl.drain_cock = True

        # 4. 离散档位螺旋逆转机摇柄 (8-16) + 经验加减阀调速
        if obs.scale_speed_kmh < 18.0:
            rev_notch = 0.85
        elif obs.scale_speed_kmh < 34.0:
            rev_notch = 0.72
        else:
            rev_notch = 0.60
        ctrl.reverser = self._slew(ctrl.reverser, self._direction * rev_notch, self.REVERSER_SLEW_PER_S, dt)

        v_err = self.target_speed_kmh - obs.scale_speed_kmh
        self._speed_err_int = max(-20.0, min(32.0, self._speed_err_int + v_err * dt))
        reg_tgt = 0.28 + 0.014 * v_err + 0.0035 * self._speed_err_int
        if ctrl.drain_cock:
            reg_tgt = min(reg_tgt, 0.42)
        reg_tgt = max(0.0, min(0.88, reg_tgt))
        ctrl.regulator = self._slew(ctrl.regulator, reg_tgt, self.REGULATOR_SLEW_PER_S, dt)

        # 5. 依据 11-7 水面计【表观】读数调节 6-9 旁通手柄 (上坡时表观水位虚高，人类司机在未察觉坡度时易偏晚关小旁通)
        w_err_apparent = obs.apparent_water_pct - 50.0
        byp_tgt = max(0.0, min(1.0, 0.44 + 0.048 * w_err_apparent))
        ctrl.bypass_valve = self._slew(ctrl.bypass_valve, byp_tgt, self.BYPASS_SLEW_PER_S, dt)

        act_tag = "ドレン暖機" if ctrl.drain_cock else f"ノッチ {int(rev_notch*100)}%"
        self.status_summary = (
            f"擬人機関士 (遅延0.4s): 加減弁 {int(ctrl.regulator*100)}% | {act_tag} | 水面計視認 {obs.apparent_water_pct:.0f}%"
        )
