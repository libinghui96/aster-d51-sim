#!/usr/bin/env python3
"""
sim/run_cosim.py
Aster D51 1D 热力学与 3D MuJoCo 闭环多体动力学实时联调主程序。
用法:
  1. 交互式 3D 实时驾驶 (默认冷炉整备状态):
     python -m sim.run_cosim --xml models/d51_full.xml
  2. 交互式 3D 实时驾驶 (直接 3.8 kgf/cm² 热炉待发):
     python -m sim.run_cosim --xml models/d51_full.xml --preset hot
  3. 无头全流程自动试车并导出曲线图 (适合无 GUI / SSH 环境验证):
     python -m sim.run_cosim --xml models/d51_full.xml --headless-demo
"""

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

import mujoco
import numpy as np

from dataclasses import asdict
from sim.auto_fireman import AutoFiremanAgent
from sim.bonsai_copilot import AsyncBonsaiCopilot
from sim.human_engineer_agent import HumanEngineerAgent
from sim.ideal_oracle_controller import IdealOracleController
from sim.thermo_1d import AsterD51Thermo1D, CabControls, CylinderCalibration, ThermoTelemetry

# GLFW 按键编码 (已避开 MuJoCo Viewer 内置渲染快捷键)
KEY_UP = 265        # 加減弁 +
KEY_DOWN = 264      # 加減弁 -
KEY_RIGHT = 262     # 逆転機 前進方向
KEY_LEFT = 263      # 逆転機 後進方向
KEY_I = ord("I")    # 点火 / 炭酸ガス消火
KEY_O = ord("O")    # 燃料弁 开/关
KEY_B = ord("B")    # 烟囱外置通風機 ON/OFF
KEY_L = ord("L")    # 通風弁 +
KEY_K = ord("K")    # 通風弁 -
KEY_X = ord("X")    # ドレン弁 开/关
KEY_W = ord("W")    # 汽笛 (11-3 / 11-8) 吹鸣开关
KEY_F = ord("F")    # 火室門 (12-2) 开/关
KEY_COMMA = ord(",")   # バイパス弁 关小 (增加入锅给水量)
KEY_PERIOD = ord(".")  # バイパス弁 开大 (回流煤水车)
KEY_Y = ord("Y")    # 手動給水ポンプ 摇动 5 次
KEY_A = ord("A")    # AI 運転モード切换 (OFF -> 擬人 Human -> 理想 Oracle)
KEY_9 = ord("9")    # 切换 1x / 15x 热力烧水加速
KEY_7 = ord("7")    # 重置为冷炉状态 (試運転 ①)
KEY_8 = ord("8")    # 一键切换至 3.8 kgf/cm² 热炉待发状态


def calibrate_model_and_valves(model: mujoco.MjModel) -> CylinderCalibration:
    """
    对标说明书第 9 节「弁調整」:
    1. 采用 4-12 逆转横轴真实 11.0mm 摇臂与 ±0.46 rad (±26.4°) 行程，达到约 74% 满档进气截止比；
    2. 在独立 MjData 副本上以中立位 N 空转车轮 1.5 圈，精确测定左右滑阀气口对称中心与外余面。
    """
    j_rev = model.joint("j_reverser").id
    act_rev = model.actuator("reverser_ctrl").id
    model.jnt_range[j_rev] = [-0.48, 0.48]
    model.actuator_ctrlrange[act_rev] = [-0.46, 0.46]
    model.actuator_gainprm[act_rev, 0] = 60.0
    model.actuator_biasprm[act_rev, 1] = -60.0

    # 铸铁轮箍 + 試運転用コロ滚子等效惯量; 按 iquat 找到轴线 (世界 Y) 对应的主轴分量
    for idx in range(1, 5):
        bid = model.body(f"wheelset_{idx}").id
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[bid])
        axle = int(np.argmax(np.abs(rot.reshape(3, 3)[1, :])))
        extra = np.full(3, 0.0002)
        extra[axle] = 0.00045
        model.body_inertia[bid] += extra

    # 从关节限位直接提取左右活塞前死点坐标 (03_generate_mjcf.py 预留了 2mm 缓冲余量)
    j_pL = model.joint("j_piston_L").id
    j_pR = model.joint("j_piston_R").id
    q_fdc_L = float(model.jnt_range[j_pL, 1] - 0.002)
    q_fdc_R = float(model.jnt_range[j_pR, 1] - 0.002)
    stroke = float((model.jnt_range[j_pL, 1] - model.jnt_range[j_pL, 0]) - 0.004)

    # 在临时副本上空转主动轮测定中立位滑阀往复极值 (临时屏蔽接触与重力，等效架空滚轮台空转测阀)
    saved_disable = int(model.opt.disableflags)
    model.opt.disableflags = saved_disable | int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    cal_data = mujoco.MjData(model)
    cal_data.ctrl[act_rev] = 0.0
    qvel_w3 = model.joint("j_wheel_3").dofadr[0]
    adr_vL = model.sensor("valve_pos_L").adr[0]
    adr_vR = model.sensor("valve_pos_R").adr[0]
    has_free = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    vL_hist, vR_hist = [], []
    omega_target = 2.0 * math.pi  # 1 rev/s 平稳慢转标定
    # 摩擦使稳态仅约 4 rad/s，采样窗 (1500~5000 步) 需覆盖 >1 圈
    for step_i in range(5000):
        if has_free:
            cal_data.qvel[:6] = 0.0
        # 温和比例驱动 (0.04 N·m·s/rad)，避免小惯量连杆瞬间过载
        cal_data.qfrc_applied[qvel_w3] = 0.04 * (omega_target - cal_data.qvel[qvel_w3])
        mujoco.mj_step(model, cal_data)
        if step_i >= 1500:
            vL_hist.append(float(cal_data.sensordata[adr_vL]))
            vR_hist.append(float(cal_data.sensordata[adr_vR]))
    model.opt.disableflags = saved_disable

    center_L = 0.5 * (max(vL_hist) + min(vL_hist))
    center_R = 0.5 * (max(vR_hist) + min(vR_hist))
    lead_amp = 0.5 * (max(vL_hist) - min(vL_hist))

    return CylinderCalibration(
        q_fdc_L=q_fdc_L,
        q_fdc_R=q_fdc_R,
        stroke=stroke,
        q_valve_center_L=center_L,
        q_valve_center_R=center_R,
        steam_lap=0.99 * lead_amp,  # 精确匹配中立先导振幅 (保留 1% 微先导)，确保 3.8kgf 中立档严密锁止且 1.0kgf 冷态正反转强劲自启动
        exhaust_lap=0.02 * lead_amp,
        port_width=0.95 * lead_amp,
    )


def bar_gauge(val: float, vmin: float, vmax: float, width: int = 20) -> str:
    ratio = max(0.0, min(1.0, (val - vmin) / max(1e-9, vmax - vmin)))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render_dashboard(
    tel: ThermoTelemetry,
    ctrl: CabControls,
    auto_fireman_on: bool = False,
    ai_mode: str = "off",
    ai_status: str = "",
    spatial_3d: Optional[Any] = None,
    copilot_diag: Optional[Any] = None,
) -> str:
    """生成 ANSI 终端实时驾驶台仪表盘"""
    sv_tag = "!! POPPING !!" if tel.safety_valve_open else "CLOSED       "
    burner_tag = "LIT (燃焼中)" if ctrl.burner_lit else "OFF (消火)  "
    fan_tag = "ON " if ctrl.chimney_fan else "OFF"
    drain_tag = "OPEN (排液)" if ctrl.drain_cock else "CLOSED     "
    if ai_mode == "human":
        af_tag = "HUMAN (擬人)"
    elif ai_mode == "oracle" or auto_fireman_on:
        af_tag = "ORACLE(理想)"
    else:
        af_tag = "MANUAL(手動)"
    rev_mode = "NEUTRAL (N)" if abs(ctrl.reverser) < 0.05 else (
        f"FWD +{ctrl.reverser*100:3.0f}%" if ctrl.reverser > 0 else f"REV {ctrl.reverser*100:4.0f}%"
    )
    if spatial_3d is not None:
        pino_str = "PINO+IAPWS" if getattr(spatial_3d, "pino_active", False) else "IAPWS"
        eng_tag = f"RTX 5080 3D {pino_str} ({spatial_3d.total_voxels/1e6:.2f}M vox @ {spatial_3d.gpu_step_ms:.1f}ms)"
    else:
        eng_tag = "1D IAPWS + CFD LUT"

    lines = [
        "\033[H\033[J" + "=" * 78,
        f"  ASTER D51 LIVE STEAM COSIM [{eng_tag}] | t={tel.sim_time:6.1f}s ({ctrl.thermal_warp:.0f}x)",
        "=" * 78,
        f"  圧力計 (PG-ø20) : {bar_gauge(tel.boiler_pressure_kgf, 0, 5.0)} {tel.boiler_pressure_kgf:4.2f} / 5.00 kgf/cm² [SV: {sv_tag}] (沸騰:{tel.boiling_progress_pct:4.0f}%)",
        f"  水面計 (11-7)   : {bar_gauge(tel.water_level_pct, 0, 100)} {tel.water_level_pct:4.1f} %  (実水位:{tel.true_water_level_pct:4.1f}% | 頂板水深:{tel.crown_submerged_mm:+5.1f}mm)",
        f"  蒸気室 / 銅火室 : {tel.steam_chest_kgf:4.2f} kgf/cm²  |  T_boil: {tel.boiler_temp_c:5.1f}°C  |  T_crown: {tel.crown_temp_c:5.1f}°C  |  T_sh: {tel.superheat_temp_c:5.1f}°C (x={tel.steam_dryness_dome:.2f}->{tel.steam_dryness_chest:.2f})",
        f"  燃料アルコール  : {tel.alcohol_ml:5.1f} / 130 ml |  炭水車水槽: {tel.tender_water_ml:5.1f}/340 ml ({tel.tender_water_temp_c:4.1f}°C, 戻水:{tel.return_water_ml_s:4.2f}ml/s) | シール:{tel.seal_health*100:3.0f}%",
        f"  シリンダ・煙室  : 壁温 {tel.cyl_wall_temp_c:5.1f}°C | 凝結水 {tel.condensate_g:4.2f}g | 煙室負圧 {tel.smokebox_vacuum_pa:4.0f}Pa | F_L={tel.force_L:+5.1f}N F_R={tel.force_R:+5.1f}N",
        f"  走行・給水収支  : 動輪 {tel.wheel_rpm:+6.1f} RPM ({tel.scale_speed_kmh:5.1f} km/h) | 蒸発 {tel.steam_gen_g_s:4.2f} g/s | 給水 {tel.feedwater_g_s:4.2f} g/s",
    ]
    if ai_status:
        lines.append(f"  [AI 運転状態]   : {ai_status}")
    if spatial_3d is not None:
        lines.append(
            f"  3D 流体場プローブ: 空燃比 φ={spatial_3d.air_fuel_phi:4.2f} (火炎傾角 {spatial_3d.flame_tilt_deg:4.1f}°) | "
            f"過熱乾き度 x={spatial_3d.sh_profile_dryness[0]:.2f}->{spatial_3d.sh_profile_dryness[-1]:.2f} | "
            f"煙管貫通 {spatial_3d.open_flue_count}/5"
        )
    if copilot_diag is not None:
        lines.append(
            f"  [L3 {copilot_diag.model_name}] [{copilot_diag.status_level}] {copilot_diag.headline} | {copilot_diag.recommended_action}"
        )
    lines.extend(
        [
            "-" * 78,
            f"  [I]着火:{burner_tag}  [O]燃料弁:{ctrl.fuel_valve*100:3.0f}%   [B]通風機:{fan_tag}    [K/L]通風弁:{ctrl.blower_valve*100:3.0f}%",
            f"  [↑/↓]加減弁:{ctrl.regulator*100:3.0f}%    [←/→]逆転機:{rev_mode}  [X]ドレン:{drain_tag} [,/ .]バイパス:{ctrl.bypass_valve*100:3.0f}%",
            f"  [A]AI司炉:{af_tag} [Y]手動ポンプ(+5) [9]15x加速 [7]冷機リセット [8]3.8kg即発車",
            "=" * 78,
        ]
    )
    if tel.hydrolock_L or tel.hydrolock_R:
        lines.append("  [HYDROLOCK!] 冷気筒ウォーターハンマー発生! 直ちに [X] ドレン弁を開いて排水して下さい!")
    if tel.flash_steam_spike:
        lines.append("  [FLASH SPIKE!] 高温空焚き火室への急冷給水により急激蒸発スパイク発生!")
    if tel.dry_fire_alarm:
        lines.append(f"  [WARNING] ボイラ低水位/空焚き危険 (銅火室頂板 {tel.crown_temp_c:.0f}°C, 損傷 {tel.copper_anneal_damage*100:.1f}%)!")
    if tel.emergency_overpressure:
        lines.append("  [EMERGENCY] 圧力 5.0 kgf/cm² 超過! 直ちに [I] 消火して下さい!")
    return "\n".join(lines)


class CoSimRunner:
    """封装 MuJoCo 与 1D / RTX 5080 3D 热流体耦合步进器"""

    def __init__(self, xml_path: Path, preset: str = "cold", use_gpu_3d: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.calib = calibrate_model_and_valves(self.model)
        self.data = mujoco.MjData(self.model)
        self.use_gpu_3d = bool(use_gpu_3d)
        if self.use_gpu_3d:
            from sim.thermo_3d_gpu import AsterD51Thermo3DGPU

            self.thermo: Any = AsterD51Thermo3DGPU(self.calib)
            print(
                f"[RTX 5080 3D] 已加载 {self.thermo.total_voxels:,} 体素五大零件域求解器 "
                f"(STL 体素化耗时 {self.thermo.voxelize_init_ms:.1f} ms, PINO={self.thermo.pino_loaded}, "
                f"5烟管贯通={self.thermo.open_flue_count}/5, 烟室气密={self.thermo.smokebox_airtight})"
            )
        else:
            self.thermo = AsterD51Thermo1D(self.calib)
        self.ctrl = CabControls()

        # 缓存传感器、执行器与关节地址
        self.s_pL = self.model.sensor("piston_pos_L").adr[0]
        self.s_vL = self.model.sensor("piston_vel_L").adr[0]
        self.s_valL = self.model.sensor("valve_pos_L").adr[0]
        self.s_pR = self.model.sensor("piston_pos_R").adr[0]
        self.s_vR = self.model.sensor("piston_vel_R").adr[0]
        self.s_valR = self.model.sensor("valve_pos_R").adr[0]
        self.s_w3 = self.model.sensor("wheel3_omega").adr[0]
        self.s_pump = self.model.sensor("pump_ram_vel").adr[0]

        self.act_fL = self.model.actuator("steam_force_L").id
        self.act_fR = self.model.actuator("steam_force_R").id
        self.act_rev = self.model.actuator("reverser_ctrl").id

        self.qvel_w3 = self.model.joint("j_wheel_3").dofadr[0]
        self.qpos_pump = self.model.joint("j_pump_ram").qposadr[0]
        self.qvel_pump = self.model.joint("j_pump_ram").dofadr[0]
        self.qvel_valL = self.model.joint("j_valve_L").dofadr[0]
        self.qvel_valR = self.model.joint("j_valve_R").dofadr[0]
        j_dc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_drain_cock")
        self.qvel_drain = int(self.model.jnt_dofadr[j_dc_id]) if j_dc_id >= 0 else -1
        self.loco_bid = self.model.body("loco_frame").id
        self.is_roller_mode = self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        self._prev_fwd_vel = 0.0

        # P3: 缓存驾驶室与煤水车全部 15 个独立可动操纵件/仪表关节地址
        self._vis_joints: dict[str, tuple[int, int]] = {}
        for jname in (
            "j_cab_regulator",
            "j_cab_blower",
            "j_cab_whistle",
            "j_cab_sight_drain",
            "j_cab_pg_needle",
            "j_cab_blowdown",
            "j_cab_fire_door",
            "j_cab_bypass",
            "j_cab_roof_hatch",
            "j_tender_handpump",
            "j_tender_drain_plug",
            "j_tender_coal_cover",
            "j_tender_water_lid",
            "j_tender_fuel_valve",
            "j_tender_fuel_cap",
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._vis_joints[jname] = (int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid]))
        jid_hpr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_tender_hp_ram")
        self.qpos_hp_ram = int(self.model.jnt_qposadr[jid_hpr]) if jid_hpr >= 0 else -1
        self.qvel_hp_ram = int(self.model.jnt_dofadr[jid_hpr]) if jid_hpr >= 0 else -1
        self._last_v_hand_ram = 0.0
        self._handpump_vis_timer = 0.0
        self._handpump_vis_phase = 0.0
        self.external_handpump_rad: Optional[float] = None

        if preset == "hot":
            self.apply_hot_preset()
        mujoco.mj_forward(self.model, self.data)

    def trigger_hand_pump(self, strokes: int = 5) -> None:
        self.thermo.trigger_hand_pump(strokes=strokes)
        self._handpump_vis_timer = max(self._handpump_vis_timer, min(4.0, 0.50 * max(1, strokes)))

    def _sync_cab_tender_visual_joints(self, dt: float, tel: ThermoTelemetry) -> None:
        if not self._vis_joints:
            return
        if self.external_handpump_rad is not None:
            hp_tgt = float(self.external_handpump_rad)
        elif self._handpump_vis_timer > 0.0:
            self._handpump_vis_timer = max(0.0, self._handpump_vis_timer - dt)
            self._handpump_vis_phase += dt * 2.0 * math.pi * 2.0
            hp_tgt = 0.34 * math.sin(self._handpump_vis_phase)
        else:
            self._handpump_vis_phase = 0.0
            hp_tgt = 0.0

        pg_val = float(getattr(tel, "pg_bourdon_kgf", tel.boiler_pressure_kgf))
        p_norm = max(0.0, min(1.05, pg_val / 6.0))
        open_water_bay = (
            bool(getattr(self.ctrl, "coal_cover_open", False))
            or float(getattr(self.ctrl, "water_fill_valve", 0.0)) > 0.02
            or bool(getattr(self.ctrl, "tender_drain_open", False))
            or abs(hp_tgt) > 0.01
        )
        targets = {
            "j_cab_regulator": 1.12 * self.ctrl.regulator,
            "j_cab_blower": 3.14 * self.ctrl.blower_valve,
            "j_cab_whistle": -0.38 * self.ctrl.whistle_valve,
            "j_cab_sight_drain": 0.0016 if bool(getattr(self.ctrl, "sight_drain_valve", False)) else 0.0,
            "j_cab_pg_needle": -2.10 + p_norm * 4.20,
            "j_cab_blowdown": 3.14 * float(getattr(self.ctrl, "blowdown_valve", 0.0)),
            "j_cab_fire_door": +1.65 * self.ctrl.firebox_door,
            "j_cab_bypass": -1.35 * self.ctrl.bypass_valve,
            "j_cab_roof_hatch": 2.10 if bool(getattr(self.ctrl, "cab_roof_open", False)) else 0.0,
            "j_tender_handpump": hp_tgt,
            "j_tender_drain_plug": 0.0050 if bool(getattr(self.ctrl, "tender_drain_open", False)) else 0.0,
            "j_tender_coal_cover": 1.75 if open_water_bay else 0.0,
            "j_tender_water_lid": 1.45 if self.ctrl.water_lid_open else 0.0,
            "j_tender_fuel_valve": 1.50 * self.ctrl.fuel_valve,
            "j_tender_fuel_cap": 0.0045 if self.ctrl.fuel_cap_open else 0.0,
        }
        alpha = min(1.0, dt * 24.0)
        for jname, tgt in targets.items():
            addrs = self._vis_joints.get(jname)
            if addrs is not None:
                qadr, dadr = addrs
                curr = float(self.data.qpos[qadr])
                nxt = curr + alpha * (tgt - curr)
                dq_dt = (nxt - curr) / max(1e-6, dt)
                self.data.qpos[qadr] = nxt
                self.data.qvel[dadr] = dq_dt if jname == "j_tender_handpump" else 0.0
                if jname == "j_tender_handpump":
                    self._last_v_hand_ram = 0.008 * dq_dt
                    if self.qpos_hp_ram >= 0:
                        self.data.qpos[self.qpos_hp_ram] = 0.008 * nxt
                    if self.qvel_hp_ram >= 0:
                        self.data.qvel[self.qvel_hp_ram] = self._last_v_hand_ram

    def apply_hot_preset(self) -> None:
        self.thermo.reset_hot_ready()
        self.ctrl.fuel_valve = 1.0
        self.ctrl.burner_lit = True
        self.ctrl.chimney_fan = False
        self.ctrl.blower_valve = 0.25
        self.ctrl.reverser = 0.85
        self.ctrl.drain_cock = True
        self.ctrl.thermal_warp = 1.0

    def step_once(self) -> ThermoTelemetry:
        sd = self.data.sensordata
        dt = float(self.model.opt.timestep)
        omega_3 = float(sd[self.s_w3])
        v_vL = float(self.data.qvel[self.qvel_valL])
        v_vR = float(self.data.qvel[self.qvel_valR])

        # 提取车体上坡俯仰角与纵向加速度 (用于 O4-1 坡道假水位与顶板浸没判定)
        pitch_rad = 0.0
        accel_x = 0.0
        if not self.is_roller_mode:
            r20 = float(self.data.xmat[self.loco_bid, 6])
            pitch_rad = math.asin(max(-0.35, min(0.35, r20)))
            fwd_vel = omega_3 * 0.0219
            accel_x = (fwd_vel - self._prev_fwd_vel) / max(1e-6, dt)
            self._prev_fwd_vel = fwd_vel

        step_kwargs = dict(
            dt=dt,
            q_pL=float(sd[self.s_pL]),
            v_pL=float(sd[self.s_vL]),
            q_vL=float(sd[self.s_valL]),
            q_pR=float(sd[self.s_pR]),
            v_pR=float(sd[self.s_vR]),
            q_vR=float(sd[self.s_valR]),
            omega_3=omega_3,
            v_pump_ram=float(sd[self.s_pump]),
            ctrl=self.ctrl,
            sim_time=float(self.data.time),
            v_vL=v_vL,
            v_vR=v_vR,
            pitch_rad=pitch_rad,
            accel_x=accel_x,
            v_hand_ram=self._last_v_hand_ram,
        )
        if self.use_gpu_3d:
            step_kwargs["q_ram"] = float(self.data.qpos[self.qpos_pump])
        F_L, F_R, F_pump, tel = self.thermo.step(**step_kwargs)

        self.data.ctrl[self.act_fL] = F_L
        self.data.ctrl[self.act_fR] = F_R
        self.data.ctrl[self.act_rev] = self.ctrl.reverser * 0.45

        # 施加平面滑阀背压摩擦反力 (O3-2)、给水泵液压负载与列车运行阻力
        self.data.qfrc_applied[self.qvel_valL] = tel.valve_friction_L
        self.data.qfrc_applied[self.qvel_valR] = tel.valve_friction_R
        self.data.qfrc_applied[self.qvel_pump] = F_pump
        if self.qvel_drain >= 0:
            self.data.qfrc_applied[self.qvel_drain] = 0.022 if self.ctrl.drain_cock else 0.0
        if self.is_roller_mode:
            self.data.qfrc_applied[self.qvel_w3] = -0.0022 * omega_3 - 0.00012 * omega_3 * abs(omega_3)
        else:
            self.data.qfrc_applied[self.qvel_w3] = -0.0055 * omega_3 - 0.00048 * omega_3 * abs(omega_3)

        self._sync_cab_tender_visual_joints(dt, tel)
        mujoco.mj_step(self.model, self.data)
        return tel


def run_interactive(
    xml_path: Path,
    preset: str,
    auto_fireman: bool = False,
    ai_mode_init: str = "off",
    use_gpu_3d: bool = False,
    enable_copilot: bool = False,
) -> None:
    import mujoco.viewer

    runner = CoSimRunner(xml_path, preset=preset, use_gpu_3d=use_gpu_3d)
    human_agent = HumanEngineerAgent()
    oracle_agent = IdealOracleController()
    ai_mode = ai_mode_init if ai_mode_init in ("off", "human", "oracle") else ("oracle" if auto_fireman else "off")
    copilot = AsyncBonsaiCopilot(enabled=enable_copilot) if enable_copilot else None

    def on_key(keycode: int) -> None:
        nonlocal ai_mode
        ctrl = runner.ctrl
        if keycode == KEY_UP:
            ctrl.regulator = min(1.0, ctrl.regulator + 0.05)
        elif keycode == KEY_DOWN:
            ctrl.regulator = max(0.0, ctrl.regulator - 0.05)
        elif keycode == KEY_RIGHT:
            ctrl.reverser = min(1.0, round(ctrl.reverser + 0.10, 2))
        elif keycode == KEY_LEFT:
            ctrl.reverser = max(-1.0, round(ctrl.reverser - 0.10, 2))
        elif keycode == KEY_I:
            if not ctrl.burner_lit:
                ctrl.fuel_valve = max(0.5, ctrl.fuel_valve)
                ctrl.burner_lit = True
            else:
                ctrl.burner_lit = False
                ctrl.fuel_valve = 0.0
        elif keycode == KEY_O:
            ctrl.fuel_valve = 0.0 if ctrl.fuel_valve > 0.1 else 1.0
        elif keycode == KEY_B:
            ctrl.chimney_fan = not ctrl.chimney_fan
        elif keycode == KEY_L:
            ctrl.blower_valve = min(1.0, ctrl.blower_valve + 0.10)
        elif keycode == KEY_K:
            ctrl.blower_valve = max(0.0, ctrl.blower_valve - 0.10)
        elif keycode == KEY_X:
            ctrl.drain_cock = not ctrl.drain_cock
        elif keycode == KEY_W:
            ctrl.whistle_valve = 0.0 if ctrl.whistle_valve > 0.1 else 1.0
        elif keycode == KEY_F:
            ctrl.firebox_door = 0.0 if ctrl.firebox_door > 0.1 else 1.0
        elif keycode == KEY_COMMA:
            ctrl.bypass_valve = max(0.0, ctrl.bypass_valve - 0.05)
        elif keycode == KEY_PERIOD:
            ctrl.bypass_valve = min(1.0, ctrl.bypass_valve + 0.05)
        elif keycode == KEY_Y:
            runner.trigger_hand_pump(strokes=5)
        elif keycode == KEY_A:
            cycle = {"off": "human", "human": "oracle", "oracle": "off"}
            ai_mode = cycle.get(ai_mode, "off")
            dir_sgn = 1.0 if ctrl.reverser >= 0.0 else -1.0
            if ai_mode == "human":
                human_agent.reset(direction=dir_sgn)
            elif ai_mode == "oracle":
                oracle_agent.reset(direction=dir_sgn)
                runner.external_handpump_rad = None
            else:
                runner.external_handpump_rad = None
        elif keycode == KEY_9:
            ctrl.thermal_warp = 15.0 if ctrl.thermal_warp < 2.0 else 1.0
        elif keycode == KEY_7:
            mujoco.mj_resetData(runner.model, runner.data)
            runner.thermo.reset_cold_prepared()
            runner.ctrl = CabControls()
            ai_mode = "off"
            runner.external_handpump_rad = None
        elif keycode == KEY_8:
            runner.apply_hot_preset()

    dt = runner.model.opt.timestep
    steps_per_frame = max(1, int(round((1.0 / 60.0) / dt)))
    last_print = 0.0

    with mujoco.viewer.launch_passive(runner.model, runner.data, key_callback=on_key) as viewer:
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -16.0
        viewer.cam.distance = 0.36
        viewer.cam.lookat[:] = [0.04, 0.0, 0.03]

        while viewer.is_running():
            t0 = time.perf_counter()
            tel = None
            for _ in range(steps_per_frame):
                tel = runner.step_once()
            ai_status = ""
            if tel is not None:
                if ai_mode == "human":
                    human_agent.update(steps_per_frame * dt, tel, runner.ctrl, hand_pump_cb=runner.trigger_hand_pump)
                    runner.external_handpump_rad = (
                        human_agent.handpump_anim_rad if abs(human_agent.handpump_anim_rad) > 1e-4 else None
                    )
                    ai_status = human_agent.status_summary
                elif ai_mode == "oracle":
                    runner.external_handpump_rad = None
                    oracle_agent.update(steps_per_frame * dt, tel, runner.ctrl, hand_pump_cb=runner.trigger_hand_pump)
                    ai_status = oracle_agent.status_summary
                else:
                    runner.external_handpump_rad = None
            viewer.sync()

            now = time.perf_counter()
            if tel is not None and (now - last_print) >= 0.10:
                sp = getattr(runner.thermo, "spatial", None)
                diag = None
                if copilot is not None:
                    copilot.submit_telemetry_nonblocking(
                        asdict(tel), asdict(runner.ctrl), sp.to_dict() if sp is not None else None
                    )
                    diag = copilot.get_latest_diagnosis()
                sys.stdout.write(
                    render_dashboard(
                        tel,
                        runner.ctrl,
                        auto_fireman_on=(ai_mode != "off"),
                        ai_mode=ai_mode,
                        ai_status=ai_status,
                        spatial_3d=sp,
                        copilot_diag=diag,
                    )
                    + "\n"
                )
                sys.stdout.flush()
                last_print = now

            elapsed = time.perf_counter() - t0
            sleep_s = (1.0 / 60.0) - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    if copilot is not None:
        copilot.shutdown()


def run_headless_demo(xml_path: Path, out_png: Path, use_gpu_3d: bool = False) -> None:
    """
    自动执行说明书試運転 ①~⑫ 标准流程并绘制遥测曲线:
    阶段 A: 开燃料阀 + 外置抽风机点火升压 (15x 热加速验证沸腾曲线)
    阶段 B: 达 1.8 kgf/cm² 撤去风机，开 35% 通风阀继续升至 4.0 kgf/cm² 安全阀喷汽
    阶段 C: 恢复 1x 物理时间，挂前进档、开疏水阀与 38% 加减阀排液发车
    阶段 D: 关疏水阀、收逆转器至 65% 膨胀档，自动微调旁通阀稳定 3.5 kgf/cm² 与 50% 水位
    """
    import matplotlib.pyplot as plt

    runner = CoSimRunner(xml_path, preset="cold", use_gpu_3d=use_gpu_3d)
    c = runner.ctrl
    c.fuel_valve = 1.0
    c.burner_lit = True
    c.chimney_fan = True
    c.reverser = 0.0
    c.thermal_warp = 15.0  # 烧水阶段使用 15x 热加速，缩短冷等待时间

    history: List[ThermoTelemetry] = []
    started_running = False
    t_run_start = 0.0

    total_steps = 80000  # 40 秒仿真步长 (覆盖加速升压 + 1x 实车运转)
    print("[DEMO] 开始执行 Aster D51 试运转全流程闭环仿真...")
    for i in range(total_steps):
        tel = runner.step_once()

        # 阶段 B: 1.8 kgf/cm² 撤下电动抽风机，改用蒸汽通风阀 (手順 ⑧)
        if not started_running and tel.boiler_pressure_kgf >= 1.80 and c.chimney_fan:
            c.chimney_fan = False
            c.blower_valve = 0.38

        # 阶段 C: 4.0 kgf/cm² 安全阀起跳后，切换 1x 实时，开疏水阀与加减阀发车 (手順 ⑧~⑨)
        if not started_running and tel.safety_valve_open:
            started_running = True
            t_run_start = tel.sim_time
            c.thermal_warp = 1.0
            c.blower_valve = 0.0
            c.drain_cock = True
            c.reverser = 0.95
            c.regulator = 0.42
            print(f"  -> [t={tel.sim_time:.1f}s] 安全阀于 {tel.boiler_pressure_kgf:.2f} kgf/cm² 起跳，开阀发车！")

        # 阶段 D: 起步 4 秒排净冷缸凝结水后关闭疏水阀，模拟司机根据水位调节旁通阀 (手順 ⑩~⑫)
        if started_running and (tel.sim_time - t_run_start) > 4.0:
            c.drain_cock = False
            c.reverser = 0.78
            c.regulator = 0.28
            # 模拟司机目视水面计维持 50% 水位的比例调节
            c.bypass_valve = max(0.0, min(1.0, 0.42 + 0.04 * (tel.water_level_pct - 50.0)))

        if i % 40 == 0:
            history.append(tel)

    t_arr = np.array([h.sim_time for h in history])
    p_boil = np.array([h.boiler_pressure_kgf for h in history])
    p_sc = np.array([h.steam_chest_kgf for h in history])
    lvl = np.array([h.water_level_pct for h in history])
    rpm = np.array([h.wheel_rpm for h in history])
    f_l = np.array([h.force_L for h in history])

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t_arr, p_boil, label="Boiler Pressure (kgf/cm²)", lw=1.8)
    axes[0].plot(t_arr, p_sc, label="Steam Chest Pressure (kgf/cm²)", lw=1.3, ls="--")
    axes[0].axhline(4.0, color="r", ls=":", label="Safety Valve Pop (4.0)")
    axes[0].set_ylabel("kgf/cm²")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, lvl, color="tab:blue", lw=1.8, label="Sight Glass Water Level (%)")
    axes[1].axhline(50.0, color="gray", ls="--", label="Target 50%")
    axes[1].set_ylabel("Level (%)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_arr, rpm, color="tab:green", lw=1.6, label="Driving Wheel Speed (RPM)")
    axes[2].set_ylabel("RPM")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_arr, f_l, color="tab:orange", lw=0.9, label="Left Piston Net Force (N)")
    axes[3].set_ylabel("Force (N)")
    axes[3].set_xlabel("Simulation Time (s)")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Aster Hobby D51 Coupled 1D-Thermo & 3D-MuJoCo Test Run Telemetry")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[DONE] 试车完成！最终表压={p_boil[-1]:.2f} kgf/cm², 水位={lvl[-1]:.1f}%, 转速={rpm[-1]:.1f} RPM")
    print(f"[DONE] 遥测曲线已保存至: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster D51 1D-3D 实时热机联合仿真器")
    parser.add_argument("--xml", type=Path, default=Path("models/d51_full.xml"))
    parser.add_argument("--preset", choices=["cold", "hot"], default="cold")
    parser.add_argument("--gpu-3d", action="store_true", help="启用本地 RTX 5080 571万体素五大零件域真 3D 多相热流 + PINO 求解器")
    parser.add_argument("--copilot", action="store_true", help="启用异步本地 Ollama bonsai-2:27b L3 机务诊断副驾")
    parser.add_argument("--auto-fireman", action="store_true", help="启动时默认开启理想最优自动司炉巡航")
    parser.add_argument(
        "--ai-mode",
        choices=["off", "human", "oracle"],
        default=None,
        help="选择 AI 驾驶模式: off(手动) | human(拟人实操司机) | oracle(理想全知极限控制)",
    )
    parser.add_argument("--headless-demo", action="store_true", help="无 GUI 自动运行标准试车流程并输出曲线图")
    parser.add_argument("--out-plot", type=Path, default=Path("logs/d51_telemetry.png"))
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(f"未找到模型 {args.xml}，请先运行 python scripts/03_generate_mjcf.py")

    if args.headless_demo:
        run_headless_demo(args.xml, args.out_plot, use_gpu_3d=args.gpu_3d)
    else:
        init_mode = args.ai_mode if args.ai_mode is not None else ("oracle" if args.auto_fireman else "off")
        run_interactive(
            args.xml,
            args.preset,
            auto_fireman=args.auto_fireman,
            ai_mode_init=init_mode,
            use_gpu_3d=args.gpu_3d,
            enable_copilot=args.copilot,
        )


if __name__ == "__main__":
    main()