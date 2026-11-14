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
from typing import List

import mujoco
import numpy as np

from sim.auto_fireman import AutoFiremanAgent
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
KEY_COMMA = ord(",")   # バイパス弁 关小 (增加入锅给水量)
KEY_PERIOD = ord(".")  # バイパス弁 开大 (回流煤水车)
KEY_Y = ord("Y")    # 手動給水ポンプ 摇动 5 次
KEY_A = ord("A")    # AI 自動司炉 (Auto-Fireman) ON/OFF
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


def render_dashboard(tel: ThermoTelemetry, ctrl: CabControls, auto_fireman_on: bool = False) -> str:
    """生成 ANSI 终端实时驾驶台仪表盘"""
    sv_tag = "!! POPPING !!" if tel.safety_valve_open else "CLOSED       "
    burner_tag = "LIT (燃焼中)" if ctrl.burner_lit else "OFF (消火)  "
    fan_tag = "ON " if ctrl.chimney_fan else "OFF"
    drain_tag = "OPEN (排液)" if ctrl.drain_cock else "CLOSED     "
    af_tag = "ON (AI巡航)" if auto_fireman_on else "OFF        "
    rev_mode = "NEUTRAL (N)" if abs(ctrl.reverser) < 0.05 else (
        f"FWD +{ctrl.reverser*100:3.0f}%" if ctrl.reverser > 0 else f"REV {ctrl.reverser*100:4.0f}%"
    )

    lines = [
        "\033[H\033[J" + "=" * 76,
        f"  ASTER HOBBY JNR D51 LIVE STEAM COSIMULATOR  |  Sim Time: {tel.sim_time:6.1f} s  (Warp: {ctrl.thermal_warp:.0f}x)",
        "=" * 76,
        f"  圧力計 (PG-ø20) : {bar_gauge(tel.boiler_pressure_kgf, 0, 5.0)} {tel.boiler_pressure_kgf:4.2f} / 5.00 kgf/cm² [SV: {sv_tag}]",
        f"  水面計 (11-7)   : {bar_gauge(tel.water_level_pct, 0, 100)} {tel.water_level_pct:4.1f} %  (実水位:{tel.true_water_level_pct:4.1f}% | 頂板水深:{tel.crown_submerged_mm:+5.1f}mm)",
        f"  蒸気室 / 銅火室 : {tel.steam_chest_kgf:4.2f} kgf/cm²  |  T_boil: {tel.boiler_temp_c:5.1f}°C  |  T_crown: {tel.crown_temp_c:5.1f}°C  |  T_sh: {tel.superheat_temp_c:5.1f}°C",
        f"  燃料アルコール  : {tel.alcohol_ml:5.1f} / 130 ml |  炭水車水槽: {tel.tender_water_ml:5.1f} ml | 油:{tel.lubricator_oil_pct:4.1f}% | シール:{tel.seal_health*100:3.0f}%",
        f"  シリンダ・煙室  : 壁温 {tel.cyl_wall_temp_c:5.1f}°C | 凝結水 {tel.condensate_g:4.2f}g | 煙室負圧 {tel.smokebox_vacuum_pa:4.0f}Pa | F_L={tel.force_L:+5.1f}N F_R={tel.force_R:+5.1f}N",
        f"  走行・給水収支  : 動輪 {tel.wheel_rpm:+6.1f} RPM ({tel.scale_speed_kmh:5.1f} km/h) | 蒸発 {tel.steam_gen_g_s:4.2f} g/s | 給水 {tel.feedwater_g_s:4.2f} g/s",
        "-" * 76,
        f"  [I]着火:{burner_tag}  [O]燃料弁:{ctrl.fuel_valve*100:3.0f}%   [B]通風機:{fan_tag}    [K/L]通風弁:{ctrl.blower_valve*100:3.0f}%",
        f"  [↑/↓]加減弁:{ctrl.regulator*100:3.0f}%    [←/→]逆転機:{rev_mode}  [X]ドレン:{drain_tag} [,/ .]バイパス:{ctrl.bypass_valve*100:3.0f}%",
        f"  [A]AI司炉:{af_tag} [Y]手動ポンプ(+5) [9]15x加速 [7]冷機リセット [8]3.8kg即発車",
        "=" * 76,
    ]
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
    """封装 MuJoCo 与 1D 热力学步进器"""

    def __init__(self, xml_path: Path, preset: str = "cold"):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.calib = calibrate_model_and_valves(self.model)
        self.data = mujoco.MjData(self.model)
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
        self.qvel_pump = self.model.joint("j_pump_ram").dofadr[0]
        self.qvel_valL = self.model.joint("j_valve_L").dofadr[0]
        self.qvel_valR = self.model.joint("j_valve_R").dofadr[0]
        j_dc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_drain_cock")
        self.qvel_drain = int(self.model.jnt_dofadr[j_dc_id]) if j_dc_id >= 0 else -1
        self.loco_bid = self.model.body("loco_frame").id
        self.is_roller_mode = self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        self._prev_fwd_vel = 0.0

        if preset == "hot":
            self.apply_hot_preset()

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

        F_L, F_R, F_pump, tel = self.thermo.step(
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
        )

        self.data.ctrl[self.act_fL] = F_L
        self.data.ctrl[self.act_fR] = F_R
        self.data.ctrl[self.act_rev] = self.ctrl.reverser * 0.45

        # 施加平面滑阀背压摩擦反力 (O3-2)、给水泵液压负载与列车运行阻力
        self.data.qfrc_applied[self.qvel_valL] = tel.valve_friction_L
        self.data.qfrc_applied[self.qvel_valR] = tel.valve_friction_R
        self.data.qfrc_applied[self.qvel_pump] = F_pump
        if self.qvel_drain >= 0:
            self.data.qfrc_applied[self.qvel_drain] = 0.0085 if self.ctrl.drain_cock else 0.0
        if self.is_roller_mode:
            self.data.qfrc_applied[self.qvel_w3] = -0.0022 * omega_3 - 0.00012 * omega_3 * abs(omega_3)
        else:
            self.data.qfrc_applied[self.qvel_w3] = -0.0055 * omega_3 - 0.00048 * omega_3 * abs(omega_3)

        mujoco.mj_step(self.model, self.data)
        return tel


def run_interactive(xml_path: Path, preset: str, auto_fireman: bool = False) -> None:
    import mujoco.viewer

    runner = CoSimRunner(xml_path, preset=preset)
    af_agent = AutoFiremanAgent()
    af_enabled = auto_fireman

    def on_key(keycode: int) -> None:
        nonlocal af_enabled
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
        elif keycode == KEY_COMMA:
            ctrl.bypass_valve = max(0.0, ctrl.bypass_valve - 0.05)
        elif keycode == KEY_PERIOD:
            ctrl.bypass_valve = min(1.0, ctrl.bypass_valve + 0.05)
        elif keycode == KEY_Y:
            runner.thermo.trigger_hand_pump(strokes=5)
        elif keycode == KEY_A:
            af_enabled = not af_enabled
            if af_enabled:
                af_agent.reset(direction=1.0 if ctrl.reverser >= 0.0 else -1.0)
        elif keycode == KEY_9:
            ctrl.thermal_warp = 15.0 if ctrl.thermal_warp < 2.0 else 1.0
        elif keycode == KEY_7:
            mujoco.mj_resetData(runner.model, runner.data)
            runner.thermo.reset_cold_prepared()
            runner.ctrl = CabControls()
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
            if af_enabled and tel is not None:
                af_agent.update(steps_per_frame * dt, tel, runner.ctrl)
            viewer.sync()

            now = time.perf_counter()
            if tel is not None and (now - last_print) >= 0.10:
                sys.stdout.write(render_dashboard(tel, runner.ctrl, auto_fireman_on=af_enabled) + "\n")
                sys.stdout.flush()
                last_print = now

            elapsed = time.perf_counter() - t0
            sleep_s = (1.0 / 60.0) - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)


def run_headless_demo(xml_path: Path, out_png: Path) -> None:
    """
    自动执行说明书試運転 ①~⑫ 标准流程并绘制遥测曲线:
    阶段 A: 开燃料阀 + 外置抽风机点火升压 (15x 热加速验证沸腾曲线)
    阶段 B: 达 1.8 kgf/cm² 撤去风机，开 35% 通风阀继续升至 4.0 kgf/cm² 安全阀喷汽
    阶段 C: 恢复 1x 物理时间，挂前进档、开疏水阀与 38% 加减阀排液发车
    阶段 D: 关疏水阀、收逆转器至 65% 膨胀档，自动微调旁通阀稳定 3.5 kgf/cm² 与 50% 水位
    """
    import matplotlib.pyplot as plt

    runner = CoSimRunner(xml_path, preset="cold")
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
    parser.add_argument("--auto-fireman", action="store_true", help="启动时默认开启 RTX 5080 标定 AI 自动司炉巡航")
    parser.add_argument("--headless-demo", action="store_true", help="无 GUI 自动运行标准试车流程并输出曲线图")
    parser.add_argument("--out-plot", type=Path, default=Path("logs/d51_telemetry.png"))
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(f"未找到模型 {args.xml}，请先运行 python scripts/03_generate_mjcf.py")

    if args.headless_demo:
        run_headless_demo(args.xml, args.out_plot)
    else:
        run_interactive(args.xml, args.preset, auto_fireman=args.auto_fireman)


if __name__ == "__main__":
    main()