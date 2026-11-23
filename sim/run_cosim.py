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
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from dataclasses import asdict
from sim.auto_fireman import AutoFiremanAgent
from sim.bonsai_copilot import AsyncBonsaiCopilot
from sim.human_engineer_agent import HumanEngineerAgent
from sim.hw_guard import HardwareThermalPowerGuard, apply_cpu_thread_and_nice_caps
from sim.ideal_oracle_controller import IdealOracleController
from sim.thermo_1d import AsterD51Thermo1D, CabControls, CylinderCalibration, ThermoTelemetry
from sim.gpu_track_endurance import (
    GPUConsistEnduranceEngine,
    TrackProfile3D,
    compute_physical_train_resistance_n,
    compute_rail_adhesion_mu,
)

apply_cpu_thread_and_nice_caps(max_threads=6, nice_delta=2)

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
    model.jnt_range[j_rev] = [-0.36, 0.36]
    model.actuator_ctrlrange[act_rev] = [-0.34, 0.34]
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
        steam_lap=0.995 * lead_amp,  # 精确匹配中立先导振幅 (0.5% 微先导), 确保 3.8kgf 中立档严密锁止 (<1 RPM) 且 1.0kgf 冷态正反转强劲自启动
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
        f"  走行・給水収支  : 動輪 {tel.wheel_rpm:+6.1f} RPM (対地 {tel.ground_speed_kmh:5.1f} km/h | 空転 {tel.wheel_slip_pct:4.1f}%) | 蒸発 {tel.steam_gen_g_s:4.2f} g/s | 給水 {tel.feedwater_g_s:4.2f} g/s",
        f"  実走行距離・航続: {tel.distance_m:7.2f} m (1:30換算 {tel.scale_distance_km:5.3f} km | {tel.lap_count:4.2f} 周) | 背圧+{tel.exhaust_backpressure_kgf:4.2f}kgf | 予測残 {tel.predicted_range_m:5.0f}m [{tel.range_limiting_factor}]",
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
    if tel.derailed:
        lines.append(f"  [DERAILED!!] 脱線落枕検知 ({tel.derail_reason}) — 枕木・道床抵抗により緊急制動中!")
    if tel.hydrolock_L or tel.hydrolock_R:
        lines.append("  [HYDROLOCK!] 冷気筒ウォーターハンマー発生! 直ちに [X] ドレン弁を開いて排水して下さい!")
    if tel.flash_steam_spike:
        lines.append("  [FLASH SPIKE!] 高温空焚き火室への急冷給水により急激蒸発スパイク発生!")
    if tel.dry_fire_alarm:
        lines.append(f"  [WARNING] ボイラ低水位/空焚き危険 (銅火室頂板 {tel.crown_temp_c:.0f}°C, 損傷 {tel.copper_anneal_damage*100:.1f}%)!")
    if tel.emergency_overpressure:
        lines.append("  [EMERGENCY] 圧力 5.0 kgf/cm² 超過! 直ちに [I] 消火して下さい!")
    if tel.gpu_monitor and tel.gpu_monitor.get("enabled"):
        gm = tel.gpu_monitor
        lines.append(
            f"  [GPU MONITOR] SM={gm.get('sm_util_pct', 0):.0f}% | MemBW={gm.get('mem_bw_util_pct', 0):.0f}% | "
            f"VRAM={gm.get('vram_used_mb', 0):.0f}/{gm.get('vram_total_mb', 0):.0f}MB (Torch {gm.get('torch_alloc_mb', 0):.0f}MB) | "
            f"{gm.get('gpu_temp_c', 0):.0f}°C / {gm.get('gpu_power_w', 0):.1f}W | PDE+PINO={gm.get('gpu_3d_pde_ms', 0)+gm.get('gpu_pino_ms', 0):.2f}ms"
        )
    return "\n".join(lines)


class CoSimRunner:
    """封装 MuJoCo 与 1D / RTX 5080 3D 热流体耦合步进器"""

    def __init__(
        self,
        xml_path: Path,
        preset: str = "cold",
        use_gpu_3d: bool = False,
        cfd_subdiv: int = 12,
        snapshot_every_cfd: int = 3,
        enable_gpu_monitor: bool = False,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.calib = calibrate_model_and_valves(self.model)
        self.data = mujoco.MjData(self.model)
        self.use_gpu_3d = bool(use_gpu_3d)
        self.enable_gpu_monitor = bool(enable_gpu_monitor)
        if self.use_gpu_3d:
            from sim.thermo_3d_gpu import AsterD51Thermo3DGPU

            self.thermo: Any = AsterD51Thermo3DGPU(
                self.calib,
                cfd_subdiv=cfd_subdiv,
                snapshot_every_cfd=snapshot_every_cfd,
                enable_gpu_monitor=enable_gpu_monitor,
            )
            print(
                f"[RTX 5080 3D] 已加载 {self.thermo.total_voxels:,} 体素八大零件域+七算子PINO求解器 "
                f"(STL 体素化耗时 {self.thermo.voxelize_init_ms:.1f} ms, PINO={self.thermo.pino_loaded}, "
                f"5烟管贯通={self.thermo.open_flue_count}/5, 烟室气密={self.thermo.smokebox_airtight})"
            )
        else:
            from sim.gpu_monitor import OptionalGPUMonitor
            self.thermo = AsterD51Thermo1D(self.calib)
            self._fallback_gpu_mon = OptionalGPUMonitor(enabled=enable_gpu_monitor)
        self.ctrl = CabControls()
        self.last_f_pump: float = 0.0
        self.derailed: bool = False
        self.derail_reason: str = ""

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
        self.tender_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tender_body")
        self.is_roller_mode = self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        self.is_curve_r2m = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r2m_track_layout") >= 0
        self.is_oval_grade = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "oval_track_layout") >= 0
        self.track_mode = (
            "roller"
            if self.is_roller_mode
            else ("oval_grade" if self.is_oval_grade else ("curve_r2m" if self.is_curve_r2m else "track"))
        )
        self.track_profile = TrackProfile3D(self.track_mode if not self.is_roller_mode else "track")
        self._prev_fwd_vel = 0.0
        self.distance_m: float = 0.0
        self.lap_count: float = 0.0
        self._prev_loco_xy: Optional[Tuple[float, float, float]] = None
        self._range_engine: Optional[GPUConsistEnduranceEngine] = None
        self._live_range_cache: Dict[str, Any] = {
            "p10_m": 0.0,
            "p50_m": 0.0,
            "p90_m": 0.0,
            "p50_min": 0.0,
            "limiting_factor": "NOMINAL",
        }

        # 缓存机车、煤水车与铰接挂车干重/基准惯量及自由关节地址 (用于实时水/甲醇消耗变质量与坡道切向重力回写)
        self._loco_base_mass = float(self.model.body_mass[self.loco_bid])
        self._loco_dry_mass = max(1.5, self._loco_base_mass - 0.210)
        self._loco_base_inertia = self.model.body_inertia[self.loco_bid].copy()
        self._tender_base_mass = float(self.model.body_mass[self.tender_bid]) if self.tender_bid >= 0 else 0.0
        self._tender_dry_mass = max(0.6, self._tender_base_mass - 0.375) if self.tender_bid >= 0 else 0.0
        self._tender_base_inertia = self.model.body_inertia[self.tender_bid].copy() if self.tender_bid >= 0 else None
        j_tfree = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tender_free")
        self._tender_dof0 = int(self.model.jnt_dofadr[j_tfree]) if j_tfree >= 0 else -1

        self._fcar_specs: List[Tuple[int, int, float, np.ndarray]] = []
        for c_idx in range(1, 12):
            cbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"freight_car_{c_idx}")
            cjid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"fcar_free_{c_idx}")
            if cbid >= 0 and cjid >= 0:
                self._fcar_specs.append(
                    (int(cbid), int(self.model.jnt_dofadr[cjid]), float(self.model.body_mass[cbid]), self.model.body_inertia[cbid].copy())
                )

        # 缓存全编组 10 轴垂向板簧/螺旋簧弹性悬挂关节与轮对刚体 (用于板簧实时挠度与真实轮轨脱线落枕物理判定)
        self._susp_qpos_addrs: List[int] = []
        for s_jname in (
            "j_leading_susp",
            "j_susp_z1",
            "j_susp_z2",
            "j_susp_z3",
            "j_susp_z4",
            "j_trailing_susp",
            "j_tsusp_z1",
            "j_tsusp_z2",
            "j_tsusp_z3",
            "j_tsusp_z4",
        ):
            sj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s_jname)
            self._susp_qpos_addrs.append(int(self.model.jnt_qposadr[sj_id]) if sj_id >= 0 else -1)

        self._lat_qpos_addrs: List[int] = []
        for l_jname in ("j_leading_lat", "j_lat_y1", "j_lat_y2", "j_lat_y3", "j_lat_y4"):
            lj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, l_jname)
            if lj_id >= 0:
                self._lat_qpos_addrs.append(int(self.model.jnt_qposadr[lj_id]))

        self._wheelset_specs: List[tuple[int, str, float]] = []
        for wb_name, r_nom in (
            ("leading_wheelset", 0.0130),
            ("wheelset_1", 0.021875),
            ("wheelset_2", 0.021875),
            ("wheelset_3", 0.021875),
            ("wheelset_4", 0.021875),
            ("trailing_wheelset", 0.0130),
            ("tender_wheelset_1", 0.0130),
            ("tender_wheelset_2", 0.0130),
            ("tender_wheelset_3", 0.0130),
            ("tender_wheelset_4", 0.0130),
        ):
            wbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, wb_name)
            if wbid >= 0:
                self._wheelset_specs.append((int(wbid), wb_name, r_nom))
        for c_idx in range(1, 12):
            for a_idx in (1, 2):
                wb_c = f"fcar_{c_idx}_wheelset_{a_idx}"
                wbid_c = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, wb_c)
                if wbid_c >= 0:
                    self._wheelset_specs.append((int(wbid_c), wb_c, 0.0130))

        self._offrail_geom_ids: set[int] = set()
        self._wheel_geom_ids: set[int] = set()
        self._drv_tread_geom_ids: List[int] = []
        self._rail_geom_ids: List[int] = []
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname == "ground" or gname.startswith("sleeper_"):
                self._offrail_geom_ids.add(gid)
            elif gname.startswith("rail_"):
                self._rail_geom_ids.append(gid)
            elif "_tread_" in gname or "_flange_" in gname or "_throat_" in gname:
                self._wheel_geom_ids.add(gid)
                if gname.startswith("drv_") and "_tread_" in gname:
                    self._drv_tread_geom_ids.append(gid)
        self._derail_consec_steps = 0

        # P3 + Stage 18: 缓存驾驶室、6-5 止回球、双突跳安全阀与煤水车全部 19 个独立可动操纵件/仪表/阀瓣关节地址
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
            "j_dome_lid",
            "j_safety_poppet_1",
            "j_safety_poppet_2",
            "j_pump_ball_suc",
            "j_pump_ball_del",
            "j_tender_hp_link",
            "j_tender_handpump",
            "j_tender_drain_plug",
            "j_tender_coal_cover",
            "j_tender_fuel_valve",
            "j_tender_fuel_cap",
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._vis_joints[jname] = (int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid]))
        jid_hpr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_tender_hp_ram")
        self.qpos_hp_ram = int(self.model.jnt_qposadr[jid_hpr]) if jid_hpr >= 0 else -1
        self.qvel_hp_ram = int(self.model.jnt_dofadr[jid_hpr]) if jid_hpr >= 0 else -1
        self._hp_psi = 0.0
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
        alpha = min(1.0, dt * 24.0)
        prev_psi = self._hp_psi
        psi = prev_psi + alpha * (hp_tgt - prev_psi)
        self._hp_psi = psi

        # 31-5 + 31-7x2 + 31-3 + 31-2 真 1-DOF 浮动支点四连杆解析闭环解 (零重合残差!)
        sin_ql = max(-0.99, min(0.99, 0.5 * (1.0 - math.cos(psi))))
        q_link = math.asin(sin_ql)
        q_lev = psi - q_link
        q_ram = 0.020 * (math.cos(q_link) - 1.0) - 0.010 * math.sin(psi)

        sin_ql_prev = max(-0.99, min(0.99, 0.5 * (1.0 - math.cos(prev_psi))))
        q_link_prev = math.asin(sin_ql_prev)
        q_ram_prev = 0.020 * (math.cos(q_link_prev) - 1.0) - 0.010 * math.sin(prev_psi)
        self._last_v_hand_ram = (q_ram - q_ram_prev) / max(1e-6, dt)
        if self.qpos_hp_ram >= 0:
            self.data.qpos[self.qpos_hp_ram] = q_ram
        if self.qvel_hp_ram >= 0:
            self.data.qvel[self.qvel_hp_ram] = self._last_v_hand_ram

        for j_hp_name, q_hp_val, q_hp_prev in (
            ("j_tender_hp_link", q_link, q_link_prev),
            ("j_tender_handpump", q_lev, prev_psi - q_link_prev),
        ):
            hp_addrs = self._vis_joints.get(j_hp_name)
            if hp_addrs is not None:
                self.data.qpos[hp_addrs[0]] = q_hp_val
                self.data.qvel[hp_addrs[1]] = (q_hp_val - q_hp_prev) / max(1e-6, dt)

        sv_lift_m = max(0.0, min(0.00075, float(getattr(tel, "safety_valve_lift_mm", 0.0)) * 1e-3))
        pb_lifts = getattr(self.thermo, "pump_ball_lifts_mm", [0.0, 0.0, 0.0])
        b_suc_m = max(0.0, min(0.0008, float(pb_lifts[0]) * 1e-3)) if len(pb_lifts) >= 1 else 0.0
        b_del_m = max(0.0, min(0.0008, float(pb_lifts[1]) * 1e-3)) if len(pb_lifts) >= 2 else 0.0
        targets = {
            "j_cab_regulator": 1.15 * self.ctrl.regulator,
            "j_cab_blower": 3.14 * self.ctrl.blower_valve,
            "j_cab_whistle": +0.38 * self.ctrl.whistle_valve,
            "j_cab_sight_drain": 0.0016 if bool(getattr(self.ctrl, "sight_drain_valve", False)) else 0.0,
            "j_cab_pg_needle": -2.10 + p_norm * 4.20,
            "j_cab_blowdown": 3.14 * float(getattr(self.ctrl, "blowdown_valve", 0.0)),
            "j_cab_fire_door": +1.40 * self.ctrl.firebox_door,
            "j_cab_bypass": +1.30 * self.ctrl.bypass_valve,
            "j_dome_lid": 1.15 if (bool(getattr(self.ctrl, "dome_lid_open", False)) or bool(getattr(self.ctrl, "cab_roof_open", False))) else 0.0,
            "j_safety_poppet_1": sv_lift_m,
            "j_safety_poppet_2": sv_lift_m * 0.94,
            "j_pump_ball_suc": b_suc_m,
            "j_pump_ball_del": b_del_m,
            "j_tender_drain_plug": 0.0050 if bool(getattr(self.ctrl, "tender_drain_open", False)) else 0.0,
            "j_tender_coal_cover": 0.026 if open_water_bay else 0.0,
            "j_tender_fuel_valve": 1.50 * self.ctrl.fuel_valve,
            "j_tender_fuel_cap": 0.0045 if self.ctrl.fuel_cap_open else 0.0,
        }
        for jname, tgt in targets.items():
            addrs = self._vis_joints.get(jname)
            if addrs is not None:
                qadr, dadr = addrs
                if jname == "j_cab_fire_door":
                    # 12-1/12-2 铸铁铰接火室门: 依赖 MJCF 隐式阻尼 (damping=0.0025) + 显式卡扣弹簧力矩，无条件数值稳定!
                    q_fd = float(self.data.qpos[qadr])
                    self.data.qfrc_applied[dadr] = 0.032 * (tgt - q_fd)
                elif jname.startswith("j_safety_poppet") or jname.startswith("j_pump_ball_"):
                    self.data.qpos[qadr] = tgt
                    self.data.qvel[dadr] = 0.0
                else:
                    curr = float(self.data.qpos[qadr])
                    nxt = curr + alpha * (tgt - curr)
                    self.data.qpos[qadr] = nxt
                    self.data.qvel[dadr] = 0.0

    def apply_hot_preset(self) -> None:
        self.thermo.reset_hot_ready()
        self.ctrl.fuel_valve = 1.0
        self.ctrl.burner_lit = True
        self.ctrl.chimney_fan = False
        self.ctrl.blower_valve = 0.25
        self.ctrl.reverser = 0.85
        self.ctrl.drain_cock = True
        self.ctrl.thermal_warp = 1.0

    def set_gpu_monitor_enabled(self, enabled: bool) -> None:
        self.enable_gpu_monitor = bool(enabled)
        if hasattr(self.thermo, "set_gpu_monitor_enabled"):
            self.thermo.set_gpu_monitor_enabled(enabled)
        elif hasattr(self, "_fallback_gpu_mon"):
            self._fallback_gpu_mon.set_enabled(bool(enabled))

    def _evaluate_wheel_rail_derailment(self) -> tuple[bool, str]:
        """
        检查全车 10 根轮对与钢轨/枕木/地面的真实三维接触与几何偏离状态:
          - 若任一车轮碰撞体与枕木 (sleeper_*) 或地面 (ground) 发生物理接触，或横向越出钢轨头 >4.8mm，或垂向跌落轨面 >1.6mm，
            判定为真实物理脱轨落枕！
        """
        if self.is_roller_mode:
            self._derail_consec_steps = 0
            return False, ""

        hit_sleeper_wheel = ""
        for ci in range(int(self.data.ncon)):
            con = self.data.contact[ci]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in self._wheel_geom_ids and g2 in self._offrail_geom_ids) or (
                g2 in self._wheel_geom_ids and g1 in self._offrail_geom_ids
            ):
                wg = g1 if g1 in self._wheel_geom_ids else g2
                hit_sleeper_wheel = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, wg) or "wheel"
                break

        worst_reason = ""
        raw_offrail = False
        if hit_sleeper_wheel:
            raw_offrail = True
            worst_reason = f"{hit_sleeper_wheel} 接触枕木/道床"

        for wbid, wname, r_nom in self._wheelset_specs:
            wx, wy, wz = float(self.data.xpos[wbid, 0]), float(self.data.xpos[wbid, 1]), float(self.data.xpos[wbid, 2])
            if not (math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz)):
                raw_offrail = True
                worst_reason = f"{wname} 数値発散保護"
                break
            proj = self.track_profile.project_xy(wx, wy)
            lat_err_m = proj.lat_err_m
            drop_m = (proj.z_rail_top_m + r_nom) - wz
            if abs(lat_err_m) > 0.0048 or drop_m > 0.0018:
                raw_offrail = True
                worst_reason = f"{wname} (横偏 {lat_err_m*1000.0:+.1f}mm, 沈下 {drop_m*1000.0:+.1f}mm)"
                break

        if raw_offrail:
            self._derail_consec_steps += 1
        else:
            self._derail_consec_steps = max(0, self._derail_consec_steps - 1)

        if self._derail_consec_steps >= 4:
            self.derailed = True
            if worst_reason:
                self.derail_reason = worst_reason
        elif self._derail_consec_steps == 0:
            self.derailed = False
            self.derail_reason = ""
        return self.derailed, self.derail_reason

    def reset_odometer(self) -> None:
        self.distance_m = 0.0
        self.lap_count = 0.0
        self._prev_loco_xy = None

    def update_gpu_range_prediction(self, ai_driver: bool = False, n_ensemble: int = 24) -> Dict[str, Any]:
        """在 RTX 5080 上从当前实时状态并行投射未来续航区间 (异步/低频调用, <12ms)"""
        if self._range_engine is None:
            self._range_engine = GPUConsistEnduranceEngine()
        core = self.thermo.core1d if hasattr(self.thermo, "core1d") else self.thermo
        lx, ly = float(self.data.xpos[self.loco_bid, 0]), float(self.data.xpos[self.loco_bid, 1])
        proj = self.track_profile.project_xy(lx, ly)
        vg_now = math.hypot(float(self.data.qvel[0]), float(self.data.qvel[1])) if not self.is_roller_mode else abs(float(self.data.sensordata[self.s_w3])) * 0.0219
        live_st = {
            "T_boiler_c": float(core.T_boiler_c),
            "P_boiler_pa": float(core.P_boiler),
            "M_water_kg": float(core.M_water_kg),
            "tender_water_ml": float(core.tender_water_ml),
            "alcohol_ml": float(core.alcohol_ml + core.sump_fuel_ml),
            "lube_oil_ml": float(core.lube_oil_ml),
            "T_cyl_wall_c": float(core.T_cyl_wall_c),
            "v_ground_ms": vg_now,
            "s_track_m": proj.s_m,
        }
        ctrl_st = {
            "tank_spec_2010": bool(getattr(self.ctrl, "tank_spec_2010", False)),
            "ai_driver": ai_driver,
            "regulator": float(self.ctrl.regulator),
            "reverser": float(self.ctrl.reverser),
            "bypass_valve": float(self.ctrl.bypass_valve),
            "blower_valve": float(self.ctrl.blower_valve),
            "trailing_cars": len(self._fcar_specs),
            "trailing_load_kg": float(getattr(self.ctrl, "trailing_load_kg", 0.0)),
            "ambient_temp_c": float(getattr(self.ctrl, "ambient_temp_c", 25.0)),
            "headwind_m_s": float(getattr(self.ctrl, "headwind_m_s", 0.0)),
            "grade_deg": float(self.ctrl.grade_deg),
        }
        t_mode = "oval_grade" if self.is_roller_mode else self.track_mode
        self._live_range_cache = self._range_engine.predict_live_remaining_range(
            track_mode=t_mode,
            live_state=live_st,
            ctrl_state=ctrl_st,
            n_ensemble=n_ensemble,
        )
        return self._live_range_cache

    def step_once(self) -> ThermoTelemetry:
        sd = self.data.sensordata
        dt = float(self.model.opt.timestep)
        omega_3 = float(sd[self.s_w3])
        v_vL = float(self.data.qvel[self.qvel_valL])
        v_vR = float(self.data.qvel[self.qvel_valR])

        # 读取全编组 10 轴板簧/螺旋簧实时弹性形变位移 (m)
        susp_z_m = [
            float(self.data.qpos[qadr]) if qadr >= 0 else 0.00065
            for qadr in self._susp_qpos_addrs
        ]
        if self.use_gpu_3d and hasattr(self.thermo, "last_susp_z_m"):
            self.thermo.last_susp_z_m = susp_z_m

        # 提取车体真实 3D 俯仰角、对地切向速度、纵向加速度与弯道离心横向加速度
        pitch_rad = 0.0
        accel_x = 0.0
        accel_lat = 0.0
        v_ground_arg: Optional[float] = None
        if not self.is_roller_mode:
            rot_loco = self.data.xmat[self.loco_bid].reshape(3, 3)
            # MuJoCo xmat 第 0 列为车架 +X 前向单位向量; 上坡时 +X 的 Z 分量 rot_loco[2, 0] > 0 -> pitch_rad > 0!
            pitch_rad = math.asin(max(-0.35, min(0.35, float(rot_loco[2, 0]))))
            e_fwd = rot_loco[:, 0]
            v_world = self.data.qvel[0:3]
            fwd_vel = float(np.dot(v_world, e_fwd))
            v_ground_arg = fwd_vel
            accel_x = (fwd_vel - self._prev_fwd_vel) / max(1e-6, dt)
            self._prev_fwd_vel = fwd_vel
            yaw_rate = float(self.data.qvel[5])
            accel_lat = fwd_vel * yaw_rate

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
            v_ground_ms=v_ground_arg,
            accel_lat=accel_lat,
        )
        if self.use_gpu_3d:
            step_kwargs["q_ram"] = float(self.data.qpos[self.qpos_pump])
        F_L, F_R, F_pump, tel = self.thermo.step(**step_kwargs)
        self.last_f_pump = float(F_pump)
        if not any(abs(v) > 1e-6 for v in tel.spring_deflect_mm):
            tel.spring_deflect_mm = [round(abs(z) * 1000.0, 3) for z in susp_z_m]
        if self._lat_qpos_addrs:
            tel.hunting_lat_mm = round(max(abs(float(self.data.qpos[qa])) for qa in self._lat_qpos_addrs) * 1000.0, 3)
        if not self.is_roller_mode and self.data.ncon > 0:
            rot_m = self.data.xmat[self.loco_bid].reshape(3, 3)
            e_lat = rot_m[:, 1]
            cw = np.zeros(6, dtype=np.float64)
            y_sum, q_sum = 0.0, 0.0
            for ci in range(self.data.ncon):
                con = self.data.contact[ci]
                if (con.geom1 in self._wheel_geom_ids) or (con.geom2 in self._wheel_geom_ids):
                    mujoco.mj_contactForce(self.model, self.data, ci, cw)
                    fw = con.frame.reshape(3, 3).T @ cw[:3]
                    if con.geom1 in self._wheel_geom_ids:
                        fw = -fw
                    y_sum += abs(float(np.dot(fw, e_lat)))
                    q_sum += max(0.0, float(fw[2]))
            tel.nadal_lv_ratio = round(min(2.5, y_sum / max(2.0, q_sum)), 3)

        self.data.ctrl[self.act_fL] = F_L
        self.data.ctrl[self.act_fR] = F_R
        self.data.ctrl[self.act_rev] = self.ctrl.reverser * 0.34

        # 施加平面滑阀背压摩擦反力 (O3-2)、给水泵双向抽吸/压送液压负载与运行阻力
        self.data.qfrc_applied[self.qvel_valL] = tel.valve_friction_L
        self.data.qfrc_applied[self.qvel_valR] = tel.valve_friction_R
        self.data.qfrc_applied[self.qvel_pump] = F_pump
        if self.qvel_drain >= 0:
            self.data.qfrc_applied[self.qvel_drain] = 0.032 if self.ctrl.drain_cock else 0.0
        if self.is_roller_mode:
            self.data.qfrc_applied[self.qvel_w3] = -0.0022 * omega_3 - 0.00012 * omega_3 * abs(omega_3)
        else:
            # 轨道模式: 1) 水/甲醇消耗与挂车载重实时变质量回写; 2) Polach 钢轨水油膜动态黏着摩擦;
            #          3) 第一性原理赫兹滚动+轴承+钝体风阻+附加坡道切向重力 (严禁硬编码 8.80*v^2!)
            core_th = self.thermo.core1d if hasattr(self.thermo, "core1d") else self.thermo
            m_loco_cur = self._loco_dry_mass + float(core_th.M_water_kg)
            self.model.body_mass[self.loco_bid] = m_loco_cur
            self.model.body_inertia[self.loco_bid] = self._loco_base_inertia * (m_loco_cur / max(0.5, self._loco_base_mass))

            extra_load_kg = max(0.0, float(getattr(self.ctrl, "trailing_load_kg", 0.0)))
            m_tdr_cur = 0.0
            if self.tender_bid >= 0 and self._tender_base_inertia is not None:
                fluid_kg = (float(core_th.tender_water_ml) * 0.997 + float(core_th.alcohol_ml + core_th.sump_fuel_ml) * 0.787) * 1e-3
                add_tdr_load = extra_load_kg if not self._fcar_specs else 0.0
                m_tdr_cur = self._tender_dry_mass + fluid_kg + add_tdr_load
                self.model.body_mass[self.tender_bid] = m_tdr_cur
                self.model.body_inertia[self.tender_bid] = self._tender_base_inertia * (m_tdr_cur / max(0.5, self._tender_base_mass))

            m_fcars_tot = 0.0
            if self._fcar_specs:
                per_car_load = extra_load_kg / len(self._fcar_specs)
                for cbid, _, dry_cm, base_ci in self._fcar_specs:
                    mc = dry_cm + per_car_load
                    self.model.body_mass[cbid] = mc
                    self.model.body_inertia[cbid] = base_ci * (mc / max(0.2, dry_cm))
                    m_fcars_tot += mc
            total_consist_kg = m_loco_cur + m_tdr_cur + m_fcars_tot + 2.15  # 含 10~14 组轮对轴箱连杆刚体总质量

            is_der, der_msg = self._evaluate_wheel_rail_derailment()
            tel.derailed = is_der
            tel.derail_reason = der_msg
            vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
            v_spd = math.hypot(vx, vy)
            v_fwd_now = float(v_ground_arg or 0.0)

            v_slip_now = omega_3 * 0.0219 - v_fwd_now
            mu_rail = compute_rail_adhesion_mu(
                v_slip_ms=v_slip_now,
                drain_cock_open=bool(self.ctrl.drain_cock),
                exhaust_gs=float(tel.exhaust_g_s),
                lubricator_oil_pct=float(tel.lubricator_oil_pct),
            )
            for gid in self._drv_tread_geom_ids:
                self.model.geom_friction[gid, 0] = mu_rail
            for gid in self._rail_geom_ids:
                self.model.geom_friction[gid, 0] = mu_rail
            tel.rail_adhesion_mu = round(mu_rail, 3)

            cur_xyz = (float(self.data.xpos[self.loco_bid, 0]), float(self.data.xpos[self.loco_bid, 1]), float(self.data.xpos[self.loco_bid, 2]))
            if self._prev_loco_xy is not None and not is_der:
                ds_3d = math.sqrt(
                    (cur_xyz[0] - self._prev_loco_xy[0]) ** 2
                    + (cur_xyz[1] - self._prev_loco_xy[1]) ** 2
                    + (cur_xyz[2] - self._prev_loco_xy[2]) ** 2
                )
                if ds_3d < 0.25:
                    self.distance_m += ds_3d
            self._prev_loco_xy = cur_xyz
            if self.track_profile.is_closed:
                self.lap_count = self.distance_m / self.track_profile.lap_length_m

            if is_der:
                if v_spd > 1e-5:
                    f_plow = min(48.0, 26.0 + 65.0 * v_spd)
                    self.data.qfrc_applied[0] = -f_plow * (vx / v_spd)
                    self.data.qfrc_applied[1] = -f_plow * (vy / v_spd)
                else:
                    self.data.qfrc_applied[0] = 0.0
                    self.data.qfrc_applied[1] = 0.0
                self.data.qfrc_applied[self.qvel_w3] = -0.35 * omega_3 - 0.08 * math.tanh(omega_3 * 4.0)
                tel.train_resistance_n = round(26.0 + 65.0 * v_spd, 3)
            else:
                cur_curv_abs = abs(self.track_profile.project_xy(cur_xyz[0], cur_xyz[1]).curvature_inv_m)
                f_res = compute_physical_train_resistance_n(
                    v_ground_ms=v_fwd_now,
                    total_mass_kg=total_consist_kg,
                    ambient_temp_c=float(getattr(self.ctrl, "ambient_temp_c", 25.0)),
                    headwind_m_s=float(getattr(self.ctrl, "headwind_m_s", 0.0)),
                    n_trailing_cars=len(self._fcar_specs),
                    include_curve_flange_1d=cur_curv_abs,
                )
                sin_extra_grade = math.sin(math.radians(max(-15.0, min(15.0, float(self.ctrl.grade_deg)))))
                e_fwd_xy = self.data.xmat[self.loco_bid].reshape(3, 3)[:, 0]
                ex_n = math.hypot(float(e_fwd_xy[0]), float(e_fwd_xy[1]))
                tx_u = float(e_fwd_xy[0]) / max(1e-6, ex_n)
                ty_u = float(e_fwd_xy[1]) / max(1e-6, ex_n)

                # 将车架+轮对附加坡道重力分量与第一性原理运行阻力共同施加于平动自由度
                m_loco_plus_wheels = m_loco_cur + 1.55 + (0.0 if self.tender_bid >= 0 else extra_load_kg)
                fx_tot = -m_loco_plus_wheels * 9.81 * sin_extra_grade * tx_u
                fy_tot = -m_loco_plus_wheels * 9.81 * sin_extra_grade * ty_u
                if v_spd > 1e-5:
                    fx_tot -= f_res * (vx / v_spd)
                    fy_tot -= f_res * (vy / v_spd)
                self.data.qfrc_applied[0] = fx_tot
                self.data.qfrc_applied[1] = fy_tot

                if self._tender_dof0 >= 0 and self.tender_bid >= 0:
                    et = self.data.xmat[self.tender_bid].reshape(3, 3)[:, 0]
                    et_n = max(1e-6, math.hypot(float(et[0]), float(et[1])))
                    m_tdr_tot = m_tdr_cur + 0.34
                    self.data.qfrc_applied[self._tender_dof0] = -m_tdr_tot * 9.81 * sin_extra_grade * (float(et[0]) / et_n)
                    self.data.qfrc_applied[self._tender_dof0 + 1] = -m_tdr_tot * 9.81 * sin_extra_grade * (float(et[1]) / et_n)

                for cbid, cdof0, _, _ in self._fcar_specs:
                    ec = self.data.xmat[cbid].reshape(3, 3)[:, 0]
                    ec_n = max(1e-6, math.hypot(float(ec[0]), float(ec[1])))
                    mc_tot = float(self.model.body_mass[cbid]) + 0.13
                    self.data.qfrc_applied[cdof0] = -mc_tot * 9.81 * sin_extra_grade * (float(ec[0]) / ec_n)
                    self.data.qfrc_applied[cdof0 + 1] = -mc_tot * 9.81 * sin_extra_grade * (float(ec[1]) / ec_n)

                self.data.qfrc_applied[self.qvel_w3] = -0.0031 * omega_3 - 0.00018 * omega_3 * abs(omega_3)
                tel.train_resistance_n = round(f_res + total_consist_kg * 9.81 * sin_extra_grade, 3)

        tel.distance_m = round(self.distance_m, 3)
        tel.scale_distance_km = round(self.distance_m * 30.0 / 1000.0, 4)
        tel.lap_count = round(self.lap_count, 3)
        tel.predicted_range_m = float(self._live_range_cache.get("p50_m", 0.0))
        tel.predicted_range_p10_m = float(self._live_range_cache.get("p10_m", 0.0))
        tel.predicted_range_p90_m = float(self._live_range_cache.get("p90_m", 0.0))
        tel.predicted_time_min = float(self._live_range_cache.get("p50_min", 0.0))
        tel.range_limiting_factor = str(self._live_range_cache.get("limiting_factor", "NOMINAL"))

        if not self.use_gpu_3d:
            tel.spring_deflect_mm = [round(abs(z) * 1000.0, 3) for z in susp_z_m]
            if hasattr(self, "_fallback_gpu_mon") and self._fallback_gpu_mon.enabled:
                tel.gpu_monitor = self._fallback_gpu_mon.snapshot().to_dict()

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
    enable_gpu_monitor: bool = False,
) -> None:
    import mujoco.viewer

    runner = CoSimRunner(
        xml_path,
        preset=preset,
        use_gpu_3d=use_gpu_3d,
        cfd_subdiv=16,
        snapshot_every_cfd=2,
        enable_gpu_monitor=enable_gpu_monitor,
    )
    hw_guard = HardwareThermalPowerGuard(max_cpu_c=78.0, max_gpu_c=72.0, max_gpu_w=230.0)
    human_agent = HumanEngineerAgent()
    oracle_agent = IdealOracleController()
    ai_mode = ai_mode_init if ai_mode_init in ("off", "human", "oracle") else ("oracle" if auto_fireman else "off")
    copilot = AsyncBonsaiCopilot(enabled=enable_copilot, interval_sec=25.0) if enable_copilot else None

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

    target_hz = 36.0 if use_gpu_3d else 60.0
    dt = runner.model.opt.timestep
    steps_per_frame = max(1, int(round((1.0 / target_hz) / dt)))
    last_print = 0.0
    frame_idx = 0

    with mujoco.viewer.launch_passive(runner.model, runner.data, key_callback=on_key) as viewer:
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -16.0
        viewer.cam.distance = 0.36
        viewer.cam.lookat[:] = [0.04, 0.0, 0.03]

        while viewer.is_running():
            frame_idx += 1
            if frame_idx % 18 == 0 and hw_guard.is_hot_or_high_power():
                time.sleep(0.035)
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
            sleep_s = (1.0 / target_hz) - elapsed
            if sleep_s > 0.002:
                time.sleep(sleep_s)
            else:
                time.sleep(0.003)
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
    parser.add_argument("--gpu-3d", action="store_true", help="启用本地 RTX 5080 578万体素八大零件域真 3D 多相热流 + 热弹性 + 7-PINO 求解器")
    parser.add_argument("--gpu-monitor", action="store_true", help="开启可选实时 GPU SM/显存/温度/功耗与 CUDA 算子耗时监控")
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
            enable_gpu_monitor=args.gpu_monitor,
        )


if __name__ == "__main__":
    main()