#!/usr/bin/env python3
"""
sim/gpu_3d_server.py
================================================================================
Aster Hobby JNR D51 — 本地 NVIDIA RTX 5080 571万体素真 3D 多相热流 HTTP + WebSocket 一体化服务
默认监听: http://127.0.0.1:8765  (同端口同时承载 Web 3D 静态页面与 /ws 实时 WebSocket 推流!)

核心特性:
  - 原生兼容 HTTP/1.1 Keep-Alive 与 WebSocket 协议升级 (彻底根除 VS Code / Remote-SSH 端口探测或浏览器直连触发的
    `InvalidUpgrade: invalid Connection header: keep-alive` 报错)；
  - 支持两种打开方式:
      方式 A (单终端一键启动): 运行本服务后直接在浏览器打开 http://127.0.0.1:8765 即可加载完整 3D 界面并直连 GPU！
      方式 B (双终端前端热更新): 配合 `npm --prefix web run dev` (5173 端口) 自动握手 ws://127.0.0.1:8765！
  - 在本机 RTX 5080 CUDA + MuJoCo C++ (2000 Hz) 上实时双向耦合步进五大三维零件流体场；
  - 无浏览器连接时自动挂起休眠 (0% GPU 占用)，浏览器打开网页即毫秒级自动点亮「RTX 5080 LIVE 3D · 5.71M VOX」！
================================================================================
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import mimetypes
import os
import signal
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional, Set

from aiohttp import WSMsgType, web
import mujoco
import numpy as np
import torch

from sim.bonsai_copilot import AsyncBonsaiCopilot, CopilotDiagnosis
from sim.human_engineer_agent import HumanEngineerAgent
from sim.hw_guard import HardwareThermalPowerGuard, apply_cpu_thread_and_nice_caps, quarantine_overclocked_cpus
from sim.ideal_oracle_controller import IdealOracleController
from sim.run_cosim import CoSimRunner
from sim.thermo_1d import CabControls, ThermoTelemetry

_QUARANTINED_OC_CPUS = quarantine_overclocked_cpus(safe_max_khz=5_200_000)
apply_cpu_thread_and_nice_caps(max_threads=6, nice_delta=2)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST_DIR = (PROJECT_ROOT / "web" / "dist").resolve()
SERVER_LOG_PATH = PROJECT_ROOT / "logs" / "gpu_server.log"


class _SafeTeeStream:
    """
    保护后台常驻进程不因 SSH / IDE 伪终端关闭 (/dev/pts/N deleted -> OSError EIO/EPIPE) 而崩溃！
    所有标准输出/错误同时落盘至 logs/gpu_server.log，并在原终端可用时同步回显。
    """

    def __init__(self, orig_stream: Any, log_path: Path):
        self._orig = orig_stream
        self._orig_dead = False
        self._log_path = log_path
        self._at_line_start = True
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: str) -> int:
        if not data:
            return 0
        if not self._orig_dead and self._orig is not None:
            try:
                self._orig.write(data)
            except OSError:
                self._orig_dead = True
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            out_chunks: list[str] = []
            for part in data.splitlines(keepends=True):
                if self._at_line_start and part.strip():
                    out_chunks.append(f"[{ts}] {part}")
                else:
                    out_chunks.append(part)
                self._at_line_start = part.endswith("\n")
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write("".join(out_chunks))
        except OSError:
            pass
        return len(data)

    def flush(self) -> None:
        if not self._orig_dead and self._orig is not None:
            try:
                self._orig.flush()
            except OSError:
                self._orig_dead = True

    def isatty(self) -> bool:
        return False


def _sanitize_json_value(obj: Any) -> Any:
    """递归过滤 NaN / Inf，确保 json.dumps(allow_nan=False) 100% 符合浏览器 JSON.parse 规范。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, dict):
        return {k: _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(x) for x in obj]
    return obj


def _terminate_stale_gpu_servers(my_pid: int) -> None:
    """启动前自动清理同用户残留的旧 sim.gpu_3d_server 进程，杜绝 Address already in use 与僵尸 PTY 进程占口。"""
    uid = os.getuid()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return
    killed_pids: list[int] = []
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == my_pid:
            continue
        try:
            if p.stat().st_uid != uid:
                continue
            cmdline = p.joinpath("cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            if "sim.gpu_3d_server" in cmdline or "sim/gpu_3d_server.py" in cmdline:
                os.kill(pid, signal.SIGTERM)
                killed_pids.append(pid)
        except Exception:
            continue
    if killed_pids:
        for _ in range(25):
            alive = [pid for pid in killed_pids if Path(f"/proc/{pid}").exists()]
            if not alive:
                break
            time.sleep(0.1)

SCENE_XML_MAP = {
    "roller": PROJECT_ROOT / "models" / "d51_full.xml",
    "track": PROJECT_ROOT / "models" / "d51_track.xml",
    "curve": PROJECT_ROOT / "models" / "d51_r2m.xml",
}

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("model/stl", ".stl")


def pack_frame_payload(
    scene_name: str,
    runner: CoSimRunner,
    tel: ThermoTelemetry,
    ai_mode: str,
    ai_status: str = "",
    copilot_diag: Optional[CopilotDiagnosis] = None,
) -> str:
    sp = runner.thermo.spatial
    b64_slice = base64.b64encode(bytes(sp.midplane_temp_u8)).decode("ascii") if sp.midplane_temp_u8 else ""
    b64_wave = base64.b64encode(bytes(sp.tender_wave_u8)).decode("ascii") if getattr(sp, "tender_wave_u8", None) else ""
    qpos_list = [round(float(x), 6) for x in runner.data.qpos]

    payload: Dict[str, Any] = {
        "type": "frame",
        "scene": scene_name,
        "nq": int(runner.model.nq),
        "qpos": qpos_list,
        "autoFireman": ai_mode != "off",
        "aiMode": ai_mode,
        "aiStatus": ai_status,
        "ctrl": {
            "fuelValve": round(runner.ctrl.fuel_valve, 3),
            "burnerLit": bool(runner.ctrl.burner_lit),
            "chimneyFan": bool(runner.ctrl.chimney_fan),
            "blowerValve": round(runner.ctrl.blower_valve, 3),
            "regulator": round(runner.ctrl.regulator, 3),
            "reverser": round(runner.ctrl.reverser, 3),
            "drainCock": bool(runner.ctrl.drain_cock),
            "bypassValve": round(runner.ctrl.bypass_valve, 3),
            "whistleValve": round(runner.ctrl.whistle_valve, 3),
            "blowdownValve": round(runner.ctrl.blowdown_valve, 3),
            "fireboxDoor": round(runner.ctrl.firebox_door, 3),
            "fuelCapOpen": bool(runner.ctrl.fuel_cap_open),
            "waterLidOpen": bool(runner.ctrl.water_lid_open),
            "waterFillValve": round(float(getattr(runner.ctrl, "water_fill_valve", 0.0)), 3),
            "tenderDrainOpen": bool(getattr(runner.ctrl, "tender_drain_open", False)),
            "sightDrainValve": bool(getattr(runner.ctrl, "sight_drain_valve", False)),
            "cabRoofOpen": bool(getattr(runner.ctrl, "cab_roof_open", False)),
            "domeLidOpen": bool(getattr(runner.ctrl, "dome_lid_open", False)),
            "coalCoverOpen": bool(getattr(runner.ctrl, "coal_cover_open", False)),
            "gradeDeg": round(runner.ctrl.grade_deg, 2),
            "thermalWarp": round(runner.ctrl.thermal_warp, 1),
        },
        "tel": {
            "simTime": round(tel.sim_time, 3),
            "boilerTempC": round(tel.boiler_temp_c, 2),
            "boilingProgressPct": round(tel.boiling_progress_pct, 1),
            "boilerPressureKgf": round(tel.boiler_pressure_kgf, 3),
            "pgBourdonKgf": round(float(getattr(tel, "pg_bourdon_kgf", tel.boiler_pressure_kgf)), 3),
            "turretPressureKgf": round(float(getattr(tel, "turret_pressure_kgf", tel.boiler_pressure_kgf)), 3),
            "steamChestKgf": round(tel.steam_chest_kgf, 3),
            "superheatTempC": round(tel.superheat_temp_c, 1),
            "waterLevelPct": round(tel.water_level_pct, 2),
            "sightGlassLevelPct": round(float(getattr(tel, "sight_glass_level_pct", tel.water_level_pct)), 2),
            "sightGlassMeniscusZMm": round(float(getattr(tel, "sight_glass_meniscus_z_mm", 55.5)), 2),
            "sightGlassVelMmS": round(float(getattr(tel, "sight_glass_vel_mm_s", 0.0)), 2),
            "sightGlassLumenConnected": bool(getattr(tel, "sight_glass_lumen_connected", True)),
            "trueWaterLevelPct": round(tel.true_water_level_pct, 2),
            "crownTempC": round(tel.crown_temp_c, 1),
            "crownSubmergedMm": round(tel.crown_submerged_mm, 2),
            "copperAnnealDamage": round(tel.copper_anneal_damage, 4),
            "sealHealth": round(tel.seal_health, 4),
            "alcoholMl": round(tel.alcohol_ml, 2),
            "tenderWaterMl": round(tel.tender_water_ml, 2),
            "tenderWaterTempC": round(tel.tender_water_temp_c, 2),
            "returnWaterMlS": round(tel.return_water_ml_s, 3),
            "waterFillMlS": round(float(getattr(tel, "water_fill_ml_s", 0.0)), 3),
            "tenderDrainMlS": round(float(getattr(tel, "tender_drain_ml_s", 0.0)), 3),
            "tenderOverflowMlS": round(float(getattr(tel, "tender_overflow_ml_s", 0.0)), 3),
            "mariotteBubbling": bool(tel.mariotte_bubbling),
            "mariotteFloodAlarm": bool(tel.mariotte_flood_alarm),
            "mariotteHeadspaceKpa": round(float(getattr(tel, "mariotte_headspace_kpa", 0.0)), 3),
            "blowdownGS": round(float(getattr(tel, "blowdown_g_s", 0.0)), 3),
            "sightPurgeGS": round(float(getattr(tel, "sight_purge_g_s", 0.0)), 3),
            "whistleSteamGS": round(tel.whistle_steam_g_s, 3),
            "exhaustGS": round(float(getattr(tel, "exhaust_g_s", 0.0)), 3),
            "blowerGS": round(float(getattr(tel, "blower_g_s", 0.0)), 3),
            "safetyGS": round(float(getattr(tel, "safety_g_s", 0.0)), 3),
            "boilerWaterMl": round(float(getattr(tel, "boiler_water_ml", 210.0)), 2),
            "waterFreeZMm": round(float(getattr(tel, "water_free_z_mm", sp.water_free_z_mm)), 2),
            "lubricatorOilPct": round(tel.lubricator_oil_pct, 2),
            "cylWallTempC": round(tel.cyl_wall_temp_c, 1),
            "condensateG": round(tel.condensate_g, 3),
            "safetyValveOpen": bool(tel.safety_valve_open),
            "dryFireAlarm": bool(tel.dry_fire_alarm),
            "flashSteamSpike": bool(tel.flash_steam_spike),
            "hydrolockL": bool(tel.hydrolock_L),
            "hydrolockR": bool(tel.hydrolock_R),
            "valveFrictionL": round(tel.valve_friction_L, 3),
            "valveFrictionR": round(tel.valve_friction_R, 3),
            "smokeboxVacuumPa": round(tel.smokebox_vacuum_pa, 1),
            "lutActive": bool(tel.lut_active),
            "pinoActive": bool(tel.pino_active),
            "steamDrynessDome": round(tel.steam_dryness_dome, 4),
            "steamDrynessChest": round(tel.steam_dryness_chest, 4),
            "heatInputW": round(tel.heat_input_w, 1),
            "boilerEfficiencyPct": round(float(getattr(tel, "boiler_efficiency_pct", 0.0)), 2),
            "shellHeatLossW": round(float(getattr(tel, "shell_heat_loss_w", 0.0)), 2),
            "shellRadLossW": round(float(getattr(tel, "shell_rad_loss_w", 0.0)), 2),
            "casingTempC": round(float(getattr(tel, "casing_temp_c", 25.0)), 1),
            "wickCapillaryMlS": round(float(getattr(tel, "wick_capillary_ml_s", 0.0)), 4),
            "wickSaturation": round(float(getattr(tel, "wick_saturation", 0.90)), 3),
            "wickTipTempC": round(float(getattr(tel, "wick_tip_temp_c", 25.0)), 1),
            "burnerLumenConnected": bool(getattr(tel, "burner_lumen_connected", True)),
            "feedwaterGS": round(tel.feedwater_g_s, 3),
            "steamGenGS": round(tel.steam_gen_g_s, 3),
            "steamOutGS": round(float(getattr(tel, "steam_out_g_s", tel.steam_gen_g_s)), 3),
            "cutoffPct": round(float(getattr(tel, "cutoff_pct", 0.0)), 1),
            "cylExpansionWorkW": round(float(getattr(tel, "cyl_expansion_work_w", 0.0)), 3),
            "specificSteamGKj": round(float(getattr(tel, "specific_steam_g_kj", 0.0)), 2),
            "pumpChamberKgf": round(float(getattr(tel, "pump_chamber_kgf", 0.0)), 3),
            "pumpSuctionVacKpa": round(float(getattr(tel, "pump_suction_vac_kpa", 0.0)), 2),
            "pumpShaftPowerW": round(float(getattr(tel, "pump_shaft_power_w", 0.0)), 4),
            "lubeOilTempC": round(float(getattr(tel, "lube_oil_temp_c", 25.0)), 1),
            "lubeOilViscosityMpas": round(float(getattr(tel, "lube_oil_viscosity_mpas", 1850.0)), 1),
            "lubeWaterMl": round(float(getattr(tel, "lube_water_ml", 0.64)), 3),
            "lubeOilMl": round(float(getattr(tel, "lube_oil_ml", 2.56)), 3),
            "lubeSettlingMmS": round(float(getattr(tel, "lube_settling_mm_s", 0.0)), 3),
            "lubeEmulsionPct": round(float(getattr(tel, "lube_emulsion_pct", 1.6)), 2),
            "lubeInterfaceZMm": round(float(getattr(tel, "lube_interface_z_mm", 2.60)), 2),
            "alcoholTempC": round(float(getattr(tel, "alcohol_temp_c", 25.0)), 1),
            "alcoholViscosityMpas": round(float(getattr(tel, "alcohol_viscosity_mpas", 1.08)), 3),
            "drainBlowdownGS": round(float(getattr(tel, "drain_blowdown_g_s", 0.0)), 3),
            "flueDeflectionMm": getattr(tel, "flue_deflection_mm", [0.0] * 5),
            "flueOvalizationMm": getattr(tel, "flue_ovalization_mm", [0.0] * 5),
            "crownBulgeMm": round(float(getattr(tel, "crown_bulge_mm", 0.0)), 4),
            "maxBoilerStressMpa": round(float(getattr(tel, "max_boiler_stress_mpa", 0.0)), 2),
            "copperYieldMpa": round(float(getattr(tel, "copper_yield_mpa", 185.0)), 1),
            "springDeflectMm": getattr(tel, "spring_deflect_mm", [0.0] * 10),
            "boilerExpansionMm": round(float(getattr(tel, "boiler_expansion_mm", 0.0)), 4),
            "rodKnockN": round(float(getattr(tel, "rod_knock_n", 0.0)), 3),
            "huntingLatMm": round(float(getattr(tel, "hunting_lat_mm", 0.0)), 3),
            "nadalLvRatio": round(float(getattr(tel, "nadal_lv_ratio", 0.0)), 3),
            "qFuelW": round(float(getattr(tel, "q_fuel_w", 0.0)), 2),
            "qEvapW": round(float(getattr(tel, "q_evap_w", 0.0)), 2),
            "qShW": round(float(getattr(tel, "q_sh_w", 0.0)), 2),
            "qSensibleW": round(float(getattr(tel, "q_sensible_w", 0.0)), 2),
            "qFlueLossW": round(float(getattr(tel, "q_flue_loss_w", 0.0)), 2),
            "qCasingLossW": round(float(getattr(tel, "q_casing_loss_w", 0.0)), 2),
            "qUnburnedW": round(float(getattr(tel, "q_unburned_w", 0.0)), 2),
            "energyBalanceErrPct": round(float(getattr(tel, "energy_balance_err_pct", 0.0)), 4),
            "derailed": bool(getattr(tel, "derailed", False)),
            "derailReason": str(getattr(tel, "derail_reason", "")),
            "pino7OpActive": bool(getattr(tel, "pino_7op_active", False)),
            "gpuMonitor": getattr(tel, "gpu_monitor", {}),
            "forceL": round(tel.force_L, 2),
            "forceR": round(tel.force_R, 2),
            "fPump": round(float(getattr(runner, "last_f_pump", 0.0)), 3),
            "wheelRpm": round(tel.wheel_rpm, 1),
            "scaleSpeedKmh": round(tel.scale_speed_kmh, 2),
        },
        "spatial": {
            "gpuActive": bool(sp.gpu_active),
            "pinoActive": bool(sp.pino_active),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "totalVoxels": int(sp.total_voxels),
            "gpuStepMs": float(sp.gpu_step_ms),
            "gpuHz": float(sp.gpu_hz),
            "stlPassagesVerified": bool(sp.stl_passages_verified),
            "openFlueCount": int(sp.open_flue_count),
            "airFuelPhi": float(sp.air_fuel_phi),
            "flameTiltDeg": float(sp.flame_tilt_deg),
            "wickTempsC": sp.wick_temps_c,
            "flueGasProfileC": sp.flue_gas_profile_c,
            "steamDrynessDome": float(sp.steam_dryness_dome),
            "boilingIntensity": float(sp.boiling_intensity),
            "waterFreeZMm": float(sp.water_free_z_mm),
            "waterSloshPitchDeg": float(sp.water_slosh_pitch_deg),
            "waterLayerTempsC": sp.water_layer_temps_c,
            "crownHotspotsC": sp.crown_hotspots_c,
            "shProfileTempC": sp.sh_profile_temp_c,
            "shProfileDryness": sp.sh_profile_dryness,
            "cylLPKgf": sp.cyl_L_p_kgf,
            "cylRPKgf": sp.cyl_R_p_kgf,
            "valvePortOpenL": sp.valve_port_open_L,
            "valvePortOpenR": sp.valve_port_open_R,
            "exhaustDryness": float(sp.exhaust_dryness),
            "oilFilmUm": float(sp.oil_film_um),
            "feedwaterVelMS": float(sp.feedwater_vel_m_s),
            "pumpChamberPKgf": float(sp.pump_chamber_p_kgf),
            "checkBallLiftsMm": sp.check_ball_lifts_mm,
            "checkBall5LiftsMm": getattr(sp, "check_ball_5_lifts_mm", sp.check_ball_lifts_mm),
            "lubeWaterFrac": float(sp.lube_water_frac),
            "pino4OpActive": bool(getattr(sp, "pino_4op_active", False)),
            "pino6OpActive": bool(getattr(sp, "pino_6op_active", False)),
            "pino7OpActive": bool(getattr(sp, "pino_7op_active", False)),
            "burnerLumenConnected": bool(getattr(sp, "burner_lumen_connected", True)),
            "sightGlassLumenConnected": bool(getattr(sp, "sight_glass_lumen_connected", True)),
            "fluidLvsVerified": bool(getattr(sp, "fluid_lvs_verified", True)),
            "fluidLvsOpenEdges": int(getattr(sp, "fluid_lvs_open_edges", 35)),
            "fluidLvsTotalEdges": int(getattr(sp, "fluid_lvs_total_edges", 35)),
            "edgeConductances": getattr(sp, "edge_conductances", {}),
            "wickCapillaryMlS": float(getattr(sp, "wick_capillary_ml_s", 0.0)),
            "wickSaturation": float(getattr(sp, "wick_saturation", 0.90)),
            "wickTipTempC": float(getattr(sp, "wick_tip_temp_c", 25.0)),
            "casingTempC": float(getattr(sp, "casing_temp_c", 25.0)),
            "shellHeatLossW": float(getattr(sp, "shell_heat_loss_w", 0.0)),
            "shellRadLossW": float(getattr(sp, "shell_rad_loss_w", 0.0)),
            "boilerEfficiencyPct": float(getattr(sp, "boiler_efficiency_pct", 0.0)),
            "insulationCoverage": float(getattr(sp, "insulation_coverage", 0.86)),
            "tenderBayTempC": float(getattr(sp, "tender_bay_temp_c", 25.0)),
            "tenderWaveRmsMm": float(getattr(sp, "tender_wave_rms_mm", 0.0)),
            "flueDeflectionMm": getattr(sp, "flue_deflection_mm", [0.0] * 5),
            "flueOvalizationMm": getattr(sp, "flue_ovalization_mm", [0.0] * 5),
            "crownBulgeMm": float(getattr(sp, "crown_bulge_mm", 0.0)),
            "maxBoilerStressMpa": float(getattr(sp, "max_boiler_stress_mpa", 0.0)),
            "copperYieldMpa": float(getattr(sp, "copper_yield_mpa", 185.0)),
            "gpuMonitor": getattr(sp, "gpu_monitor", {}),
            "tenderWaveB64": b64_wave,
            "midplaneTempB64": b64_slice,
        },
        "copilot": copilot_diag.to_dict() if copilot_diag is not None else None,
    }
    try:
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)
    except ValueError:
        return json.dumps(_sanitize_json_value(payload), separators=(",", ":"), allow_nan=False)


class GPU3DRealtimeBridge:
    def __init__(
        self,
        scene: str = "roller",
        preset: str = "hot",
        enable_copilot: bool = True,
        enable_gpu_monitor: bool = False,
    ):
        self.scene = scene if scene in SCENE_XML_MAP else "roller"
        self.preset = preset
        # 提升 GPU 利用率至 125Hz (cfd_subdiv=16) 推进 8 大三维体素/壳梁域 + 7 算子 PINO，同时消除 48 步跨度引起的面板锯齿!
        self.runner = CoSimRunner(
            SCENE_XML_MAP[self.scene],
            preset=preset,
            use_gpu_3d=True,
            cfd_subdiv=16,
            snapshot_every_cfd=2,
            enable_gpu_monitor=enable_gpu_monitor,
        )
        self.human_agent = HumanEngineerAgent()
        self.oracle_agent = IdealOracleController()
        self.ai_mode: str = "off"  # "off" | "human" | "oracle"
        self.ai_status: str = ""
        self.copilot = AsyncBonsaiCopilot(enabled=enable_copilot, interval_sec=25.0)
        self.clients: Set[web.WebSocketResponse] = set()
        self.last_tel: ThermoTelemetry = self.runner.step_once()

    def switch_scene(self, new_scene: str) -> None:
        if new_scene not in SCENE_XML_MAP or new_scene == self.scene:
            return
        xml_path = SCENE_XML_MAP[new_scene]
        if not xml_path.exists():
            return
        saved_thermo = self.runner.thermo
        saved_ctrl = self.runner.ctrl
        self.scene = new_scene
        new_runner = CoSimRunner(xml_path, preset="cold", use_gpu_3d=False)
        new_runner.use_gpu_3d = True
        saved_thermo.calib = new_runner.calib
        saved_thermo.core1d.calib = new_runner.calib
        new_runner.thermo = saved_thermo
        new_runner.ctrl = saved_ctrl
        self.runner = new_runner
        print(f"[WS-3D] 已热切换物理场景 -> {self.scene} ({xml_path.name})")

    def _set_ai_mode(self, new_mode: str) -> None:
        if new_mode not in ("off", "human", "oracle"):
            new_mode = "off"
        if new_mode == self.ai_mode:
            return
        dir_sgn = 1.0 if self.runner.ctrl.reverser >= 0.0 else -1.0
        self.ai_mode = new_mode
        if new_mode == "human":
            self.human_agent.reset(direction=dir_sgn)
            self.ai_status = self.human_agent.status_summary
        elif new_mode == "oracle":
            self.oracle_agent.reset(direction=dir_sgn)
            self.runner.external_handpump_rad = None
            self.ai_status = self.oracle_agent.status_summary
        else:
            self.runner.external_handpump_rad = None
            self.ai_status = ""

    def handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mtype = msg.get("type")
        c = self.runner.ctrl
        if mtype == "ctrl":
            if "aiMode" in msg:
                self._set_ai_mode(str(msg["aiMode"]))
            elif "autoFireman" in msg:
                self._set_ai_mode("oracle" if bool(msg["autoFireman"]) else "off")
            cd = msg.get("ctrl", {})
            if self.ai_mode == "off":
                if "fuelValve" in cd:
                    c.fuel_valve = float(cd["fuelValve"])
                if "burnerLit" in cd:
                    c.burner_lit = bool(cd["burnerLit"])
                if "chimneyFan" in cd:
                    c.chimney_fan = bool(cd["chimneyFan"])
                if "blowerValve" in cd:
                    c.blower_valve = float(cd["blowerValve"])
                if "regulator" in cd:
                    c.regulator = float(cd["regulator"])
                if "reverser" in cd:
                    c.reverser = float(cd["reverser"])
                if "drainCock" in cd:
                    c.drain_cock = bool(cd["drainCock"])
                if "bypassValve" in cd:
                    c.bypass_valve = float(cd["bypassValve"])
                if "blowdownValve" in cd:
                    c.blowdown_valve = float(cd["blowdownValve"])
                if "fireboxDoor" in cd:
                    c.firebox_door = float(cd["fireboxDoor"])
                if "fuelCapOpen" in cd:
                    c.fuel_cap_open = bool(cd["fuelCapOpen"])
            if "whistleValve" in cd:
                c.whistle_valve = float(cd["whistleValve"])
            if "waterLidOpen" in cd:
                c.water_lid_open = bool(cd["waterLidOpen"])
            if "waterFillValve" in cd:
                c.water_fill_valve = float(cd["waterFillValve"])
            if "tenderDrainOpen" in cd:
                c.tender_drain_open = bool(cd["tenderDrainOpen"])
            if "sightDrainValve" in cd:
                c.sight_drain_valve = bool(cd["sightDrainValve"])
            if "cabRoofOpen" in cd:
                c.cab_roof_open = bool(cd["cabRoofOpen"])
            if "domeLidOpen" in cd:
                c.dome_lid_open = bool(cd["domeLidOpen"])
            if "coalCoverOpen" in cd:
                c.coal_cover_open = bool(cd["coalCoverOpen"])
            if "gradeDeg" in cd:
                c.grade_deg = float(cd["gradeDeg"])
            if "thermalWarp" in cd:
                c.thermal_warp = float(cd["thermalWarp"])
        elif mtype == "action":
            act = msg.get("name")
            if act == "handpump":
                self.runner.trigger_hand_pump(int(msg.get("strokes", 5)))
            elif act == "hotPreset":
                self.runner.apply_hot_preset()
            elif act == "coldReset":
                mujoco.mj_resetData(self.runner.model, self.runner.data)
                mujoco.mj_forward(self.runner.model, self.runner.data)
                self.runner.thermo.reset_cold_prepared()
                self.runner.ctrl = CabControls()
                self._set_ai_mode("off")
                self.runner._handpump_vis_timer = 0.0
                self.runner.derailed = False
                self.runner.derail_reason = ""
            elif act == "toggleGpuMonitor":
                en = bool(msg.get("enabled", not self.runner.enable_gpu_monitor))
                self.runner.set_gpu_monitor_enabled(en)
        elif mtype == "scene":
            self.switch_scene(str(msg.get("scene", "roller")))

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024, heartbeat=55.0, autoping=True, compress=15)
        await ws.prepare(request)
        self.clients.add(ws)
        peer = request.remote or "browser"
        try:
            print(f"[WS-3D] 浏览器客户端已连接: {peer} (在线数={len(self.clients)})")
            await ws.send_str(
                pack_frame_payload(
                    self.scene,
                    self.runner,
                    self.last_tel,
                    self.ai_mode,
                    self.ai_status,
                    self.copilot.get_latest_diagnosis(),
                )
            )
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self.handle_message(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[WS-3D] 客户端会话异常 ({peer}): {exc!r}")
        finally:
            self.clients.discard(ws)
            print(f"[WS-3D] 浏览器客户端断开: {peer} (在线数={len(self.clients)})")
        return ws

    async def http_or_ws_handler(self, request: web.Request) -> web.StreamResponse:
        # 1. 若携带 WebSocket 升级头 (来自 /ws 或 Vite 5173 连接根路径 /)，走 WebSocket 实时推流
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.ws_handler(request)

        # 2. 健康检查接口 (供 IDE / 监控探针 HTTP Keep-Alive 轮询)
        rel = request.match_info.get("path", "").lstrip("/")
        if rel == "healthz":
            return web.json_response(
                {
                    "status": "ok",
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "voxels": int(self.runner.thermo.total_voxels),
                    "pino_active": bool(self.runner.thermo.pino_loaded),
                    "copilot_model": self.copilot.get_latest_diagnosis().model_name,
                    "clients": len(self.clients),
                }
            )

        # 3. 普通 HTTP 请求: 直接托管 web/dist 完整 3D 前端静态资源!
        if not WEB_DIST_DIR.exists():
            return web.Response(
                status=200,
                text=(
                    "Aster D51 RTX 5080 Live 3D Server is running.\n"
                    "Run `npm --prefix web run build` or `npm --prefix web run dev` to view the 3D Web UI.\n"
                ),
            )

        if rel == "":
            target = WEB_DIST_DIR / "index.html"
        else:
            target = (WEB_DIST_DIR / rel).resolve()
            if not target.is_relative_to(WEB_DIST_DIR):
                raise web.HTTPForbidden()

        if target.is_file():
            hdrs = {} if rel.startswith("assets/") else {"Cache-Control": "no-cache, must-revalidate"}
            return web.FileResponse(target, headers=hdrs)
        if rel == "favicon.ico":
            return web.Response(status=204)
        raise web.HTTPNotFound()

    async def sim_broadcast_loop(self, target_fps: float = 30.0) -> None:
        frame_dt = 1.0 / target_fps
        dt_mj = float(self.runner.model.opt.timestep)
        steps_per_frame = min(48, max(1, int(round(frame_dt / dt_mj))))
        hw_guard = HardwareThermalPowerGuard(max_cpu_c=76.0, max_gpu_c=70.0, max_gpu_w=215.0)
        frame_idx = 0

        while True:
            if not self.clients:
                await asyncio.sleep(0.08)
                continue

            frame_idx += 1
            if frame_idx % 12 == 0 and hw_guard.is_hot_or_high_power():
                await asyncio.sleep(0.045)

            t0 = time.perf_counter()
            try:
                tel = self.last_tel
                for s_i in range(steps_per_frame):
                    tel = self.runner.step_once()
                    # 每 14 步平滑插入 1ms 微小喘息窗口，消除 CPU/GPU 方波瞬态电流尖峰 (di/dt Vdroop) 与协程饥饿!
                    if s_i % 14 == 13:
                        await asyncio.sleep(0.001)

                if not np.all(np.isfinite(self.runner.data.qpos)):
                    print("[WS-3D] 检测到 MuJoCo qpos 非有限值，已自动重置动力学状态保持在线!")
                    mujoco.mj_resetData(self.runner.model, self.runner.data)
                    mujoco.mj_forward(self.runner.model, self.runner.data)

                self.last_tel = tel
                if self.ai_mode == "human":
                    self.human_agent.update(
                        steps_per_frame * dt_mj, tel, self.runner.ctrl, hand_pump_cb=self.runner.trigger_hand_pump
                    )
                    self.runner.external_handpump_rad = (
                        self.human_agent.handpump_anim_rad if abs(self.human_agent.handpump_anim_rad) > 1e-4 else None
                    )
                    self.ai_status = self.human_agent.status_summary
                elif self.ai_mode == "oracle":
                    self.runner.external_handpump_rad = None
                    self.oracle_agent.update(
                        steps_per_frame * dt_mj, tel, self.runner.ctrl, hand_pump_cb=self.runner.trigger_hand_pump
                    )
                    self.ai_status = self.oracle_agent.status_summary
                else:
                    self.runner.external_handpump_rad = None
                    self.ai_status = ""

                self.copilot.submit_telemetry_nonblocking(
                    asdict(tel), asdict(self.runner.ctrl), self.runner.thermo.spatial.to_dict()
                )
                frame_json = pack_frame_payload(
                    self.scene,
                    self.runner,
                    tel,
                    self.ai_mode,
                    self.ai_status,
                    self.copilot.get_latest_diagnosis(),
                )
                stale: list[web.WebSocketResponse] = []
                for ws in list(self.clients):
                    if ws.closed:
                        stale.append(ws)
                        continue
                    # SSH 隧道或后台标签页出现 TCP 发送缓冲积压 (>256KB) 时优雅跳帧，绝不挤爆缓冲区导致断连!
                    req_obj = getattr(ws, "_req", None)
                    transport = getattr(req_obj, "transport", None) if req_obj is not None else None
                    if transport is not None and transport.get_write_buffer_size() > 256 * 1024:
                        continue
                    try:
                        await ws.send_str(frame_json)
                    except Exception:
                        stale.append(ws)
                for ws in stale:
                    self.clients.discard(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[WS-3D] 广播步进捕获异常并已自愈: {exc!r}")
                await asyncio.sleep(0.05)

            elapsed = time.perf_counter() - t0
            # 强制每帧至少保留 10ms 硬件稳压冷却间隙，杜绝单线程 96% 占空比死顶!
            await asyncio.sleep(max(0.010, frame_dt - elapsed))


async def async_main(args: argparse.Namespace) -> None:
    bridge = GPU3DRealtimeBridge(
        scene=args.scene,
        preset=args.preset,
        enable_copilot=not args.no_copilot,
        enable_gpu_monitor=args.gpu_monitor,
    )
    gpu_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    app = web.Application()
    app.router.add_get("/ws", bridge.ws_handler)
    app.router.add_get("/", bridge.http_or_ws_handler)
    app.router.add_get("/{path:.*}", bridge.http_or_ws_handler)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port)
    await site.start()

    print("=" * 78)
    print("  Aster D51 RTX 5080 Live 3D Thermo-Fluid + PINO + Bonsai-2:27B Server Ready!")
    print(f"  GPU Device   : {gpu_str} ({bridge.runner.thermo.total_voxels:,} 3D STL Voxels, PINO={bridge.runner.thermo.pino_loaded})")
    if _QUARANTINED_OC_CPUS:
        print(f"  CPU Guard    : 已自动隔离 BIOS 5.4GHz 超频风险核 {_QUARANTINED_OC_CPUS} (绑定安全核: {sorted(os.sched_getaffinity(0))})")
    print(f"  Web 3D UI    : http://{args.host}:{args.port}/   (浏览器直接打开即可！)")
    print(f"  WebSocket URL: ws://{args.host}:{args.port}/ws   (兼容 Vite 5173 自动直连, {args.fps:.0f} FPS deflate)")
    print("=" * 78)

    sim_task = asyncio.create_task(bridge.sim_broadcast_loop(target_fps=args.fps))
    try:
        await sim_task
    finally:
        bridge.copilot.shutdown()
        sim_task.cancel()
        for ws in list(bridge.clients):
            await ws.close()
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster D51 RTX 5080 真 3D 多相热流 HTTP + WebSocket 一体化服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", choices=["roller", "track", "curve"], default="roller")
    parser.add_argument("--preset", choices=["cold", "hot"], default="hot")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--gpu-monitor", action="store_true", help="默认开启实时 GPU SM/VRAM/温度/功耗监控遥测")
    parser.add_argument("--no-copilot", action="store_true", help="禁用异步本地 Ollama bonsai-2:27b L3 诊断副驾")
    parser.add_argument("--no-replace", action="store_true", help="不自动清理同端口已有的旧 gpu_3d_server 进程")
    args = parser.parse_args()

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    sys.stdout = _SafeTeeStream(sys.stdout, SERVER_LOG_PATH)  # type: ignore[assignment]
    sys.stderr = _SafeTeeStream(sys.stderr, SERVER_LOG_PATH)  # type: ignore[assignment]

    if not args.no_replace:
        _terminate_stale_gpu_servers(os.getpid())

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[WS-3D] 服务已安全退出。")


if __name__ == "__main__":
    main()
