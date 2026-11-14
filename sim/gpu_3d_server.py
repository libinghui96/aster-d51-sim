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
import mimetypes
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional, Set

from aiohttp import WSMsgType, web
import mujoco
import torch

from sim.auto_fireman import AutoFiremanAgent
from sim.bonsai_copilot import AsyncBonsaiCopilot, CopilotDiagnosis
from sim.run_cosim import CoSimRunner
from sim.thermo_1d import CabControls, ThermoTelemetry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST_DIR = (PROJECT_ROOT / "web" / "dist").resolve()

SCENE_XML_MAP = {
    "roller": PROJECT_ROOT / "models" / "d51_full.xml",
    "curve": PROJECT_ROOT / "models" / "d51_r2m.xml",
}

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("model/stl", ".stl")


def pack_frame_payload(
    scene_name: str,
    runner: CoSimRunner,
    tel: ThermoTelemetry,
    auto_fireman: bool,
    copilot_diag: Optional[CopilotDiagnosis] = None,
) -> str:
    sp = runner.thermo.spatial
    b64_slice = base64.b64encode(bytes(sp.midplane_temp_u8)).decode("ascii") if sp.midplane_temp_u8 else ""
    qpos_list = [round(float(x), 6) for x in runner.data.qpos]

    payload: Dict[str, Any] = {
        "type": "frame",
        "scene": scene_name,
        "nq": int(runner.model.nq),
        "qpos": qpos_list,
        "autoFireman": auto_fireman,
        "ctrl": {
            "fuelValve": round(runner.ctrl.fuel_valve, 3),
            "burnerLit": bool(runner.ctrl.burner_lit),
            "chimneyFan": bool(runner.ctrl.chimney_fan),
            "blowerValve": round(runner.ctrl.blower_valve, 3),
            "regulator": round(runner.ctrl.regulator, 3),
            "reverser": round(runner.ctrl.reverser, 3),
            "drainCock": bool(runner.ctrl.drain_cock),
            "bypassValve": round(runner.ctrl.bypass_valve, 3),
            "gradeDeg": round(runner.ctrl.grade_deg, 2),
            "thermalWarp": round(runner.ctrl.thermal_warp, 1),
        },
        "tel": {
            "simTime": round(tel.sim_time, 3),
            "boilerTempC": round(tel.boiler_temp_c, 2),
            "boilerPressureKgf": round(tel.boiler_pressure_kgf, 3),
            "steamChestKgf": round(tel.steam_chest_kgf, 3),
            "superheatTempC": round(tel.superheat_temp_c, 1),
            "waterLevelPct": round(tel.water_level_pct, 2),
            "trueWaterLevelPct": round(tel.true_water_level_pct, 2),
            "crownTempC": round(tel.crown_temp_c, 1),
            "crownSubmergedMm": round(tel.crown_submerged_mm, 2),
            "copperAnnealDamage": round(tel.copper_anneal_damage, 4),
            "sealHealth": round(tel.seal_health, 4),
            "alcoholMl": round(tel.alcohol_ml, 2),
            "tenderWaterMl": round(tel.tender_water_ml, 2),
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
            "feedwaterGS": round(tel.feedwater_g_s, 3),
            "steamGenGS": round(tel.steam_gen_g_s, 3),
            "forceL": round(tel.force_L, 2),
            "forceR": round(tel.force_R, 2),
            "fPump": 0.0,
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
            "lubeWaterFrac": float(sp.lube_water_frac),
            "midplaneTempB64": b64_slice,
        },
        "copilot": copilot_diag.to_dict() if copilot_diag is not None else None,
    }
    return json.dumps(payload, separators=(",", ":"))


class GPU3DRealtimeBridge:
    def __init__(self, scene: str = "roller", preset: str = "hot", enable_copilot: bool = True):
        self.scene = scene if scene in SCENE_XML_MAP else "roller"
        self.preset = preset
        self.runner = CoSimRunner(SCENE_XML_MAP[self.scene], preset=preset, use_gpu_3d=True)
        self.af_agent = AutoFiremanAgent()
        self.auto_fireman = False
        self.copilot = AsyncBonsaiCopilot(enabled=enable_copilot)
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

    def handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mtype = msg.get("type")
        c = self.runner.ctrl
        if mtype == "ctrl":
            if "autoFireman" in msg:
                af_new = bool(msg["autoFireman"])
                if af_new and not self.auto_fireman:
                    self.af_agent.reset(direction=1.0 if c.reverser >= 0.0 else -1.0)
                self.auto_fireman = af_new
            cd = msg.get("ctrl", {})
            if not self.auto_fireman:
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
            if "gradeDeg" in cd:
                c.grade_deg = float(cd["gradeDeg"])
            if "thermalWarp" in cd:
                c.thermal_warp = float(cd["thermalWarp"])
        elif mtype == "action":
            act = msg.get("name")
            if act == "handpump":
                self.runner.thermo.trigger_hand_pump(int(msg.get("strokes", 5)))
            elif act == "hotPreset":
                self.runner.apply_hot_preset()
            elif act == "coldReset":
                mujoco.mj_resetData(self.runner.model, self.runner.data)
                mujoco.mj_forward(self.runner.model, self.runner.data)
                self.runner.thermo.reset_cold_prepared()
                self.runner.ctrl = CabControls()
                self.auto_fireman = False
        elif mtype == "scene":
            self.switch_scene(str(msg.get("scene", "roller")))

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024, heartbeat=25.0)
        await ws.prepare(request)
        self.clients.add(ws)
        peer = request.remote or "browser"
        print(f"[WS-3D] 浏览器客户端已连接: {peer} (在线数={len(self.clients)})")
        try:
            await ws.send_str(
                pack_frame_payload(
                    self.scene, self.runner, self.last_tel, self.auto_fireman, self.copilot.get_latest_diagnosis()
                )
            )
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self.handle_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
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
            return web.FileResponse(target)
        if rel == "favicon.ico":
            return web.Response(status=204)
        raise web.HTTPNotFound()

    async def sim_broadcast_loop(self, target_fps: float = 60.0) -> None:
        frame_dt = 1.0 / target_fps
        dt_mj = float(self.runner.model.opt.timestep)
        steps_per_frame = max(1, int(round(frame_dt / dt_mj)))

        while True:
            if not self.clients:
                await asyncio.sleep(0.06)
                continue

            t0 = time.perf_counter()
            tel = self.last_tel
            for _ in range(steps_per_frame):
                tel = self.runner.step_once()
            self.last_tel = tel
            if self.auto_fireman:
                self.af_agent.update(steps_per_frame * dt_mj, tel, self.runner.ctrl)

            self.copilot.submit_telemetry_nonblocking(
                asdict(tel), asdict(self.runner.ctrl), self.runner.thermo.spatial.to_dict()
            )
            frame_json = pack_frame_payload(
                self.scene, self.runner, tel, self.auto_fireman, self.copilot.get_latest_diagnosis()
            )
            stale: list[web.WebSocketResponse] = []
            for ws in list(self.clients):
                if ws.closed:
                    stale.append(ws)
                    continue
                try:
                    await ws.send_str(frame_json)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self.clients.discard(ws)

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.001, frame_dt - elapsed))


async def async_main(args: argparse.Namespace) -> None:
    bridge = GPU3DRealtimeBridge(scene=args.scene, preset=args.preset, enable_copilot=not args.no_copilot)
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
    print(f"  Web 3D UI    : http://{args.host}:{args.port}/   (浏览器直接打开即可！)")
    print(f"  WebSocket URL: ws://{args.host}:{args.port}/ws   (兼容 Vite 5173 自动直连)")
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
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", choices=["roller", "curve"], default="roller")
    parser.add_argument("--preset", choices=["cold", "hot"], default="hot")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-copilot", action="store_true", help="禁用异步本地 Ollama bonsai-2:27b L3 诊断副驾")
    args = parser.parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[WS-3D] 服务已安全退出。")


if __name__ == "__main__":
    main()
