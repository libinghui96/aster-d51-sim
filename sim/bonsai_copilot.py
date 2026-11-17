#!/usr/bin/env python3
"""
sim/bonsai_copilot.py
L3 高层认知诊断副驾层 (Gate 10 - L3)：
异步连接本地 Ollama `bonsai-2:27b` (1-bit/三值极低显存大语言模型)，
在不阻塞 2000 Hz MuJoCo + 250 Hz 3D PINO 物理主循环的前提下 (独立守护线程 ~0.4 Hz 慢回路)，
读取全车 IAPWS 相变遥测与三维 PINO 空间场探针，结合原厂《Aster D51 組立・運転説明書》物理规律，
输出实时机务根因诊断与操纵协同建议。
"""

from dataclasses import asdict, dataclass
import json
import threading
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.request


@dataclass
class CopilotDiagnosis:
    available: bool = False
    model_name: str = "bonsai-2:27b"
    status_level: str = "NOMINAL"     # NOMINAL | CAUTION | CRITICAL
    headline: str = "等待本地 Bonsai-2:27B 机务副驾连接..."
    root_cause: str = ""
    recommended_action: str = ""
    latency_ms: float = 0.0
    updated_sim_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT_D51 = """你是 Aster Hobby JNR D51 (1/32 真蒸汽机车) 首席机务诊断工程师。
你只接收由 L1(IAPWS真物性守恒) 与 L2(3D PINO流场算子) 算出的实时物理遥测 JSON，绝不自己捏造物理数值。
原厂物理铁律：
1. 正常工作表压 3.0~4.0 kgf/cm² (安全阀 4.0 起跳, 3.6 回座)，水面计标准水位 50%；
2. 冷炉烧水阶段 (P_boil_kgf < 0.3, T_boil_c < 105°C)：尚无蒸汽驱动通风阀 (steam_jet_gs≈0)，必须开启烟囱电动抽风机 (ctrl.fan=true) 引风升温；只要 pino_3d.smokebox_airtight=true 且 open_flues=5，严禁误报“烟室漏气或烟管堵塞”！若冷炉未开抽风机 (fan=false 且 lit=true)，才提示开启煙突通風機；
3. 唯有当高压蒸汽已建立 (P_boil_kgf > 1.0) 且蒸汽喷射已开 (steam_jet_gs > 0.12)，但 vac_pa < 15 Pa，或 pino_3d.smokebox_airtight=false / open_flues < 5 时，才判定烟室漏气或堵管；
4. 上坡时锅水后涌产生虚高假水位，必须盯紧火室紫铜顶板浸没水深 crown_sub_mm (<=0.5mm 或 T_crown>240°C 为空焚き干烧危险)；
5. 若 mariotte_flood=true (供油中拧开了 36-5 酒精箱气密盖)，马氏瓶真空破坏导致酒精漫溢富燃，提示立即旋紧 36-5 燃料箱盖；若摇手摇泵时 ctrl.bypass > 0.06，水流经 31-14 回流煤水车，提示先关旁通阀再打手摇泵；
6. 冷机缸壁 <100°C 且凝结水 >0.6g 时未开排液阀 (drain=false) 会引发不可压缩水锤 Hydrolock；注油器油量 <5% 会导致滑阀干摩损伤。

请严格只输出一行紧凑 JSON (不要 Markdown 代码块)，字段如下：
{"status_level":"NOMINAL|CAUTION|CRITICAL","headline":"18字内状态结论","root_cause":"35字内物理因果分析","recommended_action":"30字内具体阀门操作建议"}"""


class AsyncBonsaiCopilot:
    """非阻塞后台线程调用本地 Ollama bonsai-2:27b，物理主线程读取耗时为 0.000 ms"""

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        preferred_model: str = "bonsai-2:27b",
        interval_sec: float = 12.0,
        enabled: bool = True,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = preferred_model
        self.interval_sec = max(5.0, float(interval_sec))
        self.enabled = bool(enabled)

        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._payload_dirty: bool = False
        self._last_submit_mono: float = 0.0
        self._last_query_sig: Optional[tuple] = None
        self._last_query_mono: float = 0.0
        self._diagnosis = CopilotDiagnosis(model_name=self.model)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(target=self._worker_loop, name="BonsaiCopilotL3", daemon=True)
            self._thread.start()

    def submit_telemetry_nonblocking(
        self,
        tel_dict: Dict[str, Any],
        ctrl_dict: Dict[str, Any],
        spatial_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """由 60Hz/2000Hz 主循环调用，仅原子替换指针，零 I/O 阻塞"""
        if not self.enabled:
            return
        compact = {
            "sim_t": round(float(tel_dict.get("sim_time", 0.0)), 1),
            "P_boil_kgf": round(float(tel_dict.get("boiler_pressure_kgf", 0.0)), 2),
            "P_chest_kgf": round(float(tel_dict.get("steam_chest_kgf", 0.0)), 2),
            "T_boil_c": round(float(tel_dict.get("boiler_temp_c", 25.0)), 1),
            "boil_prog_pct": round(float(tel_dict.get("boiling_progress_pct", 0.0)), 1),
            "T_sh_c": round(float(tel_dict.get("superheat_temp_c", 25.0)), 1),
            "T_crown_c": round(float(tel_dict.get("crown_temp_c", 25.0)), 1),
            "crown_sub_mm": round(float(tel_dict.get("crown_submerged_mm", 11.5)), 1),
            "gauge_lvl_pct": round(float(tel_dict.get("water_level_pct", 50.0)), 1),
            "true_lvl_pct": round(float(tel_dict.get("true_water_level_pct", 50.0)), 1),
            "vac_pa": round(float(tel_dict.get("smokebox_vacuum_pa", 0.0)), 1),
            "steam_jet_gs": round(float(tel_dict.get("steam_gen_g_s", 0.0)), 2),
            "cond_g": round(float(tel_dict.get("condensate_g", 0.0)), 2),
            "oil_pct": round(float(tel_dict.get("lubricator_oil_pct", 80.0)), 1),
            "rpm": round(float(tel_dict.get("wheel_rpm", 0.0)), 1),
            "hydrolock": bool(tel_dict.get("hydrolock_L") or tel_dict.get("hydrolock_R")),
            "dry_alarm": bool(tel_dict.get("dry_fire_alarm")),
            "mariotte_flood": bool(tel_dict.get("mariotte_flood_alarm", False)),
            "return_ml_s": round(float(tel_dict.get("return_water_ml_s", 0.0)), 2),
            "ctrl": {
                "reg": round(float(ctrl_dict.get("regulator", 0.0)), 2),
                "rev": round(float(ctrl_dict.get("reverser", 0.0)), 2),
                "drain": bool(ctrl_dict.get("drain_cock", True)),
                "bypass": round(float(ctrl_dict.get("bypass_valve", 0.4)), 2),
                "blower": round(float(ctrl_dict.get("blower_valve", 0.0)), 2),
                "fan": bool(ctrl_dict.get("chimney_fan", False)),
                "lit": bool(ctrl_dict.get("burner_lit", False)),
                "fire_door": round(float(ctrl_dict.get("firebox_door", 0.0)), 2),
                "fuel_cap_open": bool(ctrl_dict.get("fuel_cap_open", False)),
                "grade_deg": round(float(ctrl_dict.get("grade_deg", 0.0)), 1),
            },
        }
        if spatial_dict:
            compact["pino_3d"] = {
                "smokebox_airtight": bool(spatial_dict.get("stl_passages_verified", True)),
                "open_flues": int(spatial_dict.get("open_flue_count", 5)),
                "phi": round(float(spatial_dict.get("air_fuel_phi", 0.95)), 2),
                "x_dome": round(float(spatial_dict.get("steam_dryness_dome", 0.96)), 3),
                "x_sh_out": round(float((spatial_dict.get("sh_profile_dryness") or [0.98])[-1]), 3),
            }
        with self._lock:
            self._latest_payload = compact
            self._payload_dirty = True
            self._last_submit_mono = time.monotonic()

    def get_latest_diagnosis(self) -> CopilotDiagnosis:
        with self._lock:
            return CopilotDiagnosis(**asdict(self._diagnosis))

    def shutdown(self) -> None:
        self._stop_evt.set()

    def _detect_model(self) -> Optional[str]:
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            if self.model in names:
                return self.model
            for cand in names:
                if "bonsai" in cand.lower():
                    return cand
            return names[0] if names else None
        except Exception:
            return None

    def _worker_loop(self) -> None:
        from sim.hw_guard import HardwareThermalPowerGuard, has_unclamped_overclocked_cpus
        hw_guard = HardwareThermalPowerGuard(max_cpu_c=78.0, max_gpu_c=70.0, max_gpu_w=240.0)
        detected = self._detect_model()
        if detected:
            self.model = detected

        while not self._stop_evt.is_set():
            snap = None
            now_m = time.monotonic()
            with self._lock:
                # 严禁无客户端连接或暂停时死循环唤醒 Ollama 27B GPU 推理!
                if (
                    self._latest_payload is not None
                    and self._payload_dirty
                    and (now_m - self._last_submit_mono) <= 2.5
                ):
                    snap = dict(self._latest_payload)
                    self._payload_dirty = False

            if snap is not None and not hw_guard.is_hot_or_high_power():
                ctrl_s = snap.get("ctrl") or {}
                sig = (
                    round(float(snap.get("P_boil_kgf", 0.0)) * 4.0),
                    round(float(snap.get("T_boil_c", 25.0)) / 5.0),
                    round(float(snap.get("gauge_lvl_pct", 50.0)) / 5.0),
                    bool(snap.get("hydrolock")),
                    bool(snap.get("dry_alarm")),
                    bool(snap.get("mariotte_flood")),
                    round(float(ctrl_s.get("reg", 0.0)) * 5.0),
                    bool(ctrl_s.get("lit", False)),
                    bool(ctrl_s.get("fan", False)),
                )
                if sig != self._last_query_sig or (now_m - self._last_query_mono) >= 45.0:
                    t0 = time.perf_counter()
                    # 若宿主机尚未执行 setup_host_power_guard.sh (仍存在 5.40GHz 超频核 CPU 8)，外部 ollama 服务未绑核，
                    # 自动采用零功耗确定性物理规则引擎，防止 ollama 与实时 3D 推流并发冲顶触发 CPU 8 崩核硬重启!
                    if has_unclamped_overclocked_cpus():
                        diag = self._rule_fallback(snap)
                    else:
                        diag = self._query_ollama(snap, t0)
                    self._last_query_sig = sig
                    self._last_query_mono = time.monotonic()
                    with self._lock:
                        self._diagnosis = diag

            self._stop_evt.wait(self.interval_sec)

    def _query_ollama(self, snap: Dict[str, Any], t0: float) -> CopilotDiagnosis:
        body = json.dumps(
            {
                "model": self.model,
                "system": SYSTEM_PROMPT_D51,
                "prompt": f"当前D51实时物理遥测: {json.dumps(snap, ensure_ascii=False)}\n请输出JSON诊断:",
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.15, "num_predict": 180},
            }
        ).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            txt = raw.get("response", "").strip()
            parsed = json.loads(txt)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lvl = str(parsed.get("status_level", "NOMINAL")).upper()
            if lvl not in ("NOMINAL", "CAUTION", "CRITICAL"):
                lvl = "NOMINAL"
            headline = str(parsed.get("headline", "工况正常"))[:40]
            root_cause = str(parsed.get("root_cause", ""))[:90]
            rec_act = str(parsed.get("recommended_action", ""))[:90]

            # 物理因果一致性硬校核: 若烟室结构完好 (airtight=True, open_flues=5) 且处于 P_boil < 0.3 冷炉烧水阶段，绝不允许误报“漏气/堵管”！
            p3d = snap.get("pino_3d") or {}
            airtight = bool(p3d.get("smokebox_airtight", True))
            open_flues = int(p3d.get("open_flues", 5))
            if airtight and open_flues >= 5 and float(snap.get("P_boil_kgf", 0.0)) < 0.35:
                combined_txt = headline + root_cause
                if ("漏气" in combined_txt) or ("堵塞" in combined_txt) or ("堵管" in combined_txt):
                    fb = self._rule_fallback(snap)
                    return CopilotDiagnosis(
                        available=True,
                        model_name=self.model,
                        status_level=fb.status_level,
                        headline=fb.headline,
                        root_cause=fb.root_cause,
                        recommended_action=fb.recommended_action,
                        latency_ms=round(dt_ms, 1),
                        updated_sim_time=float(snap.get("sim_t", 0.0)),
                    )

            return CopilotDiagnosis(
                available=True,
                model_name=self.model,
                status_level=lvl,
                headline=headline,
                root_cause=root_cause,
                recommended_action=rec_act,
                latency_ms=round(dt_ms, 1),
                updated_sim_time=float(snap.get("sim_t", 0.0)),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
            return self._rule_fallback(snap)

    def _rule_fallback(self, snap: Dict[str, Any]) -> CopilotDiagnosis:
        """确定性物理规则机务诊断引擎 (亦用于冷炉工况一致性校核)"""
        sim_t = float(snap.get("sim_t", 0.0))
        ctrl = snap.get("ctrl") or {}
        p3d = snap.get("pino_3d") or {}
        p_boil = float(snap.get("P_boil_kgf", 0.0))
        t_boil = float(snap.get("T_boil_c", 25.0))
        vac_pa = float(snap.get("vac_pa", 0.0))

        if snap.get("hydrolock"):
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CRITICAL",
                headline="气缸液态水锤锁死 (Hydrolock)",
                root_cause="冷缸蒸汽凝结积水超出死点余隙容积且排液旋塞关闭",
                recommended_action="立即拉出排液阀拉杆 8-6 [X] 吹除积水，收小加减阀暖缸",
                updated_sim_time=sim_t,
            )
        if snap.get("dry_alarm"):
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CRITICAL",
                headline="紫铜火室顶板裸露干烧告警",
                root_cause=f"顶板浸没水深 {snap.get('crown_sub_mm')}mm，壁温 {snap.get('T_crown_c')}°C",
                recommended_action="严禁高温骤打冷水！立即关 36-6 燃料阀熄火降温后再补水",
                updated_sim_time=sim_t,
            )
        if snap.get("mariotte_flood"):
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CAUTION",
                headline="马氏瓶酒精箱失真空漫油富燃",
                root_cause="供醇中打开了 36-5 气密旋盖，静液柱压迫酒精漫入火室",
                recommended_action="立即旋紧煤水车 36-5 燃料箱盖恢复恒液位真空供醇",
                updated_sim_time=sim_t,
            )
        if (not bool(p3d.get("smokebox_airtight", True))) or int(p3d.get("open_flues", 5)) < 5:
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CAUTION",
                headline="烟室气密破损或烟管堵塞",
                root_cause=f"畅通烟管 {p3d.get('open_flues', 5)}/5，烟室负压跌至 {vac_pa} Pa",
                recommended_action="锁紧烟室前门 18-3 并用通条清理 5 根铜烟管",
                updated_sim_time=sim_t,
            )
        if p_boil < 0.35:
            if ctrl.get("lit") and (not ctrl.get("fan")):
                return CopilotDiagnosis(
                    available=False,
                    model_name="physics-guard-fallback",
                    status_level="CAUTION",
                    headline=f"冷炉点火缺风 (水温 {t_boil:.0f}°C)",
                    root_cause="锅炉尚未起压无蒸汽吹风，未装烟囱电动抽风机导致火室富燃",
                    recommended_action="立即开启煙突通風機建立 -75 Pa 抽风加速烧水沸腾",
                    updated_sim_time=sim_t,
                )
            if ctrl.get("lit") and ctrl.get("fan"):
                return CopilotDiagnosis(
                    available=False,
                    model_name="physics-guard-fallback",
                    status_level="NOMINAL",
                    headline=f"冷炉抽风升温中 ({t_boil:.0f}°C / 100°C)",
                    root_cause=f"电动抽风机维持烟室负压 -{vac_pa:.0f} Pa，1x3 三头酒精灯正常加热",
                    recommended_action="待起压至 1.0 kgf/cm² 后开微通风阀并取下烟囱抽风机",
                    updated_sim_time=sim_t,
                )
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="NOMINAL",
                headline=f"冷炉待命 (水温 {t_boil:.0f}°C, 水位 {snap.get('gauge_lvl_pct')}%)",
                root_cause="锅炉处于常压冷态，双路给水与置换注油器整备完毕",
                recommended_action="旋开煤水车 36-6 燃料阀、开启煙突通風機并点火升压",
                updated_sim_time=sim_t,
            )
        return CopilotDiagnosis(
            available=False,
            model_name="physics-guard-fallback",
            status_level="NOMINAL",
            headline=f"锅炉 {p_boil:.2f} kgf/cm² 运转平稳",
            root_cause=f"烟室负压 -{vac_pa:.0f} Pa，过热汽温 {snap.get('T_sh_c')}°C",
            recommended_action="微调旁通阀维持实水位 50%，巡航收逆转器省汽",
            updated_sim_time=sim_t,
        )
