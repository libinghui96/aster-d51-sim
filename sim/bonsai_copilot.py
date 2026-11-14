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
2. 上坡时锅水后涌产生虚高假水位，必须盯紧火室紫铜顶板浸没水深 crown_submerged_mm (<=0.5mm 或 T_crown>240°C 为空焚き干烧危险)；
3. 烟室正常抽风负压 45~150 Pa；若乏汽/通风阀已开但负压极低 (<15 Pa) 且火室氧气偏低，系烟室门未关严漏气或烟管堵塞；
4. 冷机缸壁 <100°C 且凝结水 >0.6g 时未开排液阀 (drain_cock=false) 会引发不可压缩水锤 Hydrolock；
5. 置换式注油器油量 <5% 会导致平面滑阀与活塞环干摩损伤。

请严格只输出一行紧凑 JSON (不要 Markdown 代码块)，字段如下：
{"status_level":"NOMINAL|CAUTION|CRITICAL","headline":"18字内状态结论","root_cause":"35字内物理因果分析","recommended_action":"30字内具体阀门操作建议"}"""


class AsyncBonsaiCopilot:
    """非阻塞后台线程调用本地 Ollama bonsai-2:27b，物理主线程读取耗时为 0.000 ms"""

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        preferred_model: str = "bonsai-2:27b",
        interval_sec: float = 3.0,
        enabled: bool = True,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = preferred_model
        self.interval_sec = max(1.5, float(interval_sec))
        self.enabled = bool(enabled)

        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
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
            "T_sh_c": round(float(tel_dict.get("superheat_temp_c", 25.0)), 1),
            "T_crown_c": round(float(tel_dict.get("crown_temp_c", 25.0)), 1),
            "crown_sub_mm": round(float(tel_dict.get("crown_submerged_mm", 11.5)), 1),
            "gauge_lvl_pct": round(float(tel_dict.get("water_level_pct", 50.0)), 1),
            "true_lvl_pct": round(float(tel_dict.get("true_water_level_pct", 50.0)), 1),
            "vac_pa": round(float(tel_dict.get("smokebox_vacuum_pa", 0.0)), 1),
            "cond_g": round(float(tel_dict.get("condensate_g", 0.0)), 2),
            "oil_pct": round(float(tel_dict.get("lubricator_oil_pct", 80.0)), 1),
            "rpm": round(float(tel_dict.get("wheel_rpm", 0.0)), 1),
            "hydrolock": bool(tel_dict.get("hydrolock_L") or tel_dict.get("hydrolock_R")),
            "dry_alarm": bool(tel_dict.get("dry_fire_alarm")),
            "ctrl": {
                "reg": round(float(ctrl_dict.get("regulator", 0.0)), 2),
                "rev": round(float(ctrl_dict.get("reverser", 0.0)), 2),
                "drain": bool(ctrl_dict.get("drain_cock", True)),
                "bypass": round(float(ctrl_dict.get("bypass_valve", 0.4)), 2),
                "blower": round(float(ctrl_dict.get("blower_valve", 0.0)), 2),
                "fan": bool(ctrl_dict.get("chimney_fan", False)),
                "lit": bool(ctrl_dict.get("burner_lit", False)),
                "grade_deg": round(float(ctrl_dict.get("grade_deg", 0.0)), 1),
            },
        }
        if spatial_dict:
            compact["pino_3d"] = {
                "open_flues": spatial_dict.get("open_flue_count", 5),
                "phi": round(float(spatial_dict.get("air_fuel_phi", 0.95)), 2),
                "x_dome": round(float(spatial_dict.get("steam_dryness_dome", 0.96)), 3),
                "x_sh_out": round(float((spatial_dict.get("sh_profile_dryness") or [0.98])[-1]), 3),
            }
        with self._lock:
            self._latest_payload = compact

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
        detected = self._detect_model()
        if detected:
            self.model = detected

        while not self._stop_evt.is_set():
            snap = None
            with self._lock:
                if self._latest_payload is not None:
                    snap = dict(self._latest_payload)

            if snap is not None:
                t0 = time.perf_counter()
                diag = self._query_ollama(snap, t0)
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
            return CopilotDiagnosis(
                available=True,
                model_name=self.model,
                status_level=lvl,
                headline=str(parsed.get("headline", "工况正常"))[:40],
                root_cause=str(parsed.get("root_cause", ""))[:90],
                recommended_action=str(parsed.get("recommended_action", ""))[:90],
                latency_ms=round(dt_ms, 1),
                updated_sim_time=float(snap.get("sim_t", 0.0)),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
            return self._rule_fallback(snap)

    def _rule_fallback(self, snap: Dict[str, Any]) -> CopilotDiagnosis:
        """当本地 Ollama 服务未启动时的确定性物理规则兜底"""
        if snap.get("hydrolock"):
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CRITICAL",
                headline="气缸液态水锤锁死 (Hydrolock)",
                root_cause="冷缸蒸汽凝结积水超出死点余隙容积且排液旋塞关闭",
                recommended_action="立即打开排液阀 [X] 吹除积水，收小加减阀暖缸",
                updated_sim_time=float(snap.get("sim_t", 0.0)),
            )
        if snap.get("dry_alarm"):
            return CopilotDiagnosis(
                available=False,
                model_name="physics-guard-fallback",
                status_level="CRITICAL",
                headline="紫铜火室顶板裸露干烧告警",
                root_cause=f"顶板浸没水深 {snap.get('crown_sub_mm')}mm，壁温 {snap.get('T_crown_c')}°C",
                recommended_action="严禁高温骤打冷水！立即关燃料阀熄火降温后再摇手摇泵补水",
                updated_sim_time=float(snap.get("sim_t", 0.0)),
            )
        return CopilotDiagnosis(
            available=False,
            model_name="physics-guard-fallback",
            status_level="NOMINAL",
            headline=f"锅炉 {snap.get('P_boil_kgf')} kgf/cm² 运转平稳",
            root_cause=f"烟室负压 {snap.get('vac_pa')} Pa，过热汽温 {snap.get('T_sh_c')}°C",
            recommended_action="微调旁通阀维持实水位 50%，巡航收逆转器省汽",
            updated_sim_time=float(snap.get("sim_t", 0.0)),
        )
