"""
sim/gpu_monitor.py
Optional Real-Time NVIDIA GPU Utilization, Kernel Timing & VRAM Telemetry Monitor (Stage 18 Item 16).

Design invariants:
  1. Zero impact on the 2000 Hz physics loop when disabled (`enabled=False`), and < 0.005 ms/step
     when enabled (board-level SM/VRAM/Power/Temp/Clock metrics are sampled asynchronously at 2 Hz
     on a daemon thread, while sub-millisecond PyTorch CUDA kernel timings use non-blocking CUDA events).
  2. Supports both CLI (`--gpu-monitor`) and live WebSocket toggle (`set_enabled(bool)`).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class GPUMonitorSnapshot:
    enabled: bool = False
    device_name: str = "CPU Fallback"
    sm_util_pct: float = 0.0
    mem_bw_util_pct: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    torch_alloc_mb: float = 0.0
    torch_reserved_mb: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0
    gpu_clock_mhz: float = 0.0
    step_wall_ms: float = 0.0
    gpu_3d_pde_ms: float = 0.0
    gpu_pino_ms: float = 0.0
    gpu_elastic_ms: float = 0.0
    gpu_share_pct: float = 0.0
    active_voxels_m: float = 0.0
    pino_operators: int = 0
    cfd_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptionalGPUMonitor:
    """Thread-safe optional GPU telemetry collector for RTX 5080 / CUDA digital twin co-simulation."""

    def __init__(
        self,
        enabled: bool = False,
        poll_interval_s: float = 0.50,
        active_voxels: int = 0,
        pino_operators: int = 7,
        cfd_hz: float = 125.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.poll_interval_s = max(0.20, float(poll_interval_s))
        self.active_voxels = int(active_voxels)
        self.pino_operators = int(pino_operators)
        self.cfd_hz = float(cfd_hz)

        self._has_cuda = bool(torch is not None and torch.cuda.is_available())
        self._device_name = (
            torch.cuda.get_device_name(0) if self._has_cuda else "CPU Only"
        )
        self._smi_bin: Optional[str] = shutil.which("nvidia-smi")

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._sm_util_pct = 0.0
        self._mem_bw_util_pct = 0.0
        self._vram_used_mb = 0.0
        self._vram_total_mb = 0.0
        self._gpu_temp_c = 0.0
        self._gpu_power_w = 0.0
        self._gpu_clock_mhz = 0.0

        # Exponential moving averages of per-call kernel timings (ms)
        self._ema_step_wall_ms = 0.0
        self._ema_pde_ms = 0.0
        self._ema_pino_ms = 0.0
        self._ema_elastic_ms = 0.0

        if self.enabled:
            self._start_worker()

    def set_enabled(self, flag: bool) -> None:
        flag = bool(flag)
        if flag == self.enabled:
            return
        self.enabled = flag
        if flag:
            self._start_worker()

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.8)
        self._thread = None

    def _start_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._query_smi_once()
        self._thread = threading.Thread(
            target=self._poll_loop, name="d51-gpu-monitor", daemon=True
        )
        self._thread.start()

    def _query_smi_once(self) -> None:
        if not self._smi_bin:
            return
        try:
            out = subprocess.check_output(
                [
                    self._smi_bin,
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=0.45,
                text=True,
            ).strip()
            if not out:
                return
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 7:
                def _f(s: str) -> float:
                    try:
                        return float(s)
                    except ValueError:
                        return 0.0

                with self._lock:
                    self._sm_util_pct = _f(parts[0])
                    self._mem_bw_util_pct = _f(parts[1])
                    self._vram_used_mb = _f(parts[2])
                    self._vram_total_mb = _f(parts[3])
                    self._gpu_temp_c = _f(parts[4])
                    self._gpu_power_w = _f(parts[5])
                    self._gpu_clock_mhz = _f(parts[6])
        except Exception:
            pass

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            if self.enabled:
                self._query_smi_once()
            self._stop_evt.wait(self.poll_interval_s)

    def record_kernel_timings(
        self,
        step_wall_ms: float,
        pde_ms: float = 0.0,
        pino_ms: float = 0.0,
        elastic_ms: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        alpha = 0.12
        with self._lock:
            self._ema_step_wall_ms = (1.0 - alpha) * self._ema_step_wall_ms + alpha * max(0.0, float(step_wall_ms))
            if pde_ms > 0.0:
                self._ema_pde_ms = (1.0 - alpha) * self._ema_pde_ms + alpha * float(pde_ms)
            if pino_ms > 0.0:
                self._ema_pino_ms = (1.0 - alpha) * self._ema_pino_ms + alpha * float(pino_ms)
            if elastic_ms > 0.0:
                self._ema_elastic_ms = (1.0 - alpha) * self._ema_elastic_ms + alpha * float(elastic_ms)

    def snapshot(self) -> GPUMonitorSnapshot:
        if not self.enabled:
            return GPUMonitorSnapshot(
                enabled=False,
                device_name=self._device_name,
                active_voxels_m=round(self.active_voxels * 1e-6, 2),
                pino_operators=self.pino_operators,
                cfd_hz=self.cfd_hz,
            )
        alloc_mb = 0.0
        res_mb = 0.0
        if self._has_cuda and torch is not None:
            alloc_mb = float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
            res_mb = float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
        with self._lock:
            gpu_sub_ms = self._ema_pde_ms + self._ema_pino_ms + self._ema_elastic_ms
            denom_ms = max(0.05, self._ema_step_wall_ms * (2000.0 / max(1.0, self.cfd_hz)))
            gpu_share = min(99.5, 100.0 * gpu_sub_ms / denom_ms) if gpu_sub_ms > 0.0 else 0.0
            return GPUMonitorSnapshot(
                enabled=True,
                device_name=self._device_name,
                sm_util_pct=round(self._sm_util_pct, 1),
                mem_bw_util_pct=round(self._mem_bw_util_pct, 1),
                vram_used_mb=round(self._vram_used_mb, 1),
                vram_total_mb=round(self._vram_total_mb, 1),
                torch_alloc_mb=round(alloc_mb, 1),
                torch_reserved_mb=round(res_mb, 1),
                gpu_temp_c=round(self._gpu_temp_c, 1),
                gpu_power_w=round(self._gpu_power_w, 1),
                gpu_clock_mhz=round(self._gpu_clock_mhz, 0),
                step_wall_ms=round(self._ema_step_wall_ms, 3),
                gpu_3d_pde_ms=round(self._ema_pde_ms, 3),
                gpu_pino_ms=round(self._ema_pino_ms, 3),
                gpu_elastic_ms=round(self._ema_elastic_ms, 3),
                gpu_share_pct=round(gpu_share, 1),
                active_voxels_m=round(self.active_voxels * 1e-6, 2),
                pino_operators=self.pino_operators,
                cfd_hz=round(self.cfd_hz, 1),
            )
