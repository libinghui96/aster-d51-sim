#!/usr/bin/env python3
"""
sim/hw_guard.py
================================================================================
服务器热安全与瞬态功耗防过载护栏 (Hardware Thermal & Power Guard)

根因背景:
  本机配置为 Intel Core i9-12900K (24 线程, BIOS 默认解锁 PL1/PL2 = 4095W) +
  NVIDIA GeForce RTX 5080 (默认 TDP 360W)。当多核 CPU 重任务、571万体素 3D CFD、
  PINO 训练与后台 Ollama 27B 大模型推理并发叠加时，CPU+GPU 瞬态峰值功耗可达 700W+，
  易触发主板过温保护或电源 OCP 断电关机。

防护机制 (零 sudo 依赖，微秒级内核/NVML 直读):
  1. 限制 OpenMP / MKL / OpenBLAS / PyTorch CPU 并发线程数 (默认 8 线程) 与批处理 nice 优先级；
  2. 通过 /sys/class/hwmon (coretemp) 与 libnvidia-ml.so.1 直接读取 CPU 封装温度、GPU 核心温度与实时功耗；
  3. 离线重计算 (CAD 布尔、PINO 训练、21 门禁验收) 启动前自动停止后台常驻 gpu_3d_server，杜绝双倍负载叠加；
  4. 训练/仿真循环内一旦检测到 CPU >= 80°C、GPU >= 73°C 或 GPU 功耗 >= 275W，自动插入微秒/毫秒级冷却窗口。
================================================================================
"""
from __future__ import annotations

import ctypes
import os
import signal
import time
from pathlib import Path
from typing import Dict, Optional


def quarantine_overclocked_cpus(safe_max_khz: int = 5_200_000) -> list[int]:
    """
    自动检测并隔离主板 BIOS 超频核心 (零 sudo 依赖，进程及子线程自动继承):
      - Intel i9-12900K 官方最高睿频规范为 5.20 GHz (5,200,000 kHz)；
      - 实测本机 ASUS ROG MAXIMUS Z690 HERO 将 cpu4-5/8-11 (尤以物理 Core 16 = cpu8/cpu9) 超频至 5.40 GHz，
        在单主线程高频脉冲负载下触发 CPU 8 寄存器翻转 (kern.log: insn_decode_mmio Oops / libpython segfault / ACPI BERT Fatal Reset)；
      - 若检测到部分逻辑核 scaling_max_freq > safe_max_khz，自动通过 os.sched_setaffinity 将当前进程绑定至未超频安全核!
    """
    if not hasattr(os, "sched_setaffinity"):
        return []
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return []

    safe_cpus: list[int] = []
    oc_cpus: list[int] = []
    for d in sorted(cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else 9999):
        suffix = d.name[3:]
        if not suffix.isdigit():
            continue
        cid = int(suffix)
        freq_file = d / "cpufreq" / "scaling_max_freq"
        try:
            max_khz = int(freq_file.read_text().strip())
        except Exception:
            safe_cpus.append(cid)
            continue
        if max_khz > safe_max_khz:
            oc_cpus.append(cid)
        else:
            safe_cpus.append(cid)

    if oc_cpus and len(safe_cpus) >= 4:
        try:
            allowed_now = os.sched_getaffinity(0)
            target_set = set(safe_cpus) & set(allowed_now)
            if len(target_set) >= 4:
                os.sched_setaffinity(0, target_set)
                return sorted(oc_cpus)
        except OSError:
            pass
    return []


def has_unclamped_overclocked_cpus(safe_max_khz: int = 5_200_000) -> bool:
    """检查宿主机是否仍存在未通过 setup_host_power_guard.sh 压制的 >5.20GHz BIOS 超频核心 (保护外部未绑核守护进程如 ollama)。"""
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return False
    for freq_file in cpu_root.glob("cpu[0-9]*/cpufreq/scaling_max_freq"):
        try:
            if int(freq_file.read_text().strip()) > safe_max_khz:
                return True
        except Exception:
            continue
    return False


def apply_cpu_thread_and_nice_caps(
    max_threads: int = 8,
    nice_delta: int = 4,
    safe_max_khz: int = 5_200_000,
) -> None:
    """隔离 BIOS 超频不稳定核心 (如 5.40GHz CPU 8)，并限制底层数值库线程扇出与批处理优先级。"""
    quarantine_overclocked_cpus(safe_max_khz=safe_max_khz)

    t_str = str(max(1, int(max_threads)))
    for env_key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(env_key, t_str)

    if nice_delta > 0:
        try:
            cur = os.nice(0)
            if cur < nice_delta:
                os.nice(nice_delta - cur)
        except OSError:
            pass

    try:
        import torch
        if torch.get_num_threads() > max_threads:
            torch.set_num_threads(max_threads)
    except Exception:
        pass


def stop_background_gpu_server_for_heavy_job() -> int:
    """离线重负载任务 (PINO 训练 / 全量门禁) 启动前暂停同用户后台 gpu_3d_server，防止 GPU 双重满载。"""
    uid = os.getuid()
    my_pid = os.getpid()
    proc_root = Path("/proc")
    killed = 0
    if not proc_root.exists():
        return 0
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == my_pid:
            continue
        try:
            if p.stat().st_uid != uid:
                continue
            cmd = p.joinpath("cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            if "sim.gpu_3d_server" in cmd or "sim/gpu_3d_server.py" in cmd:
                os.kill(pid, signal.SIGTERM)
                killed += 1
        except Exception:
            continue
    if killed > 0:
        time.sleep(0.4)
    return killed


class HardwareThermalPowerGuard:
    """微秒级 CPU Package 温度 + NVML GPU 温度/功耗直读与自适应冷却节流器。"""

    def __init__(
        self,
        max_cpu_c: float = 80.0,
        max_gpu_c: float = 73.0,
        max_gpu_w: float = 275.0,
    ):
        self.max_cpu_c = float(max_cpu_c)
        self.max_gpu_c = float(max_gpu_c)
        self.max_gpu_w = float(max_gpu_w)
        self._cpu_temp_file: Optional[Path] = self._discover_cpu_pkg_temp()
        self._nvml: Optional[ctypes.CDLL] = None
        self._nvml_handle = ctypes.c_void_p()
        self._init_nvml()

    @staticmethod
    def _discover_cpu_pkg_temp() -> Optional[Path]:
        hwmon_root = Path("/sys/class/hwmon")
        if not hwmon_root.exists():
            return None
        fallback: Optional[Path] = None
        for hw in sorted(hwmon_root.iterdir()):
            try:
                name = (hw / "name").read_text().strip()
            except OSError:
                continue
            if name == "coretemp":
                for lbl in sorted(hw.glob("temp*_label")):
                    try:
                        txt = lbl.read_text().strip()
                        inp = hw / lbl.name.replace("_label", "_input")
                        if "Package" in txt and inp.exists():
                            return inp
                        if fallback is None and inp.exists():
                            fallback = inp
                    except OSError:
                        continue
        return fallback

    def _init_nvml(self) -> None:
        for libname in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                lib = ctypes.CDLL(libname)
                if lib.nvmlInit_v2() == 0:
                    handle = ctypes.c_void_p()
                    if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(handle)) == 0:
                        self._nvml = lib
                        self._nvml_handle = handle
                        return
            except Exception:
                continue

    def read_metrics(self) -> Dict[str, float]:
        cpu_c = -1.0
        if self._cpu_temp_file is not None:
            try:
                cpu_c = int(self._cpu_temp_file.read_text().strip()) / 1000.0
            except Exception:
                cpu_c = -1.0

        gpu_c = -1.0
        gpu_w = -1.0
        if self._nvml is not None:
            try:
                t_val = ctypes.c_uint(0)
                if self._nvml.nvmlDeviceGetTemperature(self._nvml_handle, ctypes.c_uint(0), ctypes.byref(t_val)) == 0:
                    gpu_c = float(t_val.value)
                p_mw = ctypes.c_uint(0)
                if self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle, ctypes.byref(p_mw)) == 0:
                    gpu_w = float(p_mw.value) / 1000.0
            except Exception:
                pass
        return {"cpu_c": round(cpu_c, 1), "gpu_c": round(gpu_c, 1), "gpu_w": round(gpu_w, 1)}

    def is_hot_or_high_power(self) -> bool:
        m = self.read_metrics()
        return (
            (m["cpu_c"] >= self.max_cpu_c)
            or (m["gpu_c"] >= self.max_gpu_c)
            or (m["gpu_w"] >= self.max_gpu_w)
        )

    def cooldown_if_needed(self, max_wait_s: float = 1.5) -> float:
        """若检测到温度或瞬时功耗超标，同步 GPU 并短暂休眠降温，返回累计休眠秒数。"""
        waited = 0.0
        while waited < max_wait_s:
            m = self.read_metrics()
            over_cpu = m["cpu_c"] >= self.max_cpu_c
            over_gpu_t = m["gpu_c"] >= self.max_gpu_c
            over_gpu_w = m["gpu_w"] >= self.max_gpu_w
            if not (over_cpu or over_gpu_t or over_gpu_w):
                break
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            dt_sleep = 0.18 if (over_cpu or over_gpu_t) else 0.08
            time.sleep(dt_sleep)
            waited += dt_sleep
        return waited
