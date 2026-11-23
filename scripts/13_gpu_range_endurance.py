#!/usr/bin/env python3
"""
scripts/13_gpu_range_endurance.py
Stage 27: 基于本地 NVIDIA RTX 5080 CUDA 大规模并行曲柄角级物理仿真引擎 (GPUConsistEnduranceEngine)，
评测 Aster Hobby JNR D51 在不同整备方式 (热炉/冷炉起汽)、容积规格 (经典 130ml/272ml vs 2010 规格 200ml/320ml)、
线路几何 (平直轨道 / R2m 最小曲线 / 3D ±1.5% 坡道大椭圆环线)、编组牵引载荷 (单机 / 2节 / 4节 / 6节货车)、
环境温度 (夏季 28°C / 标准 25°C / 冬季 5°C 逆风) 与司炉策略下的完整满箱轨道运行距离，
并通过 B=128 蒙特卡洛物理公差抽样给出真实跑距置信区间 [P10, P50, P90]。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.gpu_track_endurance import EnduranceScenario, GPUConsistEnduranceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5080 Aster D51 满箱全行程轨道续航跑距基准评测")
    parser.add_argument("--out", type=Path, default=Path("configs/d51_range_benchmark.json"))
    parser.add_argument("--mc-samples", type=int, default=128)
    args = parser.parse_args()

    eng = GPUConsistEnduranceEngine()
    scenarios = [
        EnduranceScenario("01_hot_straight_light_economy", track_mode="track", preset="hot", ai_driver=True, trailing_cars=0, target_speed_kmh=54.0),
        EnduranceScenario("02_hot_oval_grade_light_economy", track_mode="oval_grade", preset="hot", ai_driver=True, trailing_cars=0, target_speed_kmh=52.0),
        EnduranceScenario("03_hot_oval_grade_2cars_economy", track_mode="oval_grade", preset="hot", ai_driver=True, trailing_cars=2, target_speed_kmh=50.0),
        EnduranceScenario("04_hot_oval_grade_4cars_heavy", track_mode="oval_grade", preset="hot", ai_driver=True, trailing_cars=4, trailing_load_kg=1.2, target_speed_kmh=46.0),
        EnduranceScenario("05_hot_r2m_curve_light", track_mode="curve_r2m", preset="hot", ai_driver=True, trailing_cars=0, target_speed_kmh=48.0),
        EnduranceScenario("06_cold_start_oval_2cars_full_cycle", track_mode="oval_grade", preset="cold", ai_driver=True, trailing_cars=2, target_speed_kmh=50.0),
        EnduranceScenario("07_2010spec_200ml_cold_oval_2cars", track_mode="oval_grade", preset="cold", tank_spec_2010=True, ai_driver=True, trailing_cars=2, target_speed_kmh=50.0),
        EnduranceScenario("08_2010spec_200ml_hot_oval_light", track_mode="oval_grade", preset="hot", tank_spec_2010=True, ai_driver=True, trailing_cars=0, target_speed_kmh=54.0),
        EnduranceScenario("09_winter_5C_headwind_oval_2cars", track_mode="oval_grade", preset="hot", ai_driver=True, trailing_cars=2, ambient_temp_c=5.0, headwind_m_s=1.5, target_speed_kmh=50.0),
        EnduranceScenario("10_unnotched_full_cutoff_waste", track_mode="oval_grade", preset="hot", ai_driver=False, regulator=0.25, reverser=0.92, bypass_valve=0.92, blower_valve=0.65, trailing_cars=2),
    ]

    det_report = eng.simulate_batch(scenarios, max_sim_s=2600.0, dt=0.0025, monte_carlo_spread=0.0)

    mc_base = EnduranceScenario("mc_oval_2cars_classic", track_mode="oval_grade", preset="cold", ai_driver=True, trailing_cars=2, target_speed_kmh=50.0)
    mc_report = eng.simulate_batch([mc_base for _ in range(args.mc_samples)], max_sim_s=2600.0, dt=0.0025, monte_carlo_spread=0.08, seed=2026)
    mc_dists = np.array([r["distance_m"] for r in mc_report["results"]], dtype=np.float64)
    mc_mins = np.array([r["duration_min"] for r in mc_report["results"]], dtype=np.float64)

    summary = {
        "device": det_report["device"],
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_wall_s": det_report["wall_time_s"],
        "deterministic_speedup_x": det_report["effective_speedup_vs_realtime"],
        "scenarios": det_report["results"],
        "monte_carlo_130ml_cold_oval_2cars": {
            "samples": args.mc_samples,
            "wall_time_s": mc_report["wall_time_s"],
            "effective_speedup_x": mc_report["effective_speedup_vs_realtime"],
            "distance_p10_m": round(float(np.percentile(mc_dists, 10)), 1),
            "distance_p50_m": round(float(np.percentile(mc_dists, 50)), 1),
            "distance_p90_m": round(float(np.percentile(mc_dists, 90)), 1),
            "scale_distance_p50_km": round(float(np.percentile(mc_dists, 50)) * 30.0 / 1000.0, 3),
            "duration_p10_min": round(float(np.percentile(mc_mins, 10)), 2),
            "duration_p50_min": round(float(np.percentile(mc_mins, 50)), 2),
            "duration_p90_min": round(float(np.percentile(mc_mins, 90)), 2),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GPU-ENDURANCE] 设备={summary['device']} | 10 工况耗时 {det_report['wall_time_s']}s ({det_report['effective_speedup_vs_realtime']}x 实时)")
    for r in det_report["results"]:
        print(
            f"  - {r['name']:36s}: {r['distance_m']:7.1f} m ({r['scale_distance_km']:5.2f} km | {r['laps']:5.1f} laps) "
            f"in {r['duration_min']:5.2f} min [{r['limiting_factor']}]"
        )
    mc = summary["monte_carlo_130ml_cold_oval_2cars"]
    print(
        f"[MC-{args.mc_samples}] 130ml 冷炉起汽+3D椭圆+2节货车跑距置信区间: "
        f"P10={mc['distance_p10_m']}m | P50={mc['distance_p50_m']}m | P90={mc['distance_p90_m']}m "
        f"(耗时 {mc['wall_time_s']}s, {mc['effective_speedup_x']}x 实时)"
    )


if __name__ == "__main__":
    main()
