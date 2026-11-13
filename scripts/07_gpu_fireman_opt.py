#!/usr/bin/env python3
"""
scripts/07_gpu_fireman_opt.py
O6: 基于本地 NVIDIA RTX 5080 CUDA 大规模并行向量化仿真 (B = 4096 辆机车同步演化)，
采用交叉熵进化策略 (CEM) 对 Aster D51 自动司炉控制律参数进行全局寻优与参数辨识，
导出最优策略配置: configs/auto_fireman_policy.json，并在 3D MuJoCo 全模型上闭环验收。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@torch.no_grad()
def evaluate_fireman_population_gpu(
    params: torch.Tensor,
    target_speed_kmh: float = 52.0,
    sim_seconds: float = 60.0,
    dt: float = 0.02,
) -> torch.Tensor:
    """
    在 RTX 5080 上并行仿真 B 组候选司炉策略持续 60s (涵盖冷缸排液启动 -> 提速收截止比 -> 巡航稳压稳水位)。
    params 维度: [B, 7] ->
      0: p_target_kgf      in [3.35, 3.88]
      1: cutoff_cruise     in [0.45, 0.82]
      2: kp_reg            in [0.005, 0.045]
      3: ki_reg            in [0.001, 0.012]
      4: kp_blower         in [0.30, 1.80]
      5: kp_bypass         in [0.02, 0.14]
      6: ff_bypass_base    in [0.20, 0.70]
    """
    device = params.device
    B = params.shape[0]

    p_target = params[:, 0]
    cutoff_cruise = params[:, 1]
    kp_reg = params[:, 2]
    ki_reg = params[:, 3]
    kp_blw = params[:, 4]
    kp_byp = params[:, 5]
    ff_byp = params[:, 6]

    # 初始状态: 3.55 kgf/cm² 锅炉气压，气缸微冷 (78°C 需排液暖缸)，水位 48%
    T_boil = torch.full((B,), 147.2, device=device)
    M_water = torch.full((B,), 0.207, device=device)
    T_wall = torch.full((B,), 78.0, device=device)
    cond_g = torch.full((B,), 0.34, device=device)
    omega = torch.zeros((B,), device=device)
    spd_int = torch.zeros((B,), device=device)
    sv_open = torch.zeros((B,), dtype=torch.bool, device=device)

    cost = torch.zeros((B,), device=device)
    n_steps = int(sim_seconds / dt)

    for step_i in range(n_steps):
        # IAPWS 近似饱和表压 (kgf/cm^2)
        p_sat_pa = 101325.0 * torch.exp(13.7 - 5120.0 / (T_boil + 273.15))
        p_kgf = torch.clamp((p_sat_pa - 101325.0) / 98066.5, min=0.0)
        sv_open = torch.where(p_kgf >= 4.00, torch.ones_like(sv_open), sv_open)
        sv_open = torch.where(p_kgf <= 3.60, torch.zeros_like(sv_open), sv_open)

        water_pct = 100.0 * (M_water * 1000.0 - 140.0) / 140.0
        speed_kmh = omega * 0.0219 * 32.0 * 3.6

        # 候选司炉策略输出
        need_drain = (T_wall < 102.0) | (cond_g > 0.26)
        spd_ratio = torch.clamp(speed_kmh / 42.0, 0.0, 1.0)
        cutoff = 0.85 - (0.85 - cutoff_cruise) * spd_ratio

        v_err = target_speed_kmh - speed_kmh
        spd_int = torch.clamp(spd_int + v_err * dt, -18.0, 35.0)
        reg = torch.clamp(0.28 + kp_reg * v_err + ki_reg * spd_int, 0.0, 0.92)
        reg = torch.where(need_drain, torch.clamp(reg, max=0.42), reg)

        blw = torch.clamp(kp_blw * (p_target - p_kgf) - torch.clamp(speed_kmh * 0.006, max=0.35) + 0.12, 0.0, 1.0)

        # 蒸汽消耗率: 省汽截止比 cutoff 越小，单位功蒸汽消耗越低，但单转平均扭矩也按 sin 规律折减
        p_chest_kgf = p_kgf * (reg ** 1.15)
        torque_factor = torch.sin(0.5 * math.pi * torch.clamp(cutoff / 0.85, 0.2, 1.0))
        steam_cyl_gs = 0.16 * p_chest_kgf * (0.25 + 0.75 * cutoff) * (0.15 + 0.045 * omega)
        steam_drn_gs = torch.where(need_drain, 0.09 * p_chest_kgf, torch.zeros_like(p_kgf))
        steam_blw_gs = 0.045 * p_kgf * blw
        steam_sv_gs = torch.where(sv_open, 0.65 * p_kgf, torch.zeros_like(p_kgf))
        steam_tot_gs = steam_cyl_gs + steam_drn_gs + steam_blw_gs + steam_sv_gs

        byp = torch.clamp(ff_byp + kp_byp * (water_pct - 50.0) - 0.18 * steam_tot_gs, 0.0, 1.0)

        # 轴动泵给水与热平衡演化
        pump_max_gs = 0.026 * omega
        fw_gs = torch.clamp(pump_max_gs * (1.0 - byp ** 1.25), min=0.0)

        draft = torch.clamp(0.14 + 0.88 * torch.tanh(1.8 * steam_cyl_gs + 9.5 * steam_blw_gs), 0.14, 1.28)
        fuel_cmd = torch.where(p_kgf > 3.90, torch.clamp(1.0 - 2.5 * (p_kgf - 3.90), min=0.35), torch.ones_like(p_kgf))
        q_in_w = 640.0 * draft * fuel_cmd - 0.42 * (T_boil - 25.0)

        dT_boil = (q_in_w - steam_tot_gs * 2120.0 - fw_gs * 4.186 * (T_boil - 25.0)) / 920.0
        T_boil = torch.clamp(T_boil + dT_boil * dt, 95.0, 165.0)
        M_water = torch.clamp(M_water + (fw_gs - steam_tot_gs) * 1e-3 * dt, 0.10, 0.30)

        # 暖缸与冷凝水演化
        T_wall = T_wall + ((162.0 - T_wall) * 0.045 * torch.clamp(steam_cyl_gs, max=1.2)) * dt
        d_cond = torch.where(T_wall < 105.0, 0.18 * steam_cyl_gs, torch.zeros_like(cond_g)) - torch.where(
            need_drain, 0.45 * torch.ones_like(cond_g), 0.02 * torch.ones_like(cond_g)
        )
        cond_g = torch.clamp(cond_g + d_cond * dt, 0.0, 1.2)
        hydrolock = (~need_drain) & (cond_g > 0.55)

        # 纵向列车动力学 (含 R2m 弯道整列阻力 + 给水泵反力矩)
        tau_drive = 0.048 * p_chest_kgf * torque_factor * torch.where(hydrolock, torch.full_like(omega, -0.5), torch.ones_like(omega))
        tau_drag = 0.0055 * omega + 0.00048 * omega * torch.abs(omega) + 0.0012 * fw_gs * p_kgf
        omega = torch.clamp(omega + ((tau_drive - tau_drag) / 0.0042) * dt, 0.0, 45.0)

        # 累计多目标代价 (10s 暖缸期后严考核速度精度、50% 水位精度、零安全阀泄压与省汽经济性)
        w_spd = 1.0 if step_i * dt > 10.0 else 0.15
        step_cost = (
            w_spd * ((speed_kmh - target_speed_kmh) / 10.0) ** 2
            + 0.45 * ((water_pct - 50.0) / 5.0) ** 2
            + 0.30 * ((p_kgf - 3.70) / 0.35) ** 2
            + 18.0 * steam_sv_gs
            + 0.22 * steam_tot_gs
            + 50.0 * hydrolock.float()
        )
        cost += step_cost * dt

    return cost


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5080 自动司炉并行寻优与 3D 闭环验收")
    parser.add_argument("--out", type=Path, default=Path("configs/auto_fireman_policy.json"))
    parser.add_argument("--pop-size", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=28)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[GPU-OPT] 启动 Auto-Fireman 并行策略寻优 | 设备: {dev_name} | 种群={args.pop_size} x {args.generations} 代")

    bounds_lo = torch.tensor([3.40, 0.46, 0.006, 0.0015, 0.35, 0.025, 0.22], device=device)
    bounds_hi = torch.tensor([3.86, 0.80, 0.042, 0.0110, 1.75, 0.130, 0.68], device=device)

    mean = 0.5 * (bounds_lo + bounds_hi)
    std = 0.28 * (bounds_hi - bounds_lo)
    elite_k = max(32, args.pop_size // 16)

    t0 = time.perf_counter()
    best_cost = float("inf")
    best_vec = mean.clone()

    for gen in range(args.generations):
        samples = torch.clamp(mean + std * torch.randn((args.pop_size, 7), device=device), bounds_lo, bounds_hi)
        costs = evaluate_fireman_population_gpu(samples)
        top_vals, top_idx = torch.topk(costs, k=elite_k, largest=False)
        elites = samples[top_idx]
        mean = 0.75 * elites.mean(dim=0) + 0.25 * mean
        std = 0.75 * elites.std(dim=0) + 0.25 * std * 0.92
        if float(top_vals[0]) < best_cost:
            best_cost = float(top_vals[0])
            best_vec = elites[0].clone()

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_s = time.perf_counter() - t0
    total_traj = args.pop_size * args.generations

    vals = [round(float(x), 5) for x in best_vec.cpu().tolist()]
    policy_doc = {
        "meta": {
            "generator": "scripts/07_gpu_fireman_opt.py",
            "gpu_device": dev_name,
            "parallel_trajectories_evaluated": total_traj,
            "simulated_locomotive_hours": round(total_traj * 60.0 / 3600.0, 1),
            "compute_time_s": round(elapsed_s, 3),
            "best_objective_cost": round(best_cost, 3),
        },
        "optimal_policy": {
            "target_speed_kmh": 52.0,
            "p_target_kgf": vals[0],
            "cutoff_start": 0.85,
            "cutoff_cruise": vals[1],
            "cutoff_speed_span_kmh": 42.0,
            "kp_reg": vals[2],
            "ki_reg": vals[3],
            "kp_blower": vals[4],
            "kp_bypass": vals[5],
            "ff_bypass_base": vals[6],
            "warmup_wall_temp_c": 102.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[GPU-OPT] 寻优完成! 耗时 {elapsed_s:.2f}s 评估 {total_traj:,} 条轨迹 "
        f"(等效 {policy_doc['meta']['simulated_locomotive_hours']:,.1f} 小时实车试运转)"
    )
    print(f"          最优参数: P_target={vals[0]:.2f} kgf/cm², 巡航省汽截止比={vals[1]*100:.1f}%, kp_byp={vals[5]:.4f}")

    # 在真实 3D MuJoCo 全模型 (d51_r2m.xml 弯道牵引煤水车) 上执行 12s 闭环验证
    from sim.auto_fireman import AutoFiremanAgent
    from sim.run_cosim import CoSimRunner

    runner = CoSimRunner(ROOT / "models" / "d51_r2m.xml", preset="hot")
    runner.thermo.T_cyl_wall_c = 75.0  # 故意设微冷气缸检验自动开排液阀暖缸能力
    agent = AutoFiremanAgent(args.out, target_speed_kmh=50.0)
    sv_pop_steps = 0
    for step_i in range(20000):  # 10.0s @ 2000Hz
        if step_i % 20 == 0:
            # 100 Hz 司炉观测与操作更新
            tel_peek = runner.step_once()
            agent.update(0.01, tel_peek, runner.ctrl)
            if tel_peek.safety_valve_open:
                sv_pop_steps += 1
        else:
            tel_peek = runner.step_once()
            if tel_peek.safety_valve_open:
                sv_pop_steps += 1

    print(
        f"[3D-VERIFY] R2m 整列 10s 自动司炉验收 -> 车速={tel_peek.scale_speed_kmh:.1f} km/h | "
        f"气压={tel_peek.boiler_pressure_kgf:.2f} kgf/cm² | 水位={tel_peek.true_water_level_pct:.1f}% | "
        f"逆转机截止比={runner.ctrl.reverser*100:+.1f}% | 排液阀={'OPEN' if runner.ctrl.drain_cock else 'CLOSED'} | "
        f"安全阀起跳占比={100.0*sv_pop_steps/20000:.1f}%"
    )


if __name__ == "__main__":
    main()
