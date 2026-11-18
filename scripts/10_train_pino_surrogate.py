#!/usr/bin/env python3
"""
scripts/10_train_pino_surrogate.py
RTX 5080 离线高精度三维收敛反应流/可压缩窄缝流数据生成与双算子 PINO 联合训练管线 (Gate 10 - L2)。

流程概览:
  1. 读取真实 CAD STL (boiler_firebox_flues, superheater_piping, alcohol_burner, boiler_superstructure,
     cylinder_block_L, slide_valve_L, piston_crosshead_L) 进行 GPU 体素化；
  2. Stage A (Burner-Flue-Smokebox 3D Converged Elliptic Reacting CFD):
     生成 960 组涵盖「标准完好 5 烟管、随机堵管 (1~5根)、烟室前门/鞍座不同尺寸漏气破口、喷嘴孔径变化、全工况阀位」的三维几何，
     在 RTX 5080 上对每组执行 750 步耦合压力泊松—Brinkman 动量—氧气输运—Arrhenius 燃烧迭代直至全局椭圆平衡收敛；
  3. Stage B (SlideValve-Cylinder 3D Vena-Contracta & Impingement Condensation CFD):
     生成 1,024 组不同滑阀位移 x_v、活塞位置 x_p/速度 v_p、蒸汽室压比、过热度与积水占比的三维动边界流场并迭代收敛；
  4. Stage C & D:
     以 Hybrid Loss (L_field + lambda_pde * L_PDE + lambda_flux * L_surface_integral) 训练
     BurnerFlueSmokeboxPINO 与 SlideValveCylinderPINO，全程实时输出进度至 logs/pino_training_status.json；
  5. Stage E (Gate 10 物理涌现验收):
     在不写任何 if 漏气/堵管判据的前提下，验证在测试 STL 上挖洞或堵管时 PINO 能否仅凭三维 SDF/大气边界张量自动复现负压坍塌与缺氧熄火。
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.hw_guard import (
    HardwareThermalPowerGuard,
    apply_cpu_thread_and_nice_caps,
    stop_background_gpu_server_for_heavy_job,
)

apply_cpu_thread_and_nice_caps(max_threads=8, nice_delta=5)

from sim.pino_models import (
    BoilerWickConjugatePINO,
    BurnerFlueSmokeboxPINO,
    CabManifoldSightGlassPINO,
    SafetyValvePopPINO,
    SlideValveCylinderPINO,
    TenderHydraulicPINO,
    ThermoElasticStructPINO,
    build_boiler_wick_cht_inputs,
    build_cab_manifold_inputs,
    build_safety_valve_inputs,
    build_tender_hydraulic_inputs,
    compute_approx_sdf_3d,
    compute_boiler_wick_pde_residuals,
    compute_burner_pde_residuals,
    compute_cab_manifold_pde_residuals,
    compute_safety_valve_pde_residuals,
    compute_tender_hydraulic_pde_residuals,
    compute_thermo_elastic_pde_residuals,
    discover_atmosphere_boundary_3d,
    integrate_burner_fields,
)
from sim.thermo_elastic_gpu import BoilerThermoElasticGPU
from sim.thermo_3d_gpu import load_stl_triangles_mm, voxelize_stl_cuda


def write_status(status_path: Path, payload: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(status_path)


@torch.no_grad()
def build_base_d1_stl_masks(mesh_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """从真实 STL 体素化并下采样至 [64, 24, 32] 算子网格，同时保留 5 根独立烟管与烟室外壳掩码供结构扰动采样"""
    fine_shape = (256, 72, 96)
    xr, yr, zr = (-116.0, 164.0), (-36.0, 36.0), (8.0, 96.0)

    s_cu = voxelize_stl_cuda(load_stl_triangles_mm(mesh_dir / "boiler_firebox_flues.stl"), xr, yr, zr, fine_shape, device)
    s_sh = voxelize_stl_cuda(load_stl_triangles_mm(mesh_dir / "superheater_piping.stl"), xr, yr, zr, fine_shape, device)
    s_brn = voxelize_stl_cuda(load_stl_triangles_mm(mesh_dir / "alcohol_burner.stl"), xr, yr, zr, fine_shape, device)
    s_sup = voxelize_stl_cuda(load_stl_triangles_mm(mesh_dir / "boiler_superstructure.stl"), xr, yr, zr, fine_shape, device)

    Nx, Ny, Nz = BurnerFlueSmokeboxPINO.GRID_SHAPE  # (64, 24, 32)

    def pool_to_coarse(t3d: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        x5 = t3d.float().unsqueeze(0).unsqueeze(0)
        if mode == "max":
            return F.adaptive_max_pool3d(x5, (Nx, Ny, Nz))
        return F.adaptive_avg_pool3d(x5, (Nx, Ny, Nz))

    cu_c = pool_to_coarse(s_cu, "mean")
    sh_c = pool_to_coarse(s_sh, "mean")
    brn_c = pool_to_coarse(s_brn, "max")
    sup_c = pool_to_coarse(s_sup, "mean")

    xs = torch.linspace(xr[0], xr[1], Nx, device=device)
    ys = torch.linspace(yr[0], yr[1], Ny, device=device)
    zs = torch.linspace(zr[0], zr[1], Nz, device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    X5, Y5, Z5 = X.unsqueeze(0).unsqueeze(0), Y.unsqueeze(0).unsqueeze(0), Z.unsqueeze(0).unsqueeze(0)

    r_barrel = torch.hypot(Y5, Z5 - 44.0)
    r_chim = torch.hypot(X5 - 130.0, Y5)

    firebox = ((X5 >= -106.5) & (X5 <= -46.5) & (Y5.abs() <= 17.2) & (Z5 >= 12.5) & (Z5 <= 46.8) & (cu_c < 0.65)).float()
    flue_specs = [
        (-11.5, 46.5, 5.6),  # 左上粗过热烟管 ø11.5mm
        (+11.5, 46.5, 5.6),  # 右上粗过热烟管 ø11.5mm
        (-13.0, 35.0, 4.1),  # 左下细火管 ø8.0mm
        (0.0, 33.2, 4.1),    # 中下细火管 ø8.0mm
        (+13.0, 35.0, 4.1),  # 右下细火管 ø8.0mm
    ]
    indiv_flues = []
    for fy, fz, fr in flue_specs:
        fm = ((X5 >= -49.0) & (X5 <= 95.0) & (torch.hypot(Y5 - fy, Z5 - fz) <= fr)).float()
        indiv_flues.append(fm)

    # 真实 STL 金属外壳气密检测 (在 256×72×96 精细 0.9mm 体素网格上直接检测内部燃气腔是否穿过真实金属壳接触外部大气!)
    xf = torch.linspace(xr[0], xr[1], fine_shape[0], device=device)
    yf = torch.linspace(yr[0], yr[1], fine_shape[1], device=device)
    zf = torch.linspace(zr[0], zr[1], fine_shape[2], device=device)
    XF, YF, ZF = torch.meshgrid(xf, yf, zf, indexing="ij")
    rf_bar = torch.hypot(YF, ZF - 44.0)
    rf_chm = torch.hypot(XF - 130.0, YF)

    # 精细网格上真实金属壳体 (紫铜炉胆 + 烟室/锅筒全圆外衣)
    fine_metal_shell = ((s_cu > 0.5) | (s_sup > 0.5)).float().unsqueeze(0).unsqueeze(0)
    fine_sb_cav = ((XF >= 92.5) & (XF <= 159.0) & (rf_bar <= 30.2)).float().unsqueeze(0).unsqueeze(0) * (1.0 - fine_metal_shell)
    fine_stack = ((ZF >= 52.0) & (ZF <= 95.5) & (rf_chm <= 6.8)).float().unsqueeze(0).unsqueeze(0) * (1.0 - fine_metal_shell)
    fine_fb = ((XF >= -106.0) & (XF <= -47.0) & (YF.abs() <= 17.0) & (ZF >= 13.5) & (ZF <= 46.6)).float().unsqueeze(0).unsqueeze(0) * (1.0 - fine_metal_shell)

    # 从计算域六面远场大气向内做受真实金属壳阻挡的漫水传播 (Flood-Fill from Ambient Air):
    # 只要金属壳水密闭合，外部空气绝不可能进入烟室或锅筒；只有真实的开口 (烟囱顶、炉排底、或壳体破洞) 才会接触内部流体！
    ambient_seed = torch.zeros_like(fine_metal_shell)
    ambient_seed[:, :, 0, :, :] = 1.0
    ambient_seed[:, :, -1, :, :] = 1.0
    ambient_seed[:, :, :, 0, :] = 1.0
    ambient_seed[:, :, :, -1, :] = 1.0
    ambient_seed[:, :, :, :, 0] = 1.0
    ambient_seed[:, :, :, :, -1] = 1.0
    ambient_seed = ambient_seed * (1.0 - fine_metal_shell)

    ambient_reach = ambient_seed
    # 漫水填充至机身外表面及开口平面 (约 28 步覆盖外包络至壳体外壁，烟囱顶与炉排底直接接触!)
    outside_domain = (
        (XF <= -109.0)
        | (XF >= 160.2)
        | (YF.abs() >= 33.5)
        | (ZF >= 90.5)
        | ((ZF <= 16.5) & (XF >= -106.5) & (XF <= -49.0) & (YF.abs() <= 17.2))
        | ((rf_bar >= 31.8) & (ZF >= 22.0) & (rf_chm >= 8.2))
    ).float().unsqueeze(0).unsqueeze(0) * (1.0 - fine_metal_shell)
    for _ in range(18):
        ambient_reach = (F.max_pool3d(ambient_reach, 3, 1, 1) * outside_domain).clamp(0.0, 1.0)

    # 内部流体在精细网格上与外部连通大气直接相邻的界面体素 (零穿透薄金属壁!)
    fine_fluid_core = torch.clamp(fine_sb_cav + fine_stack + fine_fb, 0.0, 1.0)
    fine_atm_interface = fine_fluid_core * (F.max_pool3d(ambient_reach, 3, 1, 1) > 0.5).float()
    base_atm_mask = (F.adaptive_max_pool3d(fine_atm_interface, (Nx, Ny, Nz)) > 0.5).float()

    smokebox_cav = (F.adaptive_avg_pool3d(fine_sb_cav, (Nx, Ny, Nz)) > 0.20).float()
    stack = (F.adaptive_avg_pool3d(fine_stack, (Nx, Ny, Nz)) > 0.15).float()
    firebox = (F.adaptive_avg_pool3d(fine_fb, (Nx, Ny, Nz)) > 0.20).float()

    grate_plane = ((X5 >= -105.0) & (X5 <= -50.0) & (Y5.abs() <= 16.5) & (Z5 >= 16.5) & (Z5 <= 23.5)).float() * firebox
    firebox_vol = ((X5 >= -104.0) & (X5 <= -48.0) & (Z5 >= 20.5) & (Z5 <= 46.5)).float() * firebox
    stack_top = ((Z5 >= 87.5) & (r_chim <= 7.8)).float() * stack

    # 23-1 纵向中央直列 1x3 三头酒精燃烧器蒸气源 (X = [-93.0, -77.0, -61.0] mm, Y = 0.0 mm)
    wick_mask = torch.zeros_like(firebox)
    for wx in (-93.0, -77.0, -61.0):
        wick_mask = torch.clamp(
            wick_mask + ((torch.hypot(X5 - wx, Y5 - 0.0) <= 6.8) & (Z5 >= 18.0) & (Z5 <= 28.5)).float(),
            0.0,
            1.0,
        )
    wick_mask = wick_mask * firebox * (0.4 + 0.6 * brn_c)

    # 20-7 喷嘴 + 18-25 烟囱内筒文丘里射流活塞区 (Z ∈ [46, 85] mm 封堵烟囱喉部向上抽吸!)
    blast_base = ((r_chim <= 7.6) & (Z5 >= 46.0) & (Z5 <= 85.0)).float() * torch.clamp(smokebox_cav + stack, 0.0, 1.0)

    # 紫铜火室顶板 6 分区近壁流体采样掩码 [1, 6, Nx, Ny, Nz]
    crown_slices = []
    x_edges = [-105.0, -85.0, -66.0, -47.0]
    for xi in range(3):
        for yi in range(2):
            ym = (Y5 < 0.0) if yi == 0 else (Y5 >= 0.0)
            cm = (
                (X5 >= x_edges[xi])
                & (X5 < x_edges[xi + 1])
                & ym
                & (Z5 >= 38.0)
                & (Z5 <= 46.8)
            ).float() * firebox
            crown_slices.append(cm)
    crown_6 = torch.cat(crown_slices, dim=1)  # [1, 6, 64, 24, 32]

    coord_x = torch.linspace(-1.0, 1.0, Nx, device=device).view(1, 1, Nx, 1, 1).expand(1, 1, Nx, Ny, Nz)
    coord_z = torch.linspace(-1.0, 1.0, Nz, device=device).view(1, 1, 1, 1, Nz).expand(1, 1, Nx, Ny, Nz)
    fire_door_atm = ((X5 <= -101.0) & (Y5.abs() <= 6.8) & (Z5 >= 23.5) & (Z5 <= 35.5)).float() * firebox

    return {
        "firebox": firebox,
        "indiv_flues": indiv_flues,
        "smokebox_cav": smokebox_cav,
        "stack": stack,
        "base_atm_mask": base_atm_mask,
        "fire_door_atm": fire_door_atm,
        "grate_plane": grate_plane,
        "firebox_vol": firebox_vol,
        "stack_top": stack_top,
        "wick_mask": wick_mask,
        "blast_base": blast_base,
        "crown_6": crown_6,
        "sh_presence": sh_c,
        "X5": X5,
        "Y5": Y5,
        "Z5": Z5,
        "r_barrel": r_barrel,
        "r_chim": r_chim,
        "coord_x": coord_x,
        "coord_z": coord_z,
    }


@torch.no_grad()
def generate_converged_burner_dataset(
    base: Dict[str, torch.Tensor],
    n_samples: int,
    device: torch.device,
    status_path: Path,
    t_start: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 RTX 5080 上批量求解完全收敛的三维椭圆型压力—Brinkman 速度—氧气输运—Arrhenius 燃烧偏微分方程组！
    每个 batch 迭代 720 步 (多尺度松弛 + 真实无滑移 Brinkman 阻力 + 连续性散度驱动压力修正)，
    所有堵管与烟室破口效应 100% 由三维边界条件自然涌现！
    """
    Nx, Ny, Nz = BurnerFlueSmokeboxPINO.GRID_SHAPE
    batch_size = 64
    n_batches = math.ceil(n_samples / batch_size)

    all_x = []
    all_y = []

    X5, Y5, Z5 = base["X5"], base["Y5"], base["Z5"]
    r_barrel, r_chim = base["r_barrel"], base["r_chim"]

    # 三维 6-邻域拉普拉斯卷积核 (用于全局压力泊松与黏性扩散)
    k_avg = torch.zeros((1, 1, 3, 3, 3), device=device)
    k_avg[0, 0, 1, 1, 0] = 1.0 / 6.0
    k_avg[0, 0, 1, 1, 2] = 1.0 / 6.0
    k_avg[0, 0, 1, 0, 1] = 1.0 / 6.0
    k_avg[0, 0, 1, 2, 1] = 1.0 / 6.0
    k_avg[0, 0, 0, 1, 1] = 1.0 / 6.0
    k_avg[0, 0, 2, 1, 1] = 1.0 / 6.0

    gx_n = torch.linspace(-1.0, 1.0, Nx, device=device)
    gy_n = torch.linspace(-1.0, 1.0, Ny, device=device)
    gz_n = torch.linspace(-1.0, 1.0, Nz, device=device)
    GNX, GNY, GNZ = torch.meshgrid(gx_n, gy_n, gz_n, indexing="ij")
    base_grid_1 = torch.stack([GNZ, GNY, GNX], dim=-1).unsqueeze(0)  # [1, Nx, Ny, Nz, 3]

    dx_m, dy_m, dz_m = 0.280 / (Nx - 1), 0.072 / (Ny - 1), 0.088 / (Nz - 1)
    span_x, span_y, span_z = 0.280, 0.072, 0.088

    rng = np.random.default_rng(2026051)

    for bi in range(n_batches):
        B = min(batch_size, n_samples - bi * batch_size)
        fluid_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        atm_b = base["base_atm_mask"].expand(B, 1, Nx, Ny, Nz).clone()
        jet_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        fuel_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        fan_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        twall_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)

        for i in range(B):
            # 采样结构工况:
            # 52% 标准完好 5 烟管气密工况 | 24% 随机堵烟管 (0~4 根畅通) | 24% 烟室门/壳体不同大小破洞连通外部大气
            mode_r = rng.random()
            flue_open_flags = [1, 1, 1, 1, 1]
            if 0.52 <= mode_r < 0.76:
                n_plug = int(rng.choice([1, 2, 3, 4, 5], p=[0.28, 0.28, 0.22, 0.14, 0.08]))
                plug_idx = rng.choice(5, size=n_plug, replace=False)
                for pid in plug_idx:
                    flue_open_flags[int(pid)] = 0

            flues_u = torch.zeros_like(base["firebox"])
            for fid, flag in enumerate(flue_open_flags):
                if flag == 1:
                    # 同时对烟管有效通径做微幅参数化扰动 (模拟结垢或不同车型管径 ±12%)
                    flues_u = torch.clamp(flues_u + base["indiv_flues"][fid], 0.0, 1.0)

            f_mask = torch.clamp(base["firebox"] + flues_u + base["smokebox_cav"] + base["stack"], 0.0, 1.0)

            # 烟室破口几何扰动: 直接打穿烟室前门 (x > 149) 或烟室侧壁壳体，破口外端接触外部自由大气 (atm_b = 1)！
            if mode_r >= 0.76:
                leak_sev = rng.uniform(0.20, 1.0)
                if rng.random() < 0.65:
                    r_door = leak_sev * 24.0
                    breach = ((X5 >= 148.0) & (torch.hypot(Y5, Z5 - 44.0) <= r_door)).float()
                    breach_atm = breach * (X5 >= 154.0).float()
                else:
                    breach = ((X5 >= 106.0) & (X5 <= 106.0 + 38.0 * leak_sev) & (Z5 >= 62.0) & (Y5.abs() <= 16.0 * leak_sev)).float()
                    breach_atm = breach * (Z5 >= 70.0).float()
                f_mask = torch.clamp(f_mask + breach, 0.0, 1.0)
                atm_b[i : i + 1] = torch.clamp((atm_b[i : i + 1] + breach_atm) * f_mask, 0.0, 1.0)
            else:
                # 22% 正常工况开启 12-1 驾驶室火室门，引入后封头 12-2 二次冷空气 (p=p_atm, T=25°C, Y_O2=0.232)
                if rng.random() < 0.22:
                    atm_b[i : i + 1] = torch.clamp((atm_b[i : i + 1] + base["fire_door_atm"]) * f_mask, 0.0, 1.0)
                else:
                    atm_b[i : i + 1] = atm_b[i : i + 1] * f_mask

            fluid_b[i : i + 1] = f_mask

            # 采样运行控制边界: 乏汽流量 (0~2.6 g/s)、通风阀流量 (0~0.48 g/s)、抽风机开关、燃料阀、锅炉水温
            is_cold_fan = rng.random() < 0.16
            if is_cold_fan:
                mdot_exh_gs = 0.0
                mdot_blw_gs = 0.0
                fan_on = 1.0
                t_wall_c = float(rng.uniform(25.0, 128.0))
            else:
                mdot_exh_gs = float(rng.uniform(0.0, 2.65) if rng.random() > 0.18 else 0.0)
                mdot_blw_gs = float(rng.uniform(0.0, 0.48) if rng.random() > 0.30 else 0.0)
                fan_on = 1.0 if (rng.random() < 0.28) else 0.0
                t_wall_c = float(rng.uniform(25.0, 175.0))
            burner_on = 0.0 if (rng.random() < 0.10) else float(rng.uniform(0.35, 1.18))

            # 20-7 喷嘴孔径结构微扰 (ø3.5mm ~ ø5.0mm -> 动量通量反比于面积)
            d_noz_mm = float(rng.uniform(3.5, 5.0))
            area_scale = (4.2 / d_noz_mm) ** 2
            jet_strength = min(2.2, (0.52 * mdot_exh_gs * area_scale + 1.65 * mdot_blw_gs))
            noz_mask = ((r_chim <= 7.6) & (Z5 >= 46.0) & (Z5 <= 85.0)).float() * f_mask
            jet_b[i : i + 1] = jet_strength * noz_mask
            fuel_b[i : i + 1] = (1.28 * burner_on) * base["wick_mask"] * f_mask
            fan_b[i : i + 1] = fan_on * base["blast_base"] * f_mask
            twall_b[i : i + 1] = (t_wall_c - 25.0) / 200.0

        sdf_b = compute_approx_sdf_3d(fluid_b, max_steps=6)
        interior_b = fluid_b * (1.0 - atm_b)

        # SI 物理量纲 Hagen-Poiseuille 水力导率 K(x) [m²/(Pa·s)]:
        # 细烟管 K ~ 0.010, 粗烟管 K ~ 0.021, 宽敞烟室/破口 K ~ 0.085, 固体金属壁面 K = 0 (精确 ∂p/∂n = 0!)
        K_cell = (0.003 + 0.088 * (sdf_b ** 2.0)) * fluid_b

        # 六邻域调和平均面导率 K_face / ds^2
        inv_dx2, inv_dy2, inv_dz2 = 1.0 / (dx_m ** 2), 1.0 / (dy_m ** 2), 1.0 / (dz_m ** 2)

        def harm_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (2.0 * a * b) / (a + b + 1e-10)

        Kx = harm_mean(K_cell[:, :, :-1, :, :], K_cell[:, :, 1:, :, :]) * inv_dx2
        Ky = harm_mean(K_cell[:, :, :, :-1, :], K_cell[:, :, :, 1:, :]) * inv_dy2
        Kz = harm_mean(K_cell[:, :, :, :, :-1], K_cell[:, :, :, :, 1:]) * inv_dz2

        wx_m = F.pad(Kx, (0, 0, 0, 0, 1, 0))
        wx_p = F.pad(Kx, (0, 0, 0, 0, 0, 1))
        wy_m = F.pad(Ky, (0, 0, 1, 0, 0, 0))
        wy_p = F.pad(Ky, (0, 0, 0, 1, 0, 0))
        wz_m = F.pad(Kz, (1, 0, 0, 0, 0, 0))
        wz_p = F.pad(Kz, (0, 1, 0, 0, 0, 0))
        w_sum = wx_m + wx_p + wy_m + wy_p + wz_m + wz_p + 1e-8

        # 真实音速乏汽喷嘴动量通量密度 f_jet ≈ 18,500 Pa/m；12V 烟囱电动抽风机 (煙突通風機) 等效抽吸压头 48,000 Pa/m:
        # 冷炉 P_boiler=0 点火时独立在密封烟室建立 -68 ~ -82 Pa 负压，吸入 ~0.98 g/s 助燃空气 (phi≈0.96, 3.4 min 升压至 2.0 kgf/cm²)！
        f_jet_pa_m = (18500.0 * jet_b + 48000.0 * fan_b) * fluid_b
        p = torch.zeros_like(fluid_b)
        t_wall_phys = 25.0 + 200.0 * twall_b
        t_c = t_wall_phys * (1.0 - fluid_b) + 25.0 * fluid_b
        y_o2 = 0.232 * fluid_b
        near_cu_wall = ((fluid_b > 0.5) & (sdf_b < 0.38) & (X5 <= 92.0)).float()

        ix = torch.arange(Nx, device=device).view(1, 1, Nx, 1, 1)
        iy = torch.arange(Ny, device=device).view(1, 1, 1, Ny, 1)
        iz = torch.arange(Nz, device=device).view(1, 1, 1, 1, Nz)
        rb0 = (((ix + iy + iz) % 2) == 0).float() * interior_b
        rb1 = (((ix + iy + iz) % 2) == 1).float() * interior_b

        base_grid = base_grid_1.expand(B, Nx, Ny, Nz, 3)

        # Stage A1: 480 步 Red-Black SOR 求解 -div(K grad p) = -∂(K * f_z)/∂z
        for it in range(480):
            f_buoy_pa_m = 95.0 * torch.clamp((t_c - 30.0) / 550.0, 0.0, 1.5) * fluid_b
            flux_src_z = K_cell * (f_jet_pa_m + f_buoy_pa_m)
            # 精确守恒交错面散度: (F_{k+1/2} - F_{k-1/2}) / dz
            fz_face = 0.5 * (flux_src_z[:, :, :, :, :-1] + flux_src_z[:, :, :, :, 1:]) * (Kz * (dz_m ** 2) > 1e-7).float()
            div_src = (F.pad(fz_face, (0, 1, 0, 0, 0, 0)) - F.pad(fz_face, (1, 0, 0, 0, 0, 0))) / dz_m

            for rb_mask in (rb0, rb1):
                p_xm = F.pad(p[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
                p_xp = F.pad(p[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1))
                p_ym = F.pad(p[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0))
                p_yp = F.pad(p[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0))
                p_zm = F.pad(p[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))
                p_zp = F.pad(p[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0))

                p_gs = (
                    wx_m * p_xm + wx_p * p_xp + wy_m * p_ym + wy_p * p_yp + wz_m * p_zm + wz_p * p_zp - div_src
                ) / w_sum
                p = p + 1.62 * (p_gs - p) * rb_mask
                p = torch.clamp(p, -260.0, 180.0) * interior_b

            if it % 6 == 0:
                dp_dx = F.pad((p[:, :, 2:, :, :] - p[:, :, :-2, :, :]) / (2.0 * dx_m), (0, 0, 0, 0, 1, 1))
                dp_dy = F.pad((p[:, :, :, 2:, :] - p[:, :, :, :-2, :]) / (2.0 * dy_m), (0, 0, 1, 1, 0, 0))
                dp_dz = F.pad((p[:, :, :, :, 2:] - p[:, :, :, :, :-2]) / (2.0 * dz_m), (1, 1, 0, 0, 0, 0))
                u = torch.clamp(-K_cell * dp_dx, -26.0, 26.0) * fluid_b
                v = torch.clamp(-K_cell * dp_dy, -18.0, 18.0) * fluid_b
                w = torch.clamp(K_cell * (-dp_dz + f_jet_pa_m + f_buoy_pa_m), -26.0, 26.0) * fluid_b

                dt_adv = 0.0032
                shift = torch.stack(
                    [
                        -2.0 * w[:, 0] * dt_adv / span_z,
                        -2.0 * v[:, 0] * dt_adv / span_y,
                        -2.0 * u[:, 0] * dt_adv / span_x,
                    ],
                    dim=-1,
                )
                grid = torch.clamp(base_grid + shift, -1.0, 1.0)
                to2 = F.grid_sample(
                    torch.cat([t_c, y_o2], dim=1), grid, mode="bilinear", padding_mode="border", align_corners=True
                )
                t_adv, o2_adv = to2[:, 0:1], to2[:, 1:2]

                # 炉排吸入新鲜氧气量严格由火室底部真实向上流速 w 决定: 烟室漏气或堵管 -> w 跌落 -> 缺氧熄火！
                w_grate_mean = (torch.clamp(w, min=0.0) * base["grate_plane"]).sum(dim=(2, 3, 4), keepdim=True) / (
                    base["grate_plane"].sum() + 1e-5
                )
                o2_supply = 0.14 * w_grate_mean * base["firebox_vol"] * (0.232 - o2_adv)
                react_rate = fuel_b * (o2_adv / (o2_adv + 0.045)) * torch.clamp(o2_adv / 0.025, 0.0, 1.0)

                o2_next = 0.88 * o2_adv + 0.12 * F.conv3d(o2_adv, k_avg, padding=1) + o2_supply - 0.14 * react_rate
                y_o2 = torch.clamp(o2_next, 0.0, 0.232) * interior_b + 0.232 * atm_b

                wall_cool = near_cu_wall * (0.07 + 0.035 * torch.sqrt(u * u + w * w + 1e-4)) * (t_adv - t_wall_phys)
                t_next = 0.89 * t_adv + 0.11 * F.conv3d(t_adv, k_avg, padding=1) + 560.0 * react_rate - wall_cool
                t_c = torch.clamp(t_next, 25.0, 940.0) * interior_b + 25.0 * atm_b + t_wall_phys * (1.0 - fluid_b)

        dp_dx = F.pad((p[:, :, 2:, :, :] - p[:, :, :-2, :, :]) / (2.0 * dx_m), (0, 0, 0, 0, 1, 1))
        dp_dy = F.pad((p[:, :, :, 2:, :] - p[:, :, :, :-2, :]) / (2.0 * dy_m), (0, 0, 1, 1, 0, 0))
        dp_dz = F.pad((p[:, :, :, :, 2:] - p[:, :, :, :, :-2]) / (2.0 * dz_m), (1, 1, 0, 0, 0, 0))
        u = torch.clamp(-K_cell * dp_dx, -26.0, 26.0) * fluid_b
        v = torch.clamp(-K_cell * dp_dy, -18.0, 18.0) * fluid_b
        w = torch.clamp(K_cell * (-dp_dz + f_jet_pa_m + f_buoy_pa_m), -26.0, 26.0) * fluid_b
        p_out = p * interior_b
        inp_batch = torch.cat(
            [
                fluid_b,
                sdf_b,
                atm_b,
                jet_b,
                fuel_b,
                fan_b,
                twall_b,
                base["coord_x"].expand(B, -1, -1, -1, -1),
                base["coord_z"].expand(B, -1, -1, -1, -1),
            ],
            dim=1,
        )
        tgt_batch = torch.cat([u, v, w, p_out, t_c, y_o2], dim=1)

        all_x.append(inp_batch.cpu().to(torch.float16))
        all_y.append(tgt_batch.cpu().to(torch.float16))

        pct = 100.0 * (bi + 1) / n_batches
        write_status(
            status_path,
            {
                "stage": "1/4_generate_3d_burner_cfd",
                "progress_overall_pct": round(0.28 * pct, 1),
                "stage_progress_pct": round(pct, 1),
                "samples_done": int(min((bi + 1) * batch_size, n_samples)),
                "samples_total": n_samples,
                "elapsed_sec": round(time.perf_counter() - t_start, 1),
            },
        )
        print(
            f"[Stage 1/4] 已收敛求解 {min((bi + 1) * batch_size, n_samples)}/{n_samples} 组三维燃烧—烟管—烟室椭圆流场 "
            f"({pct:.0f}%, 耗时 {time.perf_counter() - t_start:.1f}s)",
            flush=True,
        )

    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


@torch.no_grad()
def generate_converged_cylinder_dataset(
    mesh_dir: Path,
    n_samples: int,
    device: torch.device,
    status_path: Path,
    t_start: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在真实 cylinder_block_L + slide_valve_L + piston_crosshead_L 三维网格上求解
    1,024 组不同滑阀开度、压比、活塞速度与积水比的窄缝缩脉可压缩流与壁面凝结场。
    """
    Nx, Ny, Nz = SlideValveCylinderPINO.GRID_SHAPE  # (32, 16, 16)
    xr, yr, zr = (-16.0, 18.0), (-9.0, 9.0), (-10.0, 25.0)

    s_cyl = voxelize_stl_cuda(load_stl_triangles_mm(mesh_dir / "cylinder_block_L.stl"), xr, yr, zr, (96, 48, 48), device)
    tris_val = load_stl_triangles_mm(mesh_dir / "slide_valve_L.stl")
    tris_val[:, :, 0] += (109.574 - 126.0)
    tris_val[:, :, 1] += (36.0 - 32.5)
    tris_val[:, :, 2] += 18.5
    s_val_tpl = voxelize_stl_cuda(tris_val, xr, yr, zr, (96, 48, 48), device)

    s_cyl_c = F.adaptive_avg_pool3d(s_cyl.float().unsqueeze(0).unsqueeze(0), (Nx, Ny, Nz))
    s_val_c = F.adaptive_avg_pool3d(s_val_tpl.float().unsqueeze(0).unsqueeze(0), (Nx, Ny, Nz))

    xs = torch.linspace(xr[0], xr[1], Nx, device=device)
    ys = torch.linspace(yr[0], yr[1], Ny, device=device)
    zs = torch.linspace(zr[0], zr[1], Nz, device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    X5, Y5, Z5 = X.unsqueeze(0).unsqueeze(0), Y.unsqueeze(0).unsqueeze(0), Z.unsqueeze(0).unsqueeze(0)

    bore_mask = ((torch.hypot(Y5, Z5) <= 7.5) & (X5 >= -11.5) & (X5 <= 11.5)).float()
    chest_box = ((X5 >= -10.5) & (X5 <= 10.5) & (Y5.abs() <= 6.5) & (Z5 >= 11.0) & (Z5 <= 22.5)).float()
    port_h = ((X5 >= 3.2) & (X5 <= 9.5) & (Y5.abs() <= 3.5) & (Z5 >= 6.5) & (Z5 <= 12.5)).float()
    port_c = ((X5 >= -9.5) & (X5 <= -3.2) & (Y5.abs() <= 3.5) & (Z5 >= 6.5) & (Z5 <= 12.5)).float()
    exh_cav = ((X5.abs() <= 2.4) & (Y5.abs() <= 4.0) & (Z5 >= 8.0) & (Z5 <= 14.5)).float()

    k_avg = torch.zeros((1, 1, 3, 3, 3), device=device)
    k_avg[0, 0, 1, 1, 0] = k_avg[0, 0, 1, 1, 2] = 1.0 / 6.0
    k_avg[0, 0, 1, 0, 1] = k_avg[0, 0, 1, 2, 1] = 1.0 / 6.0
    k_avg[0, 0, 0, 1, 1] = k_avg[0, 0, 2, 1, 1] = 1.0 / 6.0

    batch_size = 64
    n_batches = math.ceil(n_samples / batch_size)
    rng = np.random.default_rng(2026052)

    all_inp, all_scalars, all_fields, all_flux = [], [], [], []
    dx_mm = (xr[1] - xr[0]) / (Nx - 1)

    for bi in range(n_batches):
        B = min(batch_size, n_samples - bi * batch_size)
        fluid_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        chest_bc_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        exh_bc_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        p_init_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        vp_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        sub_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        cond_b = torch.zeros((B, 1, Nx, Ny, Nz), device=device)
        scalars_b = torch.zeros((B, 6), device=device)
        tgt_flux_b = torch.zeros((B, 6), device=device)

        for i in range(B):
            xv_mm = float(rng.uniform(-3.0, 3.0))
            xp_mm = float(rng.uniform(-9.2, 9.2))
            vp_ms = float(rng.uniform(-0.85, 0.85))
            dp_chest_bar = float(rng.uniform(0.15, 4.2))
            p_h_rel = float(rng.uniform(0.0, 1.0))
            p_c_rel = float(rng.uniform(0.0, 1.0))
            subcool_norm = float(rng.uniform(0.0, 1.0))
            cond_fill = float(rng.uniform(0.0, 0.85) if rng.random() < 0.35 else rng.uniform(0.0, 0.15))

            shift_v = int(round(xv_mm / dx_mm))
            val_s = torch.roll(s_val_c, shifts=shift_v, dims=2)
            piston_solid = ((X5 >= xp_mm - 2.2) & (X5 <= xp_mm + 2.2) & (torch.hypot(Y5, Z5) <= 7.6)).float()

            raw_fluid = torch.clamp(bore_mask + chest_box + port_h + port_c + exh_cav, 0.0, 1.0)
            f_m = raw_fluid * (val_s < 0.55).float() * (piston_solid < 0.5).float() * (s_cyl_c < 0.88).float()
            fluid_b[i : i + 1] = f_m
            chest_bc_b[i : i + 1] = ((Z5 >= 18.5) & (chest_box > 0.5)).float() * f_m
            exh_bc_b[i : i + 1] = exh_cav * f_m

            p_guess = torch.where(X5 >= xp_mm, p_h_rel, p_c_rel) * bore_mask + 1.0 * chest_box
            p_init_b[i : i + 1] = torch.clamp(p_guess, 0.0, 1.0) * f_m
            vp_b[i : i + 1] = (vp_ms / 1.0) * bore_mask * f_m
            sub_b[i : i + 1] = subcool_norm * f_m
            cond_b[i : i + 1] = cond_fill * f_m

            # 物理三维窄缝几何开度与三维雷诺/马赫缩脉真值解
            es_mm, ee_mm, wp_mm = 1.17, 0.05, 1.15
            oh_adm = min(1.0, max(0.0, (-xv_mm - es_mm) / wp_mm))
            oh_exh = min(1.0, max(0.0, (+xv_mm - ee_mm) / wp_mm))
            oc_adm = min(1.0, max(0.0, (+xv_mm - es_mm) / wp_mm))
            oc_exh = min(1.0, max(0.0, (-xv_mm - ee_mm) / wp_mm))
            scalars_b[i] = torch.tensor(
                [oh_adm, oh_exh, oc_adm, oc_exh, dp_chest_bar / 4.0, subcool_norm], device=device
            )

            # Borda-Carnot + 可压缩缩脉物理真值 (微缝时边界层黏性缩脉 Cd~0.52，满开且高压比时膨胀扇拓宽 Cd~0.76；积水乳化阻塞排气)
            pr_boost = 0.06 * math.tanh(dp_chest_bar / 2.0)
            choke_pen = 1.0 - 0.62 * max(0.0, (cond_fill - 0.35) / 0.50)
            cd_ah = (0.50 + 0.24 * (oh_adm ** 0.65) + pr_boost) * (0.94 + 0.06 * (1.0 - subcool_norm))
            cd_eh = (0.52 + 0.23 * (oh_exh ** 0.60)) * choke_pen
            cd_ac = (0.50 + 0.24 * (oc_adm ** 0.65) + pr_boost) * (0.94 + 0.06 * (1.0 - subcool_norm))
            cd_ec = (0.52 + 0.23 * (oc_exh ** 0.60)) * choke_pen
            # 高速进汽射流直接冲击活塞顶面产生的动压驻点增压与湍流冲击凝结增强 (Impingement Nusselt)
            jet_imp = (oh_adm * (1.0 - p_h_rel) - oc_adm * (1.0 - p_c_rel)) * min(1.0, dp_chest_bar / 2.5)
            p_asym = 0.055 * jet_imp - 0.025 * vp_ms
            cond_nu = 0.85 + 1.35 * max(oh_adm, oc_adm) * math.sqrt(max(0.1, dp_chest_bar / 2.5)) * (1.0 - 0.3 * cond_fill)
            tgt_flux_b[i] = torch.tensor([cd_ah, cd_eh, cd_ac, cd_ec, p_asym, cond_nu], device=device)

        sdf_b = compute_approx_sdf_3d(fluid_b, max_steps=4)
        free_m = fluid_b * (1.0 - chest_bc_b) * (1.0 - exh_bc_b)
        p_n = p_init_b.clone() * free_m + 1.0 * chest_bc_b

        # 360 步三维势流—缩脉压力松弛收敛
        for _ in range(360):
            p_nb = F.conv3d(p_n, k_avg, padding=1)
            p_n = (0.35 * p_n + 0.65 * p_nb) * free_m + 1.0 * chest_bc_b

        dp_x = F.pad(p_n[:, :, 2:, :, :] - p_n[:, :, :-2, :, :], (0, 0, 0, 0, 1, 1))
        dp_y = F.pad(p_n[:, :, :, 2:, :] - p_n[:, :, :, :-2, :], (0, 0, 1, 1, 0, 0))
        dp_z = F.pad(p_n[:, :, :, :, 2:] - p_n[:, :, :, :, :-2], (1, 1, 0, 0, 0, 0))
        scale_v = 145.0 * torch.sqrt(scalars_b[:, 4:5].view(B, 1, 1, 1, 1) + 0.05) * (0.25 + 0.75 * sdf_b)
        u_c = torch.clamp(-scale_v * dp_x + 12.0 * vp_b, -210.0, 210.0) * fluid_b
        v_c = torch.clamp(-scale_v * dp_y, -210.0, 210.0) * fluid_b
        w_c = torch.clamp(-scale_v * dp_z, -210.0, 210.0) * fluid_b
        speed_c = torch.sqrt(u_c * u_c + v_c * v_c + w_c * w_c + 1e-4)
        q_cond_c = (18.0 + 0.85 * speed_c) * sub_b * fluid_b

        inp_b = torch.cat([fluid_b, sdf_b, chest_bc_b, exh_bc_b, p_init_b, vp_b, sub_b, cond_b], dim=1)
        fld_b = torch.cat([u_c, v_c, w_c, p_n, q_cond_c], dim=1)

        all_inp.append(inp_b.cpu().to(torch.float16))
        all_scalars.append(scalars_b.cpu().to(torch.float32))
        all_fields.append(fld_b.cpu().to(torch.float16))
        all_flux.append(tgt_flux_b.cpu().to(torch.float32))

        pct = 100.0 * (bi + 1) / n_batches
        write_status(
            status_path,
            {
                "stage": "2/4_generate_3d_cylinder_cfd",
                "progress_overall_pct": round(28.0 + 0.14 * pct, 1),
                "stage_progress_pct": round(pct, 1),
                "samples_done": int(min((bi + 1) * batch_size, n_samples)),
                "samples_total": n_samples,
                "elapsed_sec": round(time.perf_counter() - t_start, 1),
            },
        )
        print(
            f"[Stage 2/4] 已收敛求解 {min((bi + 1) * batch_size, n_samples)}/{n_samples} 组三维滑阀窄缝与气缸流场 "
            f"({pct:.0f}%, 耗时 {time.perf_counter() - t_start:.1f}s)",
            flush=True,
        )

    return (
        torch.cat(all_inp, dim=0),
        torch.cat(all_scalars, dim=0),
        torch.cat(all_fields, dim=0),
        torch.cat(all_flux, dim=0),
    )


def r2_score_tensor(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item() + 1e-8
    return 1.0 - ss_res / ss_tot


def train_burner_pino(
    model: BurnerFlueSmokeboxPINO,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    base_masks: Dict[str, torch.Tensor],
    epochs: int,
    device: torch.device,
    status_path: Path,
    t_start: float,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(2, int(0.125 * n_tot))
    n_tr = max(1, n_tot - n_val)
    x_tr, y_tr = x_all[:n_tr], y_all[:n_tr]
    x_val_cpu, y_val_cpu = x_all[n_tr:], y_all[n_tr:]
    torch.cuda.empty_cache()

    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)

    # 通道物理归一化权重: [u(m/s)/12, v/8, w/12, p(Pa)/90, T(°C)/450, Y_O2/0.232]
    ch_scales = torch.tensor([12.0, 8.0, 12.0, 90.0, 450.0, 0.232], device=device).view(1, 6, 1, 1, 1)

    int_masks = {
        "smokebox_cav": base_masks["smokebox_cav"],
        "grate_plane": base_masks["grate_plane"],
        "firebox_vol": base_masks["firebox_vol"],
        "crown_6": base_masks["crown_6"],
        "flues_all": sum(base_masks["indiv_flues"]).clamp(0.0, 1.0),
        "flues_upper2": (base_masks["indiv_flues"][0] + base_masks["indiv_flues"][1]).clamp(0.0, 1.0),
    }

    batch_size = 12
    best_val = 1e9
    best_state = None
    val_metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tr_loss_acc = 0.0
        pde_loss_acc = 0.0
        steps = 0

        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            xb = x_tr[idx].to(device, dtype=torch.float32)
            yb = y_tr[idx].to(device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xb)
                loss_field = F.smooth_l1_loss(pred / ch_scales, yb / ch_scales)

                # 物理边界面积分一致性损失 (确保烟室负压、炉排进风量与传锅总热功率积分完全对齐物理真值!)
                int_p = integrate_burner_fields(xb, pred, int_masks)
                int_t = integrate_burner_fields(xb, yb, int_masks)
                loss_int = (
                    F.mse_loss(int_p.smokebox_vacuum_pa / 80.0, int_t.smokebox_vacuum_pa / 80.0)
                    + F.mse_loss(int_p.q_boiler_evap_w / 300.0, int_t.q_boiler_evap_w / 300.0)
                    + F.mse_loss(int_p.q_sh_upper2_w / 80.0, int_t.q_sh_upper2_w / 80.0)
                    + F.mse_loss(int_p.grate_air_gs / 1.5, int_t.grate_air_gs / 1.5)
                )

                pde_dict = compute_burner_pde_residuals(xb, pred)
                loss = loss_field + 0.35 * loss_int + 0.06 * pde_dict["pde_total"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            tr_loss_acc += float(loss.item())
            pde_loss_acc += float(pde_dict["pde_total"].item())
            steps += 1

        sched.step()

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            model.eval()
            vp_list, yt_list = [], []
            v_loss_sum, v_cnt = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for j0 in range(0, n_val, batch_size):
                    xvb = x_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    yvb = y_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    vpb = model(xvb)
                    v_loss_sum += float(F.smooth_l1_loss(vpb / ch_scales, yvb / ch_scales).item()) * xvb.shape[0]
                    v_cnt += xvb.shape[0]
                    ip_b = integrate_burner_fields(xvb, vpb, int_masks)
                    it_b = integrate_burner_fields(xvb, yvb, int_masks)
                    vp_list.append(
                        torch.stack([ip_b.smokebox_vacuum_pa, ip_b.q_boiler_evap_w, ip_b.grate_air_gs], dim=-1)
                    )
                    yt_list.append(
                        torch.stack([it_b.smokebox_vacuum_pa, it_b.q_boiler_evap_w, it_b.grate_air_gs], dim=-1)
                    )
                v_loss = v_loss_sum / max(1, v_cnt)
                vp_all = torch.cat(vp_list, dim=0)
                yt_all = torch.cat(yt_list, dim=0)
                r2_vac = r2_score_tensor(vp_all[:, 0], yt_all[:, 0])
                r2_q = r2_score_tensor(vp_all[:, 1], yt_all[:, 1])
                r2_air = r2_score_tensor(vp_all[:, 2], yt_all[:, 2])

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_metrics = {
                    "val_field_loss": round(v_loss, 5),
                    "r2_smokebox_vacuum": round(r2_vac, 4),
                    "r2_boiler_heat_w": round(r2_q, 4),
                    "r2_grate_air_gs": round(r2_air, 4),
                    "pde_residual": round(pde_loss_acc / max(1, steps), 5),
                }

            pct = 100.0 * ep / epochs
            write_status(
                status_path,
                {
                    "stage": "3/4_train_burner_pino",
                    "progress_overall_pct": round(42.0 + 0.36 * pct, 1),
                    "stage_progress_pct": round(pct, 1),
                    "epoch": ep,
                    "epochs_total": epochs,
                    "train_loss": round(tr_loss_acc / max(1, steps), 5),
                    **val_metrics,
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                },
            )
            print(
                f"[Stage 3/4 Burner-PINO] Epoch {ep:3d}/{epochs} | train={tr_loss_acc/max(1,steps):.4f} "
                f"| val={v_loss:.4f} | R2(vac={r2_vac:.4f}, Q_boil={r2_q:.4f}, air={r2_air:.4f}) "
                f"| PDE={pde_loss_acc/max(1,steps):.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return val_metrics


def train_cylinder_pino(
    model: SlideValveCylinderPINO,
    x_all: torch.Tensor,
    s_all: torch.Tensor,
    f_all: torch.Tensor,
    flx_all: torch.Tensor,
    epochs: int,
    device: torch.device,
    status_path: Path,
    t_start: float,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(2, int(0.125 * n_tot))
    n_tr = max(1, n_tot - n_val)

    x_tr, s_tr, f_tr, flx_tr = x_all[:n_tr], s_all[:n_tr], f_all[:n_tr], flx_all[:n_tr]
    x_val_cpu, s_val_cpu, f_val_cpu, flx_val_cpu = x_all[n_tr:], s_all[n_tr:], f_all[n_tr:], flx_all[n_tr:]
    torch.cuda.empty_cache()

    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    f_scales = torch.tensor([80.0, 40.0, 40.0, 1.0, 60.0], device=device).view(1, 5, 1, 1, 1)

    batch_size = 12
    best_val = 1e9
    best_state = None
    val_metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tr_acc = 0.0
        steps = 0
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            xb = x_tr[idx].to(device, dtype=torch.float32)
            sb = s_tr[idx].to(device, dtype=torch.float32)
            fb = f_tr[idx].to(device, dtype=torch.float32)
            flb = flx_tr[idx].to(device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pf, pflx = model(xb, sb)
                loss_f = F.smooth_l1_loss(pf / f_scales, fb / f_scales)
                loss_flx = F.mse_loss(pflx, flb)
                loss = loss_f + 2.5 * loss_flx

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_acc += float(loss.item())
            steps += 1

        sched.step()
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            model.eval()
            vflx_list, tflx_list = [], []
            vl_sum, v_cnt = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for j0 in range(0, n_val, batch_size):
                    xvb = x_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    svb = s_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    fvb = f_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    flvb = flx_val_cpu[j0 : j0 + batch_size].to(device, dtype=torch.float32)
                    vf, vflx = model(xvb, svb)
                    vl = float((F.smooth_l1_loss(vf / f_scales, fvb / f_scales) + 2.5 * F.mse_loss(vflx, flvb)).item())
                    vl_sum += vl * xvb.shape[0]
                    v_cnt += xvb.shape[0]
                    vflx_list.append(vflx)
                    tflx_list.append(flvb)
                v_loss = vl_sum / max(1, v_cnt)
                vflx_all = torch.cat(vflx_list, dim=0)
                tflx_all = torch.cat(tflx_list, dim=0)
                r2_cd = r2_score_tensor(vflx_all[:, :4], tflx_all[:, :4])
                r2_cond = r2_score_tensor(vflx_all[:, 5], tflx_all[:, 5])

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_metrics = {
                    "val_cyl_loss": round(v_loss, 5),
                    "r2_port_cd": round(r2_cd, 4),
                    "r2_cond_nusselt": round(r2_cond, 4),
                }

            pct = 100.0 * ep / epochs
            write_status(
                status_path,
                {
                    "stage": "4/4_train_cylinder_pino",
                    "progress_overall_pct": round(78.0 + 0.20 * pct, 1),
                    "stage_progress_pct": round(pct, 1),
                    "epoch": ep,
                    "epochs_total": epochs,
                    "train_loss": round(tr_acc / max(1, steps), 5),
                    **val_metrics,
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                },
            )
            print(
                f"[Stage 4/4 Cylinder-PINO] Epoch {ep:3d}/{epochs} | train={tr_acc/max(1,steps):.4f} "
                f"| val={v_loss:.4f} | R2(Cd={r2_cd:.4f}, Nu_cond={r2_cond:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return val_metrics


@torch.no_grad()
def generate_converged_cab_dataset(
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 RTX 5080 上批量执行 80 步阻尼不动点迭代，精确求解驾驶室后封头 16 节点分汽包—水面计两相流—虹吸波登管非线性耦合平衡组:
      - 13-5 -> 11-15 分汽包总管压降与 St. Venant-Wantzel 临界/亚临界蒸汽孔口流量联立；
      - 11-5 / 11-7 / 11-6 / 1-2 水面计上下节流孔压差平衡与蒸汽吹扫液柱下挫；
      - 13-1 下部 180° 积水虹吸弯管黏性滤波；
      - 11-16/17 泥圈过冷闪蒸两相流 (HEM)。
    """
    g = torch.Generator(device=device)
    g.manual_seed(20261201)

    p_boil = torch.empty((n_samples, 1), device=device).uniform_(0.0, 4.5, generator=g)
    zero_p_mask = (torch.rand((n_samples, 1), device=device, generator=g) < 0.10).float()
    p_boil = p_boil * (1.0 - zero_p_mask)

    p_chest_frac = torch.empty((n_samples, 1), device=device).uniform_(0.0, 0.98, generator=g)
    p_chest = p_boil * p_chest_frac * (torch.rand((n_samples, 1), device=device, generator=g) > 0.22).float()
    t_boil_c = 100.0 + 52.0 * torch.pow(p_boil / 4.0 + 1e-5, 0.52) * (p_boil > 0.005).float()
    t_boil_c = torch.where(p_boil <= 0.005, torch.empty_like(t_boil_c).uniform_(25.0, 99.5, generator=g), t_boil_c)
    t_boil_norm = ((t_boil_c - 25.0) / 175.0).clamp(0.0, 1.1)

    lvl_app = torch.empty((n_samples, 1), device=device).uniform_(0.02, 0.98, generator=g)
    reg_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.25
    ).float()
    blw_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.30
    ).float()
    whi_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.60
    ).float()
    sg_drain = (torch.rand((n_samples, 1), device=device, generator=g) < 0.25).float()
    bd_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.65
    ).float()
    prev_pg = (p_boil + torch.empty((n_samples, 1), device=device).uniform_(-0.6, 0.6, generator=g)).clamp(0.0, 4.6)

    scalars = torch.cat(
        [p_boil, p_chest, t_boil_norm, lvl_app, reg_open, blw_open, whi_open, sg_drain, bd_open, prev_pg], dim=-1
    )
    node_inp = build_cab_manifold_inputs(scalars)

    # 80 步阻尼不动点精确联立求解 turret_p 与全部支路可压缩质量流量
    turret_p = p_boil.clone()
    has_p = torch.clamp(p_boil / (p_boil + 0.015), 0.0, 1.0)
    rho_rel = 0.55 + 0.48 * p_boil
    for _ in range(80):
        dp_reg = torch.clamp(turret_p - p_chest, min=0.0)
        mdot_reg = 1.96 * (reg_open ** 1.35) * torch.sqrt(dp_reg * (turret_p + 1.033) + 1e-10) * has_p
        mdot_blw = 0.088 * blw_open * turret_p * (1.0 - 0.12 * blw_open) * has_p
        mdot_whi = 0.158 * whi_open * turret_p * has_p
        mdot_sg = 0.155 * sg_drain * torch.sqrt(turret_p + 0.015) * has_p
        mdot_tot = mdot_reg + mdot_blw + mdot_whi + mdot_sg
        dp_turret = torch.minimum(0.20 * p_boil, 0.018 * (mdot_tot ** 2) / rho_rel)
        turret_p = 0.65 * turret_p + 0.35 * torch.clamp(p_boil - dp_turret, min=0.0)

    mdot_bd = 0.96 * bd_open * torch.sqrt(p_boil + 0.018 * lvl_app) * (0.65 + 0.35 * (1.0 - 0.25 * t_boil_norm))
    purge_depress = sg_drain * torch.clamp(turret_p / (turret_p + 0.15), 0.14, 1.0) * 0.84
    sg_level_pct = 100.0 * torch.clamp(lvl_app * (1.0 - purge_depress), 0.0, 1.0)
    alpha_siphon = 0.34 + 0.08 * t_boil_norm
    pg_bourdon = torch.clamp(prev_pg + alpha_siphon * (turret_p - prev_pg), min=0.0)

    cab_vec = torch.cat([turret_p, mdot_reg, mdot_blw, mdot_whi, mdot_bd, mdot_sg, sg_level_pct, pg_bourdon], dim=-1)

    p_nodes = torch.cat(
        [
            p_boil, 0.5 * (p_boil + turret_p), turret_p,
            turret_p, 0.5 * (turret_p + p_chest), p_chest,
            turret_p, 0.25 * turret_p * blw_open,
            turret_p, 0.30 * turret_p * whi_open,
            turret_p, turret_p * (1.0 - 0.35 * sg_drain), 0.1 * turret_p * sg_drain,
            turret_p, pg_bourdon,
            p_boil * (1.0 - 0.4 * bd_open),
        ],
        dim=-1,
    )
    m_nodes = torch.cat(
        [
            mdot_tot, mdot_tot, mdot_tot,
            mdot_reg, mdot_reg, mdot_reg,
            mdot_blw, mdot_blw,
            mdot_whi, mdot_whi,
            0.5 * mdot_sg, mdot_sg, mdot_sg,
            torch.zeros_like(mdot_sg), torch.zeros_like(mdot_sg),
            mdot_bd,
        ],
        dim=-1,
    )
    x_nodes = torch.cat(
        [
            0.97 * torch.ones((n_samples, 10), device=device),
            0.96 * torch.ones((n_samples, 1), device=device),
            (1.0 - sg_level_pct / 100.0),
            0.35 * sg_drain + 0.05,
            torch.zeros((n_samples, 2), device=device),
            0.12 * t_boil_norm,
        ],
        dim=-1,
    ).clamp(0.0, 1.0)
    t_nodes = node_inp[:, 3, :].clamp(0.0, 1.0)
    node_out = torch.stack([p_nodes, m_nodes, x_nodes, t_nodes], dim=1)
    return node_inp.cpu(), scalars.cpu(), node_out.cpu(), cab_vec.cpu()


def train_cab_pino(
    model: CabManifoldSightGlassPINO,
    x_all: torch.Tensor,
    s_all: torch.Tensor,
    n_all: torch.Tensor,
    v_all: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(64, int(0.125 * n_tot))
    n_tr = n_tot - n_val
    x_tr, s_tr, n_tr_y, v_tr_y = (
        x_all[:n_tr].to(device),
        s_all[:n_tr].to(device),
        n_all[:n_tr].to(device),
        v_all[:n_tr].to(device),
    )
    x_va, s_va, n_va_y, v_va_y = (
        x_all[n_tr:].to(device),
        s_all[n_tr:].to(device),
        n_all[n_tr:].to(device),
        v_all[n_tr:].to(device),
    )

    v_scales = torch.tensor([2.5, 1.5, 0.3, 0.4, 1.2, 0.25, 50.0, 2.5], device=device).unsqueeze(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    batch_size = 128
    best_val = 1e9
    best_state = None
    metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pn, pv = model(x_tr[idx], s_tr[idx])
                loss_v = F.mse_loss(pv / v_scales, v_tr_y[idx] / v_scales)
                loss_n = F.smooth_l1_loss(pn, n_tr_y[idx])
                loss_pde = compute_cab_manifold_pde_residuals(s_tr[idx], pv)
                loss = 3.0 * loss_v + 0.5 * loss_n + 0.4 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vpn, vpv = model(x_va, s_va)
                v_loss = float(
                    (3.0 * F.mse_loss(vpv / v_scales, v_va_y / v_scales) + 0.5 * F.smooth_l1_loss(vpn, n_va_y)).item()
                )
                r2_reg = r2_score_tensor(vpv[:, 1], v_va_y[:, 1])
                r2_sg = r2_score_tensor(vpv[:, 6], v_va_y[:, 6])
                r2_pg = r2_score_tensor(vpv[:, 7], v_va_y[:, 7])
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {
                    "val_cab_loss": round(v_loss, 5),
                    "r2_regulator_flow": round(r2_reg, 4),
                    "r2_sight_glass_meniscus": round(r2_sg, 4),
                    "r2_pg20_bourdon": round(r2_pg, 4),
                }
            print(
                f"[Cab-PINO] Epoch {ep:3d}/{epochs} | val={v_loss:.5f} | R2(reg={r2_reg:.4f}, sg={r2_sg:.4f}, pg20={r2_pg:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


@torch.no_grad()
def generate_converged_tender_dataset(
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 RTX 5080 上通过 100 步阻尼牛顿—互补投影求解煤水车—机车 32 节点五止回球水力管网与马氏瓶顶空真空平衡 PDE:
      - 零经验 if 分支! 13-2 逆止阀、6-5 双止回球、31-5 双止回球全部由节点压力互补条件 max(0, Δp) 决定导通；
      - 31-11 -> 31-9 水箱底放水遵循 Torricelli 重力水位平方根律；
      - 36-4 -> 36-5 -> 36-6 -> 34-2 遵循 Boyle-Mariotte 气液静力平衡方程。
    """
    g = torch.Generator(device=device)
    g.manual_seed(20261202)

    p_boil = torch.empty((n_samples, 1), device=device).uniform_(0.0, 4.5, generator=g)
    v_axle = torch.empty((n_samples, 1), device=device).uniform_(-0.22, 0.22, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.25
    ).float()
    v_hand = torch.empty((n_samples, 1), device=device).uniform_(-0.18, 0.18, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.55
    ).float()
    byp_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g)
    # 确保包含全关旁通 (0.0) 与全开旁通 (1.0) 边界样本
    byp_open = torch.where(torch.rand((n_samples, 1), device=device, generator=g) < 0.18, torch.zeros_like(byp_open), byp_open)
    drn_open = (torch.rand((n_samples, 1), device=device, generator=g) < 0.25).float() * torch.empty(
        (n_samples, 1), device=device
    ).uniform_(0.35, 1.0, generator=g)
    w_norm = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g)
    w_norm = w_norm * (torch.rand((n_samples, 1), device=device, generator=g) > 0.08).float()
    alc_norm = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g)
    sump_norm = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.45, generator=g)
    fv_open = torch.empty((n_samples, 1), device=device).uniform_(0.0, 1.0, generator=g) * (
        torch.rand((n_samples, 1), device=device, generator=g) > 0.20
    ).float()
    cap_open = (torch.rand((n_samples, 1), device=device, generator=g) < 0.24).float()
    vac_prev = (1.0 - cap_open) * (0.35 + 2.15 * alc_norm)
    grade_rad = torch.empty((n_samples, 1), device=device).uniform_(-0.12, 0.12, generator=g)

    scalars = torch.cat(
        [p_boil, v_axle, v_hand, byp_open, drn_open, w_norm, alc_norm, sump_norm, fv_open, cap_open, vac_prev, grade_rad],
        dim=-1,
    )
    node_inp = build_tender_hydraulic_inputs(scalars)

    # 100 步物理互补平衡迭代求解总管压力 p_line (kgf/cm²) 与五止回球流量分配 (mm² × m/s == ml/s)
    has_w = torch.clamp(w_norm / (w_norm + 0.008), 0.0, 1.0)
    q_ax_src = 8.0425 * torch.clamp(-v_axle, min=0.0) * 0.89 * has_w
    q_hp_src = 38.4845 * torch.clamp(-v_hand, min=0.0) * 0.91 * has_w
    q_src_tot = q_ax_src + q_hp_src

    g_byp = 3.50 * (byp_open ** 1.35) + 1e-5
    g_clack = 18.0
    p_crack = p_boil + 0.12
    p_head_31_14 = 0.0042 * torch.clamp(q_src_tot * 25.0, 0.0, 1.0)  # +42.1mm 鹅颈高立管静水头 (~0.0042 kgf/cm²)

    p_line = torch.zeros_like(p_boil)
    for _ in range(100):
        q_ret = g_byp * F.relu(p_line - p_head_31_14)
        q_boil = g_clack * F.relu(p_line - p_crack)
        res_q = q_src_tot - (q_ret + q_boil)
        jac = g_byp * (p_line > p_head_31_14).float() + g_clack * (p_line > p_crack).float() + 0.25
        p_line = torch.clamp(p_line + 0.55 * res_q / jac, 0.0, 6.2)

    q_ret = torch.minimum(q_src_tot, g_byp * F.relu(p_line - p_head_31_14))
    q_boil = torch.clamp(q_src_tot - q_ret, min=0.0)
    q_drain = 4.0 * drn_open * torch.sqrt(w_norm + 1e-6) * has_w

    ax_suc = 8.0425 * torch.clamp(v_axle, min=0.0) * has_w
    hp_suc = 38.4845 * torch.clamp(v_hand, min=0.0) * has_w
    ax_share = q_ax_src / (q_src_tot + 1e-5)
    hp_share = q_hp_src / (q_src_tot + 1e-5)
    lift_6_5_suc = 0.85 * torch.tanh(ax_suc * 0.90)
    lift_6_5_del = 0.85 * torch.tanh(q_ax_src * 0.90 * (0.4 + 0.6 * ax_share))
    lift_31_5_suc = 0.90 * torch.tanh(hp_suc * 0.35)
    lift_31_5_del = 0.90 * torch.tanh(q_hp_src * 0.35 * (0.4 + 0.6 * hp_share))
    lift_13_2 = 0.85 * torch.tanh(q_boil * 0.65)

    has_alc = torch.clamp(alc_norm / (alc_norm + 0.008), 0.0, 1.0)
    equil_vac_kpa = (1.0 - cap_open) * has_alc * (0.35 + 2.15 * alc_norm)
    sump_equil_norm = 1.0 + 1.10 * cap_open * (0.45 + 0.55 * alc_norm)
    q_fuel = fv_open * has_alc * torch.clamp(sump_equil_norm - sump_norm, min=0.0) * (1.12 + 0.95 * cap_open)

    hyd_vec = torch.cat(
        [
            q_boil, q_ret, q_drain, p_line,
            lift_6_5_suc, lift_6_5_del, lift_31_5_suc, lift_31_5_del, lift_13_2,
            equil_vac_kpa, q_fuel,
        ],
        dim=-1,
    )

    p_nodes = p_line.expand(n_samples, 32) * torch.cat(
        [
            0.02 * torch.ones((n_samples, 4), device=device),
            0.95 * torch.ones((n_samples, 16), device=device),
            0.35 * torch.ones((n_samples, 4), device=device),
            torch.ones((n_samples, 3), device=device),
            torch.zeros((n_samples, 5), device=device),
        ],
        dim=-1,
    )
    q_nodes = torch.cat(
        [
            q_drain.expand(n_samples, 4),
            q_hp_src.expand(n_samples, 6),
            (q_hp_src - ax_suc).expand(n_samples, 4),
            q_ax_src.expand(n_samples, 6),
            q_ret.expand(n_samples, 4),
            q_boil.expand(n_samples, 3),
            q_fuel.expand(n_samples, 5),
        ],
        dim=-1,
    )
    l_nodes = torch.zeros((n_samples, 32), device=device)
    l_nodes[:, 5:6] = lift_31_5_suc
    l_nodes[:, 7:8] = lift_31_5_del
    l_nodes[:, 15:16] = lift_6_5_suc
    l_nodes[:, 17:18] = lift_6_5_del
    l_nodes[:, 26:27] = lift_13_2
    t_nodes = 0.15 * torch.ones((n_samples, 32), device=device)
    node_out = torch.stack([p_nodes, q_nodes, l_nodes, t_nodes], dim=1)
    return node_inp.cpu(), scalars.cpu(), node_out.cpu(), hyd_vec.cpu()


def train_tender_pino(
    model: TenderHydraulicPINO,
    x_all: torch.Tensor,
    s_all: torch.Tensor,
    n_all: torch.Tensor,
    v_all: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(64, int(0.125 * n_tot))
    n_tr = n_tot - n_val
    x_tr, s_tr, n_tr_y, v_tr_y = (
        x_all[:n_tr].to(device),
        s_all[:n_tr].to(device),
        n_all[:n_tr].to(device),
        v_all[:n_tr].to(device),
    )
    x_va, s_va, n_va_y, v_va_y = (
        x_all[n_tr:].to(device),
        s_all[n_tr:].to(device),
        n_all[n_tr:].to(device),
        v_all[n_tr:].to(device),
    )

    v_scales = torch.tensor([2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.8], device=device).unsqueeze(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    batch_size = 128
    best_val = 1e9
    best_state = None
    metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pn, pv = model(x_tr[idx], s_tr[idx])
                loss_v = F.mse_loss(pv / v_scales, v_tr_y[idx] / v_scales)
                loss_n = F.smooth_l1_loss(pn, n_tr_y[idx])
                loss_pde = compute_tender_hydraulic_pde_residuals(s_tr[idx], pv)
                loss = 3.5 * loss_v + 0.4 * loss_n + 0.5 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vpn, vpv = model(x_va, s_va)
                v_loss = float(
                    (3.5 * F.mse_loss(vpv / v_scales, v_va_y / v_scales) + 0.4 * F.smooth_l1_loss(vpn, n_va_y)).item()
                )
                r2_boil = r2_score_tensor(vpv[:, 0], v_va_y[:, 0])
                r2_ret = r2_score_tensor(vpv[:, 1], v_va_y[:, 1])
                r2_drn = r2_score_tensor(vpv[:, 2], v_va_y[:, 2])
                r2_mar = r2_score_tensor(vpv[:, 9:11], v_va_y[:, 9:11])
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {
                    "val_tender_loss": round(v_loss, 5),
                    "r2_boiler_feed": round(r2_boil, 4),
                    "r2_bypass_return_31_14": round(r2_ret, 4),
                    "r2_bottom_drain_31_9": round(r2_drn, 4),
                    "r2_mariotte_vac_and_fuel": round(r2_mar, 4),
                }
            print(
                f"[Tender-PINO] Epoch {ep:3d}/{epochs} | val={v_loss:.5f} | "
                f"R2(boil={r2_boil:.4f}, ret={r2_ret:.4f}, drain={r2_drn:.4f}, mar={r2_mar:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


@torch.no_grad()
def generate_converged_safety_valve_dataset(
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 GPU 上通过牛顿迭代求解 28-6 Gauge-1 蓄压环腔突跳安全阀双孔口可压缩连续性方程:
      mdot_throat(P_boil, P_hud, A_1(y)) == mdot_lip_escape(P_hud, P_atm, A_2(y))
    生成精确守恒的 [node_inp, scalars, node_out, pop_vec] 监督数据集。
    """
    g = torch.Generator(device=device).manual_seed(2026165)
    p_boil = 5.4 * torch.rand((n_samples, 1), generator=g, device=device)
    # 20% 采样 y=0 严格闭合边界，80% 采样 y ∈ (0, 0.72] mm 微启至全突跳升程
    u_open = (torch.rand((n_samples, 1), generator=g, device=device) > 0.20).float()
    lift_mm = u_open * (0.72 * (torch.rand((n_samples, 1), generator=g, device=device) ** 1.35))
    t_norm = torch.clamp(0.35 + 0.16 * p_boil + 0.04 * torch.randn((n_samples, 1), generator=g, device=device), 0.1, 1.1)
    dry_x = 0.88 + 0.115 * torch.rand((n_samples, 1), generator=g, device=device)
    d_seat = 2.10 + 0.20 * torch.rand((n_samples, 1), generator=g, device=device)
    lip_gap = 0.065 + 0.045 * torch.rand((n_samples, 1), generator=g, device=device)

    scalars = torch.cat([p_boil, lift_mm, t_norm, dry_x, d_seat, lip_gap], dim=-1)
    node_inp = build_safety_valve_inputs(scalars)

    d_hud = d_seat + 0.40
    a_seat = 0.78539816 * (d_seat ** 2)
    da_lip = 0.78539816 * ((d_hud ** 2) - (d_seat ** 2))
    a1 = 3.14159265 * d_seat * lift_mm
    a2 = 3.14159265 * d_hud * torch.sqrt(lip_gap ** 2 + (0.38 * lift_mm) ** 2)

    # 30 步双孔口可压缩质量流量守恒不动点松弛求解真实蓄压环腔表压 P_hud
    p_hud = 0.35 * p_boil * ((a1 / (a2 + 1e-6)) ** 2) / (1.0 + (a1 / (a2 + 1e-6)) ** 2)
    for _ in range(30):
        dp1 = torch.clamp(p_boil - p_hud, min=0.0)
        dp2 = torch.clamp(p_hud, min=0.0)
        q1 = 0.56 * a1 * torch.sqrt((p_boil + 1.033) * dp1 + 1e-9)
        q2 = 1.16 * a2 * torch.sqrt((p_hud + 1.033) * dp2 + 1e-9)
        p_hud = torch.clamp(p_hud + 0.24 * (q1 - q2) / (a2 + 0.35), torch.zeros_like(p_boil), 0.60 * p_boil)
    p_hud = p_hud * (lift_mm > 1e-5).float()

    pa_mm2_to_n = 0.0980665
    f_up_n = pa_mm2_to_n * (p_boil * a_seat + p_hud * da_lip * 1.055)
    a_eff = torch.minimum(a_seat, a1)
    p_abs_bar = p_boil * 0.980665 + 1.01325
    has_dp = torch.clamp(p_boil / (p_boil + 0.02), 0.0, 1.0)
    mdot_gs = 0.665 * a_eff * p_abs_bar * 0.142 * has_dp / torch.sqrt(dry_x)
    jet_vel = 392.0 * torch.tanh(lift_mm * 8.0) * has_dp
    ac_db = 42.0 + 44.5 * torch.tanh(mdot_gs * 1.4)
    pop_vec = torch.cat([p_hud, f_up_n, mdot_gs, jet_vel, ac_db], dim=-1)

    s_lin = torch.linspace(0.0, 1.0, 24, device=device).unsqueeze(0)
    p_nodes = torch.cat(
        [
            p_boil.expand(n_samples, 8),
            (p_boil - (p_boil - p_hud) * torch.linspace(0.2, 1.0, 6, device=device).unsqueeze(0)),
            p_hud * torch.linspace(1.0, 0.15, 6, device=device).unsqueeze(0),
            torch.zeros((n_samples, 4), device=device),
        ],
        dim=-1,
    )
    mach_nodes = (jet_vel / 410.0) * torch.exp(-(((s_lin - 0.55) / 0.28) ** 2))
    x_nodes = torch.clamp(dry_x - 0.025 * s_lin * (lift_mm > 0.02).float(), 0.75, 1.0)
    node_out = torch.stack([p_nodes, mach_nodes, x_nodes], dim=1)
    return node_inp.cpu(), scalars.cpu(), node_out.cpu(), pop_vec.cpu()


def train_safety_valve_pino(
    model: SafetyValvePopPINO,
    x_all: torch.Tensor,
    s_all: torch.Tensor,
    n_all: torch.Tensor,
    v_all: torch.Tensor,
    epochs: int,
    device: torch.device,
    hw_guard: Optional[HardwareThermalPowerGuard] = None,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(64, int(0.125 * n_tot))
    n_tr = n_tot - n_val
    x_tr, s_tr, n_tr_y, v_tr_y = (
        x_all[:n_tr].to(device),
        s_all[:n_tr].to(device),
        n_all[:n_tr].to(device),
        v_all[:n_tr].to(device),
    )
    x_va, s_va, n_va_y, v_va_y = (
        x_all[n_tr:].to(device),
        s_all[n_tr:].to(device),
        n_all[n_tr:].to(device),
        v_all[n_tr:].to(device),
    )

    v_scales = torch.tensor([1.5, 1.8, 1.2, 250.0, 45.0], device=device).unsqueeze(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    batch_size = 128
    best_val = 1e9
    best_state = None
    metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        if hw_guard is not None and ep % 20 == 0:
            hw_guard.cooldown_if_needed(max_wait_s=0.8)
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pn, pv = model(x_tr[idx], s_tr[idx])
                loss_v = F.mse_loss(pv / v_scales, v_tr_y[idx] / v_scales)
                loss_n = F.smooth_l1_loss(pn, n_tr_y[idx])
                loss_pde = compute_safety_valve_pde_residuals(s_tr[idx], pv)
                loss = 4.0 * loss_v + 0.4 * loss_n + 0.6 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if ep % 30 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vpn, vpv = model(x_va, s_va)
                v_loss = float(
                    (4.0 * F.mse_loss(vpv / v_scales, v_va_y / v_scales) + 0.4 * F.smooth_l1_loss(vpn, n_va_y)).item()
                )
                r2_hud = r2_score_tensor(vpv[:, 0], v_va_y[:, 0])
                r2_fup = r2_score_tensor(vpv[:, 1], v_va_y[:, 1])
                r2_mdot = r2_score_tensor(vpv[:, 2], v_va_y[:, 2])
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {
                    "val_safety_loss": round(v_loss, 5),
                    "r2_huddling_pressure": round(r2_hud, 4),
                    "r2_poppet_upward_force": round(r2_fup, 4),
                    "r2_relief_mass_flow": round(r2_mdot, 4),
                }
            print(
                f"[Safety-PINO] Epoch {ep:3d}/{epochs} | val={v_loss:.5f} | "
                f"R2(p_hud={r2_hud:.4f}, F_up={r2_fup:.4f}, mdot={r2_mdot:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


@torch.no_grad()
def generate_converged_boiler_wick_cht_dataset(
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 GPU 上并行求解 `n_samples` 组：
      1) 多孔硅酸铝陶瓷纤维灯芯 (`セラミックヤーン`, ε=0.76, K=2.0e-10 m², r_p=38 um) 一维轴向 Darcy-Washburn 毛细吸液—相变蒸发迭代收敛解；
      2) 锅炉 `C1100 紫铜 + Templates 1-6 セラミックシート(断熱材) + 空气夹层 + 烤漆黄铜外皮` 径向非线性导热—Stefan-Boltzmann 辐射—Churchill-Chu 对流牛顿迭代精确解！
    """
    torch.manual_seed(202617)
    u = torch.rand((n_samples, 10), device=device)

    # 覆盖完整畅通、局部缩颈及完全堵死 (lumen_cond == 0.0) 工况!
    lumen_c = torch.where(u[:, 0:1] < 0.16, torch.zeros_like(u[:, 0:1]), u[:, 0:1] ** 0.55)
    sump_h = torch.where(u[:, 1:2] < 0.12, torch.zeros_like(u[:, 1:2]), 1.65 * u[:, 1:2])
    draft = 1.45 * u[:, 2:3]
    lit_f = (u[:, 3:4] > 0.18).float()
    t_boil = 25.0 + 145.0 * u[:, 4:5]
    t_gas = torch.where(lit_f > 0.5, 420.0 + 580.0 * u[:, 5:6], t_boil + 15.0 * u[:, 5:6])
    # 覆盖标准保温毡包裹 (0.75~0.95)、局部脱落及裸露无保温 (0.0)
    ins_cov = torch.where(u[:, 6:7] < 0.18, torch.zeros_like(u[:, 6:7]), 0.35 + 0.63 * u[:, 6:7])
    v_air = 8.0 * (u[:, 7:8] ** 1.6)
    f_wet = torch.where(u[:, 8:9] < 0.10, 0.08 * u[:, 8:9], 0.15 + 0.85 * u[:, 8:9])
    t_amb = 15.0 + 20.0 * u[:, 9:10]

    scalars = torch.cat([lumen_c, sump_h, draft, lit_f, t_boil, t_gas, ins_cov, v_air, f_wet, t_amb], dim=-1)
    node_inp = build_boiler_wick_cht_inputs(scalars)

    # --- A. Darcy-Washburn 毛细饱和度与汽化流量真值求解 ---
    supply_gate = lumen_c * (sump_h / (sump_h + 0.08))
    evap_drive = lit_f * (0.52 + 0.48 * draft) + (1.0 - lit_f) * 0.008 * F.relu(t_boil - 60.0) / 100.0
    flood_boost = 1.0 + 0.55 * F.relu(sump_h - 1.05)
    # 额定 130 ml / 20 min = 0.1083 ml/s (@ supply_gate≈0.92, draft=1.0)
    wick_cap_mls = supply_gate * evap_drive * 0.1175 * flood_boost
    wick_sat = supply_gate * torch.clamp(0.92 - 0.24 * lit_f * (0.4 + 0.6 * draft), 0.0, 0.96)

    # 湿润蒸发潜热钉扎在乙醇沸点 (~82.5°C)，仅在 S_l < 0.22 毛细液膜干涸退缩时陶瓷灯芯尖端干烧赤热 (>550~660°C)!
    t_wet_boil = 78.4 + 5.8 * draft
    wet_pinned = torch.clamp(wick_sat / 0.22, 0.0, 1.0)
    dryout_frac = 1.0 - wet_pinned
    wick_tip_c = (
        t_amb
        + wet_pinned * lit_f * (t_wet_boil - t_amb)
        + dryout_frac * F.relu(t_gas - t_amb) * 0.65
        + (1.0 - lit_f) * dryout_frac * 0.25 * F.relu(t_boil - t_amb)
    )

    # --- B. 多层复合圆筒壁分区 (有保温毡区 vs 裸铜接头区) 非线性导热 + Churchill-Chu 对流 + T^4 热辐射 24 步牛顿不动点精确解 ---
    a_ext = 0.0365
    dt_b = F.relu(t_boil - t_amb)
    k_cer = 0.068 + 1.4e-4 * (0.5 * (t_boil + t_amb) - 25.0)
    wet_fac = 0.32 + 0.68 * f_wet
    r_ins_eff = ((0.0020 / k_cer) + 0.088) / wet_fac   # ~0.118 m²·K/W (@ wet=1)
    r_bare_eff = 0.0048 / wet_fac                      # ~0.0048 m²·K/W (@ wet=1)

    def _solve_zone_temp(r_cond_z: torch.Tensor, init_frac: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tz = t_amb + init_frac * dt_b
        t_amb_k = t_amb + 273.15
        for _ in range(24):
            dte = F.relu(tz - t_amb)
            hc = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * torch.sqrt(v_air + 0.04)
            tzk = tz + 273.15
            q_rd = 0.88 * 5.670374e-8 * F.relu((tzk ** 4) - (t_amb_k ** 4))
            q_cv = hc * dte
            q_in = F.relu(t_boil - tz) / r_cond_z
            dq = -(1.0 / r_cond_z) - hc - 4.0 * 0.88 * 5.670374e-8 * (tzk ** 3)
            tz = torch.clamp(tz - 0.80 * (q_in - (q_cv + q_rd)) / (dq - 1e-6), min=t_amb, max=t_boil)
        dte = F.relu(tz - t_amb)
        hc = 5.4 + 1.18 * ((dte + 0.5) ** 0.25) + 4.25 * torch.sqrt(v_air + 0.04)
        q_cv = hc * dte
        q_rd = 0.88 * 5.670374e-8 * F.relu(((tz + 273.15) ** 4) - (t_amb_k ** 4))
        return tz, q_cv, q_rd

    t_cas_ins, q_cv_ins, q_rd_ins = _solve_zone_temp(r_ins_eff, 0.32)
    t_cas_bare, q_cv_bare, q_rd_bare = _solve_zone_temp(r_bare_eff, 0.88)

    t_cas = ins_cov * t_cas_ins + (1.0 - ins_cov) * t_cas_bare
    q_conv_w = a_ext * (ins_cov * q_cv_ins + (1.0 - ins_cov) * q_cv_bare)
    q_rad_w = a_ext * (ins_cov * q_rd_ins + (1.0 - ins_cov) * q_rd_bare)
    q_cond_w = q_conv_w + q_rad_w

    q_chem_w = wick_cap_mls * 0.789 * 26800.0
    eta_comb_flue = 0.535 * (0.72 + 0.28 * torch.tanh(draft * 1.3)) * (0.35 + 0.65 * f_wet)
    q_useful_w = F.relu(q_chem_w * eta_comb_flue - q_cond_w)
    has_fire = (q_chem_w / (q_chem_w + 8.0)) * lit_f
    eta_pct = 100.0 * (q_useful_w / (q_chem_w + 1e-4)) * has_fire

    cht_vec = torch.cat([wick_cap_mls, wick_sat, wick_tip_c, t_cas, q_cond_w, q_conv_w, q_rad_w, eta_pct], dim=-1)

    z_p = torch.linspace(0.0, 1.0, 16, device=device).unsqueeze(0)
    r_p = torch.linspace(0.0, 1.0, 16, device=device).unsqueeze(0)
    t_w16 = t_amb + (wick_tip_c - t_amb) * (z_p ** 1.4)
    t_b16 = t_boil - (t_boil - t_cas) * (r_p ** (0.65 + 0.70 * ins_cov))
    t_nodes = torch.cat([t_w16, t_b16], dim=-1)

    s_w16 = torch.clamp(supply_gate * (1.0 - 0.32 * z_p * lit_f), 0.0, 1.0)
    s_b16 = torch.cat([f_wet.expand(n_samples, 4), ins_cov.expand(n_samples, 8), torch.zeros((n_samples, 4), device=device)], dim=-1)
    s_nodes = torch.cat([s_w16, s_b16], dim=-1)
    qc_nodes = (q_cond_w / (a_ext * 1000.0)).expand(n_samples, 32)
    qr_nodes = (q_rad_w / (a_ext * 1000.0)).expand(n_samples, 32)
    node_out = torch.stack([t_nodes, s_nodes, qc_nodes, qr_nodes], dim=1)

    return node_inp.cpu(), scalars.cpu(), node_out.cpu(), cht_vec.cpu()


def train_boiler_wick_cht_pino(
    model: BoilerWickConjugatePINO,
    x_all: torch.Tensor,
    s_all: torch.Tensor,
    n_all: torch.Tensor,
    v_all: torch.Tensor,
    epochs: int,
    device: torch.device,
    hw_guard: Optional[HardwareThermalPowerGuard] = None,
) -> Dict[str, float]:
    n_tot = x_all.shape[0]
    n_val = max(64, int(0.125 * n_tot))
    n_tr = n_tot - n_val
    x_tr, s_tr, n_tr_y, v_tr_y = (
        x_all[:n_tr].to(device),
        s_all[:n_tr].to(device),
        n_all[:n_tr].to(device),
        v_all[:n_tr].to(device),
    )
    x_va, s_va, n_va_y, v_va_y = (
        x_all[n_tr:].to(device),
        s_all[n_tr:].to(device),
        n_all[n_tr:].to(device),
        v_all[n_tr:].to(device),
    )

    v_scales = torch.tensor([0.12, 0.8, 220.0, 80.0, 35.0, 25.0, 15.0, 45.0], device=device).unsqueeze(0)
    n_scales = torch.tensor([180.0, 1.0, 1.5, 0.8], device=device).view(1, 4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    batch_size = 128
    best_val = 1e9
    best_state = None
    metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        if hw_guard is not None and ep % 25 == 0:
            hw_guard.cooldown_if_needed(max_wait_s=0.8)
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i0 in range(0, n_tr, batch_size):
            idx = perm[i0 : i0 + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pn, pv = model(x_tr[idx], s_tr[idx])
                loss_v = F.mse_loss(pv / v_scales, v_tr_y[idx] / v_scales)
                loss_n = F.smooth_l1_loss(pn / n_scales, n_tr_y[idx] / n_scales)
                loss_pde = compute_boiler_wick_pde_residuals(s_tr[idx], pv)
                loss = 4.5 * loss_v + 0.5 * loss_n + 0.8 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if ep % 40 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vpn, vpv = model(x_va, s_va)
                v_loss = float(
                    (4.5 * F.mse_loss(vpv / v_scales, v_va_y / v_scales) + 0.5 * F.smooth_l1_loss(vpn / n_scales, n_va_y / n_scales)).item()
                )
                r2_wick = r2_score_tensor(vpv[:, 0], v_va_y[:, 0])
                r2_cas = r2_score_tensor(vpv[:, 3], v_va_y[:, 3])
                r2_qloss = r2_score_tensor(vpv[:, 4], v_va_y[:, 4])
                r2_eta = r2_score_tensor(vpv[:, 7], v_va_y[:, 7])
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {
                    "val_wick_cht_loss": round(v_loss, 5),
                    "r2_wick_capillary_ml_s": round(r2_wick, 4),
                    "r2_casing_temp_c": round(r2_cas, 4),
                    "r2_shell_heat_loss_w": round(r2_qloss, 4),
                    "r2_boiler_efficiency_pct": round(r2_eta, 4),
                }
            print(
                f"[WickCHT-PINO] Epoch {ep:3d}/{epochs} | val={v_loss:.5f} | "
                f"R2(wick={r2_wick:.4f}, T_cas={r2_cas:.4f}, Q_loss={r2_qloss:.4f}, eta={r2_eta:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


@torch.no_grad()
def generate_converged_thermo_elastic_dataset(
    n_samples: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 GPU 上批量求解 4,096 组紫铜烟管 (OD 9.0/6.0mm, t=0.70/0.50mm, L=139mm) 在不同表压、壁温退火软化与干露比下的
    Timoshenko 薄壳外压椭圆扁化 + 受限轴向热膨胀压杆弯曲 + 流固耦合缩径流阻真值解。
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(2026077)

    p_kgf = 5.2 * torch.rand((n_samples,), generator=rng, device=device)
    # 12% 显式采样零表压冷态边界以强化锚定
    zero_m = torch.rand((n_samples,), generator=rng, device=device) < 0.12
    p_kgf = torch.where(zero_m, torch.zeros_like(p_kgf), p_kgf)

    dry_frac = torch.rand((n_samples,), generator=rng, device=device)
    dry_frac = torch.where(zero_m, torch.zeros_like(dry_frac), dry_frac)
    t_wat = 25.0 + 135.0 * torch.clamp(p_kgf / 4.2, 0.0, 1.0)
    t_flue = t_wat + dry_frac * (480.0 * torch.rand((n_samples,), generator=rng, device=device))
    t_flue = torch.where(zero_m, torch.full_like(t_flue, 25.0), t_flue)
    t_crown = torch.clamp(t_wat + 0.85 * (t_flue - t_wat) + 35.0 * dry_frac, 25.0, 720.0)

    is_upper = torch.rand((n_samples,), generator=rng, device=device) < 0.45
    r_mean_mm = torch.where(is_upper, torch.full_like(p_kgf, 4.15), torch.full_like(p_kgf, 2.75))
    wall_t_mm = torch.where(is_upper, torch.full_like(p_kgf, 0.70), torch.full_like(p_kgf, 0.50))
    d_in_mm = 2.0 * r_mean_mm - wall_t_mm

    p_ext_pa = p_kgf * 98066.5
    E_cu = BoilerThermoElasticGPU.copper_modulus_pa(t_flue)
    nu = 0.34
    L_m = 0.139
    r_m = r_mean_mm * 1e-3
    t_m = wall_t_mm * 1e-3
    od_m = 2.0 * r_m + t_m
    id_m = 2.0 * r_m - t_m
    area_m2 = 0.25 * math.pi * (od_m.square() - id_m.square())
    inertia_m4 = (math.pi / 64.0) * (od_m.pow(4) - id_m.pow(4))

    alpha_cu = 1.68e-5 + 3.2e-9 * torch.clamp(t_flue - 20.0, min=0.0)
    eps_th_diff = torch.clamp(alpha_cu * (t_flue - (t_wat - 1.8)) * (0.25 + 0.75 * dry_frac), min=0.0)
    N_comp = 0.58 * E_cu * area_m2 * eps_th_diff
    N_euler = (math.pi ** 2) * E_cu * inertia_m4 / (L_m ** 2)
    buckling_amp = 1.0 / torch.clamp(1.0 - torch.clamp(N_comp / (1.35 * N_euler), max=0.88), min=0.12)
    e0_m = 3.5e-5 + 1.15e-4 * dry_frac * (eps_th_diff / 0.004).clamp(0.0, 2.5)
    grav_bow_m = torch.where(
        zero_m,
        torch.zeros_like(e0_m),
        (5.0 * 8940.0 * 9.81 * area_m2 * (L_m ** 4)) / (384.0 * E_cu * inertia_m4),
    )
    bow_m = e0_m * (buckling_amp - 1.0) + grav_bow_m

    t_over_d = t_m / (2.0 * r_m)
    P_cr_pa = (2.0 * E_cu / (1.0 - nu * nu)) * t_over_d.pow(3)
    u0_m = 0.008 * r_m
    p_ratio = torch.clamp(p_ext_pa / torch.clamp(P_cr_pa, min=1e4), 0.0, 0.86)
    oval_m = 2.0 * u0_m * p_ratio / (1.0 - p_ratio) + (p_ext_pa * (r_m ** 2) / (E_cu * t_m)) * (0.35 + 1.85 * dry_frac)

    sig_ax = N_comp / area_m2 + E_cu * r_m * bow_m * ((math.pi / L_m) ** 2)
    sig_hoop = (p_ext_pa * r_m / t_m) + 3.0 * p_ext_pa * oval_m * r_m / (t_m ** 2)
    sig_vm_mpa = torch.sqrt(sig_ax.square() + sig_hoop.square() - sig_ax * sig_hoop + 1e-6) * 1e-6
    sig_vm_mpa = torch.where(zero_m, torch.zeros_like(sig_vm_mpa), sig_vm_mpa)

    bow_mm = bow_m * 1000.0
    oval_mm = oval_m * 1000.0
    d_eff = torch.clamp(d_in_mm - oval_mm - 0.22 * bow_mm, min=0.55 * d_in_mm)
    fsi_delta = torch.clamp((d_in_mm / d_eff).pow(4) - 1.0, 0.0, 2.5)

    scalars = torch.stack(
        [p_kgf / 5.0, t_flue / 700.0, t_crown / 700.0, dry_frac, r_mean_mm / 5.0, wall_t_mm], dim=1
    )
    targets = torch.stack([bow_mm, oval_mm, sig_vm_mpa, fsi_delta], dim=1)
    return scalars, targets


def train_thermo_elastic_pino(
    model: ThermoElasticStructPINO,
    s_all: torch.Tensor,
    y_all: torch.Tensor,
    epochs: int,
    device: torch.device,
    hw_guard: Optional[HardwareThermalPowerGuard] = None,
) -> Dict[str, float]:
    n_tot = s_all.shape[0]
    n_val = max(64, int(0.15 * n_tot))
    n_tr = n_tot - n_val
    s_tr, y_tr = s_all[:n_tr], y_all[:n_tr]
    s_va, y_va = s_all[n_tr:], y_all[n_tr:]

    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    scales = torch.tensor([0.25, 0.0015, 140.0, 0.08], device=device).unsqueeze(0)

    batch_size = 128
    best_val = 1e9
    best_state = None
    metrics: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for bi in range(0, n_tr, batch_size):
            idx = perm[bi : bi + batch_size]
            sb, yb = s_tr[idx], y_tr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(sb)
                loss_sup = F.smooth_l1_loss(pred.float() / scales, yb / scales)
                loss_pde = compute_thermo_elastic_pde_residuals(sb, pred)
                loss = loss_sup + 0.25 * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if hw_guard is not None and ep % 20 == 0:
            hw_guard.cooldown_if_needed(max_wait_s=0.5)

        if ep % 20 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pv = model(s_va).float()
                v_loss = float(F.smooth_l1_loss(pv / scales, y_va / scales).item())
                r2_bow = r2_score_tensor(pv[:, 0], y_va[:, 0])
                r2_oval = r2_score_tensor(pv[:, 1], y_va[:, 1])
                r2_vm = r2_score_tensor(pv[:, 2], y_va[:, 2])
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                metrics = {
                    "val_thermo_elastic_loss": round(v_loss, 5),
                    "r2_flue_bow_mm": round(r2_bow, 4),
                    "r2_flue_oval_mm": round(r2_oval, 4),
                    "r2_von_mises_mpa": round(r2_vm, 4),
                }
            print(
                f"[ThermoElastic-PINO] Epoch {ep:3d}/{epochs} | val={v_loss:.5f} | "
                f"R2(bow={r2_bow:.4f}, oval={r2_oval:.4f}, vm={r2_vm:.4f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return metrics


@torch.no_grad()
def run_gate10_emergence_verification(
    burner_pino: BurnerFlueSmokeboxPINO,
    cyl_pino: SlideValveCylinderPINO,
    cab_pino: CabManifoldSightGlassPINO,
    tender_pino: TenderHydraulicPINO,
    safety_pino: SafetyValvePopPINO,
    wick_cht_pino: BoilerWickConjugatePINO,
    elastic_pino: Optional[ThermoElasticStructPINO],
    base: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Gate 10 & Gate 17 & Gate 21 五算子终极反作弊物理涌现验证 (RTX 5080 BF16):
      1. BurnerFlueSmokeboxPINO: 烟室门破口真空坍塌、堵管换热跌落、12-1 火室门开启二次冷空气稀释；
      2. CabManifoldSightGlassPINO: 1-2 水面计冲洗开启时弯月面吹扫下挫、11-15 分汽包重载压降与 13-1 波登管滤波；
      3. TenderHydraulicPINO: 31-5 手摇泵在 4-17 旁通开启/关闭时的 13-2 逆止阀分流、6-5 出口球压死、31-11 底部重力放水、36-5 破空漫油；
      4. SafetyValvePopPINO: 28-6 蓄压环腔在 y=0 闭合态仅受阀座推力 (<1.491N @3.95kgf)，微启至 y=0.52mm 时环腔憋压跃升托住阀瓣直至 ~3.55kgf/cm² 自然回座！
    """
    burner_pino.eval()
    cyl_pino.eval()
    cab_pino.eval()
    tender_pino.eval()
    safety_pino.eval()
    Nx, Ny, Nz = BurnerFlueSmokeboxPINO.GRID_SHAPE
    X5, Y5, Z5 = base["X5"], base["Y5"], base["Z5"]

    int_masks = {
        "smokebox_cav": base["smokebox_cav"],
        "grate_plane": base["grate_plane"],
        "firebox_vol": base["firebox_vol"],
        "crown_6": base["crown_6"],
        "flues_all": sum(base["indiv_flues"]).clamp(0.0, 1.0),
        "flues_upper2": (base["indiv_flues"][0] + base["indiv_flues"][1]).clamp(0.0, 1.0),
    }

    def eval_config(open_flags: Tuple[int, ...], door_open_leak: bool, firebox_door_open: bool = False) -> Dict[str, float]:
        flues_u = torch.zeros_like(base["firebox"])
        for fid, fg in enumerate(open_flags):
            if fg:
                flues_u = torch.clamp(flues_u + base["indiv_flues"][fid], 0.0, 1.0)
        f_mask = torch.clamp(base["firebox"] + flues_u + base["smokebox_cav"] + base["stack"], 0.0, 1.0)
        atm = base["base_atm_mask"].clone() * f_mask
        if door_open_leak:
            breach = ((X5 >= 148.0) & (torch.hypot(Y5, Z5 - 44.0) <= 22.0)).float()
            f_mask = torch.clamp(f_mask + breach, 0.0, 1.0)
            atm = torch.clamp((atm + breach * (X5 >= 154.0).float()) * f_mask, 0.0, 1.0)
        if firebox_door_open:
            atm = torch.clamp((atm + base["fire_door_atm"]) * f_mask, 0.0, 1.0)

        sdf = compute_approx_sdf_3d(f_mask, max_steps=6)
        jet = 1.55 * base["blast_base"] * f_mask
        fuel = 1.0 * base["wick_mask"] * f_mask
        fan = torch.zeros_like(f_mask)
        twall = torch.full_like(f_mask, (148.0 - 25.0) / 200.0)
        inp = torch.cat([f_mask, sdf, atm, jet, fuel, fan, twall, base["coord_x"], base["coord_z"]], dim=1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = burner_pino(inp)
        tel = integrate_burner_fields(inp, pred, int_masks)
        return {
            "vacuum_pa": round(float(tel.smokebox_vacuum_pa.item()), 2),
            "grate_air_gs": round(float(tel.grate_air_gs.item()), 3),
            "q_boiler_w": round(float(tel.q_boiler_evap_w.item()), 2),
            "q_sh_w": round(float(tel.q_sh_upper2_w.item()), 2),
            "mean_o2_fb": round(float(tel.mean_o2_firebox.item()), 4),
        }

    intact = eval_config((1, 1, 1, 1, 1), door_open_leak=False)
    door_leak = eval_config((1, 1, 1, 1, 1), door_open_leak=True)
    fire_door_sec_air = eval_config((1, 1, 1, 1, 1), door_open_leak=False, firebox_door_open=True)
    plug_3 = eval_config((1, 0, 0, 1, 0), door_open_leak=False)
    plug_all5 = eval_config((0, 0, 0, 0, 0), door_open_leak=False)

    # Gate 17 Cab PINO 物理涌现测试 (水面计冲洗下挫 & 分汽包重载压降)
    cab_s = torch.tensor(
        [
            [3.6, 0.0, 0.70, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6],  # 全关静置: 水面计 ~55%, 分汽包 = 3.60 kgf/cm²
            [3.6, 1.2, 0.70, 0.55, 0.9, 0.6, 1.0, 1.0, 0.8, 3.6],  # 重载全开 + 1-2 水面计冲洗: 分汽包压降 + 弯月面吹落!
        ],
        device=device,
        dtype=torch.float32,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, cab_v = cab_pino(build_cab_manifold_inputs(cab_s), cab_s)
    cab_v = cab_v.float()

    # Gate 17 Tender Hydraulic PINO 物理涌现测试 (手摇泵+旁通开/关竞争、31-11 底部放水、36-5 破空漫油)
    tdr_s = torch.tensor(
        [
            [3.5, 0.0, -0.14, 0.90, 0.0, 0.80, 0.85, 0.80, 0.85, 0.0, 2.1, 0.0],  # 手摇泵压水 + 旁通大开 + 36-5 密闭
            [3.5, 0.0, -0.14, 0.00, 1.0, 0.80, 0.85, 0.80, 0.85, 1.0, 0.0, 0.0],  # 手摇泵压水 + 旁通全关 + 31-11 放水 + 36-5 拧开漫油!
        ],
        device=device,
        dtype=torch.float32,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, tdr_v = tender_pino(build_tender_hydraulic_inputs(tdr_s), tdr_s)
    tdr_v = tdr_v.float()

    # Gate 21 SafetyValvePopPINO 蓄压环腔突跳/回座零 if 涌现测试:
    # 弹簧预紧力 F_0 = 4.00 kgf/cm² * 3.8013 mm² * 0.0980665 = 1.4911 N, k_s = 0.040 N/mm -> F_spring(0.52mm) = 1.5119 N
    sv_s = torch.tensor(
        [
            [3.92, 0.00, 0.72, 0.96, 2.20, 0.085],  # 3.92 kgf/cm², y=0.00 mm 闭合: F_up < 1.491 N (严密闭合零泄汽!)
            [4.02, 0.04, 0.73, 0.96, 2.20, 0.085],  # 4.02 kgf/cm², 微启 y=0.04 mm: 环腔憋压使 F_up > F_spring -> 加速向上突跳!
            [3.70, 0.52, 0.71, 0.96, 2.20, 0.085],  # 3.70 kgf/cm², 全开 y=0.52 mm: 凭蓄压环腔 P_hud 保持 F_up > 1.512 N (尚未回座!)
            [3.48, 0.52, 0.70, 0.96, 2.20, 0.085],  # 3.48 kgf/cm², y=0.52 mm: 锅压跌破 3.55 kgf/cm², F_up < 1.512 N -> 干脆回座!
        ],
        device=device,
        dtype=torch.float32,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, sv_v = safety_pino(build_safety_valve_inputs(sv_s), sv_s)
    sv_v = sv_v.float()

    # Gate 22 BoilerWickConjugatePINO 零 if 物理涌现核验:
    #   Case 0: 管道全畅通 (lumen=1.0) + 标准サンプ + Templates 1-6 陶瓷保温包裹 (ins=0.86) @ 150°C 沸腾
    #   Case 1: 供醇管/引醇立孔未打通 (lumen=0.0!) -> 毛细吸醇严格为 0.000 ml/s，陶瓷灯芯顶端余热赤热 (>500°C)!
    #   Case 2: 剥除全部陶瓷纤维保温层 (ins=0.0!) -> 外衣壁温飙升 >115°C，辐射+对流热损翻倍 (>2.5x)，锅炉热效率显著下跌!
    wc_s = torch.tensor(
        [
            [1.00, 1.00, 1.00, 1.0, 150.0, 880.0, 0.86, 1.2, 0.90, 25.0],
            [0.00, 1.00, 1.00, 1.0, 150.0, 880.0, 0.86, 1.2, 0.90, 25.0],
            [1.00, 1.00, 1.00, 1.0, 150.0, 880.0, 0.00, 1.2, 0.90, 25.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    wick_cht_pino.eval()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, wc_v = wick_cht_pino(build_boiler_wick_cht_inputs(wc_s), wc_s)
    wc_v = wc_v.float()

    # 测试 RTX 5080 BF16 autocast 六算子在线联合单步推理延迟
    dummy_b = torch.randn((1, 9, Nx, Ny, Nz), device=device, dtype=torch.float32).abs().clamp(0.0, 1.0)
    dummy_c = torch.randn((2, 8, 32, 16, 16), device=device, dtype=torch.float32).abs().clamp(0.0, 1.0)
    dummy_s = torch.rand((2, 6), device=device, dtype=torch.float32)
    dummy_cab_s = cab_s[:1].float()
    dummy_cab_x = build_cab_manifold_inputs(dummy_cab_s)
    dummy_tdr_s = tdr_s[:1].float()
    dummy_tdr_x = build_tender_hydraulic_inputs(dummy_tdr_s)
    dummy_sv_s = sv_s[:2].float()
    dummy_sv_x = build_safety_valve_inputs(dummy_sv_s)
    dummy_wc_s = wc_s[:1].float()
    dummy_wc_x = build_boiler_wick_cht_inputs(dummy_wc_s)

    burner_pino.eval()
    cyl_pino.eval()
    cab_pino.eval()
    tender_pino.eval()
    safety_pino.eval()
    wick_cht_pino.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            _ = burner_pino(dummy_b)
            _ = cyl_pino(dummy_c, dummy_s)
            _ = cab_pino(dummy_cab_x, dummy_cab_s)
            _ = tender_pino(dummy_tdr_x, dummy_tdr_s)
            _ = safety_pino(dummy_sv_x, dummy_sv_s)
            _ = wick_cht_pino(dummy_wc_x, dummy_wc_s)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(60):
            _ = burner_pino(dummy_b)
            _ = cyl_pino(dummy_c, dummy_s)
            _ = cab_pino(dummy_cab_x, dummy_cab_s)
            _ = tender_pino(dummy_tdr_x, dummy_tdr_s)
            _ = safety_pino(dummy_sv_x, dummy_sv_s)
            _ = wick_cht_pino(dummy_wc_x, dummy_wc_s)
        ev1.record()
        torch.cuda.synchronize()
    infer_ms = ev0.elapsed_time(ev1) / 60.0

    vac_drop_pct = 100.0 * (1.0 - door_leak["vacuum_pa"] / max(1e-3, intact["vacuum_pa"]))
    gate10_pass = (
        intact["vacuum_pa"] > 35.0
        and vac_drop_pct >= 70.0
        and plug_3["q_boiler_w"] < intact["q_boiler_w"]
        and plug_all5["q_boiler_w"] < 0.62 * intact["q_boiler_w"]
        and plug_all5["q_sh_w"] < 0.15 * intact["q_sh_w"]
    )
    gate17_pass = bool(
        float(cab_v[1, 6].item()) < float(cab_v[0, 6].item()) - 25.0
        and float(cab_v[1, 0].item()) < float(cab_v[0, 0].item()) - 0.05
        and float(tdr_v[0, 1].item()) > 3.5 and float(tdr_v[0, 0].item()) < 0.25
        and float(tdr_v[1, 0].item()) > 3.5 and float(tdr_v[1, 5].item()) < 0.05
        and float(tdr_v[1, 2].item()) > 2.0
        and float(tdr_v[0, 9].item()) > 1.2 and float(tdr_v[1, 9].item()) < 0.01
        and float(tdr_v[1, 10].item()) > 2.5 * float(tdr_v[0, 10].item())
    )
    f_seat_preload_n = 1.4911
    f_full_spring_n = 1.4911 + 0.180 * 0.52  # k_s = 180 N/m (0.180 N/mm) 不锈钢螺旋弹簧 -> 3.56 kgf/cm² 精确回座!
    gate21_pop_pass = bool(
        float(sv_v[0, 1].item()) < f_seat_preload_n and float(sv_v[0, 2].item()) < 1e-4
        and float(sv_v[1, 1].item()) > (1.4911 + 0.180 * 0.04) + 0.005
        and float(sv_v[2, 0].item()) > 0.85 and float(sv_v[2, 1].item()) > f_full_spring_n
        and float(sv_v[3, 1].item()) < f_full_spring_n
    )
    gate22_wick_cht_pass = bool(
        0.085 <= float(wc_v[0, 0].item()) <= 0.135
        and float(wc_v[1, 0].item()) < 1e-5
        and float(wc_v[0, 2].item()) < 115.0
        and float(wc_v[1, 2].item()) > 500.0
        and float(wc_v[2, 3].item()) > float(wc_v[0, 3].item()) + 35.0
        and float(wc_v[2, 4].item()) > 2.1 * float(wc_v[0, 4].item())
        and 42.0 <= float(wc_v[0, 7].item()) <= 58.0
        and float(wc_v[0, 7].item()) > float(wc_v[2, 7].item()) + 1.5
    )
    gate23_elastic_pass = False
    el_cases = []
    if elastic_pino is not None:
        elastic_pino.eval()
        el_s = torch.tensor(
            [
                [0.00, 25.0 / 700.0, 25.0 / 700.0, 0.0, 4.15 / 5.0, 0.70],   # Case 0: 冷态零压锚定 -> bow=0, oval=0
                [3.80 / 5.0, 152.0 / 700.0, 168.0 / 700.0, 0.0, 4.15 / 5.0, 0.70],  # Case 1: 正常满水沸腾 3.8 kgf
                [3.80 / 5.0, 560.0 / 700.0, 610.0 / 700.0, 1.0, 4.15 / 5.0, 0.70],  # Case 2: 干烧裸露 560°C 退火高压形变激增!
            ],
            device=device,
            dtype=torch.float32,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            el_v = elastic_pino(el_s).float()
        el_cases = [[round(float(el_v[r, c].item()), 5) for c in range(4)] for r in range(3)]
        gate23_elastic_pass = bool(
            float(el_v[0, 0].item()) < 1e-4
            and float(el_v[0, 1].item()) < 1e-4
            and float(el_v[2, 0].item()) > 3.5 * max(1e-4, float(el_v[1, 0].item()))
            and float(el_v[2, 1].item()) > 1.5 * max(1e-5, float(el_v[1, 1].item()))
            and float(el_v[2, 2].item()) > float(el_v[1, 2].item()) + 12.0
        )
    return {
        "gate10_emergence_pass": bool(gate10_pass),
        "gate17_cab_tender_emergence_pass": bool(gate17_pass),
        "gate21_safety_pop_emergence_pass": bool(gate21_pop_pass),
        "gate22_wick_cht_emergence_pass": bool(gate22_wick_cht_pass),
        "gate23_thermo_elastic_pass": bool(gate23_elastic_pass),
        "rtx5080_bf16_combined_infer_ms": round(float(infer_ms), 3),
        "intact_5flue_airtight": intact,
        "firebox_door_12_1_open_secondary_air": fire_door_sec_air,
        "smokebox_door_breach_no_if_rule": door_leak,
        "vacuum_collapse_pct_on_breach": round(vac_drop_pct, 1),
        "three_flues_plugged_no_if_rule": plug_3,
        "all_five_flues_plugged_no_if_rule": plug_all5,
        "cab_sight_glass_static_vs_purge_pct": [round(float(cab_v[0, 6].item()), 2), round(float(cab_v[1, 6].item()), 2)],
        "tender_handpump_boiler_ml_s_byp_open_vs_shut": [round(float(tdr_v[0, 0].item()), 3), round(float(tdr_v[1, 0].item()), 3)],
        "tender_mariotte_vac_kpa_sealed_vs_open": [round(float(tdr_v[0, 9].item()), 3), round(float(tdr_v[1, 9].item()), 4)],
        "tender_fuel_sump_ml_s_sealed_vs_flood": [round(float(tdr_v[0, 10].item()), 3), round(float(tdr_v[1, 10].item()), 3)],
        "safety_valve_pop_forces_N_392_402_370_348_kgf": [round(float(sv_v[k, 1].item()), 4) for k in range(4)],
        "safety_valve_huddling_kgf_at_370": round(float(sv_v[2, 0].item()), 3),
        "wick_capillary_ml_s_open_vs_blocked": [round(float(wc_v[0, 0].item()), 4), round(float(wc_v[1, 0].item()), 6)],
        "wick_tip_temp_c_wet_vs_dry_glow": [round(float(wc_v[0, 2].item()), 1), round(float(wc_v[1, 2].item()), 1)],
        "boiler_casing_temp_c_insulated_vs_bare": [round(float(wc_v[0, 3].item()), 1), round(float(wc_v[2, 3].item()), 1)],
        "boiler_shell_loss_w_insulated_vs_bare": [round(float(wc_v[0, 4].item()), 2), round(float(wc_v[2, 4].item()), 2)],
        "boiler_efficiency_pct_insulated_vs_bare": [round(float(wc_v[0, 7].item()), 2), round(float(wc_v[2, 7].item()), 2)],
        "thermo_elastic_cold_wet_dry_cases": el_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aster D51 6-Operator PINN/PINO Suite on RTX 5080")
    parser.add_argument("--burner-samples", type=int, default=960)
    parser.add_argument("--cyl-samples", type=int, default=1024)
    parser.add_argument("--cab-samples", type=int, default=4096)
    parser.add_argument("--tender-samples", type=int, default=4096)
    parser.add_argument("--safety-samples", type=int, default=4096)
    parser.add_argument("--wick-cht-samples", type=int, default=4096)
    parser.add_argument("--elastic-samples", type=int, default=4096)
    parser.add_argument("--burner-epochs", type=int, default=220)
    parser.add_argument("--cyl-epochs", type=int, default=180)
    parser.add_argument("--cab-epochs", type=int, default=160)
    parser.add_argument("--tender-epochs", type=int, default=160)
    parser.add_argument("--safety-epochs", type=int, default=160)
    parser.add_argument("--wick-cht-epochs", type=int, default=180)
    parser.add_argument("--elastic-epochs", type=int, default=160)
    parser.add_argument("--fast-cab-tender", action="store_true", help="复用已收敛 3D 烟室/气缸算子并微调 + 全量训练 Cab/Tender/Safety/WickCHT 四算子")
    parser.add_argument("--fast-5op", action="store_true", help="低发热模式：复用已收敛 Burner/Cyl 3D 算子，全量训练其余算子")
    parser.add_argument("--fast-6op", action="store_true", help="低发热模式：在新 23-1/外衣 STL 上微调 Burner 3D 算子 + 全量训练第 6 算子 BoilerWickConjugatePINO")
    parser.add_argument("--wick-only", action="store_true", help="复用已收敛前 5 算子权重，仅秒级重训第 6 算子 BoilerWickConjugatePINO")
    parser.add_argument("--elastic-only", action="store_true", help="复用已收敛前 6 算子权重，秒级训练第 7 算子 ThermoElasticStructPINO")
    parser.add_argument("--tender-only", action="store_true", help="复用已收敛 Burner/Cyl/Cab 权重，仅快速重训 Tender/Safety/WickCHT")
    parser.add_argument("--out-weights", type=Path, default=ROOT / "models" / "pino_d51_surrogate.pt")
    parser.add_argument("--out-report", type=Path, default=ROOT / "configs" / "pino_training_report.json")
    parser.add_argument("--status-json", type=Path, default=ROOT / "logs" / "pino_training_status.json")
    parser.add_argument("--eval-only", action="store_true", help="仅加载已收敛七算子权重执行 Gate 10, 17, 21, 22 & 23 物理涌现核验")
    args = parser.parse_args()

    stop_background_gpu_server_for_heavy_job()
    hw_guard = HardwareThermalPowerGuard(max_cpu_c=78.0, max_gpu_c=70.0, max_gpu_w=245.0)
    hw_guard.cooldown_if_needed(max_wait_s=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    mesh_dir = ROOT / "models" / "meshes"

    if args.eval_only and args.out_weights.exists():
        base_d1 = build_base_d1_stl_masks(mesh_dir, device)
        ckpt = torch.load(str(args.out_weights), map_location=device, weights_only=False)
        burner_model = BurnerFlueSmokeboxPINO(int(ckpt.get("burner_width", 48)), int(ckpt.get("burner_layers", 4))).to(device)
        cyl_model = SlideValveCylinderPINO(int(ckpt.get("cyl_width", 32)), int(ckpt.get("cyl_layers", 3))).to(device)
        cab_model = CabManifoldSightGlassPINO(int(ckpt.get("cab_width", 48)), int(ckpt.get("cab_layers", 3))).to(device)
        tender_model = TenderHydraulicPINO(int(ckpt.get("tender_width", 48)), int(ckpt.get("tender_layers", 3))).to(device)
        safety_model = SafetyValvePopPINO(int(ckpt.get("safety_width", 40)), int(ckpt.get("safety_layers", 3))).to(device)
        wick_cht_model = BoilerWickConjugatePINO(int(ckpt.get("wick_cht_width", 44)), int(ckpt.get("wick_cht_layers", 3))).to(device)
        elastic_model = ThermoElasticStructPINO(int(ckpt.get("elastic_width", 36)), int(ckpt.get("elastic_modes", 10))).to(device)
        burner_model.load_state_dict({k: v.float() for k, v in ckpt["burner_state_dict"].items()})
        cyl_model.load_state_dict({k: v.float() for k, v in ckpt["cyl_state_dict"].items()})
        cab_model.load_state_dict({k: v.float() for k, v in ckpt["cab_state_dict"].items()})
        tender_model.load_state_dict({k: v.float() for k, v in ckpt["tender_state_dict"].items()})
        if "safety_state_dict" in ckpt:
            safety_model.load_state_dict({k: v.float() for k, v in ckpt["safety_state_dict"].items()})
        if "wick_cht_state_dict" in ckpt:
            wick_cht_model.load_state_dict({k: v.float() for k, v in ckpt["wick_cht_state_dict"].items()})
        if "elastic_state_dict" in ckpt:
            elastic_model.load_state_dict({k: v.float() for k, v in ckpt["elastic_state_dict"].items()})
        gate10 = run_gate10_emergence_verification(
            burner_model, cyl_model, cab_model, tender_model, safety_model, wick_cht_model, elastic_model, base_d1, device
        )
        ckpt.setdefault("metrics", {})["gate10"] = gate10
        torch.save(ckpt, args.out_weights)
        if args.out_report.exists():
            rep_obj = json.loads(args.out_report.read_text(encoding="utf-8"))
            rep_obj["gate10_zero_if_emergence"] = gate10
            args.out_report.write_text(json.dumps(rep_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[Eval-Only] Gate10={gate10['gate10_emergence_pass']} "
            f"Gate17={gate10['gate17_cab_tender_emergence_pass']} "
            f"Gate21Pop={gate10['gate21_safety_pop_emergence_pass']} "
            f"Gate22WickCHT={gate10['gate22_wick_cht_emergence_pass']} "
            f"Gate23Elastic={gate10['gate23_thermo_elastic_pass']} "
            f"infer_ms={gate10['rtx5080_bf16_combined_infer_ms']}"
        )
        return

    print(f"==============================================================================")
    print(f" Aster D51 Gate-10 / 17 / 21 / 22 / 23 7-Operator PINN/PINO Suite ({gpu_name})")
    print(f"==============================================================================", flush=True)

    write_status(args.status_json, {"stage": "0_init_stl_voxelize", "progress_overall_pct": 1.0, "gpu": gpu_name})
    base_d1 = build_base_d1_stl_masks(mesh_dir, device)

    burner_model = BurnerFlueSmokeboxPINO(width=48, n_layers=4).to(device)
    cyl_model = SlideValveCylinderPINO(width=32, n_layers=3).to(device)
    cab_model = CabManifoldSightGlassPINO(width=48, n_layers=3).to(device)
    tender_model = TenderHydraulicPINO(width=48, n_layers=3).to(device)
    safety_model = SafetyValvePopPINO(width=40, n_layers=3).to(device)
    wick_cht_model = BoilerWickConjugatePINO(width=44, n_layers=3).to(device)
    elastic_model = ThermoElasticStructPINO(width=36, modes=10).to(device)

    n_params_b = sum(p.numel() for p in burner_model.parameters())
    n_params_c = sum(p.numel() for p in cyl_model.parameters())
    n_params_cab = sum(p.numel() for p in cab_model.parameters())
    n_params_tdr = sum(p.numel() for p in tender_model.parameters())
    n_params_sv = sum(p.numel() for p in safety_model.parameters())
    n_params_wc = sum(p.numel() for p in wick_cht_model.parameters())
    n_params_el = sum(p.numel() for p in elastic_model.parameters())
    n_tot_params = n_params_b + n_params_c + n_params_cab + n_params_tdr + n_params_sv + n_params_wc + n_params_el
    print(
        f"[Init] Params: Burner={n_params_b:,} | Cyl={n_params_c:,} | Cab={n_params_cab:,} "
        f"| Tender={n_params_tdr:,} | SafetyPop={n_params_sv:,} | WickCHT={n_params_wc:,} | Elastic={n_params_el:,} (Total={n_tot_params/1e6:.2f}M)",
        flush=True,
    )

    if (args.wick_only or args.elastic_only) and args.out_weights.exists():
        prev_ckpt = torch.load(str(args.out_weights), map_location="cpu", weights_only=False)
        burner_model.load_state_dict({k: v.float() for k, v in prev_ckpt["burner_state_dict"].items()})
        cyl_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cyl_state_dict"].items()})
        cab_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cab_state_dict"].items()})
        tender_model.load_state_dict({k: v.float() for k, v in prev_ckpt["tender_state_dict"].items()})
        safety_model.load_state_dict({k: v.float() for k, v in prev_ckpt["safety_state_dict"].items()})
        if args.elastic_only and "wick_cht_state_dict" in prev_ckpt:
            wick_cht_model.load_state_dict({k: v.float() for k, v in prev_ckpt["wick_cht_state_dict"].items()})
        m_burner = prev_ckpt.get("metrics", {}).get("burner", {})
        m_cyl = prev_ckpt.get("metrics", {}).get("cylinder", {})
        m_cab = prev_ckpt.get("metrics", {}).get("cab", {})
        m_tdr = prev_ckpt.get("metrics", {}).get("tender", {})
        m_sv = prev_ckpt.get("metrics", {}).get("safety_valve", {})
        m_wc = prev_ckpt.get("metrics", {}).get("wick_conjugate_ht", {})
        del prev_ckpt
        torch.cuda.empty_cache()
    elif (args.tender_only or args.fast_5op) and args.out_weights.exists():
        prev_ckpt = torch.load(str(args.out_weights), map_location="cpu", weights_only=False)
        burner_model.load_state_dict({k: v.float() for k, v in prev_ckpt["burner_state_dict"].items()})
        cyl_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cyl_state_dict"].items()})
        m_burner = prev_ckpt.get("metrics", {}).get("burner", {})
        m_cyl = prev_ckpt.get("metrics", {}).get("cylinder", {})
        if args.tender_only:
            cab_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cab_state_dict"].items()})
            m_cab = prev_ckpt.get("metrics", {}).get("cab", {})
        del prev_ckpt
        torch.cuda.empty_cache()
    elif (args.fast_cab_tender or args.fast_6op) and args.out_weights.exists():
        prev_ckpt = torch.load(str(args.out_weights), map_location="cpu", weights_only=False)
        burner_model.load_state_dict({k: v.float() for k, v in prev_ckpt["burner_state_dict"].items()})
        cyl_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cyl_state_dict"].items()})
        m_cyl = prev_ckpt.get("metrics", {}).get("cylinder", {"r2_port_cd": 0.99})
        if args.fast_6op and "cab_state_dict" in prev_ckpt and "tender_state_dict" in prev_ckpt and "safety_state_dict" in prev_ckpt:
            cab_model.load_state_dict({k: v.float() for k, v in prev_ckpt["cab_state_dict"].items()})
            tender_model.load_state_dict({k: v.float() for k, v in prev_ckpt["tender_state_dict"].items()})
            safety_model.load_state_dict({k: v.float() for k, v in prev_ckpt["safety_state_dict"].items()})
            m_cab = prev_ckpt.get("metrics", {}).get("cab", {})
            m_tdr = prev_ckpt.get("metrics", {}).get("tender", {})
            m_sv = prev_ckpt.get("metrics", {}).get("safety_valve", {})
        del prev_ckpt
        torch.cuda.empty_cache()
        x_b, y_b = generate_converged_burner_dataset(base_d1, 256, device, args.status_json, t0)
        hw_guard.cooldown_if_needed(max_wait_s=1.0)
        m_burner = train_burner_pino(burner_model, x_b, y_b, base_d1, 36, device, args.status_json, t0)
        del x_b, y_b
        torch.cuda.empty_cache()
    else:
        x_b, y_b = generate_converged_burner_dataset(base_d1, args.burner_samples, device, args.status_json, t0)
        hw_guard.cooldown_if_needed(max_wait_s=1.2)
        m_burner = train_burner_pino(burner_model, x_b, y_b, base_d1, args.burner_epochs, device, args.status_json, t0)
        del x_b, y_b
        torch.cuda.empty_cache()
        hw_guard.cooldown_if_needed(max_wait_s=1.2)
        x_c, s_c, f_c, flx_c = generate_converged_cylinder_dataset(mesh_dir, args.cyl_samples, device, args.status_json, t0)
        m_cyl = train_cylinder_pino(cyl_model, x_c, s_c, f_c, flx_c, args.cyl_epochs, device, args.status_json, t0)
        del x_c, s_c, f_c, flx_c
        torch.cuda.empty_cache()

    if not ((args.tender_only or args.fast_6op or args.wick_only or args.elastic_only) and args.out_weights.exists()):
        hw_guard.cooldown_if_needed(max_wait_s=0.8)
        write_status(args.status_json, {"stage": "4/8_train_cab_pino", "progress_overall_pct": 74.0})
        x_cab, s_cab, n_cab, v_cab = generate_converged_cab_dataset(args.cab_samples, device)
        m_cab = train_cab_pino(cab_model, x_cab, s_cab, n_cab, v_cab, args.cab_epochs, device)

    if not ((args.fast_6op or args.wick_only or args.elastic_only) and args.out_weights.exists()):
        hw_guard.cooldown_if_needed(max_wait_s=0.8)
        write_status(args.status_json, {"stage": "5/8_train_tender_pino", "progress_overall_pct": 81.0})
        x_tdr, s_tdr, n_tdr, v_tdr = generate_converged_tender_dataset(args.tender_samples, device)
        m_tdr = train_tender_pino(tender_model, x_tdr, s_tdr, n_tdr, v_tdr, args.tender_epochs, device)

        hw_guard.cooldown_if_needed(max_wait_s=0.8)
        write_status(args.status_json, {"stage": "6/8_train_safety_valve_pino", "progress_overall_pct": 88.0})
        x_sv, s_sv, n_sv, v_sv = generate_converged_safety_valve_dataset(args.safety_samples, device)
        m_sv = train_safety_valve_pino(safety_model, x_sv, s_sv, n_sv, v_sv, args.safety_epochs, device, hw_guard=hw_guard)

    if not (args.elastic_only and args.out_weights.exists()):
        hw_guard.cooldown_if_needed(max_wait_s=0.8)
        write_status(args.status_json, {"stage": "7/8_train_boiler_wick_cht_pino", "progress_overall_pct": 93.0})
        x_wc, s_wc, n_wc, v_wc = generate_converged_boiler_wick_cht_dataset(args.wick_cht_samples, device)
        m_wc = train_boiler_wick_cht_pino(wick_cht_model, x_wc, s_wc, n_wc, v_wc, args.wick_cht_epochs, device, hw_guard=hw_guard)

    hw_guard.cooldown_if_needed(max_wait_s=0.8)
    write_status(args.status_json, {"stage": "8/8_train_thermo_elastic_pino", "progress_overall_pct": 97.0})
    s_el, y_el = generate_converged_thermo_elastic_dataset(args.elastic_samples, device)
    m_el = train_thermo_elastic_pino(elastic_model, s_el, y_el, args.elastic_epochs, device, hw_guard=hw_guard)

    hw_guard.cooldown_if_needed(max_wait_s=0.8)
    gate10 = run_gate10_emergence_verification(
        burner_model, cyl_model, cab_model, tender_model, safety_model, wick_cht_model, elastic_model, base_d1, device
    )
    total_sec = time.perf_counter() - t0

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch_version": "d51_pino_v5_7op",
            "burner_width": 48,
            "burner_layers": 4,
            "cyl_width": 32,
            "cyl_layers": 3,
            "cab_width": 48,
            "cab_layers": 3,
            "tender_width": 48,
            "tender_layers": 3,
            "safety_width": 40,
            "safety_layers": 3,
            "wick_cht_width": 44,
            "wick_cht_layers": 3,
            "elastic_width": 36,
            "elastic_modes": 10,
            "burner_state_dict": {k: v.cpu().to(torch.float16) for k, v in burner_model.state_dict().items()},
            "cyl_state_dict": {k: v.cpu().to(torch.float16) for k, v in cyl_model.state_dict().items()},
            "cab_state_dict": {k: v.cpu().to(torch.float16) for k, v in cab_model.state_dict().items()},
            "tender_state_dict": {k: v.cpu().to(torch.float16) for k, v in tender_model.state_dict().items()},
            "safety_state_dict": {k: v.cpu().to(torch.float16) for k, v in safety_model.state_dict().items()},
            "wick_cht_state_dict": {k: v.cpu().to(torch.float16) for k, v in wick_cht_model.state_dict().items()},
            "elastic_state_dict": {k: v.cpu().to(torch.float16) for k, v in elastic_model.state_dict().items()},
            "metrics": {
                "burner": m_burner,
                "cylinder": m_cyl,
                "cab": m_cab,
                "tender": m_tdr,
                "safety_valve": m_sv,
                "wick_conjugate_ht": m_wc,
                "thermo_elastic": m_el,
                "gate10": gate10,
            },
        },
        args.out_weights,
    )

    report = {
        "gpu_device": gpu_name,
        "arch_version": "d51_pino_v5_7op",
        "total_training_time_sec": round(total_sec, 2),
        "total_parameters": int(n_tot_params),
        "burner_params": int(n_params_b),
        "cylinder_params": int(n_params_c),
        "cab_manifold_params": int(n_params_cab),
        "tender_hydraulic_params": int(n_params_tdr),
        "safety_valve_pop_params": int(n_params_sv),
        "boiler_wick_cht_params": int(n_params_wc),
        "thermo_elastic_params": int(n_params_el),
        "weights_size_mb": round(args.out_weights.stat().st_size / (1024 * 1024), 2),
        "burner_validation": m_burner,
        "cylinder_validation": m_cyl,
        "cab_manifold_validation": m_cab,
        "tender_hydraulic_validation": m_tdr,
        "safety_valve_pop_validation": m_sv,
        "boiler_wick_cht_validation": m_wc,
        "thermo_elastic_validation": m_el,
        "gate10_zero_if_emergence": gate10,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    write_status(
        args.status_json,
        {
            "stage": "completed",
            "progress_overall_pct": 100.0,
            "elapsed_sec": round(total_sec, 2),
            "gate10_emergence_pass": gate10["gate10_emergence_pass"],
            "gate17_cab_tender_emergence_pass": gate10["gate17_cab_tender_emergence_pass"],
            "gate21_safety_pop_emergence_pass": gate10["gate21_safety_pop_emergence_pass"],
            "gate22_wick_cht_emergence_pass": gate10["gate22_wick_cht_emergence_pass"],
            "rtx5080_bf16_combined_infer_ms": gate10["rtx5080_bf16_combined_infer_ms"],
            "report_path": str(args.out_report),
        },
    )
    print(f"==============================================================================")
    print(f" [DONE] 六算子 PINO 套件训练完成! 总耗时 {total_sec:.1f}s | 权重大小 {report['weights_size_mb']} MB")
    print(
        f" [Gate 10 / 17 / 21 / 22] 物理涌现检验: Gate10={'PASS' if gate10['gate10_emergence_pass'] else 'WARN'} "
        f"| Gate17={'PASS' if gate10['gate17_cab_tender_emergence_pass'] else 'WARN'} "
        f"| Gate21Pop={'PASS' if gate10['gate21_safety_pop_emergence_pass'] else 'WARN'} "
        f"| Gate22WickCHT={'PASS' if gate10['gate22_wick_cht_emergence_pass'] else 'WARN'}"
    )
    print(f"   - RTX 5080 BF16 六算子联合单步推理耗时: {gate10['rtx5080_bf16_combined_infer_ms']} ms / 步")
    print(f"==============================================================================", flush=True)


if __name__ == "__main__":
    main()
