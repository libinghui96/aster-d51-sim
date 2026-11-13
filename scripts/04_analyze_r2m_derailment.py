#!/usr/bin/env python3
"""
scripts/04_analyze_r2m_derailment.py
对比先导台车支点 (20-6) 在 R2m 孔位与 R3m 孔位 (相差 10mm) 下通过 R=2.0m 弯道时的动力学表现：
  1. 外侧钢轨轮缘侧向导向力 Y (N)
  2. Nadal 脱线系数 Y/Q (安全阈值 < 0.80)
  3. 先导台车水平偏转角 (deg) 与第 1 动轮轴箱横动量 (mm)
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.run_cosim import CoSimRunner


@dataclass
class CurveMetrics:
    time_s: List[float]
    speed_kmh: List[float]
    lead_flange_y_n: List[float]
    drv1_flange_y_n: List[float]
    derail_quotient_yq: List[float]
    lead_yaw_deg: List[float]
    drv1_lat_mm: List[float]


def build_pivot_variant_xml(base_xml_path: Path, pivot_mode: str, tmp_dir: Path) -> Path:
    """基于 d51_r2m.xml 派生 R2m 与 R3m 两种先导支点 (20-6) 构型，自动保持 meshdir 绝对路径正确。"""
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        mesh_abs = (base_xml_path.parent / compiler.get("meshdir")).resolve()
        compiler.set("meshdir", str(mesh_abs))

    lead_arm = root.find(".//body[@name='leading_truck_arm']")
    lead_ws = root.find(".//body[@name='leading_wheelset']")
    j_yaw = root.find(".//joint[@name='j_leading_yaw']")
    if lead_arm is None or lead_ws is None or j_yaw is None:
        raise RuntimeError("XML 中未找到 leading_truck_arm，请确认已生成 models/d51_r2m.xml")

    lead_axle_x = float(lead_arm.attrib["pos"].split()[0]) + float(lead_ws.attrib["pos"].split()[0])
    arm_pos = [float(v) for v in lead_arm.attrib["pos"].split()]
    if pivot_mode == "R2m":
        arm_len, yaw_lim, yaw_stiff = 0.052, 0.28, 0.35
    else:
        arm_len, yaw_lim, yaw_stiff = 0.062, 0.090, 0.65

    arm_pos[0] = lead_axle_x - arm_len
    lead_arm.set("pos", f"{arm_pos[0]:.6f} {arm_pos[1]:.6f} {arm_pos[2]:.6f}")
    lead_ws.set("pos", f"{arm_len:.6f} 0 0")
    j_yaw.set("range", f"{-yaw_lim:.6f} {yaw_lim:.6f}")
    j_yaw.set("stiffness", f"{yaw_stiff:.4f}")

    # 同步调整摆臂可视化几何长度 (若存在)
    arm_bar = lead_arm.find("geom[@name='lead_arm_bar']")
    if arm_bar is not None:
        arm_bar.set("pos", f"{0.5 * arm_len:.6f} 0 0")
        old_sz = arm_bar.attrib["size"].split()
        arm_bar.set("size", f"{0.5 * arm_len:.6f} {old_sz[1]} {old_sz[2]}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_xml = tmp_dir / f"d51_pivot_{pivot_mode}.xml"
    tree.write(out_xml, encoding="utf-8")
    return out_xml


def extract_wheelset_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    loco_body_id: int,
    geom_ids_lead_L: set,
    geom_ids_drv1_L: set,
) -> Tuple[float, float, float]:
    rot_mat = data.xmat[loco_body_id].reshape(3, 3)
    e_lat = rot_mat[:, 1]
    e_vert = np.array([0.0, 0.0, 1.0])

    y_lead, q_lead, y_drv1 = 0.0, 0.0, 0.0
    c_wrench = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        is_lead = (g1 in geom_ids_lead_L) or (g2 in geom_ids_lead_L)
        is_drv1 = (g1 in geom_ids_drv1_L) or (g2 in geom_ids_drv1_L)
        if not (is_lead or is_drv1):
            continue

        mujoco.mj_contactForce(model, data, i, c_wrench)
        frame = con.frame.reshape(3, 3)
        f_world = frame.T @ c_wrench[:3]
        sign = -1.0 if (g1 in geom_ids_lead_L or g1 in geom_ids_drv1_L) else 1.0
        f_on_wheel = sign * f_world

        lat_mag = abs(float(np.dot(f_on_wheel, e_lat)))
        vert_mag = max(0.0, float(np.dot(f_on_wheel, e_vert)))

        if is_lead:
            y_lead += lat_mag
            q_lead += vert_mag
        if is_drv1:
            y_drv1 += lat_mag

    return y_lead, q_lead, y_drv1


def run_curve_trial(xml_path: Path, duration_s: float = 8.0) -> CurveMetrics:
    runner = CoSimRunner(xml_path, preset="hot")
    m, d = runner.model, runner.data

    runner.ctrl.reverser = 0.78
    runner.ctrl.regulator = 0.0

    lead_geoms_L = {
        gid for gid in range(m.ngeom)
        if (m.geom(gid).name or "").startswith("lead_") and (m.geom(gid).name or "").endswith("_L")
    }
    drv1_geoms_L = {
        gid for gid in range(m.ngeom)
        if (m.geom(gid).name or "").startswith("drv_1_") and (m.geom(gid).name or "").endswith("_L")
    }

    qpos_lead_yaw = m.joint("j_leading_yaw").qposadr[0]
    qpos_drv1_lat = m.joint("j_lat_y1").qposadr[0]
    loco_id = m.body("loco_frame").id

    n_steps = int(round(duration_s / m.opt.timestep))
    metrics = CurveMetrics([], [], [], [], [], [], [])
    win_y_lead, win_q_lead, win_y_drv1 = [], [], []

    for step_i in range(n_steps):
        t = step_i * m.opt.timestep
        runner.ctrl.regulator = min(0.45, 0.45 * (t / 1.5))
        runner.ctrl.drain_cock = (t < 1.5)

        tel = runner.step_once()
        yl, ql, yd1 = extract_wheelset_forces(m, d, loco_id, lead_geoms_L, drv1_geoms_L)
        win_y_lead.append(yl)
        win_q_lead.append(ql)
        win_y_drv1.append(yd1)

        if len(win_y_lead) > 80:
            win_y_lead.pop(0)
            win_q_lead.pop(0)
            win_y_drv1.pop(0)

        if step_i % 40 == 0 and t >= 0.4:
            avg_yl = float(np.mean(win_y_lead))
            avg_ql = max(0.35, float(np.mean(win_q_lead)))
            avg_yd1 = float(np.mean(win_y_drv1))
            yq_ratio = (avg_yl + 0.5 * avg_yd1) / avg_ql

            metrics.time_s.append(t)
            metrics.speed_kmh.append(tel.scale_speed_kmh)
            metrics.lead_flange_y_n.append(avg_yl)
            metrics.drv1_flange_y_n.append(avg_yd1)
            metrics.derail_quotient_yq.append(yq_ratio)
            metrics.lead_yaw_deg.append(math.degrees(float(d.qpos[qpos_lead_yaw])))
            metrics.drv1_lat_mm.append(float(d.qpos[qpos_drv1_lat]) * 1000.0)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Aster D51 20-6 支点孔位 R2m vs R3m 过弯脱线裕度分析")
    parser.add_argument("--xml", type=Path, default=Path("models/d51_r2m.xml"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--out-plot", type=Path, default=Path("logs/r2m_vs_r3m_derailment.png"))
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"未找到 {args.xml}，请先运行: python scripts/03_generate_mjcf.py --mode curve_r2m --out models/d51_r2m.xml"
        )

    tmp_dir = Path("logs/tmp_variants")
    results: Dict[str, CurveMetrics] = {}

    for mode in ("R2m", "R3m"):
        print(f"[ANALYZE] 正在仿真 R=2.0m 环形弯道工况 -> 先导支点 (20-6) 孔位: {mode} ...", flush=True)
        var_xml = build_pivot_variant_xml(args.xml, mode, tmp_dir)
        results[mode] = run_curve_trial(var_xml, duration_s=args.duration)

    print("\n" + "=" * 78)
    print("  20-6 先导支点孔位在 R=2.0m 弯道上的稳态动力学对比 (取后半程均值/峰值)")
    print("=" * 78)
    for mode, res in results.items():
        half = len(res.time_s) // 2
        y_d1_mean = np.mean(res.drv1_flange_y_n[half:])
        yq_peak = np.max(res.derail_quotient_yq[half:])
        lat_peak = np.max(np.abs(res.drv1_lat_mm[half:]))
        spd_end = res.speed_kmh[-1]
        status = "SAFE (余量充足)" if yq_peak < 0.80 else ("MARGINAL (轮缘硬顶)" if yq_peak < 1.20 else "DANGER (爬轨风险!)")
        print(
            f"  [{mode} 孔位] 末速={spd_end:4.1f} km/h | 第1动轮侧向力={y_d1_mean:4.2f} N | "
            f"第1轴横动={lat_peak:4.2f} mm | 峰值 Y/Q={yq_peak:4.2f} -> {status}"
        )
    print("=" * 78)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    colors = {"R2m": "tab:blue", "R3m": "tab:red"}

    for mode in ("R2m", "R3m"):
        r = results[mode]
        lbl = f"20-6 Hole: {mode}"
        axes[0].plot(r.time_s, r.derail_quotient_yq, color=colors[mode], lw=1.8, label=lbl)
        axes[1].plot(r.time_s, r.drv1_flange_y_n, color=colors[mode], lw=1.6, label=f"{lbl} (Drive Axle 1 Flange Y)")
        axes[2].plot(r.time_s, r.lead_yaw_deg, color=colors[mode], lw=1.6, label=lbl)
        axes[3].plot(r.time_s, r.drv1_lat_mm, color=colors[mode], lw=1.6, label=lbl)

    axes[0].axhline(0.80, color="orange", ls="--", lw=1.2, label="Railway Safe Limit (Y/Q = 0.80)")
    axes[0].set_ylabel("Derailment Y/Q")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Axle 1 Lateral (N)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Lead Yaw (deg)")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    axes[3].axhline(1.0, color="gray", ls=":", label="8.5mm Axlebox Extra Play (±1.0mm)")
    axes[3].axhline(-1.0, color="gray", ls=":")
    axes[3].set_ylabel("Axle 1 Shift (mm)")
    axes[3].set_xlabel("Time on R=2.0m Curve (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Aster D51 R=2.0m Curve Negotiation: 20-6 Pivot R2m vs. R3m Mounting Hole")
    fig.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=180)
    plt.close(fig)
    print(f"[DONE] 对比分析曲线已保存至: {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()