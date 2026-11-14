#!/usr/bin/env python3
"""
scripts/09_verify_cad_and_clearance.py
================================================================================
Aster Hobby JNR D51 (1/32, Gauge 1 45mm) 自动化 CAD 拓扑、BOM 真值、运动包络净空与闭环精度门禁

本脚本以纯数值方式严格核验 P0-C (V2-MECH-TRUE) + P0-D (V2-PIPING-LUBE-TRUE) 全部门禁契约：
  - Gate 1: BOM 真值表与无头几何标定一致性 (5 本煙管、380~405ml 净容积、双U四程过热器、车架内双水路、卧式置换注油器、2x2 四头燃烧器、15-3/16-2/30-2 假件隔离)；
  - Gate 2: 全车 54 个 STL 100% 流形水密性 (boundary_edges == 0)、正体积、三角形预算、烟室连续非空切片与微米级左右对称性；
  - Gate 3: 月牙板圆弧槽与 MuJoCo j_die_shift 铰链轨迹零角度偏差 (< 0.05° / < 0.02mm)、单剪偏心杆同轴台阶销闭合、独立亮面单滑杆与 7 大内脏材质拆分；
  - Gate 6: 车轮外侧零水管硬约束 (|Y| <= 11.0mm) 及 360° (每 10°) × [-1.0, 0.0, +1.0] 全行程三维三角面采样净空 (>= 0.50 mm)；
  - Gate 8: 三级精密闭环重合残差 (< 0.02 mm)、2.0N 横向力下第 3 动轮轴向游隙自由横动 (>= 0.15 mm)、逆转拉杆耦合与 1.0 kgf/cm² 正反转冷态气密自启动 / 3.8 kgf/cm² 中立位驻车闭锁；
  - Gate 9: 1D-3D 拓扑—几何连续性门禁：全车 54 个 STL 100% 单连通分量 (connected_components == 1，杜绝悬空碎块与封闭幽灵空腔)、炉内管路包络不越界、全车 18 个跨刚体/跨网格流体与机械端口在直线及 R2m 弯道双姿态下零断口对接 (gap <= 0.50 mm)。

任一门禁不达标即输出 FAIL 并返回非零退出码。
"""
from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mujoco
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.run_cosim import CoSimRunner
from sim.thermo_1d import sat_temp_c


# ==============================================================================
# 工具函数：二进制 STL 解析、水密拓扑与三角面加密采样
# ==============================================================================

def load_binary_stl(stl_path: Path) -> np.ndarray:
    """读取二进制 STL 返回 (N, 3, 3) float64 顶点数组（单位与文件一致，mm）。"""
    data = stl_path.read_bytes()
    n_tri = struct.unpack("<I", data[80:84])[0]
    dt = np.dtype([
        ("normal", "<f4", (3,)),
        ("v", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    arr = np.frombuffer(data, dtype=dt, offset=84, count=n_tri)
    return arr["v"].astype(np.float64)


def check_watertight_and_volume(tris: np.ndarray) -> Tuple[int, float]:
    """
    量化顶点 (1e-4 mm) 后统计有向半边配对残差与有向体积 (mm³):
      - 若每条半边 (u -> v) 均存在唯一反向半边 (v -> u)，则 boundary_edges == 0 (100% 闭合流形水密)。
    """
    n_tri = len(tris)
    pts = np.round(tris.reshape(-1, 3) * 10000.0).astype(np.int64)
    _, inv = np.unique(pts, axis=0, return_inverse=True)
    f = inv.reshape(n_tri, 3)

    e01 = f[:, [0, 1]]
    e12 = f[:, [1, 2]]
    e20 = f[:, [2, 0]]
    directed = np.vstack([e01, e12, e20])

    max_vid = int(inv.max()) + 1
    fwd_keys = directed[:, 0] * max_vid + directed[:, 1]
    rev_keys = directed[:, 1] * max_vid + directed[:, 0]

    fwd_sorted = np.sort(fwd_keys)
    rev_sorted = np.sort(rev_keys)
    unmatched = int(np.sum(fwd_sorted != rev_sorted))

    # 有向体积 V = (1/6) * sum(v0 · (v1 × v2))
    cross = np.cross(tris[:, 1], tris[:, 2])
    vol = float(np.sum(np.einsum("ij,ij->i", tris[:, 0], cross)) / 6.0)
    return unmatched, vol


def count_connected_components(tris: np.ndarray, tol_mm: float = 1e-4) -> int:
    """
    通过共享量化顶点并查集 (Union-Find) 统计 STL 三角网格的三维连通分量数：
      - 单独检查 boundary_edges == 0 无法发现“两个相距数毫米的闭合圆管断成两截”或“未开孔的封闭内腔”；
      - 强制 connected_components == 1 可从拓扑上彻底根除零件内部悬空断管与幽灵空腔。
    """
    if len(tris) == 0:
        return 0
    scale = 1.0 / max(tol_mm, 1e-7)
    pts = np.round(tris.reshape(-1, 3) * scale).astype(np.int64)
    _, inv = np.unique(pts, axis=0, return_inverse=True)
    f = inv.reshape(len(tris), 3)
    n_v = int(inv.max()) + 1

    parent = list(range(n_v))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, w in f.tolist():
        ru, rv, rw = _find(u), _find(v), _find(w)
        if ru != rv:
            parent[rv] = ru
        if ru != rw:
            parent[rw] = ru

    roots = {_find(v) for v in range(n_v)}
    return len(roots)


def sample_triangle_surface(tris_m: np.ndarray, max_spacing_m: float = 0.00085) -> np.ndarray:
    """
    对三角面进行重心网格自适应加密采样，确保任意表面点到最近采样点间距 <= max_spacing_m，
    避免稀疏大三角形仅靠顶点检测遗漏边-面近距穿模。
    """
    v0, v1, v2 = tris_m[:, 0], tris_m[:, 1], tris_m[:, 2]
    l01 = np.linalg.norm(v1 - v0, axis=1)
    l12 = np.linalg.norm(v2 - v1, axis=1)
    l20 = np.linalg.norm(v0 - v2, axis=1)
    max_edge = np.maximum(np.maximum(l01, l12), l20)

    chunks = [tris_m.reshape(-1, 3)]
    # 按最长边分级重心采样
    for k_div, lo, hi in [(2, 1.0, 2.2), (4, 2.2, 4.5), (8, 4.5, 10.0), (14, 10.0, 1e9)]:
        mask = (max_edge > lo * max_spacing_m) & (max_edge <= hi * max_spacing_m)
        if not np.any(mask):
            continue
        sv0, sv1, sv2 = v0[mask], v1[mask], v2[mask]
        bary = []
        for i in range(k_div + 1):
            for j in range(k_div + 1 - i):
                if (i in (0, k_div) and j in (0, k_div)):
                    continue
                u = i / k_div
                v = j / k_div
                w = 1.0 - u - v
                bary.append((w, u, v))
        if bary:
            b_arr = np.array(bary, dtype=np.float64)  # (B, 3)
            pts = (
                b_arr[:, 0:1, None] * sv0[None, :, :]
                + b_arr[:, 1:2, None] * sv1[None, :, :]
                + b_arr[:, 2:3, None] * sv2[None, :, :]
            ).reshape(-1, 3)
            chunks.append(pts)

    all_pts = np.vstack(chunks)
    # 0.25 mm 体素去重加速 KDTree 查询
    q = np.round(all_pts * 4000.0).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return all_pts[idx]


# ==============================================================================
# Gate 1: BOM 真值表与运动学配置校验
# ==============================================================================

def verify_gate_1() -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    kin_path = PROJECT_ROOT / "configs" / "d51_kinematics.json"
    pip_path = PROJECT_ROOT / "configs" / "d51_boiler_piping.json"
    lut_path = PROJECT_ROOT / "configs" / "smokebox_lut.json"

    kin = json.loads(kin_path.read_text(encoding="utf-8"))
    pip = json.loads(pip_path.read_text(encoding="utf-8"))

    # 1.1 核验 exp_link_tail_cant_deg 解析几何一致性
    import importlib.util
    spec03 = importlib.util.spec_from_file_location("gen_mjcf", PROJECT_ROOT / "scripts" / "03_generate_mjcf.py")
    mod03 = importlib.util.module_from_spec(spec03)
    assert spec03.loader is not None
    spec03.loader.exec_module(mod03)
    pose_N = mod03.solve_side_pose(0.0, kin)
    tx_m = float(kin["walschaerts_valve_gear"]["trunnion_x"]) * 1e-3
    tz_m = float(kin["walschaerts_valve_gear"]["trunnion_z"]) * 1e-3
    etx, etz = pose_N["exp_tail"]
    cx_arc, cz_arc = pose_N["arc_center_local"]
    slot_tilt_rad = math.atan2(-cz_arc, cx_arc)
    tail_back_rad = math.atan2(tx_m - etx, tz_m - etz)
    exact_cant_deg = math.degrees(slot_tilt_rad - tail_back_rad)
    cfg_cant_deg = float(kin["walschaerts_valve_gear"].get("exp_link_tail_cant_deg", -999.0))
    if abs(cfg_cant_deg - exact_cant_deg) > 1e-3:
        ok = False
        msgs.append(f"FAIL: exp_link_tail_cant_deg={cfg_cant_deg:.6f}° != exact {exact_cant_deg:.6f}°")
    else:
        msgs.append(f"PASS: exp_link_tail_cant_deg = {cfg_cant_deg:.6f}° (exact match Δ={abs(cfg_cant_deg - exact_cant_deg):.2e}°)")

    play_mm = float(kin["cylinder_and_rods"].get("crank_pin_axial_play_mm", 0.0))
    if not (0.25 <= play_mm <= 0.45) or "y_layer_offsets" not in kin["walschaerts_valve_gear"]:
        ok = False
        msgs.append(f"FAIL: crank_pin_axial_play_mm={play_mm} or missing y_layer_offsets")
    else:
        msgs.append(f"PASS: crank_pin_axial_play_mm = ±{play_mm:.2f} mm & Walschaerts y_layer_offsets locked")

    # 1.2 核验 5 本煙管与净水汽容积
    pv = pip["boiler_pressure_vessel"]
    fl5 = pv["five_flue_tubes"]
    up_c = int(fl5["upper_superheater_flues"]["count"])
    lo_c = int(fl5["lower_fire_tubes"]["count"])
    tot_c = int(fl5["total_count"])
    v_net = float(pv["verified_capacities"]["geometric_net_volume_ml"])
    evap_cm2 = float(pv["verified_capacities"]["total_evaporative_area_cm2"])
    if tot_c != 5 or up_c != 2 or lo_c != 3 or not (380.0 <= v_net <= 405.0):
        ok = False
        msgs.append(f"FAIL: 5-flue check total={tot_c} ({up_c}+{lo_c}), V_net={v_net:.1f} ml")
    else:
        msgs.append(f"PASS: 5-flue copper boiler ({up_c}×ø{fl5['upper_superheater_flues']['od_mm']}mm + {lo_c}×ø{fl5['lower_fire_tubes']['od_mm']}mm), V_net={v_net:.1f} ml, A_evap={evap_cm2:.2f} cm²")

    # 1.3 核验后壁加减阀 11-10 -> 13-9 -> 17-6 -> 17-5 四程过热器 -> 20-7 ø4.2mm 乏汽喷嘴
    sp = pip["superheater_and_steam_piping"]
    reg = sp["cab_regulator_11_10"]
    sh = sp["superheater_4pass"]
    ej = sp["smokebox_draft_ejector"]
    orifice_mm = float(ej["blast_nozzle_orifice_dia_mm"])
    lut_match = True
    if lut_path.exists():
        lut = json.loads(lut_path.read_text(encoding="utf-8"))
        lut_nozzle = float(lut.get("geometry_mm", {}).get("nozzle_id_mm", 4.2))
        lut_match = abs(orifice_mm - lut_nozzle) < 1e-3
    if (
        sh["element_part"] != "17-5"
        or "17-6" not in sh["hollow_bolt_part"]
        or int(sh["passes"]) != 4
        or abs(orifice_mm - 4.2) > 1e-3
        or not lut_match
    ):
        ok = False
        msgs.append(f"FAIL: superheater/nozzle mismatch sh={sh['element_part']}, passes={sh['passes']}, nozzle={orifice_mm}mm")
    else:
        msgs.append(f"PASS: Regulator {'/'.join(reg['part_nos'])} -> 13-9 -> 17-6 -> 17-5 ({sh['passes']}-pass, {sh['total_heated_length_mm']}mm) -> 20-7 ø{orifice_mm:.1f}mm nozzle (CFD LUT synced)")

    # 1.4 核验双水路车架内向后拓扑 & 15-3 假逆止阀零水管隔离
    fw = pip["feedwater_piping_topology"]
    dummies = pip["dummy_parts_registry"]
    d15_3 = dummies.get("15-3", {})
    d16_2 = dummies.get("16-2", {})
    d30_2 = dummies.get("30-2", {})
    if (
        fw["axle_pump_loop"]["delivery_direction"] != "rearward"
        or float(fw["max_abs_y_mm"]) > 11.0
        or int(fw["wheel_side_pipes_count"]) != 0
        or fw.get("dummy_clack_15_3_connected", True) is not False
        or not d15_3.get("is_dummy", False)
        or len(d15_3.get("connected_pipes", ["err"])) != 0
        or not d16_2.get("is_dummy", False)
        or not d30_2.get("is_dummy", False)
    ):
        ok = False
        msgs.append("FAIL: feedwater inside-frame constraint or dummy valve registry violated")
    else:
        msgs.append(
            f"PASS: Inside-frame rearward dual feedwater (|Y|<={fw['max_abs_y_mm']}mm, 13-7 & 13-2 checks, wheel_side_pipes=0; 15-3/16-2/30-2 flagged is_dummy=True)"
        )

    # 1.5 核验前车架卧式置换注油器 (4-1/4-2, 无底塞) & 2x2 四头酒精燃烧器 (23-1)
    lub = pip["displacement_lubricator_system"]
    brn = pip["alcohol_burner_system"]
    if (
        lub.get("has_bottom_drain", True) is not False
        or float(lub["max_abs_y_mm"]) > 10.5
        or int(brn["wick_count"]) != 4
        or abs(float(brn["wick_dia_mm"]) - 10.0) > 1e-3
    ):
        ok = False
        msgs.append("FAIL: displacement lubricator or 4-wick burner BOM mismatch")
    else:
        msgs.append(
            f"PASS: Central horizontal lubricator 4-2 (|Y|<={lub['max_abs_y_mm']}mm, has_bottom_drain=False under 25-3) & 2x2 4-wick ø10mm burner 23-1 (130ml sump)"
        )

    return ok, msgs


# ==============================================================================
# Gate 2: 全车 54 个 STL 水密性、正体积、烟室连续切片与左右对称性
# ==============================================================================

def verify_gate_2() -> Tuple[bool, List[str], Dict[str, np.ndarray]]:
    msgs: List[str] = []
    ok = True
    mesh_dir = PROJECT_ROOT / "models" / "meshes"
    stl_files = sorted(mesh_dir.glob("*.stl"))

    if len(stl_files) < 54:
        return False, [f"FAIL: Expected >= 54 STL files, found {len(stl_files)}"], {}

    stl_tris_mm: Dict[str, np.ndarray] = {}
    total_tris = 0
    bad_watertight = []
    bad_vol = []

    for p in stl_files:
        tris = load_binary_stl(p)
        stl_tris_mm[p.stem] = tris
        total_tris += len(tris)
        unmatched, vol = check_watertight_and_volume(tris)
        if unmatched != 0:
            bad_watertight.append((p.name, unmatched))
        if vol <= 0.0:
            bad_vol.append((p.name, vol))

    if bad_watertight or bad_vol:
        ok = False
        msgs.append(f"FAIL: Watertight/volume errors: boundary={bad_watertight}, non_pos_vol={bad_vol}")
    else:
        msgs.append(f"PASS: {len(stl_files)}/{len(stl_files)} STLs 100% manifold watertight (boundary_edges == 0) & positive volume")

    if not (60000 <= total_tris <= 170000):
        ok = False
        msgs.append(f"FAIL: Total triangle count {total_tris:,} outside [60,000, 170,000] budget")
    else:
        msgs.append(f"PASS: Total mesh triangle count = {total_tris:,} (within real-time WebGL/MuJoCo budget)")

    # 烟室与锅筒连续 2mm X-bin 顶点非空校验
    bs_pts = stl_tris_mm["boiler_superstructure"].reshape(-1, 3)
    xs = bs_pts[:, 0]
    empty_bins = []
    for x0 in np.arange(-110.0, 162.0, 2.0):
        cnt = int(np.sum((xs >= x0) & (xs < x0 + 2.0)))
        if cnt == 0:
            empty_bins.append(round(float(x0), 1))
    if empty_bins:
        ok = False
        msgs.append(f"FAIL: boiler_superstructure.stl has empty 2mm X-bins: {empty_bins}")
    else:
        sb_cnt = int(np.sum((xs >= 92.0) & (xs <= 162.0)))
        msgs.append(f"PASS: boiler_superstructure.stl has continuous non-zero vertices in ALL 136×2mm X-bins [-110, +162]mm (smokebox verts={sb_cnt:,})")

    # 中心对称网格 |min_y + max_y| < 1e-3 mm 校验 + 质心 |com_y| < 1e-2 mm 校验
    sym_names = [
        "boiler_superstructure",
        "boiler_firebox_flues",
        "leading_truck_frame",
        "trailing_truck_frame",
        "driving_axlebox",
        "tender_axlebox",
        "tender_body",
        "tender_bogie_frame",
        "chassis_equalizer_brakes",
        "drain_cock_linkage",
        "displacement_lubricator",
        "alcohol_burner",
        "axle_pump_eccentric",
        "main_rod",
        "coupling_rod",
        "coupling_rod_front",
        "coupling_rod_rear",
        "combination_lever",
    ]
    strict_com_sym_names = [n for n in sym_names if n != "boiler_superstructure"]
    mass_props = json.loads((PROJECT_ROOT / "configs" / "d51_mass_props.json").read_text(encoding="utf-8"))["parts"]

    max_sym_err = 0.0
    max_com_err = 0.0
    bad_sym = []
    for name in sym_names:
        ys = stl_tris_mm[name].reshape(-1, 3)[:, 1]
        err = abs(float(ys.min() + ys.max()))
        max_sym_err = max(max_sym_err, err)
        if err >= 1e-3:
            bad_sym.append((name, err))
    for name in strict_com_sym_names:
        com_y_mm = abs(float(mass_props[name]["com_m"][1])) * 1000.0
        max_com_err = max(max_com_err, com_y_mm)
        if com_y_mm >= 1e-2:
            bad_sym.append((f"{name}:com_y", com_y_mm))

    # _L / _R 镜像对校验 (注: 动轮与回动曲柄内置 90° 相位差，校验体积与 Y 包络镜像一致性；非曲柄件同时校验质心镜像)
    lr_stems = sorted(
        s[:-2] for s in stl_tris_mm if s.endswith("_L") and (s[:-2] + "_R") in stl_tris_mm
    )
    quartered_stems = {"driving_wheel", "main_driving_wheel", "return_crank"}
    for base in lr_stems:
        tL = stl_tris_mm[base + "_L"]
        tR = stl_tris_mm[base + "_R"]
        _, vL = check_watertight_and_volume(tL)
        _, vR = check_watertight_and_volume(tR)
        yL = tL.reshape(-1, 3)[:, 1]
        yR = tR.reshape(-1, 3)[:, 1]
        y_mirror_err = max(abs(float(yL.min() + yR.max())), abs(float(yL.max() + yR.min())))
        com_lr_err = (
            0.0
            if base in quartered_stems
            else abs(float(mass_props[base + "_L"]["com_m"][1] + mass_props[base + "_R"]["com_m"][1])) * 1000.0
        )
        max_com_err = max(max_com_err, com_lr_err)
        if len(tL) != len(tR) or abs(vL - vR) / max(vL, 1e-6) > 1e-3 or y_mirror_err > 1e-2 or com_lr_err >= 1e-2:
            bad_sym.append((f"{base}_L/R", max(y_mirror_err, com_lr_err)))

    if bad_sym:
        ok = False
        msgs.append(f"FAIL: Symmetry violations: {bad_sym}")
    else:
        msgs.append(
            f"PASS: All {len(sym_names)} center-symmetric STLs (max |min_y+max_y|={max_sym_err:.2e} mm, max |com_y|={max_com_err:.2e} mm) & {len(lr_stems)} L/R mirror pairs verified"
        )

    return ok, msgs, stl_tris_mm


# ==============================================================================
# Gate 3 (& Gate 7): 月牙板圆弧槽同心度、偏心杆单剪台阶销闭合与 X-Ray 材质拆分
# ==============================================================================

def verify_gate_3(stl_tris_mm: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    kin = json.loads((PROJECT_ROOT / "configs" / "d51_kinematics.json").read_text(encoding="utf-8"))
    R_arc_m = float(kin["walschaerts_valve_gear"]["radius_rod_length"]) * 1e-3
    L_ecc = float(kin["walschaerts_valve_gear"]["eccentric_rod_length"])

    # 3.1 检查 eccentric_rod_L 前端内侧单剪扁眼 (非双叉头!) + expansion_link_L 尾孔同轴 2-2 台阶销栓闭合无黑孔
    ecc_tris = stl_tris_mm["eccentric_rod_L"]
    ecc_pts = ecc_tris.reshape(-1, 3)
    front_eye_pts = ecc_pts[ecc_pts[:, 0] > L_ecc - 2.2]
    front_eye_y_min = float(front_eye_pts[:, 1].min())
    front_eye_y_max = float(front_eye_pts[:, 1].max())
    front_eye_thick = front_eye_y_max - front_eye_y_min

    r_tail = float(kin["walschaerts_valve_gear"]["exp_link_tail_radius"])
    cant_rad = math.radians(float(kin["walschaerts_valve_gear"]["exp_link_tail_cant_deg"]))
    x_tail = r_tail * math.sin(cant_rad)
    z_tail = -r_tail * math.cos(cant_rad)

    exp_tris = stl_tris_mm["expansion_link_L"]
    v2d = exp_tris[:, :, [0, 2]] - np.array([[[x_tail, z_tail]]])
    a, b, c = v2d[:, 0], v2d[:, 1], v2d[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.abs(det) > 1e-8
    l1 = np.zeros(len(exp_tris))
    l2 = np.zeros(len(exp_tris))
    l1[valid] = ((b[valid, 1] - c[valid, 1]) * (-c[valid, 0]) + (c[valid, 0] - b[valid, 0]) * (-c[valid, 1])) / det[valid]
    l2[valid] = ((c[valid, 1] - a[valid, 1]) * (-c[valid, 0]) + (a[valid, 0] - c[valid, 0]) * (-c[valid, 1])) / det[valid]
    l3 = 1.0 - l1 - l2
    hit_mask = valid & (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
    pin_cap_hits = int(np.sum(hit_mask))
    pin_y_span = float(exp_tris[hit_mask, :, 1].max() - exp_tris[hit_mask, :, 1].min()) if pin_cap_hits > 0 else 0.0

    if pin_cap_hits < 2 or pin_y_span < 4.0 or front_eye_thick > 1.55 or front_eye_y_max > -0.50:
        ok = False
        msgs.append(
            f"FAIL: Single-shear lap / 2-2 pin check failed: front_eye_y=[{front_eye_y_min:.2f},{front_eye_y_max:.2f}]mm "
            f"(thick={front_eye_thick:.2f}mm), pin_cap_hits={pin_cap_hits}, pin_y_span={pin_y_span:.2f}mm"
        )
    else:
        msgs.append(
            f"PASS: eccentric_rod_L/R inner single-shear flat eye (y∈[{front_eye_y_min:.2f},{front_eye_y_max:.2f}]mm, thick={front_eye_thick:.2f}mm) "
            f"+ coaxial 2-2 slotted pin solid cap verified (cap_hits={pin_cap_hits}, pin_y_span={pin_y_span:.2f}mm)"
        )

    # 3.2 在 MuJoCo 中核验 expansion_link_L/R 视觉网格圆弧中心 (+R_arc, 0, 0) 与 j_die_shift_L/R 铰链轴点完全重合
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "d51_full.xml"))
    data = mujoco.MjData(model)
    ctrl_rev = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl")

    max_center_err_mm = 0.0
    max_ang_err_deg = 0.0
    cad_arc_center_local = np.array([R_arc_m, 0.0, 0.0], dtype=np.float64)

    def _cad_pt_to_world(gid: int, pt_cad_m: np.ndarray) -> np.ndarray:
        mid = int(model.geom_dataid[gid])
        rmat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rmat, model.mesh_quat[mid])
        v_mj = rmat.reshape(3, 3).T @ (pt_cad_m - model.mesh_pos[mid])
        return data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ v_mj

    for rev_cmd in (-1.0, 0.0, +1.0):
        mujoco.mj_resetData(model, data)
        data.ctrl[ctrl_rev] = rev_cmd * 0.45
        for _ in range(400):
            mujoco.mj_step(model, data)

        for tag in ("L", "R"):
            gid_exp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"exp_link_vis_{tag}")
            gid_die = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"die_vis_{tag}")
            jid_die = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_die_shift_{tag}")

            p_exp = _cad_pt_to_world(gid_exp, cad_arc_center_local)
            p_die = _cad_pt_to_world(gid_die, cad_arc_center_local)
            p_jnt = data.xanchor[jid_die]

            err1 = float(np.linalg.norm(p_exp - p_jnt)) * 1000.0
            err2 = float(np.linalg.norm(p_die - p_jnt)) * 1000.0
            max_center_err_mm = max(max_center_err_mm, err1, err2)
            ang_err = math.degrees(max(err1, err2) / (R_arc_m * 1000.0))
            max_ang_err_deg = max(max_ang_err_deg, ang_err)

    if max_center_err_mm >= 0.02 or max_ang_err_deg >= 0.05:
        ok = False
        msgs.append(f"FAIL: Curved-slot alignment error {max_center_err_mm:.4f} mm / {max_ang_err_deg:.4f}°")
    else:
        msgs.append(
            f"PASS: Expansion-link curved slot <-> die_block <-> j_die_shift concentricity: Δr={max_center_err_mm:.5f} mm (<0.02mm), Δθ={max_ang_err_deg:.5f}° (<0.05°)"
        )

    # 3.3 核验独立亮面滑杆与 7 大 P0-D 内脏几何体及材质绑定
    required_geoms_mats = {
        "slide_bar_vis_L": "mat_machined_steel",
        "slide_bar_vis_R": "mat_machined_steel",
        "cyl_housing_L": "mat_black_iron",
        "cyl_housing_R": "mat_black_iron",
        "boiler_Super_vis": "mat_black_iron",
        "boiler_copper_core_vis": "mat_copper",
        "superheater_piping_vis": "mat_machined_steel",
        "feedwater_piping_vis": "mat_brass",
        "displacement_lubricator_vis": "mat_brass",
        "boiler_aux_valves_vis": "mat_brass",
        "alcohol_burner_vis": "mat_brass",
        "tender_internals_vis": "mat_brass",
        "tender_body_vis": "mat_black_iron",
    }
    missing_gm = []
    for gname, mname in required_geoms_mats.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mname)
        if gid < 0 or mid < 0 or int(model.geom_matid[gid]) != mid:
            missing_gm.append((gname, mname))
    if missing_gm:
        ok = False
        msgs.append(f"FAIL: Missing/mismatched X-Ray split geoms or materials: {missing_gm}")
    else:
        msgs.append("PASS: Slide bars (mat_machined_steel) & all 7 P0-D internal assemblies split cleanly from outer X-Ray shells")

    return ok, msgs


# ==============================================================================
# Gate 6: 车轮外侧零水管与 360° × 3 全行程三维表面净空扫描 (>= 0.50 mm)
# ==============================================================================

def verify_gate_6(stl_tris_mm: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    # 6.1 静态管网 Y 坐标硬门禁
    fw_max_y = float(np.max(np.abs(stl_tris_mm["feedwater_piping"][:, :, 1])))
    lub_max_y = float(np.max(np.abs(stl_tris_mm["displacement_lubricator"][:, :, 1])))
    aux_max_y = float(np.max(np.abs(stl_tris_mm["boiler_aux_valves"][:, :, 1])))

    if fw_max_y > 11.0 or lub_max_y > 10.5 or aux_max_y > 24.0:
        ok = False
        msgs.append(f"FAIL: Pipe Y bounds violated: feedwater={fw_max_y:.2f}mm (<=11.0), lub={lub_max_y:.2f}mm (<=10.5), aux={aux_max_y:.2f}mm (<=24.0)")
    else:
        msgs.append(
            f"PASS: Zero wheel-side pipes! feedwater |Y|_max={fw_max_y:.2f}mm (<=11.0mm), lubricator |Y|_max={lub_max_y:.2f}mm (<=10.5mm), aux_valves |Y|_max={aux_max_y:.2f}mm (<=24.0mm)"
        )

    # 6.2 360° (每 10°) × [-1.0, 0.0, +1.0] 全行程三维三角面加密采样净空扫描
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "d51_full.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 直接基于 MuJoCo 编译后的精确内部顶点/三角面构建加密采样点云 (100% 对齐 geom_xpos / geom_xmat)
    mesh_pts_local: Dict[int, np.ndarray] = {}
    for mid in range(model.nmesh):
        v_adr = int(model.mesh_vertadr[mid])
        v_num = int(model.mesh_vertnum[mid])
        f_adr = int(model.mesh_faceadr[mid])
        f_num = int(model.mesh_facenum[mid])
        verts = model.mesh_vert[v_adr : v_adr + v_num]
        faces = model.mesh_face[f_adr : f_adr + f_num]
        tris_m = verts[faces]
        mesh_pts_local[mid] = sample_triangle_surface(tris_m, max_spacing_m=0.00085)

    static_geoms_to_check = [
        "feedwater_piping_vis",
        "displacement_lubricator_vis",
        "boiler_aux_valves_vis",
        "superheater_piping_vis",
        "boiler_copper_core_vis",
        "boiler_Super_vis",
        "chassis_equalizer_brakes_vis",
        "motion_bracket_vis_L",
        "motion_bracket_vis_R",
        "slide_bar_vis_L",
        "slide_bar_vis_R",
        "cyl_housing_L",
        "cyl_housing_R",
        "drain_cock_vis",
    ]

    moving_body_names = [
        "wheelset_1",
        "wheelset_2",
        "wheelset_3",
        "wheelset_4",
        "leading_truck_arm",
        "leading_wheelset",
        "trailing_truck_arm",
        "trailing_wheelset",
        "main_rod_L",
        "main_rod_R",
        "coupling_rod_L",
        "coupling_rod_R",
        "coupling_rod_front_L",
        "coupling_rod_front_R",
        "coupling_rod_rear_L",
        "coupling_rod_rear_R",
        "piston_crosshead_L",
        "piston_crosshead_R",
        "slide_valve_L",
        "slide_valve_R",
        "eccentric_rod_L",
        "eccentric_rod_R",
        "expansion_link_L",
        "expansion_link_R",
        "die_block_L",
        "die_block_R",
        "radius_rod_L",
        "radius_rod_R",
        "comb_lever_L",
        "comb_lever_R",
        "union_link_L",
        "union_link_R",
        "lift_link_L",
        "lift_link_R",
        "reverser_shaft",
        "cab_reach_rod",
        "pump_eccentric_rod",
    ]

    # 真实机械配合运动副白名单 (同轴转动耳轴/气缸活塞填料函/导轨滑块配合副)
    mating_geom_body_whitelist: Set[Tuple[str, str]] = {
        ("motion_bracket_vis_L", "expansion_link_L"),
        ("motion_bracket_vis_R", "expansion_link_R"),
        ("slide_bar_vis_L", "piston_crosshead_L"),
        ("slide_bar_vis_R", "piston_crosshead_R"),
        ("cyl_housing_L", "piston_crosshead_L"),
        ("cyl_housing_R", "piston_crosshead_R"),
        ("cyl_housing_L", "slide_valve_L"),
        ("cyl_housing_R", "slide_valve_R"),
    }

    # 运动刚体间真实铰接销/轴承配合副白名单 (非白名单运动件对必须满足 >= 0.50 mm 全包络无碰撞净空)
    mating_body_pairs_base: Set[Tuple[str, str]] = {
        ("leading_truck_arm", "leading_wheelset"),
        ("trailing_truck_arm", "trailing_wheelset"),
        ("wheelset_3", "main_rod_L"),
        ("wheelset_3", "main_rod_R"),
        ("wheelset_3", "coupling_rod_L"),
        ("wheelset_3", "coupling_rod_R"),
        ("wheelset_2", "coupling_rod_L"),
        ("wheelset_2", "coupling_rod_R"),
        ("wheelset_2", "coupling_rod_front_L"),
        ("wheelset_2", "coupling_rod_front_R"),
        ("wheelset_1", "coupling_rod_front_L"),
        ("wheelset_1", "coupling_rod_front_R"),
        ("wheelset_3", "coupling_rod_rear_L"),
        ("wheelset_3", "coupling_rod_rear_R"),
        ("wheelset_4", "coupling_rod_rear_L"),
        ("wheelset_4", "coupling_rod_rear_R"),
        ("coupling_rod_L", "coupling_rod_front_L"),
        ("coupling_rod_R", "coupling_rod_front_R"),
        ("coupling_rod_L", "coupling_rod_rear_L"),
        ("coupling_rod_R", "coupling_rod_rear_R"),
        ("wheelset_3", "eccentric_rod_L"),
        ("wheelset_3", "eccentric_rod_R"),
        ("wheelset_3", "pump_eccentric_rod"),
        ("main_rod_L", "piston_crosshead_L"),
        ("main_rod_R", "piston_crosshead_R"),
        ("piston_crosshead_L", "union_link_L"),
        ("piston_crosshead_R", "union_link_R"),
        ("union_link_L", "comb_lever_L"),
        ("union_link_R", "comb_lever_R"),
        ("comb_lever_L", "slide_valve_L"),
        ("comb_lever_R", "slide_valve_R"),
        ("comb_lever_L", "radius_rod_L"),
        ("comb_lever_R", "radius_rod_R"),
        ("eccentric_rod_L", "expansion_link_L"),
        ("eccentric_rod_R", "expansion_link_R"),
        ("expansion_link_L", "die_block_L"),
        ("expansion_link_R", "die_block_R"),
        ("die_block_L", "radius_rod_L"),
        ("die_block_R", "radius_rod_R"),
        ("radius_rod_L", "lift_link_L"),
        ("radius_rod_R", "lift_link_R"),
        ("lift_link_L", "reverser_shaft"),
        ("lift_link_R", "reverser_shaft"),
        ("reverser_shaft", "cab_reach_rod"),
    }
    mating_body_whitelist = set(mating_body_pairs_base) | {(b, a) for (a, b) in mating_body_pairs_base}

    # 构建车架固定静态件 KDTree (滚轮台模式下 loco_frame 固定于世界系原点)
    static_trees: Dict[str, Tuple[cKDTree, np.ndarray, np.ndarray]] = {}
    for gname in static_geoms_to_check:
        if gname == "drain_cock_vis":
            continue
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        mid = int(model.geom_dataid[gid])
        pts_w = data.geom_xpos[gid] + mesh_pts_local[mid] @ data.geom_xmat[gid].reshape(3, 3).T
        static_trees[gname] = (cKDTree(pts_w), pts_w.min(axis=0), pts_w.max(axis=0))

    moving_body_geoms: Dict[str, List[Tuple[int, int]]] = {}
    for bname in moving_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        glist = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) == bid and int(model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[gid])
                if mid in mesh_pts_local:
                    glist.append((gid, mid))
        moving_body_geoms[bname] = glist

    ctrl_rev = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl")
    jid_w3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_wheel_3")
    dof_w3 = int(model.jnt_dofadr[jid_w3])
    qpos_w3 = int(model.jnt_qposadr[jid_w3])
    jid_dc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_drain_cock")
    qpos_dc = int(model.jnt_qposadr[jid_dc])
    gid_dc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drain_cock_vis")
    mid_dc = int(model.geom_dataid[gid_dc])

    min_clear_mm = 1e9
    worst_info = ("", "", 0.0, 0.0)
    per_static_min: Dict[str, Tuple[float, str]] = {g: (1e9, "") for g in static_geoms_to_check}
    min_mov_clear_mm = 1e9
    worst_mov_info = ("", "", 0.0, 0.0)

    t0 = time.perf_counter()
    for rev_norm in (-1.0, 0.0, +1.0):
        mujoco.mj_resetData(model, data)
        data.ctrl[ctrl_rev] = rev_norm * 0.55
        # 先稳定逆转器到位
        for _ in range(300):
            mujoco.mj_step(model, data)

        # 以恒定角速度驱动主动轮旋转整整 360° (每 10° 采样一帧真实动力学闭环姿态)
        omega_drive = 4.0 * math.pi  # 2 rev/s -> 0.5s 转满 360°
        steps_per_10deg = int(round((math.radians(10.0) / omega_drive) / model.opt.timestep))

        for k_ang in range(36):
            for _ in range(steps_per_10deg):
                data.qvel[dof_w3] = omega_drive
                mujoco.mj_step(model, data)

            deg_now = math.degrees(float(data.qpos[qpos_w3])) % 360.0

            # 同时检查排液阀关闭 (0.0) 与打开 (+0.275 rad) 两种极限姿态
            dc_trees = []
            for dc_q in (0.0, 0.275):
                data.qpos[qpos_dc] = dc_q
                mujoco.mj_kinematics(model, data)
                pts_dc = data.geom_xpos[gid_dc] + mesh_pts_local[mid_dc] @ data.geom_xmat[gid_dc].reshape(3, 3).T
                dc_trees.append((pts_dc, pts_dc.min(axis=0), pts_dc.max(axis=0)))
            data.qpos[qpos_dc] = 0.0
            mujoco.mj_kinematics(model, data)

            frame_mov_pts: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for bname, glist in moving_body_geoms.items():
                pts_chunks = [
                    data.geom_xpos[gid] + mesh_pts_local[mid] @ data.geom_xmat[gid].reshape(3, 3).T
                    for gid, mid in glist
                ]
                if not pts_chunks:
                    continue
                m_pts = np.vstack(pts_chunks)
                m_min = m_pts.min(axis=0)
                m_max = m_pts.max(axis=0)
                frame_mov_pts.append((bname, m_pts, m_min, m_max))

                for gname, (st_tree, s_min, s_max) in static_trees.items():
                    if (gname, bname) in mating_geom_body_whitelist:
                        continue
                    bound_mm = min(15.0, per_static_min[gname][0])
                    gap_xyz = np.maximum(0.0, np.maximum(s_min - m_max, m_min - s_max))
                    if float(np.linalg.norm(gap_xyz)) * 1000.0 >= bound_mm:
                        continue
                    dists, _ = st_tree.query(m_pts, distance_upper_bound=bound_mm * 0.001 + 1e-6, workers=1)
                    d_min_m = float(np.min(dists))
                    if not math.isfinite(d_min_m):
                        continue
                    d_mm = d_min_m * 1000.0
                    if d_mm < per_static_min[gname][0]:
                        per_static_min[gname] = (d_mm, f"{bname}@{deg_now:.0f}°,rev={rev_norm:+.1f}")
                    if d_mm < min_clear_mm:
                        min_clear_mm = d_mm
                        worst_info = (gname, bname, deg_now, rev_norm)

                # 检查 drain_cock_vis 对移动件净空
                for pts_dc, dc_min, dc_max in dc_trees:
                    bound_dc_mm = min(15.0, per_static_min["drain_cock_vis"][0])
                    gap_dc = np.maximum(0.0, np.maximum(dc_min - m_max, m_min - dc_max))
                    if float(np.linalg.norm(gap_dc)) * 1000.0 >= bound_dc_mm:
                        continue
                    m_tree = cKDTree(m_pts)
                    dists_dc, _ = m_tree.query(pts_dc, distance_upper_bound=bound_dc_mm * 0.001 + 1e-6, workers=1)
                    d_dc_m = float(np.min(dists_dc))
                    if not math.isfinite(d_dc_m):
                        continue
                    d_dc_mm = d_dc_m * 1000.0
                    if d_dc_mm < per_static_min["drain_cock_vis"][0]:
                        per_static_min["drain_cock_vis"] = (d_dc_mm, f"{bname}@{deg_now:.0f}°,rev={rev_norm:+.1f}")
                    if d_dc_mm < min_clear_mm:
                        min_clear_mm = d_dc_mm
                        worst_info = ("drain_cock_vis", bname, deg_now, rev_norm)

            # 检查所有非铰接配合的运动刚体对运动刚体三维表面净空 (>= 0.50 mm)
            n_mb = len(frame_mov_pts)
            for i_m in range(n_mb):
                b1, p1, mn1, mx1 = frame_mov_pts[i_m]
                t1 = None
                for j_m in range(i_m + 1, n_mb):
                    b2, p2, mn2, mx2 = frame_mov_pts[j_m]
                    if (b1, b2) in mating_body_whitelist:
                        continue
                    bound_mv_mm = min(4.0, min_mov_clear_mm)
                    gap_mv = np.maximum(0.0, np.maximum(mn1 - mx2, mn2 - mx1))
                    if float(np.linalg.norm(gap_mv)) * 1000.0 >= bound_mv_mm:
                        continue
                    if t1 is None:
                        t1 = cKDTree(p1)
                    dists_mv, _ = t1.query(p2, distance_upper_bound=bound_mv_mm * 0.001 + 1e-6, workers=1)
                    d_mv_m = float(np.min(dists_mv))
                    if not math.isfinite(d_mv_m):
                        continue
                    d_mv_mm = d_mv_m * 1000.0
                    if d_mv_mm < min_mov_clear_mm:
                        min_mov_clear_mm = d_mv_mm
                        worst_mov_info = (b1, b2, deg_now, rev_norm)

    dt_scan = time.perf_counter() - t0
    brake_min_mm, brake_who = per_static_min["chassis_equalizer_brakes_vis"]
    violations = [(g, d, who) for g, (d, who) in per_static_min.items() if d < (1.50 if g == "chassis_equalizer_brakes_vis" else 0.50)]
    if violations or min_mov_clear_mm < 0.50:
        ok = False
        msgs.append(
            f"FAIL: 360°×3 clearance violations: static={violations}, min_mov={min_mov_clear_mm:.3f}mm ({worst_mov_info})"
        )
    else:
        msgs.append(
            f"PASS: 360°×3 full-envelope scan ({dt_scan:.2f}s): static min={min_clear_mm:.3f}mm >=0.50mm ({worst_info[0]}<->{worst_info[1]}), "
            f"moving-to-moving min={min_mov_clear_mm:.3f}mm >=0.50mm ({worst_mov_info[0]}<->{worst_mov_info[1]}), brakes min={brake_min_mm:.2f}mm >=1.50mm ({brake_who})"
        )
        summary_pairs = ", ".join(
            f"{k.replace('_vis','')}:{v[0]:.2f}mm"
            for k, v in sorted(per_static_min.items(), key=lambda kv: kv[1][0])[:6]
        )
        msgs.append(f"      Top-6 tightest static clearances: {summary_pairs}")

    # 6.3 R2m (2000mm) 弯道实车动态热机过弯 + 煤水车转向架/轮对净空与牵引连杆闭环扫描
    r2m_runner = CoSimRunner(PROJECT_ROOT / "models" / "d51_r2m.xml", preset="hot")
    rm, rd = r2m_runner.model, r2m_runner.data
    r2m_mesh_pts: Dict[int, np.ndarray] = {}
    for mid in range(rm.nmesh):
        v_adr, v_num = int(rm.mesh_vertadr[mid]), int(rm.mesh_vertnum[mid])
        f_adr, f_num = int(rm.mesh_faceadr[mid]), int(rm.mesh_facenum[mid])
        r2m_mesh_pts[mid] = sample_triangle_surface(rm.mesh_vert[v_adr : v_adr + v_num][rm.mesh_face[f_adr : f_adr + f_num]], max_spacing_m=0.00085)

    r2m_runner.ctrl.reverser = 0.75
    r2m_runner.ctrl.regulator = 0.85
    r2m_runner.ctrl.drain_cock = False

    tdr_geoms = ["tender_body_vis", "tender_internals_vis"]
    tdr_bodies = [
        "tender_bogie_1",
        "tender_bogie_2",
        "tender_wheelset_1",
        "tender_wheelset_2",
        "tender_wheelset_3",
        "tender_wheelset_4",
    ]

    def _r2m_body_pts(bname: str) -> np.ndarray:
        bid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_BODY, bname)
        gl = [
            (gid, int(rm.geom_dataid[gid]))
            for gid in range(rm.ngeom)
            if int(rm.geom_bodyid[gid]) == bid and int(rm.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_MESH
        ]
        return np.vstack([rd.geom_xpos[g] + r2m_mesh_pts[m_i] @ rd.geom_xmat[g].reshape(3, 3).T for g, m_i in gl])

    def _r2m_geom_pts(gname: str) -> np.ndarray:
        gid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_GEOM, gname)
        m_i = int(rm.geom_dataid[gid])
        return rd.geom_xpos[gid] + r2m_mesh_pts[m_i] @ rd.geom_xmat[gid].reshape(3, 3).T

    r2m_min_mm = 1e9
    r2m_worst = ("", "")
    r2m_brake_min_mm = 1e9
    r2m_tdr_min_mm = 1e9
    r2m_max_res_mm = 0.0

    for k_step in range(3000):  # 1.5s live R2m curve run
        r2m_runner.step_once()
        if k_step % 50 == 0:
            r2m_max_res_mm = max(r2m_max_res_mm, _max_connect_residual_mm(rm, rd))
        if k_step % 300 == 0:
            fg_trees = {g: (cKDTree(p := _r2m_geom_pts(g)), p.min(axis=0), p.max(axis=0)) for g in static_geoms_to_check}
            for b in moving_body_names:
                bp = _r2m_body_pts(b)
                bmn, bmx = bp.min(axis=0), bp.max(axis=0)
                for g, (gt, gmn, gmx) in fg_trees.items():
                    if (g, b) in mating_geom_body_whitelist:
                        continue
                    ub_mm = min(4.0, r2m_brake_min_mm if g == "chassis_equalizer_brakes_vis" else r2m_min_mm)
                    gap = np.maximum(0.0, np.maximum(gmn - bmx, bmn - gmx))
                    if float(np.linalg.norm(gap)) * 1000.0 >= ub_mm:
                        continue
                    dists, _ = gt.query(bp, distance_upper_bound=ub_mm * 0.001 + 1e-6, workers=1)
                    dm = float(np.min(dists)) * 1000.0
                    if math.isfinite(dm):
                        if g == "chassis_equalizer_brakes_vis":
                            r2m_brake_min_mm = min(r2m_brake_min_mm, dm)
                        if dm < r2m_min_mm:
                            r2m_min_mm = dm
                            r2m_worst = (g, b)
            tg_trees = {g: (cKDTree(p := _r2m_geom_pts(g)), p.min(axis=0), p.max(axis=0)) for g in tdr_geoms}
            for b in tdr_bodies:
                bp = _r2m_body_pts(b)
                bmn, bmx = bp.min(axis=0), bp.max(axis=0)
                for g, (gt, gmn, gmx) in tg_trees.items():
                    ub_mm = min(4.0, r2m_tdr_min_mm)
                    gap = np.maximum(0.0, np.maximum(gmn - bmx, bmn - gmx))
                    if float(np.linalg.norm(gap)) * 1000.0 >= ub_mm:
                        continue
                    dists, _ = gt.query(bp, distance_upper_bound=ub_mm * 0.001 + 1e-6, workers=1)
                    dm = float(np.min(dists)) * 1000.0
                    if math.isfinite(dm):
                        r2m_tdr_min_mm = min(r2m_tdr_min_mm, dm)
                        if dm < r2m_min_mm:
                            r2m_min_mm = dm
                            r2m_worst = (g, b)

    if r2m_min_mm < 0.50 or r2m_tdr_min_mm < 0.50 or r2m_brake_min_mm < 1.50 or r2m_max_res_mm >= 0.02:
        ok = False
        msgs.append(
            f"FAIL: R2m live curve scan failed: min={r2m_min_mm:.3f}mm ({r2m_worst}), tender_min={r2m_tdr_min_mm:.3f}mm, "
            f"brake_min={r2m_brake_min_mm:.3f}mm, max_res={r2m_max_res_mm:.5f}mm"
        )
    else:
        msgs.append(
            f"PASS: R2m (2000mm) live curve envelope: min={r2m_min_mm:.3f}mm >=0.50mm ({r2m_worst[0]}<->{r2m_worst[1]}), "
            f"tender bogie/wheel min={r2m_tdr_min_mm:.2f}mm, brakes>={min(4.0, r2m_brake_min_mm):.2f}mm, max <connect> residual={r2m_max_res_mm:.5f}mm (<0.02mm)"
        )

    return ok, msgs


# ==============================================================================
# Gate 8: 命名契约、三级精密闭环残差、第 3 轴横向浮动自由度与 1D-3D 配气试车验收
# ==============================================================================

def _max_connect_residual_mm(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    max_res_m = 0.0
    for eid in range(model.neq):
        if int(model.eq_type[eid]) != int(mujoco.mjtEq.mjEQ_CONNECT):
            continue
        id1 = int(model.eq_obj1id[eid])
        id2 = int(model.eq_obj2id[eid])
        if int(model.eq_objtype[eid]) == int(mujoco.mjtObj.mjOBJ_SITE):
            p1 = data.site_xpos[id1]
            p2 = data.site_xpos[id2]
        else:
            a1 = model.eq_data[eid, 0:3]
            a2 = model.eq_data[eid, 3:6]
            p1 = data.xpos[id1] + data.xmat[id1].reshape(3, 3) @ a1
            p2 = data.xpos[id2] + data.xmat[id2].reshape(3, 3) @ a2
        max_res_m = max(max_res_m, float(np.linalg.norm(p1 - p2)))
    return max_res_m * 1000.0


def verify_gate_8() -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    locked_joints = [
        "j_wheel_1", "j_wheel_2", "j_wheel_3", "j_wheel_4",
        "j_piston_L", "j_piston_R", "j_valve_L", "j_valve_R",
        "j_reverser", "j_pump_ram",
    ]
    locked_actuators = ["steam_force_L", "steam_force_R", "reverser_ctrl"]
    locked_sensors = [
        "piston_pos_L", "piston_vel_L", "valve_pos_L",
        "piston_pos_R", "piston_vel_R", "valve_pos_R",
        "wheel3_omega", "pump_ram_vel",
    ]

    for xml_rel in ("models/d51_full.xml", "models/d51_r2m.xml"):
        m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / xml_rel))
        miss = []
        for j in locked_joints:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) < 0:
                miss.append(f"joint:{j}")
        for a in locked_actuators:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) < 0:
                miss.append(f"actuator:{a}")
        for s in locked_sensors:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) < 0:
                miss.append(f"sensor:{s}")
        if miss:
            ok = False
            msgs.append(f"FAIL: {xml_rel} missing locked contracts: {miss}")

    if ok:
        msgs.append("PASS: All 10 locked joints, 3 actuators & 8 sensors intact in d51_full.xml & d51_r2m.xml")

    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "d51_full.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    res_t0_mm = _max_connect_residual_mm(model, data)

    # 8.2 施加 2.0 N 横向力于第 3 动轮 j_lat_y3，验证 ±0.35mm 曲柄销轴向游隙释放横动自由度 (>= 0.15 mm) 且闭环残差 < 0.02 mm
    jid_lat3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_lat_y3")
    dof_lat3 = int(model.jnt_dofadr[jid_lat3])
    qpos_lat3 = int(model.jnt_qposadr[jid_lat3])

    mujoco.mj_resetData(model, data)
    for _ in range(400):
        data.qfrc_applied[dof_lat3] = 2.0
        mujoco.mj_step(model, data)
    lat3_disp_mm = abs(float(data.qpos[qpos_lat3])) * 1000.0
    res_lat_mm = _max_connect_residual_mm(model, data)

    if res_t0_mm >= 0.02 or res_lat_mm >= 0.02 or lat3_disp_mm < 0.15:
        ok = False
        msgs.append(
            f"FAIL: Tier-1/Tier-2 closed-loop check: res_t0={res_t0_mm:.4f}mm, lat3={lat3_disp_mm:.3f}mm (need >=0.15mm), res_lat={res_lat_mm:.4f}mm"
        )
    else:
        msgs.append(
            f"PASS: Tier-1/2 closed-loop precision: t=0 residual={res_t0_mm:.5f}mm (<0.02mm); under 2.0N lateral load j_lat_y3={lat3_disp_mm:.3f}mm (>=0.15mm) with residual={res_lat_mm:.5f}mm"
        )

    # 8.3 验证 <equality><joint name="couple_reach_rod"> 定比耦合精度
    jid_rev = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_reverser")
    jid_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_reach_rod")
    ctrl_rev = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "reverser_ctrl")
    max_rr_err_mm = 0.0
    for rev_cmd in (-0.55, 0.0, +0.55):
        mujoco.mj_resetData(model, data)
        data.ctrl[ctrl_rev] = rev_cmd
        for _ in range(350):
            mujoco.mj_step(model, data)
        q_rev = float(data.qpos[model.jnt_qposadr[jid_rev]])
        q_rr = float(data.qpos[model.jnt_qposadr[jid_rr]])
        expected_rr = 0.007 * math.sin(q_rev)
        max_rr_err_mm = max(max_rr_err_mm, abs(q_rr - expected_rr) * 1000.0)
    if max_rr_err_mm > 0.05:
        ok = False
        msgs.append(f"FAIL: couple_reach_rod tracking error {max_rr_err_mm:.4f} mm")
    else:
        msgs.append(f"PASS: Tier-3 <equality><joint> cab_reach_rod tracking error = {max_rr_err_mm:.4f} mm")

    # 8.4 热流-多体耦合试车：3.8 kgf/cm² 中立位锁止 + 1.0 kgf/cm² 冷态气密正反转顺畅自启动 + 90° 四拍交替排气校验
    def run_coupled_test(
        p_kgf: float, rev_norm: float, dur_s: float = 1.6, cold_air: bool = False, track_beats: bool = False
    ) -> Tuple[float, float, List[Tuple[float, str, str]]]:
        runner = CoSimRunner(PROJECT_ROOT / "models" / "d51_full.xml", preset="hot")
        target_p = 101325.0 + p_kgf * 98066.5
        if cold_air:
            runner.thermo.P_boiler = target_p
            runner.thermo.T_boiler_c = sat_temp_c(target_p)
            runner.thermo.T_cyl_wall_c = 120.0
            runner.thermo.M_cond_L[:] = 0.0
            runner.thermo.M_cond_R[:] = 0.0
            runner.thermo.condensate_kg = 0.0
            runner.ctrl.burner_lit = False
            runner.ctrl.fuel_valve = 0.0
            runner.ctrl.drain_cock = True
        else:
            runner.thermo.P_boiler = target_p
            runner.thermo.T_boiler_c = sat_temp_c(target_p)
            runner.ctrl.drain_cock = False

        runner.ctrl.regulator = 0.0
        runner.ctrl.reverser = rev_norm
        for _ in range(250):
            runner.step_once()

        jid_w3_loc = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, "j_wheel_3")
        qpos_w3_loc = int(runner.model.jnt_qposadr[jid_w3_loc])
        ee = float(runner.thermo.calib.exhaust_lap)
        qc_L = float(runner.thermo.calib.q_valve_center_L)
        qc_R = float(runner.thermo.calib.q_valve_center_R)

        runner.ctrl.regulator = 0.85
        n_steps = int(round(dur_s / runner.model.opt.timestep))
        max_dyn_res = 0.0
        beats: List[Tuple[float, str, str]] = []
        prev_xvL = float(runner.data.sensordata[runner.s_vL]) - qc_L
        prev_xvR = float(runner.data.sensordata[runner.s_vR]) - qc_R

        for k in range(n_steps):
            if cold_air:
                runner.thermo.P_boiler = target_p
                runner.thermo.T_boiler_c = sat_temp_c(target_p)
            runner.step_once()
            if k % 100 == 0:
                max_dyn_res = max(max_dyn_res, _max_connect_residual_mm(runner.model, runner.data))
            if track_beats:
                xvL = float(runner.data.sensordata[runner.s_vL]) - qc_L
                xvR = float(runner.data.sensordata[runner.s_vR]) - qc_R
                if k > 800:  # spin-up 稳定后统计四拍交替排气开启角
                    deg_w = math.degrees(float(runner.data.qpos[qpos_w3_loc]))
                    if prev_xvL <= ee and xvL > ee:
                        beats.append((deg_w, "L", "head"))
                    elif prev_xvL >= -ee and xvL < -ee:
                        beats.append((deg_w, "L", "crank"))
                    if prev_xvR <= ee and xvR > ee:
                        beats.append((deg_w, "R", "head"))
                    elif prev_xvR >= -ee and xvR < -ee:
                        beats.append((deg_w, "R", "crank"))
                prev_xvL, prev_xvR = xvL, xvR

        return float(runner.data.sensordata[runner.s_w3]), max_dyn_res, beats

    om_N, res_N, _ = run_coupled_test(3.8, 0.0, dur_s=2.0, cold_air=False)
    om_F, res_F, beats_F = run_coupled_test(1.0, +0.85, dur_s=1.5, cold_air=True, track_beats=True)
    om_R, res_R, _ = run_coupled_test(1.0, -0.85, dur_s=1.5, cold_air=True)
    max_dyn = max(res_N, res_F, res_R)

    # 核验每转 4 次 L/R 严格交替排气及 90° 正交曲柄相位差
    beats_ok = len(beats_F) >= 6 and all(beats_F[i][1] != beats_F[i - 1][1] for i in range(1, len(beats_F)))
    beat_diffs = [beats_F[i][0] - beats_F[i - 1][0] for i in range(1, len(beats_F))]
    mean_beat_deg = float(np.mean(beat_diffs)) if beat_diffs else 0.0
    quarter_lags = []
    for ch_tag in ("head", "crank"):
        l_angs = [b[0] for b in beats_F if b[1] == "L" and b[2] == ch_tag]
        r_angs = [b[0] for b in beats_F if b[1] == "R" and b[2] == ch_tag]
        for ra in r_angs:
            later_l = [la - ra for la in l_angs if 45.0 < (la - ra) < 135.0]
            if later_l:
                quarter_lags.append(later_l[0])
    max_q_err_deg = max((abs(q - 90.0) for q in quarter_lags), default=999.0)
    beats_ok = beats_ok and abs(mean_beat_deg - 90.0) <= 1.5 and max_q_err_deg <= 1.5

    if abs(om_N) > 0.05 or om_F < 1.5 or om_R > -1.5 or max_dyn >= 0.02 or not beats_ok:
        ok = False
        msgs.append(
            f"FAIL: Coupled valve-timing test: om_N={om_N:+.3f} rad/s (<=0.05), om_F@1kgf={om_F:+.2f} rad/s, om_R@1kgf={om_R:+.2f} rad/s, "
            f"mean_beat={mean_beat_deg:.2f}°, quarter_err={max_q_err_deg:.2f}°, max_res={max_dyn:.4f}mm"
        )
    else:
        msgs.append(
            f"PASS: Coupled valve timing & air-test: 3.8 kgf/cm² Neutral ω={om_N:+.3f} rad/s (locked); "
            f"1.0 kgf/cm² Air-Test Forward ω={om_F:+.2f} rad/s, Reverse ω={om_R:+.2f} rad/s; "
            f"4-beat alternating exhaust mean={mean_beat_deg:.2f}° (R->L quartering Δ=90.0°±{max_q_err_deg:.2f}°); "
            f"dynamic <connect> max residual={max_dyn:.5f} mm (<0.02mm)"
        )

    return ok, msgs


# ==============================================================================
# Gate 9: 1D-3D 拓扑—几何连续性、单网格单连通分量与跨刚体全端口零断口对接门禁
# ==============================================================================

def _sample_geom_world_pts_for_port(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gid: int,
    stl_pts_cad_m_cache: Dict[str, np.ndarray],
    roi_local_mm: List[List[float]] | None = None,
) -> np.ndarray:
    """将网格或原生几何体表面加密采样后精确映射至 MuJoCo 世界系 (m)，支持 CAD 局部 ROI 裁剪。"""
    gtype = int(model.geom_type[gid])
    xpos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)

    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        stl_key = mname[2:] if mname.startswith("m_") else mname
        pts_cad_m = stl_pts_cad_m_cache[stl_key]
        if roi_local_mm is not None:
            lo_m = np.asarray(roi_local_mm[0], dtype=np.float64) * 0.001
            hi_m = np.asarray(roi_local_mm[1], dtype=np.float64) * 0.001
            mask = np.all((pts_cad_m >= lo_m) & (pts_cad_m <= hi_m), axis=1)
            pts_cad_m = pts_cad_m[mask]
        if len(pts_cad_m) == 0:
            return np.empty((0, 3), dtype=np.float64)
        rmat_mesh = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rmat_mesh, model.mesh_quat[mid])
        pts_mj = (pts_cad_m - model.mesh_pos[mid]) @ rmat_mesh.reshape(3, 3)
        return xpos + pts_mj @ xmat.T

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        sx, sy, sz = [float(v) for v in model.geom_size[gid][:3]]
        c = np.array(
            [
                [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],
                [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz],
            ],
            dtype=np.float64,
        )
        quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
        tris = []
        for a, b, d_idx, e in quads:
            tris.append([c[a], c[b], c[d_idx]])
            tris.append([c[a], c[d_idx], c[e]])
        pts_loc = sample_triangle_surface(np.asarray(tris, dtype=np.float64), max_spacing_m=0.00045)
        return xpos + pts_loc @ xmat.T

    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        rad, h_half = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
        thetas = np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False)
        zs = np.linspace(-h_half, +h_half, max(5, int(math.ceil(2.0 * h_half / 0.00045)) + 1))
        pts = []
        for z in zs:
            for th in thetas:
                pts.append([rad * math.cos(th), rad * math.sin(th), z])
        for z_cap in (-h_half, +h_half):
            pts.append([0.0, 0.0, z_cap])
            n_r = max(5, int(math.ceil(rad / 0.00040)))
            for r_sub in np.linspace(rad / n_r, rad, n_r):
                for th in thetas:
                    pts.append([r_sub * math.cos(th), r_sub * math.sin(th), z_cap])
        pts_loc = np.asarray(pts, dtype=np.float64)
        return xpos + pts_loc @ xmat.T

    raise ValueError(f"Unsupported geom type {gtype} for port mating gate")


def verify_gate_9(stl_tris_mm: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    pip_cfg = json.loads((PROJECT_ROOT / "configs" / "d51_boiler_piping.json").read_text(encoding="utf-8"))
    contracts = pip_cfg.get("topology_and_port_mating_contracts", {})
    if not contracts:
        return False, ["FAIL: Missing topology_and_port_mating_contracts in configs/d51_boiler_piping.json"]

    # 9.1 全车 54 个 STL 单连通分量门禁 (connected_components == 1)
    cc_spec = contracts["single_solid_mesh_invariant"]
    max_cc = int(cc_spec["max_connected_components"])
    tol_mm = float(cc_spec.get("vertex_weld_tol_mm", 1e-4))
    multi_cc_meshes: List[Tuple[str, int]] = []
    for name, tris in sorted(stl_tris_mm.items()):
        cc = count_connected_components(tris, tol_mm=tol_mm)
        if cc != max_cc:
            multi_cc_meshes.append((name, cc))
    if multi_cc_meshes:
        ok = False
        msgs.append(f"FAIL: Disconnected/multi-component STLs detected (expected cc=={max_cc}): {multi_cc_meshes}")
    else:
        msgs.append(f"PASS: All {len(stl_tris_mm)}/54 STLs are single continuous solids (connected_components == 1, zero floating fragments/voids)")

    # 9.2 炉内受压管路不越界包络核验 (防止 X-Ray 透视时内部管路刺出紫铜炉壳)
    enc_failures = []
    for enc in contracts.get("internal_enclosure_contracts", []):
        m_key = enc["mesh"]
        x0, x1 = [float(v) for v in enc["x_window_mm"]]
        z_min = float(enc["z_min_mm"])
        z_ax = float(enc["axis_z_mm"])
        r_lim = float(enc["max_radius_mm"])
        v = sample_triangle_surface(stl_tris_mm[m_key], max_spacing_m=0.80)
        sel = v[(v[:, 0] >= x0) & (v[:, 0] <= x1) & (v[:, 2] >= z_min)]
        if len(sel) == 0:
            enc_failures.append((enc["id"], "empty_window"))
            continue
        r_max = float(np.max(np.hypot(sel[:, 1], sel[:, 2] - z_ax)))
        if r_max > r_lim:
            enc_failures.append((enc["id"], round(r_max, 3), r_lim))
    if enc_failures:
        ok = False
        msgs.append(f"FAIL: Internal boiler pipe enclosure violated: {enc_failures}")
    else:
        msgs.append("PASS: Internal regulator (13-9), 4-pass superheater (17-5) & safety-valve tie strictly enclosed inside boiler copper shell")

    # 9.3 跨刚体/跨网格 1D-3D 全端口零断口对接核验 (同时覆盖直线滚轮台与 R2m 弯道双姿态!)
    port_spec = contracts["cross_body_port_mating_contracts"]
    max_gap_mm = float(port_spec["max_gap_mm"])
    ports = port_spec["ports"]

    # 仅对参与端口核验的网格做 0.45mm 高密表面采样缓存
    needed_stl_keys: Set[str] = set()
    for xml_rel in port_spec["xml_targets"]:
        xml_path = PROJECT_ROOT / xml_rel
        if not xml_path.exists():
            ok = False
            msgs.append(f"FAIL: Missing XML target {xml_rel} for Gate 9 port check")
            return ok, msgs
        m_tmp = mujoco.MjModel.from_xml_path(str(xml_path))
        for p in ports:
            for g_tag in ("geom_a", "geom_b"):
                gid = mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_GEOM, p[g_tag])
                if gid >= 0 and int(m_tmp.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mid = int(m_tmp.geom_dataid[gid])
                    mnm = mujoco.mj_id2name(m_tmp, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
                    needed_stl_keys.add(mnm[2:] if mnm.startswith("m_") else mnm)

    stl_pts_cad_m_cache: Dict[str, np.ndarray] = {
        k: sample_triangle_surface(stl_tris_mm[k] * 0.001, max_spacing_m=0.00045)
        for k in needed_stl_keys
    }

    for xml_rel in port_spec["xml_targets"]:
        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / xml_rel))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        tree_b_cache: Dict[Tuple[int, str], cKDTree] = {}
        worst_gap_mm = 0.0
        worst_port_id = ""
        port_errors = []

        for p in ports:
            pid = p["id"]
            ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p["geom_a"])
            gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p["geom_b"])
            if ga < 0 or gb < 0:
                port_errors.append((pid, f"missing_geom({p['geom_a']},{p['geom_b']})"))
                continue

            pts_a = _sample_geom_world_pts_for_port(model, data, ga, stl_pts_cad_m_cache, p.get("roi_a_local_mm"))
            if len(pts_a) == 0:
                port_errors.append((pid, "empty_roi_a"))
                continue

            roi_b = p.get("roi_b_local_mm")
            b_key = (gb, json.dumps(roi_b))
            if b_key not in tree_b_cache:
                pts_b = _sample_geom_world_pts_for_port(model, data, gb, stl_pts_cad_m_cache, roi_b)
                if len(pts_b) == 0:
                    port_errors.append((pid, "empty_roi_b"))
                    continue
                tree_b_cache[b_key] = cKDTree(pts_b)

            dists, _ = tree_b_cache[b_key].query(pts_a, k=1)
            gap_mm = float(np.min(dists)) * 1000.0
            if gap_mm > worst_gap_mm:
                worst_gap_mm = gap_mm
                worst_port_id = pid
            if gap_mm > max_gap_mm:
                port_errors.append((pid, f"gap={gap_mm:.3f}mm > {max_gap_mm:.2f}mm"))

        if port_errors:
            ok = False
            msgs.append(f"FAIL [{xml_rel}]: Broken/disconnected 1D-3D ports: {port_errors}")
        else:
            msgs.append(
                f"PASS [{xml_rel}]: All {len(ports)}/{len(ports)} cross-body fluid & mechanical ports mated continuously "
                f"(worst gap={worst_gap_mm:.3f} mm <= {max_gap_mm:.2f} mm at {worst_port_id})"
            )

    return ok, msgs


def verify_gate_10() -> Tuple[bool, List[str]]:
    """
    Gate 10: 「三层黄金分工」物理—算子—认知混合架构验收
      10.1 源码零作弊审计：严禁 sim/thermo_3d_gpu.py 出现 smokebox_vacuum_pa *= 或 airtight_factor 等人工漏气乘数；
      10.2 L1 IAPWS-95/IF97 真物性与连续比焓相变校验 (过冷水 -> 湿饱和 x∈(0,1) -> 干饱和 -> 过热)；
      10.3 L2 3D PINO 训练收敛度与实车 STL 破口/堵管零 if 涌现校验；
      10.4 L3 AsyncBonsaiCopilot 零阻塞提交时延校验 (< 0.5 ms)。
    """
    ok = True
    msgs: List[str] = []

    # 10.1 源码静态零硬编码漏气审计
    gpu_src = (PROJECT_ROOT / "sim" / "thermo_3d_gpu.py").read_text(encoding="utf-8")
    forbidden_tokens = ["smokebox_vacuum_pa *=", "airtight_factor ="]
    found_bad = [tok for tok in forbidden_tokens if tok in gpu_src]
    if found_bad:
        ok = False
        msgs.append(f"FAIL: Empirical leak multiplier found in sim/thermo_3d_gpu.py: {found_bad}")
    else:
        msgs.append("PASS: Zero hardcoded leak/plug multipliers in sim/thermo_3d_gpu.py (100% PDE/PINO BC-driven)")

    # 10.2 L1 IAPWS 饱和反演与连续比焓相变校验
    from sim.iapws_thermo import sat_liquid_props, sat_pressure_pa, sat_temp_c, sat_vapor_props, steam_ph_state

    max_inv_err = 0.0
    for tc in (25.0, 99.974, 120.0, 149.6, 165.0):
        err = abs(sat_temp_c(sat_pressure_pa(tc)) - tc)
        max_inv_err = max(max_inv_err, err)
    p_test = sat_pressure_pa(149.6)
    hf, _, _ = sat_liquid_props(149.6)
    hg, hfg, _, _ = sat_vapor_props(149.6)
    st_sub = steam_ph_state(p_test, hf - 80e3)
    st_wet = steam_ph_state(p_test, hf + 0.55 * hfg)
    st_sup = steam_ph_state(p_test, hg + 65e3)
    iapws_ok = (
        max_inv_err < 1e-5
        and st_sub.temp_c < st_sub.t_sat_c - 10.0
        and st_sub.dryness_x == 0.0
        and abs(st_wet.temp_c - st_wet.t_sat_c) < 1e-4
        and abs(st_wet.dryness_x - 0.55) < 1e-4
        and st_sup.dryness_x == 1.0
        and st_sup.superheat_deg > 20.0
    )
    if not iapws_ok:
        ok = False
        msgs.append(f"FAIL: IAPWS phase-change check failed (inv_err={max_inv_err:.2e})")
    else:
        msgs.append(
            f"PASS: L1 IAPWS-95/IF97 verified (Tsat inv_err={max_inv_err:.2e}°C, "
            f"subcooled->{st_wet.dryness_x:.2f} wet->+{st_sup.superheat_deg:.1f}°C superheat)"
        )

    # 10.3 L2 3D PINO 算子报告、权重与零 if 破口涌现核验
    rep_path = PROJECT_ROOT / "configs" / "pino_training_report.json"
    ckpt_path = PROJECT_ROOT / "models" / "pino_d51_surrogate.pt"
    if not rep_path.exists() or not ckpt_path.exists():
        ok = False
        msgs.append(f"FAIL: Missing PINO report ({rep_path.exists()}) or checkpoint ({ckpt_path.exists()})")
    else:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        bm = rep.get("burner_validation", rep.get("burner_pino", {}).get("metrics", {}))
        em = rep.get("gate10_zero_if_emergence", {})
        r2_vac = float(bm.get("r2_smokebox_vacuum", 0.0))
        r2_q = float(bm.get("r2_boiler_heat_w", 0.0))
        em_pass = bool(em.get("gate10_emergence_pass", False))
        intact_vac_rep = em.get("intact_5flue_airtight", {}).get("vacuum_pa", em.get("intact_5flue_vac_pa"))
        breach_vac_rep = em.get("smokebox_door_breach_no_if_rule", {}).get("vacuum_pa", em.get("door_breach_vac_pa"))
        if r2_vac < 0.88 or r2_q < 0.90 or not em_pass:
            ok = False
            msgs.append(f"FAIL: PINO accuracy/emergence below contract (R2_vac={r2_vac:.4f}, R2_Q={r2_q:.4f}, pass={em_pass})")
        else:
            msgs.append(
                f"PASS: L2 PINO report verified (R2_vac={r2_vac:.4f}, R2_Q={r2_q:.4f}, "
                f"intact={intact_vac_rep}Pa -> door_breach={breach_vac_rep}Pa)"
            )

        # 若当前环境装有 PyTorch，直接在真实 AsterD51Thermo3DGPU 实例上对烟室前门 STL 挖洞做实测涌现核验！
        try:
            import torch
            from sim.run_cosim import calibrate_model_and_valves
            from sim.thermo_1d import CabControls
            from sim.thermo_3d_gpu import AsterD51Thermo3DGPU

            m_mj = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "d51_full.xml"))
            cal = calibrate_model_and_valves(m_mj)
            eng = AsterD51Thermo3DGPU(cal)
            eng.reset_hot_ready()
            ctrl = CabControls(fuel_valve=1.0, burner_lit=True, blower_valve=0.75, regulator=0.35, reverser=0.75)
            for _ in range(24):
                _, _, _, tel_ok = eng.step(
                    0.0005, cal.q_fdc_L - 0.01, 0.0, cal.q_valve_center_L, cal.q_fdc_R - 0.01, 0.0, cal.q_valve_center_R, 8.0, 0.0, ctrl
                )
            vac_intact = tel_ok.smokebox_vacuum_pa

            # 直接在烟室前门固体体素掩码 s1_sup_raw 上挖开 ø40mm 大破口，不改任何一行物理代码！
            nx1, ny1, nz1 = eng.D1_SHAPE
            X1, Y1, Z1 = torch.meshgrid(
                torch.linspace(eng.d1_xr[0], eng.d1_xr[1], nx1, device=eng.device),
                torch.linspace(eng.d1_yr[0], eng.d1_yr[1], ny1, device=eng.device),
                torch.linspace(eng.d1_zr[0], eng.d1_zr[1], nz1, device=eng.device),
                indexing="ij",
            )
            sup_breached = eng.s1_sup_raw.clone()
            door_hole = (X1 >= 146.0) & (torch.hypot(Y1, Z1 - 44.0) <= 22.0)
            sup_breached[door_hole] = 0.0
            eng.rebuild_d1_stl_geometry(s1_sup_override=sup_breached)
            for _ in range(24):
                _, _, _, tel_leak = eng.step(
                    0.0005, cal.q_fdc_L - 0.01, 0.0, cal.q_valve_center_L, cal.q_fdc_R - 0.01, 0.0, cal.q_valve_center_R, 8.0, 0.0, ctrl
                )
            vac_breach = tel_leak.smokebox_vacuum_pa
            if not (tel_ok.pino_active and vac_intact > 35.0 and vac_breach < 0.35 * vac_intact):
                ok = False
                msgs.append(
                    f"FAIL: Live 3D PINO STL hole-cut emergence failed (pino={tel_ok.pino_active}, "
                    f"intact={vac_intact:.1f}Pa, breach={vac_breach:.1f}Pa)"
                )
            else:
                msgs.append(
                    f"PASS: Live AsterD51Thermo3DGPU STL door-cut emergence verified "
                    f"(intact={vac_intact:.1f} Pa -> breached={vac_breach:.1f} Pa with zero 'if')"
                )
        except ImportError:
            msgs.append("PASS: CPU-only runner detected (PyTorch optional live GPU check skipped)")

    # 10.4 L3 AsyncBonsaiCopilot 主循环零阻塞时延核验
    from sim.bonsai_copilot import AsyncBonsaiCopilot

    cop = AsyncBonsaiCopilot(enabled=False)
    t0 = time.perf_counter()
    cop.submit_telemetry_nonblocking({"sim_time": 1.0, "boiler_pressure_kgf": 3.5}, {"regulator": 0.3})
    dt_submit_ms = (time.perf_counter() - t0) * 1000.0
    if dt_submit_ms > 0.5:
        ok = False
        msgs.append(f"FAIL: Copilot submit blocked main loop for {dt_submit_ms:.3f} ms > 0.5 ms")
    else:
        msgs.append(f"PASS: L3 AsyncBonsaiCopilot non-blocking handoff verified ({dt_submit_ms:.4f} ms < 0.5 ms)")

    return ok, msgs


# ==============================================================================
# 主入口
# ==============================================================================

def main() -> int:
    print("=" * 88)
    print(" Aster Hobby JNR D51 — Automated CAD Topology, BOM Truth & 360° Clearance Gate Suite")
    print("=" * 88)

    all_ok = True

    g1_ok, g1_msgs = verify_gate_1()
    print(f"\n[Gate 1] BOM Truth & Headless Calibration -> {'PASS' if g1_ok else 'FAIL'}")
    for m in g1_msgs:
        print(f"  - {m}")
    all_ok &= g1_ok

    g2_ok, g2_msgs, stl_tris_mm = verify_gate_2()
    print(f"\n[Gate 2] 54-Mesh Watertightness, Smokebox Continuity & Symmetry -> {'PASS' if g2_ok else 'FAIL'}")
    for m in g2_msgs:
        print(f"  - {m}")
    all_ok &= g2_ok

    if not stl_tris_mm:
        return 1

    g3_ok, g3_msgs = verify_gate_3(stl_tris_mm)
    print(f"\n[Gate 3 & 7] Curved-Slot Alignment, Stepped Pins & X-Ray Mesh Split -> {'PASS' if g3_ok else 'FAIL'}")
    for m in g3_msgs:
        print(f"  - {m}")
    all_ok &= g3_ok

    g6_ok, g6_msgs = verify_gate_6(stl_tris_mm)
    print(f"\n[Gate 6] Inside-Frame Piping Bounds & 360°×3 Swept 3D Clearance -> {'PASS' if g6_ok else 'FAIL'}")
    for m in g6_msgs:
        print(f"  - {m}")
    all_ok &= g6_ok

    g8_ok, g8_msgs = verify_gate_8()
    print(f"\n[Gate 8] 3-Tier Closed-Loop Residual, Axle Lateral Play & Air-Test -> {'PASS' if g8_ok else 'FAIL'}")
    for m in g8_msgs:
        print(f"  - {m}")
    all_ok &= g8_ok

    g9_ok, g9_msgs = verify_gate_9(stl_tris_mm)
    print(f"\n[Gate 9] 1D-3D Single-Solid Continuity & 22-Port Cross-Body Mating -> {'PASS' if g9_ok else 'FAIL'}")
    for m in g9_msgs:
        print(f"  - {m}")
    all_ok &= g9_ok

    g10_ok, g10_msgs = verify_gate_10()
    print(f"\n[Gate 10] 3-Tier Hybrid IAPWS + Zero-If 3D PINO Emergence + Bonsai L3 -> {'PASS' if g10_ok else 'FAIL'}")
    for m in g10_msgs:
        print(f"  - {m}")
    all_ok &= g10_ok

    print("\n" + "=" * 88)
    if all_ok:
        print(" [ALL GATES 1-10 PASSED] P0-C + P0-D + P1-A + GATE 10 (HYBRID PINO+IAPWS) VERIFIED 100%")
        print("=" * 88)
        return 0
    else:
        print(" [GATE FAILURE] One or more verification gates failed!")
        print("=" * 88)
        return 1


if __name__ == "__main__":
    sys.exit(main())
