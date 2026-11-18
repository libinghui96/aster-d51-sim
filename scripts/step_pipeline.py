#!/usr/bin/env python3
"""
scripts/step_pipeline.py
ISO-10303-21 STEP-First B-Rep Master Compiler & Analytic Lumen Verifier.

Architecture:
  1. `models/step/<part>.step` is the canonical CAD source of truth (exact NURBS / analytic B-Rep).
  2. `load_step_shape()` parses `.step` via OpenCascade `STEPControl_Reader` (zero STL approximation).
  3. `compute_step_mass_props()` integrates exact volume, COM, and 3x3 inertia tensor on the `.step` B-Rep.
  4. `tessellate_step_to_stl()` meshes the `.step` B-Rep via `BRepMesh_IncrementalMesh` + `EnsureNormalConsistency_s`
     with exact face-orientation winding into watertight runtime `.stl` meshes for MuJoCo C++/WASM & CUDA voxelization.
  5. `classify_step_points_hollow()` & `verify_step_cross_sections_open()` run exact OpenCascade
     `BRepClass3d_SolidClassifier` point-in-solid queries to guarantee end-to-end hollow fluid lumens.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_OUT, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Pnt


def load_step_shape(step_path: Path) -> TopoDS_Shape:
    """Load an ISO-10303-21 .step file into an OpenCascade TopoDS_Shape and verify solid presence."""
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPControl_Reader failed on {step_path} (status={status})")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"Loaded null TopoDS_Shape from {step_path}")
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n_solids = 0
    while exp.More():
        n_solids += 1
        exp.Next()
    if n_solids < 1:
        raise RuntimeError(f"Expected >=1 B-Rep solid in {step_path}, found {n_solids}")
    return shape


def compute_step_mass_props(step_shape: TopoDS_Shape, density_kg_m3: float) -> Dict[str, Any]:
    """Exact analytic volume, center of mass, and 3x3 inertia integration directly on STEP B-Rep (mm -> SI)."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(step_shape, props)

    vol_mm3 = float(props.Mass())
    com = props.CentreOfMass()
    mat = props.MatrixOfInertia()

    mass_kg = vol_mm3 * 1e-9 * density_kg_m3
    scale_i = 1e-15 * density_kg_m3

    ixx = max(1e-9, float(mat.Value(1, 1)) * scale_i)
    iyy = max(1e-9, float(mat.Value(2, 2)) * scale_i)
    izz = max(1e-9, float(mat.Value(3, 3)) * scale_i)
    ixy = float(mat.Value(1, 2)) * scale_i
    ixz = float(mat.Value(1, 3)) * scale_i
    iyz = float(mat.Value(2, 3)) * scale_i

    return {
        "density_kg_m3": float(density_kg_m3),
        "volume_cm3": round(vol_mm3 * 1e-3, 4),
        "mass_kg": mass_kg,
        "com_m": [float(com.X()) * 1e-3, float(com.Y()) * 1e-3, float(com.Z()) * 1e-3],
        "fullinertia": [ixx, iyy, izz, ixy, ixz, iyz],
    }


def tessellate_step_to_stl(
    step_shape: TopoDS_Shape,
    stl_path: Path,
    lin_tol: float = 0.022,
    ang_tol: float = 0.20,
) -> int:
    """
    Deterministically tessellate an OpenCascade STEP B-Rep shape into a watertight binary STL.
    Preserves exact shared boundary edge vertices and outward face orientation.
    """
    BRepTools.Clean_s(step_shape)
    BRepMesh_IncrementalMesh(step_shape, float(lin_tol), False, float(ang_tol), True)
    BRepLib.EnsureNormalConsistency_s(step_shape, 1e-3, True)

    tri_chunks: List[np.ndarray] = []
    exp = TopExp_Explorer(step_shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly is not None:
            n_nodes = poly.NbNodes()
            n_tris = poly.NbTriangles()
            if n_nodes >= 3 and n_tris >= 1:
                trsf = loc.Transformation()
                has_trsf = not loc.IsIdentity()
                nodes = np.empty((n_nodes, 3), dtype=np.float64)
                for i in range(1, n_nodes + 1):
                    p = poly.Node(i)
                    if has_trsf:
                        p = p.Transformed(trsf)
                    nodes[i - 1, 0] = p.X()
                    nodes[i - 1, 1] = p.Y()
                    nodes[i - 1, 2] = p.Z()

                rev = face.Orientation() == TopAbs_REVERSED
                idx = np.empty((n_tris, 3), dtype=np.int32)
                for t in range(1, n_tris + 1):
                    tri = poly.Triangle(t)
                    i1, i2, i3 = tri.Get()
                    if rev:
                        idx[t - 1, 0] = i1 - 1
                        idx[t - 1, 1] = i3 - 1
                        idx[t - 1, 2] = i2 - 1
                    else:
                        idx[t - 1, 0] = i1 - 1
                        idx[t - 1, 1] = i2 - 1
                        idx[t - 1, 2] = i3 - 1
                tri_chunks.append(nodes[idx])
        exp.Next()

    if not tri_chunks:
        raise RuntimeError(f"STEP tessellation produced 0 triangles for {stl_path}")

    all_tris = np.concatenate(tri_chunks, axis=0)
    v0, v1, v2 = all_tris[:, 0], all_tris[:, 1], all_tris[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    valid = nlen[:, 0] > 1e-13
    all_tris = all_tris[valid]
    nrm = nrm[valid] / nlen[valid]

    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    out_arr = np.zeros(len(all_tris), dtype=dt)
    out_arr["n"] = nrm.astype(np.float32)
    out_arr["v"] = all_tris.astype(np.float32)

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    header = b"ISO-10303-21 STEP-Compiled Watertight Mesh".ljust(80, b" ")
    with stl_path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(all_tris)))
        f.write(out_arr.tobytes())
    return int(len(all_tris))


def compile_step_to_stl_and_mass(
    step_path: Path,
    stl_path: Path,
    density_kg_m3: float,
    lin_tol: float,
    ang_tol: float,
) -> Tuple[Dict[str, Any], int]:
    """Read canonical .step file, compute exact B-Rep mass/inertia, and compile runtime .stl mesh."""
    shape = load_step_shape(step_path)
    mp = compute_step_mass_props(shape, density_kg_m3)
    n_tris = tessellate_step_to_stl(shape, stl_path, lin_tol=lin_tol, ang_tol=ang_tol)
    mp["step_file"] = f"models/step/{step_path.name}"
    mp["stl_file"] = f"models/meshes/{stl_path.name}"
    mp["stl_linear_tol_mm"] = float(lin_tol)
    mp["stl_angular_tol_rad"] = float(ang_tol)
    mp["stl_triangles"] = int(n_tris)
    mp["compiled_from_step"] = True
    return mp, n_tris


def classify_step_points_hollow(
    step_shape: TopoDS_Shape,
    pts_mm: Sequence[Tuple[float, float, float]],
    tol: float = 1e-4,
) -> List[bool]:
    """Return True for each 3D point (mm) that is strictly outside solid metal (i.e. open lumen / air)."""
    clf = BRepClass3d_SolidClassifier(step_shape)
    out: List[bool] = []
    for x, y, z in pts_mm:
        clf.Perform(gp_Pnt(float(x), float(y), float(z)), float(tol))
        out.append(clf.State() == TopAbs_OUT)
    return out


def verify_step_cross_section_open(
    step_shape: TopoDS_Shape,
    center_mm: Tuple[float, float, float],
    tangent_xyz: Tuple[float, float, float],
    r_probe_mm: float,
) -> Tuple[bool, int]:
    """
    Verify that a 5-point cross-section (center + 4 orthogonal radial offsets at r_probe_mm)
    normal to tangent_xyz is 100% open (TopAbs_OUT) inside a STEP B-Rep pipe/valve bore.
    """
    tx, ty, tz = tangent_xyz
    t_norm = math.sqrt(tx * tx + ty * ty + tz * tz)
    if t_norm < 1e-9:
        tx, ty, tz = 1.0, 0.0, 0.0
    else:
        tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm

    if abs(tx) < 0.85:
        ux, uy, uz = 0.0, tz, -ty
    else:
        ux, uy, uz = -tz, 0.0, tx
    u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm
    vx = ty * uz - tz * uy
    vy = tz * ux - tx * uz
    vz = tx * uy - ty * ux

    cx, cy, cz = center_mm
    probes = [
        (cx, cy, cz),
        (cx + r_probe_mm * ux, cy + r_probe_mm * uy, cz + r_probe_mm * uz),
        (cx - r_probe_mm * ux, cy - r_probe_mm * uy, cz - r_probe_mm * uz),
        (cx + r_probe_mm * vx, cy + r_probe_mm * vy, cz + r_probe_mm * vz),
        (cx - r_probe_mm * vx, cy - r_probe_mm * vy, cz - r_probe_mm * vz),
    ]
    flags = classify_step_points_hollow(step_shape, probes)
    n_open = sum(1 for fl in flags if fl)
    return (n_open == 5), n_open


def verify_step_wall_bounded_lumen(
    step_shape: TopoDS_Shape,
    center_mm: Tuple[float, float, float],
    tangent_xyz: Tuple[float, float, float],
    r_lumen_mm: float,
    r_wall_mm: float,
) -> Tuple[bool, int, int]:
    """
    Verify BOTH that the inner bore (5 points within r_lumen_mm) is 100% open (TopAbs_OUT)
    AND that the surrounding pipe wall ring (4 points at r_wall_mm) is solid metal (not TopAbs_OUT),
    preventing off-pipe probes in empty air from falsely passing lumen checks!
    """
    tx, ty, tz = tangent_xyz
    t_norm = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
    tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm
    ux, uy, uz = (0.0, tz, -ty) if abs(tx) < 0.85 else (-tz, 0.0, tx)
    u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm
    vx, vy, vz = ty * uz - tz * uy, tz * ux - tx * uz, tx * uy - ty * ux

    cx, cy, cz = center_mm
    bore_pts = [
        (cx, cy, cz),
        (cx + r_lumen_mm * ux, cy + r_lumen_mm * uy, cz + r_lumen_mm * uz),
        (cx - r_lumen_mm * ux, cy - r_lumen_mm * uy, cz - r_lumen_mm * uz),
        (cx + r_lumen_mm * vx, cy + r_lumen_mm * vy, cz + r_lumen_mm * vz),
        (cx - r_lumen_mm * vx, cy - r_lumen_mm * vy, cz - r_lumen_mm * vz),
    ]
    wall_pts = [
        (cx + r_wall_mm * ux, cy + r_wall_mm * uy, cz + r_wall_mm * uz),
        (cx - r_wall_mm * ux, cy - r_wall_mm * uy, cz - r_wall_mm * uz),
        (cx + r_wall_mm * vx, cy + r_wall_mm * vy, cz + r_wall_mm * vz),
        (cx - r_wall_mm * vx, cy - r_wall_mm * vy, cz - r_wall_mm * vz),
    ]
    bore_open = sum(1 for fl in classify_step_points_hollow(step_shape, bore_pts) if fl)
    wall_solid = sum(1 for fl in classify_step_points_hollow(step_shape, wall_pts) if not fl)
    return (bore_open == 5 and wall_solid >= 3), bore_open, wall_solid


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile runtime STL meshes & B-Rep mass props directly from STEP masters")
    parser.add_argument("--step-dir", type=Path, default=Path("models/step"))
    parser.add_argument("--stl-dir", type=Path, default=Path("models/meshes"))
    parser.add_argument("--mass-json", type=Path, default=Path("configs/d51_mass_props.json"))
    parser.add_argument("--compile-all", action="store_true", help="Recompile all models/step/*.step into STL + mass props")
    parser.add_argument("--only", type=str, default="", help="Comma-separated stems to compile from STEP")
    args = parser.parse_args()

    if not args.mass_json.exists():
        print(f"Missing mass metadata reference {args.mass_json}", file=sys.stderr)
        return 1

    mass_db = json.loads(args.mass_json.read_text(encoding="utf-8"))
    parts_meta = mass_db.get("parts", {})
    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()

    step_files = sorted(args.step_dir.glob("*.step"))
    if only_set:
        step_files = [p for p in step_files if p.stem in only_set]

    if not (args.compile_all or only_set):
        print(f"[STEP-PIPELINE] Found {len(step_files)} STEP B-Rep masters in {args.step_dir}. Pass --compile-all or --only.")
        return 0

    for idx, sp in enumerate(step_files, start=1):
        stem = sp.stem
        prev = parts_meta.get(stem, {})
        rho = float(prev.get("density_kg_m3", 8500.0))
        lin_tol = float(prev.get("stl_linear_tol_mm", 0.022))
        ang_tol = float(prev.get("stl_angular_tol_rad", 0.20))
        stl_out = args.stl_dir / f"{stem}.stl"
        mp, n_tris = compile_step_to_stl_and_mass(sp, stl_out, rho, lin_tol, ang_tol)
        parts_meta[stem] = mp
        print(f"[STEP->STL] {idx:02d}/{len(step_files)} {stem}.step -> {n_tris:6d} tris | mass={mp['mass_kg']*1000:6.2f} g")

    mass_db["parts"] = parts_meta
    args.mass_json.write_text(json.dumps(mass_db, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
