#!/usr/bin/env python3
"""
scripts/05_sync_web_assets.py
将 models/d51_full.xml、models/d51_r2m.xml 以及 models/meshes/*.stl 自动同步至 web/public/models/，
并为每个资产计算 SHA-256 内容指纹写入 manifest.json，确保 GitHub Pages / CDN / 浏览器在 STL 或 XML
发生任何字节变更时 100% 自动穿透旧缓存获取最新网格与模型。
"""

import hashlib
import json
from pathlib import Path
import shutil


def _sha256_short(p: Path, length: int = 10) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:length]


def main() -> None:
    src_models = Path("models")
    dst_models = Path("web/public/models")
    dst_meshes = dst_models / "meshes"
    dst_meshes.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    manifest: dict = {"xmls": [], "meshes": [], "configs": [], "hashes": hashes, "build_hash": ""}

    for xml_name in ("d51_full.xml", "d51_track.xml", "d51_r2m.xml", "d51_oval.xml"):
        p = src_models / xml_name
        if p.exists():
            dst = dst_models / xml_name
            shutil.copy2(p, dst)
            manifest["xmls"].append(xml_name)
            hashes[xml_name] = _sha256_short(dst)

    mesh_src = src_models / "meshes"
    if mesh_src.exists():
        valid_stls = {stl.name for stl in mesh_src.glob("*.stl")}
        for old_stl in dst_meshes.glob("*.stl"):
            if old_stl.name not in valid_stls:
                old_stl.unlink()
        for stl in sorted(mesh_src.glob("*.stl")):
            dst = dst_meshes / stl.name
            shutil.copy2(stl, dst)
            manifest["meshes"].append(stl.name)
            hashes[f"meshes/{stl.name}"] = _sha256_short(dst)

    for cfg_name in (
        "d51_mass_props.json",
        "d51_boiler_piping.json",
        "d51_fluid_topology.json",
        "d51_fluid_lvs_report.json",
        "aster_gauge1_prior_knowledge.json",
        "smokebox_lut.json",
        "auto_fireman_policy.json",
        "d51_range_benchmark.json",
    ):
        cfg_p = Path("configs") / cfg_name
        if cfg_p.exists():
            dst = dst_models / cfg_name
            shutil.copy2(cfg_p, dst)
            manifest["configs"].append(cfg_name)
            hashes[cfg_name] = _sha256_short(dst)

    combined = "|".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
    manifest["build_hash"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]

    (dst_models / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[SYNC] 已同步 {len(manifest['xmls'])} 个 XML 与 {len(manifest['meshes'])} 个 STL "
        f"(build_hash={manifest['build_hash']}) 至 {dst_models}"
    )


if __name__ == "__main__":
    main()
