#!/usr/bin/env python3
"""
scripts/05_sync_web_assets.py
将 models/d51_full.xml、models/d51_r2m.xml 以及 models/meshes/*.stl 自动同步至 web/public/models/，
生成 manifest.json 供浏览器端 MuJoCo WASM 虚拟文件系统 (MEMFS) 动态挂载。
"""

import json
from pathlib import Path
import shutil


def main():
    src_models = Path("models")
    dst_models = Path("web/public/models")
    dst_meshes = dst_models / "meshes"
    dst_meshes.mkdir(parents=True, exist_ok=True)

    manifest = {"xmls": [], "meshes": [], "configs": []}

    for xml_name in ("d51_full.xml", "d51_r2m.xml"):
        p = src_models / xml_name
        if p.exists():
            shutil.copy2(p, dst_models / xml_name)
            manifest["xmls"].append(xml_name)

    mesh_src = src_models / "meshes"
    if mesh_src.exists():
        for stl in sorted(mesh_src.glob("*.stl")):
            shutil.copy2(stl, dst_meshes / stl.name)
            manifest["meshes"].append(stl.name)

    for cfg_name in ("d51_mass_props.json", "smokebox_lut.json", "auto_fireman_policy.json"):
        cfg_p = Path("configs") / cfg_name
        if cfg_p.exists():
            shutil.copy2(cfg_p, dst_models / cfg_name)
            manifest["configs"].append(cfg_name)

    (dst_models / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[SYNC] 已同步 {len(manifest['xmls'])} 个 XML 与 {len(manifest['meshes'])} 个 STL 至 {dst_models}")


if __name__ == "__main__":
    main()