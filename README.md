# Aster D51 live-steam co-simulation

Aster Hobby 1 番ゲージ (1/32, 45 mm 轨距) JNR D51 真蒸汽机车的 **1D 锅炉/气缸热力学 + 3D MuJoCo 华氏配气与轮轨动力学** 双向联合仿真。

- **在线 Web 3D 实时交互数字孪生 (WASM + Three.js，由 GitHub Actions 自动构建发布):** https://libinghui96.github.io/aster-d51-sim/
- **CI 试运转遥测与离屏动画报告:** https://libinghui96.github.io/aster-d51-sim/report.html

![running](https://libinghui96.github.io/aster-d51-sim/d51_running.gif)

![telemetry](https://libinghui96.github.io/aster-d51-sim/d51_telemetry.png)

## 本地运行

### 1. Python 建模与仿真
```bash
pip install -r requirements.txt
python scripts/02_build_cad_meshes.py                           # 构建 54 件全水密 CAD STL 网格与材料惯量库
python scripts/03_generate_mjcf.py --mode roller                # 编译滚轮台模型 -> models/d51_full.xml
python scripts/03_generate_mjcf.py --mode curve_r2m --out models/d51_r2m.xml
python scripts/09_verify_cad_and_clearance.py                   # 运行 CAD 拓扑/BOM 真值/360° 净空与闭环门禁
python -m sim.run_cosim --headless-demo                         # 试运转全流程 -> logs/d51_telemetry.png
python scripts/04_analyze_r2m_derailment.py                     # R2m vs R3m 先导支点过弯 Nadal 脱线分析
MUJOCO_GL=egl python scripts/04_render_animation.py             # 离屏渲染 -> logs/d51_running.gif
```

### 2. Web 3D 实时交互前端
```bash
python scripts/05_sync_web_assets.py                            # 同步 XML 与 STL 至 web/public/models/
cd web && npm install && npm run dev                            # 启动浏览器交互端 (http://localhost:5173)
```

`data/` 下的原厂说明书 PDF 不随仓库分发；`scripts/01_extract_pdf_dims.py` 需要自行放入 `data/raw_pdfs/`。
