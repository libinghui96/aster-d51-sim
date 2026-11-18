# Aster Hobby JNR D51 Live-Steam 3D Digital Twin

Aster Hobby 1 番ゲージ (`1/32`, `45.0 mm` 轨距, `1-D-1 Mikado`) JNR D51 真蒸汽机车的 **STEP-First 101 件全水密三维空心管阀 CAD + 578 万体素七域真 3D 多相流道/多材料共轭传热（热传导·热对流·热辐射·陶瓷保温·灯芯毛细燃烧）+ 六算子 GPU PINO + 2000 Hz MuJoCo 华氏闭环多体动力学** 双向紧耦合数字孪生仿真。

- **在线 Web 3D 实时交互数字孪生 (WASM + Three.js，由 GitHub Actions 自动构建发布):** https://libinghui96.github.io/aster-d51-sim/
- **CI 试运转遥测与离屏动画报告:** https://libinghui96.github.io/aster-d51-sim/report.html

![running](https://libinghui96.github.io/aster-d51-sim/d51_running.gif)

![telemetry](https://libinghui96.github.io/aster-d51-sim/d51_telemetry.png)

## 核心物理与工程特性

1. **STEP-First 精确 B-Rep 解析建模与全流道贯通 (`models/step/*.step` -> `models/meshes/*.stl`)**：
   - 全车 **101 件** 刚体与管阀均以 `ISO-10303-21 (.step)` 精确解析曲面为唯一几何真值源（`scripts/step_pipeline.py`），直接由 OpenCascade B-Rep 拓扑积分材料质量、偏心质心与惯量张量，并离散编译运行时水密单连通网格（`boundary_edges == 0, connected_components == 1`）及 40 组跨刚体零断口流体/机械端口。
   - 全功能管道、三通、Banjo 活接、针阀/旋塞/止回阀座及 `23-1` 三杯酒精燃烧器均按原厂壁厚还原真实三维空心内腔（Lumen），并通过 `BRepClass3d_SolidClassifier` 与 GPU 三维入口种子漫水（Flood-Fill）双重校验：任何管路盲堵均自动导致物理流量与火焰归零。
2. **七域 578 万体素真 3D 多相热流、陶瓷灯芯毛细吸醇与多层共轭传热 (`sim/thermo_3d_gpu.py`)**：
   - 还原原厂 **`セラミックヤーン(バーナー芯)`** 硅酸铝多孔陶瓷纤维灯芯（3 杯 × 约 30 股，`ø10.0 × 11.7 mm`）的 Darcy-Washburn 毛细吸液、相变汽化与断醇干烧赤热（`>520 °C`）过程。
   - 按 `C1100` 无氧紫铜锅筒/火室/5 烟管、`SUS304` 双 U 四程过热器、**`Templates 1–6` 全套 `セラミックシート(断熱材)` 陶瓷纤维保温层**、空气夹层与烤漆黄铜外衣的真实尺寸及物性，耦合求解三维固体导热、自然/强迫对流及 Stefan-Boltzmann 热辐射，第一性原理涌现真实锅炉热效率（$\eta_{\text{boiler}}$）与拆除外衣保温层后的散热倍增效应。
   - 采用 **L1 IAPWS-IF97 解析守恒底座 (`2000 Hz`) + L2 六算子 GPU PINO 神经算子套件 (`1.56M` 参数, `BF16` Tensor Cores) + L3 本地 `bonsai-2:27b` 异步认知诊断 (`0.5 Hz`)** 三层混合架构，严禁任何经验 `if` 分流作弊系数。
3. **2000 Hz MuJoCo 华氏配气闭环与全可动件 1:1 还原 (`models/d51_full.xml` & `models/d51_r2m.xml`)**：
   - 精确还原外进气平面 D 型滑阀、螺旋逆转机与前弯 20% 水平逆转拉杆 `8-18`、驾驶室室内立式排液手柄 `8-7`、后拉三孔弧形汽笛柄 `11-3`、蓄压环腔突跳双安全阀 `28-6×2`、蒲鉾形砂箱铰链盖 `29-12` 及煤水车四连杆手摇泵/马氏瓶恒液位供醇系统。

## 本地运行

### 1. STEP-First CAD 建模、MJCF 编译、六算子 PINO 与门禁验证
```bash
pip install -r requirements.txt
python scripts/02_build_cad_meshes.py                           # 构建 101 件 ISO-10303-21 STEP 主模型并编译水密 STL 与材料惯量库
python scripts/03_generate_mjcf.py --mode roller                # 编译滚轮台模型 -> models/d51_full.xml
python scripts/03_generate_mjcf.py --mode curve_r2m --out models/d51_r2m.xml
python scripts/10_train_pino_surrogate.py                       # 生成收敛 3D PDE 数据并训练 6 算子 PINO 套件 -> models/pino_d51_surrogate.pt
python scripts/09_verify_cad_and_clearance.py                   # 运行 Gates 1–22 全量 STEP 贯通/多图纸证据/360° 净空/热物理门禁
python -m sim.run_cosim --headless-demo --gpu-3d                # 启动 CUDA 578万体素 + 六算子 PINO + MuJoCo 全流程联合试运转
python scripts/04_analyze_r2m_derailment.py                     # R2m vs R3m 先导支点过弯 Nadal 脱线分析
MUJOCO_GL=egl python scripts/04_render_animation.py             # 离屏渲染 -> logs/d51_running.gif
```

### 2. Web 3D 实时交互前端（支持直连本地 RTX 5080 Live 3D WebSocket 推流桥或浏览器离线 WASM）
```bash
python scripts/05_sync_web_assets.py                            # 同步 XML 与 101 件 STEP 编译 STL 至 web/public/models/
cd web && npm install && npm run dev                            # 启动 Web 3D 前端 (http://localhost:5173)
# 同时启动本地 GPU 578万体素 + 六算子 PINO 实时 WebSocket 推流服务 (ws://127.0.0.1:8765/ws):
npm --prefix web run build && python -m sim.gpu_3d_server --scene roller --preset hot
```

`data/` 下的原厂说明书扫描 PDF 不随仓库分发；运行 `scripts/01_extract_pdf_dims.py` 需先将原厂扫描件放入 `data/raw_pdfs/`。
