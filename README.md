# Aster Hobby JNR D51 Live-Steam 3D Digital Twin

Aster Hobby 1 番ゲージ (`1/32`, `45.0 mm` 轨距, `1-D-1 Mikado`) JNR D51 真蒸汽机车的 **STEP-First 104 件全水密三维空心管阀与弹性走行部 CAD + 578 万体素八域真 3D 多相流道/多材料共轭传热/紫铜烟管热弹塑性形变 + 七算子 GPU PINO (`125 Hz`) + 2000 Hz MuJoCo 华氏闭环多体与真实轮轨/枕木接触动力学** 双向紧耦合数字孪生仿真。

- **在线 Web 3D 实时交互数字孪生 (WASM + Three.js，由 GitHub Actions 自动构建发布):** https://libinghui96.github.io/aster-d51-sim/
- **CI 试运转遥测与离屏动画报告:** https://libinghui96.github.io/aster-d51-sim/report.html

![running](https://libinghui96.github.io/aster-d51-sim/d51_running.gif)

![telemetry](https://libinghui96.github.io/aster-d51-sim/d51_telemetry.png)

## 核心物理与工程特性

1. **STEP-First 104 件精确 B-Rep 解析建模与全流道管壁包络校验 (`models/step/*.step` -> `models/meshes/*.stl`)**：
   - 全车 **104 件** 刚体、走行部叠板簧/均衡梁与空心管阀均以 `ISO-10303-21 (.step)` 精确解析曲面为唯一几何真值源（`scripts/step_pipeline.py`），直接由 OpenCascade B-Rep 拓扑积分材料质量、偏心质心与惯量张量，并编译水密单连通网格（`boundary_edges == 0, connected_components == 1`）及 40 组跨刚体零断口端口。
   - 全功能管道、实心盲头 `6-5` 中央轴动给水泵（含全长 `6-1` 黄铜柱塞 + 双 `SB-ø3` 止回钢球）、阶梯车削 `4-17` 旁通阀、`23.0 mm` 分汽包 `11-15`、四孔铸造蝴蝶式火室门 `12-1`、封闭蒲鉾形砂箱 `29-11` 及极坐标贴壁撒砂管均按原厂图纸 1:1 还原，并通过 `verify_step_wall_bounded_lumen`（内腔 `TopAbs_OUT` + 管壁 `TopAbs_IN`）与 GPU 三维漫水校验。
2. **八域 578 万体素真 3D 多相热流、黏温油醇流动、开口系多变膨胀与烟管热弹塑性形变 (`sim/thermo_3d_gpu.py` & `sim/thermo_elastic_gpu.py`)**：
   - 完整物理仿真水循环（`272/340 ml` 煤水车水箱、`6-5` 泵腔抽吸负压真空与压水轴功、`4-17` 旁通分流、`0 ml` 彻底干锅零虚假蒸发）、开口系第一定律气缸多变膨胀做功（实时验证华氏螺旋逆转机截止率节汽效率）、`1-16×4` 气液两相均相闪蒸排液、以及考虑温度黏度 $\mu(T)$/密度/蒸气压/坡度重力的 `4-2` 蒸汽油与 `36-4 -> 23-1` 乙醇毛细燃烧过程。
   - GPU 实时求解 `C1100` 紫铜 5 根烟管与火室顶板在干烧退火软化及蒸汽外压下的受限热膨胀弯曲、Timoshenko 薄壳椭圆扁化与永久蠕变挠度，双向反馈烟管流固耦合阻力并在 WebGL 顶点着色器中实时呈现。
   - 采用 **L1 IAPWS-IF97 解析守恒底座 (`2000 Hz`) + L2 七算子 GPU PINO 神经算子套件 (`1.60M` 参数, `125 Hz` `BF16`) + L3 本地 `bonsai-2:27b` 异步认知诊断 (`0.5 Hz`)** 架构，内置可选实时 GPU 监控器（`--gpu-monitor` / Web HUD `📊 GPU 監視`）。
3. **2000 Hz MuJoCo 华氏闭环、板簧均衡悬挂与真实轮轨/枕木脱轨制停动力学 (`models/d51_full.xml`, `d51_track.xml`, `d51_r2m.xml`)**：
   - 还原 `3-4×6` 铰接跷跷板均衡梁、`3-5×8` 动轮叠板簧、先导 Bissel 3-DOF 转向架与煤水车 `31-25/28` 板簧/螺旋簧在整车自重下的真实弹性形变，采用四区零缝隙 Gauge-1 踏面/轮缘碰撞体与实心枕木/道床碰撞阵列，杜绝脱轨后悬空幽灵滑行。

## 本地运行

### 1. STEP-First CAD 建模、MJCF 编译、七算子 PINO 与门禁验证
```bash
pip install -r requirements.txt
python scripts/02_build_cad_meshes.py                           # 构建 104 件 ISO-10303-21 STEP 主模型并编译水密 STL 与材料惯量库
python scripts/03_generate_mjcf.py --mode roller                # 编译滚轮台模型 -> models/d51_full.xml
python scripts/03_generate_mjcf.py --mode track --out models/d51_track.xml
python scripts/03_generate_mjcf.py --mode curve_r2m --out models/d51_r2m.xml
python scripts/10_train_pino_surrogate.py                       # 生成收敛 3D PDE 数据并训练 7 算子 PINO 套件 -> models/pino_d51_surrogate.pt
python scripts/09_verify_cad_and_clearance.py                   # 运行 Gates 1–23 全量 STEP 贯通/多图纸证据/360° 净空/热弹性与脱轨门禁
python -m sim.run_cosim --headless-demo --gpu-3d --gpu-monitor  # 启动 CUDA 578万体素 + 七算子 PINO + MuJoCo 全流程联合试运转与 GPU 监控
python scripts/04_analyze_r2m_derailment.py                     # R2m vs R3m 先导支点过弯 Nadal 脱线分析
MUJOCO_GL=egl python scripts/04_render_animation.py             # 离屏渲染 -> logs/d51_running.gif
```

### 2. Web 3D 实时交互前端（支持直连本地 RTX 5080 Live 3D WebSocket 推流桥或浏览器离线 WASM）
```bash
python scripts/05_sync_web_assets.py                            # 同步 3 个 XML 与 104 件 STEP 编译 STL 至 web/public/models/
cd web && npm install && npm run dev                            # 启动 Web 3D 前端 (http://localhost:5173)
# 同时启动本地 GPU 578万体素 + 七算子 PINO (125Hz) 实时 WebSocket 推流服务 (ws://127.0.0.1:8765/ws):
npm --prefix web run build && python -m sim.gpu_3d_server --scene roller --preset hot --gpu-monitor
```

`data/` 下的原厂说明书扫描 PDF 不随仓库分发；运行 `scripts/01_extract_pdf_dims.py` 需先将原厂扫描件放入 `data/raw_pdfs/`。
