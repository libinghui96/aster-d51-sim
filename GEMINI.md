# Aster Hobby JNR D51 真蒸汽机车物理与热力学仿真项目规范 (GEMINI.md)

## 1. 项目目标与架构概述
本项目旨在对 **Aster Hobby JNR D51（1/32 比例，Gauge 1 45mm 轨距，1-D-1 Mikado 轴式真蒸汽机车）** 进行高精度三维机械结构还原，并实现 **1D 热流体网络（燃烧、相变蒸发、过热、平面滑阀配气、轴动给水）** 与 **3D 多体动力学（MuJoCo 闭环连杆与轮轨接触）** 的双向紧耦合实时仿真。

采用三层解耦架构：
1. **图纸几何真值层 (`configs/d51_kinematics.json` & `scripts/01_extract_pdf_dims.py`)**：从扫描栅格版组装图 PDF 导出纠偏后的 300 DPI 正交切片（位于 `data/ortho_views/`），提取毫米级孔距与轴距。**注意：原始 PDF 为扫描栅格图像而非原生 CAD 矢量图，请勿使用 `page.get_drawings()` 作为唯一提取手段。**
2. **多体模型编译层 (`scripts/03_generate_mjcf.py` -> `models/d51_full.xml`)**：禁止手动硬编码修改 `d51_full.xml` 中的对称刚体坐标！必须通过 `03_generate_mjcf.py` 内置的解析几何求根器（`solve_side_pose`）计算左右两侧在 $t=0$ 时的零残差位姿后自动生成 XML。
3. **1D-3D 联合仿真层 (`sim/thermo_1d.py` & `sim/run_cosim.py`)**：以 $\Delta t = 0.5\text{ ms}$ ($2000\text{ Hz}$) 同步步进。MuJoCo 解算运动学与闭环约束力，`thermo_1d.py` 读取活塞/滑阀传感器状态并回写左右活塞蒸汽推力与给水泵液压反力。

---

## 2. 图纸与运转手册提取的物理硬约束（严禁随意篡改）
修改任何物理或几何参数时，必须严格遵守原厂图纸与标准件锁死的真值：
- **轨距与曲线半径**：轨距 `45.0 mm`，最小通过曲线半径 `R2m (2000 mm)`；先导台车支点 (`20-6`) 含 `10.0 mm` 前后双孔调节。
- **主气缸与活塞 (`イラスト 1`)**：活塞环为 `PR-15`，锁死主气缸内径 **Bore = 15.0 mm**（无杆前腔面积 $176.71\text{ mm}^2$）；活塞杆密封为 `PF3-1.9`，锁死杆径 **3.0 mm**（有杆后腔面积 $169.65\text{ mm}^2$）；阀芯杆密封为 `PF2-1.5`，杆径 **2.0 mm**。
- **左右曲柄正交相位 (`Quartering`, 图 9.5)**：第 3 动轮 (`5-7`) 为主动轮，右侧动轮曲柄相对左侧严格滞后 **90°**（`theta_L = 0°`, `theta_R = -90°`），确保每转 4 次交替排气且无启动死点。
- **华氏配气机构 (`Walschaerts Valve Gear`, 图 9.2 & イラスト 2/9)**：
  - 回动曲柄 (`7-10`) 偏心销孔距动轮中心严格为 **7.0 mm**，与主曲柄呈 **90°** 相位角。
  - 采用外进气平面 D 型滑阀 (`1-6`)，中立位 `N` 时滑块 (`2-3`) 位于月牙加减链 (`2-4`) 定轴旋转中心。
- **动轮横动游隙 (图 5.1)**：第 1 动轮 (`5-5`) 轴箱内距 `8.5 mm`，第 4 动轮 (`5-8`) 轴箱内距 `6.5 mm`（首轴具备额外横动量以适应弯道）。
- **锅炉、燃烧与安全阀 (`試運転` 章节)**：
  - 锅筒前管板内衬直径 `ø53.0 mm` (`Template 6`)，烟室外径约 `64.9 mm`、长 `70.0 mm` (`Template 5`)，火室 `65 × 49 × 47 mm` (`Template 2`)。
  - 煤水车酒精箱容量 **130 ml**（恒液位サンプ供油，满载续航约 **20 分钟**）；前车架置换式注油器续航约 **60 分钟**。
  - 常用工作表压 **3.0 ~ 4.0 kgf/cm²**；双安全阀 (`28-6`) 在 **4.0 kgf/cm²** 起跳泄压，回落至 **3.5 ~ 3.6 kgf/cm²** 闭合；底盘冷态气密测试 (`エアテスト`) 在 **1.0 kgf/cm²** 气压下必须能顺畅正反转。
- **双路给水闭环 (`イラスト 4, 6, 13, 31`)**：第 3 轴偏心轮驱动中央轴动给水泵 (`6-5`, 柱塞 `ø3.2 mm`, `PN3.2-1`, 双 `SB-ø3` 止回钢球)，全程在车架内侧 (`|Y| ≤ 11 mm`) 经 `6-6/6-8` 向后送入锅炉后上部逆止阀 (`13-7`)，配合旁通阀 (`4-17`) 回水维持水面计 **50% 水位**；驻车手摇泵 (`31-5`, `ø7.0 mm`, `PN7-1.9`) 经 `13-11` 顶开火室底部独立逆止阀 (`13-2`) 摇动约 30 次补满半锅。
- **置换式注油器与 5 烟管/4 头燃烧器真值 (`イラスト 4, 12, 17, 23, 25`)**：锅筒内置 **5 根烟管**（上 2 粗过热烟管插 `17-5` 双 U 四程过热器 + 下 3 细火管），驾驶室后壁加减阀 (`11-10/11`) 供汽；火室底配 `2×2` 阵列 **4 头立式 `10 mm` 燃烧器 (`23-1`)**；前车架中央内置卧式置换油箱 (`4-2`, 顶盖 `4-1` 位于前甲板活门 `25-3` 下，无底塞，靠蒸汽凝结沉水等体积顶出蒸汽油汇入 `13-4/7-9` 给汽管)。
- **非功能性外观件 (`Dummy Parts`)**：假逆止阀 (`15-3 ダミー逆止弁`, 严禁接水管！)、假排污阀 (`16-2`)、车顶假汽笛 (`30-2`, 真汽笛 `11-8` 位于车底)、空气压缩机 (`22-11`)、储风缸 (`22-13`)、闸瓦 (`5-1~5-4`)、砂箱、发电机及除烟板 (`28-11/12`) 仅具静态质量与外观，严禁为其添加多余的动力学自由度。

---

## 3. MuJoCo MJCF 闭环建模与命名契约
为了保证 `thermo_1d.py`、`run_cosim.py` 以及未来 Web 端 (`mujoco_wasm`) 渲染器不发生断链，以下关节、执行器与传感器名称为**锁定契约（Contract）**，重构网格或扩充细节时不得改名：
- **核心关节 (`joint`)**：`j_wheel_1`..`j_wheel_4`, `j_piston_L`, `j_piston_R`, `j_valve_L`, `j_valve_R`, `j_reverser`, `j_pump_ram`
- **核心执行器 (`actuator`)**：`steam_force_L`, `steam_force_R`, `reverser_ctrl`
- **核心传感器 (`sensor`)**：`piston_pos_L`, `piston_vel_L`, `valve_pos_L`, `piston_pos_R`, `piston_vel_R`, `valve_pos_R`, `wheel3_omega`, `pump_ram_vel`
- **三级精密闭环处理铁律**（详见 `docs/LESSONS_LEARNED.md`）：
  1. **连杆销孔闭环**：MuJoCo `<worldbody>` 保持树状无环结构，连杆销孔闭环（主连杆—十字头、四轴耦合连杆、偏心杆—月牙板、心向棒—合并杆—滑阀叉头、吊杆—心向棒、给水泵偏心连杆）通过 `<equality><connect .../></equality>` 重合约束闭合（严禁对转动销孔使用锁死旋转的 `weld`），并对主传力与滑阀精密回路收紧 `solref/solimp` 确保常态重合残差 `< 0.02 mm`；
  2. **跨浮动轮对真实销套游隙**：凡跨越“浮动动轮（含轴箱横动 `j_lat_y` / 垂向悬挂 `j_susp_z`）↔ 固定车架”边界的连杆（主连杆大头 `7-11`、偏心杆 `2-12/19`、耦合连杆 `7-12/15`、轴动泵偏心环 `6-3`），必须在曲柄销侧还原原厂 `±0.25~0.4 mm` 受限轴向浮动游隙或微球面顺应度，严禁 `<connect>` 横向分量反向拽死动轮过弯横动；
  3. **齿轮啮合与定比操纵副**：对气缸排液阀齿轮旋塞 (`1-16 / 1-17`) 及逆转长拉杆定比联动 (`8-12 / 8-18`)，允许采用零齿隙 `<equality><joint .../>` 多项式耦合约束。
  4. **1D-3D 拓扑—几何连续性与跨刚体全端口零断口门禁 (`Gate 9`)**：由于 1D 集总热流网络沿逻辑拓扑图积分，单独检查 `boundary_edges == 0` 无法捕获三维断管或悬空碎块；因此每次提交必须通过 `scripts/09_verify_cad_and_clearance.py` 的 **Gate 9**：① 全车所有 `.stl` 必须为单连通固体 (`connected_components == 1`)；② `configs/*_boiler_piping.json` 声明的全部跨刚体/跨网格流体与机械端口在直线及最小曲线半径 (`R2m`) 双姿态下表面局部 ROI 对接间隙必须 `<= 0.50 mm`（跨车柔性软管须随牵引杆铰接摆动，严禁悬空硬伸）。

---

## 4. 常用开发与验证命令
- 重新提取正交图纸与无头像素标定：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --auto-calibrate`
- 交互式鼠标点选校准华氏连杆孔距：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --interactive`
- 重新提取锅炉 5 烟管/双水路/置换注油器/4 头燃烧器真值表：`python scripts/08_extract_boiler_valves_bom.py`
- 重新构建全车 CAD 网格与材料惯量库：`python scripts/02_build_cad_meshes.py`
- 重新编译全车 MJCF（滚轮台 / R2m 弯道）：`python scripts/03_generate_mjcf.py --mode roller` （或 `--mode curve_r2m --out models/d51_r2m.xml`）
- 运行 CAD 拓扑完整性、对称性与 360° 运动包络净空门禁：`python scripts/09_verify_cad_and_clearance.py`
- R2m vs R3m 先导支点过弯脱线分析：`python scripts/04_analyze_r2m_derailment.py`
- RTX 5080 烟室 3D CFD 标定查找表生成：`python scripts/06_gpu_smokebox_cfd.py`
- RTX 5080 并行参数辨识与自动司炉寻优：`python scripts/07_gpu_fireman_opt.py`
- 启动 3D 交互热机联调（热炉即发）：`python -m sim.run_cosim --xml models/d51_full.xml --preset hot`
- 启动 RTX 5080 571万体素真 3D 零件级多相热流联调：`python -m sim.run_cosim --xml models/d51_full.xml --preset hot --gpu-3d`
- 启动本地 RTX 5080 Live 3D WebSocket 推流桥（供浏览器 60 FPS 直连）：`python -m sim.gpu_3d_server --scene roller --preset hot`
- 无 GUI 自动化全流程试车验收与绘图：`python -m sim.run_cosim --xml models/d51_full.xml --headless-demo --gpu-3d`
- 同步资产并构建 Web 3D 实时交互前端：`python scripts/05_sync_web_assets.py && npm --prefix web run build`

---

## 5. 进阶精度升级契约（材料惯量、机构升级、极限热物理与 GPU 标定）
完整分阶段路线图详见 `docs/DEVELOPMENT_PLAN.md`：
- **O1 图纸无头像素标定**：支持在不依赖 X11 弹窗的远程环境下，从 `イラスト 9`、`Template 6` 及实车正交底/侧视图自动标定比例尺并交叉核验孔距/轴距（输出 `configs/d51_calibration_report.json`）。
- **O2 材料密度积分惯量（禁止空壳兜底）**：所有运动刚体（含平面滑阀 `1-6`+`1-22`、十字头 `1-31`、月牙滑块 `2-3`、逆转横轴 `4-12`、动轮/煤水车轴箱 `5-9`/`31-22`）均须由 `02_build_cad_meshes.py` 按原厂金属密度（黄铜 `8500 kg/m³`、不锈钢 `7930 kg/m³`、铸铁 `7350 kg/m³`）积分体积、偏心质心与惯量张量，输出至 `configs/d51_mass_props.json` 供 `03_generate_mjcf.py` 注入 `<inertial>`。
- **O3 机械机构三大硬精度升级**：
  1. **圆弧月牙槽**：月牙槽滑块采用 `R = radius_rod_length` 圆弧等效铰接，确保前/后死点切换正反转时保持恒定先导量（Constant Lead）；
  2. **滑阀背压摩擦**：根据蒸汽室表压压紧外进气平面 D 型滑阀的法向载荷与注油润滑状态，实时计算滑动摩擦反力回写 `j_valve_L` / `j_valve_R`；
  3. **动轮跷跷板均衡梁**：动轮垂向悬挂 `j_susp_z1..4` 通过 `<tendon>` 耦合模拟 `イラスト 3` 跷跷板均衡梁（Equalizer）。
- **O4 锅炉热损伤、坡道假水位与冷缸水锤**：`thermo_1d.py` 维护分层紫铜火室顶板壁温 `T_crown_c`、坡道/加减速表观假水位、密封圈热退化完好度 `seal_health` 及左右气缸前/后腔液态冷凝水体积，支持空焚き（干烧退火软化 + 高温补水闪蒸压力尖峰）与未开排液阀时的冷缸不可压缩水锤（Hydrolock）冲击锁死判定。
- **O5 & O6 RTX 5080 GPU 烟室 CFD、571万体素零件级真 3D 多相热流引擎与自动司炉 Agent**：支持离线 LUT（`configs/smokebox_lut.json`）与本地实时 5.71M 体素 CUDA STL 体素化三维多相求解器（`sim/thermo_3d_gpu.py` + `sim/gpu_3d_server.py` WebSocket 推流桥，直接从 `.stl` 雕刻燃烧室/5烟管/沸腾自由水面/四程过热管/滑阀气缸/给水油路三维流体域并双向耦合 MuJoCo）；支持 `AutoFiremanAgent`（`sim/auto_fireman.py`）及全套零件级 GLSL 三维流体着色器（`web/src/fluidShaders.ts`）。

---

## 6. 开发与多模态文件交互纪律（严禁循环读取高清扫描图）
- **严禁批量/循环直接读取高分辨率图片或 PDF**：在本地或远程 SSH 的 IDE Agent 会话中，**绝对禁止**循环调用文件读取工具遍历 `data/ortho_views/*.png`、`logs/*.png` 或 `data/raw_pdfs/*.pdf`。多轮对话会将已读高清图片反复打包重发（Vision Token 滚雪球效应），极易触发 API 限流重试（`Model unavailable`）与远程传输卡死。
- **本地脚本先行提取纯文本**：凡需从 PDF/切片提取零件表（BOM）、文字说明、尺寸标注或自动图像测量（如霍夫圆/边缘检测），一律编写本地 Python 脚本（`pymupdf` / `cv2` / `numpy`）在毫秒级处理并输出为纯文本 `.json` / `.md` 后再读取。
- **最小化视觉确认**：日常 CAD 建模、MJCF 编译与热流联调一律围绕纯文本代码（`.py` / `.json` / `.xml`）与数值遥测指标闭环迭代；仅在确需人工/视觉核对单一局部细节时，先通过脚本裁剪并降采样出低分辨率局部 ROI 单图，或交由独立子代理单次读取并仅返回纯文本结论。