# Aster Hobby JNR D51 真蒸汽机车物理与热力学仿真项目规范 (GEMINI.md)

## 1. 项目目标与架构概述
本项目旨在对 **Aster Hobby JNR D51（`1/30` 比例 — 日本国铁 `1067 mm` 窄轨适配 Gauge-1 `45 mm` 轨距，`2010_JNR_D51-standard.pdf` & `aster-manual.pdf` p. 6，1-D-1 Mikado 轴式真蒸汽机车）** 进行高精度三维机械结构还原，并实现 **1D/3D 热流体与气动声学网络（`Type "C"` 干式陶瓷内衬火室二次辐射燃烧、相变蒸发、过热、平面滑阀配气与 `0.50 mm` 垂向抬升泄压、`11-8` 停止管两相流汽笛声学、轴动给水）** 与 **3D 多体动力学（MuJoCo 闭环连杆与 10 轴纯滚动轮轨接触）** 的双向紧耦合实时仿真。

采用三层解耦架构：
1. **图纸几何真值层 (`configs/d51_kinematics.json` & `scripts/01_extract_pdf_dims.py`)**：从扫描栅格版组装图 PDF 导出纠偏后的 300 DPI 正交切片（位于 `data/ortho_views/`），提取毫米级孔距与轴距。**注意：原始 PDF 为扫描栅格图像而非原生 CAD 矢量图，请勿使用 `page.get_drawings()` 作为唯一提取手段。**
2. **多图纸交叉印证与全可动/行为影响件 1:1 真实还原准则（最高建模优先级）**：图纸中**所有实际能动的零件**（外进气平面滑阀 `1-6` 与驱动滑块 `1-5`、活塞 `1-28` 与十字头 `1-31/35`、螺旋逆转机 `8-12~20` 双柄曲柄/行走螺母/拱曲扁钢拉杆 `8-18` 与锅底绕行横轴 `4-12`、排液拉杆 `8-5/8-6` 与叠搭剪式摇板 `1-19`、锅炉排污针阀 `11-16/17`、加减/通风/汽笛/旁通阀、蝴蝶式火室门 `12-1/2`、煤水车可提水箱盖 `36-1/2`、醇箱盖 `36-5` / 针阀 `36-6`、摆动支点四连杆手摇泵 `31-1~7` 与排水栓 `31-11` 等）以及**所有影响流体、热力、配气或运动包络行为的零件与管道**，均须进行 1:1 真实还原。若单张图纸无法唯一确定前后层级、螺纹轴向、弯折轨迹或局部截面，**必须将多张关联组装图（`イラスト 1–37`）、局部放大子图（`図 x.y`）、1:1 模板、六面实车正交照片与《部品表 / 組立順序》放在一起联合推导**，准确还原零件的尺寸、形状、相对位置关系及真实物理相互作用。
3. **多体模型编译层 (`scripts/03_generate_mjcf.py` -> `models/d51_full.xml`)**：禁止手动硬编码修改 `d51_full.xml` 中的对称刚体坐标！必须通过 `03_generate_mjcf.py` 内置的解析几何求根器（`solve_side_pose`）计算左右两侧在 $t=0$ 时的零残差位姿后自动生成 XML。
4. **1D-3D 联合仿真层 (`sim/thermo_1d.py` & `sim/run_cosim.py`)**：以 $\Delta t = 0.5\text{ ms}$ ($2000\text{ Hz}$) 同步步进。MuJoCo 解算运动学与闭环约束力，`thermo_1d.py` 读取活塞/滑阀传感器状态并回写左右活塞蒸汽推力与给水泵液压反力。

---

## 2. 图纸与运转手册提取的物理硬约束（严禁随意篡改）
修改任何物理或几何参数时，必须严格遵守原厂图纸与标准件锁死的真值：
- **轨距与曲线半径**：轨距 `45.0 mm`，最小通过曲线半径 `R2m (2000 mm)`；先导台车支点 (`20-6`) 含 `10.0 mm` 前后双孔调节。
- **主气缸与活塞 (`イラスト 1`)**：活塞环为 `PR-15`，锁死主气缸内径 **Bore = 15.0 mm**（无杆前腔面积 $176.71\text{ mm}^2$）；活塞杆密封为 `PF3-1.9`，锁死杆径 **3.0 mm**（有杆后腔面积 $169.65\text{ mm}^2$）；阀芯杆密封为 `PF2-1.5`，杆径 **2.0 mm**。
- **左右曲柄正交相位 (`Quartering`, 图 9.5 & `aster-manual.pdf` p. 22)**：第 3 动轮 (`5-7`) 为主动轮，右侧动轮曲柄相对左侧严格**超前 90°（`Right-Hand Lead`）**（`theta_L = 0°` 前死点时 `theta_R = -90°` 位于下死点，顺时针前进转动时右缸先达死点），确保每转 4 次交替排气且无启动死点。
- **华氏配气机构 (`Walschaerts Valve Gear` + `1-9` 交叉气道板, 图 9.2, イラスト 1/2/9 & `aster-manual.pdf` pp. 22–25 `Fig. 30–33`)**：
  - 回动曲柄 (`7-10`) 偏心销孔距动轮中心严格为 **7.0 mm**，按倾斜偏心杆补偿角滞后主曲柄 **`+94.64°`**，配合月牙尾偏角 `+3.15°` 实现前/中/倒全档位恒定正先导（漂移 `<= 0.05 mm`）。
  - 合并杆 (`1-23`) **顶销接心向棒 (`2-1`)、次上销接阀杆 (`1-22`, `Z = +13.5 mm`, 间距 `2.7 mm`)**，气缸内部采用 **Aster 专用 `1-9` 交叉气道板（Cross-Port Plate）**（后阀口交叉通前缸腔、前阀口交叉通后缸腔），匹配外进气平面 D 型滑阀 (`1-6`) 与中立死锁（`3.8 kgf/cm²` 下 `|RPM| < 1.0`）。
- **动轮横动游隙 (图 5.1)**：第 1 动轮 (`5-5`) 轴箱内距 `8.5 mm`，第 4 动轮 (`5-8`) 轴箱内距 `6.5 mm`（首轴具备额外横动量以适应弯道）。
- **锅炉、燃烧与安全阀 (`試運転` 章节)**：
  - 锅筒前管板内衬直径 `ø53.0 mm` (`Template 6`)，烟室外径约 `64.9 mm`、长 `70.0 mm` (`Template 5`)，火室 `65 × 49 × 47 mm` (`Template 2`)。
  - 煤水车酒精箱容量 **130 ml**（恒液位サンプ供油，满载续航约 **20 分钟**）；前车架置换式注油器续航约 **60 分钟**。
  - 常用工作表压 **3.0 ~ 4.0 kgf/cm²**；双安全阀 (`28-6`) 在 **4.0 kgf/cm²** 起跳泄压，回落至 **3.5 ~ 3.6 kgf/cm²** 闭合；底盘冷态气密测试 (`エアテスト`) 在 **1.0 kgf/cm²** 气压下必须能顺畅正反转。
- **双路给水闭环与跨车三平行软管真值 (`イラスト 4, 6, 8, 13, 31, 34, 37` & 手册 pp.6, 9, 39)**：第 3 轴偏心轮驱动中央轴动给水泵 (`6-5`, 柱塞 `ø3.2 mm`, `PN3.2-1`, 双 `SB-ø3` 止回钢球)，动轮段全程在车架内侧经 `6-6` 至驾驶室地板下方横置圆筒形旁通阀本体 (`4-17`, 左上穿地板立杆 `6-9 + 8-7` 手柄位于左驾 `+Y` 侧)，右端经 `6-8` 垂直立管与煤水车手摇泵 (`31-5`, `ø7.0 mm`, `PN7-1.9`, 双 `SB-ø4` 止回球) 汇流至锅炉后封头火室冠板上方右内侧中位逆止阀总成 (`13-2` 阀体 + `13-7` 闷盖 + `SB-ø3` 止回球，`Z=+47.5 mm`)，配合旁通阀 (`4-17`) 回水维持水面计 **50% 水位**；机车与煤水车之间严格为 **3 根平行直连跨车弹性软管（零交叉！）**（① 牵引杆上方中央 `ST-ø7` 半透明硅胶供醇管 `34-2 -> 23-1`；② 牵引杆下方右侧 `-Y` 煤水车 `34-6` 直连机车右后 `-Y` `6-11` 外螺纹口金的 `BT-ø5` 黑色氟橡胶给水/手摇泵共用管，机车端带 `34-7/34-8` 滚花黄铜活接与 `HC-1 ×2` 卡箍；③ 机车左后 `+Y` `4-16` 宝塔嘴直连煤水车左侧 `+Y` `34-6 -> 31-14` 高位弯头回水立管的 `BT-ø5` 无压回水管）；煤水车后部开放式水箱 (`31-10`) 满容约 `340 ml`、手册标准加注 **8 分满 (`≈272 ml`)**。
- **置换式注油器与 5 烟管/3 头直列燃烧器真值 (`イラスト 4, 11, 12, 17, 23, 25, 36`)**：锅筒内置 **5 根烟管**（上 2 粗过热烟管插 `17-5` 双 U 四程过热器 + 下 3 细火管），驾驶室后壁加减阀 (`11-10/11`) 供汽，逆转机与拉杆按日本国铁 **左驾 (`+Y`)** 布置；火室底配纵向中央 **`1×3` 直列 3 头立式 `10 mm` 燃烧器 (`23-1`)**（由煤水车 `36-4 -> 34-2` 马氏瓶恒液位サンプ供醇）；前车架中央内置卧式置换油箱 (`4-2`, 顶盖 `4-1` 位于前甲板活门 `25-3` 下，无底塞，靠蒸汽凝结沉水等体积顶出蒸汽油汇入 `13-4/7-9` 给汽管)。
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
  5. **「三层黄金分工」物理—算子—认知混合架构铁律 (`Gate 10 & Gate 17`)**：严禁在 3D 热流仿真中手写 `if not airtight: *= 0.05` 等经验漏气/分流系数，亦严禁让大语言模型参与高频 PDE 数值积分或用黑盒网络替代质量/能量守恒律！统一采用：① **L1 解析守恒底座（`2000 Hz`）**：IAPWS-IF97 水/水蒸气状态方程 $(v, u) \mapsto (P, T_{\text{sat}}, x)$ + 有限体积比焓相变与煤水车敞口槽二维浅水波严格守恒；② **L2 轻量级四算子 PINO 层（`250 Hz`, `BF16`）**：对燃烧—5烟管—烟室射流、滑阀—气缸窄缝流、驾驶室分汽包/水面计冲洗及煤水车双泵五止回球/马氏瓶耦合训练 `1.28M` 参数四算子套件（大气破口通过 $p|_{\Gamma_{\text{atm}}} = p_{\text{atm}}$ 硬投影自然涌现负压坍塌、回水切换与开盖漫油）；③ **L3 本地 LLM 慢速认知副驾（`0.5 Hz` 异步）**：由 Ollama `bonsai-2:27b` 异步提供故障根因诊断与操纵建议。
  6. **五层系统工程抽象架构（`docs/SYSTEM_ARCHITECTURE.md` & `docs/ASTER_LIVE_STEAM_PRIOR.md`）、`1-BOM-1-STEP` 108 件解析 B-Rep 建模、真 van Riemsdijk `Type "C"` 锅炉（`12-6` 退台紫铜锅筒 + `12-4 SUS304` 独立干式火室罩）、`11-8` 停止管两相流汽笛气动声学、全车 38 路双侧对称声明式流体网表三维体素 `Fluid-LVS`、全列车 10 轴等线速度滚动轮轨接触、Roscoe 内立管置换注油器、闭合第一定律桑基热平衡、旷量/差热膨胀与节点连续轴向温度场铁律 (`Gate 16–Gate 28`)**：严禁跨 BOM 编号私自吞并独立功能件（如 `12-4` 不锈钢火室罩、`4-17` 旁通阀体、`6-8` 立管、`4-16` 回水管均须独立建 `.step`）、严禁忽略操纵件全行程扫掠包络或把多件重叠 `.stl` 直接拼接做奇偶射线体素化（必须逐件体素化取 `torch.maximum` 真并集！）；具体零件几何以 `d51_illustrations.pdf` / `d51_manual.pdf` 为最高真值，系统机理与隐含结构遵循 `aster-manual.pdf` 与 `2010_JNR_D51-standard.pdf`！全车 **108 件** 刚体与管阀以 **ISO 10303-21 `.step` (`models/step/*.step`)** 为唯一几何真值源，并通过 **`configs/d51_fluid_topology.json` + `sim/fluid_topology_lvs.py`（`Gate 25`）** 在直线与 `R2m` 弯道双姿态下验证全车 38/38 管道 `Zero-Open / Zero-Necking / Zero-Missing-Wall`、8/8 互斥腔室 `Zero-Short` 零串腔及 8/8 动态阀态通断切换；还原 `Type "C"` 陶瓷火室二次辐射 + Hausen 烟管入口增强热效率、外进气滑阀框 `0.50 mm` 垂向抬升防顶缸泄压、`11-8` 车底朝下排凝黄铜汽笛 Wood-Kieffer 两相湿蒸汽声速与奇次谐波多普勒声学、`1:30` 换算车速与粘着系数 `[0.20, 0.33]`、全列车耦合 Davis 运行阻力与 10 轴纯滚动等线速接触、`17-3` 平滑 `180°` J 型烟室通风喷嘴（`通風ノズル`）、`4-2` 底部 `-Z` 引出 U 形毛细管 + 罐内垂直溢流内立管（Roscoe Standpipe）重力沉水置换物理、`図 20.1` `20-6` 前盖假杆 `R3m/R2m (+3.85 mm)` 双孔切换、`3-16 / 3-5 / 3-4 / 19-8 / 19-15` 全套受载弹簧、`図 5.1` 轴箱/心盘/连杆旷量与异种金属差热伸长、闭合第一定律锅炉桑基热平衡（残差 `< 0.2%`）；驾驶室前妻板 `21-28` 100% 封闭拱顶、左驾 `8-7` 逆时针 `0..+1.30 rad` 零穿模、右驾 `8-6/8-5` 100% 位于地板上方室内水平推拉；Web 透视采用解析 `G1` 圆弧导角无扭转 Bishop 管网、`20°C..750°C` 节点连续轴向热色谱与 38 路全分支密集流线型可逆导流镖；`0.0 ml` 干锅严格锁死 `0.00 g/s` 零蒸发与强迫通风自节流。

---

## 4. 常用开发与验证命令
- 检索 Aster 1975–2023 全系 110 款历史蒸汽机车规格目录并下载车型参考 PDF：`python scripts/12_fetch_aster_catalog.py --query D51 --download`
- 重新提取正交图纸与无头像素标定：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --auto-calibrate`
- 交互式鼠标点选校准华氏连杆孔距：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --interactive`
- 重新提取锅炉 5 烟管/双水路/置换注油器/4 头燃烧器真值表：`python scripts/08_extract_boiler_valves_bom.py`
- 重新构建全车 CAD 网格与材料惯量库：`python scripts/02_build_cad_meshes.py`
- 重新编译全车 MJCF（滚轮台 / 直线轨道 / R2m 弯道）：`python scripts/03_generate_mjcf.py --mode roller` （或 `--mode track --out models/d51_track.xml` / `--mode curve_r2m --out models/d51_r2m.xml`）
- 运行全车 38 路流体网表与 8 腔室隔离 GPU 三维体素 Fluid-LVS 验证：`python -m sim.fluid_topology_lvs --xml models/d51_track.xml --topology configs/d51_fluid_topology.json --out configs/d51_fluid_lvs_report.json`
- 运行五层系统架构、CAD 拓扑完整性、Fluid-LVS、10轴等线速滚动、Type-"C" 锅炉、汽笛声学、全锅炉气体/火焰动力学与 1:1 华氏交叉气道配气门禁（Gates 1–30）：`python scripts/09_verify_cad_and_clearance.py`
- R2m vs R3m 先导支点过弯脱线分析：`python scripts/04_analyze_r2m_derailment.py`
- RTX 5080 烟室 3D CFD 标定查找表生成：`python scripts/06_gpu_smokebox_cfd.py`
- RTX 5080 并行参数辨识与自动司炉寻优：`python scripts/07_gpu_fireman_opt.py`
- RTX 5080 离线收敛 3D PDE 数据生成与轻量级七算子 PINO 套件训练：`python scripts/10_train_pino_surrogate.py`
- 启动 3D 交互热机联调（热炉即发）：`python -m sim.run_cosim --xml models/d51_full.xml --preset hot`
- 启动 RTX 5080 578万体素八域 `125 Hz` 多相热流+热弹塑性联调（含可选 GPU 监控）：`python -m sim.run_cosim --xml models/d51_full.xml --preset hot --gpu-3d --gpu-monitor`
- 启动本地 RTX 5080 Live 3D WebSocket 推流桥（默认绑定 `0.0.0.0:8765`，自带旧进程接管与防伪终端断开 `_SafeTeeStream` 高可用保护，详见 `docs/LESSONS_LEARNED.md` 失误 21）：`python -m sim.gpu_3d_server --scene roller --preset hot`
- 无 GUI 自动化全流程试车验收与绘图：`python -m sim.run_cosim --xml models/d51_full.xml --headless-demo --gpu-3d`
- 同步资产并构建 Web 3D 实时交互前端：`python scripts/05_sync_web_assets.py && npm --prefix web run build`
- 同步本项目踩坑经验至全局及远端 `live-steam-twin` 插件（已由 `.agents/hooks.json` 与 `.git/hooks/{pre-commit,post-commit,pre-push}` 全自动触发）：`python scripts/11_sync_lessons_to_plugin.py --push-plugin`

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
- **跨车型经验实时双向继承纪律（`aster-d51-sim` → `live-steam-twin`）**：凡在本项目新发现通用踩坑经验并写入 `docs/LESSONS_LEARNED.md`，或沉淀了新的通用标定/CAD/门禁算子时，必须确保同步更新至全局插件 `~/.gemini/config/plugins/live-steam-twin/` 及项目镜像 `tools/live-steam-twin-plugin/`（日常修改、提交与推送均由 `.agents/hooks.json` 与 `scripts/install_git_hooks.sh` 自动联动同步并推送到 `libinghui96/live-steam-twin`）。