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
- **双路给水闭环**：第 3 轴偏心轮驱动轴动给水泵（柱塞 `ø3.2 mm`, `PN3.2-1`），配合旁通阀 (`4-17`) 分流回水以维持水面计 **50% 水位**；驻车手摇泵（`ø7.0 mm`, `PN7-1.9`）摇动约 30 次补满半锅。
- **非功能性外观件 (Dummy Parts)**：空气压缩机 (`22-11`)、储风缸、闸瓦 (`5-1~5-4`)、砂箱、发电机及重装备版集烟装置仅具静态质量与外观，严禁为其添加多余的动力学自由度。

---

## 3. MuJoCo MJCF 闭环建模与命名契约
为了保证 `thermo_1d.py`、`run_cosim.py` 以及未来 Web 端 (`mujoco_wasm`) 渲染器不发生断链，以下关节、执行器与传感器名称为**锁定契约（Contract）**，重构网格或扩充细节时不得改名：
- **核心关节 (`joint`)**：`j_wheel_1`..`j_wheel_4`, `j_piston_L`, `j_piston_R`, `j_valve_L`, `j_valve_R`, `j_reverser`, `j_pump_ram`
- **核心执行器 (`actuator`)**：`steam_force_L`, `steam_force_R`, `reverser_ctrl`
- **核心传感器 (`sensor`)**：`piston_pos_L`, `piston_vel_L`, `valve_pos_L`, `piston_pos_R`, `piston_vel_R`, `valve_pos_R`, `wheel3_omega`, `pump_ram_vel`
- **闭环处理铁律**：MuJoCo `<worldbody>` 保持树状无环结构，所有闭环连杆（主连杆—十字头、四轴耦合连杆、偏心杆—月牙板、心向棒—合并杆—滑阀叉头、吊杆—心向棒、给水泵偏心连杆）一律通过 `<equality><connect .../></equality>` 球铰点重合约束闭合，避免平面 `weld` 造成过约束奇异。

---

## 4. 常用开发与验证命令
- 重新提取正交图纸与几何配置：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --export-default`
- 交互式鼠标点选校准华氏连杆孔距：`python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --interactive`
- 重新编译全车 MJCF（滚轮台模式 / 轨道模式）：`python scripts/03_generate_mjcf.py --mode roller` （或 `--mode track`）
- 启动 3D 交互热机联调（热炉即发）：`python -m sim.run_cosim --xml models/d51_full.xml --preset hot`
- 无 GUI 自动化全流程试车验收与绘图：`python -m sim.run_cosim --xml models/d51_full.xml --headless-demo`

---

## 5. 进阶精度升级契约（材料惯量、热损伤与 GPU 标定）
- **禁止统一空壳质量兜底**：所有运动刚体（含平面滑阀 `1-6`+`1-22`、十字头 `1-31`、月牙滑块 `2-3`、逆转横轴 `4-12`、动轮/煤水车轴箱 `5-9`/`31-22`）均须由 `02_build_cad_meshes.py` 按原厂金属密度（黄铜 `8500 kg/m³`、不锈钢 `7930 kg/m³`、铸铁 `7350 kg/m³`）积分体积、偏心质心与惯量张量，输出至 `configs/d51_mass_props.json` 供 `03_generate_mjcf.py` 注入 `<inertial>`。
- **机构与极限工况扩展契约**：
  - 月牙槽滑块采用 `R = radius_rod_length` 圆弧等效铰接，确保死点恒定先导量；
  - 动轮垂向悬挂通过 `<tendon>` 耦合模拟 `イラスト 3` 跷跷板均衡梁（Equalizer）；
  - `thermo_1d.py` 维护铜火室壁温 `T_crown_c`、密封完好度与气缸液态积水体积，支持空焚き（干烧软化/闪蒸）与冷缸水锤（Hydrolock）判定；
  - 若 `configs/smokebox_lut.json` 存在，优先通过双线性插值读取 RTX 5080 Warp 3D CFD 标定的烟室引风与火管换热功率。

---

## 6. 开发与多模态文件交互纪律（严禁循环读取高清扫描图）
- **严禁批量/循环直接读取高分辨率图片或 PDF**：在本地或远程 SSH 的 IDE Agent 会话中，**绝对禁止**循环调用文件读取工具遍历 `data/ortho_views/*.png`、`logs/*.png` 或 `data/raw_pdfs/*.pdf`。多轮对话会将已读高清图片反复打包重发（Vision Token 滚雪球效应），极易触发 API 限流重试（`Model unavailable`）与远程传输卡死。
- **本地脚本先行提取纯文本**：凡需从 PDF/切片提取零件表（BOM）、文字说明、尺寸标注或自动图像测量（如霍夫圆/边缘检测），一律编写本地 Python 脚本（`pymupdf` / `cv2` / `numpy`）在毫秒级处理并输出为纯文本 `.json` / `.md` 后再读取。
- **最小化视觉确认**：日常 CAD 建模、MJCF 编译与热流联调一律围绕纯文本代码（`.py` / `.json` / `.xml`）与数值遥测指标闭环迭代；仅在确需人工/视觉核对单一局部细节时，先通过脚本裁剪并降采样出低分辨率局部 ROI 单图，或交由独立子代理单次读取并仅返回纯文本结论。