# Aster Hobby JNR D51 进阶物理精度与 GPU 数字孪生开发路线图

本文档整合了项目全量里程碑（`M1 ~ M4+`）与六大进阶物理/GPU 优化项（`O1 ~ O6`），对比代码库完成度并记录每阶段的闭环实测验收指标。

---

## 一、全量需求盘点与当前验收状态

| 编号 | 模块名称 | 核心技术内容 | 当前状态 | 对应脚本 / 配置 |
| :--- | :--- | :--- | :---: | :--- |
| **M1~M3** | **20 件 CAD 网格与 R2m 整列过弯基线** | `build123d` 构建机车 + `8-20B` 四轴煤水车共 20 件水密 STL；`curve_r2m` 环形弯道球面钢轨与轮轨位掩码接触；R2m vs R3m 先导支点脱线分析 | ✅ 已完成 | `scripts/02_build_cad_meshes.py`<br>`scripts/03_generate_mjcf.py`<br>`scripts/04_analyze_r2m_derailment.py` |
| **O1** | **图纸真实孔距无头自动像素标定** | 基于 `イラスト 9`（第 11 页正交配气图）、`Template 6`（`ø53mm`）与第 43~44 页实车正交底/侧视照片，免 GUI 自动完成比例尺校准与轴距/孔距交叉核验 | ✅ 已完成 | `scripts/01_extract_pdf_dims.py`<br>`configs/d51_calibration_report.json` |
| **O2** | **真实金属材质密度积分惯量** | 按黄铜 (`8500 kg/m³`)、不锈钢 (`7930 kg/m³`)、铸铁 (`7350 kg/m³`) 自动积分体积、偏心质心与 `3×3` 惯量张量，通过平行轴定理合成注入全车刚体 `<inertial>` | ✅ 已完成 | `configs/d51_mass_props.json`<br>`scripts/02_build_cad_meshes.py`<br>`scripts/03_generate_mjcf.py` |
| **O3** | **机械机构三大硬精度升级** | ① **圆弧月牙槽 (`R = radius_rod_length = 64.0 mm`)**：等效圆心铰接确保死点恒定先导量（Constant Lead）；<br>② **滑阀背压摩擦**：蒸汽室表压压紧 D 型滑阀产生的动态库仑/油膜摩擦反力；<br>③ **跷跷板均衡梁 (`<tendon>`)**：模拟 `イラスト 3` 动轮垂向悬挂耦合均载 | ✅ 已完成 | `scripts/03_generate_mjcf.py`<br>`sim/thermo_1d.py`<br>`sim/run_cosim.py` |
| **O4** | **锅炉干烧热损伤 + 坡道假水位 + 冷缸水锤** | ① 分层紫铜火室顶板壁温 `T_crown_c`、干烧（空焚き）退火软化与银焊损伤；<br>② 纵坡俯仰角/加减速引起的表观假水位与顶板裸露判定；<br>③ 干烧高温补水瞬间闪蒸压力尖峰与密封圈炭化漏汽；<br>④ 未开排液阀时冷缸液态积水不可压缩水锤（Hydrolock）冲击锁死 | ✅ 已完成 | `sim/thermo_1d.py`<br>`sim/run_cosim.py` |
| **O5** | **本地 RTX 5080 烟室射流与双火管 3D CFD 标定** | 利用本地 RTX 5080 CUDA 对烟室（`ø64.9×70mm`）乏汽喷嘴 + 通风环高速射流文丘里卷吸负压及双火管/过热管换热进行 3D 并行流场标定（`165` 组网格 / `1,478` 万体素），导出 `smokebox_lut.json` 供 `thermo_1d` 双线性插值 | ✅ 已完成 | `scripts/06_gpu_smokebox_cfd.py`<br>`configs/smokebox_lut.json`<br>`sim/thermo_1d.py` |
| **O6** | **RTX 5080 并行参数辨识与自动司炉 Agent** | 基于 RTX 5080 CUDA 批量并行演化（`114,688` 条机车轨迹，等效 `1,911.5` 小时实车试运转）辨识最优省汽截止比与给水旁通前馈-反馈律，实现自动暖缸排液防 Hydrolock、稳压与恒 `50%` 水位 | ✅ 已完成 | `sim/auto_fireman.py`<br>`scripts/07_gpu_fireman_opt.py`<br>`configs/auto_fireman_policy.json` |
| **M4+** | **终极版 Web 3D 交互数字孪生全特性同步** | 将圆弧月牙槽、均衡梁、滑阀背压摩擦、`T_crown_c` 铜壁红热发光与告警、坡道假水位滑块、冷缸水锤告警、烟囱四拍白烟粒子、R2m 整列动态追车视角、`smokebox_lut.json` 查找表与一键 `AI 自動司炉` 完整同步至 `web/` 前端 | ✅ 已完成 | `scripts/05_sync_web_assets.py`<br>`web/src/thermo1d.ts`<br>`web/src/particles.ts`<br>`web/src/main.ts`<br>`web/index.html` |

---

## 二、分阶段闭环验收实测记录

1. **阶段 1：几何校准与机械机构硬精度升级 (`O1 + O2 + O3`)**
   - `O1` 无头像素标定：`Template 6` `ø53mm` 标定 `11.8415 px/mm`（偏差 `+0.26%`）；侧视整车先导轴中心 `168.59 mm`（理论 `168.0 mm`，残差 `0.59 mm`）；底视轮缘外跨 `47.52 mm`（`45.0 mm` 轨距）；`イラスト 9` 华氏配气回动曲柄偏心销 `7.0 mm`、中立位合并杆先导冲程 `2.901 mm`。
   - `O3-1` 圆弧月牙槽：前死点满幅切换逆转机 `[-100%, +100%]` 时滑阀先导漂移仅 `±0.029 mm`（峰峰值 `0.058 mm`，较直线槽提升 6 倍），`t=0` 闭环残差 `1.39e-14 mm`。
   - `O3-3` 跷跷板均衡梁：前后两组 `<tendon>` 耦合悬挂，单轴 `+1.5 mm` 冲击自动向邻轴均载；`R2m` 弯道整列脱线系数 `Y/Q = 0.78 < 0.80`（`SAFE`）对比 `R3m` 孔位 `Y/Q = 0.83`（`MARGINAL`）。
2. **阶段 2：锅炉干烧热损伤与气液两相极限物理 (`O4 + O3-2`)**
   - `O4-1` 坡道假水位：`+3.5°` 上坡时水面计表观水位升至 `78.0%`（真实静水位 `49.7%`），火室前缘浸没水深同步减浅。
   - `O4-2` 空焚き干烧与冷水闪蒸：低水位干烧紫铜火室顶板升至 `490.1°C` 累积退火损伤；高温补水瞬间触发 `flash_steam_spike`，锅筒表压瞬间跃升 `+0.74 kgf/cm²`（`3.71 -> 4.45 kgf/cm²`）顶开安全阀。
   - `O4-4` 冷缸水锤锁死与恢复：`22°C` 冷缸未开排液阀时积水 `1.33 g` 触发 `Hydrolock` 锁死（`RPM = 2.4`），打开排液阀排净凝结水后立即恢复至 `354.7 RPM`。
3. **阶段 3：RTX 5080 GPU 3D 烟室 CFD 标定与自动司炉寻优 (`O5 + O6`)**
   - `O5` 烟室 3D CFD：本机 `NVIDIA GeForce RTX 5080` `7.21s` 并行求解 `165` 组 `[40×40×56]` 三维流场，烟室负压覆盖 `6.5 ~ 147.7 Pa`，引风倍率 `0.231 ~ 1.257`，热效率 `8.1% ~ 43.4%`。
   - `O6` 自动司炉寻优：`47.35s` 并行评估 `114,688` 条轨迹，导出 `P_target = 3.86 kgf/cm²`、巡航省汽截止比 `55.9%`；在 `d51_r2m.xml` 整列 3D 闭环测试中自动开阀暖缸、随速收短截止比且安全阀白白泄压占比 `0.0%`。
4. **阶段 4：Web 3D WASM 数字孪生终极版同步 (`M4+`)**
   - `web/public/models/manifest.json` 完整挂载 `2` 个 MJCF、`20` 个 STL 及 `3` 个 JSON 配置（`d51_mass_props.json`、`smokebox_lut.json`、`auto_fireman_policy.json`），`tsc && vite build` 零告警通过，并在 `mujoco_wasm` 下完成 R2m 自动司炉巡航与水锤锁死/恢复闭环验证。
