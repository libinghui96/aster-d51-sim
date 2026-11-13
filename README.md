# Aster D51 live-steam co-simulation

Aster Hobby 1 番ゲージ JNR D51 活汽机车的 1D 锅炉/气缸热力学 + MuJoCo 华氏配气机构联合仿真。

**在线效果 (由 GitHub Actions 每次推送自动仿真并发布):** https://libinghui96.github.io/aster-d51-sim/

![running](https://libinghui96.github.io/aster-d51-sim/d51_running.gif)

![telemetry](https://libinghui96.github.io/aster-d51-sim/d51_telemetry.png)

## 本地运行

```bash
pip install -r requirements.txt
python scripts/03_generate_mjcf.py                      # 生成 models/d51_full.xml
python -m sim.run_cosim --headless-demo                 # 试运转全流程 -> logs/d51_telemetry.png
MUJOCO_GL=egl python scripts/04_render_animation.py     # 离屏渲染 -> logs/d51_running.gif
python -m sim.run_cosim --preset hot                    # 交互查看器 (需要图形显示环境)
```

`data/` 下的原厂说明书 PDF 不随仓库分发；`scripts/01_extract_pdf_dims.py` 需要自行放入 `data/raw_pdfs/`。
