#!/usr/bin/env python3
"""
离屏渲染滚轮台热炉运转动画 (GIF)，用于无显示器的服务器 / GitHub Actions。
需设置 MUJOCO_GL=egl 或 MUJOCO_GL=osmesa。
"""

import argparse
import sys
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.run_cosim import CoSimRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster D51 离屏渲染动画")
    parser.add_argument("--xml", type=Path, default=Path("models/d51_full.xml"))
    parser.add_argument("--out", type=Path, default=Path("logs/d51_running.gif"))
    parser.add_argument("--warmup", type=float, default=3.0, help="开始录制前的运转时间 (s)")
    parser.add_argument("--duration", type=float, default=1.6, help="录制的仿真时长 (s)")
    parser.add_argument("--sample-fps", type=float, default=100.0, help="仿真时间内的采样帧率")
    parser.add_argument("--play-fps", type=float, default=25.0, help="GIF 播放帧率 (低于采样帧率即为慢放)")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    runner = CoSimRunner(args.xml, preset="hot")
    c = runner.ctrl
    c.blower_valve = 0.0
    c.drain_cock = False
    c.reverser = 0.78
    c.regulator = 0.28

    model, data = runner.model, runner.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    cam = mujoco.MjvCamera()
    cam.azimuth = 135.0
    cam.elevation = -16.0
    cam.distance = 0.36
    cam.lookat[:] = [0.04, 0.0, 0.03]

    dt = model.opt.timestep
    for _ in range(int(args.warmup / dt)):
        runner.step_once()

    steps_per_frame = max(1, int(round(1.0 / (args.sample_fps * dt))))
    n_frames = int(args.duration * args.sample_fps)
    slowmo = args.sample_fps / args.play_fps
    frames = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for _ in range(n_frames):
            for _ in range(steps_per_frame):
                tel = runner.step_once()
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            ImageDraw.Draw(img).text(
                (8, 8),
                f"t={tel.sim_time:5.2f}s  {tel.wheel_rpm:+6.1f} RPM  boiler {tel.boiler_pressure_kgf:4.2f} kgf/cm2"
                f"  ({slowmo:.0f}x slow)",
                fill=(255, 255, 255),
            )
            frames.append(img.quantize(colors=128))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.play_fps), loop=0, optimize=True
    )
    print(f"[DONE] {len(frames)} 帧动画已保存至: {args.out}")


if __name__ == "__main__":
    main()
