#!/usr/bin/env python3
"""
从 Aster D51 组装图 PDF 提取正交基准视图、校准比例尺并导出 d51_kinematics.json。
支持模式：
  1. 快速生成预标定参数并导出高清纠偏图纸：
     python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --export-default
  2. 启动 Matplotlib 交互式点击拾取器精修孔距与轴距：
     python scripts/01_extract_pdf_dims.py --pdf data/raw_pdfs/d51_illustrations.pdf --interactive
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np

# 关键正交页面索引 (0-based) 与扫描旋转纠偏角度 (顺时针度数)
KEY_PAGES = {
    "step01_cylinder_scale": (2, 90),    # イラスト 1: 底部含 15/21/24/26 号零件 1:1 mm 比例框
    "step07_rods_scale":     (8, 90),    # イラスト 7: 含十字头销 7-13 (ø3x7mm) 比例框
    "step09_valve_ortho":    (10, 90),   # イラスト 9: 华氏配气与主连杆纯 2D 正交侧视图 (含 5.8mm 标尺)
    "template_1_2_5":        (40, 90),   # Template 1/2/5: 1:1 绝对尺寸 (45mm, 70x78mm, 204x70mm)
    "template_3_4_6":        (41, 90),   # Template 6: ø53.0mm 标准圆 (自动霍夫圆 DPI 标定用)
    "photo_ortho_side":      (42, 0),    # 实车正交左右侧视图 (提取四轴轴距、先导/从轮坐标)
    "photo_ortho_bottom":    (43, 0),    # 实车正交仰视图 (利用 45mm 轨距校准照片比例尺)
}


def default_d51_kinematics() -> Dict:
    """
    结合手册硬性标准件参数 (PR-15, PF3-1.9, 图5.1, 图9.2, イラスト20)
    与 1/32 Gauge 1 (45mm 轨距) D51 标准型比例预标定的几何真值表 (单位: mm)。
    """
    return {
        "meta": {
            "model": "Aster Hobby JNR D51 Standard / Heavy-Equipped (1/32, Gauge 1)",
            "units": "mm",
            "track_gauge": 45.0,
            "min_curve_radius": 2000.0,
        },
        "wheels_and_axles": {
            # 以第 3 主动轮 (5-7 主動輪) 车轴中心为局部坐标系原点 X = 0.0, Z = 0.0
            "driving_wheel_diameter": 43.8,       # 实车 1400mm / 32 = 43.75mm
            "leading_wheel_diameter": 26.8,       # 先台車 (19-9) 轮径
            "trailing_wheel_diameter": 26.8,      # 従台車 (19-14) 轮径
            "axle_x": {
                "leading":  168.0,                # 先导轮轴 X 坐标
                "drive_1":   93.0,                # 第一動輪 (5-5)
                "drive_2":   46.5,                # 第二動輪 (5-6)
                "drive_3":    0.0,                # 主動輪   (5-7, 基准原点)
                "drive_4":  -46.5,                # 第四動輪 (5-8)
                "trailing": -118.0                # 从轮轴 X 坐标
            },
            # 图 5.1: 第1动轮轴箱间距 8.5mm vs 第4动轮 6.5mm (单侧赋予 ±1.0mm 额外横动游隙)
            "axle_lateral_play_half": {
                "drive_1": 1.2,
                "drive_2": 0.3,
                "drive_3": 0.2,
                "drive_4": 0.2
            },
            "leading_pivot_r2m_offset": 10.0,     # イラスト 20 (20-6): R2m 与 R3m 支点孔距 10mm
            "quartering_deg_R_minus_L": -90.0     # 图 9.5: 右侧曲柄滞后左侧 90° 正交相位
        },
        "cylinder_and_rods": {
            "bore": 15.0,                         # PR-15 活塞环锁死内径 15.0mm
            "piston_rod_dia": 3.0,                # PF3-1.9 密封圈锁死杆径 3.0mm
            "valve_rod_dia": 2.0,                 # PF2-1.5 密封圈锁死阀芯杆径 2.0mm
            "crank_radius": 10.0,                 # 主曲柄销偏心半径 10.0mm (活塞行程 Stroke = 20.0mm)
            "cylinder_center_x": 126.0,           # 气缸腔体中心 X 坐标
            "cylinder_center_y": 32.5,            # 左右气缸中心距车架中心线横向偏移 ±32.5mm
            "cylinder_center_z": 0.0,             # 主活塞轴线与动轮轴等高
            "valve_axis_z": 18.5,                 # 上部平面滑阀 (1-6) 滑动轴线高度 Z
            "main_rod_length": 92.0,              # 主連棒 (7-11/7-16) 大小头孔距
            "crosshead_pin_dia": 3.0              # 7-13 比例框: ø3.0 x 7.0mm
        },
        "walschaerts_valve_gear": {
            "return_crank_radius": 7.0,           # 图 9.2 硬指标: 回动曲柄孔距轮心精确 7.0mm
            "return_crank_phase_deg": -90.0,      # 图 9.2 硬指标: 与主曲柄呈 90° 夹角
            "trunnion_x": 46.5,                   # 月牙加减链 (2-4) 旋转定轴 X 坐标
            "trunnion_z": 15.5,                   # 月牙加减链 (2-4) 旋转定轴 Z 坐标
            "exp_link_tail_radius": 13.5,         # 月牙板定轴中心至底部偏心杆耳孔距离
            "die_slot_half_travel": 6.5,          # 滑块 (2-3) 在月牙槽内自中立位 N 上下最大行程
            "eccentric_rod_length": 48.0,         # 偏心棒 (2-12/2-19) 两端孔距
            "radius_rod_length": 64.0,            # 心向棒 (2-1) 前端至滑块孔距
            "radius_rod_back_ext": 10.0,          # 心向棒向后延伸连接吊杆 (2-10) 的尾端长度
            "crosshead_drop": 9.5,                # 十字头中心至下方短连杆铰接孔垂向距离
            "union_link_length": 15.0,            # 短连杆 (1-27 リンク) 孔距
            "comb_lever_total_length": 28.0,      # 合併てこ (1-23) 下端孔至最上端阀叉孔总长
            "comb_lever_lap_lead_offset": 3.5,    # 合併てこ最上孔(接阀杆)与次上孔(接心向棒)间距
            "reverser_shaft_x": 34.0,             # 逆転器アームシャフト (4-12) X 坐标
            "reverser_shaft_z": 26.5,             # 逆転器アームシャフト (4-12) Z 坐标
            "reverser_arm_length": 11.0,          # 逆転アーム (2-11/2-18) 摇臂长度
            "lift_link_length": 11.0              # 吊杆 (2-10 リンク) 孔距
        },
        "boiler_and_aux": {
            "boiler_front_id": 53.0,              # Template 6: ø53mm
            "smokebox_od": 64.9,                  # Template 5: 周长 204mm / pi
            "smokebox_length": 70.0,              # Template 5: 宽 70mm
            "firebox_lwh": [65.0, 49.0, 47.0],    # Template 2: 65 x 49 x (44~50)mm
            "axle_pump_bore": 3.2,                # PN3.2-1: 轴动给水泵柱塞直径 3.2mm
            "axle_pump_eccentric": 3.5,           # 第 3 轴给水泵偏心轮偏心距
            "axle_pump_x": -22.0                  # 轴動ポンプ (6-5) 泵体中心 X 坐标
        },
        "tender_and_track": {
            "drawbar_loco_pin_x": -146.0,
            "drawbar_length": 34.0,
            "drawbar_tender_pin_local_x": 116.0,
            "tender_body_length": 226.0,
            "tender_body_width": 74.0,
            "tender_body_height": 52.0,
            "tender_deck_z": 18.0,
            "bogie_center_span": 116.0,
            "bogie_wheelbase": 48.0,
            "tender_wheel_diameter": 26.8,
            "empty_mass_kg": 1.15,
            "curve_radius_mm": 2000.0,
            "curve_segments": 144
        }
    }


def extract_and_deskew_pages(pdf_path: Path, out_dir: Path, dpi: int = 300) -> Dict[str, Path]:
    """检测 PDF 矢量/栅格属性，并将关键正交页无损导出为正向 300 DPI PNG。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    exported: Dict[str, Path] = {}

    print(f"[INFO] 打开 PDF: {pdf_path.name}, 共 {len(doc)} 页")
    for tag, (page_idx, rot_deg) in KEY_PAGES.items():
        if page_idx >= len(doc):
            continue
        page = doc[page_idx]
        drawings = page.get_drawings()
        mode_str = f"矢量图元数={len(drawings)}" if drawings else "扫描栅格页 (Raster)"

        # 导出高分辨率栅格图并纠正扫描旋转
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if rot_deg == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot_deg == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot_deg == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)

        out_png = out_dir / f"p{page_idx + 1:02d}_{tag}.png"
        cv2.imwrite(str(out_png), img)
        exported[tag] = out_png
        print(f"  -> 已导出 {out_png.name} [{mode_str}] ({img.shape[1]}x{img.shape[0]} px)")

        # 若存在原生矢量路径，同步保存 SVG 供 CAD 直接读取
        if drawings:
            svg_path = out_dir / f"p{page_idx + 1:02d}_{tag}.svg"
            svg_path.write_text(page.get_svg_image(), encoding="utf-8")

    doc.close()
    return exported


def auto_calibrate_dpi_from_template6(template6_png: Path) -> float:
    """
    在 Template 6 页面自动检测 ø53.0mm 标准圆，计算扫描仪真实 px_per_mm。
    若霍夫圆未命中则回退至 300 DPI 理论值 (300 / 25.4 = 11.811 px/mm)。
    """
    theoretical_px_per_mm = 300.0 / 25.4
    img = cv2.imread(str(template6_png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return theoretical_px_per_mm

    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    expected_r = int(26.5 * theoretical_px_per_mm)  # 半径 26.5mm 约 313 px
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=200,
        param1=100,
        param2=45,
        minRadius=int(expected_r * 0.88),
        maxRadius=int(expected_r * 1.12),
    )
    if circles is not None:
        r_px = float(circles[0, 0, 2])
        px_per_mm = (2.0 * r_px) / 53.0
        print(f"[CALIB] 自动识别 Template 6 (ø53mm): 直径={2*r_px:.1f}px -> 比例尺={px_per_mm:.4f} px/mm")
        return px_per_mm

    print(f"[CALIB] 使用 300 DPI 标准比例尺: {theoretical_px_per_mm:.4f} px/mm")
    return theoretical_px_per_mm


def interactive_measure_step09(img_path: Path, spec: Dict) -> None:
    """
    交互式标定器：打开 イラスト 9 (纯 2D 正交弁調整图)，依次点击基准锚点自动更新孔距。
    """
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    prompts = [
        "1/8 点击【第3动轮中心】",
        "2/8 点击【回动曲柄偏心孔中心】(与轮心相距 7.0mm 校准比例尺)",
        "3/8 点击【主曲柄销中心】(0°前死点位置)",
        "4/8 点击【十字头主销中心】(7-13)",
        "5/8 点击【月牙板底部偏心杆铰接孔】",
        "6/8 点击【月牙板中央定轴中心 (N位)】",
        "7/8 点击【合并杆次上孔 (接心向棒前端)】",
        "8/8 点击【合并杆最上孔 (接阀芯杆十字头)】",
    ]
    pts: List[Tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    ax.set_title(prompts[0], fontsize=13)

    def onclick(event):
        if event.xdata is None or event.ydata is None or event.button != 1:
            return
        pts.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "ro", markersize=6)
        ax.text(event.xdata + 8, event.ydata - 8, str(len(pts)), color="red", fontsize=11)
        if len(pts) < len(prompts):
            ax.set_title(prompts[len(pts)], fontsize=13)
            fig.canvas.draw()
        else:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(pts) < 8:
        print("[WARN] 点击点数不足 8 个，保留默认预标定几何参数。")
        return

    def dist_px(i: int, j: int) -> float:
        return math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    # 利用图 9.2 回动曲柄半径 = 7.0mm 反解イラスト 9 子图的精确毫米比例
    mm_per_px = 7.0 / dist_px(0, 1)
    crank_r = dist_px(0, 2) * mm_per_px
    main_rod = dist_px(2, 3) * mm_per_px
    ecc_rod = dist_px(1, 4) * mm_per_px
    exp_tail = dist_px(4, 5) * mm_per_px
    rad_rod = dist_px(5, 6) * mm_per_px
    lap_lead = dist_px(6, 7) * mm_per_px

    spec["cylinder_and_rods"]["crank_radius"] = round(crank_r, 2)
    spec["cylinder_and_rods"]["main_rod_length"] = round(main_rod, 2)
    spec["walschaerts_valve_gear"]["eccentric_rod_length"] = round(ecc_rod, 2)
    spec["walschaerts_valve_gear"]["exp_link_tail_radius"] = round(exp_tail, 2)
    spec["walschaerts_valve_gear"]["radius_rod_length"] = round(rad_rod, 2)
    spec["walschaerts_valve_gear"]["comb_lever_lap_lead_offset"] = round(lap_lead, 2)
    print("[OK] 已根据イラスト 9 实测点更新连杆孔距参数！")


def main():
    parser = argparse.ArgumentParser(description="Aster D51 PDF 尺寸提取与标定工具")
    parser.add_argument("--pdf", type=Path, default=Path("data/raw_pdfs/d51_illustrations.pdf"))
    parser.add_argument("--out-views", type=Path, default=Path("data/ortho_views"))
    parser.add_argument("--out-json", type=Path, default=Path("configs/d51_kinematics.json"))
    parser.add_argument("--export-default", action="store_true", help="直接导出预标定参数与高清正交切片")
    parser.add_argument("--interactive", action="store_true", help="启动交互式鼠标拾取窗口精修孔距")
    args = parser.parse_args()

    spec = default_d51_kinematics()

    if args.pdf.exists():
        exported = extract_and_deskew_pages(args.pdf, args.out_views)
        if "template_3_4_6" in exported:
            px_per_mm = auto_calibrate_dpi_from_template6(exported["template_3_4_6"])
            spec["meta"]["scanner_px_per_mm_300dpi"] = round(px_per_mm, 4)
        if args.interactive and "step09_valve_ortho" in exported:
            interactive_measure_step09(exported["step09_valve_ortho"], spec)
    else:
        print(f"[WARN] 未找到 {args.pdf}，仅生成预标定 d51_kinematics.json。")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] 几何参数文件已写入: {args.out_json}")


if __name__ == "__main__":
    main()