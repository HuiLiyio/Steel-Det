#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""钢坯检测的命令行入口。"""

import argparse
import sys

import config
from yolo26toDet import run as run_camera


def build_parser():
    p = argparse.ArgumentParser(
        description="取流/回放 + 双钢坯进料、横移和离场判定",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-mode", choices=config.INPUT_MODES, default=config.DEFAULT_INPUT_MODE,
                   help="输入模式：camera 真实相机，virtual 虚拟数据集")
    p.add_argument("--url", default=config.RTSP_URL, help="摄像头 RTSP 地址")
    virtual = p.add_argument_group("虚拟数据集参数")
    virtual.add_argument("--virtual-path", default=argparse.SUPPRESS, metavar="目录或图片列表",
                         help="图片目录或逗号分隔的图片列表，省略时使用 config.py 中的 DEFAULT_REPLAY")
    virtual.add_argument("--virtual-interval", type=float, default=argparse.SUPPRESS, metavar="秒",
                         help="实际读图间隔，默认 %g 秒" % config.REPLAY_READ_INTERVAL)
    virtual.add_argument("--virtual-loop", action="store_true", default=argparse.SUPPRESS,
                         help="循环读取虚拟数据集")
    p.add_argument("--entry-frames", type=int, metavar="N",
                   help="进入状态 1 所需连续命中帧数，至少为 1（实拍 %d，回放 %d）"
                   % (config.CAMERA_ENTRY_FRAMES, config.REPLAY_ENTRY_FRAMES))
    p.add_argument("--stop-frames", type=int, default=config.STOP_FRAMES, metavar="N",
                   help="执行警戒线检查前，状态 1 所需连续停稳帧数")
    p.add_argument("--det-model", default=config.DET_MODEL, help="检测权重（5 类）")
    p.add_argument("--det-imgsz", type=int, default=config.DET_IMGSZ, help="检测推理尺寸")
    p.add_argument("--conf", type=float, default=config.DET_CONF, help="检测置信度阈值")
    p.add_argument("--roi-model", default=config.ROI_MODEL, help="钢坯四类 ROI 检测权重")
    p.add_argument("--roi-imgsz", type=int, default=config.ROI_IMGSZ, help="钢坯 ROI 推理尺寸")
    p.add_argument("--roi-conf", type=float, default=config.ROI_CONF, help="顶点辅助框置信度阈值")
    p.add_argument("--roller-angles", nargs=2, type=float, metavar=("ANGLE2", "ANGLE6"),
                   help="直接指定第 2、6 根轴辊角度；从向下竖直起算，向右倾为正")
    p.add_argument("--calibrate-only", action="store_true", help="按输入模式完成标定并保存后退出")
    p.add_argument("--calibration-image", help="虚拟模式使用的本地标定参考图，默认使用 Calibration.jpg")
    p.add_argument("--calibration-file", help="读取或保存标定结果，默认使用项目内的 Calibration.json")
    p.add_argument("--device", default=config.DEFAULT_DEVICE, help="auto / mps / cpu")
    p.add_argument("--width", type=int, default=config.DISPLAY_WIDTH, help="窗口显示宽度")
    p.add_argument("--hold", type=float, default=config.ENTRY_SNAPSHOT_HOLD, help="进入后等待下一帧期间保留 0→1 证据图的最长秒数")
    p.add_argument("--log-every", type=float, default=config.LOG_INTERVAL,
                   help="状态日志间隔（秒），回放时按模拟采集时间计算")
    p.add_argument("--frames", type=int, default=config.FRAME_LIMIT, help="处理多少帧后退出；0=不限帧数")
    p.add_argument("--no-window", action="store_true", help="不开窗口")
    p.add_argument("--tcp", action="store_true", help="RTSP 强制走 TCP")
    p.add_argument("--no-frame-timeout", type=float, default=config.NO_FRAME_TIMEOUT,
                   help="启动后多少秒仍无帧则退出；0=不设上限")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.entry_frames is not None and args.entry_frames < 1:
        parser.error("--entry-frames 必须至少为 1")
    if args.stop_frames < config.STOP_FRAMES:
        parser.error("--stop-frames 必须至少为 %d" % config.STOP_FRAMES)
    if args.roi_imgsz < 1 or not 0 < args.roi_conf <= 1:
        parser.error("--roi-imgsz 必须为正数，--roi-conf 必须在 (0, 1] 内")
    if args.roller_angles is not None and not all(-90 < angle < 90 for angle in args.roller_angles):
        parser.error("--roller-angles 的两个角度必须在 (-90, 90) 内")
    if args.no_window and (args.calibrate_only or args.calibration_image):
        parser.error("--calibrate-only 和 --calibration-image 需要窗口完成标定")
    if args.input_mode not in config.INPUT_MODES:
        parser.error("DEFAULT_INPUT_MODE 必须为 camera 或 virtual")
    virtual_mode = args.input_mode == "virtual"
    if not virtual_mode and args.calibration_image:
        parser.error("--calibration-image 仅用于 virtual 模式，camera 模式使用相机首帧")
    if not virtual_mode and any(hasattr(args, option) for option in
                                ("virtual_path", "virtual_interval", "virtual_loop")):
        parser.error("--virtual-path、--virtual-interval 和 --virtual-loop 仅用于 virtual 模式")
    if virtual_mode:
        from virtual_dataset import run as run_virtual

        return run_virtual(args)
    return run_camera(args)


if __name__ == "__main__":
    sys.exit(main())
