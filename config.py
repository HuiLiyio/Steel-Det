# -*- coding: utf-8 -*-
"""钢坯检测的路径、输入模式和默认参数。"""

import os


HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_IP = "192.168.9.74"
RTSP_URL = "rtsp://admin:lgzb2593748@{ip}:554/Streaming/Channels/101".format(ip=CAMERA_IP)
DET_MODEL = os.path.join("models", "1028-640-s.pt")
ROI_MODEL = os.path.join("models", "4cls_roi_det_2.pt")
CALIBRATION_IMAGE = os.path.join(HERE, "Calibration.jpg")
CALIBRATION_FILE = os.path.join(HERE, "Calibration.json")
# 默认输入模式。camera 接真实 RTSP 相机，virtual 读取本地图片。
# 命令行 --input-mode 可覆盖此值。
INPUT_MODES = ("camera", "virtual")
DEFAULT_INPUT_MODE = "camera"
DEFAULT_REPLAY = os.path.join(HERE, "datasets", "images1021_raw")
REPLAY_READ_INTERVAL = 0.5
REPLAY_CAPTURE_INTERVAL = 3.0
CAMERA_ENTRY_FRAMES = 3
REPLAY_ENTRY_FRAMES = 1
EXIT_MISS_FRAMES = 3
STOP_FRAMES = 3
STOP_TOLERANCE_PX = 5.0
WINDOW_NAME = "yolo26toDet | steel billet"
SNAPSHOT_DIR = os.path.join("debug_out", "state")

CLS_BILLET = 0
CLS_ROLLER = 1
CLS_PLATFORM = 2
CLS_LOWER_PLATFORM = 3  # det-1 标签
ROLLER_COUNT = 7
WARNING_ROLLERS = (2, 6)
WARNING_COLORS = ((30, 180, 255), (70, 100, 255))
ROD_COUNT = 4
MIN_COVERED_RODS = 3
ROD_COLOR = (255, 215, 70)
CORNER_NAMES = {1: "K0", 2: "K1", 3: "K4"}
CORNER_COLORS = {1: (255, 188, 76), 2: (120, 235, 80), 3: (53, 193, 255)}

FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)

# 推理和运行参数，命令行可覆盖。
DET_IMGSZ = 640
DET_CONF = 0.25
ROI_IMGSZ = 1280
ROI_CONF = 0.25
DEFAULT_DEVICE = "auto"
DISPLAY_WIDTH = 1100
ENTRY_SNAPSHOT_HOLD = 0
LOG_INTERVAL = 5.0
FRAME_LIMIT = 0
NO_FRAME_TIMEOUT = 10.0
