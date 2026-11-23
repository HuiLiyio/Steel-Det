#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""相机取流与双钢坯共用检测流程。

状态 0 保留现有进料判定，并记录平台和完整的七根轴辊。
状态 1 等钢坯连续停稳后检查警戒线，通过后复用该帧顶点检查料杆覆盖。
状态 2 逐帧运行整帧检测，钢坯下边界 y2 向下越过下平台上边界 y1 时切换到状态 3。
状态 3 检查下方钢坯是否连续三帧未检出，或其左边界是否越过下平台右边界。
状态 4 确认离场并释放钢坯。
前块处于状态 3 时，后块可以确认进入上平台，并在状态 1 等待前块离场。

用 --calibrate-only 在相机首帧上标定两条警戒线和四根料杆。
"""

import json
import math
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from config import (
    CALIBRATION_FILE,
    CAMERA_ENTRY_FRAMES,
    CLS_BILLET,
    CLS_LOWER_PLATFORM,
    CLS_PLATFORM,
    CLS_ROLLER,
    CORNER_NAMES,
    EXIT_MISS_FRAMES,
    HERE,
    MIN_COVERED_RODS,
    ROD_COUNT,
    ROLLER_COUNT,
    STOP_FRAMES,
    STOP_TOLERANCE_PX,
    WARNING_ROLLERS,
)
from visualization import DetectionView, edit_calibration

try:
    from ultralytics import YOLO
except ImportError:
    sys.stderr.write(
        "\n[error] 当前解释器没有 ultralytics / torch：{0}\n"
        "        用装了推理栈的解释器运行，例如本机：\n"
        "        ~/anaconda3/envs/yolo/bin/python main.py\n\n".format(sys.executable)
    )
    raise SystemExit(2)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def area(self):
        """返回检测框面积，宽或高无效时面积为零。"""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def boxes_by_class(result):
    """将单帧 YOLO 结果 result 转成字典，键是类别 ID，值是该类的 Box 列表。"""
    out = {}
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return out
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    for c, cf, xy in zip(cls, conf, xyxy):
        out.setdefault(int(c), []).append(
            Box(float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]), float(cf), int(c)))
    return out


def pick_largest(boxes):
    """从候选框 boxes 中返回面积最大的框，候选为空时返回 None。"""
    return max(boxes, key=lambda b: b.area) if boxes else None


@dataclass(frozen=True)
class RollerLayout:
    platform: Box
    rollers: tuple
    frame_size: tuple


@dataclass(frozen=True)
class WarningLine:
    anchor_x: float
    anchor_y: float
    dx_dy: float
    y_min: float
    y_max: float

    def x_at(self, y):
        """根据传入的纵坐标 y 计算线上的横坐标。"""
        return self.anchor_x + (y - self.anchor_y) * self.dx_dy

    def covers(self, y):
        """判断纵坐标 y 是否落在线段裁剪后的有效范围内，包含两端。"""
        return self.y_min <= y <= self.y_max


def find_roller_layout(obs, platform, frame_size):
    """从单帧检测结果中提取上平台及七根轴辊的布局。

    obs 为按类别分组的检测框，platform 为上平台框，frame_size 为图像的 (高, 宽)。
    筛选中心点位于平台内的有效轴辊框，按横向位置排序。平台无效或根数不符时返回 None。
    """
    if platform is None or platform.area <= 0:
        return None
    rollers = []
    for box in obs.get(CLS_ROLLER, []):
        x, y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        if box.area > 0 and platform.x1 <= x <= platform.x2 and platform.y1 <= y <= platform.y2:
            rollers.append(box)
    rollers.sort(key=lambda box: box.x1 + box.x2)
    if len(rollers) != ROLLER_COUNT:
        return None
    return RollerLayout(platform, tuple(rollers), frame_size)


def top_midpoint(box):
    """返回检测框 box 上边沿的中点坐标 (x, y)。"""
    return (box.x1 + box.x2) / 2, box.y1


def clip_warning_line(roller, platform, angle):
    # The roller top midpoint anchors the line. Project in both directions:
    # billet vertices can lie above the anchor but still inside the platform.
    """以轴辊框 roller 的上边中点为锚点，生成 platform 内的警戒线。

    angle 为相对向下竖直方向的角度，向右倾为正。线段落不到平台内时返回 None。
    """
    x, y = top_midpoint(roller)
    return clip_platform_line(x, y, math.tan(math.radians(angle)), platform)


def clip_platform_line(x, y, slope, platform):
    """将经过原图坐标 (x, y) 的直线裁剪到平台框 platform 内。

    slope 为纵坐标每增加一像素时横坐标的增量 dx/dy。
    返回带有效纵坐标范围的 WarningLine，无有效线段时返回 None。
    """
    y_min, y_max = platform.y1, platform.y2
    if slope == 0:
        if not platform.x1 <= x <= platform.x2:
            return None
    else:
        x_limits = sorted(y + (edge - x) / slope for edge in (platform.x1, platform.x2))
        y_min, y_max = max(y_min, x_limits[0]), min(y_max, x_limits[1])
    if y_min >= y_max:
        return None
    return WarningLine(x, y, slope, y_min, y_max)


def build_warning_lines(layout, angles):
    """根据轴辊布局 layout 和第 2、6 根轴辊的角度 angles 生成左右警戒线。

    返回 (左线, 右线)。标定数据缺失、两线没有共同有效高度或无法形成左到右的通道时返回 None。
    """
    if layout is None or angles is None:
        return None
    left, right = (clip_warning_line(layout.rollers[number - 1], layout.platform, angle)
                   for number, angle in zip(WARNING_ROLLERS, angles))
    if left is None or right is None:
        return None
    start = max(left.y_min, right.y_min)
    end = min(left.y_max, right.y_max)
    if start >= end or any(left.x_at(y) >= right.x_at(y) for y in (start, end)):
        return None
    return left, right


def build_rod_lines(platform, axes):
    """根据标定端点 axes 生成平台框 platform 内的四根料杆轴线。

    每根料杆由两个原图坐标点定义，返回从左到右排列的轴线。
    端点无效、线段缺少共同高度区间或轴线相交时返回 None。
    """
    if len(axes) != ROD_COUNT:
        return None
    lines = []
    for (x1, y1), (x2, y2) in axes:
        if y1 == y2:
            return None
        line = clip_platform_line(x1, y1, (x2 - x1) / (y2 - y1), platform)
        if line is None:
            return None
        lines.append(line)
    start = max(line.y_min for line in lines)
    end = min(line.y_max for line in lines)
    if start >= end:
        return None
    lines.sort(key=lambda line: line.x_at(start))
    if any(left.x_at(start) >= right.x_at(start) or left.x_at(end) >= right.x_at(end)
           for left, right in zip(lines, lines[1:])):
        return None
    return tuple(lines)


@dataclass(frozen=True)
class Calibration:
    layout: RollerLayout
    angles: tuple
    rod_axes: tuple

    def for_frame(self, frame_size):
        """按当前图像尺寸 frame_size（高, 宽）换算平台、轴辊和料杆坐标。

        返回缩放后的轴辊布局和料杆轴线。宽高比不兼容时返回 (None, None)。
        """
        source_h, source_w = self.layout.frame_size
        height, width = frame_size
        scale = width / source_w
        if abs(height - source_h * scale) > 1:
            return None, None

        def scale_box(box):
            return Box(box.x1 * scale, box.y1 * scale, box.x2 * scale, box.y2 * scale,
                       box.conf, box.cls)

        platform = scale_box(self.layout.platform)
        rollers = tuple(scale_box(box) for box in self.layout.rollers)
        axes = tuple(tuple((x * scale, y * scale) for x, y in axis) for axis in self.rod_axes)
        return RollerLayout(platform, rollers, frame_size), build_rod_lines(platform, axes)


@dataclass(frozen=True)
class Clearance:
    left_point: tuple
    right_point: tuple
    left_limit: float
    right_limit: float

    @property
    def left_safe(self):
        """判断左极值点的横坐标是否严格位于左警戒线内侧。"""
        return self.left_point[0] > self.left_limit

    @property
    def right_safe(self):
        """判断右极值点的横坐标是否严格位于右警戒线内侧。"""
        return self.right_point[0] < self.right_limit

    @property
    def safe(self):
        """返回左右两个警戒边界是否均通过。"""
        return self.left_safe and self.right_safe


def measure_clearance(points, lines):
    """用已有钢坯顶点比较左右警戒边界。

    points 为按类别保存的非空原图顶点坐标，lines 为左右警戒线。
    分别在最左和最右顶点的高度比较横坐标，返回 Clearance。任一线在对应高度无效时返回 None。
    """
    left = min(points.values(), key=lambda point: point[0])
    right = max(points.values(), key=lambda point: point[0])
    left_line, right_line = lines
    if not left_line.covers(left[1]) or not right_line.covers(right[1]):
        return None
    return Clearance(left, right, left_line.x_at(left[1]), right_line.x_at(right[1]))


def count_covered_rods(clearance, rods):
    """统计钢坯覆盖的料杆根数。

    clearance 提供钢坯左右极值点，rods 为已标定的料杆轴线。
    料杆在左右极值点的高度均须有有效坐标，并严格落在相应的钢坯边界内。
    """
    lx, ly = clearance.left_point
    rx, ry = clearance.right_point
    return sum(rod.covers(ly) and rod.covers(ry)
               and lx < rod.x_at(ly) and rod.x_at(ry) < rx for rod in rods)


@dataclass
class CornerFrame:
    roi: np.ndarray
    boxes: dict
    points: dict


def detect_corners(frame, billet, model, imgsz, conf, device):
    # Match the training crop's base padding. Random jitter is for training only.
    """裁剪钢坯区域并识别 K0、K1 和 K4 顶点。

    frame 为原始图像，billet 为原图中的钢坯框，model 为 ROI 检测模型。
    imgsz、conf、device 分别指定推理尺寸、置信度阈值和设备。
    返回 ROI 图像、局部检测框和原图顶点坐标。裁剪区域过小则返回 None。
    """
    height, width = frame.shape[:2]
    pad_x = 0.10 * (billet.x2 - billet.x1) + 4
    pad_y = 0.10 * (billet.y2 - billet.y1) + 4
    left = max(0, math.floor(billet.x1 - pad_x))
    top = max(0, math.floor(billet.y1 - pad_y))
    right = min(width, math.ceil(billet.x2 + pad_x))
    bottom = min(height, math.ceil(billet.y2 + pad_y))
    if right - left < 16 or bottom - top < 16:
        return None
    roi = frame[top:bottom, left:right]
    result = model.predict(roi, imgsz=imgsz, conf=conf, classes=list(CORNER_NAMES),
                           device=device, verbose=False)[0]
    observations = boxes_by_class(result)
    boxes, points = {}, {}
    for cls in CORNER_NAMES:
        box = max(observations.get(cls, ()), key=lambda b: b.conf, default=None)
        if box is not None and box.area > 0:
            boxes[cls] = box
            points[cls] = (left + (box.x1 if cls == 1 else box.x2), top + box.y2)
    return CornerFrame(roi, boxes, points)


# ======================================================================
#                               状态机（重要）
# ======================================================================
class EntryStateMachine:
    def __init__(self, entry_frames=CAMERA_ENTRY_FRAMES, angles=None, billet_id=1,
                 calibration=None, stop_frames=STOP_FRAMES):
        """创建一块钢坯的跟踪状态，初始为状态 0。

        billet_id 为钢坯编号，entry_frames 和 stop_frames 分别为进料、停稳所需连续帧数。
        angles 和 calibration 分别提供警戒线角度和可选的完整标定数据。
        """
        self.billet_id = billet_id
        self.state = 0
        self.note = "等待钢坯"
        self.entry_frames = entry_frames
        self.entry_streak = 0
        self.exit_miss_streak = 0
        self.stop_frames = stop_frames
        self.stop_streak = 0
        self.stop_anchor = None
        self.stop_last_center = None
        self.stationary_clearance = None
        self.angles = angles
        self.calibration = calibration
        self.layout = None
        self.warning_lines = None
        self.rod_lines = None
        self.covered_rods = None
        self.last_box = None

    def update_entry(self, obs, frame_size, billet, platform):
        """更新状态 0 的上平台进入计数和轴辊布局。

        obs 为当前帧分组检测框，frame_size 为图像 (高, 宽)，billet 和 platform 为匹配的钢坯框和上平台框。
        钢坯左边界连续越过平台左边界达到阈值后，切换到状态 1 并准备警戒线。
        """
        if self.layout is not None and self.layout.frame_size != frame_size:
            self.layout = None
        current_layout = find_roller_layout(obs, platform, frame_size)
        if current_layout is not None:
            self.layout = current_layout

        if billet is None:
            self.note = "未检测到钢坯(0)"
        elif platform is None:
            self.note = "未检测到平台(2)"
        elif billet.x1 <= platform.x1:
            self.note = "未进入：billet.x1=%.1f <= platform.x1=%.1f" % (
                billet.x1, platform.x1)
        else:
            self.entry_streak += 1
            if self.entry_streak >= self.entry_frames:
                self.state = 1
                if self.calibration is not None:
                    self.layout, self.rod_lines = self.calibration.for_frame(frame_size)
                self.warning_lines = build_warning_lines(self.layout, self.angles)
                self.note = "钢坯已进入平台"
                return
            self.note = "正在确认：连续命中 %d/%d 帧，billet.x1=%.1f > platform.x1=%.1f" % (
                self.entry_streak, self.entry_frames, billet.x1, platform.x1)
            return

        self.entry_streak = 0

    def reset_stop_streak(self):
        """清除停稳计数、位置参照和警戒线结果，并在状态 1 清除料杆覆盖结果。"""
        self.stop_streak = 0
        self.stop_anchor = None
        self.stop_last_center = None
        self.stationary_clearance = None
        if self.state == 1:
            self.covered_rods = None

    def observe_stop(self, billet):
        """用当前钢坯框 billet 的中心位置更新停稳计数。

        中心位移超过容差时从当前帧重新计数。billet 为 None 时清零。
        达到连续停稳帧数后返回 True。
        """
        if billet is None:
            self.reset_stop_streak()
            self.note = "未检测到钢坯，重新等待停稳"
            return False

        center = ((billet.x1 + billet.x2) / 2, (billet.y1 + billet.y2) / 2)
        # 首帧参考检测累积位移，上一帧参考检测突然移动。
        if (self.stop_anchor is None
                or math.dist(center, self.stop_anchor) > STOP_TOLERANCE_PX
                or math.dist(center, self.stop_last_center) > STOP_TOLERANCE_PX):
            self.reset_stop_streak()
            self.stop_anchor = center
            self.stop_streak = 1
        else:
            self.stop_streak = min(self.stop_streak + 1, self.stop_frames)
        self.stop_last_center = center
        if self.stop_streak < self.stop_frames:
            self.note = "等待钢坯停稳：%d/%d 帧" % (self.stop_streak, self.stop_frames)
            return False
        return True

    @property
    def rod_shortage(self):
        """返回当前已有的料杆覆盖结果是否低于最低根数。"""
        return self.covered_rods is not None and self.covered_rods < MIN_COVERED_RODS

    def update_clearance(self, corners, frame_size):
        """执行状态 1 的警戒线与料杆覆盖判定。

        corners 为当前帧的钢坯顶点检测结果，frame_size 为图像 (高, 宽)。
        先检查警戒线，再在通过后统计覆盖根数。两项均通过时切换到状态 2。
        返回警戒线判定结果，缺少判定条件时返回 None。
        """
        self.covered_rods = None
        if self.angles is None:
            self.note = "等待第2、6根轴辊标定"
        elif self.calibration is not None and self.layout is None:
            self.note = "当前画面宽高比与标定图不一致"
        elif self.layout is None:
            self.note = "状态0未保存完整的七根轴辊和平台坐标"
        elif self.layout.frame_size != frame_size or self.warning_lines is None:
            self.note = "警戒线无法使用：画面尺寸改变，或两条线无法形成有效通道"
        elif corners is None:
            self.note = "当前帧未得到钢坯ROI"
        elif not corners.points:
            self.note = "当前帧未检测到钢坯顶点"
        else:
            clearance = measure_clearance(corners.points, self.warning_lines)
            if clearance is None:
                self.note = "钢坯极值点的纵坐标超出警戒线范围"
                return None
            self.note = "%s：左 %.1f vs %.1f；右 %.1f vs %.1f" % (
                "警戒线通过" if clearance.safe else "危险",
                clearance.left_point[0], clearance.left_limit,
                clearance.right_point[0], clearance.right_limit)
            if not clearance.safe:
                return clearance
            if self.rod_lines is None:
                self.note += " | 等待四根料杆标定"
                return clearance

            # 警戒线通过后，使用同一帧的左右极值检查料杆覆盖。
            self.covered_rods = count_covered_rods(clearance, self.rod_lines)
            label = "料杆预警：覆盖不足" if self.rod_shortage else "料杆覆盖"
            self.note += " | %s %d/%d，至少 %d 根" % (
                label, self.covered_rods, ROD_COUNT, MIN_COVERED_RODS)
            if not self.rod_shortage:
                self.state = 2
            return clearance
        return None

    def update_lower_platform(self, billet, platform):
        """根据当帧钢坯框 billet 和下平台框 platform 判断是否进入下平台。

        钢坯下边界 y2 严格大于下平台上边界 y1 时切换到状态 3，缺框时继续等待。
        """
        if billet is None:
            self.note = "未检测到钢坯(0)"
        elif platform is None:
            self.note = "未检测到下平台(3)，billet.y2=%.1f" % billet.y2
        elif billet.y2 > platform.y1:
            self.state = 3
            self.note = "下平台判定通过：billet.y2=%.1f > lower_platform.y1=%.1f" % (
                billet.y2, platform.y1)
        else:
            self.note = "等待下平台判定：billet.y2=%.1f <= lower_platform.y1=%.1f" % (
                billet.y2, platform.y1)

    def update_exit(self, billet, platform):
        """根据下方钢坯框 billet 和下平台框 platform 判断离场。

        钢坯左边界向右越过下平台右边界，或连续漏检达到 EXIT_MISS_FRAMES 时切换到状态 4。
        重新检测到钢坯时清零漏检计数。
        """
        if billet is None:
            self.exit_miss_streak += 1
            if self.exit_miss_streak < EXIT_MISS_FRAMES:
                self.note = "等待离场：连续 %d/%d 帧未检测到下方钢坯(0)" % (
                    self.exit_miss_streak, EXIT_MISS_FRAMES)
                return
            self.note = "钢坯已离场：连续 %d 帧未检测到下方钢坯(0)" % EXIT_MISS_FRAMES
        else:
            self.exit_miss_streak = 0
            if platform is None:
                self.note = "未检测到下平台(3)，billet.x1=%.1f" % billet.x1
                return
            if billet.x1 <= platform.x2:
                self.note = "等待离场：billet.x1=%.1f <= lower_platform.x2=%.1f" % (
                    billet.x1, platform.x2)
                return
            self.note = "钢坯已离场：billet.x1=%.1f > lower_platform.x2=%.1f" % (
                billet.x1, platform.x2)
        self.state = 4
        self.entry_streak = 0
        self.exit_miss_streak = 0
        self.reset_stop_streak()

    def step(self, obs, frame_size, billet, platform, infer_corners, blocked=False):
        """推进一块钢坯在本帧的状态，返回包含前一状态和判定结果的 BilletFrame。

        obs 为分组检测框，frame_size 为图像 (高, 宽)，billet、platform 为当前工位的匹配框。
        infer_corners 接收钢坯框并返回顶点检测结果。blocked 为 True 时，状态 1 等待前一块离场。
        """
        previous_state = self.state
        corners, clearance = None, None
        if billet is not None:
            self.last_box = billet
        if previous_state == 0:
            self.update_entry(obs, frame_size, billet, platform)
        elif previous_state == 2:
            self.update_lower_platform(billet, platform)
        elif previous_state == 3:
            self.update_exit(billet, platform)

        if self.state == 1:
            if blocked:
                self.note = "已进入上平台，等待前一块钢坯离场"
            elif self.observe_stop(billet):
                if self.stationary_clearance is None:
                    # 未通过或无法判定时，只要仍停稳，下一帧继续推理。
                    corners = infer_corners(billet)
                    clearance = self.update_clearance(corners, frame_size)
                    if clearance is not None and clearance.safe:
                        self.stationary_clearance = clearance
                else:
                    # 警戒线已通过，沿用本次停稳位置的顶点判定和料杆结果。
                    clearance = self.stationary_clearance
        return BilletFrame(self, previous_state, billet, platform, corners, clearance, blocked)


@dataclass
class BilletFrame:
    track: EntryStateMachine
    previous_state: int
    billet: Box | None
    platform: Box | None
    corners: CornerFrame | None
    clearance: Clearance | None
    blocked: bool


def match_billet(candidates, previous):
    """从当前工位候选框 candidates 中匹配一块钢坯。

    有上一帧匹配框 previous 时选中心距离最近的框，没有历史时选面积最大的框。
    候选为空时返回 None。
    """
    if previous is None or not candidates:
        return pick_largest(candidates)
    prev_x, prev_y = previous.x1 + previous.x2, previous.y1 + previous.y2
    return min(candidates, key=lambda box: (
        (box.x1 + box.x2 - prev_x) ** 2 + (box.y1 + box.y2 - prev_y) ** 2))


class BilletQueue:
    def __init__(self, entry_frames, angles, calibration=None, stop_frames=STOP_FRAMES):
        """初始化双钢坯跟踪队列，并建立第一条跟踪记录。

        entry_frames、stop_frames 分别配置进入和停稳的确认帧数，angles、calibration 提供标定数据。
        """
        self.entry_frames = entry_frames
        self.stop_frames = stop_frames
        self.angles = angles
        self.calibration = calibration
        self.tracks = []
        self.departed = 0
        self.next_id = 1
        self.frame_size = None
        self.upper_bottom = None
        self.lower_top = None
        self.add_track()

    def add_track(self):
        """为下一块钢坯创建带独立编号的状态机，加入队尾并返回该状态机。"""
        track = EntryStateMachine(self.entry_frames, self.angles, self.next_id, self.calibration,
                                  stop_frames=self.stop_frames)
        self.next_id += 1
        self.tracks.append(track)
        return track

    def step(self, obs, frame_size, infer_corners):
        """用同一帧检测结果匹配上下工位并更新双钢坯队列。

        obs 为按类别分组的检测框，frame_size 为图像 (高, 宽)，infer_corners 为按需运行顶点检测的函数。
        队首进入下平台后可跟踪下一块钢坯。确认离场时移除队首，并返回本帧的跟踪更新列表。
        """
        upper = pick_largest(obs.get(CLS_PLATFORM, []))
        lower = pick_largest(obs.get(CLS_LOWER_PLATFORM, []))
        billets = obs.get(CLS_BILLET, [])
        if self.frame_size != frame_size:
            self.frame_size = frame_size
            self.upper_bottom = self.lower_top = None
            for track in self.tracks:
                track.last_box = None
                track.reset_stop_streak()
                track.exit_miss_streak = 0
        if upper is not None:
            self.upper_bottom = upper.y2
        if lower is not None:
            self.lower_top = lower.y1
        lower_candidates = [box for box in billets
                            if self.lower_top is not None and box.y2 > self.lower_top]
        upper_candidates = [
            box for box in billets
            if (self.lower_top is None or box.y2 <= self.lower_top)
            and (self.upper_bottom is None or (box.y1 + box.y2) / 2 <= self.upper_bottom)
        ]

        first = self.tracks[0]
        if first.state == 3:
            # Without a lower boundary, a visible billet cannot establish absence.
            candidates = lower_candidates if self.lower_top is not None else billets
            platform = lower
        elif first.state == 2:
            # A billet that just left can still be visible to the right.
            candidates = [box for box in billets if lower is None
                          or box.y2 <= lower.y1 or box.x1 <= lower.x2]
            platform = lower
        else:
            candidates, platform = upper_candidates, upper
        billet = match_billet(candidates, first.last_box)
        updates = [first.step(obs, frame_size, billet, platform, infer_corners)]

        if first.state >= 3:
            waiting = self.tracks[1] if len(self.tracks) == 2 else self.add_track()
            candidates = [box for box in upper_candidates if box is not billet]
            next_billet = match_billet(candidates, waiting.last_box)
            # On the exit frame the waiting billet may still only confirm entry.
            updates.append(waiting.step(obs, frame_size, next_billet, upper, infer_corners,
                                        blocked=True))
        if first.state == 4:
            self.tracks.pop(0)
            self.departed += 1
        return updates

    def reset_frame_streaks(self):
        """取帧中断时按各钢坯的当前状态清除进料、停稳或漏检的连续计数。"""
        for track in self.tracks:
            if track.state == 0:
                track.entry_streak = 0
            elif track.state == 1:
                track.reset_stop_streak()
            elif track.state == 3:
                track.exit_miss_streak = 0


# ======================================================================
# 文件路径
# ======================================================================
def resolve_path(path, allow_directory=False):
    """依次从脚本目录和运行目录查找 path，返回存在的绝对路径。

    allow_directory 为 True 时也接受目录。找不到时返回 None。
    """
    for base in (HERE, os.getcwd()):
        candidate = os.path.abspath(os.path.join(base, path))
        if os.path.isfile(candidate) or (allow_directory and os.path.isdir(candidate)):
            return candidate
    return None


# ======================================================================
# 真实相机：检查 RTSP 地址后启动实时取流
# ======================================================================
def mask_url(url):
    """遮盖 RTSP 地址 url 中的密码，返回适合日志显示的地址。"""
    try:
        head, tail = url.split("://", 1)
        cred, host = tail.split("@", 1)
        if ":" in cred:
            return "{0}://{1}:***@{2}".format(head, cred.split(":", 1)[0], host)
    except Exception:
        pass
    return url


def check_endpoint(url, timeout=3.0):
    """检查 url 中主机和端口的 TCP 可达性，连接超时设为 timeout 秒。

    未指定端口时使用 554。返回 (能否连接, 诊断信息)。
    """
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 554
    if not host:
        return False, "URL 里解析不出主机名"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "%s:%s 可连接" % (host, port)
    except OSError as e:
        return False, "%s:%s 连不上：%s" % (host, port, e)


def start_camera(url):
    """用 RTSP 地址 url 创建并启动相机取流线程，失败时记录错误并返回 None。"""
    try:
        from opencv_ffmpeg import HikCameraStream

        cam = HikCameraStream(url, name="steel")
        cam.start()
        print("[camera] 已启动，is_running=%s" % cam.is_running)
        return cam
    except Exception as e:
        print("[error] 相机连接失败：%s" % e)
        return None


def open_camera_source(args):
    # 真实相机模式在此配置 RTSP 传输并连接相机。
    """根据命令行参数 args 中的 url 和 tcp 设置打开相机。

    先检查端口可达性，再尝试启动取流。返回 (帧源, 退出码)，成功时退出码为 0。
    """
    if args.tcp:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        print("[info] RTSP 走 TCP")
    ok, msg = check_endpoint(args.url)
    print("[check] %s" % msg)
    if not ok:
        print("[error] 相机不可达，先确认网络/地址。")
        return None, 2
    source = start_camera(args.url)
    if source is None:
        return None, 3
    print("  相机      : %s" % mask_url(args.url))
    return source, 0


def safe_stop(source):
    """停止帧源 source，短暂等待取流线程结束，并释放已有的视频捕获句柄。

    清理失败时记录警告，供退出流程继续执行。
    """
    try:
        source.is_running = False
        thread = getattr(source, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        cap = getattr(source, "cap", None)
        if cap is not None:
            cap.release()
            source.cap = None
    except Exception as e:
        print("[warn] 停止取流时异常：%s" % e)


def warmup_model(model, imgsz, device):
    """把一张空白图交给 model，按 imgsz 和 device 执行一次预热推理并记录耗时。"""
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    model.predict(dummy, imgsz=imgsz, conf=0.25, device=device, verbose=False)
    print("[warmup] det %.2f s" % (time.perf_counter() - t0))


# ======================================================================
# 标定
# ======================================================================
def calibrate_geometry(frame, layout, initial_angles=None):
    """以 frame 和轴辊布局 layout 打开交互标定，设置警戒线和料杆轴线。

    initial_angles 为可选的初始警戒线角度。用户确认后返回 Calibration，取消时返回 None。
    """
    geometry = edit_calibration(
        frame, layout.platform, tuple(map(top_midpoint, layout.rollers)), initial_angles,
        project_line=lambda x, y, slope: clip_platform_line(x, y, slope, layout.platform),
        validate_warnings=lambda angles: build_warning_lines(layout, angles) is not None,
        validate_rods=lambda axes: build_rod_lines(layout.platform, axes) is not None,
    )
    if geometry is None:
        return None
    angles, rod_axes = geometry
    return Calibration(layout, angles, rod_axes)


def read_camera_calibration_frame(source, timeout):
    """从已启动的帧源 source 读取首个有效图像作为标定参考。

    timeout 为等待秒数，非正数表示不限时。超时或取流停止时抛出 ValueError。
    """
    started = time.perf_counter()
    while source.is_running:
        ok, frame = source.read()
        if ok and frame is not None and frame.size:
            return frame
        if timeout > 0 and time.perf_counter() - started >= timeout:
            raise ValueError("等待相机标定首帧超时（%g 秒）" % timeout)
        time.sleep(0.01)
    raise ValueError("相机在取得标定首帧前停止取流")


def detect_calibration_layout(frame, args, det, device, reference_name):
    """用整帧模型 det 在标定画面 frame 中定位平台与七根轴辊。

    args 提供推理尺寸和置信度，device 指定设备，reference_name 用于错误诊断。
    返回 RollerLayout，缺少完整布局时抛出 ValueError。
    """
    result = det.predict(frame, imgsz=args.det_imgsz, conf=args.conf,
                         device=device, verbose=False)[0]
    obs = boxes_by_class(result)
    layout = find_roller_layout(obs, pick_largest(obs.get(CLS_PLATFORM, [])), frame.shape[:2])
    if layout is None:
        raise ValueError("标定图未同时检出上平台内的七根轴辊：%s" % reference_name)
    return layout


class CameraInput:
    """提供 RTSP 首帧标定、实时计时和断帧处理规则。"""

    entry_frames = CAMERA_ENTRY_FRAMES
    force_calibration = False
    reset_on_gap = True
    summary_time_label = None

    def __init__(self, args):
        """保存相机运行参数 args，供取流和首帧标定使用。"""
        self.args = args

    def open(self):
        """用已保存的相机参数打开 RTSP 取流，返回 (帧源, 退出码)。"""
        return open_camera_source(self.args)

    def calibration_reference(self, open_source):
        """通过 open_source 打开或复用取流，读取相机首帧。

        返回 (参考数据, 退出码)，参考数据包含图像、空文件路径和诊断名称。取流失败时传回原退出码。
        """
        source, status = open_source()
        if source is None:
            return None, status
        frame = read_camera_calibration_frame(source, self.args.no_frame_timeout)
        print("[calibration] 使用相机首帧标定")
        return (frame, None, "相机首帧"), 0

    def read(self, source):
        """从实时帧源 source 读取最新画面。

        返回 (是否取到帧, 画面, 是否结束)，相机暂时取不到帧时保持实时取流流程。
        """
        ok, frame = source.read()
        return ok, frame, False

    def frame_time(self, _frames, started):
        """以 started 为起点计算实际经过的秒数，_frames 为共用接口传入的帧序号。"""
        return time.perf_counter() - started


def save_calibration(path, image_path, calibration):
    """将标定数据 calibration 写入 path 指定的 JSON 文件。

    image_path 为可选的标定图路径，保存为相对路径。写完临时文件后原子替换目标文件。
    """
    directory = os.path.dirname(path)
    data = {
        "image": os.path.relpath(image_path, directory) if image_path is not None else None,
        "frame_size": calibration.layout.frame_size,
        "platform": asdict(calibration.layout.platform),
        "rollers": [asdict(box) for box in calibration.layout.rollers],
        "roller_angles": calibration.angles,
        "rod_axes": calibration.rod_axes,
    }
    with tempfile.NamedTemporaryFile("w", dir=directory, encoding="utf-8",
                                     prefix=".calibration-", delete=False) as output:
        temp_path = output.name
        try:
            json.dump(data, output, indent=2, allow_nan=False)
            output.write("\n")
            output.close()
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def load_calibration(path):
    """读取 path 指定的 JSON，校验平台、轴辊、警戒线与料杆几何后返回 Calibration。"""
    with open(path, encoding="utf-8") as source:
        data = json.load(source)
    try:
        frame_size = tuple(data["frame_size"])
        platform = Box(**data["platform"])
        rollers = [Box(**entry) for entry in data["rollers"]]
        angles = tuple(float(angle) for angle in data["roller_angles"])
        axes = tuple(tuple(tuple(float(v) for v in point) for point in axis)
                     for axis in data["rod_axes"])
        if len(axes) != ROD_COUNT or any(len(axis) != 2 or any(len(point) != 2 for point in axis)
                                         for axis in axes):
            raise ValueError("需要四根料杆，每根两个原图坐标点")
        if len(angles) != 2 or not all(-90 < angle < 90 for angle in angles):
            raise ValueError("需要两个 (-90, 90) 内的轴辊角度")
        height, width = frame_size
        if (platform.cls != CLS_PLATFORM or any(box.cls != CLS_ROLLER for box in rollers)
                or not all(0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height
                           for box in (platform, *rollers))):
            raise ValueError("平台或轴辊框超出标定图")
        if not all(0 <= x < width and 0 <= y < height for axis in axes for x, y in axis):
            raise ValueError("料杆标定点超出标定图")
        layout = find_roller_layout({CLS_ROLLER: rollers}, platform, frame_size)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("标定文件无效：%s" % exc) from exc
    if layout is None or build_warning_lines(layout, angles) is None or build_rod_lines(platform, axes) is None:
        raise ValueError("标定图尺寸、警戒线或料杆轴线无效：%s" % path)
    return Calibration(layout, angles, axes)


# ======================================================================
# 日志
# ======================================================================
def log_entry(update, frame_number):
    """用单帧跟踪更新 update 和帧号 frame_number 记录进料切换及两框的左边界。"""
    sm, billet, platform = update.track, update.billet, update.platform
    print("\n==== 状态切换 0->1 | 钢坯 #%d | 第 %d 帧 | 连续命中 %d 帧 ====" % (
        sm.billet_id, frame_number, sm.entry_frames))
    print("  钢坯 0   : xyxy=(%.1f,%.1f)-(%.1f,%.1f) conf=%.2f" % (
        billet.x1, billet.y1, billet.x2, billet.y2, billet.conf))
    print("  平台 2   : xyxy=(%.1f,%.1f)-(%.1f,%.1f) 左边界 x1=%.1f" % (
        platform.x1, platform.y1, platform.x2, platform.y2, platform.x1))
    print("  进入判据 : billet.x1=%.1f > platform.x1=%.1f" % (
        billet.x1, platform.x1))


def log_state(sm):
    """输出单块钢坯状态机 sm 的钢坯编号、当前状态和判定说明。"""
    print("[状态%d] 钢坯 #%d | %s" % (sm.state, sm.billet_id, sm.note))


# ======================================================================
# 主流程
# ======================================================================
def pick_device(name):
    """使用显式指定的设备 name。为 auto 或未指定时优先选 MPS，否则使用 CPU。"""
    if name and name != "auto":
        return name
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run(args):
    """使用命令行参数 args 启动相机检测，返回进程退出码。"""
    return run_detection(args, CameraInput(args))


def run_detection(args, frame_input):
    """加载整帧检测模型，再按帧源规则运行标定和检测。

    args 提供权重路径、设备、确认帧数等运行参数。
    frame_input 定义如何打开帧源、取得标定参考、读取画面和计时。返回流程退出码。
    """
    entry_frames = args.entry_frames or frame_input.entry_frames

    det_path = resolve_path(args.det_model)
    if not det_path:
        print("[error] 找不到检测权重：%s" % args.det_model)
        return 3

    device = pick_device(args.device)
    try:
        det = YOLO(det_path)
    except Exception as e:
        print("[error] 模型加载失败：%s" % e)
        return 3

    with ExitStack() as source_cleanup:
        return run_with_calibration(args, frame_input, entry_frames, det_path, det, device,
                                    source_cleanup)


def run_with_calibration(args, frame_input, entry_frames, det_path, det, device, source_cleanup):
    """加载或执行标定，再逐帧推理、更新状态机并输出画面与证据帧。

    args 为运行参数，frame_input 提供输入规则，entry_frames 为进料确认帧数。
    det_path 为整帧权重路径，det 为已加载模型，device 为推理设备。
    source_cleanup 登记帧源的退出清理。返回流程退出码。
    """
    source = None

    def open_source():
        nonlocal source
        # 首帧标定若已启动取流，检测阶段继续使用同一帧源。
        if source is not None:
            return source, 0
        source, status = frame_input.open()
        if source is not None:
            source_cleanup.callback(safe_stop, source)
        return source, status

    calibration_path = os.path.abspath(os.path.expanduser(args.calibration_file or CALIBRATION_FILE))
    calibration = None
    angles = args.roller_angles
    try:
        interactive = args.calibrate_only or frame_input.force_calibration or (
            not args.no_window and not os.path.exists(calibration_path))
        if interactive:
            reference_data, status = frame_input.calibration_reference(open_source)
            if reference_data is None:
                return status
            reference, image_path, reference_name = reference_data
            layout = detect_calibration_layout(reference, args, det, device, reference_name)
            calibration = calibrate_geometry(reference, layout, angles)
            if calibration is None:
                print("[info] 标定取消")
                return 0
            save_calibration(calibration_path, image_path, calibration)
            print("[calibration] 已保存：%s" % calibration_path)
            if args.calibrate_only:
                return 0
        elif args.calibration_file or os.path.exists(calibration_path):
            calibration = load_calibration(calibration_path)
            print("[calibration] 已加载：%s" % calibration_path)
        if angles is None and calibration is not None:
            angles = calibration.angles
    except (OSError, ValueError, cv2.error) as exc:
        print("[error] 标定失败：%s" % exc)
        return 4
    except KeyboardInterrupt:
        return 0

    roi_path = resolve_path(args.roi_model)
    if not roi_path:
        print("[error] 找不到 ROI 检测权重：%s" % args.roi_model)
        return 3

    print("=" * 72)
    print("  yolo26toDet | 状态 0 → 1 → 2 → 3 → 4")
    print("  输入模式  : %s" % args.input_mode)
    source, status = open_source()
    if source is None:
        return status

    print("  检测权重  : %s" % det_path)
    print("  ROI 权重  : %s" % roi_path)
    print("  设备/尺寸 : %s / imgsz=%d / conf=%.2f" % (
        device, args.det_imgsz, args.conf))
    print("  进入确认  : 连续命中 %d 帧" % entry_frames)
    print("  停稳确认  : 连续 %d 帧，中心位移容差 %g px" % (
        args.stop_frames, STOP_TOLERANCE_PX))
    print("=" * 72)

    try:
        print("[model] det: task=%s names=%s" % (det.task, det.names))
        warmup_model(det, args.det_imgsz, device)
        corner_model = YOLO(roi_path)
        if corner_model.task != "detect" or set(corner_model.names) != {0, 1, 2, 3}:
            raise ValueError("ROI 权重必须为 0–3 四类目标检测模型")
        print("[model] ROI: task=%s names=%s" % (corner_model.task, corner_model.names))
    except Exception as e:
        print("[error] 模型加载/热身失败：%s" % e)
        return 3

    if angles is not None:
        print("[calibration] 轴辊 2 / 6：%.2f°, %.2f°" % tuple(angles))
    if calibration is None:
        print("[calibration] 等待四根料杆标定，可先运行 --calibrate-only")

    view = DetectionView(args.width, args.hold, args.no_window)
    view.open()

    billet_queue = BilletQueue(entry_frames, angles, calibration, stop_frames=args.stop_frames)
    frames = 0
    started = time.perf_counter()
    frame_time = last_log = 0.0

    try:
        while True:
            try:
                ok, frame, finished = frame_input.read(source)
            except OSError as e:
                print("[error] %s" % e)
                return 3

            if ok and frame is not None:
                frames += 1
                frame_time = frame_input.frame_time(frames, started)
                result = det.predict(frame, imgsz=args.det_imgsz, conf=args.conf,
                                     device=device, verbose=False)[0]
                updates = billet_queue.step(
                    boxes_by_class(result), frame.shape[:2],
                    lambda billet: detect_corners(frame, billet, corner_model, args.roi_imgsz,
                                                  args.roi_conf, device))
                first = updates[0]
                sm = first.track
                for update in updates:
                    if update.previous_state == 0 and update.track.state >= 1:
                        view.record_entry(frame, update, frames, keep_on_screen=len(updates) == 1)
                        log_entry(update, frames)

                # 0->1 is handled above, including when 1->2 follows on the same frame.
                transitioned = sm.state > max(1, first.previous_state)
                view.update(frame, updates, transitioned)
                if transitioned:
                    transition = "%d->%d" % (max(1, first.previous_state), sm.state)
                    print("\n==== 状态切换 %s | 钢坯 #%d | 第 %d 帧 ====" % (
                        transition, sm.billet_id, frames))
                    print("  %s" % sm.note)
                    if sm.state == 2:
                        print("  原图顶点：%s" % {
                            CORNER_NAMES[cls]: tuple(round(v, 1) for v in point)
                            for cls, point in first.corners.points.items()})
                    view.save_transition(transition, sm.billet_id, frames)

                if frame_time - last_log >= args.log_every:
                    for update in updates:
                        log_state(update.track)
                    last_log = frame_time
            else:
                if finished:
                    break
                if frame_input.reset_on_gap:
                    billet_queue.reset_frame_streaks()
                if args.no_frame_timeout > 0 and frames == 0 and \
                        time.perf_counter() - started >= args.no_frame_timeout:
                    print("[error] %.0f 秒内没拿到画面，退出。" % args.no_frame_timeout)
                    return 4
                if args.no_window:
                    time.sleep(0.01)
                    continue

            if view.poll(ok):
                print("[info] 收到退出键")
                break

            if args.frames and frames >= args.frames:
                print("[info] 已达到 --frames %d，退出" % args.frames)
                break

    except KeyboardInterrupt:
        print("\n[info] Ctrl+C 退出")
    finally:
        view.close()

    print("\n===== 本次运行 =====")
    print("  帧数   : %d" % frames)
    print("  时长   : %.1f s" % (time.perf_counter() - started))
    if frame_input.summary_time_label:
        print("  %s: %.1f s" % (frame_input.summary_time_label, frame_time))
    print("  已离场钢坯: %d" % billet_queue.departed)
    for track in billet_queue.tracks:
        print("  钢坯 #%d 最终状态: %d | %s" % (track.billet_id, track.state, track.note))
    return 0
