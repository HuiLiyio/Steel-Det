# -*- coding: utf-8 -*-
"""虚拟数据集输入，包括图片回放、本地标定和模拟计时。"""

import math
import os
import re
import time

import cv2

from config import (
    CALIBRATION_IMAGE,
    DEFAULT_REPLAY,
    REPLAY_CAPTURE_INTERVAL,
    REPLAY_ENTRY_FRAMES,
    REPLAY_READ_INTERVAL,
)
from yolo26toDet import resolve_path, run_detection


def replay_sort_key(name):
    # 原始帧文件名以帧号开头，重启采集后帧号会归零。
    capture_time = re.fullmatch(r"f\d+_(\d{8}_\d{6}_\d{3})", os.path.splitext(name)[0])
    return (capture_time[1] if capture_time else name, name)


def resolve_replay(spec):
    paths = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        path = resolve_path(part, allow_directory=True)
        if path is None:
            raise ValueError("回放路径不存在：%s" % part)
        if os.path.isdir(path):
            files = [f for f in os.listdir(path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                     and os.path.isfile(os.path.join(path, f))]
            if not files:
                raise ValueError("目录里没有图片：%s" % path)
            paths.extend(os.path.join(path, f) for f in sorted(files, key=replay_sort_key))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("请提供回放目录或图片列表")
    return paths


def simulate_dataset(spec=DEFAULT_REPLAY, interval=REPLAY_READ_INTERVAL, loop=False):
    """按实际读图间隔从图片目录或列表取帧，返回可轮询的帧源。"""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("回放间隔必须为正数")
    return ReplaySource(resolve_replay(spec), interval, loop)


class ReplaySource(object):
    """到采样时间才从磁盘读取一张新图。"""

    def __init__(self, paths, interval, loop):
        self.paths = paths
        self.interval = interval
        self.loop = loop
        self.index = 0
        self.is_running = True
        self.finished = False
        self._next_at = time.perf_counter()

    def read(self):
        now = time.perf_counter()
        if not self.is_running or now < self._next_at:
            return False, None
        if self.index == len(self.paths):
            if not self.loop:
                self.finished = True
                self.is_running = False
                return False, None
            self.index = 0
        path = self.paths[self.index]
        frame = cv2.imread(path)
        if frame is None:
            raise OSError("回放图片读取失败：%s" % path)
        self.index += 1
        self._next_at = now + self.interval
        return True, frame


def open_virtual_source(args):
    # 虚拟模式在此打开图片源，读图间隔与模拟采集时间相互独立。
    spec = getattr(args, "virtual_path", DEFAULT_REPLAY)
    interval = getattr(args, "virtual_interval", REPLAY_READ_INTERVAL)
    loop = getattr(args, "virtual_loop", False)
    try:
        source = simulate_dataset(spec, interval, loop)
    except (ValueError, OSError) as e:
        print("[error] 回放源创建失败：%s" % e)
        return None, 2
    print("  输入      : 虚拟数据集 %s" % spec)
    print("  模拟采集  : %d 张图，每 %g 秒读图，模拟帧间隔 %g 秒%s" % (
        len(source.paths), source.interval, REPLAY_CAPTURE_INTERVAL,
        "，循环" if source.loop else ""))
    return source, 0


class VirtualInput:
    """提供本地标定图、图片帧和模拟采集时间。"""

    entry_frames = REPLAY_ENTRY_FRAMES
    reset_on_gap = False
    summary_time_label = "模拟时间跨度"

    def __init__(self, args):
        self.args = args
        self.force_calibration = bool(args.calibration_image)

    def open(self):
        return open_virtual_source(self.args)

    def calibration_reference(self, _open_source):
        requested = self.args.calibration_image or CALIBRATION_IMAGE
        path = resolve_path(requested)
        if path is None:
            raise ValueError("找不到标定图：%s" % requested)
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError("无法读取标定图：%s" % path)
        return (frame, path, path), 0

    def read(self, source):
        ok, frame = source.read()
        if source.finished:
            print("[replay] 播放结束")
        return ok, frame, source.finished

    def frame_time(self, frames, _started):
        # 一张图片对应一个模拟采集帧，独立于实际读图间隔。
        return (frames - 1) * REPLAY_CAPTURE_INTERVAL


def run(args):
    return run_detection(args, VirtualInput(args))
