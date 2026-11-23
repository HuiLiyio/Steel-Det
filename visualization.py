# -*- coding: utf-8 -*-
"""标定交互、检测画面和证据帧输出。"""

import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    CORNER_COLORS,
    CORNER_NAMES,
    EXIT_MISS_FRAMES,
    FONT_CANDIDATES,
    MIN_COVERED_RODS,
    ROD_COLOR,
    ROD_COUNT,
    SNAPSHOT_DIR,
    WARNING_COLORS,
    WARNING_ROLLERS,
    WINDOW_NAME,
)


# ======================================================================
# 状态标签（中文，右下角）
# ======================================================================
_FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
_LABEL_CACHE = {}
_FONT_CACHE = {}


def get_font(size):
    if size not in _FONT_CACHE:
        if _FONT_PATH:
            _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _status_texts(state, danger=False, rod_warning=False):
    if rod_warning:
        return ("危险", "状态1 · 料杆覆盖不足") if _FONT_PATH else ("DANGER", "STATE 1 | LOW ROD COVERAGE")
    if danger:
        return ("危险", "状态1 · 超出警戒范围") if _FONT_PATH else ("DANGER", "STATE 1")
    if not _FONT_PATH:
        return {
            0: ("NOT ENTERED", "STATE 0"),
            1: ("CHECKING", "STATE 1"),
            2: ("CHECKING TRANSFER", "STATE 2"),
            3: ("ON LOWER PLATFORM", "STATE 3"),
            4: ("DEPARTED", "STATE 4"),
        }[state]
    return {
        0: ("未进入", "状态0 · 未进入"),
        1: ("检查中", "状态1 · 已进入平台"),
        2: ("横移检测", "状态2 · 通过警戒线"),
        3: ("等待离场", "状态3 · 已进入下平台"),
        4: ("已离场", "状态4 · 离开下平台"),
    }[state]


def build_status_label(state, danger=False, rod_warning=False):
    key = (state, danger, rod_warning)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    main, sub = _status_texts(state, danger, rod_warning)
    bg = (188, 32, 44) if danger or rod_warning else {
        0: (30, 110, 190), 1: (180, 98, 18), 2: (0, 125, 140),
        3: (0, 135, 40), 4: (80, 105, 55),
    }[state]
    fg = (255, 255, 255)

    main_font = get_font(46)
    sub_font = get_font(19)
    pad = 14
    main_box = main_font.getbbox(main)
    sub_box = sub_font.getbbox(sub)
    main_w, main_h = main_box[2] - main_box[0], main_box[3] - main_box[1]
    sub_w, sub_h = sub_box[2] - sub_box[0], sub_box[3] - sub_box[1]
    w = max(max(main_w, sub_w) + pad * 2, 190)
    h = main_h + sub_h + pad * 3

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=12, fill=bg + (215,))
    draw.text((pad - main_box[0], pad - main_box[1]), main, font=main_font, fill=fg + (255,))
    draw.text((pad - sub_box[0], pad * 2 + main_h - sub_box[1]), sub,
              font=sub_font, fill=(255, 255, 255, 235))

    arr = np.array(img)
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    _LABEL_CACHE[key] = (bgr, alpha)
    return _LABEL_CACHE[key]


def overlay_status(frame, state, danger=False, rod_warning=False):
    bgr, alpha = build_status_label(state, danger, rod_warning)
    h, w = bgr.shape[:2]
    h, w = min(h, frame.shape[0]), min(w, frame.shape[1])
    bgr, alpha = bgr[:h, :w], alpha[:h, :w]
    x = max(0, frame.shape[1] - w - 18)
    y = max(0, frame.shape[0] - h - 18)
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    a = alpha[:, :, None]
    frame[y:y + h, x:x + w] = (roi * (1 - a) + bgr.astype(np.float32) * a).astype(np.uint8)
    return frame


# ======================================================================
# 检测框和状态 0→1 证据帧
# ======================================================================
def put_text(img, text, org, color=(255, 255, 255), scale=0.62, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_box(img, box, color, label=None, thickness=2):
    p1 = (int(round(box.x1)), int(round(box.y1)))
    p2 = (int(round(box.x2)), int(round(box.y2)))
    cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        fs = max(0.6, min(1.6, img.shape[1] / 1800.0))
        put_text(img, label, (p1[0], max(int(22 * fs), p1[1] - int(8 * fs))),
                 color, fs, 1)


def annotate_entry(frame, billet, platform):
    img = frame.copy()
    h, w = img.shape[:2]
    s = max(0.8, min(2.6, w / 1000.0))

    draw_box(img, platform, (0, 200, 0), "platform 2  %.2f" % platform.conf,
             max(2, int(round(2 * s))))
    draw_box(img, billet, (255, 120, 0), "billet 0  %.2f" % billet.conf,
             max(2, int(round(2 * s))))

    lines = [
        "STATE 0 -> 1   billet.x1=%.1f > platform.x1=%.1f" % (
            billet.x1, platform.x1),
        "billet=(%.0f,%.0f)-(%.0f,%.0f)   platform=(%.0f,%.0f)-(%.0f,%.0f)" % (
            billet.x1, billet.y1, billet.x2, billet.y2,
            platform.x1, platform.y1, platform.x2, platform.y2),
    ]
    line_h = int(round(26 * s))
    bar_h = line_h * len(lines) + int(round(16 * s))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(lines):
        put_text(img, text,
                 (int(round(18 * s)), h - bar_h + int(round(22 * s)) + i * line_h),
                 (255, 255, 255), 0.60 * s, 1)
    return img


def fit_width(frame, width):
    if not width or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * float(width) / w))),
                      interpolation=cv2.INTER_AREA)


# ======================================================================
# 标定窗口
# ======================================================================
def draw_warning_lines(img, lines, platform, shift=(0, 0), colors=WARNING_COLORS, thickness=3):
    x0, y0 = shift
    left, top = max(0, math.ceil(platform.x1) - x0), max(0, math.ceil(platform.y1) - y0)
    right = min(img.shape[1], math.floor(platform.x2) - x0 + 1)
    bottom = min(img.shape[0], math.floor(platform.y2) - y0 + 1)
    if left >= right or top >= bottom:
        return
    area = img[top:bottom, left:right]
    for line, color in zip(lines, colors):
        if line is not None:
            start, end = [(round(line.x_at(y)) - x0 - left, round(y) - y0 - top)
                          for y in (line.y_min, line.y_max)]
            cv2.line(area, start, end, color, thickness, cv2.LINE_8)


def edit_calibration(frame, platform, roller_anchors, initial_angles, *,
                     project_line, validate_warnings, validate_rods):
    """返回用户确认的警戒线角度和料杆端点，取消时返回 None。"""
    h, w = frame.shape[:2]
    left = max(0, math.floor(platform.x1) - 24)
    top = max(0, math.floor(platform.y1) - 24)
    right = min(w, math.ceil(platform.x2) + 24)
    bottom = min(h, math.ceil(platform.y2) + 24)
    base = frame[top:bottom, left:right]
    header_height = 75
    widget = "Calibrate warning lines and support rods"
    angles = list(initial_angles) if initial_angles is not None else [None, None]
    rod_axes = [[] for _ in range(ROD_COUNT)]
    marking_rods = False
    active = 0
    message = ""
    mouse_point = None

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal active, message, mouse_point
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        if not (0 <= x < view_w and header_height <= y < header_height + view_h):
            mouse_point = None
            return
        mouse_point = (left + x * base.shape[1] / view_w,
                       top + (y - header_height) * base.shape[0] / view_h)
        if event == cv2.EVENT_MOUSEMOVE:
            return
        point = mouse_point
        if marking_rods:
            axis = rod_axes[active]
            if len(axis) == 2:
                axis.clear()
            if axis and abs(point[1] - axis[0][1]) < 1:
                message = "Choose a second point at a different height."
                return
            axis.append(point)
            message = ""
            if len(axis) == 2:
                mouse_point = None
                if active + 1 < ROD_COUNT and not rod_axes[active + 1]:
                    active += 1
            return
        ax, ay = roller_anchors[WARNING_ROLLERS[active] - 1]
        dx, dy = point[0] - ax, point[1] - ay
        if abs(dy) < 1:
            message = "Choose another point above or below the anchor."
            return
        angles[active] = math.degrees(math.atan(dx / dy))
        message = ""
        mouse_point = None
        if active == 0 and angles[1] is None:
            active = 1

    def preview_line():
        if mouse_point is None:
            return None
        if marking_rods:
            axis = rod_axes[active]
            if len(axis) != 1:
                return None
            ax, ay = axis[0]
        else:
            ax, ay = roller_anchors[WARNING_ROLLERS[active] - 1]
        dx, dy = mouse_point[0] - ax, mouse_point[1] - ay
        if abs(dy) < 1:
            return None
        return project_line(ax, ay, dx / dy)

    view_h, view_w = fit_width(base, 1100).shape[:2]
    cv2.namedWindow(widget, cv2.WINDOW_AUTOSIZE)
    try:
        # HighGUI needs the window to have a displayed image before registration.
        cv2.imshow(widget, np.zeros((view_h + header_height, view_w, 3), dtype=np.uint8))
        cv2.setMouseCallback(widget, on_mouse)
        while True:
            img = base.copy()
            cv2.rectangle(img, (round(platform.x1 - left), round(platform.y1 - top)),
                          (round(platform.x2 - left), round(platform.y2 - top)),
                          (0, 200, 0), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(roller_anchors, 1):
                pt = round(x-left), round(y-top)
                color = WARNING_COLORS[WARNING_ROLLERS.index(i)] if i in WARNING_ROLLERS else (220, 220, 220)
                cv2.circle(img, pt, 5, color, -1)
                put_text(img, str(i), (pt[0]+7, pt[1]+18), color, .68, 1)
            lines = tuple(project_line(*roller_anchors[number - 1], math.tan(math.radians(angle)))
                          if angle is not None else None
                          for number, angle in zip(WARNING_ROLLERS, angles))
            draw_warning_lines(img, lines, platform, (left, top))
            heading = np.full((header_height, view_w, 3), (29, 37, 49), dtype=np.uint8)
            if marking_rods:
                for i, axis in enumerate(rod_axes, 1):
                    if len(axis) == 2:
                        (x1, y1), (x2, y2) = axis
                        line = project_line(x1, y1, (x2 - x1) / (y2 - y1))
                        draw_warning_lines(img, (line,), platform, (left, top), (ROD_COLOR,))
                    for x, y in axis:
                        cv2.circle(img, (round(x-left), round(y-top)), 5, ROD_COLOR, -1)
                    if axis:
                        x, y = axis[0]
                        put_text(img, "Rod %d" % i, (round(x-left)+8, round(y-top)+14),
                                 ROD_COLOR, .58, 1)
                instruction = ("Rod %d: click first point, move to preview, click second point.  "
                               "1-4 select | Enter save | B back | R reset | Q quit" % (active + 1))
                values = ["Rod %d: %d/2" % (i, len(axis))
                          for i, axis in enumerate(rod_axes, 1)]
            else:
                instruction = ("Roller %d: move to preview, click to set.  2/6 select  |  Enter next  |  "
                               "R reset  |  Q quit" % WARNING_ROLLERS[active])
                values = ["%d: %s" % (number, "%.2f deg" % angle if angle is not None else "unset")
                          for number, angle in zip(WARNING_ROLLERS, angles)]
            preview = preview_line()
            if preview is not None:
                preview_color = (250, 250, 250)
                draw_warning_lines(img, (preview,), platform, (left, top), (preview_color,), thickness=1)
                mx, my = mouse_point
                cv2.drawMarker(img, (round(mx-left), round(my-top)), preview_color,
                               cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
            cv2.putText(heading, instruction, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(heading, "    ".join(values) + "    " + message, (10, 51),
                        cv2.FONT_HERSHEY_SIMPLEX, .54, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.imshow(widget, np.vstack((heading, fit_width(img, view_w))))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                return None
            try:
                if cv2.getWindowProperty(widget, cv2.WND_PROP_VISIBLE) == 0:
                    return None
            except cv2.error:
                return None
            if marking_rods:
                if ord("1") <= key <= ord("4"):
                    active = key - ord("1")
                elif key == ord("b"):
                    marking_rods, active, message = False, 0, ""
                elif key == ord("r"):
                    for axis in rod_axes:
                        axis.clear()
                    active, message = 0, ""
                elif key in (10, 13):
                    if all(len(axis) == 2 for axis in rod_axes) and validate_rods(rod_axes):
                        return tuple(angles), tuple(tuple(axis) for axis in rod_axes)
                    message = "Mark four distinct, non-crossing rod axes inside the platform."
            elif key in (ord("2"), ord("6")):
                active = WARNING_ROLLERS.index(int(chr(key)))
            elif key == ord("r"):
                angles[:], active, message = [None, None], 0, ""
            elif key in (10, 13):
                if None not in angles and validate_warnings(angles):
                    marking_rods, active, message = True, 0, ""
                else:
                    message = "Set both directions and keep the lines ordered inside the platform."
    finally:
        try:
            cv2.destroyWindow(widget)
        except cv2.error:
            pass


# ======================================================================
# 运行画面和证据帧
# ======================================================================
def corner_preview(corners, width, confirmed=False, warning_lines=None):
    columns = (width // 2, width // 6, width // 6)
    columns += (width - sum(columns),)
    strip = np.full((178, width, 3), (29, 37, 49), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def panel(image, x, w, title, color, image_top=30):
        cv2.putText(strip, title, (x + 7, 23), font, .51, color, 1, cv2.LINE_AA)
        if image is not None and image.size and w > 12:
            h, image_w = image.shape[:2]
            image_height = min(118, strip.shape[0] - image_top - 6)
            ratio = min((w - 12) / image_w, image_height / h)
            tw, th = max(1, round(ratio * image_w)), max(1, round(ratio * h))
            resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
            xx, yy = x + (w - tw) // 2, image_top + (image_height - th) // 2
            strip[yy:yy+th, xx:xx+tw] = resized

    overview = corners.roi.copy() if corners is not None else None
    if corners is not None:
        for cls, box in corners.boxes.items():
            draw_box(overview, box, CORNER_COLORS[cls], CORNER_NAMES[cls], thickness=1)
            point = (round(box.x1 if cls == 1 else box.x2), round(box.y2))
            cv2.circle(overview, point, 3, CORNER_COLORS[cls], -1)
    title = "Confirmation ROI" if confirmed else "Current ROI"
    panel(overview, 0, columns[0], title if overview is not None else "Waiting for billet ROI", (230, 230, 230))
    x = columns[0]
    for cls, w in zip(CORNER_NAMES, columns[1:]):
        name = CORNER_NAMES[cls]
        box = corners.boxes.get(cls) if corners is not None else None
        if corners is not None and box is None:
            strip[:, x:x+w] = (29, 49, 75)
            cv2.putText(strip, name + " NOT DETECTED", (x + 7, 23),
                        font, min(.51, w / 300), (80, 204, 255), 1, cv2.LINE_AA)
            x += w
            continue
        crop = None
        title = name + " --"
        if box is not None:
            height, width_roi = corners.roi.shape[:2]
            l, t = max(0, math.floor(box.x1)-4), max(0, math.floor(box.y1)-4)
            r, b = min(width_roi, math.ceil(box.x2)+4), min(height, math.ceil(box.y2)+4)
            crop = corners.roi[t:b, l:r].copy()
            point = (round(box.x1 if cls == 1 else box.x2)-l, round(box.y2)-t)
            cv2.circle(crop, point, 3, CORNER_COLORS[cls], -1)
            title = name
            ox, oy = corners.points[cls]
            scale = min(.51, w / 330)
            cv2.putText(strip, "x=%.1f" % ox, (x + 36, 23),
                        font, scale, (110, 242, 137), 1, cv2.LINE_AA)
            side = 0 if cls == 1 else 1
            line = warning_lines[side] if warning_lines else None
            limit = "%.1f" % line.x_at(oy) if line is not None and line.covers(oy) else "--"
            cv2.putText(strip, "%s %s" % ("L" if side == 0 else "R", limit), (x + 7, 43),
                        font, scale, (107, 119, 255), 1, cv2.LINE_AA)
        panel(crop, x, w, title, CORNER_COLORS[cls], image_top=52)
        cv2.line(strip, (x, 0), (x, strip.shape[0]-1), (66, 78, 95), 1)
        x += w
    return strip


def render_frame(frame, updates, width, confirmed_strip=None):
    first = updates[0]
    sm = first.track
    corners, clearance = first.corners, first.clearance
    strip = confirmed_strip
    warning_lines = None
    if sm.layout is not None and sm.layout.frame_size == frame.shape[:2]:
        warning_lines = sm.warning_lines
    img = frame.copy()
    if first.previous_state >= 2 and first.platform is not None:
        draw_box(img, first.platform, (0, 200, 0), "lower platform 3")
    for update in updates:
        if update.billet is not None:
            color = (255, 120, 0) if update is first else (255, 220, 70)
            label = "Billet #%d" % update.track.billet_id
            draw_box(img, update.billet, color, label)
    if corners is not None:
        for cls, (x, y) in corners.points.items():
            color = CORNER_COLORS[cls]
            point = (round(x), round(y))
            cv2.circle(img, point, 5, color, -1, cv2.LINE_AA)
            put_text(img, CORNER_NAMES[cls], (point[0]+7, point[1]+19), color, .6, 1)
    has_clearance_result = sm.state == 1 and clearance is not None
    danger = has_clearance_result and not clearance.safe
    rod_warning = has_clearance_result and clearance.safe and sm.rod_shortage
    img = overlay_status(fit_width(img, width), sm.state, danger=danger, rod_warning=rod_warning)
    text = None
    if clearance is not None:
        text = "L %s %.1f / %.1f     R %s %.1f / %.1f" % (
            "SAFE" if clearance.left_safe else "DANGER", clearance.left_point[0], clearance.left_limit,
            "SAFE" if clearance.right_safe else "DANGER", clearance.right_point[0], clearance.right_limit)
        if clearance.safe:
            if sm.covered_rods is None:
                text += "     RODS NOT CALIBRATED"
            else:
                label = "ROD COVERAGE LOW" if rod_warning else "RODS"
                text += "     %s %d/%d (NEED %d)" % (
                    label, sm.covered_rods, ROD_COUNT, MIN_COVERED_RODS)
    elif sm.state == 1:
        if sm.stop_streak < sm.stop_frames:
            text = "WAITING FOR STOP %d/%d" % (sm.stop_streak, sm.stop_frames)
        else:
            text = "WAITING FOR VALID CLEARANCE"
    elif first.previous_state >= 2:
        billet, platform = first.billet, first.platform
        if first.previous_state == 2:
            text = "billet.y2=%s    lower_platform.y1=%s" % (
                "%.1f" % billet.y2 if billet is not None else "--",
                "%.1f" % platform.y1 if platform is not None else "--")
        elif billet is None:
            if sm.state == 4:
                text = "EXIT CONFIRMED: LOWER BILLET MISSING FOR %d FRAMES" % EXIT_MISS_FRAMES
            else:
                text = "LOWER BILLET NOT DETECTED %d/%d" % (
                    sm.exit_miss_streak, EXIT_MISS_FRAMES)
        else:
            text = "billet.x1=%s    lower_platform.x2=%s" % (
                "%.1f" % billet.x1, "%.1f" % platform.x2 if platform is not None else "--")
    status = "    |    ".join(
        "Billet #%d  State %d%s" % (update.track.billet_id, update.track.state,
                                    " (waiting)" if update.blocked and update.track.state == 1 else "")
        for update in updates)
    lines = [text, status] if text is not None else [status]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, .56
    panel_width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 20
    top = max(0, img.shape[0] - 24 * len(lines) - 8)
    cv2.rectangle(img, (0, top), (panel_width, img.shape[0] - 1), (29, 37, 49), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, top + 19 + i * 24), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if strip is None or strip.shape[1] != img.shape[1]:
        strip = corner_preview(corners, img.shape[1], confirmed=sm.state == 2,
                               warning_lines=warning_lines)
    return np.vstack((img, strip)), strip


def save_snapshot(img, kind, billet_id, frame_number):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    name = "state_%s_billet%d_%s_f%06d.jpg" % (
        kind.replace("->", "to"), billet_id, time.strftime("%Y%m%d_%H%M%S"), frame_number)
    path = os.path.join(SNAPSHOT_DIR, name)
    if cv2.imwrite(path, img):
        print("[snapshot] %s" % path)
    return path


class DetectionView:
    """管理单次运行的窗口、证据帧停留时间和顶点预览。"""

    def __init__(self, width, hold, no_window):
        self.width = width
        self.hold = hold
        self.no_window = no_window
        self.frame_view = None
        self.snapshot_view = None
        self.snapshot_expires = 0.0
        self.shown_view = None
        self.preview_track = None
        self.preview_clearance = None
        self.confirmed_strip = None

    def open(self):
        if not self.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            print("[info] 窗口已打开：q / ESC 退出")

    def record_entry(self, frame, update, frame_number, keep_on_screen):
        entry = annotate_entry(frame, update.billet, update.platform)
        view = overlay_status(fit_width(entry, self.width), 1)
        save_snapshot(view, "0->1", update.track.billet_id, frame_number)
        if keep_on_screen:
            self.snapshot_view = view
            self.snapshot_expires = time.perf_counter() + self.hold

    def update(self, frame, updates, transitioned):
        first = updates[0]
        if first.corners is not None or len(updates) == 2:
            self.snapshot_expires = 0
        if self.no_window and not transitioned:
            return

        track = first.track
        # 状态 1 重新停稳后，上一轮检查的预览失效。
        if track is not self.preview_track or (
                track.state == 1 and track.stationary_clearance is not self.preview_clearance):
            self.preview_track = track
            self.preview_clearance = None
            self.confirmed_strip = None

        self.frame_view, strip = render_frame(frame, updates, self.width, self.confirmed_strip)
        if first.corners is not None and first.clearance is not None and first.clearance.safe:
            self.preview_clearance = first.clearance
            self.confirmed_strip = strip

    def save_transition(self, transition, billet_id, frame_number):
        save_snapshot(self.frame_view, transition, billet_id, frame_number)

    def poll(self, frame_received):
        if self.no_window:
            return False
        view = self.frame_view
        if time.perf_counter() < self.snapshot_expires:
            view = self.snapshot_view
        if view is not None and view is not self.shown_view:
            cv2.imshow(WINDOW_NAME, view)
            self.shown_view = view
        return (cv2.waitKey(1 if frame_received else 30) & 0xFF) in (27, ord("q"))

    def close(self):
        if not self.no_window:
            cv2.destroyAllWindows()
