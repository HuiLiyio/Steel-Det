import os

import cv2
import threading
import queue
import time
from log import Logger
if not os.path.exists('log/info/'):
    os.makedirs('log/info/',exist_ok=True)
log = Logger('./log/info/info.log'.format(), level='info', max_size=2 * 1024 * 1024)
class HikCameraStream:
    def __init__(self, rtsp_url, queue_size=1, name="camera"):
        self.rtsp_url = rtsp_url
        self.name = name
        # 帧缓冲队列，避免丢帧
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.is_running = False
        self.thread = None

        # 初始化视频捕获对象（启用硬件加速）

        self.cap = self._init_capture()

    def _init_capture(self):
        """初始化VideoCapture，启用硬件加速"""
        cap = cv2.VideoCapture(self.rtsp_url)  # 连接config.py #38-39类似位置，相机推流
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头流: {self.rtsp_url}")

        # 设置缓冲区大小（减少延迟）
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)
        # # 启用硬件解码（根据你的系统选择，下面是主流配置）
        # # Windows/Linux 启用 FFmpeg 硬件加速
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        # # 可选：限制帧率（和摄像头配置一致，避免多余计算）
        cap.set(cv2.CAP_PROP_FPS, 5)
        return cap

    def _capture_loop(self):
        """独立线程读取帧"""
        while self.is_running:
            try:
                if self.cap is None:
                    self.cap = self._init_capture()
                    if self.cap is None:
                        raise Exception(f"{self.name} _capture_loop初始化失败")
                ret, frame = self.cap.read()  # opencv内置函数，返回一个tuple(bool,numpy),frame即为图像帧。[1440, 2560, 3]
                if not ret:
                    raise IOError(f"{self.name} _capture_loop读取帧失败")
                # 缓冲队列满了就丢弃旧帧，保证最新帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # 立即读取无需等待，意味着队列中只有最新帧
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
                time.sleep(0.01)  # 10ms读取一次
            except Exception as e:
                try:
                    print(f"{self.name} 读取帧失败，尝试重连...")
                    log.logger.info(f"{self.name} 读取帧失败，尝试重连...错误:"+e)
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        time.sleep(1)
                except:
                    pass

    def start(self):
        """启动拉流线程"""
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True) # 执行self._capture_loop开始拉流
        self.thread.start()

    def read(self):
        """读取最新帧（非阻塞）"""
        try:
            return True, self.frame_queue.get(timeout=1)
        except queue.Empty:
            return False, None

    def stop(self):
        """停止拉流"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        # 清空队列
        while not self.frame_queue.empty():
            self.frame_queue.get()