# core/frame_diff.py
import cv2
import numpy as np
from collections import deque


class FrameDiffBuffer:
    """
    최근 N개의 그레이스케일 프레임을 저장하고,
    2프레임/10프레임/100프레임 수준에서의
    amplitude map(진폭)과 freq map(변화 빈도)을 계산하는 클래스.
    """

    def __init__(self, max_frames_list=None):
        if max_frames_list is None:
            max_frames_list = [2, 10, 100]
        self.max_frames_list = max_frames_list
        self.buffers = {n: deque(maxlen=n) for n in max_frames_list}

    def push(self, gray_frame: np.ndarray):
        """
        gray_frame: 단일 채널(np.uint8) 이미지
        """
        for _, buf in self.buffers.items():
            buf.append(gray_frame.copy())

    def diff_map(self, n: int):
        """
        n프레임 윈도우 안에서의 평균 진폭(amplitude) 맵과
        변화 빈도(freq) 맵을 계산한다.

        반환:
          amp_map: float32, 0~255 정도 범위
          freq_map: float32, 0~1 범위 (해당 픽셀에서 변화가 얼마나 자주 일어나는지)
        """
        buf = self.buffers[n]
        if len(buf) < 2:
            return None, None

        diffs = []
        change_masks = []
        for i in range(1, len(buf)):
            d = cv2.absdiff(buf[i], buf[i - 1]).astype(np.float32)
            diffs.append(d)
            change_masks.append((d > 0).astype(np.float32))

        amp_map = np.mean(diffs, axis=0)
        freq_map = np.mean(change_masks, axis=0)

        return amp_map, freq_map
