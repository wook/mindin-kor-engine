# runtime/analyze_video.py
import os
import glob
import cv2

from core.frame_diff import FrameDiffBuffer
from core.features import A_features, F_features, S_features
from core.histo import freq_hist_features
from core.params import compute_T_params


def find_video_path(video_dir: str, video_id: str):
    """
    video_dir 안에서 video_id.* 파일을 찾아서 첫 번째 것을 리턴.
    예: test_data_01.avi, test_data_01.mp4 등
    """
    pattern = os.path.join(video_dir, f"{video_id}.*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return candidates[0]


def analyze_video_file(video_path: str,
                       max_duration_sec: float = 60.0):
    """
    비디오 파일(avi/mp4 등)을 열어 프레임을 순차적으로 읽고,
    약 60초 분량까지 FrameDiffBuffer에 넣으면서
    A/F/S/H/T 파라미터를 프레임마다 계산한 뒤,
    1분 간의 평균 T 값을 리턴한다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot_open"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # 메타데이터가 없으면 기본 30fps 가정
    frame_limit = int(fps * max_duration_sec)

    buff = FrameDiffBuffer([2, 10, 100])
    T_series = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > frame_limit:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: 여기에서 얼굴 검출 후 머리/목 ROI만 crop 하면 더 정확해짐
        roi = gray

        buff.push(roi)

        A = A_features(buff)
        F = F_features(buff)
        S = S_features(buff)
        H = freq_hist_features(buff)
        T = compute_T_params(A, F, S, H)
        T_series.append(T)

    cap.release()

    if not T_series:
        return {"error": "no_frames"}

    # 1분 동안의 평균 T 값
    final_report = {}
    keys = T_series[0].keys()
    for k in keys:
        vals = [t[k] for t in T_series]
        final_report[k] = float(sum(vals) / len(vals))

    return final_report
