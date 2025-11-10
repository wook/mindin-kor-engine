# runtime/analyze_video.py
import os
import glob
import cv2
import numpy as np

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


# 얼굴 검출기 캐싱 (전역 변수로 한 번만 초기화)
_face_cascade = None


def _get_face_cascade():
    """얼굴 검출기 singleton 인스턴스 반환"""
    global _face_cascade
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
            if _face_cascade.empty():
                _face_cascade = None
        except:
            _face_cascade = None
    return _face_cascade


def detect_face_and_crop_roi(gray_frame):
    """
    얼굴을 검출하고 머리/목 ROI 영역을 반환.
    얼굴이 검출되지 않으면 전체 프레임을 반환.
    
    Args:
        gray_frame: 그레이스케일 이미지 (numpy array)
        
    Returns:
        roi: 머리/목 영역이 crop된 이미지
    """
    h, w = gray_frame.shape
    
    # 얼굴 검출기 가져오기
    face_cascade = _get_face_cascade()
    if face_cascade is None:
        # 얼굴 검출기가 없으면 전체 프레임 반환
        return gray_frame
    
    # 얼굴 검출
    try:
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(30, w // 20), max(30, h // 20))  # 프레임 크기에 비례한 최소 크기
        )
    except:
        return gray_frame
    
    # 얼굴이 검출되지 않으면 전체 프레임 반환
    if len(faces) == 0:
        return gray_frame
    
    # 가장 큰 얼굴 선택
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, face_w, face_h = largest_face
    
    # 머리/목 ROI 영역 계산
    # 얼굴 위쪽 50% (머리) + 얼굴 + 얼굴 아래쪽 30% (목)
    head_ratio = 0.5  # 얼굴 위쪽 머리 영역
    neck_ratio = 0.3  # 얼굴 아래쪽 목 영역
    
    # ROI의 y 좌표와 높이 계산
    roi_y = max(0, int(y - face_h * head_ratio))
    roi_height = int(face_h * (1 + head_ratio + neck_ratio))
    roi_y_end = min(h, roi_y + roi_height)
    
    # ROI의 너비는 얼굴 너비의 1.2배로 설정 (좌우 여유 공간)
    roi_width = int(face_w * 1.2)
    roi_x = max(0, int(x - (roi_width - face_w) / 2))
    roi_x_end = min(w, roi_x + roi_width)
    
    # ROI 추출
    roi = gray_frame[roi_y:roi_y_end, roi_x:roi_x_end]
    
    # ROI가 너무 작거나 비어있으면 전체 프레임 반환
    if roi.shape[0] < 50 or roi.shape[1] < 50 or roi.size == 0:
        return gray_frame
    
    return roi


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

    # 첫 번째 ROI 크기를 저장하여 이후 모든 프레임을 같은 크기로 리사이즈
    target_roi_size = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > frame_limit:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출 후 머리/목 ROI만 crop
        roi = detect_face_and_crop_roi(gray)

        # 첫 번째 프레임에서 ROI 크기를 결정
        if target_roi_size is None:
            target_roi_size = (roi.shape[1], roi.shape[0])  # (width, height)
        else:
            # 이후 프레임은 첫 번째 프레임과 같은 크기로 리사이즈
            if roi.shape[:2] != (target_roi_size[1], target_roi_size[0]):
                roi = cv2.resize(roi, target_roi_size, interpolation=cv2.INTER_AREA)

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
