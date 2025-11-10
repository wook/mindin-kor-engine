# core/features.py
import numpy as np

from .frame_diff import FrameDiffBuffer


def _split_left_right(roi: np.ndarray):
    """
    머리/얼굴 ROI를 좌/우 반으로 나눠 평균값 비교용으로 사용.
    """
    h, w = roi.shape
    left = roi[:, : w // 2]
    right = roi[:, w // 2 :]
    return left, right


def A_features(buff: FrameDiffBuffer):
    """
    A 계열 특징 (진폭 중심).
    간단히:
    - A1: 2프레임 윈도우 평균 진폭
    - A2: 10프레임 윈도우 평균 진폭
    - A3: 100프레임 윈도우 평균 진폭
    - A4: A1과 유사(placeholder)
    """
    out = {}
    for n, name in [(2, "A1"), (10, "A2"), (100, "A3")]:
        amp_map, _ = buff.diff_map(n)
        if amp_map is None:
            out[name] = 0.0
            continue
        out[name] = float(np.mean(amp_map))

    out["A4"] = out["A1"]
    return out


def F_features(buff: FrameDiffBuffer):
    """
    F 계열 특징 (변화 빈도/고주파 비율 등).
    - F1: 2프레임 윈도우 freq 평균
    - F2: 10프레임 윈도우 freq 평균
    - F3: 100프레임 윈도우 freq 평균
    - F5: '고주파 비율' proxy (freq2 / freq10)
    - F6/F9: 추후 calibration에서 다듬을 placeholder
    """
    out = {}

    _, freq2 = buff.diff_map(2)
    _, freq10 = buff.diff_map(10)
    _, freq100 = buff.diff_map(100)

    if freq2 is not None:
        out["F1"] = float(np.mean(freq2))
    else:
        out["F1"] = 0.0

    if freq10 is not None:
        out["F2"] = float(np.mean(freq10))
    else:
        out["F2"] = 0.0

    if freq100 is not None:
        out["F3"] = float(np.mean(freq100))
    else:
        out["F3"] = 0.0

    # 고주파 비율 proxy
    if freq2 is not None and freq10 is not None and np.mean(freq10) > 1e-6:
        out["F5"] = float(np.mean(freq2) / (np.mean(freq10) + 1e-6))
    else:
        out["F5"] = 0.0

    # 반응 지연 / 신경증 경향 (일단 0으로 두고, calibration에서 설계)
    out["F6"] = 0.0
    out["F9"] = 0.0

    return out


def S_features(buff: FrameDiffBuffer):
    """
    좌우 비대칭(S 계열):
    - Stress_asym: 좌우 진폭/변화 비대칭 (클수록 스트레스↑)
    - Charm_sym  : 좌우가 비슷할수록 높게 (매력/친화성↑)
    """
    amp_map, freq_map = buff.diff_map(10)
    if amp_map is None or freq_map is None:
        return {"Stress_asym": 0.0, "Charm_sym": 0.0}

    L_amp, R_amp = _split_left_right(amp_map)
    L_freq, R_freq = _split_left_right(freq_map)

    amp_diff = abs(float(np.mean(L_amp) - np.mean(R_amp)))
    freq_diff = abs(float(np.mean(L_freq) - np.mean(R_freq)))

    stress_asym = amp_diff + freq_diff
    charm_sym = 1.0 / (1.0 + stress_asym)

    return {
        "Stress_asym": float(stress_asym),
        "Charm_sym": float(charm_sym),
    }
