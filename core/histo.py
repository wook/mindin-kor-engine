# core/histo.py
import numpy as np
from scipy.stats import norm

from .frame_diff import FrameDiffBuffer


def freq_hist_features(buff: FrameDiffBuffer, bins: int = 32):
    """
    freq_map(10프레임 수준)을 히스토그램으로 만들고,
    - hist_peak: 히스토그램 최대 높이
    - hist_std : 분포 표준편차
    - gauss_similarity: 가우시안과의 유사도 (Balance 관련)
    - energy_metric: peak - std (Energy 관련)
    """
    _, freq_map = buff.diff_map(10)
    if freq_map is None:
        return {
            "hist_peak": 0.0,
            "hist_std": 0.0,
            "gauss_similarity": 0.0,
            "energy_metric": 0.0,
        }

    vals = freq_map.flatten()
    hist, edges = np.histogram(vals, bins=bins, range=(0, 1), density=True)
    peak = float(np.max(hist))
    mean = float(np.mean(vals))
    std = float(np.std(vals) + 1e-6)

    x_centers = 0.5 * (edges[:-1] + edges[1:])
    gauss = norm.pdf(x_centers, loc=mean, scale=std)
    gauss /= gauss.sum() + 1e-9

    hist_n = hist / (hist.sum() + 1e-9)
    gauss_sim = float(np.sum(np.minimum(hist_n, gauss)))

    energy_metric = float(peak - std)

    return {
        "hist_peak": peak,
        "hist_std": std,
        "gauss_similarity": gauss_sim,
        "energy_metric": energy_metric,
    }
