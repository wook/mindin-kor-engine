# core/params.py
from typing import Dict


def compute_T_params(A: Dict[str, float],
                     F: Dict[str, float],
                     S: Dict[str, float],
                     H: Dict[str, float]):
    """
    Vibra 계열 T 파라미터 구조를 참고한 국산 버전 근사치.
    - T1 Aggression ~ hist_peak + hist_std
    - T2 Stress ~ 좌우 비대칭
    - T3 Tension ~ F5 (고주파 비율)
    - T4 Negative combo ~ (T1+T2+T3)/3
    - T5 Balance ~ gauss_similarity
    - T6 Charm ~ Charm_sym
    - T7 Energy ~ energy_metric
    - T8 Selfreg ~ (T5+T6)/2
    - T9 Inhibition ~ F6
    - T10 Neuroticism ~ F9
    - T11 Depression-like ~ hist_std * hist_peak
    - T12 Happiness-like ~ 1 / (1 + T11)
    """

    T1 = H["hist_peak"] + H["hist_std"]
    T2 = S["Stress_asym"]
    T3 = F["F5"]
    T4 = (T1 + T2 + T3) / 3.0
    T5 = H["gauss_similarity"]
    T6 = S["Charm_sym"]
    T7 = H["energy_metric"]
    T8 = (T5 + T6) / 2.0
    T9 = F["F6"]
    T10 = F["F9"]
    T11 = H["hist_std"] * H["hist_peak"]
    T12 = 1.0 / (1.0 + T11)

    return {
        "T1_aggression": float(T1),
        "T2_stress": float(T2),
        "T3_tension": float(T3),
        "T4_negative": float(T4),
        "T5_balance": float(T5),
        "T6_charm": float(T6),
        "T7_energy": float(T7),
        "T8_selfreg": float(T8),
        "T9_inhibition": float(T9),
        "T10_neuroticism": float(T10),
        "T11_depression_like": float(T11),
        "T12_happiness_like": float(T12),
    }
