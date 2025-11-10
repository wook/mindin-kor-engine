# runtime/report_builder.py
from typing import Dict

RANGE_DEFS_0_100 = {
    "ATTACK": [
        ("낮음", 0.0, 20.0),
        ("보통", 20.0, 50.0),
        ("높음", 50.0, 100.0),
    ],
    "STRESS": [
        ("낮음", 0.0, 20.0),
        ("보통", 20.0, 40.0),
        ("높음", 40.0, 100.0),
    ],
    "UNSTABLE": [
        ("낮음", 0.0, 15.0),
        ("보통", 15.0, 40.0),
        ("높음", 40.0, 100.0),
    ],
    "SUSPICION": [
        ("낮음", 0.0, 20.0),
        ("보통", 20.0, 50.0),
        ("높음", 50.0, 100.0),
    ],
    "BALANCE": [
        ("낮음", 0.0, 40.0),
        ("보통", 40.0, 80.0),
        ("높음", 80.0, 100.0),
    ],
    "CHARM": [
        ("낮음", 0.0, 50.0),
        ("보통", 50.0, 80.0),
        ("높음", 80.0, 100.0),
    ],
    "ENERGY": [
        ("낮음", 0.0, 20.0),
        ("보통", 20.0, 40.0),
        ("높음", 40.0, 100.0),
    ],
    "SELFCONTROL": [
        ("낮음", 0.0, 50.0),
        ("보통", 50.0, 80.0),
        ("높음", 80.0, 100.0),
    ],
    "HYSTERIA": [
        ("낮음", 0.0, 10.0),
        ("보통", 10.0, 50.0),
        ("높음", 50.0, 100.0),
    ],
}


def classify_range_0_100(metric: str, value: float) -> str:
    if metric not in RANGE_DEFS_0_100:
        return "해석범위없음"

    ranges = RANGE_DEFS_0_100[metric]
    for label, vmin, vmax in ranges:
        if value >= vmin and (value < vmax or vmax == 100.0):
            return label

    if value < ranges[0][1]:
        return "매우낮음(범위외)"
    return "매우높음(범위외)"


def classify_brainfatigue(value: float) -> str:
    if value >= 2.5:
        return "매우건강"
    elif value >= -2.0:
        return "건강"
    elif value >= -5.0:
        return "주의"
    else:
        return "위험"


def classify_concentrate(value: float) -> str:
    if value >= 70.0:
        return "높음"
    elif value >= 35.0:
        return "보통"
    else:
        return "낮음"


def classify_lifepower(value: float) -> str:
    if value >= 2.5:
        return "높음"
    elif value >= 1.0:
        return "보통"
    else:
        return "낮음"


def classify_metric(name: str, value: float) -> str:
    if name == "brainfatigue":
        return classify_brainfatigue(value)
    if name == "concentrate":
        return classify_concentrate(value)
    if name == "LIFEPOWER":
        return classify_lifepower(value)
    if name in RANGE_DEFS_0_100:
        return classify_range_0_100(name, value)
    return "해석범위없음"


KOREAN_LABELS = {
    "ATTACK": "공격성",
    "STRESS": "스트레스",
    "UNSTABLE": "불안정성",
    "SUSPICION": "의심/경계",
    "BALANCE": "정서 균형",
    "CHARM": "매력도/친화성",
    "ENERGY": "에너지 수준",
    "SELFCONTROL": "자기조절",
    "HYSTERIA": "히스테리 경향",
    "brainfatigue": "두뇌 피로도",
    "concentrate": "집중도",
    "LIFEPOWER": "생명 에너지",
    "depression": "우울 지수",
    "happines": "행복 지수",
}


def build_text_report(metrics: Dict[str, float]) -> str:
    lines = []
    lines.append("▶ 감정/심리 상태 요약 리포트")
    lines.append("")

    primary_order = [
        "STRESS",
        "UNSTABLE",
        "ATTACK",
        "SUSPICION",
        "brainfatigue",
        "concentrate",
        "LIFEPOWER",
        "depression",
        "happines",
    ]

    for key in primary_order:
        if key not in metrics:
            continue
        val = float(metrics[key])
        cat = classify_metric(key, val)
        label = KOREAN_LABELS.get(key, key)
        lines.append(f"- {label} ({key}) : {cat} ({val:.2f})")

    lines.append("")
    lines.append("▶ 세부 지표")

    for key, val in metrics.items():
        if key in primary_order:
            continue
        if key not in KOREAN_LABELS:
            continue
        cat = classify_metric(key, float(val))
        label = KOREAN_LABELS[key]
        lines.append(f"- {label} ({key}) : {cat} ({float(val):.2f})")

    return "\n".join(lines)
