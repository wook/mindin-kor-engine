# offline/build_dataset_offline.py
import csv
import os
from typing import Dict, Iterable, List

from runtime.analyze_video import analyze_video_file, find_video_path

DATA_DIR = "data"
VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
LABEL_PATH = os.path.join(DATA_DIR, "labels.csv")
OUT_DATASET_PATH = os.path.join(DATA_DIR, "dataset_generated.csv")
def _load_labels(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"라벨 CSV 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [dict(row) for row in reader]


def _write_dataset(rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(OUT_DATASET_PATH), exist_ok=True)

    with open(OUT_DATASET_PATH, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    labels = _load_labels(LABEL_PATH)

    # 네 CSV에 들어있는 라벨 컬럼들
    label_cols = [
        "brainfatigue",
        "concentrate",
        "LIFEPOWER",
        "ATTACK",
        "STRESS",
        "UNSTABLE",
        "SUSPICION",
        "BALANCE",
        "CHARM",
        "ENERGY",
        "SELFCONTROL",
        "HYSTERIA",
        "depression",
        "happines",
    ]

    rows: List[Dict[str, object]] = []
    fieldnames: List[str] = []

    for row in labels:
        video_id = row.get("video_id")
        if not video_id:
            print("[WARN] video_id 가 없는 행을 건너뜁니다.")
            continue

        video_path = find_video_path(VIDEO_DIR, video_id)

        if video_path is None:
            print(f"[WARN] video file not found for {video_id}")
            continue

        print(f"[INFO] analyzing {video_id} -> {video_path}")
        T_report = analyze_video_file(video_path, max_duration_sec=60.0)

        if "error" in T_report:
            print(f"[WARN] video {video_id} failed: {T_report['error']}")
            continue

        combined: Dict[str, object] = {"video_id": video_id}
        for col in label_cols:
            combined[col] = row.get(col)

        combined.update(T_report)
        rows.append(combined)

        # fieldnames 업데이트
        for key in combined.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["video_id", *label_cols]

    _write_dataset(rows, fieldnames)

    print(f"[OK] saved dataset to {OUT_DATASET_PATH}")
    print("[INFO] 첫 5개 샘플:")
    for sample in rows[:5]:
        print(sample)


if __name__ == "__main__":
    main()
