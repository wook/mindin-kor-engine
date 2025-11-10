## 프로젝트 개요

이 저장소는 비디오의 얼굴 영역을 분석하여 다양한 심리·정서 지표(T1~T12 등)를 추정하고, 분석 결과를 보고서 형태로 정리하거나 오프라인 학습용 데이터셋을 생성·학습하는 도구를 제공합니다. 핵심 로직은 `core` 패키지의 프레임 차분 기반 특징 추출기이며, `runtime` 모듈은 실시간/배치 분석 파이프라인을, `offline` 모듈은 데이터셋 구축과 모델 학습 파이프라인을 제공합니다.

---

## 핵심 데이터 구조

### `core.frame_diff.FrameDiffBuffer`

- **설명**: 최근 N개의 그레이스케일 프레임을 저장하여 프레임 간 차이를 기반으로 진폭(amplitude) 맵과 변화 빈도(freq) 맵을 계산합니다.
- **초기화**
  - `max_frames_list: Iterable[int] = [2, 10, 100]`
  - 각 윈도 크기별로 내부 `deque` 버퍼를 생성합니다.
- **메서드**
  - `push(gray_frame: np.ndarray) -> None`
    - 단일 채널(`np.uint8`) 프레임을 모든 버퍼에 복사 저장합니다.
  - `diff_map(n: int) -> Tuple[np.ndarray | None, np.ndarray | None]`
    - 길이가 `n`인 버퍼에서 인접 프레임의 절대 차를 평균내어 `amp_map`(float32)과 변화가 있었던 비율 `freq_map`(float32, 0~1)을 반환합니다.
    - 프레임 수가 2 미만이면 `(None, None)`을 반환합니다.
- **사용 예시**

```python
import cv2
import numpy as np
from core.frame_diff import FrameDiffBuffer

buff = FrameDiffBuffer([2, 10])
gray = cv2.imread("face.png", cv2.IMREAD_GRAYSCALE)
buff.push(gray)
buff.push(gray)  # 최소 두 프레임 필요
amp_map, freq_map = buff.diff_map(2)
```

---

## 특징 추출 API (`core.features`)

### `A_features(buff: FrameDiffBuffer) -> Dict[str, float>`
- 2·10·100 프레임 윈도의 평균 진폭을 기반으로 A1~A4 특징을 계산합니다.
- 내부적으로 `FrameDiffBuffer.diff_map`을 호출하며, 프레임이 충분하지 않은 경우 0.0을 반환합니다.
- **예시**

```python
from core.features import A_features
features = A_features(buff)
# {'A1': 12.3, 'A2': 8.1, 'A3': 5.4, 'A4': 12.3}
```

### `F_features(buff: FrameDiffBuffer) -> Dict[str, float>`
- 2·10·100 프레임 윈도의 변화 빈도 평균을 사용해 F1~F3을 계산하고, 고주파 비율을 `F5`로 제공합니다.
- `F6`, `F9`는 후처리용 placeholder로 0.0을 반환합니다.

### `S_features(buff: FrameDiffBuffer) -> Dict[str, float>`
- 좌우 ROI 분할 후 진폭/빈도 맵의 비대칭 정도를 정량화하여 `Stress_asym`과 `Charm_sym`을 산출합니다.
- 데이터 부족 시 두 값 모두 0.0을 반환합니다.

---

## 히스토그램 기반 특징 (`core.histo`)

### `freq_hist_features(buff: FrameDiffBuffer, bins: int = 32) -> Dict[str, float>`
- 10프레임 윈도의 `freq_map` 히스토그램을 분석하여 다음 지표를 제공합니다.
  - `hist_peak`: 히스토그램 최대 높이
  - `hist_std`: 변화 빈도의 표준편차
  - `gauss_similarity`: 동일 평균·분산 가우시안과의 유사도
  - `energy_metric`: `hist_peak - hist_std`
- **예시**

```python
from core.histo import freq_hist_features
H = freq_hist_features(buff, bins=64)
```

---

## 통합 파라미터 계산 (`core.params`)

### `compute_T_params(A, F, S, H) -> Dict[str, float]`
- A/F/S/H 특징 딕셔너리를 입력받아 12개의 T 지표(`T1_aggression` 등)를 계산합니다.
- 내부적으로 선형 결합과 간단한 비선형 조합(예: 역수, 평균)을 수행합니다.
- **예시**

```python
from core.params import compute_T_params
T = compute_T_params(A, F, S, H)
# {'T1_aggression': ..., 'T2_stress': ..., ...}
```

---

## 런타임 분석 컴포넌트 (`runtime.analyze_video`)

### `find_video_path(video_dir: str, video_id: str) -> str | None`
- `video_dir` 아래에서 `video_id.*` 패턴에 맞는 첫 파일 경로를 반환합니다.
- 존재하지 않으면 `None`.

### `detect_face_and_crop_roi(gray_frame: np.ndarray) -> np.ndarray`
- Haar Cascade를 사용해 얼굴을 탐지하고, 머리~목 영역을 포함한 ROI를 반환합니다.
- 얼굴을 찾지 못하면 입력 프레임 전체를 반환합니다.

### `analyze_video_file(video_path: str, max_duration_sec: float = 60.0) -> Dict[str, float] | Dict[str, str]`
- 비디오를 순차적으로 읽어 ROI를 정규화하고, 매 프레임마다 A/F/S/H/T 특징을 계산합니다.
- 최대 `max_duration_sec`만큼 처리하며, 최종적으로 T 지표의 평균을 반환합니다.
- 실패 시 `{"error": <사유>}`를 돌려줍니다.
- **예시**

```python
from runtime.analyze_video import analyze_video_file
report = analyze_video_file("samples/test.mp4", max_duration_sec=30)
if "error" in report:
    print("분석 실패:", report["error"])
else:
    print("평균 T 지표:", report)
```

---

## 보고서 빌더 (`runtime.report_builder`)

### 범주 정의 상수
- `RANGE_DEFS_0_100`: 0~100 범위 지표에 대한 구간-레이블 매핑.
- `KOREAN_LABELS`: 각 지표의 한국어 라벨.

### 분류 유틸리티
- `classify_range_0_100(metric: str, value: float) -> str`
- `classify_brainfatigue(value: float) -> str`
- `classify_concentrate(value: float) -> str`
- `classify_lifepower(value: float) -> str`
- `classify_metric(name: str, value: float) -> str`
  - 이름에 따라 위 함수를 라우팅하며, 범위를 벗어나면 “해석범위없음”을 반환합니다.

### `build_text_report(metrics: Dict[str, float]) -> str`
- T 지표나 기타 지표를 받아 요약/세부 보고서 텍스트를 생성합니다.
- **예시**

```python
from runtime.report_builder import build_text_report
text = build_text_report({"STRESS": 35.2, "ATTACK": 18.0, "BALANCE": 72.0})
print(text)
```

---

## 오프라인 데이터셋 생성 (`offline.build_dataset_offline`)

### 상수
- `DATA_DIR`, `VIDEO_DIR`, `LABEL_PATH`, `OUT_DATASET_PATH`: 데이터 위치 설정.

### 내부 유틸리티
- `_load_labels(path)` / `_write_dataset(rows, fieldnames)` 는 파일 입출력 헬퍼입니다.

### `main()`
- 라벨 CSV를 읽어 `video_id`별 영상 파일을 찾고, `analyze_video_file`로 T 지표를 산출합니다.
- 라벨과 분석 결과를 결합해 `dataset_generated.csv`로 저장합니다.
- **CLI 사용**

```bash
python -m offline.build_dataset_offline
```

---

## 모델 학습 및 평가 (`offline.train_eval_from_dataset`)

### 상수
- `DATASET_PATH`, `TARGETS`, `RANDOM_STATE`, `N_FOLDS`: 실험 설정.

### `extract_linear_formula(model, feature_names) -> str`
- 학습된 Lasso 모델을 사람이 읽을 수 있는 선형 식으로 변환합니다.

### `main()`
- `dataset_generated.csv`를 읽고 특징/라벨을 분리합니다.
- 각 타깃에 대해 K-Fold 교차 검증을 수행하고 평균 MAE/R²를 계산합니다.
- 전체 데이터를 다시 학습해 `modeling/checkpoints/{target}_lasso.joblib`로 저장하고, 결과 로그를 타임스탬프 파일로 기록합니다.
- **CLI 사용**

```bash
python -m offline.train_eval_from_dataset
```

---

## 통합 파이프라인 예시

```python
from runtime.analyze_video import analyze_video_file
from runtime.report_builder import build_text_report

video_report = analyze_video_file("data/raw_videos/sample.mp4")
if "error" in video_report:
    raise RuntimeError(video_report["error"])

text_report = build_text_report(video_report)
print(text_report)
```

---

## 요구 사항

필요 패키지는 `requirements.txt`를 참고하여 설치합니다.

```bash
pip install -r requirements.txt
```

OpenCV의 Haar Cascade 파일은 `opencv-python` 패키지 설치 시 자동 포함되며, 런타임 시 `cv2.data.haarcascades` 경로에서 로드됩니다.
