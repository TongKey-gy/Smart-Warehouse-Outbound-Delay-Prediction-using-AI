# Smart Warehouse Outbound Delay Prediction using AI

데이콘 `스마트 창고 출고 지연 예측 AI 경진대회`를 위한 정형(tabular) 회귀 실험 저장소입니다.
이 저장소는 `karpathy/autoresearch` 스타일을 참고해, 안정적인 데이터 준비 레이어와 AI가 반복 수정할 실험 레이어를 분리한 구조로 재구성되었습니다.

## 문제 개요

- 문제 유형: 회귀
- 입력 데이터: 창고 운영 상태 스냅샷
- 예측 대상: 향후 30분 평균 출고 지연 시간
- 주요 특성: `layout_info.csv` 병합 가능, `scenario_id` 기반 그룹 검증 가능

## 데이터 위치

원본 데이터는 GitHub에 포함하지 않습니다.
`prepare.py`를 실행하면 `open/` 폴더가 없을 때 Google Drive 공유 링크에서 `open.zip`을 자동 다운로드하고 압축을 해제합니다.
이미 `open/` 폴더가 있으면 다운로드와 압축 해제를 건너뜁니다.

```text
open.zip
open/
├── train.csv
├── test.csv
├── layout_info.csv
└── sample_submission.csv
```

`open/` 폴더의 raw 데이터는 수정하지 않습니다.

## 저장소 구조

```text
.
├── prepare.py
├── train.py
├── program.md
├── cache/
├── logs/
│   └── results.csv
├── outputs/
│   └── submissions/
└── README.md
```

## 파일 역할

### `prepare.py`

고정 데이터 준비 레이어입니다.

- `open/` 데이터 존재 여부 검사
- `open/` 폴더가 없으면 Google Drive에서 `open.zip` 자동 다운로드
- `open.zip` 압축 해제 및 expected 파일 검증
- `train.csv`, `test.csv`, `layout_info.csv`, `sample_submission.csv` 로드
- `layout_info.csv` 자동 병합
- 타깃 컬럼 자동 탐지
- `scenario_id` 존재 시 group split 준비
- 재사용 가능한 데이터 준비 함수 제공

### `train.py`

반복 실험 파일입니다.

- `prepare.py`에서 준비된 데이터 사용
- 베이스라인 모델 학습
- 교차검증 수행 및 fold score 출력
- 전체 score 집계
- submission 파일 생성
- `logs/results.csv`에 실험 결과 기록

### `program.md`

autoresearch 루프용 실험 규칙입니다.

- 목표 metric 개선 방향 정의
- 수정 가능한 파일과 수정 금지 파일 정의
- 성능 개선 시 유지, 악화 시 롤백 규칙 정의

## 실행 방법

가상환경이 활성화된 상태에서 아래처럼 실행합니다.

먼저 `gdown`이 없으면 설치하세요.

```bash
pip install gdown
```

또는 현재 작업 환경의 가상환경을 쓴다면:

```bash
./.venv/bin/pip install gdown
```

그 다음 아래처럼 실행합니다.

```bash
python prepare.py
python train.py
```

현재 작업 환경에서는 시스템 `python`이 없을 수 있으므로, 필요하면 아래처럼 실행하세요.

```bash
./.venv/bin/python prepare.py
./.venv/bin/python train.py
```

기본 Google Drive 링크를 바꾸고 싶으면 `OPEN_DATA_URL` 환경변수를 사용할 수 있습니다.

## 베이스라인 동작 요약

- `layout_id` 기준으로 layout 메타데이터 병합
- 타깃은 `train`에는 있고 `test`에는 없는 컬럼 중 자동 탐지
- `scenario_id`가 있으면 `GroupKFold`, 없으면 `KFold`
- 범주형 컬럼은 ordinal encoding
- 수치형 컬럼은 median imputation
- 기본 모델은 `HistGradientBoostingRegressor`

## 비고

- 제출 형식은 `sample_submission.csv`를 유지합니다.
- 생성된 제출 파일은 `outputs/submissions/` 아래 저장됩니다.
- 실험 로그는 `logs/results.csv`에 누적됩니다.
