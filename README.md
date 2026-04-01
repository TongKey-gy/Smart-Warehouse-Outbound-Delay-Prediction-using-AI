# Smart Warehouse Outbound Delay Prediction using AI

## 1. 프로젝트 소개

이 프로젝트는 데이콘 `스마트 창고 출고 지연 예측 AI 경진대회` 참가를 위해 구축한 머신러닝 프로젝트입니다. 스마트 물류 운영 데이터에 기반해 향후 출고 지연을 예측하고, 창고 운영 효율화에 활용할 수 있는 예측 모델을 개발하는 것을 목표로 합니다.

본 프로젝트는 **AMR(Autonomous Mobile Robot) 기반 스마트 물류창고 시뮬레이션 환경**에서 생성된 데이터를 활용하며, 실제 물류 창고에서 발생할 수 있는 다양한 병목 상황을 반영한 운영 데이터를 기반으로 분석 및 예측을 수행합니다.

핵심 목표는 시계열 기반 창고 운영 데이터를 활용하여 **물류 병목을 사전에 감지하고 출고 지연을 정량적으로 예측하는 모델**을 개발하는 것입니다.

> Reference: https://dacon.io/competitions/official/236696/overview/description

## 2. 문제 정의

본 프로젝트는 **스마트 창고 출고 지연 예측 AI 경진대회**를 기반으로 진행된 회귀 문제입니다.

- 입력 데이터: 특정 시나리오의 특정 시점에서 관측된 창고 운영 상태 스냅샷
- 예측 목표: 향후 30분 동안 발생할 평균 출고 지연 시간 예측
- 문제 유형: `Regression`
- Target 변수: `avg_delay_minutes_next_30m`

현재 시점의 운영 상태를 바탕으로 향후 출고 지연을 추정해야 하므로, 시점 간 관계와 운영 변수 간 상호작용을 함께 고려하는 것이 중요합니다.

## 3. 데이터 설명

대회 데이터는 GitHub의 [`open/`](https://github.com/TongKey-gy/Smart-Warehouse-Outbound-Delay-Prediction-using-AI/tree/main/open) 경로에 업로드되어 있으며, 배포 구조는 아래와 같습니다.

```text
open.zip
├── train.csv
├── test.csv
├── layout_info.csv
└── sample_submission.csv
```

데이터는 각 행이 특정 시나리오의 특정 시점에서 관측된 창고 운영 상태를 나타내는 구조이며, 현재 상태를 기반으로 미래 30분 평균 출고 지연 시간을 예측하는 데 사용됩니다.

### 데이터 파일 구성

| File Name | Shape | Description |
| --- | ---: | --- |
| `train.csv` | 250,000 × 94 | 모델 학습용 데이터. 식별자, 시나리오 정보, 90개 운영 피처, 타깃 컬럼 포함 |
| `test.csv` | 50,000 × 93 | 평가용 데이터. `train.csv`와 동일한 입력 구조이며 타깃 컬럼 제외 |
| `layout_info.csv` | 300 × 15 | 창고 레이아웃 관련 보조 테이블. `layout_id` 기준으로 병합 가능한 메타 정보 제공 |
| `sample_submission.csv` | 50,000 × 2 | 제출 형식 예시. 테스트 샘플별 예측값 작성용 |

### `train.csv`

| Column Group | Column Name | Description |
| --- | --- | --- |
| Identifier | `ID` | 각 샘플의 고유 식별자 |
| Scenario Metadata | `layout_id` | 창고 레이아웃 식별자. `layout_info.csv`와 조인 가능 |
| Scenario Metadata | `scenario_id` | 운영 시나리오 식별자 |
| Features | `feature_1` ~ `feature_90` | 특정 시점의 창고 운영 상태를 나타내는 90개 피처 컬럼 |
| Target | `avg_delay_minutes_next_30m` | 향후 30분 동안의 평균 출고 지연 시간(분). 예측 대상 |

### `test.csv`

| Column Group | Column Name | Description |
| --- | --- | --- |
| Identifier | `ID` | 각 평가 샘플의 고유 식별자 |
| Scenario Metadata | `layout_id` | 창고 레이아웃 식별자 |
| Scenario Metadata | `scenario_id` | 운영 시나리오 식별자 |
| Features | `feature_1` ~ `feature_90` | `train.csv`와 동일한 입력 피처 |

### `layout_info.csv`

| Column Group | Column Name | Description |
| --- | --- | --- |
| Key | `layout_id` | 창고 레이아웃 식별자 |
| Layout Metadata | `layout_feature_1` ~ `layout_feature_14` | 레이아웃 구조, 설비 배치, 운영 환경 등 레이아웃 관련 보조 정보 |

### `sample_submission.csv`

| Column Name | Description |
| --- | --- |
| `ID` | `test.csv`의 샘플 고유 식별자 |
| `avg_delay_minutes_next_30m` | 예측한 향후 30분 평균 출고 지연 시간(분) |

### 주요 입력 변수

- 입력 피처는 창고 운영 상태를 표현하는 시점별 스냅샷 변수로 구성됩니다.
- 주요 정보에는 로봇 운영 상태, 주문 유입량, 배터리 및 충전 상태, 통로 혼잡도, 기타 물류 운영 지표가 포함될 수 있습니다.
- `layout_info.csv`는 정적 메타데이터 성격이 강하므로, `layout_id` 기준 병합 후 파생변수 생성에 활용할 수 있습니다.

### 예측 타깃

| Target | Meaning | Unit |
| --- | --- | --- |
| `avg_delay_minutes_next_30m` | 현재 시점 이후 향후 30분 동안 발생한 평균 출고 지연 시간 | minutes |

### 데이터 누수 및 검증 시 주의사항

- 타깃 `avg_delay_minutes_next_30m`는 **미래 30분 정보**를 요약한 값이므로, 이를 직접 또는 간접적으로 복원할 수 있는 파생변수 생성은 데이터 누수로 이어질 수 있습니다.
- 동일한 `scenario_id` 내 샘플은 서로 강하게 연관될 가능성이 높으므로, 무작위 행 단위 분할보다 **시나리오 단위 분할** 또는 그룹 기반 검증을 우선 고려하는 것이 안전합니다.
- `layout_id`를 통해 `layout_info.csv`를 병합하는 것은 가능하지만, 검증 데이터에만 존재해야 할 통계값을 전체 데이터 기준으로 집계해 사용하는 방식은 피해야 합니다.
- 전처리 과정에서 스케일링, 인코딩, 결측치 대체, 타깃 인코딩 등을 수행할 경우 반드시 **훈련 데이터 기준으로만 적합(fit)** 하고 검증/테스트에는 동일 변환만 적용해야 합니다.
- `ID`는 고유 식별자이므로 일반적으로 예측 신호보다는 인덱스 역할에 가깝습니다. 모델 입력에 사용할 경우 과적합 여부를 반드시 점검해야 합니다.

### 데이터 활용 흐름

1. `train.csv`를 기준으로 모델 학습 및 검증 세트를 구성합니다.
2. 필요 시 `layout_info.csv`를 `layout_id` 기준으로 병합합니다.
3. 검증 전략은 `scenario_id` 또는 시나리오 단위 그룹 분할을 우선 검토합니다.
4. 학습된 모델로 `test.csv`를 예측한 뒤, `sample_submission.csv` 형식에 맞춰 제출 파일을 생성합니다.

## 4. EDA

## 5. 모델링

## 6. 실험 결과

## 7. 실행 방법

## 8. 폴더 구조

## 9. 기술 스택

## 10. 향후 개선 방향
