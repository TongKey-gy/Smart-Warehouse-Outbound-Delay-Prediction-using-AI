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
│   ├── submissions/
│   └── submissions_local/
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
- 교차검증 수행 및 fold별 `MAE`, `RMSE` 출력
- 전체 `OOF MAE` 중심 score 집계
- `layout_info.csv` 메타데이터 사용 여부 ablation 가능
- submission 파일 생성
- `outputs/submissions_local/submission_xx.csv` 복사본 생성
- `logs/results.csv`에 실험 결과 기록
- README `실험기록` 표 자동 갱신

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

현재 `train.py` 기본 설정은 `scenario_id group_kfold` 기준의 `capacity ratios + bottleneck ratios + log target + target weighting + layout-aware blend` 조합입니다.

## 베이스라인 동작 요약

- `layout_id` 기준으로 layout 메타데이터 병합
- 타깃은 `train`에는 있고 `test`에는 없는 컬럼 중 자동 탐지
- `CONFIG`에서 `validation_type`, `group_column`, `use_layout_info`, `use_layout_id`, `use_scenario_id`, seed, LightGBM 하이퍼파라미터를 직접 제어
- `CONFIG`에서 간단한 feature engineering 토글을 켜고 끌 수 있음
- 범주형 컬럼은 ordinal encoding
- 수치형 컬럼은 median imputation
- 기본 모델은 `LightGBMRegressor`
- 리더보드 일반화와 맞추기 위해 기본 검증은 `scenario_id` 기준 `group_kfold`를 권장
- 최근 10회 탐색(`submission_48.csv` ~ `submission_57.csv`) 중 최고 `OOF MAE`는 `submission_49.csv`의 `9.217497`이었다

## 실험기록

<!-- EXPERIMENT_LOG_START -->
| 실험 번호 | 저장 파일명 (submission_xx.csv) | 원본 파일명 | 실험 시각 | 모델/전략 | 성능 점수 | 개선 사항 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | submission_01.csv | submission_baseline_lightgbm_v1_20260402_143455.csv | 2026-04-02 14:34:55 | baseline_lightgbm_v1 | 10.176448 | - |
| 2 | submission_02.csv | submission_baseline_lightgbm_v1_20260403_192737.csv | 2026-04-03 19:27:37 | baseline_lightgbm_v1 | 10.176448 | - |
| 3 | submission_03.csv | submission_baseline_lightgbm_rmse_v2_20260403_202532.csv | 2026-04-03 20:25:32 | baseline_lightgbm_rmse_v2 | 21.774106 | - |
| 4 | submission_04.csv | submission_baseline_lightgbm_rmse_v2_20260403_204521.csv | 2026-04-03 20:45:21 | baseline_lightgbm_rmse_v2 | 21.774106 | - |
| 5 | submission_05.csv | submission_baseline_lightgbm_rmse_v3_20260403_205311.csv | 2026-04-03 20:53:11 | baseline_lightgbm_rmse_v3 | 21.764786 | - |
| 6 | submission_06.csv | submission_baseline_lightgbm_rmse_v4_20260403_205948.csv | 2026-04-03 20:59:48 | baseline_lightgbm_rmse_v4 | 21.754744 | - |
| 7 | submission_07.csv | submission_blend_v2_030_v3_025_v4_045_20260403_210130.csv | 2026-04-03 21:01:30 | blend_v2_030_v3_025_v4_045 | - | - |
| 8 | submission_08.csv | submission_baseline_lightgbm_rmse_v4_child15_20260403_210741.csv | 2026-04-03 21:07:41 | baseline_lightgbm_rmse_v4_child15 | - | - |
| 9 | submission_09.csv | submission_blend_v2_024_v3_024_v4_032_child15_020_20260403_210755.csv | 2026-04-03 21:07:55 | blend_v2_024_v3_024_v4_032_child15_020 | - | - |
| 10 | submission_10.csv | submission_baseline_lightgbm_rmse_v4_seed7_20260403_211207.csv | 2026-04-03 21:12:07 | baseline_lightgbm_rmse_v4_seed7 | 21.776224 | - |
| 11 | submission_11.csv | submission_baseline_lightgbm_rmse_v4_seed21_20260403_211321.csv | 2026-04-03 21:13:21 | baseline_lightgbm_rmse_v4_seed21 | 21.769780 | - |
| 12 | submission_12.csv | submission_baseline_lightgbm_rmse_v4_seed84_20260403_211412.csv | 2026-04-03 21:14:12 | baseline_lightgbm_rmse_v4_seed84 | 21.769471 | - |
| 13 | submission_13.csv | submission_blend_nnls7_20260403_211727.csv | 2026-04-03 21:17:27 | blend_nnls7 | - | - |
| 14 | submission_14.csv | submission_exp01_group_scenario_baseline_20260404_013753.csv | 2026-04-04 01:37:53 | exp01_group_scenario_baseline | 9.844030 | baseline scenario groupkfold MAE tracking |
| 15 | submission_15.csv | submission_exp02_group_scenario_layoutid_20260404_022314.csv | 2026-04-04 02:23:14 | exp02_group_scenario_layoutid | 9.849037 | enable layout_id categorical feature |
| 16 | submission_16.csv | submission_exp03_group_layout_baseline_20260404_023028.csv | 2026-04-04 02:30:28 | exp03_group_layout_baseline | 9.907822 | switch validation grouping from scenario_id to layout_id |
| 17 | submission_17.csv | submission_exp04_kfold_baseline_20260404_023208.csv | 2026-04-04 02:32:08 | exp04_kfold_baseline | 8.417999 | switch validation to shuffled kfold baseline |
| 18 | submission_18.csv | submission_exp05_kfold_tuned_trees_20260404_023521.csv | 2026-04-04 02:35:21 | exp05_kfold_tuned_trees | 7.909776 | kfold with deeper trees and milder regularization |
| 19 | submission_19.csv | submission_exp06_kfold_regularized_20260404_023629.csv | 2026-04-04 02:36:29 | exp06_kfold_regularized | 9.170873 | kfold with smaller leaves and stronger regularization |
| 20 | submission_20.csv | submission_exp07_kfold_robot_features_20260404_023733.csv | 2026-04-04 02:37:33 | exp07_kfold_robot_features | 8.709932 | kfold plus robot balance engineered features |
| 21 | submission_21.csv | submission_exp08_kfold_workload_features_20260404_023837.csv | 2026-04-04 02:38:37 | exp08_kfold_workload_features | 8.711440 | kfold plus workload engineered features and layout_id |
| 22 | submission_22.csv | submission_exp09_kfold_env_workload_20260404_023942.csv | 2026-04-04 02:39:42 | exp09_kfold_env_workload | 8.714634 | kfold plus environment and workload features |
| 23 | submission_23.csv | submission_exp10_kfold_logtarget_combo_20260404_024046.csv | 2026-04-04 02:40:46 | exp10_kfold_logtarget_combo | 8.201912 | kfold with log target and workload features |
| 24 | submission_24.csv | submission_exp11_kfold_tuned_no_layoutinfo_20260404_094941.csv | 2026-04-04 09:49:41 | exp11_kfold_tuned_no_layoutinfo | 9.187121 | kfold tuned trees without layout_info metadata |
| 25 | submission_25.csv | submission_exp12_kfold_tuned_with_layoutinfo_20260404_095122.csv | 2026-04-04 09:51:22 | exp12_kfold_tuned_with_layoutinfo | 7.909776 | kfold tuned trees with layout_info metadata restored |
| 26 | submission_26.csv | submission_exp13_kfold_tuned_log_layoutid_20260404_111006.csv | 2026-04-04 11:10:06 | exp13_kfold_tuned_log_layoutid | 7.783874 | tuned trees with layout_info, layout_id, and log target |
| 27 | submission_27.csv | submission_exp14_kfold_tuned_log_workload_layoutid_20260404_111211.csv | 2026-04-04 11:12:11 | exp14_kfold_tuned_log_workload_layoutid | 7.794796 | tuned trees plus workload features, layout_info, layout_id, and log target |
| 28 | submission_28.csv | submission_exp15_seed7_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_20260404_123530.csv | 2026-04-04 12:35:30 | exp15_seed7_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 | 7.765284 | exp15_seed7_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13 |
| 29 | submission_29.csv | submission_exp16_seed21_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_20260404_123723.csv | 2026-04-04 12:37:23 | exp16_seed21_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 | 7.789322 | exp16_seed21_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13 |
| 30 | submission_30.csv | submission_exp17_seed84_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_20260404_123917.csv | 2026-04-04 12:39:17 | exp17_seed84_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 | 7.781999 | exp17_seed84_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13 |
| 31 | submission_31.csv | submission_exp18_lr002_leaf127_depth10_mc20_sub09_col09_ra003_rl003_20260404_124210.csv | 2026-04-04 12:42:10 | exp18_lr002_leaf127_depth10_mc20_sub09_col09_ra003_rl003 | 7.698430 | exp18_lr002_leaf127_depth10_mc20_sub09_col09_ra003_rl003 sweep around exp13 |
| 32 | submission_32.csv | submission_exp19_lr0025_leaf159_depth11_mc20_sub095_col085_ra002_rl002_20260404_124449.csv | 2026-04-04 12:44:49 | exp19_lr0025_leaf159_depth11_mc20_sub095_col085_ra002_rl002 | 7.495884 | exp19_lr0025_leaf159_depth11_mc20_sub095_col085_ra002_rl002 sweep around exp13 |
| 33 | submission_33.csv | submission_exp20_lr0035_leaf95_depth9_mc40_sub085_col095_ra008_rl008_20260404_124639.csv | 2026-04-04 12:46:39 | exp20_lr0035_leaf95_depth9_mc40_sub085_col095_ra008_rl008 | 7.747959 | exp20_lr0035_leaf95_depth9_mc40_sub085_col095_ra008_rl008 sweep around exp13 |
| 34 | submission_34.csv | submission_exp21_lr003_leaf191_depth12_mc15_sub09_col09_ra001_rl001_20260404_124924.csv | 2026-04-04 12:49:24 | exp21_lr003_leaf191_depth12_mc15_sub09_col09_ra001_rl001 | 7.297656 | exp21_lr003_leaf191_depth12_mc15_sub09_col09_ra001_rl001 sweep around exp13 |
| 35 | submission_35.csv | submission_exp22_lr0025_leaf127_depth10_mc25_sub10_col08_ra005_rl005_scenario_20260404_125142.csv | 2026-04-04 12:51:42 | exp22_lr0025_leaf127_depth10_mc25_sub10_col08_ra005_rl005_scenario | 7.631533 | exp22_lr0025_leaf127_depth10_mc25_sub10_col08_ra005_rl005_scenario sweep around exp13 |
| 36 | submission_36.csv | submission_exp23_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_robot_20260404_125339.csv | 2026-04-04 12:53:39 | exp23_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_robot | 7.757103 | exp23_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_robot sweep around exp13 |
| 37 | submission_37.csv | submission_exp24_lr0025_leaf143_depth10_mc25_sub09_col09_ra003_rl003_robot_20260404_125620.csv | 2026-04-04 12:56:20 | exp24_lr0025_leaf143_depth10_mc25_sub09_col09_ra003_rl003_robot | 7.535328 | exp24_lr0025_leaf143_depth10_mc25_sub09_col09_ra003_rl003_robot sweep around exp13 |
| 38 | submission_38.csv | submission_exp25_lr0025_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_20260404_131802.csv | 2026-04-04 13:18:02 | exp25_lr0025_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005 | 7.212686 | exp25_lr0025_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005 sweep around exp21 |
| 39 | submission_39.csv | submission_exp26_lr002_leaf191_depth12_mc10_sub09_col09_ra0_rl0_20260404_132153.csv | 2026-04-04 13:21:53 | exp26_lr002_leaf191_depth12_mc10_sub09_col09_ra0_rl0 | 7.235242 | exp26_lr002_leaf191_depth12_mc10_sub09_col09_ra0_rl0 sweep around exp21 |
| 40 | submission_40.csv | submission_exp27_lr003_leaf223_depth13_mc12_sub09_col09_ra0_rl0_20260404_132450.csv | 2026-04-04 13:24:50 | exp27_lr003_leaf223_depth13_mc12_sub09_col09_ra0_rl0 | 7.123511 | exp27_lr003_leaf223_depth13_mc12_sub09_col09_ra0_rl0 sweep around exp21 |
| 41 | submission_41.csv | submission_exp28_lr003_leaf255_depth14_mc10_sub09_col09_ra0_rl0_20260404_132749.csv | 2026-04-04 13:27:49 | exp28_lr003_leaf255_depth14_mc10_sub09_col09_ra0_rl0 | 7.109556 | exp28_lr003_leaf255_depth14_mc10_sub09_col09_ra0_rl0 sweep around exp21 |
| 42 | submission_42.csv | submission_exp29_lr0028_leaf223_depth12_mc15_sub095_col09_ra0005_rl0005_20260404_133120.csv | 2026-04-04 13:31:20 | exp29_lr0028_leaf223_depth12_mc15_sub095_col09_ra0005_rl0005 | 7.064350 | exp29_lr0028_leaf223_depth12_mc15_sub095_col09_ra0005_rl0005 sweep around exp21 |
| 43 | submission_43.csv | submission_exp30_lr0025_leaf255_depth13_mc20_sub09_col085_ra001_rl001_20260404_133518.csv | 2026-04-04 13:35:18 | exp30_lr0025_leaf255_depth13_mc20_sub09_col085_ra001_rl001 | 6.962523 | exp30_lr0025_leaf255_depth13_mc20_sub09_col085_ra001_rl001 sweep around exp21 |
| 44 | submission_44.csv | submission_exp31_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_robot_20260404_133805.csv | 2026-04-04 13:38:05 | exp31_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_robot | 7.282706 | exp31_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_robot sweep around exp21 |
| 45 | submission_45.csv | submission_exp32_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_workload_20260404_134056.csv | 2026-04-04 13:40:56 | exp32_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_workload | 7.314872 | exp32_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_workload sweep around exp21 |
| 46 | submission_46.csv | submission_exp33_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_envwork_20260404_134351.csv | 2026-04-04 13:43:51 | exp33_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_envwork | 7.314545 | exp33_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_envwork sweep around exp21 |
| 47 | submission_47.csv | submission_exp34_lr0022_leaf223_depth14_mc8_sub095_col095_ra0_rl0_20260404_134830.csv | 2026-04-04 13:48:30 | exp34_lr0022_leaf223_depth14_mc8_sub095_col095_ra0_rl0 | 6.992231 | exp34_lr0022_leaf223_depth14_mc8_sub095_col095_ra0_rl0 sweep around exp21 |
| 48 | submission_48.csv | submission_exp48_group_scenario_baseline_tuned_v1_20260404_143149.csv | 2026-04-04 14:31:49 | exp48_group_scenario_baseline_tuned_v1 | 9.843006 | groupkfold scenario baseline tuned v1 |
| 49 | submission_49.csv | submission_exp49_group_scenario_baseline_tuned_log_20260404_143238.csv | 2026-04-04 14:32:38 | exp49_group_scenario_baseline_tuned_log | 9.217497 | groupkfold baseline tuned v1 with log target |
| 50 | submission_50.csv | submission_exp50_group_scenario_compact_regularized_20260404_143315.csv | 2026-04-04 14:33:15 | exp50_group_scenario_compact_regularized | 9.863333 | groupkfold more regularized compact trees |
| 51 | submission_51.csv | submission_exp51_group_scenario_mid_deep_20260404_143405.csv | 2026-04-04 14:34:05 | exp51_group_scenario_mid_deep | 9.839829 | groupkfold mid depth larger leaves |
| 52 | submission_52.csv | submission_exp52_group_scenario_slow_large_20260404_143504.csv | 2026-04-04 14:35:04 | exp52_group_scenario_slow_large | 9.833989 | groupkfold slower larger model |
| 53 | submission_53.csv | submission_exp53_group_scenario_no_layoutid_20260404_143552.csv | 2026-04-04 14:35:52 | exp53_group_scenario_no_layoutid | 9.831477 | groupkfold remove raw layout_id while keeping layout metadata |
| 54 | submission_54.csv | submission_exp54_group_scenario_workload_20260404_143643.csv | 2026-04-04 14:36:43 | exp54_group_scenario_workload | 9.835349 | groupkfold tuned trees plus workload features |
| 55 | submission_55.csv | submission_exp55_group_scenario_robot_20260404_143733.csv | 2026-04-04 14:37:33 | exp55_group_scenario_robot | 9.845183 | groupkfold tuned trees plus robot balance features |
| 56 | submission_56.csv | submission_exp56_group_scenario_env_workload_20260404_143826.csv | 2026-04-04 14:38:26 | exp56_group_scenario_env_workload | 9.838691 | groupkfold tuned trees plus environment and workload features |
| 57 | submission_57.csv | submission_exp57_group_scenario_seed7_mid_deep_20260404_143916.csv | 2026-04-04 14:39:16 | exp57_group_scenario_seed7_mid_deep | 9.833067 | groupkfold mid depth larger leaves alternate seed |
| 58 | submission_58.csv | submission_default_groupkfold_scenario_log_tuned_v2_20260404_144812.csv | 2026-04-04 14:48:12 | default_groupkfold_scenario_log_tuned_v2 | 9.217497 | - |
| 59 | submission_59.csv | submission_exp59_group_log_clip_floor_20260404_145111.csv | 2026-04-04 14:51:11 | exp59_group_log_clip_floor | 9.217488 | clip negative predictions at zero for non-negative delay target |
| 60 | submission_60.csv | submission_exp60_group_log_capacity_20260404_145211.csv | 2026-04-04 14:52:11 | exp60_group_log_capacity | 9.167677 | add capacity pressure ratios between inflow, pack stations, docks, and chargers |
| 61 | submission_61.csv | submission_exp61_group_log_capacity_congestion_20260404_145307.csv | 2026-04-04 14:53:07 | exp61_group_log_capacity_congestion | 9.169492 | stack capacity pressure features with congestion interactions |
| 62 | submission_62.csv | submission_exp62_group_log_capacity_temporal_20260404_145407.csv | 2026-04-04 14:54:07 | exp62_group_log_capacity_temporal | 9.169118 | combine capacity ratios with cyclical shift and weekday features |
| 63 | submission_63.csv | submission_exp63_group_log_capacity_layout_20260404_145504.csv | 2026-04-04 14:55:04 | exp63_group_log_capacity_layout | 9.174876 | combine capacity ratios with layout density and charger coverage features |
| 64 | submission_64.csv | submission_exp64_group_log_capacity_sqrt_weight_20260404_145606.csv | 2026-04-04 14:56:06 | exp64_group_log_capacity_sqrt_weight | 9.148250 | apply mild sqrt target weighting on top of capacity ratios to reduce tail underprediction |
| 65 | submission_65.csv | submission_exp65_group_log_capacity_log_weight_20260404_145713.csv | 2026-04-04 14:57:13 | exp65_group_log_capacity_log_weight | 9.146279 | swap sqrt weighting for smoother log target weighting on capacity features |
| 66 | submission_66.csv | submission_exp66_group_capacity_log_weight_mae_obj_20260404_145827.csv | 2026-04-04 14:58:27 | exp66_group_capacity_log_weight_mae_obj | 9.175939 | align objective with MAE while keeping best capacity and log-weight settings |
| 67 | submission_67.csv | submission_exp67_group_capacity_log_weight_deeper_20260404_145944.csv | 2026-04-04 14:59:44 | exp67_group_capacity_log_weight_deeper | 9.145386 | increase tree capacity for the best capacity plus log-weight configuration |
| 68 | submission_68.csv | submission_exp68_group_capacity_log_weight_deeper_relaxed_20260404_150112.csv | 2026-04-04 15:01:12 | exp68_group_capacity_log_weight_deeper_relaxed | 9.148869 | relax regularization and enlarge trees for the best weighted capacity setup |
| 69 | submission_69.csv | submission_exp69_group_capacity_log_weight_deeper_no_layoutid_20260404_150231.csv | 2026-04-04 15:02:31 | exp69_group_capacity_log_weight_deeper_no_layoutid | 9.140415 | drop raw layout_id from the best weighted capacity setup to reduce layout-specific overfit |
| 70 | submission_70.csv | submission_default_groupkfold_capacity_weighted_v1_20260404_150436.csv | 2026-04-04 15:04:36 | default_groupkfold_capacity_weighted_v1 | 9.140415 | - |
| 71 | submission_71.csv | submission_exp71_capacity_weighted_bottleneck_v1_20260404_151420.csv | 2026-04-04 15:14:20 | exp71_capacity_weighted_bottleneck_v1 | 9.126865 | add bottleneck ratio features from blocked paths, truck wait, queues, and staffing pressure |
| 72 | submission_72.csv | submission_exp72_bottleneck_logweight_025_20260404_151543.csv | 2026-04-04 15:15:43 | exp72_bottleneck_logweight_025 | 9.131888 | raise log target weighting strength on top of bottleneck features |
| 73 | submission_73.csv | submission_exp73_bottleneck_logweight_015_20260404_151656.csv | 2026-04-04 15:16:56 | exp73_bottleneck_logweight_015 | 9.128075 | lower log target weighting after adding bottleneck features |
| 74 | submission_74.csv | submission_exp74_bottleneck_deeper_v1_20260404_151827.csv | 2026-04-04 15:18:27 | exp74_bottleneck_deeper_v1 | 9.141117 | increase tree capacity for bottleneck ratio features |
| 75 | submission_75.csv | submission_exp75_bottleneck_blend_layoutid_unweighted_20260404_152042.csv | 2026-04-04 15:20:42 | exp75_bottleneck_blend_layoutid_unweighted | 9.120463 | blend weighted no-layoutid model with unweighted layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none |
| 76 | submission_76.csv | submission_exp76_bottleneck_blend_layoutid_unweighted_w15_20260404_152256.csv | 2026-04-04 15:22:56 | exp76_bottleneck_blend_layoutid_unweighted_w15 | 9.122095 | reduce secondary layoutid blend weight to 0.15 | blend_secondary_model weight=0.15 secondary_use_layout_id=True secondary_target_weight_mode=none |
| 77 | submission_77.csv | submission_exp77_bottleneck_blend_unweighted_only_20260404_152506.csv | 2026-04-04 15:25:06 | exp77_bottleneck_blend_unweighted_only | 9.121387 | blend weighted bottleneck model with unweighted no-layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=False secondary_target_weight_mode=none |
| 78 | submission_78.csv | submission_exp78_bottleneck_blend_simple_secondary_20260404_152801.csv | 2026-04-04 15:28:01 | exp78_bottleneck_blend_simple_secondary | 9.120463 | blend bottleneck primary with simpler layoutid secondary baseline | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none |
| 79 | submission_79.csv | submission_exp78_bottleneck_blend_simple_secondary_20260404_152827.csv | 2026-04-04 15:28:27 | exp78_bottleneck_blend_simple_secondary | 9.122371 | blend bottleneck primary with simpler capacity-only layoutid secondary model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none |
| 80 | submission_80.csv | submission_exp79_bottleneck_blend_layoutid_unweighted_w35_20260404_153045.csv | 2026-04-04 15:30:45 | exp79_bottleneck_blend_layoutid_unweighted_w35 | 9.120201 | increase layoutid secondary blend weight to 0.35 | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none |
| 81 | submission_81.csv | submission_exp80_bottleneck_blend_layoutid_softweighted_20260404_153306.csv | 2026-04-04 15:33:06 | exp80_bottleneck_blend_layoutid_softweighted | 9.121146 | use lightly weighted layoutid secondary model in the best bottleneck blend | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=log |
| 82 | submission_82.csv | submission_default_groupkfold_bottleneck_blend_v1_20260404_153545.csv | 2026-04-04 15:35:45 | default_groupkfold_bottleneck_blend_v1 | 9.120201 | - | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none |
<!-- EXPERIMENT_LOG_END -->

## 비고

- 제출 형식은 `sample_submission.csv`를 유지합니다.
- 생성된 제출 파일 원본은 `outputs/submissions/` 아래 저장됩니다.
- README 기록용 복사본은 `outputs/submissions_local/submission_xx.csv` 형식으로 로컬에만 저장됩니다.
- 실험 로그는 `logs/results.csv`에 누적됩니다.
- `train.py` 실행이 끝나면 README `실험기록` 표가 자동 갱신됩니다.
