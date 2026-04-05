# Session Memory 2026-04-05

## Goal

`seen_layout` 성능을 개선할 수 있는 방향을 찾고, 필요하면 멀티모델 앙상블까지 검증한다.

## Current Best Result

- Best baseline in `train.py` family:
  - experiment: `default_groupkfold_bottleneck_blend_v1`
  - `seen_layout_oof_mae=9.745907`
  - `unseen_layout_oof_mae=9.134554`
  - `hybrid_layout_score=9.501366`

- Best overall result found in this session:
  - pipeline: `pb_10_2_pipeline.py`
  - model families: `LightGBM + XGBoost + CatBoost`
  - notebook feature system + notebook hybrid CV used as-is
  - `seen_layout ensemble MAE=9.450760`
  - `unseen_layout ensemble MAE=8.799830`
  - `Hybrid ensemble CV score=9.190388`

## Key Conclusion

`train.py` 기반 미세조정보다, `PB_10.2코드공유.ipynb`를 추출한 `pb_10_2_pipeline.py` 기반 풀 멀티모델 앙상블이 훨씬 강했다.

즉 다음 세션에서는 `train.py` 계열을 더 미세조정하기보다, `pb_10_2_pipeline.py`를 기준선으로 삼아 추가 튜닝하거나 저장소 메인 실험 흐름에 통합하는 쪽이 우선순위가 높다.

## What Was Tried

### 1. Notebook Extraction

- Created `pb_10_2_pipeline.py` from `PB_10.2코드공유.ipynb`
- Adjusted data paths to `open/`
- Added `main()` guard

### 2. `train.py` Enhancements

- Added hybrid layout CV reporting:
  - `seen_layout`
  - `unseen_layout`
  - `hybrid_layout_score`
- Added notebook-inspired feature port switches
- Added grouped notebook port feature sets:
  - `battery`
  - `flow`
  - `mass`
  - `interaction_light`
  - `all`

### 3. `train.py` Experiment Findings

- Ported 10 notebook-style features did not beat the baseline on `seen_layout`
- `battery` group was the least harmful, but still worse than baseline
- `flow` and `mass` groups were clearly worse
- `secondary_use_layout_id=False` did not improve over baseline
- Lightweight ensemble attempt with `LightGBM + HistGradientBoosting` also failed:
  - `seen_layout OOF MAE=9.767518`
  - `unseen_layout OOF MAE=9.137802`
  - `hybrid_layout_score=9.515631`

### 4. Full Notebook-Style Multi-Model Run

- Installed `xgboost` and `catboost`
- Ran full `pb_10_2_pipeline.py`
- Output files:
  - `outputs/pb_10_2_pipeline/cv_results.json`
  - `outputs/pb_10_2_pipeline/model_bundle.joblib`
  - `outputs/pb_10_2_pipeline/submission.csv`

## Recommended Next Step

Resume from the notebook-style pipeline, not from the `train.py` baseline path.

Suggested order:

1. Read `outputs/pb_10_2_pipeline/cv_results.json`
2. Inspect auto weights and recommended iterations
3. Run one follow-up multi-model experiment by adjusting notebook config
4. If stable, integrate the notebook pipeline into the repo's main experiment workflow

## Local Code Changes Not Yet Pushed

These files were modified or created locally during this session:

- `train.py`
- `pb_10_2_pipeline.py`
- `lightweight_ensemble_experiment.py`
- `README.md`
- `EXPERIMENT_PORTFOLIO.md`
- `logs/results.csv`

If a later session needs the exact work state, inspect these local files first.
