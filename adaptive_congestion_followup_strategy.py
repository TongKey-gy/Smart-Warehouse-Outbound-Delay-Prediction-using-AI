"""Follow-up congestion strategy state for the next training run."""

ITERATION = 3
ANALYSIS = '직전 실험 `adaptive_congestion_followup_02`의 OOF MAE는 9.114007, fold std는 0.237610였다. baseline 대비 +0.006525, congestion reference 대비 +0.006272다.'
PROPOSAL = '직전 개선이 있으면 보조모델에 아주 약한 log weighting을 더해 tail 보정을 미세하게 확인한다.'
CURRENT_STRATEGY = {
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 95,
    'max_depth': 11,
    'min_child_samples': 27,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'add_congestion_features': True,
    'target_weight_strength': 0.3,
    'secondary_weight': 0.22,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'log',
    'secondary_target_weight_strength': 0.02,
    'experiment_name': 'adaptive_congestion_followup_03',
}
