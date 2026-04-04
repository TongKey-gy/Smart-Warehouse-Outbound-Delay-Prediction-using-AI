"""Adaptive gkf tuning strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_gkf_tuning_09`의 OOF MAE는 9.113235, fold std는 0.236388였다. baseline 대비 +0.005753, mean residual 3.3602, tail residual mean 163.8906다. 세션 최고는 `adaptive_gkf_tuning_05`의 9.108616다.'
PROPOSAL = '현재 최고 설정을 마지막으로 한 번 더 좁게 재확인한다.'
CURRENT_STRATEGY = {
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 127,
    'max_depth': 11,
    'min_child_samples': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'target_weight_strength': 0.4,
    'secondary_weight': 0.22,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_gkf_tuning_10',
}
