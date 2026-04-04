"""Adaptive delay-risk strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_delay_risk_09`의 OOF MAE는 9.116918, fold std는 0.234705였다. baseline 대비 +0.009436, mean residual 3.2715, tail residual mean 163.0921다. 세션 최고는 `adaptive_delay_risk_06`의 9.108772다.'
PROPOSAL = '마지막으로 보조모델에도 같은 flow risk를 얕게 공유해 진짜 역할 분리가 필요한지 확인한다.'
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
    'delay_risk_feature_set': 'plus_flow',
    'secondary_delay_risk_feature_set': 'plus_flow',
    'target_weight_strength': 0.4,
    'secondary_weight': 0.25,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_delay_risk_10',
}
