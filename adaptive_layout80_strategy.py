"""Adaptive layout strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_layout80_09`의 OOF MAE는 9.113797, fold std는 0.236183였다. baseline 대비 -0.006404, mean residual 3.4746, tail residual mean 164.7175다. 세션 최고는 `adaptive_layout80_06`의 9.108352다.'
PROPOSAL = '현재 최고 조합을 더 촘촘한 fold로 다시 확인해 layout 신호의 안정성을 본다.'
CURRENT_STRATEGY = {
    'n_splits': 7,
    'use_layout_info': True,
    'use_layout_id': True,
    'add_layout_interaction_features': True,
    'layout_feature_set': 'plus_density',
    'add_delay_risk_features': False,
    'delay_risk_feature_set': 'base',
    'target_weight_strength': 0.24,
    'secondary_weight': 0.25,
    'secondary_use_layout_id': True,
    'secondary_layout_feature_set': 'plus_density',
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_layout80_10',
}
