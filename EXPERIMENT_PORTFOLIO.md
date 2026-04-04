# Experiment Portfolio

실험 로그를 포트폴리오 형식으로 자동 정리한 문서입니다.

### Experiment 01 — Baseline scenario groupkfold MAE tracking

**Objective**
baseline scenario groupkfold MAE tracking를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=800, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.844030

**Conclusion**
첫 기준 실험으로 사용한 설정이며, 이후 탐색의 출발점으로 삼을 수 있는 점수를 확보했다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 02 — Enable layout_id categorical feature

**Objective**
enable layout_id categorical feature를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.03, n_estimators=800, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.849037

**Conclusion**
이전 실험 대비 MAE 변화가 0.005007로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 03 — Switch validation grouping from scenario_id to layout_id

**Objective**
switch validation grouping from scenario_id to layout_id를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `layout_id`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=800, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.907822

**Conclusion**
이전 실험 대비 MAE가 0.058785 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 04 — Switch validation to shuffled kfold baseline

**Objective**
switch validation to shuffled kfold baseline를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=800, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 8.417999

**Conclusion**
이전 실험 대비 MAE가 1.489823 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 05 — Kfold with deeper trees and milder regularization

**Objective**
kfold with deeper trees and milder regularization를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=600, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 7.909776

**Conclusion**
이전 실험 대비 MAE가 0.508223 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 06 — Kfold with smaller leaves and stronger regularization

**Objective**
kfold with smaller leaves and stronger regularization를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=500, num_leaves=31, max_depth=6, min_child_samples=50.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.170873

**Conclusion**
이전 실험 대비 MAE가 1.261097 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 07 — Kfold plus robot balance engineered features

**Objective**
kfold plus robot balance engineered features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: robot balance. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=500, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 robot balance 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 8.709932

**Conclusion**
이전 실험 대비 MAE가 0.460941 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 08 — Kfold plus workload engineered features and layout_id

**Objective**
kfold plus workload engineered features and layout_id를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: workload. layout_id: enabled. hyperparameters: learning_rate=0.03, n_estimators=500, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 8.711440

**Conclusion**
이전 실험 대비 MAE 변화가 0.001508로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 09 — Kfold plus environment and workload features

**Objective**
kfold plus environment and workload features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: environment, workload. layout_id: enabled. hyperparameters: learning_rate=0.03, n_estimators=500, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 environment, workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 8.714634

**Conclusion**
이전 실험 대비 MAE 변화가 0.003194로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 10 — Kfold with log target and workload features

**Objective**
kfold with log target and workload features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: workload. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.05, n_estimators=500, num_leaves=63, max_depth=7, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 8.201912

**Conclusion**
이전 실험 대비 MAE가 0.512722 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 11 — Kfold tuned trees without layout_info metadata

**Objective**
kfold tuned trees without layout_info metadata를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=600, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.187121

**Conclusion**
이전 실험 대비 MAE가 0.985209 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 12 — Kfold tuned trees with layout_info metadata restored

**Objective**
kfold tuned trees with layout_info metadata restored를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: disabled. hyperparameters: learning_rate=0.03, n_estimators=600, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 7.909776

**Conclusion**
이전 실험 대비 MAE가 1.277345 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 13 — Tuned trees with layout_info, layout_id, and log target

**Objective**
tuned trees with layout_info, layout_id, and log target를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.783874

**Conclusion**
이전 실험 대비 MAE가 0.125902 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 14 — Tuned trees plus workload features, layout_info, layout_id, and log target

**Objective**
tuned trees plus workload features, layout_info, layout_id, and log target를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: workload. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.794796

**Conclusion**
이전 실험 대비 MAE가 0.010922 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 15 — Exp15_seed7_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13

**Objective**
exp15_seed7_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.765284

**Conclusion**
이전 실험 대비 MAE가 0.029512 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 16 — Exp16_seed21_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13

**Objective**
exp16_seed21_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.789322

**Conclusion**
이전 실험 대비 MAE가 0.024039 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 17 — Exp17_seed84_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13

**Objective**
exp17_seed84_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.781999

**Conclusion**
이전 실험 대비 MAE 변화가 -0.007323로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 18 — Exp18_lr002_leaf127_depth10_mc20_sub09_col09_ra003_rl003 sweep around exp13

**Objective**
exp18_lr002_leaf127_depth10_mc20_sub09_col09_ra003_rl003 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.02, n_estimators=1200, num_leaves=127, max_depth=10, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.698430

**Conclusion**
이전 실험 대비 MAE가 0.083569 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 19 — Exp19_lr0025_leaf159_depth11_mc20_sub095_col085_ra002_rl002 sweep around exp13

**Objective**
exp19_lr0025_leaf159_depth11_mc20_sub095_col085_ra002_rl002 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=159, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.495884

**Conclusion**
이전 실험 대비 MAE가 0.202546 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 20 — Exp20_lr0035_leaf95_depth9_mc40_sub085_col095_ra008_rl008 sweep around exp13

**Objective**
exp20_lr0035_leaf95_depth9_mc40_sub085_col095_ra008_rl008 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.035, n_estimators=800, num_leaves=95, max_depth=9, min_child_samples=40.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.747959

**Conclusion**
이전 실험 대비 MAE가 0.252075 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 21 — Exp21_lr003_leaf191_depth12_mc15_sub09_col09_ra001_rl001 sweep around exp13

**Objective**
exp21_lr003_leaf191_depth12_mc15_sub09_col09_ra001_rl001 sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=191, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.297656

**Conclusion**
이전 실험 대비 MAE가 0.450303 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 22 — Exp22_lr0025_leaf127_depth10_mc25_sub10_col08_ra005_rl005_scenario sweep around exp13

**Objective**
exp22_lr0025_leaf127_depth10_mc25_sub10_col08_ra005_rl005_scenario sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.631533

**Conclusion**
이전 실험 대비 MAE가 0.333877 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 23 — Exp23_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_robot sweep around exp13

**Objective**
exp23_lr003_leaf127_depth10_mc30_sub09_col09_ra005_rl005_robot sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: robot balance. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=700, num_leaves=127, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; robot balance 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.757103

**Conclusion**
이전 실험 대비 MAE가 0.125571 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 24 — Exp24_lr0025_leaf143_depth10_mc25_sub09_col09_ra003_rl003_robot sweep around exp13

**Objective**
exp24_lr0025_leaf143_depth10_mc25_sub09_col09_ra003_rl003_robot sweep around exp13를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: robot balance. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=143, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; robot balance 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.535328

**Conclusion**
이전 실험 대비 MAE가 0.221776 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 25 — Exp25_lr0025_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005 sweep around exp21

**Objective**
exp25_lr0025_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.025, n_estimators=1200, num_leaves=191, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.212686

**Conclusion**
이전 실험 대비 MAE가 0.322642 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 26 — Exp26_lr002_leaf191_depth12_mc10_sub09_col09_ra0_rl0 sweep around exp21

**Objective**
exp26_lr002_leaf191_depth12_mc10_sub09_col09_ra0_rl0 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.02, n_estimators=1500, num_leaves=191, max_depth=12, min_child_samples=10.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.235242

**Conclusion**
이전 실험 대비 MAE가 0.022556 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 27 — Exp27_lr003_leaf223_depth13_mc12_sub09_col09_ra0_rl0 sweep around exp21

**Objective**
exp27_lr003_leaf223_depth13_mc12_sub09_col09_ra0_rl0 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=1000, num_leaves=223, max_depth=13, min_child_samples=12.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.123511

**Conclusion**
이전 실험 대비 MAE가 0.111731 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 28 — Exp28_lr003_leaf255_depth14_mc10_sub09_col09_ra0_rl0 sweep around exp21

**Objective**
exp28_lr003_leaf255_depth14_mc10_sub09_col09_ra0_rl0 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=255, max_depth=14, min_child_samples=10.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.109556

**Conclusion**
이전 실험 대비 MAE가 0.013955 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 29 — Exp29_lr0028_leaf223_depth12_mc15_sub095_col09_ra0005_rl0005 sweep around exp21

**Objective**
exp29_lr0028_leaf223_depth12_mc15_sub095_col09_ra0005_rl0005 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.028, n_estimators=1100, num_leaves=223, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 7.064350

**Conclusion**
이전 실험 대비 MAE가 0.045207 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 30 — Exp30_lr0025_leaf255_depth13_mc20_sub09_col085_ra001_rl001 sweep around exp21

**Objective**
exp30_lr0025_leaf255_depth13_mc20_sub09_col085_ra001_rl001 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.025, n_estimators=1200, num_leaves=255, max_depth=13, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 6.962523

**Conclusion**
이전 실험 대비 MAE가 0.101827 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 31 — Exp31_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_robot sweep around exp21

**Objective**
exp31_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_robot sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: robot balance. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=191, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; robot balance 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.282706

**Conclusion**
이전 실험 대비 MAE가 0.320183 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 32 — Exp32_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_workload sweep around exp21

**Objective**
exp32_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_workload sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: workload. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=191, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.314872

**Conclusion**
이전 실험 대비 MAE가 0.032166 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 33 — Exp33_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_envwork sweep around exp21

**Objective**
exp33_lr003_leaf191_depth12_mc15_sub09_col09_ra0005_rl0005_envwork sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. features: environment, workload. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=191, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; environment, workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 7.314545

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000327로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 34 — Exp34_lr0022_leaf223_depth14_mc8_sub095_col095_ra0_rl0 sweep around exp21

**Objective**
exp34_lr0022_leaf223_depth14_mc8_sub095_col095_ra0_rl0 sweep around exp21를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: `kfold`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.022, n_estimators=1600, num_leaves=223, max_depth=14, min_child_samples=8.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 6.992231

**Conclusion**
이전 실험 대비 MAE가 0.322314 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 48 — Groupkfold scenario baseline tuned v1

**Objective**
groupkfold scenario baseline tuned v1를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.843006

**Conclusion**
이전 실험 대비 MAE가 2.850775 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 49 — Groupkfold baseline tuned v1 with log target

**Objective**
groupkfold baseline tuned v1 with log target를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 9.217497

**Conclusion**
이전 실험 대비 MAE가 0.625509 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 50 — Groupkfold more regularized compact trees

**Objective**
groupkfold more regularized compact trees를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.035, n_estimators=700, num_leaves=63, max_depth=8, min_child_samples=40.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.863333

**Conclusion**
이전 실험 대비 MAE가 0.645836 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 51 — Groupkfold mid depth larger leaves

**Objective**
groupkfold mid depth larger leaves를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.839829

**Conclusion**
이전 실험 대비 MAE가 0.023504 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 52 — Groupkfold slower larger model

**Objective**
groupkfold slower larger model를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.02, n_estimators=1200, num_leaves=159, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.833989

**Conclusion**
이전 실험 대비 MAE 변화가 -0.005839로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 53 — Groupkfold remove raw layout_id while keeping layout metadata

**Objective**
groupkfold remove raw layout_id while keeping layout metadata를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: disabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.831477

**Conclusion**
이전 실험 대비 MAE 변화가 -0.002513로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 54 — Groupkfold tuned trees plus workload features

**Objective**
groupkfold tuned trees plus workload features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: workload. layout_id: enabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.835349

**Conclusion**
이전 실험 대비 MAE 변화가 0.003872로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 55 — Groupkfold tuned trees plus robot balance features

**Objective**
groupkfold tuned trees plus robot balance features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: robot balance. layout_id: enabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 robot balance 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.845183

**Conclusion**
이전 실험 대비 MAE 변화가 0.009834로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 56 — Groupkfold tuned trees plus environment and workload features

**Objective**
groupkfold tuned trees plus environment and workload features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: environment, workload. layout_id: enabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 environment, workload 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.838691

**Conclusion**
이전 실험 대비 MAE 변화가 -0.006492로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 57 — Groupkfold mid depth larger leaves alternate seed

**Objective**
groupkfold mid depth larger leaves alternate seed를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. hyperparameters: learning_rate=0.025, n_estimators=1000, num_leaves=127, max_depth=10, min_child_samples=25.

**Hypothesis**
가설은 검증 설정이나 트리 구조 조정이 현재 기준보다 더 나은 일반화를 만들 수 있다는 점이었다.

**Result**
OOF MAE 9.833067

**Conclusion**
이전 실험 대비 MAE 변화가 -0.005624로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 58 — Default groupkfold scenario log tuned v2

**Objective**
default groupkfold scenario log tuned v2 설정을 평가해 `group_kfold` 기준 일반화 성능 변화를 확인하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 9.217497

**Conclusion**
이전 실험 대비 MAE가 0.615570 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 59 — Clip negative predictions at zero for non-negative delay target

**Objective**
clip negative predictions at zero for non-negative delay target를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다는 점이었다.

**Result**
OOF MAE 9.217488

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000008로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 설정을 유지한 채 검증 방식이나 하이퍼파라미터를 한 단계씩 조정한다.

### Experiment 60 — Add capacity pressure ratios between inflow, pack stations, docks, and chargers

**Objective**
add capacity pressure ratios between inflow, pack stations, docks, and chargers를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.167677

**Conclusion**
이전 실험 대비 MAE가 0.049811 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 61 — Stack capacity pressure features with congestion interactions

**Objective**
stack capacity pressure features with congestion interactions를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity, congestion. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity, congestion 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.169492

**Conclusion**
이전 실험 대비 MAE 변화가 0.001816로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 62 — Combine capacity ratios with cyclical shift and weekday features

**Objective**
combine capacity ratios with cyclical shift and weekday features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity, temporal. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity, temporal 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.169118

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000374로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 63 — Combine capacity ratios with layout density and charger coverage features

**Objective**
combine capacity ratios with layout density and charger coverage features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity, layout interaction. layout_id: enabled. target: log1p. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity, layout interaction 피처가 운영 병목을 더 직접적으로 설명할 수 있다는 점이었다.

**Result**
OOF MAE 9.174876

**Conclusion**
이전 실험 대비 MAE 변화가 0.005758로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 64 — Apply mild sqrt target weighting on top of capacity ratios to reduce tail underprediction

**Objective**
apply mild sqrt target weighting on top of capacity ratios to reduce tail underprediction를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. weighting: sqrt x 0.05. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.148250

**Conclusion**
이전 실험 대비 MAE가 0.026626 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 65 — Swap sqrt weighting for smoother log target weighting on capacity features

**Objective**
swap sqrt weighting for smoother log target weighting on capacity features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.146279

**Conclusion**
이전 실험 대비 MAE 변화가 -0.001972로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 66 — Align objective with MAE while keeping best capacity and log-weight settings

**Objective**
align objective with MAE while keeping best capacity and log-weight settings를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.03, n_estimators=900, num_leaves=95, max_depth=10, min_child_samples=30.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.175939

**Conclusion**
이전 실험 대비 MAE가 0.029661 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 67 — Increase tree capacity for the best capacity plus log-weight configuration

**Objective**
increase tree capacity for the best capacity plus log-weight configuration를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.145386

**Conclusion**
이전 실험 대비 MAE가 0.030553 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 68 — Relax regularization and enlarge trees for the best weighted capacity setup

**Objective**
relax regularization and enlarge trees for the best weighted capacity setup를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: enabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.023, n_estimators=1300, num_leaves=159, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.148869

**Conclusion**
이전 실험 대비 MAE 변화가 0.003482로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 69 — Drop raw layout_id from the best weighted capacity setup to reduce layout-specific overfit

**Objective**
drop raw layout_id from the best weighted capacity setup to reduce layout-specific overfit를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.140415

**Conclusion**
이전 실험 대비 MAE 변화가 -0.008454로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 70 — Default groupkfold capacity weighted v1

**Objective**
default groupkfold capacity weighted v1 설정을 평가해 `group_kfold` 기준 일반화 성능 변화를 확인하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.140415

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000000로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 71 — Add bottleneck ratio features from blocked paths, truck wait, queues, and staffing pressure

**Objective**
add bottleneck ratio features from blocked paths, truck wait, queues, and staffing pressure를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.126865

**Conclusion**
이전 실험 대비 MAE가 0.013550 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 72 — Raise log target weighting strength on top of bottleneck features

**Objective**
raise log target weighting strength on top of bottleneck features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.25. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.131888

**Conclusion**
이전 실험 대비 MAE 변화가 0.005023로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 73 — Lower log target weighting after adding bottleneck features

**Objective**
lower log target weighting after adding bottleneck features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.15. hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.128075

**Conclusion**
이전 실험 대비 MAE 변화가 -0.003813로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
효과가 있었던 피처 축을 유지하고, 가중치나 트리 용량을 인접 값으로 미세 조정한다.

### Experiment 74 — Increase tree capacity for bottleneck ratio features

**Objective**
increase tree capacity for bottleneck ratio features를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. hyperparameters: learning_rate=0.022, n_estimators=1300, num_leaves=159, max_depth=12, min_child_samples=15.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.141117

**Conclusion**
이전 실험 대비 MAE가 0.013042 악화되어 해당 변화는 과적합 또는 일반화 저하로 해석된다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 75 — Blend weighted no-layoutid model with unweighted layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
blend weighted no-layoutid model with unweighted layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.25, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.120463

**Conclusion**
이전 실험 대비 MAE가 0.020654 개선되어 이 방향을 유지할 가치가 확인됐다.

**Next Step**
블렌드 비중이나 보조 모델 설정을 근처 값으로 조정해 추가 개선 가능성을 탐색한다.

### Experiment 76 — Reduce secondary layoutid blend weight to 0.15 | blend_secondary_model weight=0.15 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
reduce secondary layoutid blend weight to 0.15 | blend_secondary_model weight=0.15 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.15, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.122095

**Conclusion**
이전 실험 대비 MAE 변화가 0.001632로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 77 — Blend weighted bottleneck model with unweighted no-layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=False secondary_target_weight_mode=none

**Objective**
blend weighted bottleneck model with unweighted no-layoutid model | blend_secondary_model weight=0.25 secondary_use_layout_id=False secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.25, layout_id=False, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.121387

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000708로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
블렌드 비중이나 보조 모델 설정을 근처 값으로 조정해 추가 개선 가능성을 탐색한다.

### Experiment 78 — Blend bottleneck primary with simpler layoutid secondary baseline | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
blend bottleneck primary with simpler layoutid secondary baseline | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.25, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.120463

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000924로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
블렌드 비중이나 보조 모델 설정을 근처 값으로 조정해 추가 개선 가능성을 탐색한다.

### Experiment 78 — Blend bottleneck primary with simpler capacity-only layoutid secondary model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
blend bottleneck primary with simpler capacity-only layoutid secondary model | blend_secondary_model weight=0.25 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.25, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.122371

**Conclusion**
이전 실험 대비 MAE 변화가 0.001908로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 79 — Increase layoutid secondary blend weight to 0.35 | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
increase layoutid secondary blend weight to 0.35 | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.35, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.120201

**Conclusion**
이전 실험 대비 MAE 변화가 -0.002170로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
블렌드 비중이나 보조 모델 설정을 근처 값으로 조정해 추가 개선 가능성을 탐색한다.

### Experiment 80 — Use lightly weighted layoutid secondary model in the best bottleneck blend | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=log

**Objective**
use lightly weighted layoutid secondary model in the best bottleneck blend | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=log를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.35, layout_id=True, weighting=log). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.121146

**Conclusion**
이전 실험 대비 MAE 변화가 0.000945로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
이번 변화는 되돌리고, 신호가 있었던 이전 설정을 기준으로 다른 피처 축이나 블렌드 방향을 검증한다.

### Experiment 82 — - | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none

**Objective**
- | blend_secondary_model weight=0.35 secondary_use_layout_id=True secondary_target_weight_mode=none를 검증해 교차검증 점수를 개선하는 것이 목적이었다.

**Change**
validation: group k-fold on `scenario_id`. features: bottleneck, capacity. layout_id: disabled. target: log1p. weighting: log x 0.2. blend: secondary model (weight=0.35, layout_id=True, weighting=none). hyperparameters: learning_rate=0.025, n_estimators=1100, num_leaves=127, max_depth=11, min_child_samples=20.

**Hypothesis**
가설은 타깃 분포의 긴 꼬리를 완화해 MAE를 안정화할 수 있다; bottleneck, capacity 피처가 운영 병목을 더 직접적으로 설명할 수 있다; 고지연 샘플에 더 많은 학습 비중을 주면 tail 오차를 줄일 수 있다; 편향이 다른 보조 모델을 섞으면 fold 분산을 줄일 수 있다는 점이었다.

**Result**
OOF MAE 9.120201

**Conclusion**
이전 실험 대비 MAE 변화가 -0.000945로 작아, 영향은 제한적이지만 후속 미세 조정 여지는 남았다.

**Next Step**
블렌드 비중이나 보조 모델 설정을 근처 값으로 조정해 추가 개선 가능성을 탐색한다.
