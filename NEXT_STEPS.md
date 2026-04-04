# Next Steps

현재 기준 최선은 `adaptive_gkf_01`의 `OOF MAE 9.107482`이다.
후속 실험은 이 기준점을 유지하면서, 신호가 있었던 축만 좁게 다시 검증하는 방향이 가장 효율적이다.

## 1. adaptive_gkf_01 + temporal

- 기준 설정은 `adaptive_gkf_01`을 그대로 사용한다.
- 변경점은 `add_temporal_features=true`만 추가한다.
- 목적은 `delay_risk` 기반 성능을 유지하면서 시간대/요일 패턴이 일반화 성능에 실제로 기여하는지 확인하는 것이다.
- `delay_risk + congestion` 단독 추가는 미세 악화였지만, fast 탐색에서는 temporal 축이 간헐적으로 살아남았기 때문에 가장 먼저 확인할 가치가 있다.

## 2. adaptive_gkf_01 + temporal + very mild congestion

- 기준 설정은 역시 `adaptive_gkf_01`이다.
- `add_temporal_features=true`, `add_congestion_features=true`를 함께 켠다.
- 대신 모델 복잡도는 약간 낮춘다.
- 권장 시작값:
  - `num_leaves`: `127 -> 111`
  - `min_child_samples`: `20 -> 24` 또는 `26`
  - `target_weight_strength`: `0.36 -> 0.30` 또는 `0.32`
- 목적은 congestion 계열 feature가 직접 추가될 때 생기는 과적합을, temporal signal과 약한 규제로 상쇄할 수 있는지 보는 것이다.

## 3. adaptive_gkf_01 주변 미세 튜닝

- feature set은 `adaptive_gkf_01`과 동일하게 유지한다.
- 튜닝 대상은 아래 두 값으로 제한한다.
  - `secondary_weight`
  - `target_weight_strength`
- 권장 탐색 범위:
  - `secondary_weight`: `0.20`, `0.25`, `0.30`
  - `target_weight_strength`: `0.30`, `0.34`, `0.36`, `0.40`
- 목적은 feature를 늘리지 않고 현재 챔피언 설정의 bias-variance 균형을 더 정교하게 맞추는 것이다.

## Validation Rule

- 탐색 단계에서는 `2-fold` 또는 `3-fold group_kfold`를 사용해 속도를 확보한다.
- 승격 후보는 반드시 `5-fold group_kfold`에서 `adaptive_gkf_01`과 직접 비교한다.
- fast 탐색에서 좋아 보인 설정이 full 검증에서 유지되지 않은 경우가 이미 있었기 때문에, 최종 채택은 `5-fold` 결과만 기준으로 판단한다.
