# Experiment Portfolio

Smart Warehouse Outbound Delay Prediction  
LightGBM 기반 실험 기록

---

## 🧪 Experiment 01 — Baseline (Scenario GroupKFold)

**Objective**  
Baseline 모델 성능 확인

**Key Changes**  
- GroupKFold (scenario_id)
- learning_rate=0.03
- num_leaves=63
- max_depth=7

**Result**  
OOF MAE: **9.8440**

**Insight**  
Baseline 성능 확보

---

## 🧪 Experiment 04 — KFold Validation

**Objective**  
검증 방식 변경 효과 확인

**Key Changes**  
- GroupKFold → KFold

**Result**  
OOF MAE: **8.4180** ↓

**Insight**  
KFold가 더 안정적인 성능 제공

---

## 🧪 Experiment 05 — Deeper Trees

**Objective**  
모델 표현력 증가

**Key Changes**  
- num_leaves: 127
- max_depth: 10

**Result**  
OOF MAE: **7.9098** ↓

**Insight**  
Tree complexity 증가가 효과적

---

## 🧪 Experiment 21 — Large Tree Model

**Key Changes**
- num_leaves: 191
- depth: 12
- learning_rate: 0.03

**Result**
OOF MAE: **7.2977** ↓

---

## 🧪 Experiment 25 — Best Model

**Key Changes**
- num_leaves: 191
- depth: 12
- learning_rate: 0.025
- strong regularization tuning

**Result**
🏆 **Best OOF MAE: 7.2127**
