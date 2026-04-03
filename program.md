# Program

This repository follows an autoresearch-style loop for a tabular regression competition.
The goal is to improve cross-validation performance on the Dacon Smart Warehouse Outbound Delay Prediction task while preserving a valid submission pipeline.

## Objective

- Improve the primary validation score reported by `train.py`
- Keep experiments reproducible and easy to compare
- Preserve the raw data under `open/`
- Always keep submission generation working

## Repository Roles

- `prepare.py`
  - Stable data preparation layer
  - Handles file checks, loading, layout merge, target detection, and CV setup
  - Treat this file as infrastructure
- `train.py`
  - Main experiment surface
  - This is the primary file to edit during iterative research
- `program.md`
  - Rules for autonomous experimentation

## Editable Files

- `train.py`
- `README.md` only if execution instructions or experiment reporting become inaccurate

## Do Not Modify

- `prepare.py` unless there is a real infrastructure bug
- Anything under `open/`
- Submission schema expected by `sample_submission.csv`
- Historical entries in `logs/results.csv`

## Data Constraints

- The problem is tabular regression
- Raw data is local-only and not committed to GitHub
- Expected local files:
  - `open/train.csv`
  - `open/test.csv`
  - `open/layout_info.csv`
  - `open/sample_submission.csv`
- Never overwrite or mutate raw CSV files

## Experiment Loop

1. Read `program.md`
2. Inspect the current `train.py`
3. Propose one clear improvement at a time
4. Run `python train.py`
5. Compare the new validation result against the best known result in `logs/results.csv`
6. Keep the change only if the result improves meaningfully and the submission pipeline still works
7. Revert or discard the change if the score gets worse or the run becomes invalid
8. Repeat

## Allowed Experiment Directions

- Model choice
- Feature selection inside `train.py`
- Encoding strategy
- Validation-safe feature engineering
- Hyperparameters
- Ensemble logic

## Forbidden Experiment Directions

- Using test labels or any leaked future information
- Editing raw files in `open/`
- Breaking `python prepare.py` or `python train.py`
- Changing the target column manually without evidence that auto-detection is wrong

## Keep Or Roll Back

- If CV performance improves and the run remains valid, keep the change
- If CV performance is flat but the implementation becomes significantly simpler or more stable, keeping it is allowed
- If CV performance degrades, roll back
- If submission generation fails, roll back
- If a change introduces leakage risk, roll back immediately

## 실험 기록 규칙

- 모든 실험은 `README.md`의 `실험기록` 표에 기록한다
- 제출 복사본은 `outputs/submissions_local/submission_xx.csv` 형식을 사용한다
- 성능 점수는 `logs/results.csv` 또는 `metrics.json` 기준으로 기록한다
- 각 실험의 개선 사항을 함께 기록한다

## Notes For Agents

- Prefer small, reviewable diffs
- Keep the experiment surface concentrated in `train.py`
- Do not expand scope unless the current bottleneck is clear
- Record every successful run through the existing logging path in `logs/results.csv`
