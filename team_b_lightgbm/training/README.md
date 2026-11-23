# Team B — Model Training & Leakage Verification Code

This folder contains the full training code for both Team B modeling regimes (`fixed_train` and `rolling_daily_refit`), organized for straightforward code review and data-leakage verification.

## 1. Two Training Modes & How `30d / 90d` Windows Work

### Mode A: `fixed_train/` (Primary Team B Model: `lightgbm_l2`, `fixed_train_feb_aug`, 30d)
- **Role of `30d / 90d`:** Lookback feature windows from the repository's canonical feature pipeline (`data/processed/features/features_30d.csv.gz`, `features_90d.csv.gz`, and `predictors_{30|90}d.json`).
- **Training Splits:**
  1. `fixed_train_feb_jun`: Trained once on Feb–Jun 2026 (`train` split), with Jul–Aug 2026 (`validation` split) used strictly for hyperparameter tuning and early stopping (`tune_lgbm.py` -> `best_lgbm_config.json`).
  2. `fixed_train_feb_aug`: Trained once on Feb–Aug 2026 (`train + validation`) using the exact locked hyperparameters and `best_iteration` from Validation, without ever touching September test labels.
- **Anti-Leakage Safeguards:**
  - `evaluation.py` enforces `FORBIDDEN_PREDICTORS` (blocking `sessions_target`, `split`, `target_valid`, `label_eligible`, etc.).
  - `com_id` categorical levels are fit strictly on the training slice (`train_lgbm_shared.py`).
  - Missing lag/rolling values remain `NaN` (never filled with `0`).
  - High-usage threshold (`P90 = 4.0`) is computed strictly on the Feb–Jun training split.
- **What the model sees when predicting day `D` in September:** Fixed tree weights learned up to end of August (or June), calendar features of day `D`, `com_id`, and past-only lag/rolling features (`<= D - 1`) produced by the repo's `build_features.py` (one-step-ahead forecasting).

### Mode B: `rolling_daily_refit/` (Daily-Refit Benchmark: `lightgbm_poisson` 90d & `lightgbm_l1` 90d)
- **Role of `30d / 90d`:** Both the **rolling training window** and the **feature lookback window**. For each forecast day `D`, the code slices strictly `h = wide.loc[D - W : D - 1]` from `data/processed/daily_panel.csv.gz`, computes features strictly inside `[D - W, D - 1]`, and fits a fresh LightGBM model for day `D`.
- **Eligibility & Missingness:** Training rows require a valid label and at least `7` prior valid observations; forecast devices require at least `7` valid history days inside `[D - W, D - 1]`. Missing values stay `NaN`.
- **Anti-Leakage Safeguards:**
  - `make_features()` in `experiment.py` receives only the history matrix `[D - W, D - 1]` with `NaN` appended for the forecast origin `D`, making same-day or future target leakage structurally impossible.
  - Run `python check_features_v2.py` to execute automated unit tests verifying **future invariance** (mutating future values does not alter past features), **strict window isolation**, and **missing != zero**.
  - Run `python verify_test.py` after `run_test.py` to verify target alignment, eligibility counts, and saved-model reload consistency.
- **What the model sees when predicting day `D` in September:** Calendar features of day `D`, `com_id`, and observed history strictly within `[D - W, D - 1]` (one-step-ahead daily refit; observed September days prior to `D` enter subsequent windows `[D - W, D - 1]`).

## 2. Quick Leakage Unit Test (Runs in < 2 seconds)

```bash
cd team_b_lightgbm/training/rolling_daily_refit
python check_features_v2.py
```
