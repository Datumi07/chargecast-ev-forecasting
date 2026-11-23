# ChargeCast (DATA2010) — Team B: Shared LightGBM Forecasting

Unified September 2026 test forecasts, reproducible evaluation script, and finalized benchmark results for **Team B (Shared LightGBM)** on the ChargeCast EV charging demand forecasting task.

## 1. Final Model Selection

Team B prioritizes **overall RMSE** and **high-usage ("busy day") accuracy** because peak demand days carry greater operational value than low-demand days. All models are evaluated under one unified naming scheme (`model`, `training_mode`, `window_days`).

| Role | Model | Training Mode | Window | Key Result |
| --- | --- | --- | :---: | --- |
| **Primary Model (Team B)** | `lightgbm_l2` | `fixed_train_feb_aug` | 30d | Best overall RMSE (`1.1332`), best Busy RMSE (`2.5382`), best Busy MAE (`2.0877`), least under-prediction on busy days (`Busy Bias = -1.6182`), and **100.00% coverage** (`39,564 / 39,564`). |
| **Daily-Refit Benchmark** | `lightgbm_poisson` | `rolling_daily_refit` | 90d | Best rolling daily-refit model on RMSE (`1.1383` on 39,207 matched rows; `1.1356` on 39,466 90d rows), `99.75%` coverage. |
| **Best Overall MAE Reference** | `lightgbm_l1` | `rolling_daily_refit` | 90d | Lowest overall MAE (`0.7291`), but weak on busy days (`Busy RMSE = 2.8111`, `Busy Bias = -2.0166`, `86.4%` under-predicted). |
| **Supplementary Reference** | `lightgbm_l2` | `rolling_daily_refit` | 30d / 90d | Post-inspection L2 daily-refit experiment (`reg_lambda=0` for 30d, `reg_lambda=5` for 90d); did not outperform `lightgbm_poisson (90d)` or `lightgbm_l2 (fixed_train_feb_aug, 30d)` on RMSE. |
| **Baselines** | `baseline_mean7`, `baseline_lag7` | `history_only` | 7d | Trailing 7-day mean and same-weekday lag-7 baselines evaluated on the 36,483-row matched intersection. |

## 2. Evaluation Protocol (`matched_all_models`)

- **Test Period:** September 2–29, 2026 (`28` days × `1,413` devices = `39,564` eligible device-day rows with valid labels).
- **Fair Row Alignment:** Each `(model, training_mode, window_days)` configuration is checked for unique `(com_id, reported_date)` keys and identical `y_true` labels before taking the exact matched intersection (`matched_all_models`). Coverage is reported separately.
- **Fixed High-Usage Threshold:** Defined as **`y_true >= 4.0`** sessions/device-day, computed once as the 90th percentile (`P90 = 4.0`) of all `209,496` valid labels in the Feb–Jun 2026 training split (pooled across devices) and held fixed across all models.
- **Methodological Transparency Note:** Final candidate selection after comparing September test performance across training modes is a **supplementary post-hoc selection**. September 2026 should not be described as a completely unseen test set for the final selection or for the supplementary L2 daily-refit evaluation.

## 3. Evaluation Results (Fixed P90 >= 4.0)

### 3.1. Matched LightGBM Models (30d & 90d — `39,207` rows, `3,541` busy days)

| Model | Training Mode | Window | Coverage | MAE | RMSE | Busy MAE | Busy RMSE | Busy Bias | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lightgbm_l2` | `fixed_train_feb_aug` | 30d | 100.00% | 0.7806 | 1.1332 | 2.0877 | 2.5382 | -1.6182 | Primary |
| `lightgbm_l2` | `fixed_train_feb_aug` | 90d | 100.00% | 0.7800 | 1.1348 | 2.1085 | 2.5581 | -1.6567 | Primary |
| `lightgbm_l2` | `fixed_train_feb_jun` | 30d | 100.00% | 0.7820 | 1.1377 | 2.1381 | 2.5935 | -1.7522 | Primary |
| `lightgbm_poisson` | `rolling_daily_refit` | 90d | 99.75% | 0.7764 | 1.1383 | 2.1094 | 2.5730 | -1.6606 | Primary |
| `lightgbm_l2` | `rolling_daily_refit` | 90d | 99.75% | 0.7786 | 1.1389 | 2.0997 | 2.5636 | -1.6449 | Supplementary |
| `lightgbm_l2` | `fixed_train_feb_jun` | 90d | 100.00% | 0.7818 | 1.1397 | 2.1550 | 2.6106 | -1.7710 | Primary |
| `lightgbm_l2` | `rolling_daily_refit` | 30d | 99.10% | 0.7902 | 1.1614 | 2.1549 | 2.6267 | -1.7106 | Supplementary |
| `lightgbm_poisson` | `rolling_daily_refit` | 30d | 99.10% | 0.7870 | 1.1622 | 2.1838 | 2.6581 | -1.7474 | Primary |
| `lightgbm_l1` | `rolling_daily_refit` | 90d | 99.75% | 0.7291 | 1.1900 | 2.3297 | 2.8111 | -2.0166 | Primary |
| `lightgbm_l1` | `rolling_daily_refit` | 30d | 99.10% | 0.7441 | 1.2176 | 2.3770 | 2.8960 | -2.0513 | Primary |

### 3.2. Matched 90-Day LightGBM Models (`39,466` rows, `3,543` busy days)

| Model | Training Mode | Window | Coverage | MAE | RMSE | Busy MAE | Busy RMSE | Busy Bias | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lightgbm_l2` | `fixed_train_feb_aug` | 90d | 100.00% | 0.7772 | 1.1321 | 2.1093 | 2.5589 | -1.6578 | Primary |
| `lightgbm_poisson` | `rolling_daily_refit` | 90d | 99.75% | 0.7736 | 1.1356 | 2.1102 | 2.5738 | -1.6617 | Primary |
| `lightgbm_l2` | `rolling_daily_refit` | 90d | 99.75% | 0.7758 | 1.1362 | 2.1005 | 2.5644 | -1.6459 | Supplementary |
| `lightgbm_l2` | `fixed_train_feb_jun` | 90d | 100.00% | 0.7790 | 1.1370 | 2.1558 | 2.6114 | -1.7720 | Primary |
| `lightgbm_l1` | `rolling_daily_refit` | 90d | 99.75% | 0.7257 | 1.1872 | 2.3305 | 2.8118 | -2.0175 | Primary |

### 3.3. Matched All Models Including Baselines (`36,483` rows, `3,281` busy days)

| Model | Training Mode | Window | Coverage | MAE | RMSE | Busy MAE | Busy RMSE | Busy Bias | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lightgbm_l2` | `fixed_train_feb_aug` | 30d | 100.00% | 0.7785 | 1.1304 | 2.0869 | 2.5328 | -1.6164 | Primary |
| `lightgbm_l2` | `fixed_train_feb_aug` | 90d | 100.00% | 0.7780 | 1.1318 | 2.1061 | 2.5517 | -1.6548 | Primary |
| `lightgbm_l2` | `fixed_train_feb_jun` | 30d | 100.00% | 0.7797 | 1.1344 | 2.1346 | 2.5863 | -1.7506 | Primary |
| `lightgbm_l2` | `rolling_daily_refit` | 90d | 99.75% | 0.7769 | 1.1358 | 2.0979 | 2.5566 | -1.6413 | Supplementary |
| `lightgbm_poisson` | `rolling_daily_refit` | 90d | 99.75% | 0.7745 | 1.1358 | 2.1098 | 2.5684 | -1.6615 | Primary |
| `lightgbm_l2` | `fixed_train_feb_jun` | 90d | 100.00% | 0.7798 | 1.1365 | 2.1519 | 2.6034 | -1.7694 | Primary |
| `lightgbm_l2` | `rolling_daily_refit` | 30d | 99.10% | 0.7882 | 1.1586 | 2.1536 | 2.6212 | -1.7101 | Supplementary |
| `lightgbm_poisson` | `rolling_daily_refit` | 30d | 99.10% | 0.7848 | 1.1600 | 2.1858 | 2.6572 | -1.7479 | Primary |
| `lightgbm_l1` | `rolling_daily_refit` | 90d | 99.75% | 0.7276 | 1.1869 | 2.3289 | 2.8044 | -2.0173 | Primary |
| `baseline_mean7` | `history_only` | 7d | 99.66% | 0.8012 | 1.2028 | 2.1576 | 2.6427 | -1.6439 | Baseline |
| `lightgbm_l1` | `rolling_daily_refit` | 30d | 99.10% | 0.7419 | 1.2146 | 2.3781 | 2.8923 | -2.0528 | Primary |
| `baseline_lag7` | `history_only` | 7d | 92.21% | 0.9382 | 1.5531 | 2.5520 | 3.1956 | -1.5315 | Baseline |

## 4. Folder Structure

```text
team_b_lightgbm/
├── README.md                         # Overview, model selection, results, and reproduction
├── scripts/
│   └── evaluate.py                   # Reproduces coverage.csv and model_summary.csv
├── data/
│   └── forecasts_september.csv.gz    # Unified Team B September test forecasts (~3.4 MB)
└── results/
    ├── coverage.csv                  # Per-configuration coverage & date span
    └── model_summary.csv             # Consolidated metrics across all 3 matched cuts
```

## 5. Quick Reproduction

Requires Python 3.9+ with `pandas` and `numpy`. From inside `team_b_lightgbm/`:

```bash
python scripts/evaluate.py
```
