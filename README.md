# ChargeCast — Next-day EV charging session forecasts

DATA2010 · VinUniversity

Predict recorded session starts for each ChargePlace Scotland charging device on the next day. The cleaned panel contains 1,678 devices across September 2025–September 2026. Main experiments evaluate July–August validation and September test forecasts in 2026.

## Start here

| Task | Entry point |
|---|---|
| Inspect cleaning and EDA | [Notebook](notebooks/01_data_quality_and_eda.ipynb) |
| Understand coverage and inferred zeros | [Cleaning methodology](docs/data_cleaning.md) |
| Generate canonical features | [Feature definitions](docs/feature_engineering.md) |
| Review primary shared/local Poisson | [Poisson experiment](reports/poisson/README.md) |
| Review alternative four-feature Poisson | [Alternative results](reports/group_a/results.txt) |
| Review LightGBM | [Team B guide](team_b_lightgbm/README.md) |
| Review Ridge | [Ridge guide](reports/ridge/README.md) |
| Compare three models and two baselines | [Matched scores](reports/comparison/metrics.csv) |

## Run from the repository root

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python src/check_features.py
python src/build_features.py
python src/run_ridge.py
python src/compare_models.py
```

The comparison reads the committed forecasts. Refit the primary Poisson experiment with `python src/poisson_forecast.py`. See each experiment's guide for its training protocol.

For fixed LightGBM, generate canonical features first, then run:

```bash
python team_b_lightgbm/training/fixed_train/evaluate_final_test.py
```

This script uses a locally tuned configuration in `reports/modeling/` when present, otherwise the bundled configuration. The full LightGBM refit has not been independently reproduced in the integration review. For the published forecast scores, run `python scripts/evaluate.py` from `team_b_lightgbm/`.

## Research questions and current findings

1. Do models improve on historical baselines? Compare three model families with lag-7 and seven-day-average forecasts on identical September device-day keys.
2. Does pooling devices help with 30/90 days of training history? The primary Poisson experiment finds lower September errors for local models at both lengths. Its shared specification retains a device-specific activity offset.
3. How do errors change on busy days? Use at least four recorded sessions, the training-period threshold. Report busy-day RMSE, signed bias and underprediction rate.

### Matched September results

All five methods are scored on the same 36483 device-days.

| Model | MAE | RMSE | Busy RMSE | Busy bias | Busy underprediction |
|---|---:|---:|---:|---:|---:|
| poisson_local90 | 0.8049 | 1.1779 | 2.6617 | -1.7008 | 81.8% |
| ridge | 0.7933 | 1.1582 | 2.5916 | -1.7341 | 84.3% |
| lightgbm_fixed | 0.7785 | 1.1304 | 2.5328 | -1.6164 | 82.7% |
| lag7 | 0.9382 | 1.5531 | 3.1956 | -1.5315 | 67.6% |
| mean7 | 0.8012 | 1.2028 | 2.6427 | -1.6439 | 80.8% |

Bias is prediction minus actual. Conditioning on high realized counts selects upward surprises. These busy-day results describe conditional errors. Compare forecast coverage separately in [coverage.csv](reports/comparison/coverage.csv).

## Experiment definitions

| Experiment | Model fit | Historical features |
|---|---|---|
| Primary Poisson | Daily refit on previous 30/90 calendar days | Recent device activity offset and weekday effect, smoothed using validation |
| Alternative Poisson | Daily refit on previous 30/90 calendar days | Four predictors; short local histories produce extreme validation forecasts |
| Team B primary LightGBM | One fit on February–August | Canonical 30-day feature lookback; L2 objective |
| Ridge | One fit on previous 90 days before each evaluation period | Weekday indicators, lags 1/7/14/28 and trailing means 7/28/90; alpha 10 |

Ridge and Poisson settings were selected using validation. The team had already inspected September during development. Team B's final configuration choice also considered September scores. Treat the table as a retrospective comparison. Different training schedules and feature sets make this a comparison of complete forecasting pipelines.

## Data and layout

- `data/raw/` describes original workbooks, excluded from Git.
- `data/processed/` contains the daily panel and registry. Generated feature tables remain local.
- `src/` contains cleaning, feature creation, model and comparison entry points.
- `notebooks/` contains EDA.
- `reports/` contains audits and model outputs.
- `team_b_lightgbm/` retains the submitted LightGBM code, forecasts and evaluation.

Rebuild the panel with `python src/prepare_data.py --raw /path/to/workbooks`. Exact input filenames are in `config/input_manifest.json`. Download the original workbooks from [ChargePlace Scotland](https://chargeplacescotland.org/monthly-charge-point-performance/). See the [daily dictionary](docs/data_dictionary.csv) and [session/registry dictionary](docs/session_registry_dictionary.csv).

### Essential limitations

- Dates are **source-reported dates**, not verified Scotland local dates. No unverified timezone shift has been applied.
- A missing record becomes zero only under the documented within-month coverage assumption; this does not prove continuous logging. Entire device-months without sessions are treated as unknown.
- August 1–2, 2026 is masked. Month boundaries and DST-transition dates are also masked conservatively; this does not resolve every possible timezone-related day shift.
- Selecting devices present throughout the period is a retrospective cohort choice and introduces survivorship bias. Results do not represent all CPS devices.
- Session deduplication uses a documented composite rule because a reliable unique session ID is unavailable.
- Data ends in September 2026. Any initial demo should use historical forecast origins, not claim live next-day predictions.

See [the cleaning methodology](docs/data_cleaning.md), [daily data dictionary](docs/data_dictionary.csv) and [session/registry dictionary](docs/session_registry_dictionary.csv).


## Remaining deliverables

- Independently reproduce complete model training from the cleaned panel.
- Complete the historical device-selection pilot and final report.
- Document permitted AI assistance according to the course requirements.

The historical panel supports recorded-session forecasting. Transfer to Vietnam requires local data and validation.

## Team

Dang Tuan Minh · Nguyen Quy Tu · Nguyen Dinh Nam Khanh · Ngo Van Thang · Bui Anh Minh
