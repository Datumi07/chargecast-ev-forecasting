# ChargeCast: EV Charging Session Forecasting

**DATA2010 — Data Science Programming | VinUniversity**

ChargeCast forecasts the number of recorded charging sessions starting on the next day at an individual charging device (`com_id`) in the ChargePlace Scotland (CPS) network. We investigate whether a shared model trained across devices improves forecasting over device-specific models under simulated **30-day and 90-day history windows** before each forecast date. The dataset spans 395 calendar days, providing multiple forecast dates for chronological evaluation.

The current release contains a reproducible cleaning pipeline, a cleaned daily panel and a verified feature engineering pipeline for 30-day and 90-day histories. **Forecasting models, evaluation results and an interactive application have not yet been implemented.**

Feature engineering: run `python src/check_features.py`, then `python src/build_features.py`.
See [feature definitions and model handoff](docs/feature_engineering.md) for predictor allowlists, missing-history handling, forecast timing and eligibility rules.

## Research questions

1. Can forecasting models outperform simple historical baselines?
2. Does shared learning outperform device-specific learning under identical 30-day and 90-day history windows?
3. How do errors and underprediction change on high-usage days?

The target measures **recorded sessions**, not all latent charging demand, charger availability or waiting time. A device is not necessarily a separate physical site. Site-level aggregation requires a verified device-to-site mapping.

## Dataset and current release

**Source:** [ChargePlace Scotland — Monthly Charge Point Performance](https://chargeplacescotland.org/monthly-charge-point-performance/)

| Item | Clean v1 |
|---|---:|
| Supplied reporting period | September 2025 – September 2026 |
| Devices present in all 13 monthly device reports | 1,756 |
| Devices with matching sessions, included in the daily panel | 1,678 |
| Retained session records in this cohort | 788,796 |
| Daily panel rows, including unavailable labels | 662,810 |
| Usable target rows under documented coverage assumptions | 550,274 |
| Zero targets inferred under those assumptions | 247,818 |

The complete cleaned session table also retains devices outside the main cohort. The 78 registry devices with no matching sessions remain in the audit rather than receiving fabricated zero histories.

**Recommended initial experiment:** February–September 2026. Earlier months remain available for EDA and sensitivity analysis because their detailed session counts do not reconcile with published monthly totals.

### Essential limitations

- Dates are **source-reported dates**, not verified Scotland local dates. No unverified timezone shift has been applied.
- A missing record becomes zero only under the documented within-month coverage assumption; this does not prove continuous logging. Entire device-months without sessions are treated as unknown.
- August 1–2, 2026 is masked. Month boundaries and DST-transition dates are also masked conservatively; this does not resolve every possible timezone-related day shift.
- Selecting devices present throughout the period is a retrospective cohort choice and introduces survivorship bias. Results do not represent all CPS devices.
- Session deduplication uses a documented composite rule because a reliable unique session ID is unavailable.
- Data ends in September 2026. Any initial demo should use historical forecast origins, not claim live next-day predictions.

See [the cleaning methodology](docs/data_cleaning.md), [daily data dictionary](docs/data_dictionary.csv) and [session/registry dictionary](docs/session_registry_dictionary.csv).

## Repository layout

| Path | Purpose |
|---|---|
| `config/input_manifest.json` | Exact filenames, reporting months and source types for 26 input workbooks |
| `data/raw/` | Original XLSX inputs, excluded from Git |
| `data/processed/daily_panel.csv.gz` | Main continuous device-day panel, included in Git |
| `data/processed/registry_monthly.csv` | Monthly metadata for the stable cohort, included in Git |
| `data/processed/sessions_clean.csv.gz` | Full session-level output, bundled in delivery ZIP but excluded from Git |
| `src/prepare_data.py` | Rebuild cleaning outputs from original workbooks |
| `src/load_example.py` | Load data and demonstrate time-safe lag creation |
| `docs/` | Cleaning decisions and column dictionaries |
| `reports/data_quality/` | Audit counts, excluded IDs, removed duplicates and input fingerprints |

## Quick start

Run commands from the repository root with Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python src/load_example.py
```

The example reads the supplied panel, creates lag-1 and lag-7 using the continuous calendar, and reports usable rows for each split. It does **not** train a model.

```python
import pandas as pd

panel = pd.read_csv(
    "data/processed/daily_panel.csv.gz",
    dtype={"com_id": "string", "sessions_target": "Int64"},
    parse_dates=["reported_date"],
).sort_values(["com_id", "reported_date"])

# Create historical features BEFORE removing unavailable target dates.
panel["lag_7"] = panel.groupby("com_id")["sessions_target"].shift(7)
usable = panel["recommended_period"] & panel["target_valid"]
```

### Rebuild from raw workbooks

Place the files listed in `config/input_manifest.json` directly under `data/raw/`, or provide the path to the original `ChargeCast/Data/Raw` folder:

```bash
python src/prepare_data.py
# Alternative source directory:
python src/prepare_data.py --raw /path/to/ChargeCast/Data/Raw
```

This regenerates processed data and audit tables without editing the raw sources. The original workbooks are not included in this repository delivery. Their SHA256 fingerprints are recorded in `reports/data_quality/ingestion_audit.csv`.

## Cleaning policy

- Standardize identifiers and schemas; exclude rows without a usable device ID or start time.
- Remove duplicate copies using the declared composite key and retain an audit of removed rows.
- Preserve zero-energy sessions for the recorded-session target. Flag invalid energy/duration values without automatically deleting the session.
- Intersect monthly device registries, then aggregate sessions by device and reported date.
- Retain a continuous calendar. Unreliable labels remain missing rather than becoming zero or being interpolated.
- Keep monthly reliability and ticket information out of the initial predictive feature set. Information published after a forecast origin must not be used at that origin.

The pipeline asserts unique daily keys, nonnegative available targets, masked unavailable targets, continuous calendar size and reconciliation between session counts and the daily panel. These checks establish internal consistency, not source completeness.

## Modelling and evaluation plan

| Partition | Recommended period |
|---|---|
| Train | February–June 2026 |
| Validation | July–August 2026 |
| Test | September 2026 |

- Start with same-weekday-last-week and historical-average baselines.
- Candidate models: Poisson regression and LightGBM; model selection remains pending.
- Compare shared and local models using identical forecast dates, devices and 30/90-day calendar windows. Define the rolling retraining schedule before experimentation.
- Use chronological evaluation; never random-split device-day rows.
- Compute lag/rolling statistics only from prior dates. Keep missing history visible and report coverage requirements.
- Report MAE, RMSE and high-usage underprediction. Define high-usage thresholds using training history, not test outcomes.
- Evaluate competing models on the same eligible rows; report the number of devices/days retained.

**Do not automatically use every column as a predictor.** `observed_session_rows` reveals the same-day outcome. `month_session_rows` uses the entire reporting month. Coverage flags and target-validity fields are retrospective audit controls, not forecast-time features. `sessions_target` is the target.

## Planned pilot and optional extensions

The core pilot will let users select a device and inspect historical usage and a next-day forecast. Two-day forecasting needs separate horizon evaluation. A chatbot may later query model outputs; RAG is optional for retrieving station documentation. Daily session counts alone cannot support guaranteed recommendations about available connectors or waiting times.

Findings may inform future methodological work in Vietnam, but this dataset does not establish predictive performance in Vietnam.

## Team members

- Dang Tuan Minh
- Nguyen Quy Tu
- Nguyen Dinh Nam Khanh
- Ngo Van Thang
- Bui Anh Minh

Individual responsibilities will be finalised during project planning.

## Project status and academic context

- Research scope defined.
- Clean v1 data, reproducible preparation code and quality documentation supplied.
- Feature engineering implemented with calendar lags, historical statistics, coverage reports and leakage checks.
- Model comparisons and pilot implementation are next.

This repository is maintained for DATA2010 at VinUniversity. Cite CPS as the original data source and describe the cohort, inferred zeros, exclusions and timestamp limitations in the final report. Public access alone is not a substitute for checking source reuse terms before wider redistribution.
