# Feature engineering

Convert the cleaned device-day panel into numeric predictors for next-day
recorded session counts. The two configurations use the same devices, dates,
calendar features and lags; their longest rolling window and historical coverage
denominator are 30 and 90 days respectively. Each output has 26 predictors.

Run from the repository root with the project requirements installed:

```bash
python src/check_features.py
python src/build_features.py
```

Outputs: `data/processed/features/features_30d.csv.gz`, `features_90d.csv.gz`,
one `predictors_*d.json` allowlist per window, and
`reports/feature_engineering/coverage.csv`. Feature tables are generated locally
and excluded from Git. Input/output paths can be overridden with `--input`,
`--output`, and `--report`.

## Forecast convention

Each row predicts `sessions_target` on `reported_date`, using history ending on
the preceding calendar day. This assumes that the previous day's label is
available when forecasting. It supports historical rolling one-day evaluation,
not a fixed-origin forecast of an entire validation/test month. Actual source
publication delays are unknown. The source dates and retrospective coverage
assumptions retain the limitations described in `data_cleaning.md`.

Both outputs retain every input device-day. Sorting and validation precede
feature creation; duplicate keys, missing calendar days, invalid counts and
inconsistent label masks raise errors. Missing labels stay missing. No
interpolation, zero filling, scaling, encoding or fitting uses future rows.

## Predictors

| Columns | Definition |
|---|---|
| `weekday`, `is_weekend`, `month` | Known prediction-date calendar fields; Monday = 0 |
| `weekday_sin`, `weekday_cos` | Weekly cyclic encoding, period 7 |
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | Target exactly that many calendar days earlier, within device |
| `rolling_mean_W`, `rolling_std_W`, `rolling_min_W`, `rolling_max_W` | Previous W calendar days; W = 7, 14 and experiment window (30 or 90) |
| `rolling_count_W` | Number of nonmissing labels in that same window; zero for no observed history |
| `zero_fraction` | Fraction of available historical labels equal to zero in the experiment window |
| `history_observed_fraction` | Available historical label count divided by 30 or 90 |

Statistics ignore missing labels but never compress the calendar. Means/min/max
need one available label; sample standard deviation (`ddof=1`) needs two.
Empty statistics remain missing. The zero fraction is missing when no historical
label is available. Inferred historical zeros remain zeros under Clean v1 policy.
All predictors access at most the experiment's history window.

Calendar representations are alternatives for different model families; model
owners should choose appropriate subsets. Device ID remains a key; encoding it
for a shared model must be fitted using training data only. Monthly registry,
reliability, ticket, same-day session and coverage audit columns are excluded.
Historical missingness features reflect retrospective label availability; they
do not establish live logging coverage.

## Model handoff

Use only the JSON predictor allowlist for X and `sessions_target` for y.
`history_window_available` requires the full calendar window to exist in the
input, even when some labels in it are missing. `label_eligible` combines that
condition with `recommended_period` and `target_valid`. `complete_features`
indicates whether every listed predictor is present; it is an evaluation
control, never a predictor. Missing-aware models may use `label_eligible`;
complete-case models additionally require `complete_features`. Compare models
on the intersection of their eligible keys and report exclusions.

Example for the 30-day experiment (use `90d` for the other configuration):

```python
import json
from pathlib import Path
import pandas as pd

folder = Path("data/processed/features")
features = pd.read_csv(folder / "features_30d.csv.gz",
                       dtype={"com_id": "string"}, parse_dates=["reported_date"])
columns = json.loads((folder / "predictors_30d.json").read_text())
# Complete cases suit models that cannot accept NaN, such as PoissonRegressor.
eligible = features["label_eligible"] & features["complete_features"]
train = features.loc[eligible & features["split"].eq("train")]
X_train, y_train = train[columns], train["sessions_target"]
```

For a missing-aware model such as LightGBM, the complete-case condition can be
omitted. Do not fill missing session history with zero. This example selects the
training partition only; rolling retraining still requires the date restrictions
below. There is no model training implementation in the current repository, so
compatibility is verified at the numeric feature/label interface.

Preserve the supplied chronological splits: February–June training,
July–August validation, September test. Earlier observations may supply history
within the window. Previous validation/test outcomes may supply later features
only under the stated daily rolling forecast convention. Fit preprocessing on
training observations only. Feature lookback bounds do not enforce model
training windows: the model owner must independently restrict retraining to the
same 30/90 calendar days and define retraining dates. Do not treat the feature
tables as permission to train on all earlier data.

Coverage reports include usable labels, device counts, complete-case rows,
missing-label fractions, zero-label fractions and mean historical coverage.
These summarize retrospective data quality, not model performance.

`src/check_features.py` checks exact calendar lags, device isolation, missing
labels, numeric window statistics, first-window eligibility, future-label
perturbations, equivalence to explicitly truncated history, and invalid input.
It also compares every lag and rolling statistic against independent calendar
slices, checks entirely missing history, and verifies stable output after input
row shuffling. The CLI fixes gzip timestamps; reruns with identical input and
the same Python/pandas/NumPy versions should produce identical output hashes.
Dependencies specify version ranges, so byte-identical results across different
library versions or operating systems are not guaranteed.

## Verified delivery

The supplied Clean v1 input produces 662,810 rows and 35 columns per window:
26 predictors, two keys, four label/evaluation fields and three eligibility
controls. Eligible labels are 209,496 train, 81,912 validation and 39,564 test
for either window. Requiring every predictor reduces these to 150,319, 55,125
and 28,602 respectively; compare models on matching keys.

Validation used Python 3.14.5, pandas 2.3.3 and NumPy 2.5.5. The runnable checks,
full output schema and coverage checks, cross-window shared-feature comparison,
and rebuild SHA256 comparison passed. Both feature tables, both predictor
allowlists and the coverage report were byte-identical after rebuilding.
