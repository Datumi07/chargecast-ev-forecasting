# Group A Poisson regression

The 90-day device-specific model had the lowest September MAE and RMSE. Sharing the weekday effects across devices increased forecast coverage, but on matching rows it slightly increased both errors at both history lengths.

## Reproduce

Run from the repository root with Python 3.10 or later and the supplied Clean v1 panel. The model and tests use only NumPy and pandas beyond the Python standard library. The recorded run used Python 3.13.7, NumPy 2.3.2 and pandas 2.3.2.

```sh
python -m pip install "numpy>=1.26" "pandas>=2.2,<3"
python -m unittest discover -s tests
python src/poisson_forecast.py
```

For a validation-only cross-check, run `python src/poisson_forecast.py --validation-only`. It writes to `reports/poisson_validation/`. Use `--output DIRECTORY` to select another destination. Read the CSVs with `dtype={"com_id": "string"}` to preserve identifiers.

| Output | Contents |
|---|---|
| [forecasts.csv](forecasts.csv) | All 617,504 device-date/configuration rows. A blank `y_true` denotes a masked target. A blank `y_pred` denotes an unavailable forecast. Predictions are fractional expected counts. |
| [device_fit_failures.csv](device_fit_failures.csv) | Device, forecast date, split, window and reason for every unavailable device-specific fit, including dates with masked actuals. |
| [metrics.csv](metrics.csv) | Matched-row scores, exclusions, eligible rows before matching, distinct devices and busy-day errors. |
| [validation_grid.csv](validation_grid.csv) | All 12 validation scores used to choose the common smoothing setting. |

## Model and time windows

Each morning the model forecasts the device's recorded sessions for that date. For forecast date D and window W, it uses dates D−W through D−1 inclusive. February–June supplies the initial training pool. July–August is validation, and September is the rolling test. As each date passes, its usable target can enter later fits. Each training row's features are rebuilt inside the same window, so even the earliest training feature cannot reach before D−W. This historical experiment assumes prior-day usable counts are available by the next forecast. Actual publication latency is unknown.

Both versions use a Poisson regression with a log link, a device-specific recent activity level as an offset, and a categorical weekday coefficient. For each training or forecast row, take up to seven preceding calendar days inside the window. The level is (sum of usable sessions + 1) / (number of usable days + 1). An observed zero counts as a day. A missing target contributes neither a session nor a day. With no recent observations the level is 1.

The expected count equals the recent level times a weekday multiplier. A multiplier of 1.2 raises that device's recent level by 20% for that weekday. The local model estimates seven multipliers separately for each device. The shared model estimates seven across the entire fleet while keeping each device's own recent level. The shared model pools the counts and exposures before fitting, so each device can still receive a different forecast on the same date.

For one weekday, let Y be the sum of valid training counts on that weekday, and E the sum of their corresponding recent levels. The fitted multiplier is (Y + a) / (E + a), where a is the smoothing strength. This is the exact Poisson fit with a weighted pseudo-observation of rate 1 and exposure a per weekday. It shrinks sparse estimates toward a multiplier of 1. The implementation computes the coefficient for the forecast weekday directly because the seven coefficients fit independently. A local model with no valid targets anywhere in the window is marked unavailable. The shared model can still produce a fleet-based forecast for that device.

Validation chose a single common strength by minimizing the average of the four configurations' matched-row MAEs. Strengths 1, 7 and 28 gave 0.843800, 0.833400 and 0.828932 respectively. The run fixed **a = 28** before evaluating September. Daily refitting, the seven-day feature and this candidate grid were implementation choices. No same-day counts, monthly totals, or retrospective coverage flags are predictors.

## Results

Errors are in recorded sessions per device-day. Each row below compares both scopes on identical device-date keys with valid targets and both forecasts available. RMSE gives more weight to larger misses.

| Split | History | Matched rows | Device MAE | Shared MAE | Device RMSE | Shared RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 30 | 81,790 | 0.8297 | 0.8331 | 1.2269 | 1.2246 |
| Validation | 90 | 81,879 | 0.8215 | 0.8314 | 1.2129 | 1.2244 |
| Test | 30 | 39,513 | 0.8119 | 0.8195 | 1.1904 | 1.1967 |
| Test | 90 | 39,550 | 0.8042 | 0.8173 | 1.1784 | 1.1947 |

The September local MAE advantage was 0.0076 sessions at 30 days and 0.0131 at 90 days. Sharing had a small RMSE advantage on 30-day validation. On the 39,513 September rows common to **all four** configurations, the local 90-day model also led with MAE 0.8041 and RMSE 1.1786. These differences describe this cohort and month.

**Coverage.** Validation had 104,036 device-dates, of which 22,124 had unavailable actuals. Among the 81,912 usable actuals, the local model lacked history on 122 rows at 30 days and 33 at 90 days. September had 50,340 device-dates, including 10,776 unavailable actuals. Among the 39,564 usable actuals, 51 and 14 respectively lacked a local forecast. The shared model covered all usable actuals. These extra rows were excluded from both sides of each paired accuracy comparison.

**Busy days.** For this handoff, high usage means at least four sessions, the 90th percentile of valid February–June device-day counts using the higher observed quantile. On the 3,543 matched high-usage September rows at 90 days, the local model underpredicted 81.9% of the time, averaging 1.72 sessions below actual. The shared model underpredicted 87.2%, averaging 1.97 below actual. The local forecast averaged about 3.99 sessions against an actual mean of 5.71. Selecting days by high *realized* counts emphasizes upward surprises, so this conditional shortfall should not be read as overall bias. The team can recompute with its agreed threshold and baseline models from the row-level CSV.

## Sample rows for peer verification

For device `54143` on Tuesday, September 15, the 90-day window is June 17 through September 14. Its recent level was 3.375. Ten usable training Tuesdays had 41 sessions and a summed exposure of 24.957143. The local forecast was 3.375 × (41 + 28) / (24.957143 + 28) = **4.3974**. The shared forecast was **3.5524** and the panel actual was **5**. The 30-day local fit began August 16 and predicted **3.9650**.

For a validation check using the standalone command, filter to the same device on August 3. The 30-day and 90-day local predictions are **5.288623** and **5.053111**, and the actual is **2**. August 1–2 remain masked calendar dates in the seven-day feature. The validation-only rerun reproduced all four exported validation tables. A separate calculation checked the archived targets and matched MAE/RMSE, plus local forecasts on July 3, August 3 and September 15 at both window lengths. Independent team review and comparisons with the team baselines are still pending.
