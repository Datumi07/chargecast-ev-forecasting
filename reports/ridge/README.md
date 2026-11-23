# Shared Ridge regression

From the repository root, install the project requirements and run `python src/run_ridge.py`. The script regenerates the three files in this folder from the daily panel and the merged Team B forecast export.

Each forecast estimates next-day session starts for one device. A pooled linear model uses weekday indicators, device-specific lags (1, 7, 14, 28 days) and trailing means (7, 28, 90 days). The device ID groups history. Trailing means exclude the target day and omit unavailable prior labels. Historical features assume that eligible previous-day counts are available by forecast time and inherit the panel's retrospective zero-coverage assumptions.

Fit a single model before each evaluation period using the preceding 30 or 90 calendar days of eligible labels. Features may use older history. Impute medians, add missingness indicators and standardize using the training slice. Ridge penalizes squared coefficient magnitude. Clip negative forecasts to zero for scoring. Features update daily while coefficients stay fixed.

Select the window and penalty by lowest overall July–August RMSE on all eligible rows. Refit the selected configuration before September. The team had already inspected September in earlier experiments. Treat it as a retrospective comparison. Validation RMSE was effectively tied for two penalties at 90 days.

- `validation.csv` records every candidate, ordered by selection score.
- `forecasts.csv.gz` contains September forecasts keyed by device and date.
- `test_metrics.csv` reports full Ridge coverage and common-row comparisons with the two published baselines and Team B's fixed model. Busy days have at least four recorded sessions. Bias is prediction minus actual. Underprediction rate is the fraction of strictly negative errors.

This shared Ridge experiment addresses baseline comparison and busy-day errors. Team A's matched shared/local experiment addresses the effect of pooling devices.
