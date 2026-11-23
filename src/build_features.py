"""Build next-day predictors on the continuous Clean v1 calendar."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["com_id", "reported_date"]
CONTROLS = ["sessions_target", "target_valid", "recommended_period", "split"]


def build_features(panel, history_days):
    """Row date is the prediction date; history ends on the preceding day."""
    if history_days not in (30, 90):
        raise ValueError("history_days must be 30 or 90")
    required = KEYS + CONTROLS
    if missing := set(required) - set(panel.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    p = panel[required].copy()
    p["reported_date"] = pd.to_datetime(p.reported_date, errors="raise")
    if p.empty or p[KEYS].isna().any().any() or p.duplicated(KEYS).any():
        raise ValueError("Expected nonempty, unique, nonmissing device-date keys")
    if p.reported_date.dt.tz is not None or not p.reported_date.eq(p.reported_date.dt.normalize()).all():
        raise ValueError("Expected timezone-naive calendar dates at midnight")
    for column in ("target_valid", "recommended_period"):
        if p[column].isna().any() or not p[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain booleans")
        p[column] = p[column].astype(bool)
    y = pd.to_numeric(p.sessions_target, errors="raise").astype(float)
    if not y.notna().equals(p.target_valid) or not (np.isfinite(y.dropna()) & y.dropna().ge(0) & y.dropna().mod(1).eq(0)).all():
        raise ValueError("Targets must be nonnegative integer counts exactly where target_valid")
    p["sessions_target"] = y
    p = p.sort_values(KEYS).reset_index(drop=True)
    differences = p.groupby("com_id", sort=False).reported_date.diff().dropna()
    if not differences.dt.total_seconds().eq(86400).all():
        raise ValueError("Keep every calendar day before constructing features")

    features = []
    def add(name, values):
        p[name] = values
        features.append(name)

    weekday = p.reported_date.dt.dayofweek
    add("weekday", weekday)
    add("is_weekend", weekday.ge(5).astype(int))
    add("month", p.reported_date.dt.month)
    add("weekday_sin", np.sin(2 * np.pi * weekday / 7))
    add("weekday_cos", np.cos(2 * np.pi * weekday / 7))
    grouped = p.groupby("com_id", sort=False).sessions_target
    for lag in (1, 7, 14, 28):
        add(f"lag_{lag}", grouped.shift(lag))
    # Shift first: no statistic can include the outcome on its own row.
    prior = grouped.shift(1)
    for window in sorted({7, 14, history_days}):
        rolling = prior.groupby(p.com_id, sort=False).rolling(window, min_periods=1)
        for stat in ("mean", "std", "min", "max", "count"):
            values = getattr(rolling, stat)().reset_index(level=0, drop=True).sort_index()
            if stat == "count":
                values = values.fillna(0).astype(int)
            add(f"rolling_{stat}_{window}", values)
    zero_prior = p.sessions_target.eq(0).astype(float).where(p.sessions_target.notna()).groupby(p.com_id, sort=False).shift(1)
    add("zero_fraction", zero_prior.groupby(p.com_id, sort=False).rolling(history_days, min_periods=1).mean().reset_index(level=0, drop=True).sort_index())
    add("history_observed_fraction", p[f"rolling_count_{history_days}"] / history_days)
    # Eligibility controls are never predictors. Missing history stays missing.
    p["history_window_available"] = p.groupby("com_id", sort=False).cumcount().ge(history_days)
    p["label_eligible"] = p.recommended_period & p.target_valid & p.history_window_available
    p["complete_features"] = p[features].notna().all(axis=1)
    return p, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/daily_panel.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/features")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/feature_engineering/coverage.csv")
    args = parser.parse_args()
    panel = pd.read_csv(args.input, dtype={"com_id": "string"}, parse_dates=["reported_date"])
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for window in (30, 90):
        frame, columns = build_features(panel, window)
        frame.to_csv(args.output / f"features_{window}d.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        (args.output / f"predictors_{window}d.json").write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")
        for split in ("train", "validation", "test"):
            subset = frame[frame.split.eq(split) & frame.recommended_period]
            eligible = subset[subset.label_eligible]
            summaries.append({"history_days": window, "split": split, "rows": len(subset), "valid_labels": int(subset.target_valid.sum()), "eligible_labels": len(eligible), "devices": eligible.com_id.nunique(), "complete_feature_rows": int(eligible.complete_features.sum()), "missing_label_fraction": float(1 - subset.target_valid.mean()), "zero_label_fraction": float(eligible.sessions_target.eq(0).mean()), "mean_history_observed_fraction": float(eligible.history_observed_fraction.mean())})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.report, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
