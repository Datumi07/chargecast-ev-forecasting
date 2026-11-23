#!/usr/bin/env python3
"""
Team B — Unified Evaluation Script (matched_all_models)
Reproduces coverage reports and fair matched-row evaluation tables for
ChargeCast (DATA2010) September 2026 EV charging demand forecasts.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "com_id",
    "reported_date",
    "y_true",
    "y_pred",
    "model",
    "training_mode",
    "window_days",
]
CONFIG_COLS = ["model", "training_mode", "window_days"]
KEY_COLS = ["com_id", "reported_date"]
TOTAL_ELIGIBLE_TEST_ROWS = 39564

MODEL_ROLES = {
    ("lightgbm_l2", "fixed_train_feb_aug", 30): ("Primary Model (Team B)", "Primary"),
    ("lightgbm_l2", "fixed_train_feb_aug", 90): ("Fixed Train Reference (90d)", "Primary"),
    ("lightgbm_l2", "fixed_train_feb_jun", 30): ("Fixed Train Feb-Jun Reference (30d)", "Primary"),
    ("lightgbm_l2", "fixed_train_feb_jun", 90): ("Fixed Train Feb-Jun Reference (90d)", "Primary"),
    ("lightgbm_poisson", "rolling_daily_refit", 90): ("Daily-Refit Benchmark (90d)", "Primary"),
    ("lightgbm_poisson", "rolling_daily_refit", 30): ("Daily-Refit Poisson (30d)", "Primary"),
    ("lightgbm_l1", "rolling_daily_refit", 90): ("Best Overall MAE Model (90d)", "Primary"),
    ("lightgbm_l1", "rolling_daily_refit", 30): ("Daily-Refit L1 (30d)", "Primary"),
    ("lightgbm_l2", "rolling_daily_refit", 90): ("L2 Daily Refit (90d)", "Supplementary"),
    ("lightgbm_l2", "rolling_daily_refit", 30): ("L2 Daily Refit (30d)", "Supplementary"),
    ("baseline_mean7", "history_only", 7): ("Baseline (7-Day Trailing Mean)", "Baseline"),
    ("baseline_lag7", "history_only", 7): ("Baseline (Same Weekday Lag-7)", "Baseline"),
}


def evaluate_cut(df: pd.DataFrame, cut_name: str, busy_threshold: float) -> pd.DataFrame:
    slices: dict[tuple[str, str, int], pd.DataFrame] = {}
    matched_keys: set[tuple[str, str]] | None = None

    for cfg, g in df.groupby(CONFIG_COLS, sort=True):
        keys = set(zip(g["com_id"], g["reported_date"]))
        matched_keys = keys if matched_keys is None else (matched_keys & keys)
        slices[cfg] = g

    if not matched_keys:
        raise ValueError(f"Empty matched_all_models intersection for cut '{cut_name}'.")

    m_idx = pd.DataFrame(list(matched_keys), columns=KEY_COLS)
    ref_y: pd.DataFrame | None = None
    rows = []

    for cfg, g in slices.items():
        m = g.merge(m_idx, on=KEY_COLS, how="inner").sort_values(KEY_COLS).reset_index(drop=True)
        if ref_y is None:
            ref_y = m[KEY_COLS + ["y_true"]].rename(columns={"y_true": "y_ref"})
        else:
            chk = ref_y.merge(m[KEY_COLS + ["y_true"]], on=KEY_COLS, how="inner")
            max_diff = float((chk["y_true"] - chk["y_ref"]).abs().max())
            if max_diff > 1e-9:
                raise ValueError(f"y_true mismatch detected for {cfg} (max_diff={max_diff}).")

        err = m["y_pred"] - m["y_true"]
        busy = m[m["y_true"] >= busy_threshold]
        eb = busy["y_pred"] - busy["y_true"]
        role, status = MODEL_ROLES.get(cfg, ("Other", "Other"))

        rows.append(
            {
                "eval_set": cut_name,
                "model": cfg[0],
                "training_mode": cfg[1],
                "window_days": int(cfg[2]),
                "role": role,
                "status": status,
                "coverage_rows": int(len(g)),
                "coverage_pct": round(len(g) / TOTAL_ELIGIBLE_TEST_ROWS * 100.0, 2),
                "n_matched": int(len(m)),
                "MAE": round(float(np.abs(err).mean()), 4),
                "RMSE": round(float(np.sqrt((err ** 2).mean())), 4),
                "Bias": round(float(err.mean()), 4),
                "n_busy_days": int(len(busy)),
                "Busy_MAE": round(float(np.abs(eb).mean()), 4),
                "Busy_RMSE": round(float(np.sqrt((eb ** 2).mean())), 4),
                "Busy_Bias": round(float(eb.mean()), 4),
                "Busy_UnderPred_pct": round(float((eb < 0).mean() * 100.0), 1),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["RMSE", "Busy_RMSE"], ascending=[True, True])
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Team B unified LightGBM forecasts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/forecasts_september.csv.gz"),
        help="Path to Team B unified forecast file.",
    )
    parser.add_argument(
        "--busy-threshold",
        type=float,
        default=4.0,
        help="Fixed high-usage threshold P90 from Train Feb-Jun 2026 (default: 4.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for evaluation CSV tables (default: results).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["y_true"].notna() & df["y_pred"].notna()].copy()
    df["com_id"] = df["com_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["reported_date"] = pd.to_datetime(df["reported_date"]).dt.strftime("%Y-%m-%d")
    df["window_days"] = df["window_days"].astype(int)

    for cfg, g in df.groupby(CONFIG_COLS):
        dups = int(g.duplicated(subset=KEY_COLS).sum())
        if dups > 0:
            raise ValueError(f"Duplicate (com_id, reported_date) keys found for {cfg}: {dups}")

    cov_rows = []
    for cfg, g in df.groupby(CONFIG_COLS, sort=True):
        role, status = MODEL_ROLES.get(cfg, ("Other", "Other"))
        cov_rows.append(
            {
                "model": cfg[0],
                "training_mode": cfg[1],
                "window_days": int(cfg[2]),
                "role": role,
                "status": status,
                "valid_rows": int(len(g)),
                "eligible_test_rows": TOTAL_ELIGIBLE_TEST_ROWS,
                "coverage_pct": round(len(g) / TOTAL_ELIGIBLE_TEST_ROWS * 100.0, 2),
                "min_date": g["reported_date"].min(),
                "max_date": g["reported_date"].max(),
                "n_devices": int(g["com_id"].nunique()),
            }
        )
    cov_df = pd.DataFrame(cov_rows)

    is_base = df["model"].str.startswith("baseline_")
    df_lgbm = df[~is_base]

    cut_lgbm_all = evaluate_cut(df_lgbm, "matched_lgbm_all", args.busy_threshold)
    cut_lgbm_90 = evaluate_cut(df_lgbm[df_lgbm["window_days"] == 90], "matched_lgbm_90d", args.busy_threshold)
    cut_all_base = evaluate_cut(df, "matched_all_with_baselines", args.busy_threshold)

    summary_df = pd.concat([cut_lgbm_all, cut_lgbm_90, cut_all_base], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cov_df.to_csv(args.output_dir / "coverage.csv", index=False)
    summary_df.to_csv(args.output_dir / "model_summary.csv", index=False)

    print("=== TEAM B: MATCHED LIGHTGBM ALL (39,207 rows, P90 >= 4.0) ===")
    print(cut_lgbm_all[["model", "training_mode", "window_days", "coverage_pct", "MAE", "RMSE", "Busy_MAE", "Busy_RMSE", "Busy_Bias", "status"]].to_string(index=False))
    print("\n=== TEAM B: MATCHED ALL WITH BASELINES (36,483 rows, P90 >= 4.0) ===")
    print(cut_all_base[["model", "training_mode", "window_days", "MAE", "RMSE", "Busy_RMSE", "status"]].to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
