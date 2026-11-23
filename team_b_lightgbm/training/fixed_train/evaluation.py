"""Shared evaluation utilities for ChargeCast EV charging session forecasting.

Follows the feature-engineering handoff contract in docs/feature_engineering.md:
- Predictors are loaded strictly from predictors_{window}d.json.
- Target is `sessions_target`.
- LightGBM filters on `label_eligible` (missing-aware; never fills missing lags with 0).
- Fair comparison with complete-case models (e.g., PoissonRegressor) is always
  reported on the `label_eligible & complete_features` subset alongside `all_eligible`.
- High-usage threshold is computed strictly on the training split to prevent data leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_PREDICTORS = {
    "com_id",
    "reported_date",
    "sessions_target",
    "split",
    "recommended_period",
    "target_valid",
    "history_window_available",
    "label_eligible",
    "complete_features",
}


def load_predictors(json_path: Path) -> list[str]:
    """Load predictor column names from predictors_{window}d.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        predictors = payload
    elif isinstance(payload, dict):
        for key in ("predictors", "features", "columns"):
            if key in payload and isinstance(payload[key], list):
                predictors = payload[key]
                break
        else:
            raise ValueError(
                f"Unrecognized JSON structure in {json_path}. Keys: {list(payload.keys())}"
            )
    else:
        raise ValueError(f"Unsupported JSON payload type in {json_path}: {type(payload)}")

    leaked = set(predictors).intersection(FORBIDDEN_PREDICTORS)
    if leaked:
        raise ValueError(f"Forbidden non-predictor columns found in allowlist: {sorted(leaked)}")

    return list(predictors)


def load_feature_dataset(
    window: int,
    features_dir: str | Path = "data/processed/features",
) -> tuple[pd.DataFrame, list[str]]:
    """Load features_{window}d.csv.gz and its predictor allowlist JSON."""
    if window not in (30, 90):
        raise ValueError(f"Expected window to be 30 or 90, got {window}")

    base = Path(features_dir)
    csv_path = base / f"features_{window}d.csv.gz"
    json_path = base / f"predictors_{window}d.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {csv_path}. Run `python src/build_features.py` first."
        )
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing predictor allowlist: {json_path}. Run `python src/build_features.py` first."
        )

    predictors = load_predictors(json_path)
    df = pd.read_csv(
        csv_path,
        dtype={"com_id": str},
        parse_dates=["reported_date"],
    )

    for col in ("label_eligible", "complete_features"):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df, predictors


def filter_split(df: pd.DataFrame, split_name: str, eligible_only: bool = True) -> pd.DataFrame:
    """Filter DataFrame by canonical split name ('train', 'validation', 'test')."""
    split_lower = df["split"].astype(str).str.strip().str.lower()
    target = split_name.strip().lower()

    if target in ("validation", "val"):
        mask = split_lower.isin(["validation", "val"])
    else:
        mask = split_lower.eq(target)

    if eligible_only:
        mask = mask & df["label_eligible"]

    return df.loc[mask].copy()


def compute_high_usage_threshold(
    df_train_eligible: pd.DataFrame,
    quantile: float = 0.90,
) -> float:
    """Compute high-usage session threshold strictly on the eligible training split."""
    y_train = df_train_eligible["sessions_target"].dropna()
    if y_train.empty:
        raise ValueError("Training split has no valid sessions_target values.")
    return float(np.quantile(y_train, quantile))


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    high_usage_threshold: float,
) -> dict[str, Any]:
    """Compute MAE, RMSE, bias, and high-usage underprediction metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(yt) & np.isfinite(yp)
    n_total = int(len(yt))
    n_eval = int(valid.sum())
    n_excluded = n_total - n_eval

    if n_eval == 0:
        return {
            "n_total": n_total,
            "n_eval": 0,
            "n_excluded": n_excluded,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "high_usage_threshold": high_usage_threshold,
            "high_usage_n": 0,
            "high_usage_mae": np.nan,
            "high_usage_rmse": np.nan,
            "high_usage_underpred_rate": np.nan,
            "high_usage_mean_underpred": np.nan,
            "high_usage_bias": np.nan,
        }

    yt = yt[valid]
    yp = yp[valid]
    err = yp - yt  # negative => underprediction

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    high_mask = yt >= high_usage_threshold
    high_n = int(high_mask.sum())

    if high_n > 0:
        err_hi = err[high_mask]
        hi_mae = float(np.mean(np.abs(err_hi)))
        hi_rmse = float(np.sqrt(np.mean(err_hi ** 2)))
        hi_underpred_rate = float(np.mean(err_hi < 0))
        hi_mean_underpred = float(np.mean(np.maximum(0.0, -err_hi)))
        hi_bias = float(np.mean(err_hi))
    else:
        hi_mae = hi_rmse = hi_underpred_rate = hi_mean_underpred = hi_bias = np.nan

    return {
        "n_total": n_total,
        "n_eval": n_eval,
        "n_excluded": n_excluded,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "high_usage_threshold": round(float(high_usage_threshold), 4),
        "high_usage_n": high_n,
        "high_usage_mae": round(hi_mae, 4) if np.isfinite(hi_mae) else np.nan,
        "high_usage_rmse": round(hi_rmse, 4) if np.isfinite(hi_rmse) else np.nan,
        "high_usage_underpred_rate": round(hi_underpred_rate, 4) if np.isfinite(hi_underpred_rate) else np.nan,
        "high_usage_mean_underpred": round(hi_mean_underpred, 4) if np.isfinite(hi_mean_underpred) else np.nan,
        "high_usage_bias": round(hi_bias, 4) if np.isfinite(hi_bias) else np.nan,
    }


def evaluate_predictions(
    df_split_eligible: pd.DataFrame,
    y_pred: np.ndarray | pd.Series,
    high_usage_threshold: float,
    model_name: str,
    window: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Evaluate predictions on both `all_eligible` and `complete_case` subsets."""
    y_true = df_split_eligible["sessions_target"].to_numpy(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    complete_mask = df_split_eligible["complete_features"].to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    subsets = [
        ("all_eligible", np.ones(len(df_split_eligible), dtype=bool)),
        ("complete_case", complete_mask),
    ]

    for subset_name, mask in subsets:
        metrics = compute_metrics(
            y_true=y_true[mask],
            y_pred=y_pred_arr[mask],
            high_usage_threshold=high_usage_threshold,
        )
        rows.append(
            {
                "model": model_name,
                "window_days": window,
                "split": split_name,
                "subset": subset_name,
                **metrics,
            }
        )

    return rows