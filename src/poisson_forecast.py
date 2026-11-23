"""Daily rolling-origin Poisson forecasts for ChargeCast Group A."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import PoissonRegressor


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (30, 90)
STRENGTHS = (1, 7, 28)
SCOPES = ("device_specific", "shared")
WEEK = 7
WEEKDAY_FEATURES = np.eye(WEEK)


def load_panel(path, last_day):
    panel = pd.read_csv(
        path, usecols=["com_id", "reported_date", "sessions_target", "target_valid"],
        dtype={"com_id": "string", "target_valid": bool}, parse_dates=["reported_date"],
    )
    panel = panel[panel.reported_date.between("2026-02-01", last_day)]
    target = panel.sessions_target
    if not target.notna().equals(panel.target_valid):
        raise ValueError("Target validity mask disagrees with target availability")
    if ((target < 0) | (target.mod(1) > 0)).any() or panel.com_id.isna().any():
        raise ValueError("Expected nonnegative integer counts and nonmissing device IDs")
    counts = panel.pivot(index="reported_date", columns="com_id", values="sessions_target")
    calendar = pd.date_range("2026-02-01", last_day)
    if not counts.index.equals(calendar) or counts.size != len(panel):
        raise ValueError("Expected one panel row per device per calendar date")
    return counts


def weekly_levels(history):
    """Smoothed prior-seven-calendar-day mean for each training row and the origin."""
    days = np.arange(len(history) + 1)
    starts = np.maximum(0, days - WEEK)
    zero = np.zeros((1, history.shape[1]))
    events = np.vstack((zero, np.nancumsum(history, axis=0)))
    valid = np.vstack((zero, np.cumsum(np.isfinite(history), axis=0)))
    return (1 + events[days] - events[starts]) / (1 + valid[days] - valid[starts])


def fit_weekday_model(weekdays, counts, exposure, strength):
    """Fit weekday rate multipliers using the recent activity level as exposure."""
    valid = exposure > 0
    # One binary feature per weekday. Append one rate-1 observation per
    # weekday so that rare or unseen weekdays shrink toward the recent level.
    X = np.vstack([WEEKDAY_FEATURES[weekdays[valid]], WEEKDAY_FEATURES])
    rates = np.r_[counts[valid] / exposure[valid], np.ones(WEEK)]
    weights = np.r_[exposure[valid], np.full(WEEK, strength)]
    model = PoissonRegressor(alpha=0, fit_intercept=False, solver="newton-cholesky", tol=1e-10)
    model.fit(X, rates, sample_weight=weights)
    return model


def forecast_origin(history, weekdays, forecast_weekday, strengths):
    """Fit one GLM per eligible device and one GLM across all devices."""
    levels = weekly_levels(history)
    available = np.isfinite(history)
    exposure = np.where(available, levels[:-1], 0)
    devices = np.flatnonzero(available.any(axis=0))
    # Within a date all devices have identical weekday features, so their
    # counts and exposures can be summed for the shared fit.
    pooled_counts = np.nansum(history, axis=1)
    pooled_exposure = exposure.sum(axis=1)
    next_weekday = WEEKDAY_FEATURES[[forecast_weekday]]
    predictions = {}
    for strength in strengths:
        local = np.full(history.shape[1], np.nan)
        shared = np.full(history.shape[1], np.nan)
        for device in devices:
            model = fit_weekday_model(weekdays, history[:, device], exposure[:, device], strength)
            local[device] = levels[-1, device] * model.predict(next_weekday)[0]
        if devices.size:
            model = fit_weekday_model(weekdays, pooled_counts, pooled_exposure, strength)
            shared = levels[-1] * model.predict(next_weekday)[0]
        predictions[strength, "device_specific"] = local
        predictions[strength, "shared"] = shared
    return predictions


def roll(panel, split, strengths, jobs=1):
    dates = panel.index
    days = dates.dayofweek.to_numpy()
    counts = panel.to_numpy(dtype=float)
    start, end = ("2026-07-01", "2026-08-31") if split == "validation" else ("2026-09-01", "2026-09-30")
    origins = np.flatnonzero((dates >= start) & (dates <= end))
    blocks = {(strength, window, scope): [] for strength in strengths for window in WINDOWS for scope in SCOPES}

    def forecast_date(origin):
        by_window = {}
        for window in WINDOWS:
            if origin < window:
                raise ValueError("Incomplete calendar history")
            past = slice(origin - window, origin)
            by_window[window] = forecast_origin(counts[past], days[past], days[origin], strengths)
        return by_window

    with parallel_config(backend="loky", inner_max_num_threads=1):
        results_by_date = Parallel(n_jobs=jobs)(delayed(forecast_date)(origin) for origin in origins)
    for by_window in results_by_date:
        for window, results in by_window.items():
            for (strength, scope), pred in results.items():
                blocks[strength, window, scope].append(pred)
    return dates[origins], counts[origins], {key: np.stack(values) for key, values in blocks.items()}


def score(y, pred, common, high_cutoff):
    valid = np.isfinite(y)
    available = np.isfinite(pred)
    error = pred[common] - y[common]
    high = error[y[common] >= high_cutoff]
    return {
        "total_rows": y.size, "invalid_target": int((~valid).sum()),
        "pred_unavailable": int((~available).sum()),
        "valid_target_no_pred": int((valid & ~available).sum()),
        "eligible_own_rows": int((valid & available).sum()),
        "excluded_for_matching": int((valid & available & ~common).sum()),
        "n_scored": len(error), "devices_scored": int(common.any(axis=0).sum()),
        "mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error**2)),
        "high_cutoff": high_cutoff, "high_rows": len(high),
        "high_under_fraction": (high < 0).mean() if len(high) else np.nan,
        "high_mean_error": high.mean() if len(high) else np.nan,
    }


def metrics(split, truth, predictions, high_cutoff):
    rows = []
    for (strength, window, scope), pred in predictions.items():
        common = np.isfinite(truth)
        for other in SCOPES:
            common &= np.isfinite(predictions[strength, window, other])
        rows.append({
            "split": split, "strength": strength, "history_days": window, "training_scope": scope,
            **score(truth, pred, common, high_cutoff),
        })
    return pd.DataFrame(rows)


def export_split(dest, split, ids, dates, truth, predictions, strength):
    keys = pd.MultiIndex.from_product([dates, ids], names=["reported_date", "com_id"]).to_frame(index=False)
    for i, (window, scope) in enumerate((w, s) for w in WINDOWS for s in SCOPES):
        pred = predictions[strength, window, scope].ravel()
        rows = keys.assign(split=split, model="poisson", training_scope=scope, history_days=window,
                           y_true=pd.array(truth.ravel(), dtype="Int64"), y_pred=pred)
        first = split == "validation" and i == 0
        rows.to_csv(dest / "forecasts.csv", mode="w" if first else "a", header=first, index=False,
                    columns=["com_id", "reported_date", "split", "model", "training_scope",
                             "history_days", "y_true", "y_pred"], float_format="%.12g")
        if scope == "device_specific":
            failures = rows.loc[~np.isfinite(pred), ["com_id", "reported_date", "split", "history_days"]].copy()
            failures["reason"] = "no_valid_training_targets"
            failures.to_csv(dest / "device_fit_failures.csv", mode="w" if first else "a",
                            header=first, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=4, help="Forecast dates to fit in parallel (default: 4)")
    args = parser.parse_args()
    dest = args.output or ROOT / "reports" / ("poisson_validation" if args.validation_only else "poisson")
    dest.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/processed/daily_panel.csv.gz"
    panel = load_panel(path, "2026-08-31")
    high_cutoff = int(np.nanquantile(panel.loc[:"2026-06-30"].to_numpy(), 0.90, method="higher"))
    dates, truth, predictions = roll(panel, "validation", STRENGTHS, args.jobs)
    grid = metrics("validation", truth, predictions, high_cutoff)
    grid.to_csv(dest / "validation_grid.csv", index=False)
    # Equal weight for both scopes and both windows. Smallest strength wins a tie.
    strength = int(grid.groupby("strength", sort=True).mae.mean().idxmin())
    selected = grid.loc[grid.strength == strength]
    export_split(dest, "validation", panel.columns, dates, truth, predictions, strength)
    if not args.validation_only:
        panel = load_panel(path, "2026-09-30")
        dates, truth, predictions = roll(panel, "test", [strength], args.jobs)
        selected = pd.concat([selected, metrics("test", truth, predictions, high_cutoff)])
        export_split(dest, "test", panel.columns, dates, truth, predictions, strength)
    selected.to_csv(dest / "metrics.csv", index=False)
    print(selected[["split", "strength", "history_days", "training_scope", "n_scored", "mae", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
