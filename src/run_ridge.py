"""Train and evaluate pooled next-day Ridge forecasts from the cleaned daily panel."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ridge"
KEY = ["com_id", "reported_date"]


def make_features(panel):
    history = panel.groupby("com_id", sort=False)["sessions_target"]
    weekday = panel.reported_date.dt.weekday
    features = pd.get_dummies(pd.Categorical(weekday, categories=range(7)), prefix="weekday", dtype=float)
    for days in (1, 7, 14, 28):
        features[f"lag_{days}"] = history.shift(days)
    for days in (7, 28, 90):
        features[f"mean_{days}"] = history.transform(
            lambda y: y.shift(1).rolling(days, min_periods=1).mean()
        )
    return features


def score(frame):
    error = frame.y_pred - frame.y_true
    busy = error[frame.y_true >= 4]
    return dict(rows=len(error), mae=error.abs().mean(), rmse=np.sqrt(error.pow(2).mean()),
                busy_rows=len(busy), busy_rmse=np.sqrt(busy.pow(2).mean()),
                busy_bias=busy.mean(), busy_under_rate=(busy < 0).mean())


def forecast(panel, features, origin, end, window, alpha):
    train = panel.reported_date.between(origin - pd.Timedelta(days=window),
                                        origin - pd.Timedelta(days=1)) & panel.target_valid
    test = panel.reported_date.between(origin, end) & panel.target_valid
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), Ridge(alpha=alpha))
    model.fit(features.loc[train], panel.loc[train, "sessions_target"])
    result = panel.loc[test, KEY + ["sessions_target"]].rename(columns={"sessions_target": "y_true"}).copy()
    result["y_pred"] = np.maximum(0, model.predict(features.loc[test]))
    return result


def main():
    panel = pd.read_csv(ROOT / "data/processed/daily_panel.csv.gz", dtype={"com_id": str},
                        parse_dates=["reported_date"]).sort_values(KEY).reset_index(drop=True)
    assert not panel.duplicated(KEY).any()
    assert panel.groupby("com_id").reported_date.diff().dropna().eq(pd.Timedelta(days=1)).all()
    features = make_features(panel)
    validation = []
    for window in (30, 90):
        for alpha in (0.1, 10, 1000):
            pred = forecast(panel, features, pd.Timestamp("2026-07-01"), "2026-08-31", window, alpha)
            validation.append(dict(window=window, alpha=alpha, **score(pred)))
    trials = pd.DataFrame(validation).sort_values(["rmse", "window", "alpha"])
    OUT.mkdir(parents=True, exist_ok=True)
    trials.to_csv(OUT / "validation.csv", index=False)
    best = trials.iloc[0]
    pred = forecast(panel, features, pd.Timestamp("2026-09-01"), "2026-09-30",
                    int(best.window), float(best.alpha))
    pred.to_csv(OUT / "forecasts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    published = pd.read_csv(ROOT / "team_b_lightgbm/data/forecasts_september.csv.gz",
                            dtype={"com_id": str}, parse_dates=["reported_date"])
    chosen = {"lag7": published.model.eq("baseline_lag7"),
              "mean7": published.model.eq("baseline_mean7"),
              "team_b_fixed": published.model.eq("lightgbm_l2") &
                  published.training_mode.eq("fixed_train_feb_aug") & published.window_days.eq(30)}
    matched = pred.rename(columns={"y_pred": "ridge"})
    for name, mask in chosen.items():
        ref = published.loc[mask, KEY + ["y_true", "y_pred"]].rename(columns={"y_pred": name})
        matched = matched.merge(ref, on=KEY, validate="one_to_one", suffixes=("", "_ref"))
        assert matched.y_true.eq(matched.pop("y_true_ref")).all()
    results = [dict(model="ridge", population="all", **score(pred))]
    for name in ["ridge", *chosen]:
        results.append(dict(model=name, population="matched", **score(matched.assign(y_pred=matched[name]))))
    pd.DataFrame(results).to_csv(OUT / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
