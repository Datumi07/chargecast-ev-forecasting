"""Run with: python src/check_features.py"""
import numpy as np
import pandas as pd

from build_features import build_features


def main():
    dates = pd.date_range("2026-02-01", periods=120)
    panel = pd.DataFrame([
        {"com_id": device, "reported_date": date, "sessions_target": float(i + offset),
         "target_valid": True, "recommended_period": True, "split": "train"}
        for device, offset in (("A", 0), ("B", 1000))
        for i, date in enumerate(dates)
    ])
    gap = panel.com_id.eq("A") & panel.reported_date.eq(dates[95])
    panel.loc[gap, "sessions_target"] = np.nan
    panel.loc[gap, "target_valid"] = False
    for window in (30, 90):
        result, predictors = build_features(panel.sample(frac=1, random_state=1), window)
        a = result[result.com_id.eq("A")].reset_index(drop=True)
        assert len(result) == len(panel) and len(predictors) == 26
        assert a.weekday.tolist() == dates.dayofweek.tolist()
        assert a.month.tolist() == dates.month.tolist()
        assert a.is_weekend.tolist() == (dates.dayofweek >= 5).astype(int).tolist()
        np.testing.assert_allclose(a.weekday_sin, np.sin(2 * np.pi * dates.dayofweek / 7))
        np.testing.assert_allclose(a.weekday_cos, np.cos(2 * np.pi * dates.dayofweek / 7))
        # Independent calendar slices check every statistic, including empty history.
        for _, device in result.groupby("com_id"):
            device = device.reset_index(drop=True)
            for i in range(len(device)):
                for lag in (1, 7, 14, 28):
                    expected_lag = device.sessions_target.iloc[i-lag] if i >= lag else np.nan
                    np.testing.assert_allclose(device.loc[i, f"lag_{lag}"], expected_lag, equal_nan=True)
                for size in (7, 14, window):
                    history = device.sessions_target.iloc[max(0, i-size):i].dropna()
                    for stat in ("mean", "std", "min", "max", "count"):
                        np.testing.assert_allclose(device.loc[i, f"rolling_{stat}_{size}"], getattr(history, stat)(), equal_nan=True)
                history = device.sessions_target.iloc[max(0, i-window):i].dropna()
                zero_fraction = history.eq(0).mean() if len(history) else np.nan
                np.testing.assert_allclose(device.loc[i, "zero_fraction"], zero_fraction, equal_nan=True)
                assert device.loc[i, "history_observed_fraction"] == len(history) / window
        assert a.loc[96, "lag_1"] != a.loc[96, "lag_1"]
        assert a.loc[102, "lag_7"] != a.loc[102, "lag_7"]
        expected = panel.loc[panel.com_id.eq("A"), "sessions_target"].iloc[100-window:100]
        assert np.isclose(a.loc[100, f"rolling_mean_{window}"], expected.mean())
        assert a.loc[100, f"rolling_count_{window}"] == window - 1
        assert np.isclose(a.loc[100, "history_observed_fraction"], (window - 1) / window)
        assert a.loc[window, "history_window_available"]
        assert not a.loc[window - 1, "history_window_available"]
        assert a.loc[1, "zero_fraction"] == 1
        assert pd.isna(a.loc[0, "rolling_mean_7"])
        assert result.loc[result.com_id.eq("B"), "lag_1"].iloc[1] == 1000
        assert not set(predictors) & (set(panel.columns) | {"label_eligible", "complete_features", "history_window_available"})
        assert result.label_eligible.equals(result.target_valid & result.recommended_period & result.history_window_available)
        assert result.complete_features.equals(result[predictors].notna().all(axis=1))
        pd.testing.assert_frame_equal(result, build_features(panel, window)[0])
        empty_history = panel.copy()
        empty_history["sessions_target"] = np.nan
        empty_history["target_valid"] = False
        empty, _ = build_features(empty_history, window)
        assert empty[f"rolling_count_{window}"].eq(0).all()
        assert empty.history_observed_fraction.eq(0).all()
        assert empty[["lag_1", "zero_fraction", f"rolling_mean_{window}", f"rolling_std_{window}"]].isna().all().all()
        assert not empty.label_eligible.any()
        changed = panel.copy()
        changed.loc[changed.reported_date.ge(dates[100]), "sessions_target"] = 9999
        changed.loc[changed.reported_date.ge(dates[100]), "target_valid"] = True
        future, _ = build_features(changed, window)
        pd.testing.assert_frame_equal(result.loc[result.reported_date.le(dates[100]), predictors], future.loc[future.reported_date.le(dates[100]), predictors])
        bounded = panel[panel.reported_date.ge(dates[100-window]) & panel.reported_date.le(dates[100])]
        truncated, _ = build_features(bounded, window)
        pd.testing.assert_frame_equal(result.loc[result.reported_date.eq(dates[100]), predictors].reset_index(drop=True), truncated.loc[truncated.reported_date.eq(dates[100]), predictors].reset_index(drop=True))
    for invalid in (panel.drop(index=5), pd.concat([panel, panel.iloc[:1]])):
        try:
            build_features(invalid, 30)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid calendar accepted")
    for value in (-1, 0.5, np.inf, np.nan):
        invalid = panel.copy()
        invalid.loc[0, "sessions_target"] = value
        try:
            build_features(invalid, 30)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid target accepted: {value}")
    print("Passed: calendar lags, device isolation, missing labels, window bounds, future invariance, input validation")


if __name__ == "__main__":
    main()
