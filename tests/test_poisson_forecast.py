import unittest

import numpy as np
import pandas as pd

from src.poisson_forecast import forecast_origin, roll, weekly_levels


class PoissonForecastTests(unittest.TestCase):
    def test_missing_days_age_out_and_zero_counts_have_exposure(self):
        history = np.array([[9, np.nan], [np.nan, 0], *([[np.nan, np.nan]] * 6)])
        levels = weekly_levels(history)
        np.testing.assert_allclose(levels[0], [1, 1])
        np.testing.assert_allclose(levels[7], [5, 0.5])
        np.testing.assert_allclose(levels[8], [1, 0.5])

    def test_separate_and_pooled_weekday_effects(self):
        history = np.full((14, 3), np.nan)
        history[[0, 7], 0] = [4, 2]
        history[[0, 7], 1] = [0, 0]
        pred = forecast_origin(history, np.arange(14) % 7, 0, [1])
        # Two Mondays per observed device. Prior levels on those Mondays are
        # (1, 2.5) and (1, 0.5). Next Monday's levels are 1.5, 0.5 and 1.
        np.testing.assert_allclose(pred[1, "device_specific"], [1.5 * 7 / 4.5, 0.5 / 2.5, np.nan])
        np.testing.assert_allclose(pred[1, "shared"], np.array([1.5, 0.5, 1]) * 7 / 6)

    def test_origin_uses_only_its_calendar_window(self):
        dates = pd.date_range("2026-03-01", "2026-07-02")
        panel = pd.DataFrame({"001": np.ones(len(dates))}, index=dates)
        for window in (30, 90):
            changed = panel.copy()
            start = pd.Timestamp("2026-07-01") - pd.Timedelta(days=window)
            changed.loc[changed.index < start] = 900
            changed.loc["2026-07-01":] = 700
            _, _, forecasts = roll(changed, "validation", [7])
            for scope in ("device_specific", "shared"):
                np.testing.assert_allclose(forecasts[7, window, scope][0], [1])
