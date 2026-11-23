"""Runnable leakage, window, failure and scoring checks: python src/check_poisson.py."""
from datetime import timedelta
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from build_features import build_features
from run_poisson import FEATURES, experiment, fit_predict, metrics
from threadpoolctl import threadpool_limits


def main():
    dates = pd.date_range('2026-02-01', '2026-07-02')
    panel = pd.DataFrame([dict(com_id=device, reported_date=date, sessions_target=float(i % 5), target_valid=True, recommended_period=True, split='validation' if date.month == 7 else 'train') for device in ('001', '002') for i, date in enumerate(dates)])
    with threadpool_limits(limits=1):
        for days in (30, 90):
            frame, _ = build_features(panel, days)
            predictions, audit = experiment(frame, days, 'validation', 1.)
            assert len(predictions) == 8
            assert predictions.y_pred.notna().all()
            assert audit.n_train.min() == days - 7
            assert (audit.train_end < audit.reported_date).all()
            assert (audit.reported_date - audit.train_start).dt.days.eq(days).all()
            # Independent truncated rebuild reproduces both scopes on first origin.
            origin = pd.Timestamp('2026-07-01')
            bounded = panel[panel.reported_date.between(origin-timedelta(days=int(days)), origin)]
            cropped, _ = build_features(bounded, days)
            train = cropped.reported_date.between(origin-timedelta(days=int(days)-7), origin-timedelta(days=1))
            for scope in ('shared', 'device_specific'):
                use = train if scope == 'shared' else train & cropped.com_id.eq('001')
                future = cropped.com_id.eq('001') & cropped.reported_date.eq(origin)
                expected, _ = fit_predict(cropped.loc[use, FEATURES], cropped.loc[use, 'sessions_target'], cropped.loc[future, FEATURES], 1.)
                actual = predictions.query('training_scope == @scope and com_id == "001" and reported_date == @origin').y_pred.iloc[0]
                np.testing.assert_allclose(actual, expected[0])
            # A changed forecast-day outcome cannot alter that day's prediction.
            changed = panel.copy()
            changed.loc[changed.reported_date.ge(origin), 'sessions_target'] = 99.
            perturbed, _ = experiment(build_features(changed, days)[0], days, 'validation', 1.)
            np.testing.assert_allclose(predictions.loc[predictions.reported_date.eq(origin), 'y_pred'], perturbed.loc[perturbed.reported_date.eq(origin), 'y_pred'])
            old = panel.copy()
            old.loc[old.reported_date.lt(origin-timedelta(days=int(days))), 'sessions_target'] = 500.
            older, _ = experiment(build_features(old, days)[0], days, 'validation', 1.)
            np.testing.assert_allclose(predictions.y_pred, older.y_pred)
            missing_history = panel.copy()
            gap = missing_history.reported_date.eq(origin-timedelta(days=7))
            missing_history.loc[gap, 'sessions_target'] = np.nan
            missing_history.loc[gap, 'target_valid'] = False
            unavailable, _ = experiment(build_features(missing_history, days)[0], days, 'validation', 1.)
            assert unavailable.loc[unavailable.reported_date.eq(origin), 'y_pred'].isna().all()
            assert unavailable.loc[unavailable.reported_date.eq(origin), 'status'].eq('missing_forecast_features').all()
            scored = metrics(predictions, 3.)
            assert scored.n_scored.eq(4).all()
            missing = predictions.copy()
            missing.loc[0, 'y_pred'] = np.nan
            assert metrics(missing, 3.).query('population == "matched"').n_scored.eq(3).all()
        assert fit_predict(np.ones((6, 4)), np.ones(6), np.ones((1, 4)), 1.)[0] is None
        pred, status = fit_predict(np.ones((7, 4)), np.zeros(7), np.ones((1, 4)), 1.)
        assert pred[0] == 0 and status == 'all_zero_boundary'
    # Exercise selection/export CLI, not just the individual fitting functions.
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        panel.to_csv(folder/'panel.csv', index=False)
        subprocess.run([sys.executable, str(Path(__file__).with_name('run_poisson.py')),
                        '--input', str(folder/'panel.csv'), '--output', str(folder/'output'),
                        '--validation-only', '--jobs', '1'], check=True, capture_output=True, text=True)
        settings = json.loads((folder/'output/settings.json').read_text())
        assert len(settings['selected_alpha']) == 4
        output = pd.read_csv(folder/'output/forecasts.csv', dtype={'com_id': 'string'})
        assert len(output) == 16 and output.y_pred.notna().all()
        assert output.split.eq('validation').all()
    check_delivery()
    print('Passed: bounded history, daily refits, future invariance, string IDs, matched scoring, insufficient/zero history')


def check_delivery():
    folder = Path('reports/group_a')
    if not (folder/'forecasts.csv').exists():
        return
    forecast = pd.read_csv(folder/'forecasts.csv', dtype={'com_id': 'string'}, parse_dates=['reported_date'])
    panel = pd.read_csv('data/processed/daily_panel.csv.gz', dtype={'com_id': 'string'}, parse_dates=['reported_date'])
    keys = ['com_id', 'reported_date']
    assert not forecast.duplicated(keys+['training_scope', 'history_days']).any()
    valid = panel[panel.target_valid & panel.recommended_period & panel.split.isin(['validation', 'test'])]
    assert len(forecast) == 4 * len(valid)
    merged = forecast.merge(valid[keys+['sessions_target', 'split']], on=keys, validate='many_to_one', suffixes=('', '_actual'))
    assert len(merged) == len(forecast)
    assert merged.y_true.eq(merged.sessions_target).all()
    assert merged.split.eq(merged.split_actual).all()
    assert forecast.loc[forecast.y_pred.notna(), 'y_pred'].ge(0).all()
    assert forecast.loc[forecast.y_pred.notna(), 'status'].isin(['ok', 'all_zero_boundary']).all()
    settings = json.loads((folder/'settings.json').read_text())
    expected_threshold = panel.loc[panel.target_valid & panel.split.eq('train') & panel.reported_date.ge('2026-02-01'), 'sessions_target'].quantile(.9)
    assert expected_threshold == settings['high_usage_threshold']
    expected = metrics(forecast, expected_threshold)
    actual = pd.read_csv(folder/'metrics.csv')
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, rtol=1e-12)
    audit = pd.read_csv(folder/'fit_audit.csv', dtype={'com_id': 'string'}, parse_dates=['reported_date', 'train_start', 'train_end'])
    assert (audit.reported_date - audit.train_start).dt.days.eq(audit.history_days).all()
    assert (audit.reported_date - audit.train_end).dt.days.eq(1).all()
    assert audit.loc[audit.status.eq('ok'), 'n_train'].ge(7).all()
    # Refit real validation sample from the raw panel, independent of full feature cache.
    with threadpool_limits(limits=1):
        for days in (30, 90):
            origin = pd.Timestamp('2026-07-10')
            bounded = panel[panel.reported_date.between(origin-timedelta(days=days), origin)]
            frame, _ = build_features(bounded, days)
            train = frame.reported_date.between(origin-timedelta(days=days-7), origin-timedelta(days=1))
            train &= frame.target_valid & frame.recommended_period & frame[FEATURES].notna().all(axis=1)
            sample = forecast[forecast.reported_date.eq(origin) & forecast.history_days.eq(days) & forecast.y_pred.notna()]
            for scope in ('shared', 'device_specific'):
                row = sample[sample.training_scope.eq(scope)].iloc[0]
                use = train if scope == 'shared' else train & frame.com_id.eq(row.com_id)
                future = frame.reported_date.eq(origin) & frame.com_id.eq(row.com_id)
                pred, _ = fit_predict(frame.loc[use, FEATURES], frame.loc[use, 'sessions_target'], frame.loc[future, FEATURES], settings['selected_alpha'][f'{days}_{scope}'])
                np.testing.assert_allclose(row.y_pred, pred[0], rtol=1e-10)
    print(f'Delivery verified: {len(forecast):,} forecasts, actuals, metrics, audit windows, four real-data refits')


if __name__ == '__main__':
    main()
