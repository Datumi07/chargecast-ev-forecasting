"""Daily rolling Poisson experiments. Run from repository root; see README.md, Group A section."""
import argparse
import hashlib
import json
import platform
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from build_features import ROOT, build_features

FEATURES = ['weekday_sin', 'weekday_cos', 'lag_7', 'rolling_mean_7']
ALPHAS = (0.1, 1.0)


def fit_predict(x, y, future, alpha):
    if len(y) < 7:
        return None, 'fewer_than_7_training_rows'
    if not np.any(y):
        # ponytail: zero-only Poisson MLE is a boundary limit; use exact zero.
        return np.zeros(len(future)), 'all_zero_boundary'
    scaler = StandardScaler().fit(x)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model = PoissonRegressor(alpha=alpha, max_iter=300, tol=1e-6)
            model.fit(scaler.transform(x), y)
            prediction = model.predict(scaler.transform(future))
        if not np.isfinite(prediction).all():
            return None, 'nonfinite_prediction'
        return prediction, 'ok'
    except (ValueError, FloatingPointError, ConvergenceWarning) as error:
        return None, type(error).__name__ + ': ' + str(error)


def experiment(frame, days, split, alpha):
    dates = frame.reported_date.to_numpy()
    x = frame[FEATURES].to_numpy(float)
    y = frame.sessions_target.to_numpy(float)
    complete = np.isfinite(x).all(axis=1)
    train_ok = complete & frame.target_valid.to_numpy() & frame.recommended_period.to_numpy() & (dates >= np.datetime64('2026-02-01'))
    outputs, audits = [], []
    for origin in sorted(frame.loc[frame.split.eq(split), 'reported_date'].unique()):
        start = origin - timedelta(days=int(days))
        # First seven rows are context only: every training predictor must also
        # use observations inside this origin's strict calendar history limit.
        train = train_ok & (dates >= (start + timedelta(days=7)).to_datetime64()) & (dates < origin.to_datetime64())
        forecast = np.flatnonzero((dates == origin.to_datetime64()) & frame.target_valid.to_numpy() & frame.recommended_period.to_numpy())
        if not len(forecast):
            continue
        eligible = forecast[complete[forecast] & frame.history_window_available.to_numpy()[forecast]]
        local_groups = frame.loc[train].groupby('com_id', sort=False).indices
        train_indices = np.flatnonzero(train)
        device_indices = {device: train_indices[idx] for device, idx in local_groups.items()}
        for scope in ('device_specific', 'shared'):
            result = frame.iloc[forecast][['com_id', 'reported_date', 'split']].copy()
            result['model'] = 'poisson'
            result['training_scope'] = scope
            result['history_days'] = days
            result['y_true'] = y[forecast]
            result['y_pred'] = np.nan
            result['status'] = 'missing_forecast_features'
            if scope == 'shared':
                pred, status = fit_predict(x[train], y[train], x[eligible], alpha) if len(eligible) else (None, 'no_eligible_forecasts')
                audits.append(dict(reported_date=origin, history_days=days, split=split, training_scope=scope, com_id='ALL', train_start=start, train_end=origin-timedelta(days=1), n_train=int(train.sum()), status=status))
                if pred is not None:
                    result.loc[eligible, 'y_pred'] = pred
                result.loc[eligible, 'status'] = status
            else:
                for idx in eligible:
                    device = frame.at[idx, 'com_id']
                    rows = device_indices.get(device, np.array([], dtype=int))
                    pred, status = fit_predict(x[rows], y[rows], x[idx:idx+1], alpha)
                    if pred is not None:
                        result.loc[idx, 'y_pred'] = pred[0]
                    result.loc[idx, 'status'] = status
                    audits.append(dict(reported_date=origin, history_days=days, split=split, training_scope=scope, com_id=device, train_start=start, train_end=origin-timedelta(days=1), n_train=len(rows), status=status))
            outputs.append(result)
        print(f'{split} {days}d {origin:%Y-%m-%d} alpha={alpha}', flush=True)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits)


def metrics(forecasts, threshold):
    records = []
    keys = ['com_id', 'reported_date']
    for (days, split), group in forecasts.groupby(['history_days', 'split']):
        available = group.pivot(index=keys, columns='training_scope', values='y_pred').notna().all(axis=1)
        common = available[available].index
        for scope, part in group.groupby('training_scope'):
            for population in ('available', 'matched'):
                rows = part[part.y_pred.notna()]
                if population == 'matched':
                    rows = rows.set_index(keys).loc[common].reset_index()
                error = rows.y_pred - rows.y_true
                high = rows.y_true.ge(threshold)
                records.append(dict(history_days=days, split=split, training_scope=scope, population=population, n_total=len(part), n_scored=len(rows), n_excluded=len(part)-len(rows), mae=float(error.abs().mean()), rmse=float(np.sqrt((error**2).mean())), high_usage_threshold=threshold, n_high=int(high.sum()), high_mean_shortfall=float((-error[high]).clip(lower=0).mean()), high_underprediction_rate=float(error[high].lt(0).mean())))
    return pd.DataFrame(records)


def run_experiment(frame, days, split, alpha):
    with threadpool_limits(limits=1):
        return experiment(frame, days, split, alpha)


def write_report(result, summary, folder):
    """Keep narrative and review sample reproducible from the delivered forecasts."""
    sample = result.groupby(['history_days', 'split', 'training_scope'], sort=True).head(5)
    sample.to_csv(folder/'sample_forecasts.csv', index=False)
    matched = summary[summary.population.eq('matched')]
    lines = ['GROUP A - POISSON REGRESSION', '',
             'Daily refits; 30/90 calendar-day bounds; identical features and forecast dates.',
             'All four alpha settings selected on validation before running test.',
             'Read README.md (Group A section) for commands, features and exclusions.', '',
             'MATCHED-ROW RESULTS (sessions/day)', matched.to_string(index=False), '',
             'DOES SHARING HELP?']
    for (days, split), group in matched.groupby(['history_days', 'split']):
        scores = group.set_index('training_scope')
        local, shared = scores.loc['device_specific'], scores.loc['shared']
        gain = 100 * (local.mae - shared.mae) / local.mae
        lines.append(f'{split} {days}d: shared MAE {shared.mae:.6f}, local {local.mae:.6f}; '
                     f'shared MAE reduction {gain:.2f}%; matched rows {int(shared.n_scored):,}.')
    lines += ['', 'ERROR PATTERNS']
    available = result[result.y_pred.notna()].copy()
    available['absolute_error'] = (available.y_pred - available.y_true).abs()
    available['signed_error'] = available.y_pred - available.y_true
    for (split, days, scope), part in available.groupby(['split', 'history_days', 'training_scope']):
        zero = part[part.y_true.eq(0)]
        lines.append(f'{split} {days}d {scope}: median absolute error {part.absolute_error.median():.4f}; '
                     f'max forecast {part.y_pred.max():.6g}; zero-day mean forecast {zero.y_pred.mean():.4f}; '
                     f'overall signed error {part.signed_error.mean():.4f}. '
                     'See high-day shortfall/rate above for busy-day underprediction.')
    lines += ['Short-window local models can extrapolate to extreme finite counts through the exponential link.',
              'Compare median errors with MAE/RMSE and inspect the largest-error rows below. Extreme outputs are retained,',
              'not clipped after evaluation. Small per-device samples/feature variation are a likely source of instability.',
              'Any broader alpha grid, transformed predictors or constrained predictions require a new validation protocol.',
              '', 'LARGEST ABSOLUTE ERRORS (available rows; descriptive, not tuning)',
              available.nlargest(8, 'absolute_error')[['com_id', 'reported_date', 'split', 'history_days', 'training_scope', 'y_true', 'y_pred', 'absolute_error']].to_string(index=False), '',
              'COVERAGE / FAILED FITS',
              result.groupby(['history_days', 'split', 'training_scope', 'status']).size().rename('rows').to_string(), '',
              'LIMITATIONS / HANDOFF',
              'All-zero histories use the exact-zero Poisson boundary limit, logged separately from fitted models.',
              'Shared model pools four predictors without device fixed effects; conclusions apply to this specification.',
              'High usage is the frozen pooled training 90th percentile, not a test-derived threshold.',
              'Retrospective coverage flags and assumed next-day label availability limit operational interpretation.',
              'Earlier seven window days are context only; usable training-label spans are at most 23/83 days.',
              'Masked targets are excluded before forecast export; missing predictions remain in forecasts.csv.',
              'No baseline or LightGBM comparison claimed; those runs belong to the other assigned owners.',
              'Automated reproducibility checks are not independent teammate reproduction.',
              'Pending human steps: Group B validation cross-check, authentic member commits, branch/PR sharing.',
              'No GitHub push performed.']
    (folder/'results.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/daily_panel.csv.gz')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/group_a')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.input, dtype={'com_id': 'string'}, parse_dates=['reported_date'])
    threshold = float(panel.loc[panel.split.eq('train') & panel.target_valid & panel.reported_date.ge('2026-02-01'), 'sessions_target'].quantile(.9))
    selected, forecasts, audits, trials, frames = {}, [], [], [], {}
    frames = {days: build_features(panel, days)[0] for days in (30, 90)}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        validation_jobs = {(days, alpha): pool.submit(run_experiment, frame, days, 'validation', alpha)
                           for days, frame in frames.items() for alpha in ALPHAS}
        for days, frame in frames.items():
            candidates = []
            for alpha in ALPHAS:
                pred, audit = validation_jobs[days, alpha].result()
                pred.to_csv(args.output/f'validation_{days}d_alpha{alpha}_forecasts.csv', index=False)
                score = metrics(pred, threshold)
                score['alpha'] = alpha
                trials.append(score)
                candidates.append((alpha, pred, audit, score))
            for scope in ('device_specific', 'shared'):
                def objective(candidate):
                    scores = candidate[3]
                    row = scores[scores.training_scope.eq(scope) & scores.population.eq('matched')].iloc[0]
                    return row.mae if row.n_scored else float('inf')
                best = min(candidates, key=objective)
                if not np.isfinite(objective(best)):
                    raise RuntimeError('No matched validation predictions')
                selected[f'{days}_{scope}'] = best[0]
                forecasts.append(best[1][best[1].training_scope.eq(scope)])
                audits.append(best[2][best[2].training_scope.eq(scope)])
            # Lock settings on disk before reading/scoring final test predictions.
            (args.output/'settings.json').write_text(json.dumps(dict(selected_alpha=selected, features=FEATURES, alpha_candidates=ALPHAS, refit='daily', min_training_rows=7, training_context_days=7, high_usage_threshold=threshold, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__), indent=2)+'\n')
        if not args.validation_only:
            test_jobs = {(days, alpha): pool.submit(run_experiment, frame, days, 'test', alpha)
                         for days, frame in frames.items()
                         for alpha in {selected[f'{days}_{scope}'] for scope in ('device_specific', 'shared')}}
            for days, frame in frames.items():
                for alpha in sorted({selected[f'{days}_{scope}'] for scope in ('device_specific', 'shared')}):
                    pred, audit = test_jobs[days, alpha].result()
                    scopes = [scope for scope in ('device_specific', 'shared') if selected[f'{days}_{scope}'] == alpha]
                    forecasts.append(pred[pred.training_scope.isin(scopes)])
                    audits.append(audit[audit.training_scope.isin(scopes)])
    result = pd.concat(forecasts, ignore_index=True).sort_values(['split', 'history_days', 'training_scope', 'com_id', 'reported_date'])
    result.to_csv(args.output/'forecasts.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(args.output/'fit_audit.csv', index=False)
    pd.concat(trials, ignore_index=True).to_csv(args.output/'validation_trials.csv', index=False)
    summary = metrics(result, threshold)
    summary.to_csv(args.output/'metrics.csv', index=False)
    result.groupby(['history_days', 'split', 'training_scope', 'status'], dropna=False).size().rename('rows').to_csv(args.output/'coverage.csv')
    panel.assign(split=panel.split.where(panel.reported_date.ge('2026-02-01'), 'history_context')).groupby('split').agg(calendar_rows=('target_valid', 'size'), valid_targets=('target_valid', 'sum')).assign(masked_targets=lambda table: table.calendar_rows-table.valid_targets).to_csv(args.output/'input_coverage.csv')
    write_report(result, summary, args.output)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
