from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    return s.astype('string').str.strip().str.upper().str.replace(
        '\\.0$',
        '',
        regex=True
    ).replace('', pd.NA)


def col(s):
    return re.sub('[^a-z0-9]+', '_', str(s).lower()).strip('_')


def duration(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (datetime.timedelta, pd.Timedelta)):
        return v.total_seconds() / 3600
    if isinstance(v, datetime.time):
        return v.hour + v.minute / 60 + v.second / 3600
    if isinstance(v, (datetime.datetime, pd.Timestamp)):
        from openpyxl.utils.datetime import to_excel
        return to_excel(v) * 24
    if isinstance(v, (int, float)):
        return v * 24
    try:
        return pd.to_timedelta(v).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(ROOT / 'data/raw'))
    args = ap.parse_args()

    raw = Path(args.raw)
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    audit = ROOT / 'reports/data_quality'
    audit.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / 'config/input_manifest.json').read_text())
    parts = []
    regs = []
    ingest = []

    for m in manifest:
        p = raw / m['file']
        d = pd.read_excel(p)
        d.columns = [col(c) for c in d.columns]
        d = d.rename(columns={
            'charging_point_id': 'com_id',
            'consum': 'energy_kwh',
            'reliability': 'reliability_pct',
            'local_authority_area': 'local_authority'
        })
        n = len(d)
        d['source_row'] = np.arange(2, n + 2)
        d['source_file'] = m['file']
        d['report_month'] = m['month']
        d['com_id'] = norm(d.com_id)
        if m['kind'] == 'sessions':
            d['start_time'] = pd.to_datetime(d.start_time, errors='coerce')
            bad = d.com_id.isna() | d.start_time.isna()
            d = d.loc[~bad].copy()
            # September/October duration resides in an unnamed column.
            dc = next(
                (c for c in ['duration_time', 'duration', 'unnamed_7'] if c in d),
                None
            )
            d['duration_raw'] = d[dc].astype('string') if dc else pd.NA
            d['duration_hours_raw'] = d[dc].map(duration) if dc else np.nan
            cc = next((c for c in ['connector_id', 'connector'] if c in d), None)
            d['connector_id'] = norm(d[cc]) if cc else pd.NA
            for c in ['energy_kwh', 'amount']:
                d[c] = pd.to_numeric(d[c], errors='coerce')
            for c in ['site_name', 'charging_type', 'currency']:
                if c not in d:
                    d[c] = pd.NA
                d[c] = d[c].astype('string').str.strip()
            keep = [
                'com_id',
                'start_time',
                'site_name',
                'charging_type',
                'connector_id',
                'currency',
                'energy_kwh',
                'amount',
                'duration_raw',
                'duration_hours_raw',
                'source_file',
                'source_row',
                'report_month'
            ]
            parts.append(d[keep])
            removed = int(bad.sum())
        else:
            d['cpid'] = norm(d.cpid)
            v = pd.to_numeric(d.reliability_pct, errors='coerce')
            # All 13 supplied reports store 0..100 percentages (verified per file).
            if not v.dropna().between(0, 100).all():
                raise ValueError('Unexpected uptime scale: ' + m['file'])
            d['reliability_pct'] = v
            d = d.dropna(subset=['com_id'])
            regs.append(d[[
                'com_id',
                'cpid',
                'report_month',
                'reliability_pct',
                'local_authority',
                'host_owner',
                'manufacturer',
                'source_file'
            ]])
            removed = n - len(d)
        ingest.append({
            **m,
            'raw_rows': n,
            'excluded_missing_key_or_time': removed,
            'retained_rows': len(d),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        })
        print(m['file'], len(d), flush=True)

    s = pd.concat(parts, ignore_index=True)
    r = pd.concat(regs, ignore_index=True)
    assert not r.duplicated(['com_id', 'report_month']).any()

    keys = [
        'com_id',
        'start_time',
        'site_name',
        'charging_type',
        'connector_id',
        'currency',
        'energy_kwh',
        'amount',
        'duration_raw'
    ]
    s['duplicate_candidate'] = s.duplicated(keys, keep=False)
    dup = s.duplicated(keys, keep='first')
    s.loc[dup].to_csv(audit / 'removed_duplicate_rows.csv', index=False)
    before = len(s)
    s = s.loc[~dup].copy()

    s['energy_invalid'] = s.energy_kwh.isna() | s.energy_kwh.lt(0)
    # Review threshold, not physical truth.
    s['energy_extreme_review'] = s.energy_kwh.gt(500)
    s['duration_invalid'] = (
        s.duration_hours_raw.isna()
        | s.duration_hours_raw.lt(0)
        | s.duration_hours_raw.gt(24 * 365)
    )
    s['duration_long_review'] = s.duration_hours_raw.gt(48)
    s['energy_kwh_clean'] = (
        s.energy_kwh.mask(s.energy_invalid
        | s.energy_extreme_review)
    )
    s['duration_hours_clean'] = s.duration_hours_raw.mask(s.duration_invalid)
    s['reported_date'] = s.start_time.dt.normalize()
    s['report_month_mismatch'] = s.start_time.dt.strftime('%Y-%m').ne(s.report_month)

    months = sorted(r.report_month.unique())
    sets = [set(r.loc[r.report_month.eq(m), 'com_id']) for m in months]
    common = set.intersection(*sets)
    s['stable_registry_cohort'] = s.com_id.isin(common)
    registry = r[r.com_id.isin(common)].copy()
    registry.to_csv(out / 'registry_monthly.csv', index=False)

    summary = s[s.stable_registry_cohort].groupby('com_id').agg(
        session_rows=('com_id', 'size'),
        months_with_sessions=('report_month', 'nunique'),
        first_record=('start_time', 'min'),
        last_record=('start_time', 'max')
    )
    cohort = pd.DataFrame({'com_id': sorted(common)}).join(summary, on='com_id')
    cohort['session_rows'] = cohort.session_rows.fillna(0).astype(int)
    cohort['months_with_sessions'] = cohort.months_with_sessions.fillna(0).astype(int)
    cohort['cohort_status'] = np.where(
        cohort.session_rows.gt(0),
        'observed_in_sessions',
        'no_matching_sessions_review'
    )
    cohort.to_csv(audit / 'cohort_audit.csv', index=False)

    ids = cohort.loc[cohort.session_rows.gt(0), 'com_id']
    g = s[s.com_id.isin(ids)]

    dates = pd.date_range('2025-09-01', '2026-09-30')
    panel = pd.MultiIndex.from_product(
        [ids, dates],
        names=['com_id', 'reported_date']
    ).to_frame(index=False)

    counts = g.groupby([
        'com_id',
        'reported_date'
    ]).size().rename('observed_session_rows').reset_index()
    panel = panel.merge(counts, how='left', validate='one_to_one')
    panel['observed_session_rows'] = panel.observed_session_rows.fillna(0).astype(int)
    panel['report_month'] = panel.reported_date.dt.strftime('%Y-%m')

    activity = g.groupby([
        'com_id',
        'report_month'
    ]).size().rename('month_session_rows').reset_index()
    panel = panel.merge(activity, how='left', validate='many_to_one')
    panel.month_session_rows = panel.month_session_rows.fillna(0).astype(int)
    panel['coverage_status'] = np.where(
        panel.month_session_rows.eq(0),
        'unknown_whole_month',
        'assumed_observed'
    )

    # Conservative boundary masks: source timezones unverified, possible +/-1h day shifts.
    boundary = panel.reported_date.dt.is_month_start | panel.reported_date.dt.is_month_end | panel.reported_date.isin(pd.to_datetime([
        '2025-10-26',
        '2026-03-29'
    ]))
    panel.loc[boundary, 'coverage_status'] = 'uncertain_time_boundary'
    panel.loc[
        panel.reported_date.between('2026-08-01', '2026-08-02'),
        'coverage_status'
    ] = 'known_network_gap'
    panel['target_valid'] = panel.coverage_status.eq('assumed_observed')
    panel['sessions_target'] = panel.observed_session_rows.where(panel.target_valid).astype('Int64')
    panel['zero_is_assumed'] = panel.target_valid & panel.observed_session_rows.eq(0)

    panel['split'] = np.select(
        [panel.reported_date.lt('2026-07-01'), panel.reported_date.lt('2026-09-01')],
        ['train', 'validation'],
        default='test'
    )
    panel['recommended_period'] = panel.reported_date.ge('2026-02-01')

    # Monthly metadata intentionally NOT joined as predictive features.
    panel.to_csv(out / 'daily_panel.csv.gz', index=False)
    s.to_csv(
        out / 'sessions_clean.csv.gz',
        index=False,
        compression={'method': 'gzip', 'compresslevel': 1}
    )
    s[~s.stable_registry_cohort].groupby('com_id').size().rename('session_rows').to_csv(audit / 'excluded_cohort_ids.csv')

    unmatched = s[~s.com_id.isin(set(r.com_id))].groupby([
        'com_id',
        'report_month'
    ]).size().rename('session_rows')
    unmatched.to_csv(audit / 'unmatched_ids.csv')
    pd.DataFrame(ingest).to_csv(audit / 'ingestion_audit.csv', index=False)

    stats = {
        'input_valid_session_rows': before,
        'duplicate_copies_removed': int(dup.sum()),
        'clean_session_rows': len(s),
        'stable_registry_ids': len(common),
        'stable_ids_with_sessions': len(ids),
        'stable_session_rows': len(g),
        'daily_rows': len(panel),
        'valid_targets': int(panel.target_valid.sum()),
        'assumed_zero_targets': int(panel.zero_is_assumed.sum()),
        'coverage_status': panel.coverage_status.value_counts().to_dict(),
        'valid_targets_by_split': panel[panel.target_valid].groupby('split').size().to_dict()
    }

    assert len(panel) == len(ids) * len(dates)
    assert not panel.duplicated(['com_id', 'reported_date']).any()
    assert panel.loc[~panel.target_valid, 'sessions_target'].isna().all()
    assert panel.sessions_target.dropna().ge(0).all()
    assert panel.observed_session_rows.sum() == len(g[g.reported_date.between(
        dates.min(),
        dates.max()
    )])
    assert not s.duplicated(keys).any()

    stats['validation_checks'] = 'passed: unique keys, continuous calendar, invalid labels masked, nonnegative counts, session/day reconciliation, duplicate rule'
    (audit / 'audit_summary.json').write_text(json.dumps(stats, indent=2))

    quality = {k: int(s[k].sum()) for k in [
        'energy_invalid',
        'energy_extreme_review',
        'duration_invalid',
        'duration_long_review',
        'report_month_mismatch'
    ]}
    (audit / 'session_quality_summary.json').write_text(json.dumps(quality, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
