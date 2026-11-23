"""Independent ChargeCast Model B. Validation only; no repository writes."""
from pathlib import Path
import argparse, hashlib, json, platform, time, urllib.request
import numpy as np
import pandas as pd
import lightgbm as lgb

COMMIT = '90bbc04b9cde3e168ff347de0875df0c973a15e7'
SHA256 = 'bb8bf9c4623ba71e1c1455093491af6bbacd8a9709496ed86c6fa05227a00dd5'
URL = f'https://raw.githubusercontent.com/Datumi07/chargecast-ev-forecasting/{COMMIT}/data/processed/daily_panel.csv.gz'
CANDIDATES = [dict(name=f'{size}_{"id" if use_id else "no_id"}', use_id=use_id, n_estimators=trees, num_leaves=leaves, min_child_samples=minimum)
              for size, trees, leaves, minimum in [('small',80,15,50),('medium',140,31,100)] for use_id in [False,True]]

def load_panel(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(URL,path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA256, 'Unexpected dataset version'
    d=pd.read_csv(path,dtype={'com_id':'string'},parse_dates=['reported_date'])
    assert not d.duplicated(['com_id','reported_date']).any()
    assert d.loc[~d.target_valid,'sessions_target'].isna().all()
    assert d.loc[d.target_valid,'sessions_target'].notna().all()
    # Test runner: labels loaded for scoring; features still use only history before origin.
    d=d[d.reported_date.lt('2026-10-01')].copy()
    wide=d.pivot(index='reported_date',columns='com_id',values='sessions_target').sort_index()
    assert wide.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
    return d,wide

def features(history,dates,ids):
    """Input is ONLY the allowed history. Last row is forecast origin, no target.
    Row-major order: dates, then devices. Rolling std uses ddof=0.
    """
    h=np.asarray(history,dtype=float); n,m=h.shape
    z=np.vstack([h,np.full((1,m),np.nan)])
    f={'weekday':np.repeat(pd.DatetimeIndex(dates).dayofweek,m),
       'weekend':np.repeat(pd.DatetimeIndex(dates).dayofweek>=5,m)}
    for lag in [1,7,14]:
        a=np.full_like(z,np.nan)
        if lag<len(z):a[lag:]=z[:-lag]
        f[f'lag_{lag}']=a.ravel()
    sums=np.vstack([np.zeros(m),np.cumsum(np.nan_to_num(z),axis=0)])
    counts=np.vstack([np.zeros(m),np.cumsum(np.isfinite(z),axis=0)])
    squares=np.vstack([np.zeros(m),np.cumsum(np.nan_to_num(z)**2,axis=0)])
    end=np.arange(n+1)
    for w in [7,14]:
        start=np.maximum(0,end-w); c=counts[end]-counts[start]
        s=sums[end]-sums[start]; ss=squares[end]-squares[start]
        mean=np.divide(s,c,out=np.full_like(s,np.nan),where=c>0)
        var=np.divide(ss,c,out=np.full_like(ss,np.nan),where=c>0)-mean**2
        f[f'mean_{w}']=mean.ravel(); f[f'std_{w}']=np.sqrt(np.maximum(var,0)).ravel(); f[f'count_{w}']=c.ravel()
    x=pd.DataFrame(f)
    x['device']=pd.Categorical(np.tile(ids,n+1),categories=ids)
    return x

def score(g):
    ok=g.y_true.notna() & g.y_pred.notna(); q=g[ok]
    err=q.y_pred-q.y_true
    busy=q[q.y_true.ge(q.high_threshold)]
    return dict(n_eligible=len(g),n_scored=len(q),n_excluded=len(g)-len(q),
        MAE=float(err.abs().mean()),RMSE=float(np.sqrt((err**2).mean())),
        n_busy=len(busy),busy_underprediction=float((busy.y_true-busy.y_pred).clip(lower=0).mean()),
        busy_underprediction_rate=float((busy.y_pred<busy.y_true).mean()) if len(busy) else None)

def run(data='data/daily_panel.csv.gz',output='outputs',smoke=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    d,wide=load_panel(data);ids=wide.columns.tolist();m=len(ids)
    dates=pd.date_range('2026-07-01','2026-08-31')
    if smoke:dates=dates[:3]
    metadata=dict(commit=COMMIT,data_sha256=SHA256,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,refit='daily',minimum_history=7,rolling_std_ddof=0,seed=2010,smoke=smoke,candidates=CANDIDATES,test_run=False)
    manifest=out/'run_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==metadata, 'Output configuration differs; choose a new output folder.'
    manifest.write_text(json.dumps(metadata,indent=2))
    for w in [30,90]:
        for day in dates:
            dest=out/f'checkpoint_{w}_{day:%Y%m%d}.csv.gz'
            if dest.exists():continue
            t=time.time(); start=day-pd.Timedelta(days=w)
            h=wide.loc[start:day-pd.Timedelta(days=1)].to_numpy()
            assert len(h)==w and start>=pd.Timestamp('2026-02-01')
            x=features(h,pd.date_range(start,day),ids)
            y=h.ravel(); fit=np.isfinite(y)
            train=x.iloc[:w*m];pred=x.iloc[w*m:]
            counts=np.isfinite(h).sum(axis=0);eligible=counts>=7
            truth=wide.loc[day].to_numpy()
            threshold=max(1.0,float(np.quantile(y[fit],.9)))
            base=pd.DataFrame(dict(com_id=ids,reported_date=day,split='validation',training_scope='shared',history_days=w,y_true=truth,history_valid_days=counts,high_threshold=threshold))
            parts=[]
            for cfg in CANDIDATES:
                columns=[c for c in x.columns if cfg['use_id'] or c!='device']
                model=lgb.LGBMRegressor(objective='poisson',learning_rate=.05,random_state=2010,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True,**{k:cfg[k] for k in ['n_estimators','num_leaves','min_child_samples']})
                model.fit(train.loc[fit,columns],y[fit])
                v=model.predict(pred[columns]);assert np.isfinite(v).all() and (v>=0).all()
                q=base.copy();q['model']='lightgbm';q['candidate']=cfg['name'];q['y_pred']=np.where(eligible,v,np.nan)
                q['unavailable_reason']=np.where(eligible,'','fewer_than_7_valid_history_days');parts.append(q)
            for name,v in [('same_weekday_last_week',h[-7]),('seven_day_average',pred.mean_7.to_numpy())]:
                q=base.copy();q['model']=name;q['candidate']=name;q['training_scope']='device_specific';q['y_pred']=np.where(eligible,v,np.nan)
                q['unavailable_reason']=np.where(~eligible,'fewer_than_7_valid_history_days',np.where(np.isfinite(v),'','missing_baseline_history'));parts.append(q)
            pd.concat(parts,ignore_index=True).to_csv(dest,index=False)
            print(f'{w}d {day.date()} completed in {time.time()-t:.1f}s',flush=True)
    frames=[pd.read_csv(out/f'checkpoint_{w}_{day:%Y%m%d}.csv.gz',dtype={'com_id':'string'}) for w in [30,90] for day in dates]
    allp=pd.concat(frames,ignore_index=True)
    valid=allp[allp.y_true.notna()]
    metrics=[]
    for (w,c),g in valid.groupby(['history_days','candidate']):metrics.append(dict(history_days=w,candidate=c,**score(g)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'validation_metrics_all_candidates.csv',index=False)
    chosen={}
    for w in [30,90]:
        table=metrics[metrics.history_days.eq(w)&metrics.candidate.isin([c['name'] for c in CANDIDATES])]
        chosen[w]=table.sort_values(['MAE','RMSE','candidate']).iloc[0]['candidate']
    (out/'selected_settings.json').write_text(json.dumps({str(w):next(c for c in CANDIDATES if c['name']==name) for w,name in chosen.items()},indent=2))
    selected=allp[allp['candidate'].eq(allp['history_days'].map(chosen))].copy()
    required=['com_id','reported_date','split','model','training_scope','history_days','y_true','y_pred']
    selected[required+[c for c in selected if c not in required]].to_csv(out/'forecasts_validation.csv',index=False)
    comparisons=[];matched_keys=[]
    for w in [30,90]:
        q=valid[valid.history_days.eq(w)&valid.candidate.isin([chosen[w],'same_weekday_last_week','seven_day_average'])]
        counts=q[q.y_pred.notna()].groupby(['com_id','reported_date']).size();keys=counts[counts.eq(3)].index
        match=q.set_index(['com_id','reported_date']).loc[keys].reset_index()
        matched_keys.append(match[['com_id','reported_date','history_days']].drop_duplicates())
        for name,g in match.groupby('candidate'):comparisons.append(dict(history_days=w,candidate=name,**score(g)))
    pd.DataFrame(comparisons).to_csv(out/'matched_validation_metrics.csv',index=False)
    pd.concat(matched_keys).to_csv(out/'matched_validation_keys.csv',index=False)
    valid_selected=selected[selected.y_true.notna()&selected.y_pred.notna()].copy()
    valid_selected['underprediction']=valid_selected.y_true-valid_selected.y_pred
    valid_selected[valid_selected.y_true.ge(valid_selected.high_threshold)].sort_values('underprediction',ascending=False).head(30).to_csv(out/'busy_day_examples.csv',index=False)
    print(metrics.to_string(index=False));print('Selected:',chosen,flush=True)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/daily_panel.csv.gz');p.add_argument('--output',default='outputs');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.data,a.output,a.smoke)
