"""ChargeCast v2: July-only selection, August stability check, NO September test."""
from pathlib import Path
import json,hashlib,time,argparse,platform
import numpy as np
import pandas as pd
import lightgbm as lgb
from base_v1 import load_panel,features as base_features,score,COMMIT,SHA256

SEED=2010
ANCHOR=dict(group='base',min_train_history=0,objective='poisson',n_estimators=140,learning_rate=.05,num_leaves=31,min_child_samples=100,reg_lambda=0.0)
GROUP_COLS={'weekly':['same_weekday_mean','same_weekday_count'], 'trend':['previous_mean7','trend7'], 'activity':['past_count','past_mean','past_zero_fraction','days_since_positive','has_past_positive']}

def make_features(h,dates,ids):
    x=base_features(h,dates,ids);h=np.asarray(h,float);n,m=h.shape
    z=np.vstack([h,np.full((1,m),np.nan)]);valid=np.isfinite(z)
    count=np.vstack([np.zeros(m),np.cumsum(valid,axis=0)])[:-1]
    sums=np.vstack([np.zeros(m),np.cumsum(np.nan_to_num(z),axis=0)])[:-1]
    zeros=np.vstack([np.zeros(m),np.cumsum(z==0,axis=0)])[:-1]
    x['past_count']=count.ravel()
    x['past_mean']=np.divide(sums,count,out=np.full_like(sums,np.nan),where=count>0).ravel()
    x['past_zero_fraction']=np.divide(zeros,count,out=np.full_like(zeros,np.nan),where=count>0).ravel()
    index=np.arange(n+1)[:,None]
    last=np.maximum.accumulate(np.where(z>0,index,-1),axis=0)
    last=np.vstack([np.full(m,-1),last[:-1]])
    x['days_since_positive']=np.where(last>=0,index-last,np.nan).ravel()
    x['has_past_positive']=(last>=0).ravel()
    wc=np.zeros_like(z);ws=np.zeros_like(z)
    for lag in [7,14,21,28]:
        if lag<len(z):
            wc[lag:]+=np.isfinite(z[:-lag]);ws[lag:]+=np.nan_to_num(z[:-lag])
    x['same_weekday_mean']=np.divide(ws,wc,out=np.full_like(ws,np.nan),where=wc>0).ravel()
    x['same_weekday_count']=wc.ravel()
    prev=np.full_like(z,np.nan)
    # Previous-week interval [t-14, t-7), clipped to allowed history.
    for t in range(n+1):
        a=z[max(0,t-14):max(0,t-7)];c=np.isfinite(a).sum(axis=0)
        prev[t]=np.divide(np.nansum(a,axis=0),c,out=np.full(m,np.nan),where=c>0)
    x['previous_mean7']=prev.ravel();x['trend7']=x.mean_7-x.previous_mean7
    return x

def columns_for(x,group):
    extras=sum(GROUP_COLS.values(),[])
    cols=[c for c in x if c not in extras]
    if group=='all':return cols+extras
    if group!='base':cols+=GROUP_COLS[group]
    return cols

def config_id(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]

def evaluate_many(w,configs,dates,wide,out,stage):
    ids=wide.columns.tolist();m=len(ids); records=[]
    for day in dates:
        pending=[]
        for cfg in configs:
            cid=config_id(cfg);folder=out/'cache'/str(w)/cid;folder.mkdir(parents=True,exist_ok=True)
            cfgpath=folder/'config.json'
            if cfgpath.exists():assert json.loads(cfgpath.read_text())==cfg
            else:cfgpath.write_text(json.dumps(cfg,indent=2))
            if not (folder/f'{day:%Y%m%d}.csv.gz').exists():pending.append((cfg,folder))
        if not pending:continue
        start=day-pd.Timedelta(days=w);h=wide.loc[start:day-pd.Timedelta(days=1)].to_numpy()
        assert len(h)==w and start>=pd.Timestamp('2026-02-01') and pd.Timestamp('2026-09-01')<=day<pd.Timestamp('2026-10-01')
        x=make_features(h,pd.date_range(start,day),ids);y=h.ravel();train=x.iloc[:w*m];predict=x.iloc[w*m:]
        threshold=max(1,float(np.quantile(y[np.isfinite(y)],.9)));count=np.isfinite(h).sum(axis=0)
        base=pd.DataFrame(dict(com_id=ids,reported_date=day.strftime('%Y-%m-%d'),split='test',model='lightgbm',training_scope='shared',history_days=w,y_true=wide.loc[day].to_numpy(),history_valid_days=count,high_threshold=threshold))
        t=time.time()
        for cfg,folder in pending:
            mask=np.isfinite(y)&(train.past_count.to_numpy()>=cfg['min_train_history'])
            cols=columns_for(x,cfg['group'])
            model=lgb.LGBMRegressor(**{k:v for k,v in cfg.items() if k not in ['group','min_train_history']},random_state=SEED,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(train.loc[mask,cols],y[mask])
            raw=model.predict(predict[cols]);v=np.maximum(raw,0)
            assert np.isfinite(v).all()
            q=base.copy();q['y_pred']=np.where(count>=7,v,np.nan);q['config_id']=config_id(cfg)
            q['unavailable_reason']=np.where(count>=7,'','fewer_than_7_valid_history_days')
            q['training_rows']=int(mask.sum());q['clipped_negative']=raw<0
            dest=folder/f'{day:%Y%m%d}.csv.gz';tmp=dest.with_suffix('.tmp')
            q.to_csv(tmp,index=False,compression='gzip');tmp.replace(dest)
            # Final validation-origin snapshots; not a live or September model.
            if day==pd.Timestamp('2026-09-30'):
                model.booster_.save_model(str(folder/'model_20260930.txt'))
                (folder/'feature_columns.json').write_text(json.dumps(cols))
        print(f'{stage} W={w} {day.date()} {len(pending)} fits {time.time()-t:.1f}s',flush=True)
    for cfg in configs:
        q=read_forecasts(out,w,cfg,dates)
        valid=q[q.y_true.notna()]
        records.append(dict(stage=stage,history_days=w,config_id=config_id(cfg),**cfg,**score(valid)))
    return pd.DataFrame(records)

def read_forecasts(out,w,cfg,dates):
    folder=out/'cache'/str(w)/config_id(cfg)
    return pd.concat([pd.read_csv(folder/f'{day:%Y%m%d}.csv.gz',dtype={'com_id':'string'}) for day in dates],ignore_index=True)

def choose(metrics,configs):
    # All configurations have identical forecast eligibility; only train selection differs.
    assert metrics.n_scored.nunique()==1
    win=metrics.sort_values(['MAE','RMSE','config_id']).iloc[0].config_id
    return next(c for c in configs if config_id(c)==win)

def run(data,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_commit=COMMIT,data_sha256=SHA256,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_code_sha256=hashlib.sha256(Path(__file__).with_name('base_v1.py').read_bytes()).hexdigest(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,python=platform.python_version(),selection_month='2026-07',stability_month='2026-08',test_run=False,weighted_training=False)
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Different code/environment; choose a new output directory.'
    mp.write_text(json.dumps(manifest,indent=2))
    _,wide=load_panel(data);july=pd.date_range('2026-07-01','2026-07-31');aug=pd.date_range('2026-08-01','2026-09-30')
    logs=[];locked={};stages={}
    # Finish ALL selections using July before starting ANY August candidate evaluation.
    for w in [30,90]:
        cfgs=[dict(ANCHOR,group=g) for g in ['base','weekly','trend','activity','all']]
        met=evaluate_many(w,cfgs,july,wide,out,'features');logs.append(met);best_feature=choose(met,cfgs)
        stages[w]={'original':ANCHOR.copy(),'features':best_feature.copy()}
        pd.concat(logs).to_csv(out/'july_trials.csv',index=False)
        cfgs=[best_feature,dict(best_feature,min_train_history=7)]
        met=evaluate_many(w,cfgs,july,wide,out,'train_history');logs.append(met);best_history=choose(met,cfgs)
        stages[w]['train_history']=best_history.copy()
        pd.concat(logs).to_csv(out/'july_trials.csv',index=False)
        # Each comparison has a stated parent; no August-based adaptation.
        cfgs=[best_history,dict(best_history,n_estimators=280),dict(best_history,n_estimators=280,learning_rate=.025),dict(best_history,num_leaves=15),dict(best_history,min_child_samples=50),dict(best_history,reg_lambda=5.0),dict(best_history,objective='regression_l1')]
        met=evaluate_many(w,cfgs,july,wide,out,'tuning');logs.append(met);locked[w]=choose(met,cfgs)
        stages[w]['final']=locked[w].copy()
        pd.concat(logs).to_csv(out/'july_trials.csv',index=False)
        print('JULY CHOICE',w,json.dumps(locked[w]),flush=True)
    (out/'locked_july_settings.json').write_text(json.dumps(locked,indent=2))
    (out/'stage_settings.json').write_text(json.dumps(stages,indent=2))
    for w in [30,90]:
        unique={config_id(c):c for c in stages[w].values()}
        met=evaluate_many(w,list(unique.values()),aug,wide,out,'august_stability')
        met.to_csv(out/f'august_metrics_{w}.csv',index=False)
    selected=[]
    for w,cfg in locked.items():selected.append(read_forecasts(out,w,cfg,july.append(aug)))
    selected=pd.concat(selected,ignore_index=True)
    required=['com_id','reported_date','split','model','training_scope','history_days','y_true','y_pred']
    selected[required+[c for c in selected if c not in required]].to_csv(out/'forecasts_validation_v2.csv',index=False)
    print('Complete. July settings frozen; August only for stability. No September evaluation.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/daily_panel.csv.gz');p.add_argument('--output',default='outputs');a=p.parse_args();run(a.data,a.output)
