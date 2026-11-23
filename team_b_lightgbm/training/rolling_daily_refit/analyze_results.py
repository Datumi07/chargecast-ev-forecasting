"""Recompute fair baseline comparisons from saved validation forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment import load_panel,read_forecasts,score

def baseline_forecasts(w,dates,wide):
    ids=wide.columns.tolist();rows=[]
    for day in dates:
        h=wide.loc[day-pd.Timedelta(days=w):day-pd.Timedelta(days=1)].to_numpy();count=np.isfinite(h).sum(0)
        c7=np.isfinite(h[-7:]).sum(0);mean7=np.divide(np.nansum(h[-7:],axis=0),c7,out=np.full(len(ids),np.nan),where=c7>0)
        for name,pred in [('mean7',mean7),('lag7',h[-7])]:
            rows.append(pd.DataFrame(dict(com_id=ids,reported_date=day.strftime('%Y-%m-%d'),history_days=w,model=name,y_true=wide.loc[day].to_numpy(),y_pred=np.where(count>=7,pred,np.nan),high_threshold=max(1,float(np.nanquantile(h,.9))))))
    return pd.concat(rows,ignore_index=True)

def summarize(data,output):
    out=Path(output);_,wide=load_panel(data);dates=pd.date_range('2026-07-01','2026-08-31')
    settings=json.loads((out/'stage_settings.json').read_text());all_metrics=[];groups=[];matched_keys=[];all_rows=[];progress=[]
    for w in [30,90]:
        baseline=baseline_forecasts(w,dates,wide)
        common=baseline[baseline.y_true.notna()&baseline.y_pred.notna()].groupby(['com_id','reported_date']).size();common=common[common.eq(2)].reset_index()[['com_id','reported_date']]
        common['history_days']=w;matched_keys.append(common)
        frames=[baseline]
        for label,cfg in settings[str(w)].items():
            q=read_forecasts(out,w,cfg,dates);q['model']=label;frames.append(q)
            progress.append(dict(history_days=w,stage=label,**cfg,config_id=q.config_id.iloc[0]))
        combined=pd.concat(frames,ignore_index=True)
        all_rows.append(combined)
        for scope,z in [('all_available',combined),('matched_baselines',combined.merge(common,on=['com_id','reported_date','history_days'],validate='many_to_one'))]:
            for month in ['2026-07','2026-08','all']:
                q=z if month=='all' else z[z.reported_date.str.startswith(month)]
                for label,g in q[q.y_true.notna()].groupby('model'):
                    all_metrics.append(dict(history_days=w,scope=scope,period=month,model=label,**score(g)))
        matched=combined.merge(common,on=['com_id','reported_date','history_days'])
        matched['usage_group']=pd.cut(matched.y_true,[-1,0,2,4,np.inf],labels=['zero','1-2','3-4','5+'])
        for month in ['2026-07','2026-08','all']:
            q=matched if month=='all' else matched[matched.reported_date.str.startswith(month)]
            for (label,band),g in q.groupby(['model','usage_group'],observed=True):
                groups.append(dict(history_days=w,period=month,model=label,usage_group=band,bias=float((g.y_pred-g.y_true).mean()),**score(g)))
    met=pd.DataFrame(all_metrics);met.to_csv(out/'comparison_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'error_by_usage.csv',index=False)
    pd.DataFrame(progress).to_csv(out/'stage_configs.csv',index=False)
    pd.concat(matched_keys).to_csv(out/'matched_keys.csv',index=False)
    combined=pd.concat(all_rows,ignore_index=True)
    combined.to_csv(out/'stage_and_baseline_forecasts.csv.gz',index=False)
    poisson=combined[combined.model.eq('poisson_best')].copy();poisson['model']='lightgbm_poisson_v2'
    required=['com_id','reported_date','split','model','training_scope','history_days','y_true','y_pred']
    poisson[required+[c for c in poisson if c not in required]].to_csv(out/'forecasts_validation_poisson_candidate.csv',index=False)
    q=combined[combined.model.eq('final')&combined.y_true.notna()&combined.y_pred.notna()].copy();q['underprediction']=q.y_true-q.y_pred
    q[q.y_true.ge(q.high_threshold)].sort_values('underprediction',ascending=False).head(40).to_csv(out/'busy_day_examples.csv',index=False)
    print(met[(met.scope=='matched_baselines') & met.model.isin(['original','final','mean7'])].to_string(index=False))
    return met

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/daily_panel.csv.gz');p.add_argument('--output',default='outputs');a=p.parse_args();summarize(a.data,a.output)
