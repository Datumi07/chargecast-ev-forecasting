"""Frozen September evaluation. No model selection or tuning on test."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import lightgbm as lgb
from experiment import load_panel, evaluate_many, read_forecasts, score, COMMIT, SHA256
from analyze_results import baseline_forecasts

def run(data='data/daily_panel.csv.gz', output='outputs_test'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    settings=Path(__file__).parent/'settings'
    locked={label:json.loads((settings/name).read_text()) for label,name in [('lightgbm_l1','locked_july_settings.json'),('lightgbm_poisson','locked_july_poisson_comparison.json')]}
    protocol=dict(primary_model='lightgbm_l1',secondary_model='lightgbm_poisson',history_days=[30,90],primary_metric='MAE',test_start='2026-09-01',test_end='2026-09-30',refit='daily; previous observed days may enter later windows',tuning_on_test=False,settings=locked,data_sha256=SHA256,source_commit=COMMIT,numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,python=platform.python_version(),code_hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['run_test.py','experiment.py','base_v1.py','analyze_results.py']})
    dest=out/'test_protocol.json'
    if dest.exists(): assert json.loads(dest.read_text())==protocol,'Use a fresh output folder for changed code/configuration.'
    else: dest.write_text(json.dumps(protocol,indent=2))
    _,wide=load_panel(data);dates=pd.date_range('2026-09-01','2026-09-30')
    assert wide.index.max()==dates[-1]
    all_rows=[];metrics=[];keys=[];groups=[]
    for w in [30,90]:
        configs=[cfg[str(w)] for cfg in locked.values()]
        evaluate_many(w,configs,dates,wide,out,'september_test')
        frames=[]
        for label,cfg in locked.items():
            q=read_forecasts(out,w,cfg[str(w)],dates);q['model']=label;frames.append(q)
        b=baseline_forecasts(w,dates,wide);b['split']='test';b['training_scope']='device_specific';frames.append(b)
        q=pd.concat(frames,ignore_index=True)
        assert not q.duplicated(['com_id','reported_date','model']).any()
        available=q[q.y_true.notna()&q.y_pred.notna()]
        counts=available.groupby(['com_id','reported_date']).model.nunique()
        common=counts[counts==4].reset_index()[['com_id','reported_date']];common['history_days']=w;keys.append(common)
        match=q.merge(common,on=['com_id','reported_date','history_days'],validate='many_to_one')
        assert match.groupby(['com_id','reported_date']).y_true.nunique().eq(1).all()
        for scope,rows in [('all_available',q),('matched_all_models',match)]:
            for label,g in rows[rows.y_true.notna()].groupby('model'):
                metrics.append(dict(history_days=w,scope=scope,period='2026-09',model=label,**score(g)))
        match['usage_group']=pd.cut(match.y_true,[-1,0,2,4,np.inf],labels=['zero','1-2','3-4','5+'])
        for (label,band),g in match.groupby(['model','usage_group'],observed=True):
            groups.append(dict(history_days=w,model=label,usage_group=str(band),bias=float((g.y_pred-g.y_true).mean()),**score(g)))
        all_rows.append(q)
    allp=pd.concat(all_rows,ignore_index=True)
    required=['com_id','reported_date','split','model','training_scope','history_days','y_true','y_pred']
    allp=allp[required+[c for c in allp if c not in required]]
    for label,name in [('lightgbm_l1','forecasts_test_september_l1.csv'),('lightgbm_poisson','forecasts_test_september_poisson.csv')]:
        allp[allp.model.eq(label)].to_csv(out/name,index=False)
    allp[allp.model.isin(['mean7','lag7'])].to_csv(out/'baselines_test_september.csv',index=False)
    allp.to_csv(out/'all_test_forecasts.csv.gz',index=False)
    pd.concat(keys).to_csv(out/'matched_test_keys.csv',index=False)
    met=pd.DataFrame(metrics);met.to_csv(out/'september_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'error_by_usage_september.csv',index=False)
    busy=allp[allp.model.str.startswith('lightgbm')&allp.y_true.ge(allp.high_threshold)&allp.y_pred.notna()].copy()
    busy['underprediction']=(busy.y_true-busy.y_pred).clip(lower=0)
    busy.sort_values('underprediction',ascending=False).head(50).to_csv(out/'busy_day_examples_september.csv',index=False)
    print(met[met.scope.eq('matched_all_models')].to_string(index=False),flush=True)
    return met

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/daily_panel.csv.gz');p.add_argument('--output',default='outputs_test');a=p.parse_args();run(a.data,a.output)
