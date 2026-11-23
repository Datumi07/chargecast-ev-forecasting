from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from experiment import load_panel,make_features,columns_for,config_id

def verify(data,output):
    out=Path(output);_,wide=load_panel(data)
    q=pd.read_csv(out/'all_test_forecasts.csv.gz',dtype={'com_id':'string'})
    assert len(q)==len(wide.columns)*30*2*4
    assert not q.duplicated(['com_id','reported_date','history_days','model']).any()
    assert q.split.eq('test').all()
    assert q.reported_date.between('2026-09-01','2026-09-30').all()
    truth=wide.stack(dropna=False).rename('actual').reset_index()
    truth['reported_date']=truth.reported_date.dt.strftime('%Y-%m-%d')
    joined=q.merge(truth,on=['com_id','reported_date'],validate='many_to_one')
    np.testing.assert_allclose(joined.y_true,joined.actual,equal_nan=True)
    settings=json.loads((out/'test_protocol.json').read_text())['settings']
    for w in [30,90]:
        for day in pd.date_range('2026-09-01','2026-09-30'):
            h=wide.loc[day-pd.Timedelta(days=w):day-pd.Timedelta(days=1)].to_numpy()
            counts=np.isfinite(h).sum(0)
            rows=q[q.history_days.eq(w)&q.reported_date.eq(str(day.date()))]
            for label in ['lightgbm_l1','lightgbm_poisson']:
                r=rows[rows.model.eq(label)].set_index('com_id').loc[wide.columns]
                np.testing.assert_array_equal(r.y_pred.notna(),counts>=7)
                np.testing.assert_allclose(r.high_threshold,max(1,np.nanquantile(h,.9)))
                cfg=settings[label][str(w)]
                past=np.vstack([np.zeros(len(wide.columns)),np.cumsum(np.isfinite(h),axis=0)])[:-1]
                expected=(np.isfinite(h)&(past>=cfg['min_train_history'])).sum()
                assert r.training_rows.eq(expected).all()
            c=np.isfinite(h[-7:]).sum(0)
            mean=np.divide(np.nansum(h[-7:],axis=0),c,out=np.full(len(c),np.nan),where=c>0)
            for label,pred in [('mean7',mean),('lag7',h[-7])]:
                r=rows[rows.model.eq(label)].set_index('com_id').loc[wide.columns]
                np.testing.assert_allclose(r.y_pred,np.where(counts>=7,pred,np.nan),equal_nan=True)
        day=pd.Timestamp('2026-09-30');start=day-pd.Timedelta(days=w)
        h=wide.loc[start:day-pd.Timedelta(days=1)].to_numpy()
        x=make_features(h,pd.date_range(start,day),wide.columns.tolist())
        for label in ['lightgbm_l1','lightgbm_poisson']:
            cfg=settings[label][str(w)];folder=out/'cache'/str(w)/config_id(cfg)
            booster=lgb.Booster(model_file=str(folder/'model_20260930.txt'))
            cols=json.loads((folder/'feature_columns.json').read_text())
            pred=np.maximum(booster.predict(x.iloc[-len(wide.columns):][cols]),0)
            r=q[q.history_days.eq(w)&q.reported_date.eq('2026-09-30')&q.model.eq(label)].set_index('com_id').loc[wide.columns]
            np.testing.assert_allclose(r.y_pred,np.where(np.isfinite(h).sum(0)>=7,pred,np.nan),equal_nan=True,rtol=1e-12,atol=1e-12)
    keys=pd.read_csv(out/'matched_test_keys.csv',dtype={'com_id':'string'})
    matched=q.merge(keys,on=['com_id','reported_date','history_days'],validate='many_to_one')
    met=pd.read_csv(out/'september_metrics.csv')
    for r in met[met.scope.eq('matched_all_models')].itertuples():
        g=matched[matched.history_days.eq(r.history_days)&matched.model.eq(r.model)]
        assert len(g)==r.n_scored and g.y_pred.notna().all() and g.y_true.notna().all()
        err=g.y_pred-g.y_true
        np.testing.assert_allclose([err.abs().mean(),np.sqrt((err**2).mean())],[r.MAE,r.RMSE],rtol=1e-12)
    result={'status':'PASS','rows':len(q),'unique_keys':True,'targets_match_source':True,'prediction_eligibility_all_origins':True,'training_row_counts_all_origins':True,'baseline_recalculation':True,'busy_thresholds':True,'saved_models_reload':True,'matched_metrics_recomputed':True}
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/daily_panel.csv.gz');p.add_argument('--output',default='outputs_test');a=p.parse_args();verify(a.data,a.output)
