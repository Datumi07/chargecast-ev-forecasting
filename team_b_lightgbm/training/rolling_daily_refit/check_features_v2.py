import numpy as np
import pandas as pd
from experiment import make_features,columns_for
rng=np.random.default_rng(2010);h=rng.integers(0,6,(30,3)).astype(float);h[rng.random(h.shape)<.2]=np.nan;h[:,2]=np.nan;h[10,2]=0
ids=['001','APT2','UC3'];dates=pd.date_range('2026-06-01',periods=31);x=make_features(h,dates,ids)
for t in range(31):
 for j in range(3):
  r=x.iloc[t*3+j];a=h[:t,j];valid=a[np.isfinite(a)];pos=np.where(a>0)[0]
  expected={'past_count':len(valid),'past_mean':valid.mean() if len(valid) else np.nan,'past_zero_fraction':(valid==0).mean() if len(valid) else np.nan,'days_since_positive':t-pos[-1] if len(pos) else np.nan,'has_past_positive':bool(len(pos))}
  week=np.array([h[t-k,j] for k in [7,14,21,28] if t>=k]);week=week[np.isfinite(week)]
  expected.update(same_weekday_count=len(week),same_weekday_mean=week.mean() if len(week) else np.nan)
  a=h[max(0,t-14):max(0,t-7),j];a=a[np.isfinite(a)];b=h[max(0,t-7):t,j];b=b[np.isfinite(b)]
  expected.update(previous_mean7=a.mean() if len(a) else np.nan,trend7=b.mean()-a.mean() if len(a) and len(b) else np.nan)
  for key,value in expected.items():assert np.allclose(float(r[key]),float(value),equal_nan=True),(t,j,key)
changed=h.copy();changed[15:]=99999
pd.testing.assert_frame_equal(x.iloc[:16*3],make_features(changed,dates,ids).iloc[:16*3])
full=np.vstack([np.full((90,3),-1),h]);before=make_features(full[-30:],dates,ids);full[:90]=99999
pd.testing.assert_frame_equal(before,make_features(full[-30:],dates,ids))
assert 'past_count' not in columns_for(x,'base')
assert 'past_zero_fraction' not in columns_for(x,'weekly')
assert x.iloc[:3].past_count.eq(0).all()
print('PASS new features: direct oracle, missing != zero, future invariance, strict window, group isolation.')
