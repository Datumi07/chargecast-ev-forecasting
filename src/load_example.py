from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
p=pd.read_csv(root/'data/processed/daily_panel.csv.gz',dtype={'com_id':'string','sessions_target':'Int64'},parse_dates=['reported_date']).sort_values(['com_id','reported_date'])

# Calendar is continuous. Shift before filtering invalid labels or selecting splits.
p['lag_1']=p.groupby('com_id').sessions_target.shift(1)
p['lag_7']=p.groupby('com_id').sessions_target.shift(7)

# The target is the date on the same row; features only access earlier dates.
p['weekday']=p.reported_date.dt.dayofweek

# Complete-case example only: explicit row loss, NOT a mandate for all models.
usable=p.recommended_period & p.target_valid & p[['lag_1','lag_7']].notna().all(axis=1)
X_columns=['lag_1','lag_7','weekday']
for split in ['train','validation','test']:
    q=p.loc[usable & p.split.eq(split)]
    X=q[X_columns]; y=q.sessions_target
print(split, 'rows=',len(q),'devices=',q.com_id.nunique())

