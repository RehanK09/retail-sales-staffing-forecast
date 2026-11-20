import sys
sys.path.insert(0, '.')
from src.data_loader  import load_merged_train, load_stores, load_features
from src.preprocessing import engineer_features
from src.models        import models_exist, load_all
from src.evaluation    import load_results
from src.staffing      import estimate_staffing, staffing_from_forecast_df, staffing_recommendation
from src.forecasting   import predict, predict_range
import pandas as pd

print('All imports OK')
print('Models exist:', models_exist())

df = load_merged_train()
print('Stores:', df['Store'].nunique(), '| Depts:', df['Dept'].nunique())

result = predict(1, 1, pd.Timestamp('2012-11-02'), False)
lr_w = result['lr_pred']
dt_w = result['dt_pred']
print("Next week LR:", lr_w, " DT:", dt_w)

month_dates = pd.date_range(pd.Timestamp('2012-11-02'), periods=4, freq='W-FRI')
total_dt = 0.0
for d in month_dates:
    r2 = predict(1, 1, d, False)
    total_dt += r2['dt_pred']
print("4-week DT total:", round(total_dt, 2))

fdf = predict_range(1, 1, pd.Timestamp('2012-11-02'), pd.Timestamp('2012-11-30'))
print('Range forecast rows:', len(fdf))

est = estimate_staffing(dt_w, 300.0, 40.0, 2)
print('Staff:', est['estimated_staff'], ' Hrs:', est['labour_hours'])

r3 = load_results()
print('DT R2:', r3['Decision Tree']['overall']['R2'])
print('LR R2:', r3['Linear Regression']['overall']['R2'])
print('ALL CHECKS PASSED')

