import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

sys.path.insert(0, './ml')
from data_prep import load_cpi

meta = joblib.load('ml/models/cpi_meta.joblib')
df = load_cpi()

# In-sample predictions (Historical fit)
df_ml = df[["cpi", "lag_1", "lag_3", "time_idx"]].dropna()
X_lin = df_ml[["lag_1", "lag_3", "time_idx"]].values
lin_model = meta["linear_model"]
y_pred_hist = lin_model.predict(X_lin)

# Out-of-sample predictions (Future forecast)
forecast = meta["forecast_24m"] # Series with dates and values

# Combine Predicted indices and values
pred_dates = list(df_ml.index) + list(forecast.index)
pred_values = list(y_pred_hist) + list(forecast.values)

# Full historical ACTUAL data from 2021
hist_dates = pd.to_datetime([
    '2021-01-01', '2021-02-01', '2022-01-01', '2022-02-01', 
    '2023-01-01', '2023-02-01', '2024-01-01', '2024-02-01', 
    '2025-01-01', '2025-02-01'
])
hist_cpi = [113.9, 115.1, 121.1, 122.8, 127.8, 128.2, 135.0, 134.2, 132.4, 132.7]

plt.figure(figsize=(11, 6))

# 1. Plot the actual historical CPI
plt.plot(hist_dates, hist_cpi, label='Actual Historical Data', color='#3b82f6', marker='o', linewidth=2.5)

# 2. Plot the predicted CPI (combining historical fit and future forecast)
plt.plot(pred_dates, pred_values, label='Predicted (Fit + Forecast)', color='#eab308', linestyle='--', marker='x', linewidth=2.5)

# Formatting
plt.title('Algorithms Accuracy: Actual vs Predicted CPI (2021-2027)', fontsize=14, pad=15)
plt.xlabel('Date (Year/Month)', fontsize=12)
plt.ylabel('Consumer Price Index', fontsize=12)

# Make X-axis show years cleanly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.xlim(pd.Timestamp('2020-09-01'), pd.Timestamp('2027-06-01'))

plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

output_path = 'cpi_actual_vs_predicted.png'
plt.savefig(output_path, dpi=300)
print(f"Vs graph generated: {output_path}")
