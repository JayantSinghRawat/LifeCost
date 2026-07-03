import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib.dates as mdates

os.makedirs('artifacts', exist_ok=True)
meta_path = Path('ml/models/cpi_meta.joblib')
meta = joblib.load(meta_path)

hist = meta['historical_series']
fcst = meta['forecast_24m']

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Historical
plt.plot(hist.index, hist.values, marker='o', color='#2c3e50', linewidth=2.5, label='Actual Historical CPI')

# Forecast Connect forecast to historical properly
last_hist_date = hist.index[-1]
last_hist_val = hist.values[-1]

fcst_idx = [last_hist_date] + list(fcst.index)
fcst_val = [last_hist_val] + list(fcst.values)

plt.plot(fcst_idx, fcst_val, linestyle='--', marker='s', markersize=4, color='#e74c3c', linewidth=2.5, label='24-Month Forecast (Linear Baseline)')

plt.title('CPI Historical Data and Forecast Visualization', fontsize=16, pad=20)
plt.xlabel('Date (Month/Year)', fontsize=12)
plt.ylabel('Consumer Price Index (Base 2016=100)', fontsize=12)

# Shade the forecast region
plt.axvspan(last_hist_date, fcst_idx[-1], color='#e74c3c', alpha=0.1, label='Forecast Zone')

plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle="--", alpha=0.7)

# Format x-axis with nice date formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()

output_path = '/Users/jayant/Desktop/MP-Life/artifacts/Figure_10_CPI_Forecast.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
