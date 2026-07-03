import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# Full 10 historical points (bypassing the lag_3 drop in data_prep to get all data)
hist_dates = pd.to_datetime([
    '2021-01-01', '2021-02-01', '2022-01-01', '2022-02-01', 
    '2023-01-01', '2023-02-01', '2024-01-01', '2024-02-01', 
    '2025-01-01', '2025-02-01'
])
hist_cpi = [113.9, 115.1, 121.1, 122.8, 127.8, 128.2, 135.0, 134.2, 132.4, 132.7]

# Load the 24-month forecast from the generated models
meta = joblib.load('ml/models/cpi_meta.joblib')
forecast = meta["forecast_24m"] # Series with DatetimeIndex and 24 future points

plt.figure(figsize=(11, 6))

# 1. Plot the actual historical CPI
plt.plot(hist_dates, hist_cpi, label='Actual Historical CPI (Jan 2021 - Feb 2025)', color='#3b82f6', marker='o', linewidth=2.5)

# 2. To make the line continuous, prepend the last historical point to the forecast
fcst_dates = [hist_dates[-1]] + list(forecast.index)
fcst_values = [hist_cpi[-1]] + list(forecast.values)

# 3. Plot the predicted 24-month forecast CPI
plt.plot(fcst_dates, fcst_values, label='Predicted 24-Month Forecast', color='#8b5cf6', linestyle='--', marker='x', linewidth=2.5)

# Formatting
plt.title('Bhopal Consumer Price Index: Actual vs Forecast (2021-2027)', fontsize=14, pad=15)
plt.ylabel('General Index (Base 2016=100)', fontsize=12)
plt.xlabel('Year', fontsize=12)

# Make X-axis show years clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Save it as cpi_forecast_graph.png (overwriting the first one)
plt.savefig('cpi_forecast_graph.png', dpi=300)
print("Graph generated successfully: cpi_forecast_graph.png")
