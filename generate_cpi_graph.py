import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Load CPI Meta
meta = joblib.load('ml/models/cpi_meta.joblib')
historical = meta["historical_series"]
forecast = meta["forecast_24m"]

plt.figure(figsize=(10, 6))

# Plot historical
plt.plot(historical.index, historical.values, label='Historical CPI', color='blue', linewidth=2)

# Plot forecast
# Prepend the last historical point to forecast so the lines connect smoothly
concat_index = [historical.index[-1]] + list(forecast.index)
concat_values = [historical.iloc[-1]] + list(forecast.values)
plt.plot(concat_index, concat_values, label='24-Month Forecast (ML)', color='orange', linestyle='--', linewidth=2)

plt.title('Bhopal Consumer Price Index: Historical vs Forecast')
plt.xlabel('Year')
plt.ylabel('CPI (Base 2016=100)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

output_path = 'cpi_forecast_graph.png'
plt.savefig(output_path, dpi=300)
print(f"Graph saved exactly as {output_path}")
