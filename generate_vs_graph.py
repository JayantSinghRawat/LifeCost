import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, './ml')
from data_prep import load_cpi

meta = joblib.load('ml/models/cpi_meta.joblib')
df = load_cpi()

df_ml = df[["cpi", "lag_1", "lag_3", "time_idx"]].dropna()
X_lin = df_ml[["lag_1", "lag_3", "time_idx"]].values

lin_model = meta["linear_model"]
y_pred = lin_model.predict(X_lin)

plt.figure(figsize=(10, 6))

plt.plot(df.index, df['cpi'], label='Actual Historical CPI', color='blue', linewidth=2.5, marker='o')
plt.plot(df_ml.index, y_pred, label='Model Fitted (Predicted) CPI', color='red', linestyle='--', linewidth=2.5, marker='x')

plt.title('Algorithms Accuracy: Actual vs Predicted CPI (Historical Validation)')
plt.xlabel('Date')
plt.ylabel('Consumer Price Index')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

output_path = 'cpi_actual_vs_predicted.png'
plt.savefig(output_path, dpi=300)
print(f"Vs graph saved as {output_path}")
