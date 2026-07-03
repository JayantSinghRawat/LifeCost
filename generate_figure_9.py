import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('artifacts', exist_ok=True)

# Add ml folder to path to import data_prep and rent_model
ml_path = Path('ml').resolve()
sys.path.insert(0, str(ml_path))

try:
    from data_prep import load_rent
    from rent_model import prepare_features
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

model_path = ml_path / 'models' / 'rent_model.joblib'
if not model_path.exists():
    print("Rent model not found. Train it first.")
    sys.exit(1)

# Load data and prepare identical test split
df = load_rent()
X, y, feat_cols = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model and predict
model = joblib.load(model_path)
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

# Scatter plot
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#4c72b0", edgecolor="w", s=80)

# Ideal line
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

plt.title('Actual vs Predicted Rent Values Scatter Plot', fontsize=16, pad=20)
plt.xlabel('Actual Rent (₹)', fontsize=12)
plt.ylabel('Predicted Rent (₹)', fontsize=12)

# Format axes to thousands
from matplotlib.ticker import FuncFormatter
def format_k(x, pos):
    return f'₹{int(x/1000)}k'

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(format_k))
ax.yaxis.set_major_formatter(FuncFormatter(format_k))

plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
output_path = '/Users/jayant/Desktop/MP-Life/artifacts/Figure_9_Actual_vs_Predicted_Rent.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
