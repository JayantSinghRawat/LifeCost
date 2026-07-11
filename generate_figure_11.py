import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

os.makedirs('artifacts', exist_ok=True)

# setup imports
ml_path = Path('ml').resolve()
sys.path.insert(0, str(ml_path))

try:
    from data_prep import load_metro
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

model_path = ml_path / 'models' / 'metro_model.joblib'
meta_path = ml_path / 'models' / 'metro_meta.joblib'
if not model_path.exists() or not meta_path.exists():
    print("Metro model not found. Train it first.")
    from metro_model import train_metro_model
    train_metro_model(verbose=False)

# Load data
df = load_metro()
feature_cols = ["distance_km", "num_stops", "travel_min"]
X = df[feature_cols].values
y_true = df["fare"].values

# Load model and meta
model = joblib.load(model_path)
meta = joblib.load(meta_path)

if meta['use_scaler']:
    X = meta['scaler'].transform(X)

y_pred = model.predict(X)

# Confusion Matrix
labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

# Plot heatmap
ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                 xticklabels=[f"₹{l}" for l in labels], 
                 yticklabels=[f"₹{l}" for l in labels],
                 annot_kws={"size": 15}, linewidths=1, linecolor='black')

plt.title('Confusion Matrix for Metro Fare Classification', fontsize=16, pad=20)
plt.xlabel('Predicted Fare Class', fontsize=14, labelpad=10)
plt.ylabel('Actual Fare Class', fontsize=14, labelpad=10)

plt.tight_layout()
output_path = '/Users/jayant/Desktop/LifeCost/artifacts/Figure_11_Metro_Confusion_Matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
