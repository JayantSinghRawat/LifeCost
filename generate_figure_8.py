import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

os.makedirs('artifacts', exist_ok=True)

MODEL_DIR = Path('ml/models')
model_path = MODEL_DIR / 'rent_model.joblib'
meta_path = MODEL_DIR / 'rent_meta.joblib'

if not model_path.exists() or not meta_path.exists():
    print("Model or metadata not found. Please train the rent model first.")
    exit(1)

model = joblib.load(model_path)
meta = joblib.load(meta_path)

feat_cols = meta['feature_cols']

# Some ensemble models (RandomForest, XGBoost) have feature_importances_
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
    print(f"Model {meta.get('best')} does not have feature importances.")
    exit(1)

# Create a DataFrame
df_imp = pd.DataFrame({
    'Feature': feat_cols,
    'Importance': importances
})

# Aggregate city dummy variables if they exist
city_mask = df_imp['Feature'].str.startswith('city_')
if city_mask.any():
    city_importance = df_imp.loc[city_mask, 'Importance'].sum()
    df_imp = df_imp[~city_mask]
    new_row = pd.DataFrame({'Feature': ['City/Zone'], 'Importance': [city_importance]})
    df_imp = pd.concat([df_imp, new_row], ignore_index=True)

# Map ugly feature names to readable ones
feature_name_map = {
    'bhk': 'Bedrooms (BHK)',
    'bathrooms': 'Bathrooms',
    'sqft': 'Square Footage',
    'City/Zone': 'Geographic Zone/City'
}
df_imp['Feature'] = df_imp['Feature'].replace(feature_name_map)

df_imp = df_imp.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', palette="rocket", legend=False)

plt.title('Feature Importance for Rent Prediction Model', fontsize=16, pad=20)
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('Property Features', fontsize=12)

# Format the x-axis to be percentages
from matplotlib.ticker import PercentFormatter
ax.xaxis.set_major_formatter(PercentFormatter(1.0))

plt.tight_layout()
output_path = '/Users/jayant/Desktop/LifeCost/artifacts/Figure_8_Feature_Importance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
