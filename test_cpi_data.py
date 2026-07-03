import sys
import joblib
sys.path.insert(0, './ml')
from data_prep import load_cpi

meta = joblib.load('ml/models/cpi_meta.joblib')
df = load_cpi()

print("Original DF shape:", df.shape)
print("Original DF tail:")
print(df.tail(15))

df_ml = df[["cpi", "lag_1", "lag_3", "time_idx"]].dropna()

X_lin = df_ml[["lag_1", "lag_3", "time_idx"]].values
lin_model = meta["linear_model"]
y_pred = lin_model.predict(X_lin)

print("\ndf_ml shape:", df_ml.shape)
for i in range(len(df_ml)):
    print(f"Date: {df_ml.index[i].date()}, Actual: {df_ml['cpi'].iloc[i]:.2f}, Predicted: {y_pred[i]:.2f}")
