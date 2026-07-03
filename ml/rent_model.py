"""
rent_model.py — Rent Price Predictor
Train a RandomForest + XGBoost ensemble on OLX rental listings from MP.
Predicts monthly rent (₹) given: BHK, bathrooms, sqft, city.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_rent

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def prepare_features(df: pd.DataFrame):
    """Encode categoricals and return X, y."""
    df = df.copy()

    # One-hot encode city
    df = pd.get_dummies(df, columns=["city"], drop_first=False)

    # Drop non-numeric / unneeded cols
    drop_cols = [c for c in ["bhk_type"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    feature_cols = [c for c in df.columns if c != "rent"]
    X = df[feature_cols].values
    y = df["rent"].values
    return X, y, feature_cols


def train_rent_model(verbose=True):
    df = load_rent()
    if verbose:
        print(f"[Rent] Loaded {len(df)} rows after cleaning.")

    X, y, feat_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )

    best_rmse = float("inf")
    best_name = None
    best_model = None
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        results[name] = dict(rmse=rmse, mae=mae, r2=r2)
        if verbose:
            print(f"  {name:20s} | RMSE=₹{rmse:,.0f}  MAE=₹{mae:,.0f}  R²={r2:.3f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    if verbose:
        print(f"\n✅ Best model: {best_name}  (RMSE=₹{best_rmse:,.0f})")

    # Persist artefacts
    model_path = MODELS_DIR / "rent_model.joblib"
    meta_path  = MODELS_DIR / "rent_meta.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump({"feature_cols": feat_cols, "results": results, "best": best_name}, meta_path)
    if verbose:
        print(f"💾 Saved → {model_path}")
    return best_model, feat_cols, results


def predict_rent(bhk: int, bathrooms: int, sqft: float, city: str = "BHOPAL") -> float:
    """
    Quick inference. Returns predicted monthly rent in ₹.
    Load once; call many times.
    """
    model_path = MODELS_DIR / "rent_model.joblib"
    meta_path  = MODELS_DIR / "rent_meta.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run train_rent_model() first.")

    model = joblib.load(model_path)
    meta  = joblib.load(meta_path)
    feat_cols = meta["feature_cols"]

    # Build input row
    row = {"bhk": bhk, "bathrooms": bathrooms, "sqft": sqft}
    # One-hot encode city columns
    for col in feat_cols:
        if col.startswith("city_"):
            c = col[len("city_"):]
            row[col] = 1 if c == city.upper() else 0
    X = np.array([[row.get(c, 0) for c in feat_cols]])
    pred = float(model.predict(X)[0])
    return max(pred, 0)


if __name__ == "__main__":
    print("=" * 60)
    print("  🏠  RENT PRICE PREDICTOR")
    print("=" * 60)
    train_rent_model(verbose=True)

    print("\n--- Interactive Prediction ---")
    while True:
        try:
            print("\nEnter details (or press Ctrl+C to exit):")
            bhk = int(input("  BHK (e.g., 2): "))
            bath = int(input("  Bathrooms: "))
            sqft = float(input("  Square Footage: "))
            city = input("  City (e.g., BHOPAL, INDORE): ").strip().upper()
            
            p = predict_rent(bhk, bath, sqft, city)
            print(f"\n  ✅ Predicted Rent: ₹{p:,.0f}/month\n" + "-"*40)
        except ValueError:
            print("  ❌ Invalid input. Please enter valid numbers for BHK, Bathrooms, and Sqft.")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
