"""
metro_model.py — Bhopal Metro Fare Classifier
Classifies metro fare tier (₹10 / ₹20 / ₹30) from distance and num_stops.
DecisionTree is perfectly interpretable and is the right tool for this tiny dataset.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_metro

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_metro_model(verbose=True):
    df = load_metro()
    if verbose:
        print(f"[Metro] Loaded {len(df)} route pairs.")
        print("  Fare distribution:\n", df["fare"].value_counts().to_string())

    feature_cols = ["distance_km", "num_stops", "travel_min"]
    X = df[feature_cols].values
    y = df["fare"].values  # 10, 20, 30

    # Decision Tree — interpretable rules
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y)
    dt_cv   = cross_val_score(dt, X, y, cv=LeaveOneOut(), scoring="accuracy").mean()
    dt_pred = dt.predict(X)
    dt_acc  = accuracy_score(y, dt_pred)

    # Logistic Regression
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr   = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_sc, y)
    lr_cv  = cross_val_score(lr, X_sc, y, cv=LeaveOneOut(), scoring="accuracy").mean()
    lr_acc = accuracy_score(y, lr.predict(X_sc))

    if verbose:
        print(f"\n  DecisionTree   | train_acc={dt_acc:.3f}  LOO_cv={dt_cv:.3f}")
        print(f"  LogisticRegr   | train_acc={lr_acc:.3f}  LOO_cv={lr_cv:.3f}")
        print("\n  🌳 Decision Tree rules:")
        print(export_text(dt, feature_names=feature_cols))

    # Pick best by LOO cross-val
    if dt_cv >= lr_cv:
        best_model, best_name, best_cv = dt, "DecisionTree", dt_cv
        use_scaler = False
    else:
        best_model, best_name, best_cv = lr, "LogisticRegression", lr_cv
        use_scaler = True

    if verbose:
        print(f"✅ Best: {best_name}  (LOO-accuracy={best_cv:.3f})")
        print("\n  Classification Report:")
        preds = best_model.predict(X_sc if use_scaler else X)
        print(classification_report(y, preds, target_names=["₹10", "₹20", "₹30"]))

    # Save
    model_path = MODELS_DIR / "metro_model.joblib"
    meta_path  = MODELS_DIR / "metro_meta.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(
        {"feature_cols": feature_cols, "scaler": scaler if use_scaler else None,
         "best_name": best_name, "loo_accuracy": best_cv, "use_scaler": use_scaler},
        meta_path,
    )
    if verbose:
        print(f"\n💾 Saved → {model_path}")
    return best_model


def predict_fare(distance_km: float, num_stops: int, travel_min: float = None) -> int:
    """Predict metro fare tier.  travel_min is optional (will be estimated)."""
    model = joblib.load(MODELS_DIR / "metro_model.joblib")
    meta  = joblib.load(MODELS_DIR / "metro_meta.joblib")

    if travel_min is None:
        travel_min = num_stops * 2.0  # ~2 min per stop estimate

    X = np.array([[distance_km, num_stops, travel_min]])
    if meta["use_scaler"]:
        X = meta["scaler"].transform(X)
    return int(model.predict(X)[0])


if __name__ == "__main__":
    print("=" * 60)
    print("  🚇  METRO FARE CLASSIFIER")
    print("=" * 60)
    train_metro_model(verbose=True)

    print("\n--- Interactive Prediction ---")
    while True:
        try:
            print("\nEnter details (or press Ctrl+C to exit):")
            dist = float(input("  Distance in km (e.g., 5.0): "))
            stops = int(input("  Number of stops: "))
            
            fare = predict_fare(dist, stops)
            print(f"\n  ✅ Predicted Fare: ₹{fare}\n" + "-"*40)
        except ValueError:
            print("  ❌ Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
