"""
grocery_model.py — Daily Grocery Basket Estimator
Price prediction per grocery item + daily basket cost for Bhopal.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_grocery

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Realistic single-person daily consumption
# (basket is conservative — breakfast + dinner vegs, not restaurant)
DAILY_BASKET = {
    # category : (amount, unit)   unit: 'ml' | 'g' | 'pcs'
    "milk":       (500,  "ml"),   # 500 ml/day  (≈ 1 glass + chai)
    "bread":      (4,    "pcs"),  # 4 slices  →  ≈100 g
    "eggs":       (2,    "pcs"),  # 2 eggs
    "vegetables": (200,  "g"),   # 200 g (serving of 2 sabzis)
    "fruits":     (150,  "g"),   # 150 g (1 medium piece of fruit)
}


def _median_unit_price(df: pd.DataFrame, cat: str, unit: str) -> float:
    """
    For a category compute the median price per BASE unit:
      'ml'  → price per 1 ml  (from items sold in ml/ltr)
      'g'   → price per 1 g   (from items sold in g/kg)
      'pcs' → median item price directly (the item IS 1 unit)
    """
    sub = df[df["category"] == cat].copy()
    sub = sub.dropna(subset=["price", "quantity_norm_g"])
    sub = sub[sub["quantity_norm_g"] > 0]

    if sub.empty:
        return np.nan

    if unit == "pcs":
        # Just use median price of items in the category (each listing ≈ 1 pack)
        return float(sub["price"].median())
    else:
        # price_per_unit = price / qty_norm_g  (qty_norm_g stores both g and ml on same scale)
        sub = sub[sub["price_per_100g"].notna()]
        sub = sub[sub["price_per_100g"] < 2000]   # sanity cap
        if sub.empty:
            return np.nan
        return float(sub["price_per_100g"].median()) / 100  # price per 1 g or ml


def train_grocery_model(verbose=True):
    df = load_grocery()
    if verbose:
        print(f"[Grocery] Loaded {len(df)} priced items.")

    # Ridge on price_per_100g (for ML baseline)
    df_pp = df.dropna(subset=["price_per_100g"]).copy()
    df_pp = df_pp[df_pp["price_per_100g"] < 2000]

    le = LabelEncoder()
    df_pp["cat_enc"] = le.fit_transform(df_pp["category"])
    X = df_pp[["cat_enc", "quantity_norm_g"]].values
    y = df_pp["price_per_100g"].values

    mae, r2 = np.nan, np.nan
    if len(X) > 5:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)
    else:
        model = Ridge(alpha=1.0).fit(X, y)

    # Per-category median stats for reporting
    cat_stats = (
        df.groupby("category")["price"]
        .agg(["median", "mean", "std", "count"])
    )
    if verbose:
        print("\n  Per-category median price (₹):\n", cat_stats.round(2))

    # ── Compute daily basket using unit-aware pricing ─────────────────────
    basket = {}
    for cat, (amount, unit) in DAILY_BASKET.items():
        price_per_unit = _median_unit_price(df, cat, unit)

        if unit == "pcs":
            # price_per_unit is the median pack price; estimate cost per piece
            # e.g. eggs median pack = ₹75 for 6 → ₹12.5/egg
            # bread median pack = ₹35 for ~10 slices → ₹3.5/slice
            # We'll divide by typical pieces per pack
            pieces_per_pack = {"eggs": 6, "bread": 8}.get(cat, 1)
            cost = (price_per_unit / pieces_per_pack) * amount if not np.isnan(price_per_unit) else np.nan
        else:
            # price_per_unit in ₹/ml or ₹/g
            cost = price_per_unit * amount if not np.isnan(price_per_unit) else np.nan

        basket[cat] = {
            "amount": amount,
            "unit": unit,
            "cost_inr": round(cost, 2) if (cost and not np.isnan(cost)) else None,
        }

    total = sum(v["cost_inr"] for v in basket.values() if v["cost_inr"] is not None)
    if verbose:
        print("\n  --- Estimated Daily Basket (1 person, Bhopal) ---")
        for cat, info in basket.items():
            cost_str = f"₹{info['cost_inr']:.2f}" if info["cost_inr"] else "N/A"
            print(f"    {cat:12s} {info['amount']:>4}{info['unit']:<4} →  {cost_str}")
        print(f"  {'TOTAL':12s}               →  ₹{total:.2f}/day  (~₹{total*30:.0f}/month)")

    # Save
    model_path = MODELS_DIR / "grocery_model.joblib"
    meta_path  = MODELS_DIR / "grocery_meta.joblib"
    joblib.dump(model, model_path)
    joblib.dump(
        {"label_encoder": le, "cat_stats": cat_stats, "basket": basket, "total_daily": total,
         "mae": mae, "r2": r2},
        meta_path,
    )
    if verbose:
        print(f"\n💾 Saved → {model_path}")
    return model, basket, total


def get_basket_cost() -> dict:
    """Return saved basket breakdown.  Run train_grocery_model() first."""
    meta = joblib.load(MODELS_DIR / "grocery_meta.joblib")
    return {"basket": meta["basket"], "total_daily_inr": meta["total_daily"],
            "total_monthly_inr": meta["total_daily"] * 30}


if __name__ == "__main__":
    print("=" * 60)
    print("  🛒  GROCERY BASKET ESTIMATOR")
    print("=" * 60)
    train_grocery_model(verbose=True)
    
    print("\n--- Interactive Basket Cost ---")
    try:
        input("Press Enter to fetch the estimated daily basket cost (or Ctrl+C to exit)...")
        basket_data = get_basket_cost()
        print("\n✅ Daily Basket (1 person, Bhopal):")
        for cat, info in basket_data['basket'].items():
            cost_str = f"₹{info['cost_inr']:.2f}" if info["cost_inr"] else "N/A"
            print(f"  {cat:12s} {info['amount']:>4}{info['unit']:<4} → {cost_str}")
        print("-" * 35)
        print(f"  {'TOTAL DAILY':12s}             → ₹{basket_data['total_daily_inr']:.2f}")
        print(f"  {'TOTAL MONTHLY':12s}           → ₹{basket_data['total_monthly_inr']:.2f}")
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
