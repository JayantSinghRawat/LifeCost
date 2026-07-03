"""
composite_index.py — MP Cost of Living Composite Index
Combines rent, grocery, transport, and CPI-adjusted values into a single index.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent / "models"

# Component weights
WEIGHTS = {
    "rent":      0.50,
    "grocery":   0.30,
    "transport": 0.05,
    "cpi_adj":   0.15,
}

# Normalisation baseline (reference costs in ₹/month for Bhopal)
# These are approximate medians used to compute the index (100 = baseline June 2021)
BASELINE = {
    "rent_1bhk_monthly":     7_500,   # ₹/month (1BHK Bhopal median ~2021)
    "grocery_monthly":       1_500,   # ₹/month (basket * 30 days ~2021 estimate)
    "transport_10_trips":      200,   # ₹/month (10 metro commutes @ ₹20 avg)
    "cpi_base":               114.3,  # Jan 2021 CPI (first data point)
}


def compute_index(
    rent_monthly: float = None,
    grocery_daily: float = None,
    avg_metro_fare: float = 20.0,
    cpi_latest: float = None,
    verbose: bool = True,
) -> dict:
    """
    Compute the MP Cost of Living Composite Index.

    Parameters
    ----------
    rent_monthly   : median 1BHK rent for city (₹).  Auto-loaded from rent model if None.
    grocery_daily  : daily basket cost (₹).            Auto-loaded from grocery model if None.
    avg_metro_fare : average metro fare per trip (₹).  Default ₹20 (Bhopal median).
    cpi_latest     : latest CPI index value.           Auto-loaded from CPI model if None.
    """
    # ── Auto-load from saved models if not provided ──────────────────────────
    if rent_monthly is None:
        try:
            from rent_model import predict_rent
            rent_monthly = predict_rent(1, 1, 500, "BHOPAL")
        except Exception:
            rent_monthly = BASELINE["rent_1bhk_monthly"]

    if grocery_daily is None:
        try:
            grocery_meta = joblib.load(MODELS_DIR / "grocery_meta.joblib")
            grocery_daily = grocery_meta["total_daily"]
        except Exception:
            grocery_daily = BASELINE["grocery_monthly"] / 30

    if cpi_latest is None:
        try:
            cpi_meta = joblib.load(MODELS_DIR / "cpi_meta.joblib")
            cpi_latest = cpi_meta["last_actual_cpi"]
        except Exception:
            cpi_latest = BASELINE["cpi_base"]

    grocery_monthly  = grocery_daily * 30
    transport_monthly = avg_metro_fare * 10  # assume 10 commutes/month

    # ── Raw component costs ──────────────────────────────────────────────────
    costs = {
        "rent":      rent_monthly,
        "grocery":   grocery_monthly,
        "transport": transport_monthly,
    }
    total_raw = sum(costs.values())

    # ── CPI adjustment factor (relative to baseline CPI) ────────────────────
    cpi_factor = cpi_latest / BASELINE["cpi_base"]

    # ── Index computation  ───────────────────────────────────────────────────
    # Normalise each component against its 2021 baseline
    norm = {
        "rent":      rent_monthly      / BASELINE["rent_1bhk_monthly"],
        "grocery":   grocery_monthly   / BASELINE["grocery_monthly"],
        "transport": transport_monthly / BASELINE["transport_10_trips"],
        "cpi_adj":   cpi_factor,
    }
    col_index = sum(norm[k] * WEIGHTS[k] for k in WEIGHTS) * 100  # scale to 100 = baseline

    if verbose:
        print("=" * 60)
        print("  🧮  MP COST OF LIVING COMPOSITE INDEX")
        print("=" * 60)
        print(f"  📍  Location       : Bhopal (MP)")
        print(f"  🏠  Rent (1BHK)    : ₹{rent_monthly:,.0f}/month")
        print(f"  🛒  Grocery        : ₹{grocery_daily:.2f}/day  →  ₹{grocery_monthly:.0f}/month")
        print(f"  🚇  Transport      : ₹{avg_metro_fare:.0f} × 10 trips  →  ₹{transport_monthly:.0f}/month")
        print(f"  📈  CPI Latest     : {cpi_latest:.1f}  (factor: {cpi_factor:.3f})")
        print(f"  ─────────────────────────────────────────")
        print(f"  💰  Total Monthly  : ₹{total_raw:,.0f}")
        print(f"  📊  CoL Index      : {col_index:.1f}  (100 = Jan 2021 baseline)")
        print("=" * 60)

    return {
        "city": "Bhopal",
        "rent_monthly": round(rent_monthly, 2),
        "grocery_monthly": round(grocery_monthly, 2),
        "transport_monthly": round(transport_monthly, 2),
        "total_monthly_inr": round(total_raw, 2),
        "cpi_latest": round(cpi_latest, 2),
        "cpi_factor": round(cpi_factor, 4),
        "col_index": round(col_index, 2),
        "weights": WEIGHTS,
    }


if __name__ == "__main__":
    result = compute_index(verbose=True)
