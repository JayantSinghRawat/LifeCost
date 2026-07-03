"""
evaluate.py — Unified Evaluation Report for all MP CoL ML Models
Runs training for every model and prints metrics in a clean summary table.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import time


def section(title: str):
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║  {title:<56}║")
    print("╚" + "═" * 58 + "╝")


def run_all():
    start = time.time()
    results = {}

    # ── 1. Rent ───────────────────────────────────────────────────────────────
    section("🏠  RENT PRICE PREDICTOR")
    from rent_model import train_rent_model
    _, _, rent_results = train_rent_model(verbose=True)
    results["rent"] = rent_results

    # ── 2. Grocery ────────────────────────────────────────────────────────────
    section("🛒  GROCERY BASKET ESTIMATOR")
    from grocery_model import train_grocery_model
    _, basket, daily_total = train_grocery_model(verbose=True)
    results["grocery"] = {"basket": basket, "daily_total": daily_total}

    # ── 3. CPI ────────────────────────────────────────────────────────────────
    section("📈  CPI INFLATION FORECASTER")
    from cpi_model import train_cpi_model
    cpi_meta = train_cpi_model(verbose=True)
    results["cpi"] = {
        "arima_mape": cpi_meta.get("arima_mape"),
        "linear_mape": cpi_meta.get("linear_mape"),
    }

    # ── 4. Metro ──────────────────────────────────────────────────────────────
    section("🚇  METRO FARE CLASSIFIER")
    from metro_model import train_metro_model
    train_metro_model(verbose=True)

    # ── 5. Composite Index ────────────────────────────────────────────────────
    section("🧮  COMPOSITE COST-OF-LIVING INDEX")
    from composite_index import compute_index
    col_result = compute_index(verbose=True)
    results["composite"] = col_result

    # ── Summary Table ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print()
    print("┌" + "─" * 58 + "┐")
    print(f"│  {'EVALUATION SUMMARY':^56}│")
    print("├" + "─" * 58 + "┤")

    # Rent best model
    best_rent = min(rent_results.items(), key=lambda x: x[1]["rmse"])
    name, m = best_rent
    target_rent  = "✅" if m["rmse"] < 5000 else "⚠️ "
    target_r2    = "✅" if m["r2"]   > 0.70  else "⚠️ "
    print(f"│  Rent  ({name:<19}) RMSE ₹{m['rmse']:>6,.0f} {target_rent}  R²={m['r2']:.3f} {target_r2}│")

    # Grocery
    g_msg = f"Daily basket ₹{daily_total:.2f}  →  ₹{daily_total*30:.0f}/month"
    print(f"│  Grocery  {g_msg:<47}│")

    # CPI
    arima_mape = cpi_meta.get("arima_mape")
    lin_mape   = cpi_meta.get("linear_mape")
    if arima_mape is not None:
        target_cpi = "✅" if arima_mape < 5 else "⚠️ "
        print(f"│  CPI ARIMA(1,1,1)           MAPE={arima_mape:.2f}% {target_cpi:<26}│")
    print(f"│  CPI Linear Baseline        MAPE={lin_mape:.2f}%{' ' * 28}│")

    # Metro (loaded from saved meta)
    import joblib
    metro_meta = joblib.load(Path(__file__).parent / "models" / "metro_meta.joblib")
    m_acc = metro_meta["loo_accuracy"]
    target_metro = "✅" if m_acc > 0.9 else "⚠️ "
    print(f"│  Metro ({metro_meta['best_name']:<20}) LOO-Acc={m_acc:.3f} {target_metro:<18}│")

    # CoL Index
    print(f"│  CoL Composite Index:  {col_result['col_index']:.1f}  (100 = Jan 2021 baseline)      │")
    print(f"│  Total monthly cost:   ₹{col_result['total_monthly_inr']:,.0f}/month                 │")
    print("├" + "─" * 58 + "┤")
    print(f"│  ⏱  Training completed in {elapsed:.1f}s{' ' * (29 - len(f'{elapsed:.1f}'))}│")
    print("└" + "─" * 58 + "┘")


if __name__ == "__main__":
    run_all()
