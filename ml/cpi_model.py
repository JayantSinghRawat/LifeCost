"""
cpi_model.py — CPI Inflation Trend Forecaster
Uses ARIMA(1,1,1) on Bhopal Food & Beverages CPI (base year 2016).
Falls back to LinearRegression on lag features if statsmodels fails.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import joblib

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_cpi

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_cpi_model(verbose=True):
    df = load_cpi()
    series = df["cpi"]

    if verbose:
        print(f"[CPI] {len(series)} monthly data points  ({series.index[0].date()} → {series.index[-1].date()})")
        print(f"      Range: {series.min():.1f} – {series.max():.1f}  |  Mean: {series.mean():.1f}")

    # ── ARIMA ────────────────────────────────────────────────────────────────
    arima_model = None
    arima_forecast = None
    arima_mape = None

    if HAS_ARIMA and len(series) >= 20:
        try:
            # Ensure monthly frequency so statsmodels handles date index correctly
            series = series.asfreq("MS")

            # Train/test split: last 6 months as test
            train_s = series.iloc[:-6]
            test_s  = series.iloc[-6:]

            arima_fit   = ARIMA(train_s, order=(1, 1, 1)).fit()
            test_pred   = arima_fit.forecast(steps=6)
            arima_mape  = mean_absolute_percentage_error(test_s, test_pred) * 100

            # Refit on full series for forecasting
            arima_full  = ARIMA(series, order=(1, 1, 1)).fit()
            arima_forecast = arima_full.forecast(steps=24)
            arima_model = arima_full

            if verbose:
                print(f"\n  ARIMA(1,1,1)  | MAPE={arima_mape:.2f}%")
                print(f"  24-month forecast (from {series.index[-1].date()}):")
                for dt, val in zip(pd.date_range(series.index[-1], periods=25, freq="MS")[1:], arima_forecast):
                    print(f"    {dt.strftime('%b %Y')} → {val:.1f}")
        except Exception as e:
            if verbose:
                print(f"  ARIMA failed ({e}), falling back to linear baseline.")

    # ── Linear Regression fallback ─────────────────────────────────────────
    df_ml    = df[["cpi", "lag_1", "lag_3", "time_idx"]].dropna()
    X_lin    = df_ml[["lag_1", "lag_3", "time_idx"]].values
    y_lin    = df_ml["cpi"].values
    lin_model = LinearRegression().fit(X_lin, y_lin)
    lin_pred  = lin_model.predict(X_lin)
    lin_mape  = mean_absolute_percentage_error(y_lin, lin_pred) * 100

    if verbose:
        print(f"\n  Linear Regression (lag)  | MAPE={lin_mape:.2f}%")

    # ── Generate 24-month linear forecast ───────────────────────────────────
    last_row  = df_ml.iloc[-1]
    lin_fcst  = []
    l1 = last_row["lag_1"]
    l0 = last_row["cpi"]
    t  = last_row["time_idx"]
    for i in range(1, 25):
        xi = np.array([[l0, l1, t + i]])
        yi = float(lin_model.predict(xi)[0])
        lin_fcst.append(yi)
        l1, l0 = l0, yi

    lin_forecast_dates = pd.date_range(series.index[-1], periods=25, freq="MS")[1:]
    lin_forecast_series = pd.Series(lin_fcst, index=lin_forecast_dates)

    if verbose and arima_forecast is None:
        print(f"  24-month linear forecast (from {series.index[-1].date()}):")
        for dt, val in zip(lin_forecast_dates, lin_fcst):
            print(f"    {dt.strftime('%b %Y')} → {val:.1f}")

    # Prefer ARIMA forecast if available
    final_forecast = arima_forecast if arima_forecast is not None else lin_forecast_series

    # ── Save ──────────────────────────────────────────────────────────────
    meta = {
        "arima_model":        arima_model,
        "linear_model":       lin_model,
        "arima_mape":         arima_mape,
        "linear_mape":        lin_mape,
        "forecast_24m":        final_forecast,
        "last_actual_date":   series.index[-1],
        "last_actual_cpi":    float(series.iloc[-1]),
        "historical_series":  series,
    }
    meta_path = MODELS_DIR / "cpi_meta.joblib"
    joblib.dump(meta, meta_path)
    if verbose:
        print(f"\n💾 Saved → {meta_path}")
    return meta


def get_cpi_forecast() -> pd.Series:
    """Return 24-month CPI forecast Series.  Run train_cpi_model() first."""
    meta = joblib.load(MODELS_DIR / "cpi_meta.joblib")
    return meta["forecast_24m"]


if __name__ == "__main__":
    print("=" * 60)
    print("  📈  CPI INFLATION FORECASTER")
    print("=" * 60)
    train_cpi_model(verbose=True)
    
    print("\n--- Interactive Forecast ---")
    try:
        input("Press Enter to fetch the latest 24-month forecast (or Ctrl+C to exit)...")
        forecast = get_cpi_forecast()
        print("\n✅ CPI Forecast:")
        for date, val in forecast.items():
            print(f"  {date.strftime('%b %Y')} → {val:.1f}")
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
