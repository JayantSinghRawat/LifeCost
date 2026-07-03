"""
data_prep.py — MP Cost of Living ML Platform
Parse and clean all 4 scraped datasets into model-ready DataFrames.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path constants ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent / "Scrapping_Manual"
RENT_PATH    = BASE / "renting"   / "listings.json"
GROCERY_PATH = BASE / "grocery"   / "Bhopal_blinkit_results_462010.json"
CPI_PATH     = BASE / "cpi"       / "bhopal_cpi_clean.xlsx"
METRO_PATH   = BASE / "Metro"     / "bhopal_metro_fares.json"

# ── Month lookup ──────────────────────────────────────────────────────────────
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ── 1. RENTING ────────────────────────────────────────────────────────────────
def load_rent() -> pd.DataFrame:
    """Parse OLX rental listings into a clean DataFrame."""
    with open(RENT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        details  = str(item.get("details") or "")
        price_s  = str(item.get("price") or "")
        location = str(item.get("location") or "")

        # BHK / RK type
        m_bhk = re.search(r"(\d+)\s*(BHK|RK|BRK)", details, re.I)
        bhk = int(m_bhk.group(1)) if m_bhk else np.nan
        bhk_type = m_bhk.group(2).upper() if m_bhk else "UNK"

        # Bathrooms
        m_bath = re.search(r"(\d+)\s*Bathroom", details, re.I)
        bathrooms = int(m_bath.group(1)) if m_bath else np.nan

        # Sqft
        m_sqft = re.search(r"(\d[\d,]*)\s*sqft", details, re.I)
        sqft = float(m_sqft.group(1).replace(",", "")) if m_sqft else np.nan

        # Price → float
        m_price = re.search(r"[\d,]+", price_s.replace(" ", ""))
        rent = float(m_price.group().replace(",", "")) if m_price else np.nan

        # City (last comma-separated token)
        city = location.split(",")[-1].strip().upper() if "," in location else location.strip().upper()

        rows.append(
            dict(bhk=bhk, bhk_type=bhk_type, bathrooms=bathrooms, sqft=sqft, city=city, rent=rent)
        )

    df = pd.DataFrame(rows)

    # Drop rows missing critical features or target
    df = df.dropna(subset=["bhk", "sqft", "rent"])
    # Cap obvious outliers (very low / very high rents)
    df = df[(df["rent"] >= 1_000) & (df["rent"] <= 500_000)]
    df = df[(df["sqft"] >= 100) & (df["sqft"] <= 20_000)]
    # BHK cap at 5
    df = df[df["bhk"].between(1, 5)]
    df["bhk"] = df["bhk"].astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(df["bhk"]).astype(int)

    # Encode city — keep top-N cities, rest → 'OTHER'
    top_cities = df["city"].value_counts().head(10).index.tolist()
    df["city"] = df["city"].apply(lambda c: c if c in top_cities else "OTHER")

    return df.reset_index(drop=True)


# ── 2. GROCERY ────────────────────────────────────────────────────────────────
def load_grocery() -> pd.DataFrame:
    """Parse Blinkit grocery items into a clean DataFrame."""
    with open(GROCERY_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        price_s = str(item.get("price") or "")
        if "₹" not in price_s:
            continue

        # Price
        m_price = re.search(r"[\d.]+", price_s.replace(",", ""))
        if not m_price:
            continue
        price = float(m_price.group())

        # Quantity → numeric ml / g
        qty_s = str(item.get("quantity") or "").lower()
        m_qty = re.search(r"([\d.]+)\s*(kg|g|ltr|l|ml|pcs?|dozen|nos?)", qty_s, re.I)
        if m_qty:
            val  = float(m_qty.group(1))
            unit = m_qty.group(2).lower()
            # Normalise everything to grams / ml (same scale)
            if unit == "kg":
                val *= 1000
            elif unit in ("ltr", "l"):
                val *= 1000
            elif unit in ("pcs", "pc", "nos", "no", "dozen"):
                val *= 100  # assign 100 g fictional weight per piece
            qty_norm = val
        else:
            qty_norm = np.nan

        # Price per 100 g/ml
        price_per_100 = (price / qty_norm * 100) if (qty_norm and qty_norm > 0) else np.nan

        rows.append(
            dict(
                category=str(item.get("term") or ""),
                name=str(item.get("name") or ""),
                quantity_raw=qty_s,
                quantity_norm_g=qty_norm,
                price=price,
                price_per_100g=price_per_100,
            )
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["price"])
    # Drop extreme outliers (sanity: price per 100g > ₹2000 → data error)
    df = df[df["price"] <= 2000]
    return df.reset_index(drop=True)


# ── 3. CPI  ──────────────────────────────────────────────────────────────────
def load_cpi(group: str = "General Index") -> pd.DataFrame:
    """
    Load and feature-engineer CPI data for time-series modelling.

    The scraped CPI file contains 7 CPI groups (Food & Beverages, Housing, etc.)
    for Bhopal.  By default we use the 'General Index' (headline CPI).
    If requested group is missing, we average across all groups per month.

    NOTE: Only Jan & Feb are scraped for each year (2021-2025) → 10 rows max.
    ARIMA requires ≥15 pts, so we fall back to a linear lag model automatically.
    """
    df = pd.read_excel(CPI_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Index": "cpi", "Year": "year", "Month": "month_name"})
    df["month"] = df["month_name"].map(MONTH_MAP)
    df = df.dropna(subset=["cpi", "year", "month"])

    # Filter to requested group; fall back to mean across groups
    if "Group" in df.columns and group in df["Group"].values:
        df = df[df["Group"] == group].copy()
    elif "Group" in df.columns:
        df = (
            df.groupby(["year", "month", "month_name"], as_index=False)["cpi"]
            .mean()
        )

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
    )
    df = df.set_index("date")
    df = df[~df.index.duplicated(keep="last")]

    # Lag features
    df["lag_1"]   = df["cpi"].shift(1)
    df["lag_3"]   = df["cpi"].shift(3)
    df["time_idx"] = range(len(df))

    return df[["cpi", "lag_1", "lag_3", "time_idx"]].dropna()


# ── 4. METRO ─────────────────────────────────────────────────────────────────
def load_metro() -> pd.DataFrame:
    """Parse Bhopal metro fare data into a clean DataFrame."""
    with open(METRO_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        # Distance
        m_dist = re.search(r"([\d.]+)", str(item.get("Distance") or ""))
        distance_km = float(m_dist.group(1)) if m_dist else np.nan

        # Stops (num before keyword)
        m_stops = re.search(r"(\d+)\s*stop", str(item.get("Stations") or ""), re.I)
        num_stops = int(m_stops.group(1)) if m_stops else np.nan

        # Travel time (minutes)
        m_time = re.search(r"(\d+)\s*minute", str(item.get("Est. Time") or ""), re.I)
        travel_min = int(m_time.group(1)) if m_time else np.nan

        # Fare
        m_fare = re.search(r"(\d+)", str(item.get("Est. Fare") or ""))
        fare = int(m_fare.group(1)) if m_fare else np.nan

        rows.append(
            dict(
                from_station=item.get("From", ""),
                to_station=item.get("To", ""),
                distance_km=distance_km,
                num_stops=num_stops,
                travel_min=travel_min,
                fare=fare,
            )
        )

    df = pd.DataFrame(rows)
    df = df.dropna()
    df["fare"] = df["fare"].astype(int)
    return df.reset_index(drop=True)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    rent    = load_rent()
    grocery = load_grocery()
    cpi     = load_cpi()
    metro   = load_metro()

    print("─" * 60)
    print(f"🏠  Rent      : {len(rent):>4} rows | columns: {list(rent.columns)}")
    print(f"🛒  Grocery   : {len(grocery):>4} rows | columns: {list(grocery.columns)}")
    print(f"📈  CPI       : {len(cpi):>4} rows | columns: {list(cpi.columns)}")
    print(f"🚇  Metro     : {len(metro):>4} rows | columns: {list(metro.columns)}")
    print("─" * 60)
    print("\nRent stats:\n", rent[["bhk", "sqft", "rent"]].describe())
    print("\nGrocery stats:\n", grocery[["price", "price_per_100g"]].describe())
    print("\nCPI stats:\n", cpi["cpi"].describe())
    print("\nMetro fare distribution:\n", metro["fare"].value_counts())
