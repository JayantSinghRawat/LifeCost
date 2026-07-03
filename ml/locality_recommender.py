"""
locality_recommender.py
=======================
Recommends the best Bhopal localities based on user profile:
  - Monthly salary (₹)
  - Workplace area (locality name)
  - BHK preference
  - Lifestyle priority (budget / comfort / premium)
  - Commute tolerance (km)

Algorithm:
  1. Build a locality profile from OLX rent data
     (median rent, listing count, rent range, affordability tier)
  2. Score each locality using a weighted multi-criteria formula:
       - Budget fit        (40%)  — does median rent fit 25% of salary rule?
       - Commute score     (30%)  — distance to workplace (Bhopal locality map)
       - Availability      (15%)  — number of listings (proxy for options)
       - Lifestyle match   (15%)  — premium / value / family index
  3. Return ranked top-N recommendations with explanations

Saved artefact: ml/models/locality_meta.joblib
"""

from __future__ import annotations

import json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
DATA   = BASE / "Scrapping_Manual" / "renting" / "listings.json"
MODEL  = BASE / "ml" / "models" / "locality_meta.joblib"

# ── Known workplace hubs in Bhopal & their approx coordinates ─────────────────
# (lat, lng)  — used to estimate relative distances between locality clusters
WORKPLACE_HUBS: dict[str, tuple[float, float]] = {
    "MP NAGAR":          (23.2340, 77.4340),
    "ARERA COLONY":      (23.2175, 77.4409),
    "GOVINDPURA":        (23.2656, 77.4724),
    "NEW MARKET":        (23.2289, 77.4081),
    "HABIBGANJ":         (23.2291, 77.4357),
    "BHOPAL RAILWAY":    (23.2677, 77.4118),
    "KOLAR ROAD":        (23.1705, 77.4536),
    "NISHATPURA":        (23.2789, 77.4033),
    "BAIRAGARH":         (23.2954, 77.3672),
    "MANDIDEEP":         (23.1056, 77.5321),
    "BERASIA":           (23.6336, 77.4382),
    "UNIVERSITY":        (23.2755, 77.3975),
    "AIIMS BHOPAL":      (23.1987, 77.5310),
    "IT PARK":           (23.1890, 77.4870),
    "BHEL":              (23.2710, 77.3550),
    "OTHER":             (23.2599, 77.4126),  # city centre fallback
}

# ── Locality cluster map (locality → approximate coords) ──────────────────────
LOCALITY_COORDS: dict[str, tuple[float, float]] = {
    "KOLAR ROAD":              (23.1705, 77.4536),
    "ARERA COLONY":            (23.2175, 77.4409),
    "AYODHYA NAGAR":           (23.2544, 77.4698),
    "ASHOKA GARDEN":           (23.2406, 77.4218),
    "ROHIT NAGAR":             (23.2060, 77.4481),
    "BHANPURA":                (23.2770, 77.4580),
    "AWADHPURI":               (23.2056, 77.4270),
    "GULMOHAR COLONY":         (23.2223, 77.4258),
    "INDRAPURI C SECTOR":      (23.2472, 77.4159),
    "KAROD KALAN":             (23.1906, 77.4693),
    "SHAHPURA":                (23.2042, 77.4386),
    "KATARA HILLS":            (23.2089, 77.4494),
    "NEW MINAL RESIDENCY":     (23.2534, 77.4325),
    "KOKTA":                   (23.2918, 77.4622),
    "CHUNABHATTI":             (23.2265, 77.4562),
    "AAMAR ESTATE":            (23.2040, 77.4210),
    "MP NAGAR":                (23.2340, 77.4340),
    "AAKRITI ECOCITY":         (23.1890, 77.4870),
    "MISROD":                  (23.2045, 77.5080),
    "RAJEEV NAGAR":            (23.2600, 77.3910),
    "SANJEEV NAGAR":           (23.2430, 77.4500),
    "SAKET NAGAR":             (23.2390, 77.4220),
    "DANISH SQUARE":           (23.2660, 77.4460),
    "INDRAPURI":               (23.2410, 77.4196),
    "BAGH MUNGALIYA":          (23.2305, 77.4145),
    "BAWARIYA KALAN":          (23.2590, 77.3810),
    "NEHRU NAGAR":             (23.2571, 77.4023),
    "OLD SUBHASH NAGAR":       (23.2490, 77.4145),
    "AYODHYA NAGAR EXTENTION": (23.2600, 77.4740),
    "BARKHERI":                (23.2780, 77.4890),
    "NANBI BAGH":              (23.2380, 77.4070),
    "PATEL NAGAR":             (23.2510, 77.4565),
    "RUCHI LIFE SCAPES":       (23.1950, 77.4720),
    "NEW MARKET TT NAGAR":     (23.2289, 77.4081),
    "ISRO COLONY":             (23.2126, 77.4612),
    "GOVINDPURA":              (23.2656, 77.4724),
    "HABIBGANJ":               (23.2291, 77.4357),
    "NISHATPURA":              (23.2789, 77.4033),
}

# ── Lifestyle tags ─────────────────────────────────────────────────────────────
LOCALITY_LIFESTYLE: dict[str, list[str]] = {
    "ARERA COLONY":     ["premium", "established", "urban", "family"],
    "MP NAGAR":         ["commercial", "urban", "well-connected"],
    "KOLAR ROAD":       ["affordable", "mid-range", "residential", "family"],
    "ASHOKA GARDEN":    ["value", "budget", "residential"],
    "HABIBGANJ":        ["urban", "railway", "well-connected"],
    "NEW MARKET TT NAGAR": ["commercial", "urban", "central"],
    "AYODHYA NAGAR":    ["mid-range", "residential", "family"],
    "AAKRITI ECOCITY":  ["premium", "gated", "modern"],
    "SHAHPURA":         ["residential", "family", "mid-range"],
    "KATARA HILLS":     ["mid-range", "hillside", "residential"],
    "ROHIT NAGAR":      ["mid-range", "family", "residential"],
    "BAGH MUNGALIYA":   ["central", "mid-range", "urban"],
    "DANISH SQUARE":    ["mid-range", "residential"],
    "ISRO COLONY":      ["residential", "family", "scientific-community"],
    "GOVINDPURA":       ["industrial", "mid-range", "commercial"],
    "BAWARIYA KALAN":   ["premium", "well-connected", "residential"],
    "GULMOHAR COLONY":  ["residential", "mid-range", "family"],
    "AWADHPURI":        ["budget", "residential"],
    "NEW MINAL RESIDENCY": ["gated", "mid-range", "modern", "residential"],
    "KOKTA":            ["budget", "outskirts", "affordable"],
    "NISHATPURA":       ["industrial", "affordable"],
    "MISROD":           ["budget", "outskirts"],
    "INDRAPURI":        ["budget", "affordable"],
    "NEHRU NAGAR":      ["central", "mid-range"],
    "BHANPURA":         ["mid-range", "residential", "family"],
}

LIFESTYLE_PRIORITY_MAP = {
    "budget":    ["budget", "affordable", "value"],
    "comfort":   ["mid-range", "residential", "family", "gated", "modern"],
    "premium":   ["premium", "established", "urban", "well-connected"],
    "family":    ["family", "residential", "gated", "mid-range"],
    "connected": ["urban", "well-connected", "commercial", "central"],
}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def build_locality_profiles() -> dict:
    """Parse OLX listings → per-locality rent statistics."""
    with open(DATA) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else list(data.values())

    locality_prices: dict[str, list[int]] = defaultdict(list)
    bhk_data: dict[str, list[int]]        = defaultdict(list)

    def parse_price(p) -> Optional[int]:
        try:   return int(str(p).replace('₹','').replace(',','').replace(' ','').strip())
        except: return None

    def parse_bhk(details: str) -> Optional[int]:
        try:
            part = details.strip().split()[0]
            return int(part)
        except: return None

    for item in items:
        loc_raw = item.get('location') or ''
        parts   = [x.strip() for x in loc_raw.split(',')]
        if len(parts) < 2: continue
        loc  = parts[0]
        city = parts[-1]
        if city != 'BHOPAL' or not loc: continue

        price = parse_price(item.get('price'))
        if not price or price < 1000 or price > 200000: continue
        locality_prices[loc].append(price)

        bhk = parse_bhk(item.get('details',''))
        if bhk: bhk_data[loc].append(bhk)

    profiles = {}
    for loc, prices in locality_prices.items():
        if len(prices) < 2: continue
        profiles[loc] = {
            "locality":      loc,
            "listing_count": len(prices),
            "median_rent":   int(statistics.median(prices)),
            "min_rent":      min(prices),
            "max_rent":      max(prices),
            "avg_bhk":       round(statistics.mean(bhk_data[loc]), 1) if bhk_data[loc] else 2.0,
            "coords":        LOCALITY_COORDS.get(loc),
            "lifestyle":     LOCALITY_LIFESTYLE.get(loc, ["residential"]),
        }

    return profiles


def train_locality_recommender():
    """Build and save locality profiles + metadata."""
    profiles = build_locality_profiles()
    # Attach affordability tier
    rents = [p["median_rent"] for p in profiles.values()]
    p33   = statistics.quantiles(rents, n=3)[0]
    p66   = statistics.quantiles(rents, n=3)[1]
    for p in profiles.values():
        r = p["median_rent"]
        if r <= p33:   p["tier"] = "budget"
        elif r <= p66: p["tier"] = "mid-range"
        else:           p["tier"] = "premium"

    meta = {
        "profiles":    profiles,
        "workplace_hubs": WORKPLACE_HUBS,
        "n_localities":   len(profiles),
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta, MODEL)
    print(f"[Locality] {len(profiles)} Bhopal localities profiled → {MODEL}")
    return meta


def recommend_locality(
    salary_monthly: float,
    workplace: str = "OTHER",
    bhk: int = 2,
    lifestyle: str = "comfort",
    commute_tolerance_km: float = 15.0,
    top_n: int = 5,
) -> list[dict]:
    """
    Score and rank Bhopal localities for a user profile.

    Parameters
    ----------
    salary_monthly       Monthly take-home salary (₹)
    workplace            Name of workplace hub (e.g. 'MP NAGAR', 'GOVINDPURA')
    bhk                  Preferred BHK size (1, 2, 3 …)
    lifestyle            Priority: 'budget' | 'comfort' | 'premium' | 'family' | 'connected'
    commute_tolerance_km Max acceptable commute distance in km
    top_n                Number of recommendations to return

    Returns  list of locality dicts ranked by composite score
    """
    import joblib as _jl
    meta     = _jl.load(MODEL)
    profiles = meta["profiles"]
    hubs     = meta["workplace_hubs"]

    # Budget rule: rent should be ≤ 30% of monthly salary
    max_rent = salary_monthly * 0.30
    budget_hard_limit = salary_monthly * 0.40  # hard cap

    # Workplace coordinates
    workplace_key = workplace.upper().strip()
    wp_coords = hubs.get(workplace_key, hubs["OTHER"])

    # Lifestyle tags we want
    lifestyle_tags = LIFESTYLE_PRIORITY_MAP.get(lifestyle, ["residential", "mid-range"])

    results = []
    for loc, prof in profiles.items():
        median_rent = prof["median_rent"]
        # Hard budget filter (must be within 40% of salary)
        if median_rent > budget_hard_limit: continue

        # ── Score components ────────────────────────────────────────────────
        # 1. Budget fit (40%) — closer to 25% rule = better
        ideal_rent = salary_monthly * 0.25
        rent_diff  = abs(median_rent - ideal_rent) / ideal_rent
        budget_score = max(0.0, 1.0 - rent_diff)   # 0–1

        # 2. Commute score (30%)
        loc_coords = prof.get("coords")
        if loc_coords:
            dist_km = _haversine(*loc_coords, *wp_coords)
        else:
            dist_km = 15.0   # unknown → pessimistic default
        commute_score = max(0.0, 1.0 - dist_km / commute_tolerance_km)

        # 3. Listing availability (15%) — more listings = more choice
        avail_score = min(prof["listing_count"] / 30.0, 1.0)

        # 4. Lifestyle match (15%)
        tags = prof.get("lifestyle", [])
        hits = sum(1 for t in lifestyle_tags if t in tags)
        lifestyle_score = min(hits / max(len(lifestyle_tags), 1), 1.0)

        # ── Weighted composite ───────────────────────────────────────────────
        composite = (
            0.40 * budget_score   +
            0.30 * commute_score  +
            0.15 * avail_score    +
            0.15 * lifestyle_score
        )

        results.append({
            "locality":         loc,
            "median_rent":      median_rent,
            "min_rent":         prof["min_rent"],
            "max_rent":         prof["max_rent"],
            "tier":             prof["tier"],
            "listing_count":    prof["listing_count"],
            "avg_bhk":          prof["avg_bhk"],
            "lifestyle_tags":   prof["lifestyle"],
            "distance_to_work_km": round(dist_km, 1) if loc_coords else None,
            "budget_fit_pct":   round(median_rent / salary_monthly * 100, 1),
            "scores": {
                "budget":    round(budget_score, 3),
                "commute":   round(commute_score, 3),
                "availability": round(avail_score, 3),
                "lifestyle": round(lifestyle_score, 3),
                "composite": round(composite, 3),
            },
            "why": _explain(loc, median_rent, salary_monthly, dist_km, lifestyle_tags, prof),
        })

    results.sort(key=lambda x: -x["scores"]["composite"])
    return results[:top_n]


def _explain(loc, rent, salary, dist, lifestyle_tags, prof) -> str:
    """Generate a one-line human-readable explanation."""
    pct = round(rent / salary * 100)
    parts = []
    if pct <= 25:
        parts.append(f"rent is only {pct}% of salary (excellent value)")
    elif pct <= 30:
        parts.append(f"rent is {pct}% of salary (within budget)")
    else:
        parts.append(f"rent is {pct}% of salary (slightly above ideal)")
    if dist < 5:
        parts.append("very close to workplace")
    elif dist < 10:
        parts.append("comfortable commute")
    elif dist < 20:
        parts.append("manageable commute")
    else:
        parts.append("longer commute")
    matched = [t for t in lifestyle_tags if t in prof.get("lifestyle", [])]
    if matched:
        parts.append(f"matches your {'/'.join(matched[:2])} lifestyle")
    return "; ".join(parts).capitalize()


if __name__ == "__main__":
    train_locality_recommender()
