"""
main.py — MP Cost of Living FastAPI REST Server
Serves all 5 ML model predictions via clean, documented HTTP endpoints.

Run:
    uvicorn api.main:app --reload --port 8000

Docs available at: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure ml/ is importable from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MP Cost of Living API",
    description=(
        "AI-powered cost-of-living insights for Madhya Pradesh, India.\n\n"
        "Provides rent predictions, grocery basket estimates, CPI forecasts, "
        "Bhopal metro fare classification, and a composite cost-of-living index."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model cache ──────────────────────────────────────────────────────────
_models: Dict[str, Any] = {}

def _ensure_models_loaded():
    """Load all model artefacts once, cache in memory."""
    if _models:
        return
    import joblib
    models_dir = PROJECT_ROOT / "ml" / "models"
    required = [
        "rent_model", "rent_meta",
        "grocery_meta",
        "cpi_meta",
        "metro_model", "metro_meta",
        "locality_meta",
    ]
    for name in required:
        path = models_dir / f"{name}.joblib"
        if not path.exists():
            raise RuntimeError(
                f"Model file not found: {path}\n"
                "Run 'python3 ml/train_all.py' first."
            )
        _models[name] = joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════════

class RentRequest(BaseModel):
    bhk: int = Field(..., ge=1, le=6, description="Number of BHK rooms (1-6)")
    bathrooms: int = Field(..., ge=1, le=6, description="Number of bathrooms")
    sqft: float = Field(..., gt=0, le=20_000, description="Area in square feet")
    city: str = Field("BHOPAL", description="City name (e.g. BHOPAL, DEWAS, SAGAR)")

    @field_validator("city")
    @classmethod
    def upper_city(cls, v: str) -> str:
        return v.strip().upper()


class RentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_rent_inr: float
    bhk: int
    bathrooms: int
    sqft: float
    city: str
    model_used: str
    note: str


class GroceryResponse(BaseModel):
    basket: Dict[str, Any]
    total_daily_inr: float
    total_monthly_inr: float
    note: str


class CpiResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    latest_cpi: float
    cpi_group: str
    date_of_latest: str
    forecast_24m: Dict[str, float]
    model_used: str
    mape_pct: Optional[float]


class MetroRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=50, description="Route distance in km")
    num_stops: int = Field(..., ge=0, le=30, description="Number of stops")
    travel_min: Optional[float] = Field(None, description="Travel time in minutes (auto-estimated if omitted)")


class MetroResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_fare_inr: int
    distance_km: float
    num_stops: int
    travel_min: float
    model_used: str
    decision_rule: str


class CompositeRequest(BaseModel):
    rent_monthly: Optional[float] = Field(None, description="Override rent (₹/month)")
    grocery_daily: Optional[float] = Field(None, description="Override daily grocery cost (₹)")
    avg_metro_fare: float = Field(20.0, description="Average metro fare per trip (₹)")
    bhk: int = Field(1, ge=1, le=6, description="BHK size for rent prediction")
    sqft: float = Field(500.0, gt=0, description="Sqft for rent prediction")
    city: str = Field("BHOPAL", description="City for rent prediction")

    @field_validator("city")
    @classmethod
    def upper_city(cls, v: str) -> str:
        return v.strip().upper()


class CompositeResponse(BaseModel):
    city: str
    rent_monthly: float
    grocery_monthly: float
    transport_monthly: float
    total_monthly_inr: float
    cpi_latest: float
    cpi_factor: float
    col_index: float
    weights: Dict[str, float]
    note: str


class LocalityRequest(BaseModel):
    salary_monthly: float = Field(..., gt=0, description="Monthly take-home salary (₹)")
    workplace: str = Field("OTHER", description="Workplace hub name (e.g. MP NAGAR, GOVINDPURA, AIIMS BHOPAL)")
    bhk: int = Field(2, ge=1, le=4, description="Preferred BHK size")
    lifestyle: str = Field("comfort", description="Priority: budget | comfort | premium | family | connected")
    commute_tolerance_km: float = Field(15.0, gt=0, le=50, description="Max acceptable commute distance in km")
    top_n: int = Field(5, ge=1, le=10, description="Number of recommendations to return")


class LocalityResult(BaseModel):
    locality: str
    median_rent: int
    min_rent: int
    max_rent: int
    tier: str
    listing_count: int
    avg_bhk: float
    lifestyle_tags: list[str]
    distance_to_work_km: Optional[float]
    budget_fit_pct: float
    scores: Dict[str, float]
    why: str


class LocalityResponse(BaseModel):
    recommendations: list[LocalityResult]
    user_profile: Dict[str, Any]
    workplace_hubs: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Meta"])
def health():
    """Server health check. Also reports whether models are loaded."""
    loaded = bool(_models)
    return {"status": "ok", "models_loaded": loaded, "timestamp": time.time()}


@app.post("/train", tags=["Meta"])
def train_models():
    """
    Trigger a fresh training run for all models.
    This may take up to 30 seconds.
    """
    _models.clear()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "ml"))
        import importlib
        for mod_name in ["data_prep","rent_model","grocery_model","cpi_model","metro_model","composite_index","evaluate"]:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        from evaluate import run_all
        run_all()
        _models.clear()   # flush cache so next request reloads fresh files
        return {"status": "training_complete"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /predict/rent ─────────────────────────────────────────────────────────────
@app.post("/predict/rent", response_model=RentResponse, tags=["Predictions"])
def predict_rent(req: RentRequest):
    """
    Predict monthly rent (₹) for a residential property in MP.

    **Model**: RandomForest trained on 952 OLX listings.
    **Features**: BHK count, bathrooms, area (sqft), city.
    """
    try:
        _ensure_models_loaded()
        import numpy as np

        model     = _models["rent_model"]
        feat_cols = _models["rent_meta"]["feature_cols"]
        best_name = _models["rent_meta"]["best"]

        row = {"bhk": req.bhk, "bathrooms": req.bathrooms, "sqft": req.sqft}
        for col in feat_cols:
            if col.startswith("city_"):
                row[col] = 1 if col[len("city_"):] == req.city else 0
        X = np.array([[row.get(c, 0) for c in feat_cols]])
        pred = float(model.predict(X)[0])
        pred = max(pred, 0)

        return RentResponse(
            predicted_rent_inr=round(pred, 2),
            bhk=req.bhk,
            bathrooms=req.bathrooms,
            sqft=req.sqft,
            city=req.city,
            model_used=best_name,
            note=(
                "Prediction is Bhopal-centric (900/952 training samples). "
                "Accuracy lower for other MP cities."
            ) if req.city != "BHOPAL" else "Based on OLX listings data.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict/grocery ──────────────────────────────────────────────────────────
@app.get("/predict/grocery", response_model=GroceryResponse, tags=["Predictions"])
def predict_grocery():
    """
    Get the estimated daily and monthly grocery basket cost for Bhopal.

    **Basket**: 500ml milk, 4 bread slices, 2 eggs, 200g vegetables, 150g fruits.
    **Source**: Blinkit (quick-commerce) prices — includes delivery premium.
    """
    try:
        _ensure_models_loaded()
        meta = _models["grocery_meta"]
        return GroceryResponse(
            basket=meta["basket"],
            total_daily_inr=round(meta["total_daily"], 2),
            total_monthly_inr=round(meta["total_daily"] * 30, 2),
            note=(
                "Prices sourced from Blinkit (Bhopal pin 462010). "
                "Quick-commerce prices may be 10-20% higher than local market."
            ),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict/cpi ──────────────────────────────────────────────────────────────
@app.get("/predict/cpi", response_model=CpiResponse, tags=["Predictions"])
def predict_cpi():
    """
    CPI (Consumer Price Index) latest value and 24-month forecast for Bhopal.

    **Data**: Bhopal General Index from Labour Bureau (base year 2016).
    **Model**: Linear Regression on lag features (MAPE ~1.3%).
    """
    try:
        _ensure_models_loaded()
        meta = _models["cpi_meta"]
        forecast = meta["forecast_24m"]

        # Serialise forecast — index may be DatetimeIndex or int
        try:
            fcst_dict = {str(k.strftime("%b %Y")): round(float(v), 2) for k, v in forecast.items()}
        except AttributeError:
            fcst_dict = {f"Month+{i+1}": round(float(v), 2) for i, v in enumerate(forecast)}

        mape = meta.get("arima_mape") or meta.get("linear_mape")
        used = "ARIMA(1,1,1)" if meta.get("arima_mape") else "LinearRegression (lag features)"

        return CpiResponse(
            latest_cpi=round(float(meta["last_actual_cpi"]), 2),
            cpi_group="General Index",
            date_of_latest=str(meta["last_actual_date"])[:10] if hasattr(meta["last_actual_date"], "strftime") else str(meta["last_actual_date"]),
            forecast_24m=fcst_dict,
            model_used=used,
            mape_pct=round(float(mape), 2) if mape else None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict/metro ────────────────────────────────────────────────────────────
@app.post("/predict/metro", response_model=MetroResponse, tags=["Predictions"])
def predict_metro(req: MetroRequest):
    """
    Predict Bhopal metro fare tier (₹10 / ₹20 / ₹30).

    **Model**: DecisionTree — 100% Leave-One-Out accuracy on 42 routes.
    **Rule**: stops ≤ 3 → ₹10 | stops > 3 & time ≤ 11 min → ₹20 | else ₹30
    """
    try:
        _ensure_models_loaded()
        import numpy as np

        model    = _models["metro_model"]
        meta     = _models["metro_meta"]
        t_min    = req.travel_min if req.travel_min is not None else req.num_stops * 2.0

        X = np.array([[req.distance_km, req.num_stops, t_min]])
        if meta["use_scaler"] and meta["scaler"]:
            X = meta["scaler"].transform(X)
        fare = int(model.predict(X)[0])

        rule = (
            f"stops ≤ 3 → ₹10" if req.num_stops <= 3 else
            f"stops > 3 & time ≤ 11 min → ₹20" if t_min <= 11 else
            f"stops > 3 & time > 11 min → ₹30"
        )

        return MetroResponse(
            predicted_fare_inr=fare,
            distance_km=req.distance_km,
            num_stops=req.num_stops,
            travel_min=t_min,
            model_used=meta["best_name"],
            decision_rule=rule,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /composite-index ──────────────────────────────────────────────────────────
@app.post("/composite-index", response_model=CompositeResponse, tags=["Composite"])
def composite_index(req: CompositeRequest):
    """
    Compute the MP Cost-of-Living Composite Index.

    Combines rent (50%), grocery (30%), transport (5%), and CPI-adjustment (15%)
    into a single score normalised to **100 = January 2021 Bhopal baseline**.
    """
    try:
        _ensure_models_loaded()
        import numpy as np

        # --- Rent ----------------------------------------------------------
        if req.rent_monthly is not None:
            rent = req.rent_monthly
        else:
            model     = _models["rent_model"]
            feat_cols = _models["rent_meta"]["feature_cols"]
            row = {"bhk": req.bhk, "bathrooms": req.bhk, "sqft": req.sqft}
            for col in feat_cols:
                if col.startswith("city_"):
                    row[col] = 1 if col[len("city_"):] == req.city else 0
            X = np.array([[row.get(c, 0) for c in feat_cols]])
            rent = max(float(model.predict(X)[0]), 0)

        # --- Grocery -------------------------------------------------------
        if req.grocery_daily is not None:
            grocery_daily = req.grocery_daily
        else:
            grocery_daily = _models["grocery_meta"]["total_daily"]

        # --- CPI -----------------------------------------------------------
        cpi_latest = float(_models["cpi_meta"]["last_actual_cpi"])

        # --- Composite calculation (mirrors composite_index.py) ─────────────
        BASELINE = {
            "rent_1bhk_monthly":  7_500,
            "grocery_monthly":    1_500,
            "transport_10_trips":   200,
            "cpi_base":           113.9,   # Jan 2021 General Index Bhopal
        }
        WEIGHTS = {"rent": 0.50, "grocery": 0.30, "transport": 0.05, "cpi_adj": 0.15}

        grocery_monthly   = grocery_daily * 30
        transport_monthly = req.avg_metro_fare * 10
        total_raw         = rent + grocery_monthly + transport_monthly
        cpi_factor        = cpi_latest / BASELINE["cpi_base"]

        norm = {
            "rent":      rent              / BASELINE["rent_1bhk_monthly"],
            "grocery":   grocery_monthly   / BASELINE["grocery_monthly"],
            "transport": transport_monthly / BASELINE["transport_10_trips"],
            "cpi_adj":   cpi_factor,
        }
        col_index = sum(norm[k] * WEIGHTS[k] for k in WEIGHTS) * 100

        return CompositeResponse(
            city=req.city,
            rent_monthly=round(rent, 2),
            grocery_monthly=round(grocery_monthly, 2),
            transport_monthly=round(transport_monthly, 2),
            total_monthly_inr=round(total_raw, 2),
            cpi_latest=round(cpi_latest, 2),
            cpi_factor=round(cpi_factor, 4),
            col_index=round(col_index, 2),
            weights=WEIGHTS,
            note="Index 100 = January 2021 Bhopal baseline. Higher = more expensive.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /recommend/locality ────────────────────────────────────────────────────────
@app.post("/recommend/locality", response_model=LocalityResponse, tags=["Recommender"])
def recommend_locality(req: LocalityRequest):
    """
    Recommend the best Bhopal localities to stay in, based on your personal profile.

    **Scoring weights**: Budget fit 40% · Commute 30% · Listing availability 15% · Lifestyle 15%

    **Workplace hubs supported**: MP NAGAR, ARERA COLONY, GOVINDPURA, HABIBGANJ,
    KOLAR ROAD, AIIMS BHOPAL, BHEL, IT PARK, BAIRAGARH, MANDIDEEP, NISHATPURA,
    BHOPAL RAILWAY, NEW MARKET, UNIVERSITY, BERASIA, OTHER
    """
    try:
        _ensure_models_loaded()
        from locality_recommender import recommend_locality as _rec

        results = _rec(
            salary_monthly        = req.salary_monthly,
            workplace             = req.workplace,
            bhk                   = req.bhk,
            lifestyle             = req.lifestyle,
            commute_tolerance_km  = req.commute_tolerance_km,
            top_n                 = req.top_n,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No localities match your budget. Try increasing salary or commute tolerance."
            )

        hubs = list(_models["locality_meta"]["workplace_hubs"].keys())

        return LocalityResponse(
            recommendations=[LocalityResult(**r) for r in results],
            user_profile={
                "salary_monthly":       req.salary_monthly,
                "max_rent_budget":      round(req.salary_monthly * 0.30),
                "workplace":            req.workplace.upper(),
                "bhk":                  req.bhk,
                "lifestyle":            req.lifestyle,
                "commute_tolerance_km": req.commute_tolerance_km,
            },
            workplace_hubs=hubs,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/locality/workplaces", tags=["Recommender"])
def get_workplace_hubs():
    """List all supported workplace hub names for the locality recommender."""
    try:
        _ensure_models_loaded()
        return {"workplaces": list(_models["locality_meta"]["workplace_hubs"].keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/locality/map", tags=["Recommender"])
def get_map_data():
    """Get coordinates and basic stats for all localities and workplace hubs to plot on a map."""
    try:
        _ensure_models_loaded()
        meta = _models["locality_meta"]
        # 1. Load rent listings and scatter them around their locality center
        import random
        import json as _json
        import numpy as np
        from scipy.spatial import ConvexHull
        
        rentings = []
        loc_scattered_pts = {}
        try:
            listings_path = PROJECT_ROOT / "Scrapping_Manual" / "renting" / "listings.json"
            with open(listings_path) as f:
                items = _json.load(f)
                if not isinstance(items, list):
                    items = list(items.values())

                for item in items:
                    loc_raw = item.get("location", "")
                    parts = [x.strip() for x in loc_raw.split(',')]
                    if len(parts) < 2 or parts[-1] != "BHOPAL": continue
                    loc_name = parts[0]
                    
                    if loc_name in meta["profiles"] and meta["profiles"][loc_name].get("coords"):
                        base_lat, base_lng = meta["profiles"][loc_name]["coords"]
                        
                        # Add ~1.3km jitter (0.012 degrees)
                        lat_jitter = random.uniform(-0.012, 0.012)
                        lng_jitter = random.uniform(-0.012, 0.012)
                        
                        lat = base_lat + lat_jitter
                        lng = base_lng + lng_jitter
                        
                        rentings.append({
                            "title": item.get("title", "—"),
                            "price": item.get("price", "—"),
                            "lat": lat,
                            "lng": lng
                        })
                        if loc_name not in loc_scattered_pts:
                            loc_scattered_pts[loc_name] = []
                        loc_scattered_pts[loc_name].append([lat, lng])
        except Exception:
            pass # Gracefully handle if listings.json not found

        # 1.5 Prepare all points array for nearest-neighbor fallback hulls
        all_pts = []
        for r in rentings:
            all_pts.append([r["lat"], r["lng"]])
        all_pts_arr = np.array(all_pts) if all_pts else None

        # 2. Format localities and build Convex Hull polygons / Load OSM boundaries
        osm_boundaries = {}
        try:
            with open(PROJECT_ROOT / "Scrapping_Manual" / "osm_boundaries.json") as f:
                osm_boundaries = _json.load(f)
        except Exception:
            pass

        voronoi_boundaries = {}
        try:
            with open(PROJECT_ROOT / "Scrapping_Manual" / "voronoi_boundaries.json") as f:
                voronoi_boundaries = _json.load(f)
        except Exception:
            pass

        localities = []
        for loc_name, data in meta["profiles"].items():
            if data.get("coords"):
                loc_obj = {
                    "name": loc_name,
                    "lat": data["coords"][0],
                    "lng": data["coords"][1],
                    "median_rent": data.get("median_rent", 0),
                    "tier": data.get("tier", "unknown")
                }
                
                if loc_name in osm_boundaries:
                    loc_obj["osm_geojson"] = osm_boundaries[loc_name]
                elif loc_name in voronoi_boundaries:
                    loc_obj["osm_geojson"] = voronoi_boundaries[loc_name]
                    
                localities.append(loc_obj)
        
        # 3. Format workplace hubs
        hubs = []
        for loc_name, coords in meta["workplace_hubs"].items():
            hubs.append({
                "name": loc_name,
                "lat": coords[0],
                "lng": coords[1]
            })

        return {
            "localities": localities,
            "workplaces": hubs,
            "rentings": rentings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Data Explorer — raw scraped data endpoints
# ═══════════════════════════════════════════════════════════════════════════════

import json as _json

_DATA_DIR = PROJECT_ROOT / "Scrapping_Manual"


def _load_json(path: Path):
    with open(path) as f:
        return _json.load(f)


@app.get("/data/rent", tags=["Data Explorer"])
def get_rent_data(search: str = "", page: int = 1, per_page: int = 50):
    """
    Browse raw OLX rental listings (1,080 rows).
    Use `search` to filter by locality, BHK, city, etc.
    """
    try:
        items = _load_json(_DATA_DIR / "renting" / "listings.json")
        if not isinstance(items, list):
            items = list(items.values())

        # Normalise each row
        rows = []
        for item in items:
            rows.append({
                "title":    item.get("title", "—"),
                "price":    item.get("price", "—"),
                "details":  item.get("details", "—"),
                "location": item.get("location", "—"),
                "link":     item.get("link", ""),
            })

        # Search filter
        q = search.strip().lower()
        if q:
            rows = [r for r in rows if
                    q in str(r.get("location", "")).lower() or
                    q in str(r.get("title", "")).lower() or
                    q in str(r.get("details", "")).lower() or
                    q in str(r.get("price", "")).lower()]

        total = len(rows)
        start = (page - 1) * per_page
        return {"total": total, "page": page, "per_page": per_page,
                "data": rows[start: start + per_page]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/grocery", tags=["Data Explorer"])
def get_grocery_data(search: str = "", page: int = 1, per_page: int = 50):
    """Browse raw Blinkit grocery price records (197 items)."""
    try:
        items = _load_json(_DATA_DIR / "grocery" / "Bhopal_blinkit_results_462010.json")
        rows = []
        for item in items:
            if item.get("price") and "₹" in str(item.get("price", "")):
                rows.append({
                    "name":     item.get("name", "—"),
                    "quantity": item.get("quantity", "—"),
                    "price":    item.get("price", "—"),
                    "category": item.get("term", "—"),
                    "delivery": item.get("delivery_time", "—"),
                })

        q = search.strip().lower()
        if q:
            rows = [r for r in rows if
                    q in str(r.get("name", "")).lower() or
                    q in str(r.get("category", "")).lower() or
                    q in str(r.get("quantity", "")).lower()]

        total = len(rows)
        start = (page - 1) * per_page
        return {"total": total, "page": page, "per_page": per_page,
                "data": rows[start: start + per_page]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/metro", tags=["Data Explorer"])
def get_metro_data(search: str = "", page: int = 1, per_page: int = 50):
    """Browse all 42 Bhopal metro route records."""
    try:
        items = _load_json(_DATA_DIR / "Metro" / "bhopal_metro_fares.json")
        rows = []
        for item in items:
            rows.append({
                "from":        item.get("From", "—"),
                "to":          item.get("To", "—"),
                "distance":    item.get("Distance", "—"),
                "time":        item.get("Est. Time", "—"),
                "stops":       item.get("Stations", "—"),
                "fare":        item.get("Est. Fare", "—"),
                "interchange": item.get("Interchanges", "None"),
            })

        q = search.strip().lower()
        if q:
            rows = [r for r in rows if
                    q in str(r.get("from", "")).lower() or
                    q in str(r.get("to", "")).lower() or
                    q in str(r.get("fare", "")).lower()]

        total = len(rows)
        start = (page - 1) * per_page
        return {"total": total, "page": page, "per_page": per_page,
                "data": rows[start: start + per_page]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/metro/stations", tags=["Data Explorer"])
def get_metro_stations():
    """Return list of all Bhopal metro station names."""
    try:
        items = _load_json(_DATA_DIR / "Metro" / "bhopal_metro_fares.json")
        stations = sorted(set([r["From"] for r in items] + [r["To"] for r in items]))
        return {"stations": stations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/metro/route", tags=["Data Explorer"])
def get_metro_route(from_station: str, to_station: str):
    """
    Look up fare and route details between two stations.
    Example: /data/metro/route?from_station=AIIMS&to_station=Subhash+Nagar
    """
    try:
        items = _load_json(_DATA_DIR / "Metro" / "bhopal_metro_fares.json")
        from_s = from_station.strip()
        to_s   = to_station.strip()
        for item in items:
            if item.get("From", "").lower() == from_s.lower() and \
               item.get("To", "").lower() == to_s.lower():
                return {
                    "from":        item["From"],
                    "to":          item["To"],
                    "distance":    item.get("Distance", "—"),
                    "time":        item.get("Est. Time", "—"),
                    "stops":       item.get("Stations", "—"),
                    "fare":        item.get("Est. Fare", "—"),
                    "interchange": item.get("Interchanges", "None"),
                    "route_steps": item.get("Route Steps", []),
                }
        # Try reverse
        for item in items:
            if item.get("From", "").lower() == to_s.lower() and \
               item.get("To", "").lower() == from_s.lower():
                return {
                    "from":        item["To"],   # swapped
                    "to":          item["From"],
                    "distance":    item.get("Distance", "—"),
                    "time":        item.get("Est. Time", "—"),
                    "stops":       item.get("Stations", "—"),
                    "fare":        item.get("Est. Fare", "—"),
                    "interchange": item.get("Interchanges", "None"),
                    "route_steps": item.get("Route Steps", []),
                    "note":        "Reverse direction — same fare applies.",
                }
        raise HTTPException(status_code=404,
            detail=f"No route found between '{from_s}' and '{to_s}'. Same station = ₹0.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/cpi", tags=["Data Explorer"])
def get_cpi_data(search: str = ""):
    """Browse raw CPI (Consumer Price Index) records for Bhopal."""
    try:
        import pandas as pd
        df = pd.read_excel(_DATA_DIR / "cpi" / "bhopal_cpi_clean.xlsx")
        df.columns = [c.strip() for c in df.columns]
        rows = df.to_dict(orient="records")

        q = search.strip().lower()
        if q:
            rows = [r for r in rows if
                    q in str(r.get("Group","")).lower() or
                    q in str(r.get("Year","")).lower() or
                    q in str(r.get("Month","")).lower()]

        return {"total": len(rows), "data": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

