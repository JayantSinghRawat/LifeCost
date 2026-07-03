import json
import urllib.request
import urllib.parse
import time
import sys
from pathlib import Path

# Add project root to sys path to import ml
sys.path.append(str(Path(__file__).parent))
from ml.locality_recommender import LOCALITY_COORDS, WORKPLACE_HUBS

boundaries = {}
localities = list(LOCALITY_COORDS.keys()) + list(WORKPLACE_HUBS.keys())
localities = list(set(localities))

print(f"Fetching boundaries for {len(localities)} localities...")

ALIAS_MAP = {
    "AYODHYA NAGAR": "Ayodhya Bypass",
    "AWADHPURI": "Awadhpuri",
    "NEW MARKET": "T. T. Nagar",
    "NEW MARKET TT NAGAR": "T. T. Nagar",
    "MISROD": "Misrod",
    "MANDIDEEP": "Mandideep",
    "SHAHPURA": "Shahpura",
    "GOVINDPURA": "Govindpura",
    "AIIMS BHOPAL": "AIIMS",
    "BAGH MUNGALIYA": "Baghmughalia",
    "BAWARIYA KALAN": "Bawadia Kalan",
    "CHUNABHATTI": "Chuna Bhatti",
    "INDRAPURI": "Indrapuri",
    "KAROD KALAN": "Karond",
    "DANISH SQUARE": "Danish Nagar",
    "ASHOKA GARDEN": "Ashoka Garden",
    "AKRITI ECOCITY": "Aakriti Ecocity",
    "AAKRITI ECOCITY": "Aakriti Ecocity",
    "NEW MINAL RESIDENCY": "Minal Residency",
    "INDRAPURI C SECTOR": "Indrapuri",
    "IT PARK": "IT Park"
}

for loc in localities:
    search_name = ALIAS_MAP.get(loc, loc)
    query = f"{search_name}, Bhopal"
    url = "https://nominatim.openstreetmap.org/search?q=" + urllib.parse.quote(query) + "&format=json&polygon_geojson=1&limit=5"
    req = urllib.request.Request(url, headers={"User-Agent": "MP-Life-Bot/1.0"})
    
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode())
            
            found_poly = False
            for item in data:
                if "geojson" in item:
                    geojson = item["geojson"]
                    if geojson["type"] in ["Polygon", "MultiPolygon"]:
                        boundaries[loc] = geojson
                        print(f"✅ Found polygon for {loc} (type: {item.get('type', 'unknown')})")
                        found_poly = True
                        break
            
            if not found_poly:
                if data:
                    print(f"❌ Found point only for {loc}")
                else:
                    print(f"❌ No data for {loc}")
    except Exception as e:
        print(f"⚠️ Error {loc}: {e}")
    time.sleep(1.2)

out_file = Path("Scrapping_Manual/osm_boundaries.json")
with open(out_file, "w") as f:
    json.dump(boundaries, f)
print(f"Saved to {out_file}")
