import urllib.request
import urllib.parse
import json

def overpass_query(query_str):
    url = "https://overpass-api.de/api/interpreter"
    data = query_str.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Life-Cost-Bot/1.0", "Content-Type": "application/x-www-form-urlencoded"})
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error: {e}")
        return None

query = """
[out:json][timeout:25];
area[name="Bhopal"]->.searchArea;
(
  nwr["name"~"Ashoka Garden",i](area.searchArea);
);
out geom;
"""
res = overpass_query(query)
if res and "elements" in res:
    for el in res["elements"]:
        print(f"ID: {el['id']}, Type: {el['type']}, Tags: {el.get('tags', {})}")
        if el["type"] in ["way", "relation"] and "geometry" in el:
            print("  Has geometry!")
