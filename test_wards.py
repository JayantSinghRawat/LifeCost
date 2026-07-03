import urllib.request, json
query = """
[out:json][timeout:25];
area["name"="Bhopal"]->.searchArea;
(
  relation["admin_level"~"9|10|11"](area.searchArea);
  way["admin_level"~"9|10|11"](area.searchArea);
);
out geom;
"""
url = "https://overpass-api.de/api/interpreter"
req = urllib.request.Request(url, data=query.encode("utf-8"))
import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    with urllib.request.urlopen(req, context=ctx) as response:
        res = json.loads(response.read().decode())
        print("Found wards:", len(res.get("elements", [])))
        for el in res.get("elements", [])[:10]:
            print(el.get("tags", {}).get("name", "Unnamed"))
except Exception as e:
    print(e)
