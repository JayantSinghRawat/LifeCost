import urllib.request
query = """
[out:json][timeout:25];
area["name"="Bhopal"]->.searchArea;
nwr["name"~"Awadhpuri",i](area.searchArea);
out geom;
"""
url = "https://overpass-api.de/api/interpreter"
req = urllib.request.Request(url, data=query.encode("utf-8"))
import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
with urllib.request.urlopen(req, context=ctx) as response:
    data = response.read().decode()
    print(data[:500])
