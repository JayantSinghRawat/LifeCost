import urllib.request
import urllib.parse
import json

def search(q):
    url = "https://nominatim.openstreetmap.org/search?q=" + urllib.parse.quote(q) + "&format=json&polygon_geojson=1&limit=10"
    req = urllib.request.Request(url, headers={"User-Agent": "MP-Life-Bot/1.1"})
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode())
            for item in data:
                print(f"{q}: {item.get('display_name')} | Class: {item.get('class')} | Type: {item.get('type')} | Geo: {item.get('geojson', {}).get('type')}")
    except Exception as e:
        print(f"Error {q}: {e}")

search("Kolar Road, Bhopal")
search("Kolar, Bhopal")
search("Awadhpuri, Bhopal")
search("Ashoka Garden, Bhopal")
search("Indrapuri, Bhopal")
search("Ayodhya Bypass, Bhopal")
search("Ayodhya Nagar, Bhopal")
