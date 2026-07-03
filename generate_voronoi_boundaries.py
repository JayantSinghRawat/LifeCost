import json
import math
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, mapping
import sys
from pathlib import Path

# Improvised script to create Voronoi polygons
sys.path.append(str(Path.cwd()))
from ml.locality_recommender import LOCALITY_COORDS, WORKPLACE_HUBS

names = []
coords = []

for k, v in LOCALITY_COORDS.items():
    if k not in names:
        names.append(k)
        coords.append([v[0], v[1]]) # lat, lng

for k, v in WORKPLACE_HUBS.items():
    if k not in names:
        names.append(k)
        coords.append([v[0], v[1]])

# Bhopal bounding box approx
min_lat = 23.15
max_lat = 23.35
min_lng = 77.30
max_lng = 77.58

bhopal_box = box(min_lat, min_lng, max_lat, max_lng)

dummy_pts = [
    [min_lat - 0.5, min_lng - 0.5],
    [min_lat - 0.5, max_lng + 0.5],
    [max_lat + 0.5, min_lng - 0.5],
    [max_lat + 0.5, max_lng + 0.5],
    [23.25, 76.5], [23.25, 78.5], [22.5, 77.45], [24.0, 77.45]
]
coords_with_dummy = coords + dummy_pts
vor = Voronoi(coords_with_dummy)

voronoi_dict = {}
for idx, name in enumerate(names):
    region_index = vor.point_region[idx]
    region = vor.regions[region_index]
    
    if -1 in region or len(region) == 0:
        continue
    
    polygon_vertices = [vor.vertices[i] for i in region]
    try:
        poly = Polygon(polygon_vertices)
        clipped = poly.intersection(bhopal_box)
        
        if not clipped.is_empty and clipped.geom_type == 'Polygon':
             geom = mapping(clipped)
             # Leaflet geoJSON expects [lng, lat]
             lng_lat_coords = [[ [pt[1], pt[0]] for pt in ring ] for ring in geom['coordinates']]
             voronoi_dict[name] = {"type": "Polygon", "coordinates": lng_lat_coords}
    except Exception as e:
        print(f"Error for {name}: {e}")

out_file = Path("Scrapping_Manual/voronoi_boundaries.json")
with open(out_file, "w") as f:
    json.dump(voronoi_dict, f)
print(f"Generated {len(voronoi_dict)} Voronoi boundaries.")
