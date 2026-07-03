import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs('artifacts', exist_ok=True)

ml_path = Path('ml').resolve()
sys.path.insert(0, str(ml_path))

try:
    from locality_recommender import recommend_locality, LOCALITY_COORDS, WORKPLACE_HUBS, train_locality_recommender
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Ensure trained metadata exists
try:
    train_locality_recommender()
except Exception as e:
    print(f"Failed to train locality recommender: {e}")

# Generate recommendations for all localities
# Profile: 50,000 INR/mo, works in MP NAGAR, comfort lifestyle
recs = recommend_locality(
    salary_monthly=50000.0,
    workplace="MP NAGAR",
    lifestyle="comfort",
    commute_tolerance_km=20.0,
    top_n=100
)

# Build dataframe
data = []
for r in recs:
    loc = r["locality"]
    coords = LOCALITY_COORDS.get(loc)
    if coords:
        data.append({
            "Locality": loc,
            "Latitude": coords[0],
            "Longitude": coords[1],
            "Score": r["scores"]["composite"] * 100, # 0-100 scale
        })

df = pd.DataFrame(data)

# Workplace
wp_coords = WORKPLACE_HUBS["MP NAGAR"]

plt.figure(figsize=(12, 10))
sns.set_theme(style="white")

# Scatter map
scatter = sns.scatterplot(
    data=df, 
    x="Longitude", 
    y="Latitude", 
    size="Score", 
    hue="Score", 
    palette="YlOrRd", 
    sizes=(100, 1500), 
    alpha=0.8, 
    edgecolor="black",
    linewidth=1
)

# Plot workplace
plt.plot(wp_coords[1], wp_coords[0], marker='*', color='blue', markersize=20, label='Workplace (MP Nagar)')

# Annotate the top 5
for i, row in df.head(5).iterrows():
    plt.text(row['Longitude'] + 0.005, row['Latitude'] + 0.005, row['Locality'], 
             fontsize=10, weight='bold', color='black', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

plt.title('Geographic Heatmap of Recommended Localities\n(Profile: ₹50k/mo | Work: MP Nagar | Priority: Comfort)', fontsize=16, pad=20)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# Custom legend positioning since scatterplot overlaps
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Heatmap Legend")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

output_path = '/Users/jayant/Desktop/MP-Life/artifacts/Figure_12_Locality_Heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
