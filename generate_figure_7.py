import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

os.makedirs('artifacts', exist_ok=True)

with open('Scrapping_Manual/renting/listings.json', encoding='utf-8') as f:
    raw = json.load(f)

# Extract localities for Bhopal only
localities = []
for item in raw:
    location = str(item.get("location") or "").upper()
    if 'BHOPAL' in location:
        # Get the part before Bhopal if it exists
        parts = [p.strip() for p in location.split(',')]
        if len(parts) > 1:
            locality = parts[-2]
            localities.append(locality)
        elif location.strip() != 'BHOPAL':
            localities.append(location.replace(' BHOPAL', '').strip())

df = pd.DataFrame(localities, columns=['Locality'])
top_localities = df['Locality'].value_counts().head(15)

# Plotting
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x=top_localities.values, y=top_localities.index, palette="viridis")

plt.title('Distribution of Rental Data Across Bhopal Localities', fontsize=16, pad=20)
plt.xlabel('Number of Listings', fontsize=12)
plt.ylabel('Locality', fontsize=12)

# Add counts on the bars
for i, v in enumerate(top_localities.values):
    ax.text(v + 3, i, str(v), color='black', va='center')

plt.tight_layout()
output_path = '/Users/jayant/Desktop/MP-Life/artifacts/Figure_7_Distribution_of_Rental_Data.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
