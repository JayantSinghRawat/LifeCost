import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('artifacts', exist_ok=True)

# Data from Table 12 in the report
data = {
    'Concurrent Users': [10, 50, 100, 200],
    'Avg Response Time (ms)': [87, 124, 198, 312]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a bar plot
ax = sns.barplot(
    x="Concurrent Users", 
    y="Avg Response Time (ms)", 
    data=df, 
    palette="Blues_d",
    hue="Concurrent Users",
    legend=False
)

# Add a line plot overlay to show the trend
sns.lineplot(
    x=range(len(df)), 
    y="Avg Response Time (ms)", 
    data=df, 
    marker='o', 
    color='#e74c3c', 
    linewidth=2.5, 
    markersize=10,
    ax=ax
)

# Annotate bars
for i, v in enumerate(df["Avg Response Time (ms)"]):
    ax.text(i, v + 8, f"{v} ms", color='black', ha='center', fontweight='bold', fontsize=11)

plt.title('Figure 13. API Response Time Distribution Chart', fontsize=16, pad=20)
plt.xlabel('Concurrent Users (Simulated Load)', fontsize=12, labelpad=10)
plt.ylabel('Average Response Time (ms)', fontsize=12, labelpad=10)

# Add a threshold line for 200ms objective mentioned in the report
plt.axhline(y=200, color='#f39c12', linestyle='--', linewidth=2, label='SLA Target (<200ms)')
plt.legend(loc='upper left', fontsize=11)

plt.ylim(0, 360) # Give semantic headroom for text
plt.tight_layout()

output_path = '/Users/jayant/Desktop/MP-Life/artifacts/Figure_13_API_Response_Time.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
