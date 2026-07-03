import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the trained Metro Classifier
model = joblib.load('ml/models/metro_model.joblib')
meta = joblib.load('ml/models/metro_meta.joblib')

feature_names = meta.get("feature_cols", ["Distance (km)", "Stops", "Travel Time (min)"])
class_names = [f"₹{c}" for c in model.classes_]

plt.figure(figsize=(10, 6))
plot_tree(
    model, 
    feature_names=feature_names, 
    class_names=class_names, 
    filled=True, 
    rounded=True, 
    precision=1, 
    fontsize=11
)

plt.title('Bhopal Metro Fare Decision Tree', fontsize=16, pad=20)
plt.tight_layout()

output_path = 'metro_decision_tree.png'
plt.savefig(output_path, dpi=300)
print(f"Decision tree saved as {output_path}")
