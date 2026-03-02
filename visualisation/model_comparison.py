import matplotlib.pyplot as plt

# Updated with your actual terminal results
models = ['Linear Regression', 'Random Forest', 'XGBoost (Final)']
scores = [0.2662, 0.4538, 0.4614] 
colors = ['#ffdc2d', '#8fb339', '#1f77b4'] # Changed colors for a cleaner look

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors, edgecolor='black')
plt.ylim(0, 0.6)
plt.ylabel('Mean R^2 Score')
plt.title('Model Selection Comparison', fontweight='bold')

# Add the specific labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualisation/model_comparison.png', dpi=300)
print("Saved: visualisation/model_comparison.png")