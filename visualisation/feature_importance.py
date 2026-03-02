import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBRegressor
import os

# 1. Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(base_dir, 'CW1_train.csv'))

# 2. Preprocessing
def preprocess(df):
    temp = df.copy()
    temp['cut']     = temp['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    temp['color']   = temp['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    temp['clarity'] = temp['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    temp['vol']     = temp['x'] * temp['y'] * temp['z']
    return temp

trn = preprocess(train_df)
X, y = trn.drop(columns=['outcome']), trn['outcome']

# 3. Train final model
np.random.seed(42)
model = XGBRegressor(
    n_estimators=1500, learning_rate=0.015, max_depth=5,
    min_child_weight=10, gamma=0.2, subsample=0.7,
    colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=3.0,
    random_state=42, n_jobs=-1
)
model.fit(X, y)

# 4. Feature importances sorted ascending
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

# Colour: blue = physical/categorical, red = a/b columns
colors = ['#d73027' if (f.startswith('a') or f.startswith('b')) else '#4575b4'
          for f in importance.index]

# 5. Plot
fig, ax = plt.subplots(figsize=(8, 9))
bars = ax.barh(importance.index, importance.values, color=colors, edgecolor='white', height=0.7)

ax.set_xlabel('Feature Importance (F-score)', fontsize=11)
ax.set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold', pad=10)

for bar, val in zip(bars, importance.values):
    if val > 0.005:
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=7)

legend_elements = [Patch(facecolor='#4575b4', label='Physical / Categorical'),
                   Patch(facecolor='#d73027', label='a/b columns')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_importance.png')
plt.savefig("visualisation/feature_importance.png", dpi=300, bbox_inches='tight')
print(f"Saved: visualisation/feature_importance.png")
