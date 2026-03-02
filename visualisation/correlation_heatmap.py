import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Path to CSV
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'CW1_train.csv')

train_df = pd.read_csv(csv_path)

# 2. Preprocessing
def get_processed(df):
    temp = df.copy()
    temp['cut']     = temp['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    temp['color']   = temp['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    temp['clarity'] = temp['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    temp['vol']     = temp['x'] * temp['y'] * temp['z']
    return temp

df = get_processed(train_df)

# 3. All features in a logical order: outcome, physical, categorical, price, then a/b columns
cols = [
    'outcome',
    'depth', 'table', 'cut', 'color', 'clarity', 'price',
    'carat', 'x', 'y', 'z', 'vol',
    'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
    'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10'
]

corr_matrix = df[cols].corr()

# 4. Lower triangle mask
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            fmt='.2f',
            center=0,
            vmin=-0.5,
            vmax=0.5,
            linewidths=0.3,
            square=True,
            ax=ax,
            annot_kws={'size': 5.5},
            cbar_kws={'shrink': 0.6})

ax.set_title('Correlation Heatmap: All Features vs. Outcome', fontsize=14, fontweight='bold', pad=10)
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.tick_params(axis='y', labelsize=7, rotation=0)

plt.tight_layout()

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correlation_heatmap.png')
plt.savefig('visualisation/correlation_heatmap.png', dpi=300)
print("Saved: visualisation/correlation_heatmap.png")