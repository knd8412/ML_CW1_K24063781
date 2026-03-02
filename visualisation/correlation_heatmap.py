import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Correct the path to the root CSV
# This goes up one folder from 'visualisation' to the root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'CW1_train.csv')

train_df = pd.read_csv(csv_path)

# 2. Preprocessing
def get_processed(df):
    temp = df.copy()
    # Adding volume as it is a key physical signal
    temp['vol'] = temp['x'] * temp['y'] * temp['z']
    return temp

df = get_processed(train_df)

# 3. Selection: Include target, signals, and noise columns
cols = ['outcome', 'vol', 'carat', 'x', 'y', 'z', 'depth', 'table', 'a1', 'a2', 'b1', 'b2']
corr_matrix = df[cols].corr()

# 4. Professional Styling (Lower Triangle Mask)
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Matches the style of your red heatmap

sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            center=0,
            linewidths=.5)

plt.title('Correlation Heatmap: Physical Signals vs. Synthetic Noise', fontsize=15, fontweight='bold')
plt.tight_layout()

# Save the output
plt.savefig('visualisation/correlation_heatmap.png', dpi=300)
print("Saved: visualisation/correlation_heatmap.png")