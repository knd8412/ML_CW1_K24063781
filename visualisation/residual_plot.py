import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('CW1_train.csv')

def preprocess(df):
    temp = df.copy()
    temp['cut'] = temp['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    temp['color'] = temp['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    temp['clarity'] = temp['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    temp['vol'] = temp['x'] * temp['y'] * temp['z']
    return temp

trn = preprocess(train_df)
X, y = trn.drop(columns=['outcome']), trn['outcome']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized Hyperparameters
model = XGBRegressor(n_estimators=1500, learning_rate=0.015, max_depth=5, 
                     subsample=0.7, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=3.0)
model.fit(X_train, y_train)
preds = model.predict(X_val)
residuals = y_val - preds

plt.figure(figsize=(10, 6))
plt.scatter(preds, residuals, alpha=0.4, color='red', s=15)
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel('Predicted Outcome')
plt.ylabel('Residual (Error)')
plt.title('Residual Plot: Error Distribution (XGBoost)', fontweight='bold')

plt.tight_layout()
plt.savefig('visualisation/residual_plot.png', dpi=300)
print("Saved: visualisation/residual_plot.png")