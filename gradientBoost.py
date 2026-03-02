import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# 1. Setup & Data Loading
np.random.seed(42)
train_df = pd.read_csv('CW1_train.csv')
test_df = pd.read_csv('CW1_test.csv')

# 2. Preprocessing
def preprocess(df):
    temp = df.copy()
    temp['cut'] = temp['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    temp['color'] = temp['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    temp['clarity'] = temp['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    temp['vol'] = temp['x'] * temp['y'] * temp['z']
    return temp

trn = preprocess(train_df)
tst = preprocess(test_df)
X, y = trn.drop(columns=['outcome']), trn['outcome']

# 3. Hyperparameters (Optimized for Noise Reduction)
model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=5,
    min_child_weight=10,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_alpha=1.5,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=-1
)

# 4. Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"XGBoost Average R^2: {np.mean(cv_scores):.4f}")

# 5. Final Submission Output
model.fit(X, y)
yhat = model.predict(tst)
pd.DataFrame({'yhat': yhat}).to_csv('CW1_submission_k24063781.csv', index=False)
print("XGBoost submission saved.")