# File: src/train_model_sma.py
# This script trains an XGBoost classifier for the SMA strategy,
# tunes hyperparameters and probability threshold using the validation set,
# and saves the final model.

import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import json

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 1. Load feature files and original data
# -------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_sma_features.csv'), index_col=0, parse_dates=True)
val_df = pd.read_csv(os.path.join(DATA_DIR, 'validation_sma_features.csv'), index_col=0, parse_dates=True)

# Load original ETH data to get open price
original_df = pd.read_csv(os.path.join(DATA_DIR, 'ETHUSDT_1h.csv'), index_col=0, parse_dates=True)
original_df = original_df[['open', 'close']]

# Merge open price into train and validation sets
train_df = train_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')
val_df = val_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')

# Drop rows with missing open (should be none)
train_df = train_df.dropna(subset=['open'])
val_df = val_df.dropna(subset=['open'])

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")

# -------------------------------
# 2. Separate features and target
# -------------------------------
feature_cols = [
    'SMA_10', 'SMA_20', 'SMA_50',
    'price_to_SMA_10', 'price_to_SMA_20', 'price_to_SMA_50',
    'SMA_10_over_20', 'SMA_20_over_50',
    'SMA_10_slope_3', 'SMA_20_slope_3', 'SMA_50_slope_3',
    'zscore_price_to_SMA_20'
]

X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']

# -------------------------------
# 3. Hyperparameter tuning (small grid)
# -------------------------------
param_grid = [
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 400, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 400, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 3, 'learning_rate': 0.1,  'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8},
]

best_auc = 0
best_params = None
best_model = None

for params in param_grid:
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        **params
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred_proba)
    print(f"Params: {params} -> Validation AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_params = params
        best_model = xgb_model

print(f"\nBest validation AUC: {best_auc:.4f}")
print(f"Best hyperparameters: {best_params}")

# -------------------------------
# 4. Retrain final model on train+validation
# -------------------------------
X_combined = pd.concat([X_train, X_val], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    **best_params
)
final_model.fit(X_combined, y_combined, verbose=False)
print("Final model trained on train+validation.")

# -------------------------------
# 5. Threshold tuning on validation set
# -------------------------------
def backtest_returns(probabilities, threshold, close_prices, fee=0.001):
    """
    Backtest with signal shifted by one bar.
    Inputs:
        probabilities: pandas Series of predicted probabilities (index aligned with close_prices)
        threshold: float
        close_prices: pandas Series of close prices (same index)
        fee: transaction fee per side
    Returns: pandas Series of net period returns
    """
    signal = (probabilities > threshold).astype(int)
    position = signal.shift(1).fillna(0).astype(int)
    price_ret = close_prices.pct_change().fillna(0)
    strat_ret = position * price_ret
    trade = position.diff().abs()
    costs = trade * fee
    net_ret = strat_ret - costs
    return net_ret

# Get validation probabilities from the final model
val_proba = final_model.predict_proba(X_val)[:, 1]
# Create a pandas Series with same index as val_df
val_proba_series = pd.Series(val_proba, index=val_df.index)
val_close = val_df['close']

thresholds = np.arange(0.30, 0.71, 0.01)
best_sharpe = -np.inf
best_threshold = 0.5

for th in thresholds:
    net_ret = backtest_returns(val_proba_series, th, val_close, fee=0.001)
    if net_ret.std() > 0:
        sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(24*365)
    else:
        sharpe = 0
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_threshold = th

print(f"Best threshold on validation: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")

# -------------------------------
# 6. Save final model and metadata
# -------------------------------
final_model.save_model(os.path.join(MODEL_DIR, 'xgboost_sma.json'))

metadata = {
    'threshold': float(best_threshold),
    'best_params': best_params,
    'validation_auc': best_auc,
    'feature_cols': feature_cols
}
with open(os.path.join(MODEL_DIR, 'sma_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model and metadata saved to {MODEL_DIR}/")
print("Step 3 complete.")