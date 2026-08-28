# File: src/long_short/train_model_rsi_short.py
# This script trains an XGBoost classifier for RSI with long/short signals,
# tunes hyperparameters and two thresholds (upper for long, lower for short)
# using the validation set, and saves the model to models/long_short/.

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
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', '..', 'models', 'long_short')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 1. Load feature files and original data
# -------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_rsi_features.csv'), index_col=0, parse_dates=True)
val_df = pd.read_csv(os.path.join(DATA_DIR, 'validation_rsi_features.csv'), index_col=0, parse_dates=True)

# Load original ETH data to get open price
original_df = pd.read_csv(os.path.join(DATA_DIR, 'ETHUSDT_1h.csv'), index_col=0, parse_dates=True)
original_df = original_df[['open', 'close']]

# Merge open price into train and validation sets
train_df = train_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')
val_df = val_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')

# Drop rows with missing open
train_df = train_df.dropna(subset=['open'])
val_df = val_df.dropna(subset=['open'])

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")

# -------------------------------
# 2. Separate features and target
# -------------------------------
feature_cols = [
    'RSI_14',
    'RSI_14_lag_1', 'RSI_14_lag_2', 'RSI_14_lag_3',
    'RSI_14_momentum_3',
    'RSI_14_rolling_mean_10', 'RSI_14_rolling_std_10', 'RSI_14_rolling_mean_50',
    'RSI_14_dist_30', 'RSI_14_dist_50', 'RSI_14_dist_70'
]

X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']

# -------------------------------
# 3. Hyperparameter tuning (same grid as before)
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
# 5. Long/short threshold tuning on validation set
# -------------------------------
def backtest_long_short(signal_series, open_prices, fee=0.001):
    """
    signal_series: pandas Series with values 1 (long), -1 (short), 0 (flat)
    open_prices: pandas Series of open prices (same index)
    Returns: equity curve, net returns per bar
    """
    position = signal_series.shift(1).fillna(0).astype(int)
    next_open = open_prices.shift(-1)
    long_ret = (next_open / open_prices) - 1
    short_ret = (open_prices / next_open) - 1
    hold_ret = np.where(position == 1, long_ret, np.where(position == -1, short_ret, 0))
    hold_ret = pd.Series(hold_ret, index=open_prices.index).fillna(0)
    
    trade = position.diff().abs()
    costs = trade * fee
    net_ret = hold_ret - costs
    equity = (1 + net_ret).cumprod()
    return equity, net_ret

val_proba = final_model.predict_proba(X_val)[:, 1]
val_proba_series = pd.Series(val_proba, index=val_df.index)
val_open = val_df['open']

upper_grid = np.arange(0.50, 0.81, 0.02)
lower_grid = np.arange(0.20, 0.51, 0.02)

best_sharpe = -np.inf
best_upper = 0.60
best_lower = 0.40

for upper in upper_grid:
    for lower in lower_grid:
        if lower >= upper:
            continue
        signal = np.where(val_proba > upper, 1, np.where(val_proba < lower, -1, 0))
        signal_series = pd.Series(signal, index=val_df.index)
        _, net_ret = backtest_long_short(signal_series, val_open, fee=0.001)
        if net_ret.std() > 0:
            sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(24*365)
        else:
            sharpe = 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_upper = upper
            best_lower = lower

print(f"\nBest thresholds on validation:")
print(f"  Upper (long) : {best_upper:.2f}")
print(f"  Lower (short): {best_lower:.2f}")
print(f"  Sharpe       : {best_sharpe:.2f}")

# -------------------------------
# 6. Save final model and metadata
# -------------------------------
final_model.save_model(os.path.join(MODEL_DIR, 'xgboost_rsi_short.json'))

metadata = {
    'upper_threshold': float(best_upper),
    'lower_threshold': float(best_lower),
    'best_params': best_params,
    'validation_auc': best_auc,
    'feature_cols': feature_cols
}
with open(os.path.join(MODEL_DIR, 'rsi_short_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel and metadata saved to {MODEL_DIR}/")
print("RSI short training complete.")