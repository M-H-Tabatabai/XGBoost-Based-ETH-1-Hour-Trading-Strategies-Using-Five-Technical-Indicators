# File: src/feature_engineering_rsi.py
# This script loads the cleaned ETH/USDT data, computes RSI-based features,
# creates binary labels per split (no leakage), and saves the feature sets.

import pandas as pd
import numpy as np
import os

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# 1. Load cleaned full data
# -------------------------------
data_path = os.path.join(DATA_DIR, 'ETHUSDT_1h.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"Loaded data shape: {df.shape}")

# -------------------------------
# 2. RSI-based feature engineering
# -------------------------------
# All features use only past/current data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Raw RSI_14
df['RSI_14'] = compute_rsi(df['close'], period=14)

# Lagged RSI values
for lag in [1, 2, 3]:
    df[f'RSI_14_lag_{lag}'] = df['RSI_14'].shift(lag)

# RSI momentum (difference over 3 bars)
df['RSI_14_momentum_3'] = df['RSI_14'] - df['RSI_14'].shift(3)

# Rolling mean and std of RSI
df['RSI_14_rolling_mean_10'] = df['RSI_14'].rolling(window=10).mean()
df['RSI_14_rolling_std_10'] = df['RSI_14'].rolling(window=10).std()
df['RSI_14_rolling_mean_50'] = df['RSI_14'].rolling(window=50).mean()

# Distance from key levels
df['RSI_14_dist_30'] = df['RSI_14'] - 30
df['RSI_14_dist_50'] = df['RSI_14'] - 50
df['RSI_14_dist_70'] = df['RSI_14'] - 70

# Keep only relevant columns: features + close (for label construction)
feature_cols = [
    'RSI_14',
    'RSI_14_lag_1', 'RSI_14_lag_2', 'RSI_14_lag_3',
    'RSI_14_momentum_3',
    'RSI_14_rolling_mean_10', 'RSI_14_rolling_std_10', 'RSI_14_rolling_mean_50',
    'RSI_14_dist_30', 'RSI_14_dist_50', 'RSI_14_dist_70'
]
df_features = df[['close'] + feature_cols].copy()

# -------------------------------
# 3. Chronological split (same as before)
# -------------------------------
n = len(df_features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df_features.iloc[:train_end].copy()
val_df = df_features.iloc[train_end:val_end].copy()
test_df = df_features.iloc[val_end:].copy()

# -------------------------------
# 4. Label construction per split (no leakage)
# -------------------------------
def add_labels(split_df):
    split_df['target'] = (split_df['close'].shift(-1) > split_df['close']).astype(int)
    split_df = split_df.iloc[:-1]   # drop last row with NaN target
    return split_df

train_df = add_labels(train_df)
val_df = add_labels(val_df)
test_df = add_labels(test_df)

# Drop rows with NaN in features (due to rolling windows)
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

print(f"Train features shape: {train_df.shape}")
print(f"Validation features shape: {val_df.shape}")
print(f"Test features shape: {test_df.shape}")

# -------------------------------
# 5. Save feature sets
# -------------------------------
train_df.to_csv(os.path.join(DATA_DIR, 'train_rsi_features.csv'))
val_df.to_csv(os.path.join(DATA_DIR, 'validation_rsi_features.csv'))
test_df.to_csv(os.path.join(DATA_DIR, 'test_rsi_features.csv'))

print(f"RSI feature files saved to {DATA_DIR}/")