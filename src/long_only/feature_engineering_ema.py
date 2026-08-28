# File: src/feature_engineering_ema.py
# This script loads the cleaned ETH/USDT data, computes EMA-based features,
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
# 2. EMA-based feature engineering
# -------------------------------
# All features use only past/current data

# Raw EMA values
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

# Price-to-EMA ratios
df['price_to_EMA_12'] = df['close'] / df['EMA_12'] - 1
df['price_to_EMA_26'] = df['close'] / df['EMA_26'] - 1

# EMA ratio
df['EMA_12_over_26'] = df['EMA_12'] / df['EMA_26'] - 1

# EMA slopes over 3 bars
df['EMA_12_slope_3'] = df['EMA_12'].diff(3) / df['EMA_12'].shift(3)
df['EMA_26_slope_3'] = df['EMA_26'].diff(3) / df['EMA_26'].shift(3)

# Rolling z-score of price relative to EMA_20? We don't have EMA20, but we can use EMA_12
rolling_std_12 = df['close'].rolling(window=12).std()
df['zscore_price_to_EMA_12'] = (df['close'] - df['EMA_12']) / rolling_std_12

feature_cols = [
    'EMA_12', 'EMA_26',
    'price_to_EMA_12', 'price_to_EMA_26',
    'EMA_12_over_26',
    'EMA_12_slope_3', 'EMA_26_slope_3',
    'zscore_price_to_EMA_12'
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
train_df.to_csv(os.path.join(DATA_DIR, 'train_ema_features.csv'))
val_df.to_csv(os.path.join(DATA_DIR, 'validation_ema_features.csv'))
test_df.to_csv(os.path.join(DATA_DIR, 'test_ema_features.csv'))

print(f"EMA feature files saved to {DATA_DIR}/")