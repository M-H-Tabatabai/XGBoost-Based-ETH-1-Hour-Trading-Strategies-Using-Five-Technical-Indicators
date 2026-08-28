# File: src/feature_engineering_volume.py
# This script loads the cleaned ETH/USDT data, computes Volume-based features,
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
# 2. Volume-based feature engineering
# -------------------------------
# All features use only past/current data

# Raw volume (already exists)
# Volume moving averages
for w in [10, 20, 50]:
    df[f'volume_SMA_{w}'] = df['volume'].rolling(window=w).mean()

# Volume ratio
df['volume_ratio_20'] = df['volume'] / df['volume_SMA_20'] - 1
df['volume_ratio_50'] = df['volume'] / df['volume_SMA_50'] - 1

# Rolling z-score of volume over 50 bars
rolling_mean_vol_50 = df['volume'].rolling(window=50).mean()
rolling_std_vol_50 = df['volume'].rolling(window=50).std()
df['volume_zscore_50'] = (df['volume'] - rolling_mean_vol_50) / rolling_std_vol_50

# Volume slope (3-bar)
df['volume_slope_3'] = df['volume'].diff(3) / df['volume'].shift(3)

# On-Balance Volume (OBV) slope as a volume-derived feature
obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
df['OBV_slope_3'] = obv.diff(3)

feature_cols = [
    'volume',
    'volume_SMA_10', 'volume_SMA_20', 'volume_SMA_50',
    'volume_ratio_20', 'volume_ratio_50',
    'volume_zscore_50',
    'volume_slope_3',
    'OBV_slope_3'
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
train_df.to_csv(os.path.join(DATA_DIR, 'train_volume_features.csv'))
val_df.to_csv(os.path.join(DATA_DIR, 'validation_volume_features.csv'))
test_df.to_csv(os.path.join(DATA_DIR, 'test_volume_features.csv'))

print(f"Volume feature files saved to {DATA_DIR}/")