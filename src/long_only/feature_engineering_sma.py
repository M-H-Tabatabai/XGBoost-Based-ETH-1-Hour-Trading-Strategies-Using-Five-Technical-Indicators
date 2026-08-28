# File: src/feature_engineering_sma.py
# This script loads the cleaned ETH/USDT data, computes SMA‑based features,
# creates binary labels per split (no leakage), and saves the feature sets.

import pandas as pd
import numpy as np
import os

# -------------------------------
# 1. Locate and load the cleaned data
# -------------------------------
# Possible locations for ETHUSDT_1h.csv
candidate_paths = [
    '../data/ETHUSDT_1h.csv',   # recommended: in data/ folder one level up
    'ETHUSDT_1h.csv',           # fallback: same directory as script
    '../ETHUSDT_1h.csv'         # fallback: project root
]

data_path = None
for path in candidate_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    raise FileNotFoundError("Could not find ETHUSDT_1h.csv. Please place it in the data/ folder or the src/ folder.")

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"Loaded data shape: {df.shape}")

# -------------------------------
# 2. SMA‑based feature engineering
# -------------------------------
# All features use only past/current data (backward‑looking rolling windows)

windows = [10, 20, 50]

# Raw SMA values
for w in windows:
    df[f'SMA_{w}'] = df['close'].rolling(window=w).mean()

# Price-to-SMA ratios
for w in windows:
    df[f'price_to_SMA_{w}'] = df['close'] / df[f'SMA_{w}'] - 1

# SMA cross ratios
df['SMA_10_over_20'] = df['SMA_10'] / df['SMA_20'] - 1
df['SMA_20_over_50'] = df['SMA_20'] / df['SMA_50'] - 1

# SMA slopes over 3 bars
for w in windows:
    df[f'SMA_{w}_slope_3'] = df[f'SMA_{w}'].diff(3) / df[f'SMA_{w}'].shift(3)

# Rolling z‑score of price relative to SMA_20
rolling_std_20 = df['close'].rolling(window=20).std()
df['zscore_price_to_SMA_20'] = (df['close'] - df['SMA_20']) / rolling_std_20

feature_cols = [
    'SMA_10', 'SMA_20', 'SMA_50',
    'price_to_SMA_10', 'price_to_SMA_20', 'price_to_SMA_50',
    'SMA_10_over_20', 'SMA_20_over_50',
    'SMA_10_slope_3', 'SMA_20_slope_3', 'SMA_50_slope_3',
    'zscore_price_to_SMA_20'
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
# Create output directory if it doesn't exist
output_dir = '../data'
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, 'train_sma_features.csv'))
val_df.to_csv(os.path.join(output_dir, 'validation_sma_features.csv'))
test_df.to_csv(os.path.join(output_dir, 'test_sma_features.csv'))

print(f"SMA feature files saved to {output_dir}/")