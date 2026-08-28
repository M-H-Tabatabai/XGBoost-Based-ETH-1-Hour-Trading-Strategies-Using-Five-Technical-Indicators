# File: src/feature_engineering_macd.py
# This script loads the cleaned ETH/USDT data, computes MACD-based features,
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
# 2. MACD-based feature engineering
# -------------------------------
# All features use only past/current data

# MACD line = EMA_12 - EMA_26
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD_line'] = ema_12 - ema_26

# Signal line = 9-period EMA of MACD_line
df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()

# MACD histogram
df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

# Lagged MACD, signal, histogram
for lag in [1, 2, 3]:
    df[f'MACD_line_lag_{lag}'] = df['MACD_line'].shift(lag)
    df[f'MACD_signal_lag_{lag}'] = df['MACD_signal'].shift(lag)
    df[f'MACD_hist_lag_{lag}'] = df['MACD_hist'].shift(lag)

# MACD slope (3-bar)
df['MACD_line_slope_3'] = df['MACD_line'].diff(3) / df['MACD_line'].shift(3)
df['MACD_hist_slope_3'] = df['MACD_hist'].diff(3)

# Rolling z-score of histogram (20-bar)
rolling_mean_hist = df['MACD_hist'].rolling(window=20).mean()
rolling_std_hist = df['MACD_hist'].rolling(window=20).std()
df['MACD_hist_zscore_20'] = (df['MACD_hist'] - rolling_mean_hist) / rolling_std_hist

feature_cols = [
    'MACD_line', 'MACD_signal', 'MACD_hist',
    'MACD_line_lag_1', 'MACD_signal_lag_1', 'MACD_hist_lag_1',
    'MACD_line_lag_2', 'MACD_signal_lag_2', 'MACD_hist_lag_2',
    'MACD_line_lag_3', 'MACD_signal_lag_3', 'MACD_hist_lag_3',
    'MACD_line_slope_3', 'MACD_hist_slope_3',
    'MACD_hist_zscore_20'
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
train_df.to_csv(os.path.join(DATA_DIR, 'train_macd_features.csv'))
val_df.to_csv(os.path.join(DATA_DIR, 'validation_macd_features.csv'))
test_df.to_csv(os.path.join(DATA_DIR, 'test_macd_features.csv'))

print(f"MACD feature files saved to {DATA_DIR}/")