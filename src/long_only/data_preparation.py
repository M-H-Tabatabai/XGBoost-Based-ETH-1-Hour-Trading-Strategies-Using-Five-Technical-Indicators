# File: src/data_preparation.py
# This script fetches/loads ETH/USDT 1-hour data, cleans it, and splits chronologically into train/validation/test sets.

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime   # <-- FIX: import the datetime class

# -------------------------------
# 1. Fetch or load data
# -------------------------------

def fetch_binance_klines(symbol='ETHUSDT', interval='1h', start_date='2020-01-01', end_date=None):
    """Fetch historical 1h klines from Binance public API."""
    base_url = 'https://api.binance.com/api/v3/klines'
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    if end_date:
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)
    
    all_klines = []
    limit = 1000
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': limit
        }
        resp = requests.get(base_url, params=params)
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        # Move to next batch
        current_start = data[-1][0] + 1
        # Respect rate limits
        time.sleep(0.2)
    
    if not all_klines:
        raise ValueError("No data fetched.")
    
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only needed columns
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

# Attempt to load from local file if exists, otherwise fetch
local_file = 'ETHUSDT_1h.csv'
if os.path.exists(local_file):
    print(f"Loading data from {local_file}")
    df = pd.read_csv(local_file, index_col=0, parse_dates=True)
else:
    print("Fetching data from Binance...")
    # Adjust start_date as needed – we need at least ~5000 rows for useful training
    df = fetch_binance_klines(start_date='2020-01-01')
    df.to_csv(local_file)
    print(f"Data saved to {local_file}")

# -------------------------------
# 2. Data cleaning
# -------------------------------

print(f"Raw data shape: {df.shape}")
# Remove duplicates
df = df[~df.index.duplicated(keep='first')]
# Sort by timestamp
df.sort_index(inplace=True)
# Drop rows with missing or invalid prices/volume
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
df = df[(df['volume'] > 0) & (df['close'] > 0)]
print(f"Cleaned data shape: {df.shape}")

# -------------------------------
# 3. Chronological split (70/15/15)
# -------------------------------

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Train: {train_df.shape[0]} rows ({train_df.index[0]} to {train_df.index[-1]})")
print(f"Validation: {val_df.shape[0]} rows ({val_df.index[0]} to {val_df.index[-1]})")
print(f"Test: {test_df.shape[0]} rows ({test_df.index[0]} to {test_df.index[-1]})")

# Save splits for later steps
train_df.to_csv('train_ETHUSDT_1h.csv')
val_df.to_csv('validation_ETHUSDT_1h.csv')
test_df.to_csv('test_ETHUSDT_1h.csv')
print("Split files saved.")