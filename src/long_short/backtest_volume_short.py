# File: src/long_short/backtest_volume_short.py
# This script loads the trained Volume short-enabled model, generates long/short signals,
# backtests on the test set, and compares with rule-based long/short and Buy & Hold.

import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', '..', 'models', 'long_short')
RESULT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'long_short')
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------
# 1. Load test features and original data
# -------------------------------
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_volume_features.csv'), index_col=0, parse_dates=True)
original_df = pd.read_csv(os.path.join(DATA_DIR, 'ETHUSDT_1h.csv'), index_col=0, parse_dates=True)
original_df = original_df[['open', 'close']]

test_df = test_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')
test_df = test_df.dropna(subset=['open'])
print(f"Test set shape: {test_df.shape}")

# -------------------------------
# 2. Load trained model and metadata
# -------------------------------
model = xgb.XGBClassifier()
model.load_model(os.path.join(MODEL_DIR, 'xgboost_volume_short.json'))

with open(os.path.join(MODEL_DIR, 'volume_short_metadata.json'), 'r') as f:
    metadata = json.load(f)

upper_th = metadata['upper_threshold']
lower_th = metadata['lower_threshold']
feature_cols = metadata['feature_cols']

print(f"Upper threshold (long): {upper_th:.2f}")
print(f"Lower threshold (short): {lower_th:.2f}")

# -------------------------------
# 3. Generate predictions on test set
# -------------------------------
X_test = test_df[feature_cols]
test_proba = model.predict_proba(X_test)[:, 1]
test_proba_series = pd.Series(test_proba, index=test_df.index)

# -------------------------------
# 4. Long/short backtest function
# -------------------------------
def backtest_long_short(signal_series, open_prices, fee=0.001):
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

# -------------------------------
# 5. XGBoost long/short strategy on test set
# -------------------------------
signal_xgb = np.where(test_proba > upper_th, 1, np.where(test_proba < lower_th, -1, 0))
signal_xgb_series = pd.Series(signal_xgb, index=test_df.index)
equity_xgb, net_ret_xgb = backtest_long_short(signal_xgb_series, test_df['open'], fee=0.001)

total_return_xgb = (equity_xgb.iloc[-1] - 1) * 100
max_dd_xgb = ((equity_xgb / equity_xgb.cummax()) - 1).min() * 100
sharpe_xgb = (net_ret_xgb.mean() / net_ret_xgb.std()) * np.sqrt(24*365) if net_ret_xgb.std() > 0 else 0

print("\n--- XGBoost Volume Long/Short (Test) ---")
print(f"Total Return: {total_return_xgb:.2f}%")
print(f"Max Drawdown: {max_dd_xgb:.2f}%")
print(f"Sharpe Ratio: {sharpe_xgb:.2f}")

# -------------------------------
# 6. Rule-based Volume long/short baseline
# -------------------------------
# Rule: long if volume > 1.5*avg AND close > open ; short if volume > 1.5*avg AND close < open ; else flat
condition_long = (test_df['volume'] > 1.5 * test_df['volume_SMA_20']) & (test_df['close'] > test_df['open'])
condition_short = (test_df['volume'] > 1.5 * test_df['volume_SMA_20']) & (test_df['close'] < test_df['open'])
test_df['signal_rule'] = 0
test_df.loc[condition_long, 'signal_rule'] = 1
test_df.loc[condition_short, 'signal_rule'] = -1

equity_rule, net_ret_rule = backtest_long_short(test_df['signal_rule'], test_df['open'], fee=0.001)

total_return_rule = (equity_rule.iloc[-1] - 1) * 100
max_dd_rule = ((equity_rule / equity_rule.cummax()) - 1).min() * 100
sharpe_rule = (net_ret_rule.mean() / net_ret_rule.std()) * np.sqrt(24*365) if net_ret_rule.std() > 0 else 0

print("\n--- Rule-Based Volume Long/Short (Test) ---")
print(f"Total Return: {total_return_rule:.2f}%")
print(f"Max Drawdown: {max_dd_rule:.2f}%")
print(f"Sharpe Ratio: {sharpe_rule:.2f}")

# -------------------------------
# 7. Buy & Hold ETH
# -------------------------------
buy_hold_return = (test_df['close'].iloc[-1] / test_df['open'].iloc[0] - 1) * 100 - 0.1
close_series = test_df['close']
buy_hold_equity = close_series / close_series.iloc[0]
max_dd_bh = ((buy_hold_equity / buy_hold_equity.cummax()) - 1).min() * 100
bh_ret = close_series.pct_change().fillna(0)
sharpe_bh = (bh_ret.mean() / bh_ret.std()) * np.sqrt(24*365) if bh_ret.std() > 0 else 0

print("\n--- Buy & Hold ETH (Test) ---")
print(f"Total Return: {buy_hold_return:.2f}%")
print(f"Max Drawdown: {max_dd_bh:.2f}%")
print(f"Sharpe Ratio: {sharpe_bh:.2f}")

# -------------------------------
# 8. Save results
# -------------------------------
results = {
    'Strategy': ['XGBoost Volume Long/Short', 'Rule-based Volume Long/Short', 'Buy & Hold'],
    'Total Return (%)': [total_return_xgb, total_return_rule, buy_hold_return],
    'Max Drawdown (%)': [max_dd_xgb, max_dd_rule, max_dd_bh],
    'Sharpe Ratio': [sharpe_xgb, sharpe_rule, sharpe_bh]
}
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULT_DIR, 'volume_results_short.csv'), index=False)
print(f"\nResults saved to {os.path.join(RESULT_DIR, 'volume_results_short.csv')}")
print("Volume short backtest complete.")