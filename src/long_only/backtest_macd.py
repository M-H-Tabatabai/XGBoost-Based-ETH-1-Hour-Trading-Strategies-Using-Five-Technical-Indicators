# File: src/backtest_macd.py
# This script loads the trained MACD XGBoost model and evaluates it on the test set.
# It also computes the rule-based MACD baseline and saves the results.

import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
RESULT_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------
# 1. Load test features and original data
# -------------------------------
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_macd_features.csv'), index_col=0, parse_dates=True)
original_df = pd.read_csv(os.path.join(DATA_DIR, 'ETHUSDT_1h.csv'), index_col=0, parse_dates=True)
original_df = original_df[['open', 'close']]

# Merge open price into test set
test_df = test_df.merge(original_df[['open']], left_index=True, right_index=True, how='left')
test_df = test_df.dropna(subset=['open'])
print(f"Test set shape: {test_df.shape}")

# -------------------------------
# 2. Load trained model and metadata
# -------------------------------
model = xgb.XGBClassifier()
model.load_model(os.path.join(MODEL_DIR, 'xgboost_macd.json'))

with open(os.path.join(MODEL_DIR, 'macd_metadata.json'), 'r') as f:
    metadata = json.load(f)

threshold = metadata['threshold']
feature_cols = metadata['feature_cols']

print(f"Threshold: {threshold:.2f}")

# -------------------------------
# 3. Generate predictions on test set
# -------------------------------
X_test = test_df[feature_cols]
test_proba = model.predict_proba(X_test)[:, 1]
test_proba_series = pd.Series(test_proba, index=test_df.index)

# -------------------------------
# 4. Accurate backtest function (execution at next open)
# -------------------------------
def backtest_accurate(signal_series, open_prices, close_prices, fee=0.001):
    """
    Backtest with execution at next open.
    signal_series: 1 for long, 0 for flat.
    Returns: equity curve, net returns per bar
    """
    position = signal_series.shift(1).fillna(0).astype(int)
    next_open = open_prices.shift(-1)
    hold_ret = (next_open / open_prices) - 1
    hold_ret = hold_ret.fillna(0)
    strat_ret = position * hold_ret
    trade = position.diff().abs()
    costs = trade * fee
    net_ret = strat_ret - costs
    equity = (1 + net_ret).cumprod()
    return equity, net_ret

# -------------------------------
# 5. XGBoost MACD Strategy on test set
# -------------------------------
signal_xgb = (test_proba_series > threshold).astype(int)
equity_xgb, net_ret_xgb = backtest_accurate(signal_xgb, test_df['open'], test_df['close'], fee=0.001)

total_return_xgb = (equity_xgb.iloc[-1] - 1) * 100
max_dd_xgb = ((equity_xgb / equity_xgb.cummax()) - 1).min() * 100
sharpe_xgb = (net_ret_xgb.mean() / net_ret_xgb.std()) * np.sqrt(24*365) if net_ret_xgb.std() > 0 else 0

print("\n--- XGBoost MACD Strategy (Test) ---")
print(f"Total Return: {total_return_xgb:.2f}%")
print(f"Max Drawdown: {max_dd_xgb:.2f}%")
print(f"Sharpe Ratio: {sharpe_xgb:.2f}")

# -------------------------------
# 6. Rule-based MACD Baseline (MACD line > signal line)
# -------------------------------
test_df['MACD_line'] = test_df['close'].ewm(span=12, adjust=False).mean() - test_df['close'].ewm(span=26, adjust=False).mean()
test_df['MACD_signal'] = test_df['MACD_line'].ewm(span=9, adjust=False).mean()
test_df['signal_rule'] = (test_df['MACD_line'] > test_df['MACD_signal']).astype(int)

equity_rule, net_ret_rule = backtest_accurate(test_df['signal_rule'], test_df['open'], test_df['close'], fee=0.001)

total_return_rule = (equity_rule.iloc[-1] - 1) * 100
max_dd_rule = ((equity_rule / equity_rule.cummax()) - 1).min() * 100
sharpe_rule = (net_ret_rule.mean() / net_ret_rule.std()) * np.sqrt(24*365) if net_ret_rule.std() > 0 else 0

print("\n--- Rule-Based MACD (Test) ---")
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
# 8. Save results for later comparison
# -------------------------------
results = {
    'Strategy': ['XGBoost MACD', 'Rule-based MACD', 'Buy & Hold'],
    'Total Return (%)': [total_return_xgb, total_return_rule, buy_hold_return],
    'Max Drawdown (%)': [max_dd_xgb, max_dd_rule, max_dd_bh],
    'Sharpe Ratio': [sharpe_xgb, sharpe_rule, sharpe_bh]
}
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULT_DIR, 'macd_results.csv'), index=False)
print(f"\nResults saved to {os.path.join(RESULT_DIR, 'macd_results.csv')}")
print("Step 13 complete.")