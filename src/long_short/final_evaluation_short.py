# File: src/long_short/final_evaluation_short.py
# This script loads all long-only and long-short result CSVs,
# combines them into a single comparison table,
# sorts by Sharpe Ratio, and saves the final comparison.

import pandas as pd
import os

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Long-only results are saved directly in the main results folder
RESULTS_LONG_ONLY = os.path.join(BASE_DIR, '..', '..', 'results')
# Long-short results are in the long_short subfolder
RESULTS_LONG_SHORT = os.path.join(RESULTS_LONG_ONLY, 'long_short')
OUTPUT_DIR = RESULTS_LONG_SHORT
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Load all long-only result files
# -------------------------------
long_only_files = [
    'sma_results.csv',
    'rsi_results.csv',
    'ema_results.csv',
    'macd_results.csv',
    'volume_results.csv'
]

all_results = []

for file in long_only_files:
    file_path = os.path.join(RESULTS_LONG_ONLY, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Type'] = 'Long-Only'
        all_results.append(df)
        print(f"Loaded long-only: {file}")
    else:
        print(f"Warning: {file} not found in results/")

# -------------------------------
# 2. Load all long-short result files
# -------------------------------
long_short_files = [
    'sma_results_short.csv',
    'rsi_results_short.csv',
    'ema_results_short.csv',
    'macd_results_short.csv',
    'volume_results_short.csv'
]

for file in long_short_files:
    file_path = os.path.join(RESULTS_LONG_SHORT, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Type'] = 'Long-Short'
        all_results.append(df)
        print(f"Loaded long-short: {file}")
    else:
        print(f"Warning: {file} not found in long_short/")

# -------------------------------
# 3. Combine all results
# -------------------------------
if not all_results:
    raise FileNotFoundError("No result files found. Run backtest scripts first.")

combined = pd.concat(all_results, ignore_index=True)

# Remove duplicate Buy & Hold entries (keep only one)
combined = combined.drop_duplicates(subset=['Strategy'], keep='first')

# Reorder columns: put Type after Strategy
cols = combined.columns.tolist()
if 'Type' in cols:
    cols.remove('Type')
    strategy_idx = cols.index('Strategy')
    cols.insert(strategy_idx + 1, 'Type')
    combined = combined[cols]

# -------------------------------
# 4. Sort by Sharpe Ratio descending
# -------------------------------
combined_sorted = combined.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)
combined_sorted.insert(0, 'Rank', range(1, len(combined_sorted) + 1))

# -------------------------------
# 5. Save final comparison table
# -------------------------------
final_path = os.path.join(OUTPUT_DIR, 'final_comparison_short.csv')
combined_sorted.to_csv(final_path, index=False)

# -------------------------------
# 6. Display final table
# -------------------------------
print("\n" + "="*80)
print("FINAL COMPARISON TABLE (Long-Only + Long-Short + Buy & Hold)")
print("="*80)
print(combined_sorted.to_string(index=False))
print("\nSaved to:", final_path)

# -------------------------------
# 7. Conclusion
# -------------------------------
best_strategy = combined_sorted.iloc[0]
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Best strategy: {best_strategy['Strategy']}")
print(f"Type: {best_strategy['Type']}")
print(f"Total Return: {best_strategy['Total Return (%)']:.2f}%")
print(f"Max Drawdown: {best_strategy['Max Drawdown (%)']:.2f}%")
print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")

buy_hold = combined_sorted[combined_sorted['Strategy'] == 'Buy & Hold']
if not buy_hold.empty:
    bh = buy_hold.iloc[0]
    print(f"\nBuy & Hold Return: {bh['Total Return (%)']:.2f}%")
    print(f"Buy & Hold Sharpe: {bh['Sharpe Ratio']:.2f}")