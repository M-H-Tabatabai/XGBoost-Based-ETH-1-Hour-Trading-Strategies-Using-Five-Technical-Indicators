# File: src/final_evaluation.py
# This script loads all individual strategy result CSVs,
# combines them into a single comparison table,
# sorts by Sharpe Ratio, and saves the final comparison.

import pandas as pd
import os

# -------------------------------
# 0. Robust path setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------
# 1. Load all result files
# -------------------------------
result_files = [
    'sma_results.csv',
    'rsi_results.csv',
    'ema_results.csv',
    'macd_results.csv',
    'volume_results.csv'
]

all_results = []
for file in result_files:
    file_path = os.path.join(RESULT_DIR, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        all_results.append(df)
        print(f"Loaded {file}")
    else:
        print(f"Warning: {file} not found. Skipping.")

if not all_results:
    raise FileNotFoundError("No result files found. Run individual backtest scripts first.")

# Combine all results
combined = pd.concat(all_results, ignore_index=True)

# Remove duplicate Buy & Hold entries (keep only one)
combined = combined.drop_duplicates(subset=['Strategy'], keep='first')

# -------------------------------
# 2. Sort by Sharpe Ratio descending
# -------------------------------
combined_sorted = combined.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)

# Add rank column
combined_sorted.insert(0, 'Rank', range(1, len(combined_sorted) + 1))

# -------------------------------
# 3. Save final comparison table
# -------------------------------
final_path = os.path.join(RESULT_DIR, 'final_comparison.csv')
combined_sorted.to_csv(final_path, index=False)

# -------------------------------
# 4. Display final table
# -------------------------------
print("\n" + "="*80)
print("FINAL COMPARISON TABLE (Test Set Results)")
print("="*80)
print(combined_sorted.to_string(index=False))
print("\nSaved to:", final_path)

# -------------------------------
# 5. Conclusion
# -------------------------------
best_strategy = combined_sorted.iloc[0]
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Best strategy: {best_strategy['Strategy']}")
print(f"Total Return: {best_strategy['Total Return (%)']:.2f}%")
print(f"Max Drawdown: {best_strategy['Max Drawdown (%)']:.2f}%")
print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")

# Compare with Buy & Hold
buy_hold = combined_sorted[combined_sorted['Strategy'] == 'Buy & Hold']
if not buy_hold.empty:
    bh = buy_hold.iloc[0]
    print(f"\nBuy & Hold Return: {bh['Total Return (%)']:.2f}%")
    print(f"Buy & Hold Sharpe: {bh['Sharpe Ratio']:.2f}")