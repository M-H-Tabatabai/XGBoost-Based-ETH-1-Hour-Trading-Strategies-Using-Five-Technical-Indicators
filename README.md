# XGBoost-Based ETH 1-Hour Trading Strategies Using Five Technical Indicators

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

This project implements and compares **five independent XGBoost-based trading strategies** for **Ethereum (ETH)** using **1-hour data**. Each strategy relies on a single technical indicator as its primary input:

- **SMA** – Simple Moving Average
- **RSI** – Relative Strength Index
- **EMA** – Exponential Moving Average
- **MACD** – Moving Average Convergence Divergence
- **Volume** – Trading volume and derived features

The main goal is to determine which indicator-based strategy performs best on **unseen test data**, and to compare the XGBoost models against **rule-based baselines** and **Buy & Hold**.

---

## 📁 Project Structure

```text
eth_trading_project/
│
├── data/                              # All raw and split data files
│   ├── ETHUSDT_1h.csv                 # Raw 1h OHLCV data
│   ├── train_*.csv                    # Training feature sets (70%)
│   ├── validation_*.csv               # Validation feature sets (15%)
│   └── test_*.csv                     # Test feature sets (15%)
│
├── src/
│   ├── long_only/                     # Long-only strategies
│   │   ├── data_preparation.py
│   │   ├── feature_engineering_*.py
│   │   ├── train_model_*.py
│   │   └── backtest_*.py
│   │
│   └── long_short/                    # Long-short extensions
│       ├── train_model_*_short.py
│       ├── backtest_*_short.py
│       └── final_evaluation_short.py
│
├── models/
│   ├── long_only/                     # Saved XGBoost models
│   │   └── xgboost_*.json
│   └── long_short/                    # Saved XGBoost models
│       └── xgboost_*_short.json
│
└── results/
    ├── long_only/                     # Results for long-only strategies
    │   └── *_results.csv
    └── long_short/                    # Results for long-short strategies
        └── *_results_short.csv
```

---

## 📊 Data Description

- **Asset:** Ethereum (ETH/USDT)
- **Timeframe:** 1 hour
- **Period:** ~2019-12-31 to 2026-08-28
- **Dataset size:** Approximately 58,000 candles
- **Source:** Binance public API (or local CSV fallback)
- **Fields used:** `open`, `high`, `low`, `close`, `volume`

### Chronological Split

To avoid data leakage in time-series data, the dataset is split **chronologically**, not randomly:

| Split | Percentage | Rows | Purpose |
|---|---:|---:|---|
| **Training** | 70% | ~40,827 | Model training |
| **Validation** | 15% | ~8,749 | Hyperparameter tuning and threshold selection |
| **Test** | 15% | ~8,749 | Final unseen evaluation |

> **Important:** The test set is never used during training, validation, or threshold selection.

---

## 🧠 Methodology

### 1. Feature Engineering

Each strategy uses only its corresponding indicator as the primary input. Features are backward-looking and do not use future information.

| Indicator | Feature Examples |
|---|---|
| **SMA** | SMA (10, 20, 50), price-to-SMA ratios, SMA cross ratios, slopes, z-score |
| **RSI** | RSI(14), lags (1, 2, 3), momentum, rolling mean/std, distance from 30/50/70 |
| **EMA** | EMA(12, 26), price-to-EMA ratios, EMA ratio, slopes, z-score |
| **MACD** | MACD line, signal line, histogram, lags, slopes, histogram z-score |
| **Volume** | Volume, volume SMAs, volume ratios, z-score, slope, OBV slope |

### 2. Target Label

The task is a binary classification problem:

```text
1 → next hourly close > current close
0 → otherwise
```

### 3. Model Training

- **Algorithm:** XGBoost Classifier
- **Hyperparameter tuning:** Small grid search with 6 configurations
- **Selection metric:** Validation AUC
- **Early stopping:** 20 rounds on the validation set
- **Final model:** Retrained on training + validation data after selecting the best hyperparameters

### 4. Signal Generation & Backtesting

#### Long-Only

```text
P(up) > threshold → Long
P(up) ≤ threshold → Flat
```

#### Long-Short

```text
P(up) > upper_threshold → Long
P(up) < lower_threshold → Short
Otherwise → Flat
```

#### Execution Assumptions

- Signals generated at the close of bar `t`
- Orders executed at the open of bar `t+1`
- Transaction fee: **0.1% per side**
- Round-trip transaction cost: approximately **0.2%**
- Both long-only and long-short positions are simulated

### 5. Evaluation Metrics

- **Total Return (%)** – Final equity relative to initial capital
- **Maximum Drawdown (%)** – Largest peak-to-trough decline
- **Sharpe Ratio** – Annualised using 24 × 365 periods with a 0% risk-free rate

---

## 📈 Results

### Test Set

**Test period:** 2025-08-28 to 2026-08-28

During this period, ETH experienced a **strong downtrend**, with Buy & Hold losing **44.25%**.

### Final Comparison

Results are sorted by Sharpe Ratio.

| Rank | Strategy | Type | Total Return | Max Drawdown | Sharpe Ratio |
|---:|---|---|---:|---:|---:|
| 1 | Rule-based RSI Long/Short | Long-Short | -24.14% | -47.61% | -0.15 |
| 2 | Buy & Hold ETH | — | -44.25% | -67.99% | -0.66 |
| 3 | Rule-based RSI | Long-Only | -34.94% | -53.21% | -0.73 |
| 4 | **XGBoost RSI** | Long-Only | **-11.49%** | **-11.78%** | -1.61 |
| 5 | XGBoost EMA Long/Short | Long-Short | -18.22% | -18.75% | -2.17 |
| 6 | XGBoost RSI Long/Short | Long-Short | -19.30% | -19.30% | -2.45 |
| 7 | Rule-based MACD Long/Short | Long-Short | -81.53% | -82.45% | -2.46 |
| 8 | Rule-based MACD | Long-Only | -68.02% | -71.37% | -2.48 |
| 9 | XGBoost EMA | Long-Only | -22.39% | -23.68% | -2.58 |
| 10 | XGBoost SMA | Long-Only | -32.80% | -34.74% | -2.60 |
| 11 | Rule-based EMA Long/Short | Long-Short | -87.46% | -89.64% | -3.09 |
| 12 | Rule-based EMA | Long-Only | -73.65% | -78.91% | -3.17 |
| 13 | Rule-based SMA | Long-Only | -75.83% | -80.90% | -3.31 |
| 14 | Rule-based SMA Long/Short | Long-Short | -89.45% | -91.46% | -3.38 |
| 15 | Rule-based Volume | Long-Only | -60.76% | -60.81% | -3.69 |
| 16 | Rule-based Volume Long/Short | Long-Short | -80.73% | -81.27% | -4.40 |
| 17 | XGBoost Volume | Long-Only | -57.41% | -58.52% | -5.12 |
| 18 | XGBoost SMA Long/Short | Long-Short | -69.15% | -70.63% | -5.35 |
| 19 | XGBoost MACD | Long-Only | -60.48% | -61.27% | -5.46 |
| 20 | XGBoost MACD Long-Short | Long-Short | -74.67% | -75.12% | -6.57 |
| 21 | XGBoost Volume Long-Short | Long-Short | -89.13% | -89.49% | -8.95 |

---

## 🏆 Key Findings

### 1. Best XGBoost Strategy: RSI Long-Only

The **XGBoost RSI** strategy achieved the best performance among the long-only XGBoost models:

- **Total Return:** -11.49%
- **Maximum Drawdown:** -11.78%
- **Buy & Hold:** -44.25% return
- **Buy & Hold Max Drawdown:** -67.99%

Although its Sharpe Ratio was still negative (-1.61), the strategy preserved capital significantly better than the other strategies.

### 2. Best XGBoost Long-Short Strategy: EMA

The **XGBoost EMA Long-Short** strategy achieved:

- **Total Return:** -18.22%
- **Maximum Drawdown:** -18.75%
- It outperformed its rule-based counterpart by **69.23 percentage points**.

### 3. XGBoost vs Rule-Based Strategies

XGBoost outperformed rule-based baselines in most same-indicator comparisons, particularly for:

- RSI
- EMA
- SMA
- MACD

### 4. Long-Short Strategies Were Not Consistently Better

The long-short versions did not consistently improve performance.

A major reason is that the binary classification model was not specifically designed for short-signal prediction. Short entries were sometimes triggered late or incorrectly.

### 5. The Test Period Was Highly Bearish

The test period was extremely challenging for long-only strategies.

Therefore, the primary objective was not generating positive returns, but rather:

- Minimising losses
- Reducing drawdown
- Preserving capital
- Avoiding poor trades

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- pandas
- numpy
- xgboost
- scikit-learn
- requests
- joblib

### Install Dependencies

```bash
pip install pandas numpy xgboost scikit-learn requests joblib
```

### Step 1 — Data Preparation

```bash
python src/long_only/data_preparation.py
```

This downloads ETH/USDT 1-hour data, or loads it from a local CSV fallback, cleans the data, and performs the chronological split.

### Step 2 — Run Long-Only Strategy

Example: SMA

```bash
python src/long_only/feature_engineering_sma.py
python src/long_only/train_model_sma.py
python src/long_only/backtest_sma.py
```

Repeat the same process for:

- RSI
- EMA
- MACD
- Volume

by changing the filename suffix.

### Step 3 — Run Long-Short Strategy

Example: SMA

```bash
python src/long_short/train_model_sma_short.py
python src/long_short/backtest_sma_short.py
```

Repeat for the other indicators.

### Step 4 — Final Evaluation

```bash
python src/long_short/final_evaluation_short.py
```

This combines all results, sorts them by Sharpe Ratio, and saves the final comparison table to:

```text
results/long_short/final_comparison_short.csv
```

---

## 📁 Output Files

### Models

```text
models/long_only/*.json
models/long_short/*.json
```

### Results

```text
results/long_only/*_results.csv
results/long_short/*_results_short.csv
```

### Final Comparison

```text
results/long_short/final_comparison_short.csv
```

---

## ⚠️ Limitations & Assumptions

- Transaction costs are fixed at **0.1% per side**.
- Slippage and funding rates are not included.
- Short selling is simulated without borrowing fees.
- The prediction horizon is **1 hour**.
- Results may change with longer prediction horizons.
- The test period represents a strong bear market and may not generalise to bullish market regimes.
- Validation AUC was approximately **0.53–0.54**, indicating weak predictive power.
- The strategies primarily improve performance by avoiding poor trades rather than strongly predicting market direction.
- Signals generated at the close are executed at the next open.
- No intra-bar stop-loss or take-profit mechanism is used.

---

## 📝 Conclusion

This project demonstrates a systematic approach to evaluating indicator-based **XGBoost trading strategies** using strict chronological data splits and avoiding data leakage.

The results show that even models with relatively weak predictive power can provide meaningful risk-management benefits in adverse market conditions.

The strongest result came from the **XGBoost RSI long-only strategy**, which reduced the loss to **-11.49%**, compared with **-44.25% for Buy & Hold**, while also reducing maximum drawdown from **-67.99% to -11.78%**.

### Future Improvements

Potential directions for improving the system include:

- Using regression targets based on future return magnitude instead of binary classification.
- Developing separate models specifically for long and short predictions.
- Adding volatility filters.
- Adding market-regime detection.
- Improving short-selling signal generation.
- Testing multiple prediction horizons.
- Incorporating additional market and on-chain features.

---

## 📄 License

This project is released under the **MIT License**.

Feel free to use, modify, and distribute the project.

---

## 🤝 Contributing

Pull requests and suggestions are welcome.

For major changes, please open an issue first to discuss the proposed modification.

---

<div align="center">

**Made with ❤️ and XGBoost**

</div>
