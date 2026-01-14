# 📈 Machine Learning Trading Strategy (End-to-End Pipeline)

## Overview 
This project implements an end-to-end machine learning pipeline for historical price–based market direction prediction.
The goal is to predict next-day price direction (up/down) using technical indicators and evaluate performance using realistic backtesting methods.

The project emphasizes good quant research practices, including:

- Feature engineering from market data
- Avoiding data leakage
- Walk-forward (time-series) validation
- Strategy-level evaluation via equity curves

## Key Features

📊 Historical price data processing
🧠 Feature engineering using technical indicators
🤖 Machine learning models (Logistic Regression, Random Forest)
🔁 Walk-forward validation (time-series aware)
💰 Strategy backtesting & equity curve analysis

### Features Used

- Daily returns
- Rolling volatility
- Relative Strength Index (RSI)
- Moving averages
- Momentum-based indicators

These features are constructed using only past information to prevent lookahead bias.

## Labels 

- Binary classification target:

  - 1 → next-day return > 0
  - 0 → next-day return ≤ 0
 
## Models Implemented

- Logistic Regression (baseline, interpretable)
- Random Forest (non-linear, higher capacity)

Models are trained using walk-forward validation to simulate real trading conditions.

## Validation Methodology

- Walk-forward / expanding window
- -No random shuffling
- Strict chronological order

This ensures realistic out-of-sample performance.

## Backtesting Logic

- Go long when model predicts upward movement
- Stay flat otherwise
- Equity curve calculated via cumulative strategy returns


Performance metrics include:
- Cumulative return
- Equity curve visualization
- (Optional) Sharpe ratio and drawdowns

## Results & Observations

- Demonstrates how ML signals translate into tradable strategies
- Highlights the difficulty of achieving persistent alpha
- Emphasizes robustness over overfitting

**This project is intended for educational and research purposes, not live trading.**
