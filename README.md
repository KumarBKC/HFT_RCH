# Strategic HFT Project

A modular Python implementation of algorithmic trading strategies, converted from a research Jupyter notebook.

## 🚀 Overview

This project implements **two advanced trading strategies** derived from quantitative research. The codebase has been refactored from a monolithic notebook into a structured, production-ready Python application.

### Working Strategies

#### 1. Twitter Sentiment Strategy (`sentiment`)
**Goal**: Capitalize on social media momentum.
- **Data Source**: Local `sentiment_data.csv` (Twitter metrics) + Yahoo Finance (Prices).
- **Logic**: 
  - Calculates an **Engagement Ratio** (Comments + Likes / Impressions) for each stock.
  - Ranks stocks monthly.
  - Selects top 5 stocks with the highest engagement.
  - Rebalances monthly to maintain an equally weighted portfolio.
- **Robustness**: Includes fallback to simulated price data if live downloads fail (e.g., for delisted stocks).

#### 2. Intraday GARCH Strategy (`garch`)
**Goal**: Profitable intraday trading using volatility prediction.
- **Data Source**: Local `simulated_daily_data.csv` and `simulated_5min_data.csv`.
- **Logic**:
  - fits a **GARCH(1,1)** model on daily returns (rolling window).
  - Predicts 1-day ahead volatility.
  - Generates a **daily signal** based on the "Prediction Premium" (Predicted vs Realized Volatility).
  - Combines with intraday **RSI** and **Larry Williams' %R (LBOS)** for precise entry/exit timing.
  - Holds positions until the end of the day.

---

## � Project Structure

The project is organized for modularity and scalability:

```text
HFT/
├── config/                 # Configuration (Paths, Parameters)
│   └── settings.py
├── strategies/             # Strategy Logic
│   ├── base.py             # Abstract base class
│   ├── sentiment.py        # Twitter Strategy implementation
│   └── intraday_garch.py   # GARCH Strategy implementation
├── utils/                  # Shared Components
│   ├── data_loader.py      # CSV loading & cleaning
│   ├── features.py         # Technical Indicators (RSI, MACD, Volatility)
│   └── visualization.py    # Matplotlib plotting
├── output/                 # Results (Backtest charts)
├── main.py                 # CLI Entry Point
└── requirements.txt        # Dependencies
```

---

## ⚡ Quick Start

### 1. Setup Environment
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Strategies
You can run strategies directly from the command line using `main.py`.

**Run All Strategies:**
```bash
python main.py --all
```

**Run Twitter Sentiment Strategy:**
```bash
python main.py --strategy sentiment
```

**Run Intraday GARCH Strategy:**
```bash
python main.py --strategy garch
```

**List Options:**
```bash
python main.py --help
```

---

## 📊 Outputs

The application generates performance reports and visualizations in the `output/` directory:
- `sentiment_strategy_returns.png`: Cumulative returns vs Benchmark (QQQ).
- `garch_strategy_returns.png`: Cumulative returns of the GARCH model vs Buy & Hold.

Console output includes key metrics:
- **Sharpe Ratio**
- **Annualized Return**
- **Max Drawdown**
- **Win Rate**

---

## 🔧 Configuration

You can adjust strategy parameters in `config/settings.py`.

**Examples:**
- Change `TOP_N_STOCKS` for the Sentiment strategy.
- Adjust `ROLLING_WINDOW` or `RSI_PERIOD` for the GARCH strategy.
- Modify date ranges for backtesting.

---

## 📝 Notes on Data
- **Sentiment Data**: `sentiment_data.csv` (provided locally).
- **HFT Data**: `simulated_5min_data.csv` (provided locally).
- **Unsupervised Strategy**: The original notebook contained a third strategy (K-Means Clustering). This was **excluded** from the CLI implementation to avoid heavy dependency on downloading large external S&P 500 datasets, but the logic remains available in `notebooks/Algorithmic_Trading_Machine_Learning_Quant_Strategies.ipynb`.
