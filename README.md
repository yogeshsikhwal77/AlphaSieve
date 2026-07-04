# AlphaSieve 🔍📈

**An ML-filtered algorithmic trading strategy backtester.**  
Combines rule-based momentum/mean-reversion signals with a classical ML classifier that filters out low-quality trades — then rigorously backtests the result with realistic transaction costs, slippage, and walk-forward validation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

🔗 **Live Demo:** [Coming soon](#) *(add your Streamlit link here once deployed)*

---

## 📖 Overview

Most trading strategy projects stop at "backtest looks great, therefore it works." AlphaSieve is built to avoid that trap — it uses machine learning not to predict prices, but to **filter which rule-based signals are actually worth taking**, and evaluates everything the way a real quant/risk desk would: Sharpe ratio, max drawdown, and cost-adjusted returns, benchmarked against simple buy-and-hold.

## ✨ Features

- 📊 Rule-based signal generation (momentum / mean-reversion)
- 🤖 ML classifier (Logistic Regression / Random Forest / XGBoost) to filter false signals
- ⏳ Walk-forward validation (no lookahead bias, no random shuffling of time-series data)
- 💰 Realistic backtesting — transaction costs, slippage, position sizing
- 📉 Risk metrics: Sharpe Ratio, Max Drawdown, CAGR, Win Rate
- 🔍 Feature importance / interpretability via SHAP
- 📈 Visualizations: equity curve, drawdown chart, trade markers on price
- 🌍 Regime-awareness (bull / bear / sideways market detection)

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `jugaad-data`, `ccxt` |
| ML | `scikit-learn`, `xgboost`, `shap` |
| Backtesting | `vectorbt` / `backtrader` |
| Visualization | `matplotlib`, `plotly` |
| Deployment | `streamlit` |

## 📂 Project Structure

```
alphasieve/
├── data/                  # raw & processed price data
├── notebooks/             # EDA and experimentation
├── src/
│   ├── data_loader.py     # fetches OHLCV data
│   ├── features.py        # technical indicator engineering
│   ├── signals.py         # rule-based strategy logic
│   ├── model.py           # ML classifier training/inference
│   ├── backtest.py        # backtesting engine + cost modeling
│   └── metrics.py         # Sharpe, drawdown, CAGR, etc.
├── app.py                 # Streamlit dashboard (live demo)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/alphasieve.git
cd alphasieve

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/backtest.py

# Or launch the interactive dashboard
streamlit run app.py
```

## 📊 Results (Sample)

> *(Replace this section with your actual output once you run it)*

| Metric | Strategy | Buy & Hold |
|---|---|---|
| CAGR | — | — |
| Sharpe Ratio | — | — |
| Max Drawdown | — | — |
| Win Rate | — | — |

![Equity Curve](assets/equity_curve.png)

## ⚠️ Limitations & Disclaimer

- This project has **survivorship bias** — data reflects currently listed stocks/index constituents, not delisted/bankrupt ones.
- Backtested performance is **not a guarantee of future results** and typically overstates real-world performance.
- This is a **research/educational project**, not financial advice. Do not use it to make real trading decisions without further validation.

## 🗺️ Roadmap

- [ ] Add intraday data support
- [ ] Add crypto market support (24/7 data)
- [ ] Multi-asset portfolio backtesting
- [ ] Regime-switching model (HMM/GMM)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙋 Contact

Built by **[Your Name]** — feel free to reach out or open an issue for suggestions.
