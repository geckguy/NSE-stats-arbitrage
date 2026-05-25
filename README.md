# NSE Pairs Trading Engine

A market-neutral statistical arbitrage strategy for the Indian equity market (NSE), built with cointegration-based pair selection and validated through rolling walk-forward testing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

The strategy identifies cointegrated stock pairs within the same NSE sector and exploits short-term deviations in their spread. Positions are market-neutral (long one stock, short the other), so returns are driven purely by relative price movements rather than broad market direction.

Key design choices:
- Dual cointegration testing (ADF + Engle–Granger) with rolling stability checks
- Rolling hedge ratio (OLS β) recomputed on a 120-day window to avoid stale estimates
- Velocity filter on entries to confirm the spread is reverting before opening a position
- 3% per-pair drawdown stop and hard z-score stop-loss to control tail risk
- Walk-forward validation across 13 out-of-sample periods (Jan 2023 – Apr 2026)

---

## Project Structure

```
.
├── src/                        # Core strategy library
│   ├── data_loader.py          # Downloads & caches NSE price data via yfinance
│   ├── pair_selector.py        # Cointegration tests, stability scoring, pair ranking
│   ├── signal_generator.py     # Rolling β, spread, z-score & signal logic
│   ├── backtester.py           # P&L computation and trade log extraction
│   └── metrics.py              # Sharpe, CAGR, drawdown, Calmar, win rate
│
├── research/                   # Experiment & parameter tuning scripts
│   ├── tune_strategy.py        # Grid search over entry/exit/stop parameters
│   ├── tune_walk_forward.py    # Walk-forward parameter sweep (1,280 combos)
│   └── ...                     # Earlier walk-forward iterations (v2–v5)
│
├── notebooks/                  # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_pair_selection.ipynb
│   ├── 03_signal_generation.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_long_only_constraint.ipynb   # NSE overnight short restriction analysis
│
├── docs/
│   └── learn.md                # Technical reference (statistical methods)
│
├── backtest.py                 # Runs the full walk-forward backtest
├── dashboard.py                # Streamlit dashboard for interactive analysis
├── config.py                   # All strategy parameters
└── requirements.txt
```

---

## Quickstart

**1. Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generate backtest results**

Downloads ~5 years of NSE price data, runs the walk-forward backtest, and saves outputs to `data/`.

```bash
python backtest.py
```

**3. Launch the dashboard**

```bash
streamlit run dashboard.py
```

---

## Strategy Parameters

All parameters are defined in [`config.py`](config.py). Values were selected via grid search over the rolling walk-forward sweep (`research/tune_walk_forward.py`).

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BETA_LOOKBACK_WINDOW` | 120 days | Rolling OLS window for hedge ratio |
| `Z_LOOKBACK_WINDOW` | 30 days | Rolling window for z-score normalisation |
| `Z_ENTRY_THRESHOLD` | 2.2 | Enter when spread exceeds ±2.2σ |
| `Z_EXIT_THRESHOLD` | 0.5 | Exit when spread returns within ±0.5σ |
| `Z_STOP_LOSS` | 3.5 | Hard stop — cointegration likely broken |
| `TRAIN_WINDOW_DAYS` | 504 | Training window (~2 years) per period |
| `REBALANCE_DAYS` | 126 | Out-of-sample window (~6 months) |
| `MAX_PAIRS` | 5 | Maximum concurrent pairs per period |
| `MIN_STABILITY_PCT` | 60% | Minimum rolling cointegration stability |
| `PAIR_DRAWDOWN_STOP` | 3% | Per-pair drawdown circuit breaker |

---

## Walk-Forward Results (1× Leverage)

Tested across 7 out-of-sample periods from January 2023 to April 2026, benchmarked against the Nifty 50.

| Metric | Portfolio | Nifty 50 |
|--------|-----------|----------|
| Total Return | +42.9% | +35.1% |
| CAGR | +11.7% | +9.8% |
| Sharpe Ratio | 1.54 | 0.30 |
| Max Drawdown | -1.3% | -15.8% |
| Win Rate | 52.6% (114 trades) | — |
| Profitable Periods | 7/7 | — |

**2025–2026 out-of-sample period**

| Metric | Portfolio | Nifty 50 |
|--------|-----------|----------|
| Total Return | +21.55% | -0.62% |
| CAGR | +21.55% | -0.62% |
| Sharpe Ratio | 1.91 | -0.43 |
| Max Drawdown | -3.37% | -15.18% |
| Volatility | 7.23% | 13.26% |
| Calmar Ratio | 6.40 | -0.04 |

---

## Dashboard

```bash
streamlit run dashboard.py
```

| Tab | Description |
|-----|-------------|
| Portfolio Performance | Equity curves, drawdown profile, per-period table |
| Pair Signal Explorer | Reconstruct spread, z-score & signals for any traded pair |
| Trade Logs | Filter and analyse all executed trades |
| Parameter Sweep | Visualise the 1,280-combination grid search results |
| Reference | Technical notes on the statistical methods used |

---

## Methodology Reference

See [`docs/learn.md`](docs/learn.md) for a detailed walkthrough of the statistical methods: cointegration vs. correlation, stationarity, ADF testing, OLS hedge ratios, z-score signals, half-life estimation, walk-forward validation, and transaction cost modelling.

---

## Practical Considerations

NSE does not allow retail investors to carry short equity positions overnight — intraday only. This means the market-neutral backtest results above assume access to the **F&O (Futures & Options) segment**, where short positions can be held via stock futures.

- **140 of 154 tickers** in the universe have NSE F&O contracts, so the strategy is largely executable using futures for the short leg
- The 14 tickers without F&O are mostly smaller names (CUB, IOB, STARCEMENT, PGHH, etc.) that rarely appear in the final pair selection
- A live implementation would short the future and long the cash equity, introducing basis risk and rollover costs not modelled here

For a version of the strategy that works within the overnight short constraint (long leg only, no futures required), see [`notebooks/05_long_only_constraint.ipynb`](notebooks/05_long_only_constraint.ipynb). This sacrifices market neutrality but shows what is achievable with cash equity alone.

---

## Disclaimer

This project is for research purposes only. It is not financial advice. Past backtest performance does not guarantee future returns.

---

<sub>Built with assistance from [Antigravity](https://antigravity.dev) (AI coding assistant).</sub>
