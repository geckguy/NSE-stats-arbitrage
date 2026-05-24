# Statistical Arbitrage & Pairs Trading — Technical Reference

---

## Table of Contents

1. [What is Pairs Trading?](#1-what-is-pairs-trading)
2. [Correlation vs Cointegration](#2-correlation-vs-cointegration)
3. [Stationarity — The Foundation](#3-stationarity--the-foundation)
4. [The Augmented Dickey-Fuller (ADF) Test](#4-the-augmented-dickey-fuller-adf-test)
5. [OLS Regression & The Hedge Ratio](#5-ols-regression--the-hedge-ratio)
6. [Why Log Prices?](#6-why-log-prices)
7. [The Spread & Z-Score](#7-the-spread--z-score)
8. [Mean Reversion & Half-Life](#8-mean-reversion--half-life)
9. [Rolling Windows & Lookahead Bias](#9-rolling-windows--lookahead-bias)
10. [Walk-Forward Validation](#10-walk-forward-validation)
11. [Transaction Costs & Slippage](#11-transaction-costs--slippage)
12. [Risk Management Basics](#12-risk-management-basics)
13. [Performance Metrics](#13-performance-metrics)
14. [Putting It All Together](#14-putting-it-all-together)
15. [Further Reading](#15-further-reading)

---

## 1. What is Pairs Trading?

### The Intuition

Imagine two coffee shops on the same street. Both charge roughly the same price for a latte. Sometimes Shop A raises prices slightly, sometimes Shop B does — but they always converge back because they're competing for the same customers.

Pairs trading applies this logic to stocks. Two stocks in the same sector (say HDFCBANK and ICICIBANK) are exposed to the same economic forces. Their prices don't move identically, but the *difference* between them tends to stay in a range. When that difference stretches unusually far, you bet it snaps back.

### The Mechanics

- **Long-short position:** You simultaneously buy one stock and sell the other.
- **Market-neutral:** Because you're both long and short, broad market moves cancel out. You only profit (or lose) from the *relative* movement.
- **Mean reversion bet:** You're betting the spread returns to its historical average.

### Example

| Day | HDFC Price | ICICI Price | Spread | Action |
|-----|-----------|-------------|--------|--------|
| 1   | ₹1600     | ₹1000       | 600    | Normal — do nothing |
| 2   | ₹1650     | ₹1000       | 650    | Spread widening... |
| 3   | ₹1700     | ₹1000       | 700    | Too wide! Buy ICICI, Short HDFC |
| 4   | ₹1650     | ₹1020       | 630    | Converging... |
| 5   | ₹1620     | ₹1010       | 610    | Back to normal. Close both. Profit! |

You profited because HDFC fell ₹80 (you were short) and ICICI rose ₹10 (you were long). Net gain even though both stocks moved.

---

## 2. Correlation vs Cointegration

> **Note:** High correlation ≠ good pair for trading. This is the most common source of error in pair selection.

### Correlation

- Measures whether two stocks *move in the same direction* over time.
- HDFC and ICICI might have 0.95 correlation — when one goes up, the other usually does too.
- **Problem:** Two stocks can be highly correlated but drift apart permanently. Think of two stocks both trending upward but at different rates — correlated, but the gap keeps growing.

### Cointegration

- Measures whether the *difference* between two stocks stays bounded.
- Formally: two series are cointegrated if there exists a linear combination of them that is stationary (more on this below).
- **This is what we need.** We don't care if both go up or down — we care if the spread between them reverts to a mean.

### Analogy

- **Correlation:** A man and his friend walk in the same direction. (They might end up miles apart.)
- **Cointegration:** A man and his dog on a leash. They wander differently, but the leash keeps them within a bounded distance.

### Implementation

```python
# Correlation — easy but not what we want
corr = stock_a.corr(stock_b)  # e.g., 0.95

# Cointegration — this is what matters
from statsmodels.tsa.stattools import coint
score, p_value, _ = coint(stock_a, stock_b)
# p_value < 0.05 → cointegrated → good pair
```

---

## 3. Stationarity — The Foundation

### What is Stationarity?

A time series is **stationary** if its statistical properties (mean, variance) don't change over time.

- **Stationary:** Temperature fluctuations around a mean (e.g., 25°C ± 5°C year-round).
- **Non-stationary:** Stock prices — they trend upward/downward with no fixed mean.

### Why Do We Care?

Our entire strategy depends on the spread reverting to a mean. If the spread is non-stationary, there's no mean to revert to — it could drift forever. A stationary spread is a predictable spread.

### Types of Non-Stationarity

1. **Trend:** The series goes up or down over time (stock prices).
2. **Unit root:** Random shocks have permanent effects (a random walk — each step is equally likely up or down, and there's no pull back to center).
3. **Changing variance:** Volatility clusters (common in financial data).

### The Key Insight

Individual stock prices are almost always non-stationary (they have trends, they don't revert to a mean). But the *spread* between two cointegrated stocks can be stationary. That's the magic — we combine two non-stationary series to create a stationary one.

---

## 4. The Augmented Dickey-Fuller (ADF) Test

### What It Does

The ADF test checks whether a time series has a "unit root" — which means it's non-stationary. We use it to test if our spread is stationary.

### The Hypothesis

- **Null hypothesis (H₀):** The series has a unit root → non-stationary → BAD for us.
- **Alternative hypothesis (H₁):** The series is stationary → GOOD for us.

We want to **reject** H₀ (get a low p-value).

### The Math

The test fits this regression:

```
Δyₜ = α + βt + γyₜ₋₁ + δ₁Δyₜ₋₁ + δ₂Δyₜ₋₂ + ... + εₜ
```

Where:
- `Δyₜ = yₜ - yₜ₋₁` (the change in the series)
- `γ` is the key coefficient — if γ < 0, the series tends to revert (stationary)
- The "augmented" part adds lagged differences (δ terms) to handle autocorrelation

**If γ is significantly negative → reject H₀ → series is stationary.**

### In Python

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(spread, autolag='AIC')
adf_statistic = result[0]  # More negative = stronger evidence of stationarity
p_value = result[1]         # < 0.05 = stationary at 95% confidence

print(f"ADF Statistic: {adf_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Spread is stationary — good pair!")
else:
    print("Spread is non-stationary — skip this pair.")
```

### Interpretation Guide

| P-value | Verdict |
|---------|---------|
| < 0.01  | Strong evidence of stationarity ✅ |
| 0.01–0.05 | Moderate evidence ✅ |
| 0.05–0.10 | Weak evidence ⚠️ |
| > 0.10 | Not stationary ❌ |

---

## 5. OLS Regression & The Hedge Ratio

### Why We Need a Hedge Ratio

You can't just compute `Price_A - Price_B` as the spread. If HDFC trades at ₹1600 and ICICI at ₹1000, the raw difference is meaningless — it's dominated by the price level, not the relationship.

The **hedge ratio (β)** tells you how many units of Stock B to trade for every unit of Stock A, so that your position is truly market-neutral.

### OLS Regression

We run a simple linear regression:

```
log(Price_A) = α + β × log(Price_B) + ε
```

- **β (slope)** = the hedge ratio. If β = 0.8, for every 1 share of A, you trade 0.8 shares of B.
- **ε (residuals)** = the spread. This is what we test for stationarity.
- **α (intercept)** = the long-run equilibrium level of the spread.

### In Python

```python
import statsmodels.api as sm
import numpy as np

# Log prices
y = np.log(price_a)  # Dependent variable
x = np.log(price_b)  # Independent variable
x = sm.add_constant(x)  # Add intercept

model = sm.OLS(y, x).fit()
beta = model.params[1]      # Hedge ratio
spread = model.resid         # The spread (residuals)

print(f"Hedge ratio (β): {beta:.4f}")
# If β = 1.2, for every ₹1 of Stock A, you need ₹1.2 of Stock B
```

### Why the Hedge Ratio Matters

Without it, your "market-neutral" position isn't actually neutral. If HDFC moves 1% and you're holding equal rupee amounts, but HDFC is more volatile, you have hidden directional exposure.

> **Note:** The hedge ratio is non-stationary. It is recomputed on a rolling 120-day window to track structural drift in the relationship.

---

## 6. Why Log Prices?

Log-transformed prices are used throughout the strategy for the following reasons:

### Reason 1: Percentage Changes

- `log(Price_today) - log(Price_yesterday) ≈ percentage return`
- This means our spread represents a *percentage* relationship, not a rupee one.
- A ₹10 move on a ₹100 stock (10%) is very different from ₹10 on a ₹1000 stock (1%). Logs normalize this.

### Reason 2: Multiplicative → Additive

- Stock prices are multiplicative (returns compound).
- Logs turn multiplication into addition, making the math cleaner.
- `log(A/B) = log(A) - log(B)` — ratios become differences.

### Reason 3: Better Statistical Properties

- Log returns are more normally distributed than raw returns.
- OLS regression assumptions are better satisfied with log-transformed data.

```python
# Raw spread (bad)
spread_raw = price_a - beta * price_b  # Dominated by price levels

# Log spread (good)
spread_log = np.log(price_a) - beta * np.log(price_b)  # Percentage relationship
```

---

## 7. The Spread & Z-Score

### Computing the Spread

```
spread = log(Price_A) − β × log(Price_B)
```

This is the residual from our regression. When it's near zero, the pair is in equilibrium. When it's far from zero, the pair has diverged.

### The Z-Score

The raw spread value is hard to interpret — is a spread of 0.05 "big"? Depends on context. The z-score standardizes it:

```
z-score = (spread − mean(spread)) / std(spread)
```

Now we're measuring "how many standard deviations away from the mean" the spread is. This is universal and interpretable.

### Trading Rules

| Z-Score | Interpretation | Action |
|---------|---------------|--------|
| z < −2.0 | Spread is unusually low (A is cheap relative to B) | **Go long:** Buy A, Short B |
| z > +2.0 | Spread is unusually high (A is expensive relative to B) | **Go short:** Short A, Buy B |
| abs(z) < 0.5 | Spread is near normal | **Exit** any open position |

### Critical: Rolling Window

> **Warning:** Computing mean and std over the full dataset introduces lookahead bias — future information leaks into the signal. A rolling window is required:

```python
# WRONG — uses future data
z_score = (spread - spread.mean()) / spread.std()

# RIGHT — only uses past 60 days
rolling_mean = spread.rolling(window=60).mean()
rolling_std = spread.rolling(window=60).std()
z_score = (spread - rolling_mean) / rolling_std
```

---

## 8. Mean Reversion & Half-Life

### What is Mean Reversion?

Mean reversion means the spread tends to return to its average over time. After stretching to z = +2.5, it doesn't keep going to +10 — it pulls back toward 0.

### Half-Life

The **half-life** tells you: *how many days does it take the spread to revert halfway back to the mean?*

- Half-life = 5 days → Fast reversion → Good for trading (quick trades).
- Half-life = 50 days → Slow reversion → Bad (your money is tied up too long).
- **Sweet spot: 5–30 trading days.**

### The Math

We model the spread as an **Ornstein-Uhlenbeck (OU) process** — a continuous-time mean-reverting process. In discrete time, this simplifies to an AR(1) model:

```
spreadₜ - spreadₜ₋₁ = λ × spreadₜ₋₁ + εₜ
```

Where:
- `λ` (lambda) is the mean-reversion speed. Should be **negative** (pulls back toward 0).
- Half-life = `−log(2) / log(1 + λ)` ≈ `−log(2) / λ` (when λ is small)

### In Python

```python
import numpy as np
import statsmodels.api as sm

# Fit AR(1) model
spread_lag = spread.shift(1).dropna()
spread_diff = spread.diff().dropna()

# Align
spread_lag = spread_lag.iloc[1:]
spread_diff = spread_diff.iloc[1:]

model = sm.OLS(spread_diff, spread_lag).fit()
lambda_param = model.params[0]

half_life = -np.log(2) / lambda_param
print(f"Half-life: {half_life:.1f} days")

if 5 <= half_life <= 30:
    print("Good mean-reversion speed")
else:
    print("Too fast or too slow for practical trading")
```

---

## 9. Rolling Windows & Lookahead Bias

### What is Lookahead Bias?

Lookahead bias occurs when a strategy uses information that would not have been available at the time of the trade — the primary cause of backtest overfitting.

### Examples

| What You Did | Why It's Wrong |
|-------------|---------------|
| Used full-period mean/std for z-score | On Day 100, you're using data from Day 500 |
| Selected pairs using all 5 years, then backtested on all 5 years | You already know which pairs work |
| Computed β once over the full period | β might have been different at different times |

### The Fix: Rolling Windows

At each point in time, you only use data from the **past N days**:

```python
lookback = 60  # Only use the last 60 days

for each day t:
    training_data = data[t-60 : t]  # Past only!
    beta = compute_hedge_ratio(training_data)
    z_score = compute_z_score(spread[t], training_data)
    signal = generate_signal(z_score)
```

In practice, we vectorize this with pandas `.rolling()` — no explicit loops.

---

## 10. Walk-Forward Validation

### Why Not a Simple Train/Test Split?

A single split is fragile — your test period might happen to be favorable or unfavorable. Walk-forward validation gives you multiple out-of-sample tests.

### How It Works

```
|--- Train 1 ---|--- Test 1 ---|
      |--- Train 2 ---|--- Test 2 ---|
            |--- Train 3 ---|--- Test 3 ---|
                  |--- Train 4 ---|--- Test 4 ---|
```

1. Train on Window 1 (e.g., 252 trading days = 1 year).
2. Test on the next window (e.g., 63 trading days = 1 quarter).
3. Slide both windows forward by the test length.
4. Repeat.

### What You're Checking

- Does the pair stay cointegrated across windows?
- Does the strategy make money consistently, or only in one lucky period?
- Does β stay stable, or does it drift wildly?

### Implementation Detail

- **Training window:** 504 trading days (~2 years) — pair selection and parameter fitting.
- **Test window:** 126 trading days (~6 months) — out-of-sample execution, never seen during training.
- **Within the test window:** Rolling 120-day β and 30-day z-score windows.

---

## 11. Transaction Costs & Slippage

### Why They Matter

Pairs trading generates many trades (often 30–100+ per pair per year). Each trade has costs that eat into returns. A strategy showing 15% annual return with zero costs might actually lose money with realistic costs.

### Types of Costs

| Cost | What It Is | Typical for NSE |
|------|-----------|----------------|
| Brokerage | Fee to your broker per trade | 0.01–0.03% |
| STT (Securities Transaction Tax) | Government tax on sell-side | 0.025% |
| Exchange fees | NSE transaction charges | ~0.003% |
| GST | Tax on brokerage | 18% of brokerage |
| **Total per leg** | Combined | **~0.05%** |
| Slippage | Price moves between decision and execution | ~0.02–0.03% |

### Implementation

```python
cost_per_leg = 0.0005  # 0.05%
total_cost_per_trade = cost_per_leg * 2  # Two legs: buy one stock, sell another
# Add slippage
total_cost = total_cost_per_trade + 0.0003  # 0.03% slippage

# Apply to returns
trade_return = raw_return - total_cost  # Per round-trip trade
```

> **Note:** Ignoring transaction costs is the most common source of backtest inflation in high-frequency relative-value strategies.

---

## 12. Risk Management Basics

### Why It Matters

Even a profitable strategy can blow up without risk management. One pair might break cointegration permanently (e.g., a company gets acquired), and the spread diverges forever.

### Strategy-Specific Controls

**1. Stop-Loss**
If z-score exceeds ±4.0, the cointegration relationship might be broken. Cut the position instead of waiting for a reversion that may never come.

**2. Position Sizing**
Don't put all your capital in one pair. If you have 6 pairs, allocate roughly 15% of capital to each, keeping some cash buffer.

**3. Maximum Holding Period**
If a trade hasn't reverted in, say, 2× the half-life, close it. Something may have changed fundamentally.

**4. Net Market Exposure**
Monitor that your combined long and short positions approximately cancel out in rupee terms. If they don't, you're taking directional bets.

---

## 13. Performance Metrics

### Sharpe Ratio

**What it measures:** Risk-adjusted return — how much return you get per unit of risk (volatility).

```
Sharpe = (Mean Return − Risk-Free Rate) / Std Dev of Returns
```

Annualized (for daily returns):
```
Sharpe_annual = Sharpe_daily × √252
```

| Sharpe | Interpretation |
|--------|---------------|
| < 0.5  | Poor |
| 0.5–1.0 | Acceptable |
| 1.0–2.0 | Good |
| > 2.0  | Excellent (or suspicious — check for overfitting) |

### Maximum Drawdown

**What it measures:** The worst peak-to-trough decline in your portfolio. "How much did I lose at the worst moment?"

```python
cumulative = (1 + returns).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_drawdown = drawdown.min()  # e.g., -0.15 = 15% max drawdown
```

### CAGR (Compound Annual Growth Rate)

**What it measures:** Your average annual return, accounting for compounding.

```
CAGR = (Final Value / Initial Value)^(1/years) − 1
```

### Win Rate

**What it measures:** Percentage of trades that were profitable.

```
Win Rate = Profitable Trades / Total Trades
```

Note: Win rate alone is misleading. A 40% win rate is fine if winners are much larger than losers.

### Average Holding Period

How many days you hold each trade on average. For pairs trading, expect 5–20 days (related to the half-life).

---

## 14. Putting It All Together

Here's the conceptual flow of the entire strategy:

```
Step 1: Download Data
   ↓
Step 2: Group by Sector → Form Candidate Pairs
   ↓
Step 3: For Each Pair:
   ├── Compute spread (log prices + OLS)
   ├── Run ADF test on spread
   ├── Compute half-life
   └── Rank by ADF p-value and half-life
   ↓
Step 4: Select Top Pairs (ADF p < 0.05, half-life 5–30 days)
   ↓
Step 5: On Held-Out Test Data:
   ├── Compute rolling β (60-day window)
   ├── Compute rolling z-score
   ├── Generate signals (enter at ±2.0, exit at ±0.5)
   ├── Apply transaction costs
   └── Track P&L per trade
   ↓
Step 6: Compute Metrics (Sharpe, Drawdown, CAGR, Win Rate)
   ↓
Step 7: Dashboard (Streamlit)
   ├── Spread plots
   ├── Z-score with signal overlays
   ├── Equity curves
   └── Per-pair metric tables
```

---

## 15. Further Reading

### Foundational
- **"Pairs Trading" by Ganapathy Vidyamurthy** — The definitive book on the topic.
- **"Algorithmic Trading" by Ernie Chan** — Chapters on mean reversion are excellent and practical.

### Papers
- Gatev, Goetzmann & Rouwenhorst (2006) — "Pairs Trading: Performance of a Relative Value Arbitrage Rule" — the seminal academic paper.

### Online
- [QuantStart: Pairs Trading](https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/) — Good practical intro.
- [statsmodels ADF documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
- [Hudson & Thames: Cointegration-based Pairs Trading](https://hudsonthames.org/) — Advanced implementations.

### Python-Specific
- statsmodels docs for `OLS`, `adfuller`, `coint`
- pandas `.rolling()` documentation
