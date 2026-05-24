# ============================================================
# backtester.py — Vectorized Backtest Engine
# ============================================================
# Simulates the pairs trading strategy over historical data
# and computes P&L, equity curves, and trade logs.
#
# Key features:
#   - Vectorized P&L computation (fast, no loops for returns)
#   - Realistic transaction costs and slippage
#   - Per-pair and portfolio-level equity curves
#   - Detailed trade log (entry/exit dates, P&L, holding period)
#   - Nifty 50 benchmark comparison
#
# Usage:
#   from src.backtester import run_backtest
#   results = run_backtest(prices, selected_pairs)
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.signal_generator import generate_signals
from src.metrics import compute_all_metrics
from src.risk_manager import compute_position_sizes, compute_net_exposure


# ============================================================
# Step 1: Compute per-pair returns
# ============================================================

def compute_pair_returns(
    signals_df: pd.DataFrame,
    cost_per_leg: float = None,
    slippage: float = None,
) -> pd.DataFrame:
    """
    Compute daily returns for a single pair from its signals.

    The correct way to compute pair returns:
        log_return_A = log(A_t / A_{t-1})
        log_return_B = log(B_t / B_{t-1})
        pair_return_t = signal_{t-1} × (log_return_A_t - β_{t-1} × log_return_B_t)

    Key details:
    - We use yesterday's signal AND yesterday's β (no lookahead)
    - Returns come from individual stock log-returns, NOT spread.diff()
      (spread.diff() would include fake jumps from β changes)
    - Transaction costs deducted on every signal change

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output from signal_generator (has columns: price_a, price_b, beta, signal)
    cost_per_leg : float
        Cost per leg of the trade. Default from config.
    slippage : float
        Slippage per trade. Default from config.

    Returns
    -------
    pd.DataFrame with columns:
        - pair_return: raw return from the pair position
        - gross_return: return before costs
        - cost: transaction cost incurred (on signal changes)
        - net_return: final return after costs
    """
    if cost_per_leg is None:
        cost_per_leg = config.COST_PER_LEG
    if slippage is None:
        slippage = config.SLIPPAGE

    # Total cost per entry or exit (both legs)
    total_cost_per_trade = (cost_per_leg * 2) + slippage

    # Individual stock log-returns
    log_return_a = np.log(signals_df["price_a"] / signals_df["price_a"].shift(1))
    log_return_b = np.log(signals_df["price_b"] / signals_df["price_b"].shift(1))

    # Pair return using PREVIOUS day's beta and signal
    # This is the return you'd earn from being long A / short β×B
    prev_beta = signals_df["beta"].shift(1)
    prev_signal = signals_df["signal"].shift(1)

    pair_return = log_return_a - prev_beta * log_return_b
    gross_return = prev_signal * pair_return

    # Transaction costs: incurred on each signal change
    signal_change = signals_df["signal"].diff().abs()
    cost = signal_change.fillna(0) * total_cost_per_trade

    # Net return = gross - costs
    net_return = gross_return - cost

    return pd.DataFrame({
        "pair_return": pair_return,
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
    }, index=signals_df.index)


# ============================================================
# Step 2: Extract trade log
# ============================================================

def extract_trade_log(signals_df: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    """
    Extract individual trades from the signal series.

    A trade starts when signal goes from 0 to ±1 (entry)
    and ends when signal returns to 0 (exit).

    Returns a DataFrame with one row per trade:
        - pair, direction, entry_date, exit_date
        - entry_z, exit_z, holding_days
        - trade_return (accumulated log-return, consistent with equity curve)
    """
    signal = signals_df["signal"]
    zscore = signals_df["zscore"]
    price_a = signals_df["price_a"]
    price_b = signals_df["price_b"]
    beta = signals_df["beta"]

    # Precompute daily log-returns
    log_ret_a = np.log(price_a / price_a.shift(1))
    log_ret_b = np.log(price_b / price_b.shift(1))

    trades = []
    in_trade = False
    entry_date = None
    entry_z = None
    direction = 0
    cumulative_return = 0.0

    for i, (date, sig) in enumerate(signal.items()):
        if not in_trade and sig != 0:
            # Entry
            in_trade = True
            entry_date = date
            entry_z = zscore.loc[date]
            direction = sig
            cumulative_return = 0.0

        elif in_trade:
            # Accumulate return using previous day's beta
            if i > 0 and not np.isnan(log_ret_a.iloc[i]) and not np.isnan(log_ret_b.iloc[i]):
                day_return = direction * (log_ret_a.iloc[i] - beta.iloc[i-1] * log_ret_b.iloc[i])
                cumulative_return += day_return

            if sig == 0:
                # Exit
                exit_z = zscore.loc[date]

                trades.append({
                    "pair": pair_name,
                    "direction": "Long" if direction == 1 else "Short",
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_z": round(float(entry_z), 2) if not np.isnan(entry_z) else 0,
                    "exit_z": round(float(exit_z), 2) if not np.isnan(exit_z) else 0,
                    "holding_days": (date - entry_date).days,
                    "trade_return": round(float(cumulative_return), 6),
                })

                in_trade = False
                entry_date = None

    return pd.DataFrame(trades)


# ============================================================
# Step 3: Download benchmark
# ============================================================

def get_benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Download Nifty 50 returns for benchmark comparison.

    Uses ^NSEI (Nifty 50 index) from yfinance.
    Falls back to equal-weight portfolio of our tickers if download fails.
    """
    test_start = prices.index[-config.TEST_PERIOD_DAYS]
    test_end = prices.index[-1]

    try:
        nifty = yf.download(
            "^NSEI",
            start=test_start - pd.Timedelta(days=10),  # Buffer
            end=test_end + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )

        if isinstance(nifty.columns, pd.MultiIndex):
            nifty_close = nifty["Close"].squeeze()
        else:
            nifty_close = nifty["Close"]

        nifty_returns = nifty_close.pct_change().dropna()

        # Align with test period
        common_idx = nifty_returns.index.intersection(prices.index[-config.TEST_PERIOD_DAYS:])
        nifty_returns = nifty_returns.loc[common_idx]

        print(f"📈 Loaded Nifty 50 benchmark ({len(nifty_returns)} days)")
        return nifty_returns

    except Exception as e:
        print(f"⚠️  Could not download Nifty 50: {e}")
        print("   Using equal-weight portfolio as benchmark instead.")

        test_prices = prices.iloc[-config.TEST_PERIOD_DAYS:]
        benchmark = test_prices.pct_change().mean(axis=1).dropna()
        return benchmark


# ============================================================
# Step 4: Main backtest runner
# ============================================================

def run_backtest(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    verbose: bool = True,
    leverage: float = None,
) -> dict:
    """
    Run the complete backtest pipeline.

    Steps:
    1. Generate signals for all pairs on the test period
    2. Compute per-pair daily returns (with transaction costs)
    3. Build per-pair equity curves
    4. Combine into portfolio returns (equal-weighted)
    5. Extract trade logs
    6. Compute performance metrics
    7. Compare against Nifty 50 benchmark

    Parameters
    ----------
    prices : pd.DataFrame
        Full price data (train + test).
    selected_pairs : pd.DataFrame
        Selected pairs from pair_selector.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with:
        - pair_returns: dict of per-pair return DataFrames
        - pair_equity: dict of per-pair equity curves
        - pair_metrics: dict of per-pair metrics
        - pair_trades: dict of per-pair trade logs
        - portfolio_returns: combined daily returns
        - portfolio_equity: combined equity curve
        - portfolio_metrics: overall metrics
        - benchmark_returns: Nifty 50 daily returns
        - benchmark_equity: Nifty 50 equity curve
        - all_trades: combined trade log
        - position_sizes: capital allocation info
        - signals: dict of signal DataFrames
    """
    if verbose:
        print("=" * 60)
        print("BACKTEST ENGINE")
        print("=" * 60)

    # --- Step 1: Generate signals ---
    if verbose:
        print("\n📡 Step 1: Generating signals...")

    all_signals = generate_signals(
        prices, selected_pairs,
        use_test_period=True, verbose=verbose
    )

    # --- Step 2: Position sizing ---
    position_sizes = compute_position_sizes(len(all_signals))
    if verbose:
        print(f"\n💰 Step 2: Position sizing")
        print(f"   Capital: ₹{position_sizes['initial_capital']:,.0f}")
        print(f"   Per pair: ₹{position_sizes['capital_per_pair']:,.0f}")
        print(f"   Cash buffer: ₹{position_sizes['cash_buffer']:,.0f} "
              f"({position_sizes['cash_buffer_pct']}%)")

    # --- Step 3: Compute per-pair returns ---
    if verbose:
        print(f"\n📊 Step 3: Computing returns...")

    pair_returns = {}
    pair_equity = {}
    pair_metrics = {}
    pair_trades = {}
    all_trades_list = []

    for pair_name, signals_df in all_signals.items():
        # Daily returns
        returns_df = compute_pair_returns(signals_df)
        pair_returns[pair_name] = returns_df

        # Equity curve (starting from 1.0)
        equity = (1 + returns_df["net_return"].fillna(0)).cumprod()
        pair_equity[pair_name] = equity

        # Trade log
        trades = extract_trade_log(signals_df, pair_name)
        pair_trades[pair_name] = trades
        all_trades_list.append(trades)

        # Per-pair metrics
        trade_pnls = trades["trade_return"] if len(trades) > 0 else pd.Series(dtype=float)
        metrics = compute_all_metrics(
            returns_df["net_return"].dropna(),
            trade_pnls
        )
        pair_metrics[pair_name] = metrics

        if verbose:
            print(f"   {pair_name}: "
                  f"Return={metrics['total_return']:.1f}% | "
                  f"Sharpe={metrics['sharpe_ratio']:.2f} | "
                  f"MaxDD={metrics['max_drawdown']:.1f}% | "
                  f"Trades={metrics.get('n_trades', 0)}")

    # --- Step 4: Portfolio returns (equal-weighted) ---
    if verbose:
        print(f"\n📈 Step 4: Building portfolio...")

    if leverage is None:
        leverage = getattr(config, "LEVERAGE", 1.0)

    # Combine all pair returns into portfolio
    returns_matrix = pd.DataFrame({
        name: ret["net_return"] for name, ret in pair_returns.items()
    })

    # Equal-weight: average across all pairs, scaled by leverage
    portfolio_returns = returns_matrix.mean(axis=1).fillna(0) * leverage
    portfolio_equity = (1 + portfolio_returns).cumprod()

    # Combined trade log
    all_trades = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()
    if len(all_trades) > 0 and "trade_return" in all_trades.columns:
        all_trades["trade_return"] = all_trades["trade_return"] * leverage
    trade_pnls = all_trades["trade_return"] if len(all_trades) > 0 else pd.Series(dtype=float)

    # Portfolio metrics
    portfolio_metrics = compute_all_metrics(portfolio_returns.dropna(), trade_pnls)

    if verbose:
        print(f"\n   PORTFOLIO SUMMARY:")
        print(f"   Total Return: {portfolio_metrics['total_return']:.2f}%")
        print(f"   CAGR: {portfolio_metrics['cagr']:.2f}%")
        print(f"   Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {portfolio_metrics['max_drawdown']:.2f}%")
        print(f"   Volatility: {portfolio_metrics['volatility']:.2f}%")
        if "n_trades" in portfolio_metrics:
            print(f"   Total Trades: {portfolio_metrics['n_trades']}")
            print(f"   Win Rate: {portfolio_metrics['win_rate']:.1f}%")
            print(f"   Profit Factor: {portfolio_metrics['profit_factor']:.2f}")

    # --- Step 5: Benchmark ---
    if verbose:
        print(f"\n🏦 Step 5: Loading benchmark...")

    benchmark_returns = get_benchmark_returns(prices)
    benchmark_equity = (1 + benchmark_returns).cumprod()

    benchmark_metrics = compute_all_metrics(benchmark_returns.dropna())

    if verbose:
        print(f"   Nifty 50 Return: {benchmark_metrics['total_return']:.2f}%")
        print(f"   Nifty 50 Sharpe: {benchmark_metrics['sharpe_ratio']:.2f}")
        print(f"   Nifty 50 MaxDD:  {benchmark_metrics['max_drawdown']:.2f}%")

    # --- Compile results ---
    results = {
        "pair_returns": pair_returns,
        "pair_equity": pair_equity,
        "pair_metrics": pair_metrics,
        "pair_trades": pair_trades,
        "portfolio_returns": portfolio_returns,
        "portfolio_equity": portfolio_equity,
        "portfolio_metrics": portfolio_metrics,
        "benchmark_returns": benchmark_returns,
        "benchmark_equity": benchmark_equity,
        "benchmark_metrics": benchmark_metrics,
        "all_trades": all_trades,
        "position_sizes": position_sizes,
        "signals": all_signals,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ Backtest complete!")
        print(f"{'='*60}")

    return results


# ============================================================
# Run directly for quick testing
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.pair_selector import run_pair_selection, get_selected_pairs

    prices = load_data()
    pair_results = run_pair_selection(prices, verbose=False)
    selected = get_selected_pairs(pair_results)

    results = run_backtest(prices, selected)
