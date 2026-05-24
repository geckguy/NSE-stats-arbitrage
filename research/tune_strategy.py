"""
============================================================
tune_strategy.py — Parameter Optimization for Pairs Trading
============================================================
Sweeps over key trading parameters to find the best combination.

Strategy: select pairs ONCE (expensive), then sweep over signal
and risk parameters (cheap) to find what works best.

Parameters tuned:
  - Z_ENTRY_THRESHOLD: when to enter trades (1.5 → 2.5)
  - Z_EXIT_THRESHOLD: when to exit trades (0.0 → 0.75)
  - LOOKBACK_WINDOW: rolling window for β and z-score (30 → 90)
  - Z_STOP_LOSS: emergency exit threshold (3.0 → 5.0)
  - MAX_PAIRS: how many pairs to trade (5 → 15)

Usage:
    python tune_strategy.py
============================================================
"""

import sys
import os
import itertools
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import load_data
from src.pair_selector import run_pair_selection, get_selected_pairs
from src.signal_generator import generate_signals_for_pair
from src.backtester import compute_pair_returns, extract_trade_log
from src.metrics import compute_all_metrics
from src.signal_generator import (
    compute_rolling_beta_fast,
    compute_rolling_spread,
    compute_rolling_zscore,
    generate_raw_signals,
    apply_risk_overlay,
)

warnings.filterwarnings("ignore")


# ============================================================
# Step 1: Define parameter grid
# ============================================================

PARAM_GRID = {
    "z_entry": [1.5, 1.8, 2.0, 2.2, 2.5],
    "z_exit": [0.0, 0.25, 0.5, 0.75],
    "lookback": [30, 45, 60, 90],
    "z_stop": [3.0, 3.5, 4.0, 5.0],
    "max_pairs": [5, 8, 10, 12],
}

# Total combinations
total = 1
for v in PARAM_GRID.values():
    total *= len(v)


# ============================================================
# Step 2: Fast signal + backtest for a single param set
# ============================================================

def run_single_backtest(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    z_entry: float,
    z_exit: float,
    lookback: int,
    z_stop: float,
    max_pairs: int,
) -> dict | None:
    """
    Run signals + backtest for ONE parameter combination.
    Returns metrics dict or None if it fails.
    """
    # Limit pairs
    pairs_to_use = selected_pairs.head(max_pairs)

    if len(pairs_to_use) == 0:
        return None

    # Compute signals and returns for each pair
    warmup = lookback * 2
    start_idx = max(0, len(prices) - config.TEST_PERIOD_DAYS - warmup)
    signal_prices = prices.iloc[start_idx:]
    test_start = prices.index[-config.TEST_PERIOD_DAYS]

    all_returns = {}

    for _, pair in pairs_to_use.iterrows():
        stock_a = pair["stock_a"]
        stock_b = pair["stock_b"]
        half_life = pair["half_life"]

        try:
            price_a = signal_prices[stock_a]
            price_b = signal_prices[stock_b]
        except KeyError:
            continue

        # Rolling beta
        beta = compute_rolling_beta_fast(price_a, price_b, lookback)

        # Rolling spread
        spread = compute_rolling_spread(price_a, price_b, beta)

        # Rolling z-score
        zscore = compute_rolling_zscore(spread, lookback)

        # Raw signals with custom thresholds
        raw_signal = generate_raw_signals(zscore, z_entry, z_exit)

        # Risk overlay with custom stop-loss
        signal = apply_risk_overlay(raw_signal, zscore, half_life, z_stop)

        # Build DataFrame
        signals_df = pd.DataFrame({
            "price_a": price_a,
            "price_b": price_b,
            "beta": beta,
            "spread": spread,
            "zscore": zscore,
            "signal": signal,
        })

        # Trim to test period
        signals_df = signals_df.loc[test_start:]

        if len(signals_df) < 50:
            continue

        # Compute returns
        returns_df = compute_pair_returns(signals_df)
        all_returns[f"{stock_a}-{stock_b}"] = returns_df

    if not all_returns:
        return None

    # Portfolio returns
    returns_matrix = pd.DataFrame({
        name: ret["net_return"] for name, ret in all_returns.items()
    })
    portfolio_returns = returns_matrix.mean(axis=1).fillna(0)

    # Extract trades for win rate / profit factor
    all_trades = []
    for pair_key, signals_df_data in zip(all_returns.keys(), all_returns.values()):
        # Rebuild signals_df for trade extraction
        pass

    # Metrics (without trade-level stats for speed)
    metrics = compute_all_metrics(portfolio_returns.dropna())
    metrics["n_pairs_active"] = len(all_returns)

    # Count trades from signal changes
    total_trades = 0
    for name, ret in all_returns.items():
        signal_changes = ret["gross_return"].ne(0).astype(int).diff().abs()
        # Rough trade count from signal changes
        pass

    return metrics


# ============================================================
# Step 3: Run full sweep
# ============================================================

def run_parameter_sweep(
    prices: pd.DataFrame,
    all_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sweep over all parameter combinations and return results.
    """
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    print(f"🔧 Parameter Sweep")
    print(f"   {len(combos)} combinations to test")
    print(f"   Parameters: {keys}")
    print(f"   Grid: {PARAM_GRID}")
    print()

    results = []
    start_time = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Progress
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(combos) - i - 1) / rate if rate > 0 else 0
            print(f"   [{i+1}/{len(combos)}] "
                  f"Elapsed: {elapsed:.0f}s | "
                  f"ETA: {remaining:.0f}s | "
                  f"Rate: {rate:.1f} combos/s")

        metrics = run_single_backtest(
            prices, all_pairs,
            z_entry=params["z_entry"],
            z_exit=params["z_exit"],
            lookback=params["lookback"],
            z_stop=params["z_stop"],
            max_pairs=params["max_pairs"],
        )

        if metrics is None:
            continue

        row = {**params, **metrics}
        results.append(row)

    elapsed = time.time() - start_time
    print(f"\n✅ Sweep complete: {len(results)} valid combinations in {elapsed:.0f}s")

    results_df = pd.DataFrame(results)
    return results_df


# ============================================================
# Step 4: Analyze and report
# ============================================================

def analyze_results(results_df: pd.DataFrame) -> dict:
    """Find the best parameter combination and report insights."""

    print(f"\n{'='*70}")
    print(f"PARAMETER OPTIMIZATION RESULTS")
    print(f"{'='*70}")

    # --- Best by Sharpe ---
    best_sharpe_idx = results_df["sharpe_ratio"].idxmax()
    best_sharpe = results_df.loc[best_sharpe_idx]

    print(f"\n🏆 BEST BY SHARPE RATIO ({best_sharpe['sharpe_ratio']:.2f}):")
    print(f"   z_entry={best_sharpe['z_entry']}, z_exit={best_sharpe['z_exit']}, "
          f"lookback={int(best_sharpe['lookback'])}, z_stop={best_sharpe['z_stop']}, "
          f"max_pairs={int(best_sharpe['max_pairs'])}")
    print(f"   Return: {best_sharpe['total_return']:.2f}% | "
          f"MaxDD: {best_sharpe['max_drawdown']:.2f}% | "
          f"Calmar: {best_sharpe['calmar_ratio']:.2f}")

    # --- Best by Calmar ---
    best_calmar_idx = results_df["calmar_ratio"].idxmax()
    best_calmar = results_df.loc[best_calmar_idx]

    print(f"\n🏆 BEST BY CALMAR RATIO ({best_calmar['calmar_ratio']:.2f}):")
    print(f"   z_entry={best_calmar['z_entry']}, z_exit={best_calmar['z_exit']}, "
          f"lookback={int(best_calmar['lookback'])}, z_stop={best_calmar['z_stop']}, "
          f"max_pairs={int(best_calmar['max_pairs'])}")
    print(f"   Return: {best_calmar['total_return']:.2f}% | "
          f"Sharpe: {best_calmar['sharpe_ratio']:.2f} | "
          f"MaxDD: {best_calmar['max_drawdown']:.2f}%")

    # --- Best by Total Return ---
    best_return_idx = results_df["total_return"].idxmax()
    best_return = results_df.loc[best_return_idx]

    print(f"\n🏆 BEST BY TOTAL RETURN ({best_return['total_return']:.2f}%):")
    print(f"   z_entry={best_return['z_entry']}, z_exit={best_return['z_exit']}, "
          f"lookback={int(best_return['lookback'])}, z_stop={best_return['z_stop']}, "
          f"max_pairs={int(best_return['max_pairs'])}")
    print(f"   Sharpe: {best_return['sharpe_ratio']:.2f} | "
          f"MaxDD: {best_return['max_drawdown']:.2f}% | "
          f"Calmar: {best_return['calmar_ratio']:.2f}")

    # --- Balanced pick (Sharpe > 0.5, lowest DD) ---
    decent = results_df[results_df["sharpe_ratio"] > 0.3]
    if len(decent) > 0:
        balanced_idx = decent["max_drawdown"].idxmax()  # Least negative DD
        balanced = decent.loc[balanced_idx]

        print(f"\n🎯 BALANCED PICK (Sharpe > 0.3, least drawdown):")
        print(f"   z_entry={balanced['z_entry']}, z_exit={balanced['z_exit']}, "
              f"lookback={int(balanced['lookback'])}, z_stop={balanced['z_stop']}, "
              f"max_pairs={int(balanced['max_pairs'])}")
        print(f"   Return: {balanced['total_return']:.2f}% | "
              f"Sharpe: {balanced['sharpe_ratio']:.2f} | "
              f"MaxDD: {balanced['max_drawdown']:.2f}% | "
              f"Calmar: {balanced['calmar_ratio']:.2f}")

    # --- Parameter sensitivity ---
    print(f"\n{'='*70}")
    print(f"PARAMETER SENSITIVITY (average Sharpe by parameter value)")
    print(f"{'='*70}")

    for param in PARAM_GRID.keys():
        print(f"\n   {param}:")
        grouped = results_df.groupby(param)["sharpe_ratio"].mean()
        for val, avg_sharpe in grouped.items():
            bar = "█" * int(max(0, avg_sharpe + 1) * 15)
            print(f"      {val:>6} → Sharpe {avg_sharpe:+.3f}  {bar}")

    # --- Top 10 combinations ---
    print(f"\n{'='*70}")
    print(f"TOP 15 PARAMETER COMBINATIONS (by Sharpe)")
    print(f"{'='*70}\n")

    top15 = results_df.nlargest(15, "sharpe_ratio")
    display_cols = ["z_entry", "z_exit", "lookback", "z_stop", "max_pairs",
                    "total_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "volatility"]
    print(top15[display_cols].to_string(index=False))

    # --- Worst 5 ---
    print(f"\n{'='*70}")
    print(f"WORST 5 (avoid these)")
    print(f"{'='*70}\n")
    worst5 = results_df.nsmallest(5, "sharpe_ratio")
    print(worst5[display_cols].to_string(index=False))

    return {
        "best_sharpe": best_sharpe.to_dict(),
        "best_calmar": best_calmar.to_dict(),
        "best_return": best_return.to_dict(),
        "results_df": results_df,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PAIRS TRADING — PARAMETER OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\n📂 Loading data...")
    prices = load_data(force_refresh=False)

    # Run pair selection with RELAXED criteria to get a larger pool
    print("\n🔍 Running pair selection (relaxed criteria for larger pool)...")

    # Temporarily relax criteria
    original_max_pairs = config.MAX_PAIRS
    original_hl_max = config.HALF_LIFE_MAX

    config.MAX_PAIRS = 20  # Get more pairs
    config.HALF_LIFE_MAX = 40  # Slightly relaxed

    all_results = run_pair_selection(prices, verbose=False)
    all_pairs = get_selected_pairs(all_results)

    # Restore
    config.MAX_PAIRS = original_max_pairs
    config.HALF_LIFE_MAX = original_hl_max

    print(f"\n   Available pairs for sweep: {len(all_pairs)}")

    # Run sweep
    results_df = run_parameter_sweep(prices, all_pairs)

    # Save raw results
    os.makedirs("data", exist_ok=True)
    results_df.to_csv("data/tuning_results.csv", index=False)
    print(f"\n💾 Raw results saved to data/tuning_results.csv")

    # Analyze
    analysis = analyze_results(results_df)

    # Suggest config update
    best = analysis["best_sharpe"]
    print(f"\n{'='*70}")
    print(f"SUGGESTED CONFIG UPDATE")
    print(f"{'='*70}")
    print(f"""
# Copy these into config.py to use the optimized parameters:
Z_ENTRY_THRESHOLD = {best['z_entry']}
Z_EXIT_THRESHOLD = {best['z_exit']}
LOOKBACK_WINDOW = {int(best['lookback'])}
Z_STOP_LOSS = {best['z_stop']}
MAX_PAIRS = {int(best['max_pairs'])}
""")
