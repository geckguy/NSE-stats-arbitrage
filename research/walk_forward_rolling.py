"""
============================================================
walk_forward_rolling.py — Rolling Walk-Forward Validation
============================================================
The realistic approach: re-select pairs every 6 months.

How a real quant fund would run this:
  1. At the start of each 6-month period, look back 2 years
  2. Find the best cointegrated pairs in that window
  3. Trade those pairs for 6 months
  4. Repeat — drop broken pairs, pick new ones

This adapts to changing market conditions instead of
assuming pairs discovered in 2021 still work in 2025.

Usage:
    python walk_forward_rolling.py
============================================================
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import load_data
from src.pair_selector import (
    generate_candidate_pairs,
    compute_spread,
    test_cointegration,
    test_cointegration_engle_granger,
    compute_half_life,
    test_cointegration_stability,
    score_pair,
)
from src.signal_generator import (
    compute_rolling_beta_fast,
    compute_rolling_spread,
    compute_rolling_zscore,
    generate_raw_signals,
    apply_risk_overlay,
)
from src.backtester import compute_pair_returns, extract_trade_log
from src.metrics import compute_all_metrics

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

TRAIN_WINDOW_DAYS = 504    # ~2 years of training data
REBALANCE_DAYS = 126       # Re-select pairs every ~6 months
WARMUP_DAYS = 150          # Buffer for rolling calculations

# Use conservative (un-tuned) parameters for fair test
Z_ENTRY = 2.0
Z_EXIT = 0.5
LOOKBACK = 60
Z_STOP = 4.0
MAX_PAIRS = 8
MIN_STABILITY = 30.0      # Minimum stability % to accept a pair


# ============================================================
# Select pairs on a training window
# ============================================================

def select_pairs_on_window(prices: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Run pair selection on the given price slice."""
    candidates = generate_candidate_pairs(prices)
    results = []

    for pair in candidates:
        stock_a, stock_b, sector = pair["stock_a"], pair["stock_b"], pair["sector"]
        try:
            pa, pb = prices[stock_a], prices[stock_b]
            ols = compute_spread(pa, pb)
            adf = test_cointegration(ols["spread"])
            hl = compute_half_life(ols["spread"])

            if adf["is_cointegrated"]:
                stab = test_cointegration_stability(pa, pb)
                stability_pct = stab["stability_pct"]
            else:
                stability_pct = 0.0

            pair_score = score_pair(adf["adf_pvalue"], hl, stability_pct)

            results.append({
                "stock_a": stock_a, "stock_b": stock_b, "sector": sector,
                "beta": round(ols["beta"], 4),
                "adf_pvalue": round(adf["adf_pvalue"], 4),
                "is_cointegrated": adf["is_cointegrated"],
                "half_life": round(hl, 1),
                "stability_pct": round(stability_pct, 1),
                "score": pair_score,
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    selected = results_df[
        (results_df["is_cointegrated"])
        & (results_df["half_life"] >= config.HALF_LIFE_MIN)
        & (results_df["half_life"] <= config.HALF_LIFE_MAX)
        & (results_df["stability_pct"] >= MIN_STABILITY)
    ].head(MAX_PAIRS)

    return selected


# ============================================================
# Generate signals for one rebalance period
# ============================================================

def run_period(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict:
    """
    Generate signals and compute returns for one 6-month period.
    """
    # Get prices with warmup buffer
    warmup_start = period_start - pd.Timedelta(days=WARMUP_DAYS)
    signal_prices = prices.loc[warmup_start:period_end]

    all_returns = {}
    all_trades = []

    for _, pair in selected.iterrows():
        stock_a, stock_b = pair["stock_a"], pair["stock_b"]
        half_life = pair["half_life"]

        try:
            price_a = signal_prices[stock_a]
            price_b = signal_prices[stock_b]
        except KeyError:
            continue

        # Rolling calculations
        beta = compute_rolling_beta_fast(price_a, price_b, LOOKBACK)
        spread = compute_rolling_spread(price_a, price_b, beta)
        zscore = compute_rolling_zscore(spread, LOOKBACK)
        raw_signal = generate_raw_signals(zscore, Z_ENTRY, Z_EXIT)
        signal = apply_risk_overlay(raw_signal, zscore, half_life, Z_STOP)

        signals_df = pd.DataFrame({
            "price_a": price_a, "price_b": price_b,
            "beta": beta, "spread": spread,
            "zscore": zscore, "signal": signal,
        })

        # Trim to period
        signals_df = signals_df.loc[period_start:period_end]
        if len(signals_df) < 10:
            continue

        returns_df = compute_pair_returns(signals_df)
        pair_key = f"{stock_a.replace('.NS','')}-{stock_b.replace('.NS','')}"
        all_returns[pair_key] = returns_df

        trades = extract_trade_log(signals_df, pair_key)
        all_trades.append(trades)

    return {
        "returns": all_returns,
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
    }


# ============================================================
# Main: Rolling walk-forward
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ROLLING WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"\nRe-selects pairs every {REBALANCE_DAYS} trading days (~6 months)")
    print(f"Training window: {TRAIN_WINDOW_DAYS} days (~2 years)")
    print(f"Parameters: z_entry={Z_ENTRY}, z_exit={Z_EXIT}, lookback={LOOKBACK}")
    print(f"Min stability: {MIN_STABILITY}%\n")

    # Load data
    print("📂 Loading data...")
    prices = load_data(force_refresh=False)
    all_dates = prices.index

    print(f"   Data: {all_dates[0].date()} → {all_dates[-1].date()} ({len(all_dates)} days)")

    # Define rebalance points
    # Start testing after we have enough training data
    first_test_idx = TRAIN_WINDOW_DAYS
    rebalance_indices = list(range(first_test_idx, len(all_dates), REBALANCE_DAYS))

    print(f"   Rebalance points: {len(rebalance_indices)}")

    # Run each period
    all_period_returns = []
    all_period_trades = []
    period_summaries = []

    for i, rb_idx in enumerate(rebalance_indices):
        # Training window
        train_start_idx = max(0, rb_idx - TRAIN_WINDOW_DAYS)
        train_prices = prices.iloc[train_start_idx:rb_idx]

        # Test window (next 6 months or until data ends)
        test_end_idx = min(rb_idx + REBALANCE_DAYS, len(all_dates))
        period_start = all_dates[rb_idx]
        period_end = all_dates[test_end_idx - 1]

        print(f"\n{'─'*60}")
        print(f"  Period {i+1}/{len(rebalance_indices)}")
        print(f"  Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()}")
        print(f"  Test:  {period_start.date()} → {period_end.date()}")

        # Select pairs
        selected = select_pairs_on_window(train_prices)

        if len(selected) == 0:
            print(f"  ⚠️ No pairs found. Skipping period.")
            continue

        print(f"  Selected {len(selected)} pairs:")
        for _, row in selected.iterrows():
            a = row['stock_a'].replace('.NS', '')
            b = row['stock_b'].replace('.NS', '')
            print(f"    {a} ↔ {b} (HL={row['half_life']:.0f}d, stab={row['stability_pct']:.0f}%)")

        # Run signals + backtest
        result = run_period(prices, selected, period_start, period_end)

        if not result["returns"]:
            print(f"  ⚠️ No valid returns. Skipping.")
            continue

        # Portfolio returns for this period
        returns_matrix = pd.DataFrame({
            name: ret["net_return"] for name, ret in result["returns"].items()
        })
        period_returns = returns_matrix.mean(axis=1).fillna(0)

        all_period_returns.append(period_returns)
        all_period_trades.append(result["trades"])

        # Period metrics
        metrics = compute_all_metrics(
            period_returns.dropna(),
            result["trades"]["trade_return"] if len(result["trades"]) > 0 else None
        )

        period_summaries.append({
            "Period": f"{period_start.date()} → {period_end.date()}",
            "Return (%)": metrics["total_return"],
            "Sharpe": metrics["sharpe_ratio"],
            "Max DD (%)": metrics["max_drawdown"],
            "Vol (%)": metrics["volatility"],
            "Trades": metrics.get("n_trades", 0),
            "Win Rate (%)": metrics.get("win_rate", 0),
            "Pairs": len(result["returns"]),
        })

        print(f"  📊 Return: {metrics['total_return']:+.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
              f"MaxDD: {metrics['max_drawdown']:.2f}%")

    # ============================================================
    # Full-period combined equity curve
    # ============================================================

    print(f"\n\n{'='*70}")
    print(f"ROLLING WALK-FORWARD — FULL RESULTS")
    print(f"{'='*70}\n")

    if not all_period_returns:
        print("❌ No valid periods. Check data availability.")
        sys.exit(1)

    # Concatenate all periods into one continuous return series
    combined_returns = pd.concat(all_period_returns).sort_index()
    # Remove duplicate indices (overlap at rebalance boundaries)
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]

    combined_trades = pd.concat(all_period_trades, ignore_index=True) if all_period_trades else pd.DataFrame()
    trade_pnls = combined_trades["trade_return"] if len(combined_trades) > 0 else None

    overall_metrics = compute_all_metrics(combined_returns.dropna(), trade_pnls)

    # Per-period summary
    summary_df = pd.DataFrame(period_summaries)
    print("PER-PERIOD BREAKDOWN:\n")
    print(summary_df.to_string(index=False))

    # Overall
    n_periods = len(summary_df)
    n_profitable = (summary_df["Return (%)"] > 0).sum()

    print(f"\n{'─'*70}")
    print(f"OVERALL (combined across all periods):")
    print(f"  Total Return:   {overall_metrics['total_return']:+.2f}%")
    print(f"  CAGR:           {overall_metrics['cagr']:+.2f}%")
    print(f"  Sharpe Ratio:   {overall_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:   {overall_metrics['max_drawdown']:.2f}%")
    print(f"  Calmar Ratio:   {overall_metrics['calmar_ratio']:.2f}")
    print(f"  Volatility:     {overall_metrics['volatility']:.2f}%")
    print(f"  Total Trades:   {overall_metrics.get('n_trades', 0)}")
    print(f"  Win Rate:       {overall_metrics.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:  {overall_metrics.get('profit_factor', 0):.2f}")

    print(f"\n  Profitable periods: {n_profitable}/{n_periods} "
          f"({n_profitable/n_periods*100:.0f}%)")
    print(f"  Testing span: {combined_returns.index[0].date()} → "
          f"{combined_returns.index[-1].date()} "
          f"({len(combined_returns)} trading days)")

    # Verdict
    print(f"\n{'─'*70}")
    if overall_metrics["sharpe_ratio"] > 0.5 and n_profitable >= n_periods * 0.6:
        print("✅ Strategy shows a CONSISTENT edge across multiple periods!")
    elif overall_metrics["sharpe_ratio"] > 0 and n_profitable >= n_periods * 0.5:
        print("🟡 Strategy has a WEAK edge — works more often than not, but inconsistent.")
    else:
        print("❌ Strategy does NOT show a reliable edge across time periods.")

    # Save
    summary_df.to_csv("data/walk_forward_rolling_results.csv", index=False)
    combined_returns.to_csv("data/walk_forward_equity.csv")
    print(f"\n💾 Saved to data/walk_forward_rolling_results.csv")
