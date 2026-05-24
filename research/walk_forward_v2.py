"""
============================================================
walk_forward_v2.py — Improved Rolling Walk-Forward
============================================================
Changes from v1:
  1. Higher stability filter (60%+ from config)
  2. Require both ADF + Engle-Granger cointegration tests
  3. Spread velocity filter (don't enter diverging spreads)
  4. Pair-level drawdown stop (cut losing pairs early)

Usage:
    python walk_forward_v2.py
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

TRAIN_WINDOW_DAYS = 504    # ~2 years training
REBALANCE_DAYS = 126       # Re-select every ~6 months
WARMUP_DAYS = 150          # Buffer for rolling calcs

# Use conservative (un-tuned) parameters
Z_ENTRY = 2.0
Z_EXIT = 0.5
LOOKBACK = 60
Z_STOP = 4.0


# ============================================================
# Select pairs (with improved filtering)
# ============================================================

def select_pairs_on_window(prices: pd.DataFrame) -> pd.DataFrame:
    """Run pair selection with dual cointegration + stability filter."""
    candidates = generate_candidate_pairs(prices)
    results = []

    for pair in candidates:
        stock_a, stock_b, sector = pair["stock_a"], pair["stock_b"], pair["sector"]
        try:
            pa, pb = prices[stock_a], prices[stock_b]
            ols = compute_spread(pa, pb)
            adf = test_cointegration(ols["spread"])

            # IMPROVEMENT 2: Run both cointegration tests
            eg = test_cointegration_engle_granger(pa, pb)

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
                "eg_pvalue": round(eg["eg_pvalue"], 4),
                "eg_is_cointegrated": eg["eg_is_cointegrated"],
                "half_life": round(hl, 1),
                "stability_pct": round(stability_pct, 1),
                "score": pair_score,
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    # IMPROVEMENT 1 + 2: Filter by stability AND both tests
    mask = (
        (results_df["is_cointegrated"])
        & (results_df["half_life"] >= config.HALF_LIFE_MIN)
        & (results_df["half_life"] <= config.HALF_LIFE_MAX)
        & (results_df["stability_pct"] >= config.MIN_STABILITY_PCT)
    )

    if config.REQUIRE_BOTH_COINT_TESTS:
        mask = mask & (results_df["eg_is_cointegrated"])

    selected = results_df[mask].head(config.MAX_PAIRS)
    return selected


# ============================================================
# Run one period with pair-level drawdown stops
# ============================================================

def run_period(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict:
    """Generate signals and returns with pair drawdown stops."""
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

        # Rolling calculations (IMPROVEMENT 3: velocity filter is now in generate_raw_signals)
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

        # Compute returns
        returns_df = compute_pair_returns(signals_df)

        # IMPROVEMENT 4: Pair-level drawdown stop
        # If cumulative loss exceeds threshold, zero out remaining signals
        cum_return = (1 + returns_df["net_return"].fillna(0)).cumprod()
        cum_drawdown = (cum_return - cum_return.cummax()) / cum_return.cummax()
        stop_date = cum_drawdown[cum_drawdown < -config.PAIR_DRAWDOWN_STOP].index

        if len(stop_date) > 0:
            first_stop = stop_date[0]
            # Zero out returns after stop
            returns_df.loc[first_stop:, "net_return"] = 0
            returns_df.loc[first_stop:, "gross_return"] = 0

        pair_key = f"{stock_a.replace('.NS','')}-{stock_b.replace('.NS','')}"
        all_returns[pair_key] = returns_df

        trades = extract_trade_log(signals_df, pair_key)
        all_trades.append(trades)

    return {
        "returns": all_returns,
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IMPROVED ROLLING WALK-FORWARD (v2)")
    print("=" * 60)
    print(f"\nImprovements:")
    print(f"  1. Min stability: {config.MIN_STABILITY_PCT}% (was 30%)")
    print(f"  2. Require BOTH ADF + Engle-Granger: {config.REQUIRE_BOTH_COINT_TESTS}")
    print(f"  3. Spread velocity filter: {config.SPREAD_VELOCITY_LOOKBACK}-day lookback")
    print(f"  4. Pair drawdown stop: {config.PAIR_DRAWDOWN_STOP*100:.0f}%")
    print(f"\n  Rebalance: every {REBALANCE_DAYS} days | Train: {TRAIN_WINDOW_DAYS} days")
    print(f"  Params: z_entry={Z_ENTRY}, z_exit={Z_EXIT}, lookback={LOOKBACK}\n")

    # Load data
    prices = load_data(force_refresh=False)
    all_dates = prices.index
    print(f"  Data: {all_dates[0].date()} → {all_dates[-1].date()} ({len(all_dates)} days)")

    # Rebalance points
    first_test_idx = TRAIN_WINDOW_DAYS
    rebalance_indices = list(range(first_test_idx, len(all_dates), REBALANCE_DAYS))
    print(f"  Rebalance points: {len(rebalance_indices)}")

    # Run
    all_period_returns = []
    all_period_trades = []
    period_summaries = []

    for i, rb_idx in enumerate(rebalance_indices):
        train_start_idx = max(0, rb_idx - TRAIN_WINDOW_DAYS)
        train_prices = prices.iloc[train_start_idx:rb_idx]

        test_end_idx = min(rb_idx + REBALANCE_DAYS, len(all_dates))
        period_start = all_dates[rb_idx]
        period_end = all_dates[test_end_idx - 1]

        print(f"\n{'─'*60}")
        print(f"  Period {i+1}/{len(rebalance_indices)}")
        print(f"  Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()}")
        print(f"  Test:  {period_start.date()} → {period_end.date()}")

        selected = select_pairs_on_window(train_prices)

        if len(selected) == 0:
            print(f"  ⚠️ No qualifying pairs. Skipping (flat period).")
            # Record a flat period
            period_summaries.append({
                "Period": f"{period_start.date()} → {period_end.date()}",
                "Return (%)": 0.0, "Sharpe": 0.0, "Max DD (%)": 0.0,
                "Vol (%)": 0.0, "Trades": 0, "Win Rate (%)": 0.0, "Pairs": 0,
            })
            continue

        print(f"  Selected {len(selected)} pairs:")
        for _, row in selected.iterrows():
            a = row['stock_a'].replace('.NS', '')
            b = row['stock_b'].replace('.NS', '')
            print(f"    {a} ↔ {b} (HL={row['half_life']:.0f}d, stab={row['stability_pct']:.0f}%)")

        result = run_period(prices, selected, period_start, period_end)

        if not result["returns"]:
            print(f"  ⚠️ No valid returns.")
            continue

        returns_matrix = pd.DataFrame({
            name: ret["net_return"] for name, ret in result["returns"].items()
        })
        period_returns = returns_matrix.mean(axis=1).fillna(0)

        all_period_returns.append(period_returns)
        all_period_trades.append(result["trades"])

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
    # Summary
    # ============================================================

    print(f"\n\n{'='*70}")
    print(f"IMPROVED WALK-FORWARD — RESULTS")
    print(f"{'='*70}\n")

    summary_df = pd.DataFrame(period_summaries)
    print("PER-PERIOD BREAKDOWN:\n")
    print(summary_df.to_string(index=False))

    if all_period_returns:
        combined_returns = pd.concat(all_period_returns).sort_index()
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]
        combined_trades = pd.concat(all_period_trades, ignore_index=True) if all_period_trades else pd.DataFrame()
        trade_pnls = combined_trades["trade_return"] if len(combined_trades) > 0 else None

        overall = compute_all_metrics(combined_returns.dropna(), trade_pnls)

        n_periods = len(summary_df)
        n_profitable = (summary_df["Return (%)"] > 0).sum()
        n_nonneg = (summary_df["Return (%)"] >= 0).sum()

        print(f"\n{'─'*70}")
        print(f"OVERALL:")
        print(f"  Total Return:   {overall['total_return']:+.2f}%")
        print(f"  CAGR:           {overall['cagr']:+.2f}%")
        print(f"  Sharpe Ratio:   {overall['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {overall['max_drawdown']:.2f}%")
        print(f"  Calmar Ratio:   {overall['calmar_ratio']:.2f}")
        print(f"  Volatility:     {overall['volatility']:.2f}%")
        print(f"  Total Trades:   {overall.get('n_trades', 0)}")
        print(f"  Win Rate:       {overall.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor:  {overall.get('profit_factor', 0):.2f}")
        print(f"\n  Profitable: {n_profitable}/{n_periods} | "
              f"Non-negative: {n_nonneg}/{n_periods}")

        print(f"\n{'─'*70}")
        if overall["sharpe_ratio"] > 0.5 and n_profitable >= n_periods * 0.6:
            print("✅ CONSISTENT EDGE detected across time periods!")
        elif overall["sharpe_ratio"] > 0 and n_nonneg >= n_periods * 0.6:
            print("🟡 WEAK EDGE — positive overall, but inconsistent across periods.")
        else:
            print("❌ NO reliable edge across time periods.")

    summary_df.to_csv("data/walk_forward_v2_results.csv", index=False)
    print(f"\n💾 Saved to data/walk_forward_v2_results.csv")
