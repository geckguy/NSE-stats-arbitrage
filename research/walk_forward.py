"""
============================================================
walk_forward.py — Walk-Forward Validation
============================================================
The gold standard for strategy confidence.

Instead of 1 train/test split, we do MULTIPLE:
  Window 1: Train 2021-2023 → Test 2023-2024
  Window 2: Train 2022-2024 → Test 2024-2025
  Window 3: Train 2023-2025 → Test 2025-2026

For each window:
  1. Select pairs on training data only
  2. Generate signals on test data only
  3. Backtest on test data
  4. Record metrics

If the strategy works across ALL windows → real edge.
If it only works in one window → overfitting / luck.

Usage:
    python walk_forward.py
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
from src.pair_selector import run_pair_selection, get_selected_pairs
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
# Define walk-forward windows
# ============================================================

# Each window: (train_start, train_end, test_start, test_end)
# Using ~2.5 year training, ~1 year testing, sliding by 1 year
WINDOWS = [
    {
        "name": "Window 1 (Test: 2023)",
        "train_start": "2021-01-01",
        "train_end": "2023-04-30",
        "test_start": "2023-05-01",
        "test_end": "2024-04-30",
    },
    {
        "name": "Window 2 (Test: 2024)",
        "train_start": "2022-01-01",
        "train_end": "2024-04-30",
        "test_start": "2024-05-01",
        "test_end": "2025-04-30",
    },
    {
        "name": "Window 3 (Test: 2025-26)",
        "train_start": "2023-01-01",
        "train_end": "2025-04-30",
        "test_start": "2025-05-01",
        "test_end": "2026-04-30",
    },
]


# ============================================================
# Run one window
# ============================================================

def run_window(
    prices: pd.DataFrame,
    window: dict,
    verbose: bool = True,
) -> dict:
    """
    Run pair selection + backtest for a single walk-forward window.
    """
    name = window["name"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Train: {window['train_start']} → {window['train_end']}")
        print(f"  Test:  {window['test_start']} → {window['test_end']}")
        print(f"{'='*60}")

    # --- Slice data ---
    train_prices = prices.loc[window["train_start"]:window["train_end"]]
    test_prices = prices.loc[window["test_start"]:window["test_end"]]

    if len(train_prices) < 200:
        print(f"  ⚠️ Not enough training data ({len(train_prices)} days). Skipping.")
        return None
    if len(test_prices) < 50:
        print(f"  ⚠️ Not enough test data ({len(test_prices)} days). Skipping.")
        return None

    if verbose:
        print(f"  Training days: {len(train_prices)}")
        print(f"  Test days: {len(test_prices)}")

    # --- Step 1: Pair selection on TRAINING data ---
    # Temporarily override config to use full training period
    original_test_days = config.TEST_PERIOD_DAYS
    config.TEST_PERIOD_DAYS = 0  # Use all data for training

    if verbose:
        print(f"\n  🔍 Selecting pairs on training data...")

    # Run pair selection — we need to pass training prices directly
    # Override: run_pair_selection uses train split internally,
    # but here we want it to use ALL of train_prices
    from src.pair_selector import (
        generate_candidate_pairs,
        compute_spread,
        test_cointegration,
        test_cointegration_engle_granger,
        compute_half_life,
        test_cointegration_stability,
        score_pair,
    )

    candidates = generate_candidate_pairs(train_prices)
    results = []

    for pair in candidates:
        stock_a, stock_b, sector = pair["stock_a"], pair["stock_b"], pair["sector"]
        try:
            pa = train_prices[stock_a]
            pb = train_prices[stock_b]
            ols = compute_spread(pa, pb)
            adf = test_cointegration(ols["spread"])
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
                "half_life": round(hl, 1),
                "stability_pct": round(stability_pct, 1),
                "score": pair_score,
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    # Select top pairs
    selected = results_df[
        (results_df["is_cointegrated"])
        & (results_df["half_life"] >= config.HALF_LIFE_MIN)
        & (results_df["half_life"] <= config.HALF_LIFE_MAX)
    ].head(config.MAX_PAIRS)

    config.TEST_PERIOD_DAYS = original_test_days

    if len(selected) == 0:
        print(f"  ⚠️ No cointegrated pairs found. Skipping.")
        return None

    if verbose:
        print(f"  Selected {len(selected)} pairs:")
        for _, row in selected.iterrows():
            a = row['stock_a'].replace('.NS', '')
            b = row['stock_b'].replace('.NS', '')
            print(f"    {a} ↔ {b} (score={row['score']:.3f}, HL={row['half_life']:.0f}d)")

    # --- Step 2: Generate signals on TEST data ---
    if verbose:
        print(f"\n  📡 Generating signals on test data...")

    # Need warmup before test period for rolling calculations
    warmup_start = pd.Timestamp(window["test_start"]) - pd.Timedelta(days=config.LOOKBACK_WINDOW * 3)
    signal_prices = prices.loc[warmup_start:window["test_end"]]
    test_start_ts = pd.Timestamp(window["test_start"])

    all_returns = {}
    all_trades_list = []

    for _, pair in selected.iterrows():
        stock_a = pair["stock_a"]
        stock_b = pair["stock_b"]
        half_life = pair["half_life"]

        try:
            price_a = signal_prices[stock_a]
            price_b = signal_prices[stock_b]
        except KeyError:
            continue

        # Rolling calculations
        beta = compute_rolling_beta_fast(price_a, price_b, config.LOOKBACK_WINDOW)
        spread = compute_rolling_spread(price_a, price_b, beta)
        zscore = compute_rolling_zscore(spread, config.LOOKBACK_WINDOW)
        raw_signal = generate_raw_signals(zscore, config.Z_ENTRY_THRESHOLD, config.Z_EXIT_THRESHOLD)
        signal = apply_risk_overlay(raw_signal, zscore, half_life, config.Z_STOP_LOSS)

        signals_df = pd.DataFrame({
            "price_a": price_a, "price_b": price_b,
            "beta": beta, "spread": spread,
            "zscore": zscore, "signal": signal,
        })

        # Trim to test period
        signals_df = signals_df.loc[test_start_ts:]

        if len(signals_df) < 50:
            continue

        # Compute returns
        returns_df = compute_pair_returns(signals_df)
        pair_key = f"{stock_a.replace('.NS','')}-{stock_b.replace('.NS','')}"
        all_returns[pair_key] = returns_df

        # Trade log
        trades = extract_trade_log(signals_df, pair_key)
        all_trades_list.append(trades)

    if not all_returns:
        print(f"  ⚠️ No valid returns. Skipping.")
        return None

    # --- Step 3: Portfolio returns ---
    returns_matrix = pd.DataFrame({
        name: ret["net_return"] for name, ret in all_returns.items()
    })
    portfolio_returns = returns_matrix.mean(axis=1).fillna(0)

    all_trades = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()
    trade_pnls = all_trades["trade_return"] if len(all_trades) > 0 else pd.Series(dtype=float)

    metrics = compute_all_metrics(portfolio_returns.dropna(), trade_pnls)
    metrics["n_pairs"] = len(all_returns)
    metrics["test_days"] = len(portfolio_returns)

    if verbose:
        print(f"\n  📊 Results:")
        print(f"     Return:   {metrics['total_return']:+.2f}%")
        print(f"     Sharpe:   {metrics['sharpe_ratio']:.2f}")
        print(f"     MaxDD:    {metrics['max_drawdown']:.2f}%")
        print(f"     Calmar:   {metrics['calmar_ratio']:.2f}")
        print(f"     Trades:   {metrics.get('n_trades', 0)}")
        print(f"     Win Rate: {metrics.get('win_rate', 0):.1f}%")

    return {
        "window": name,
        "metrics": metrics,
        "trades": all_trades,
        "portfolio_returns": portfolio_returns,
        "pairs": list(all_returns.keys()),
    }


# ============================================================
# Main: Run all windows
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"\nTests the strategy across {len(WINDOWS)} independent time periods.")
    print("If it works in ALL of them, the edge is real.\n")

    # Load full price data
    print("📂 Loading data...")
    prices = load_data(force_refresh=False)
    print(f"   Data range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Run each window
    window_results = []

    for window in WINDOWS:
        result = run_window(prices, window)
        if result:
            window_results.append(result)

    # ============================================================
    # Summary
    # ============================================================

    print(f"\n\n{'='*70}")
    print(f"WALK-FORWARD SUMMARY")
    print(f"{'='*70}\n")

    summary_rows = []
    for res in window_results:
        m = res["metrics"]
        summary_rows.append({
            "Window": res["window"],
            "Return (%)": m["total_return"],
            "Sharpe": m["sharpe_ratio"],
            "Max DD (%)": m["max_drawdown"],
            "Calmar": m["calmar_ratio"],
            "Volatility (%)": m["volatility"],
            "Trades": m.get("n_trades", 0),
            "Win Rate (%)": m.get("win_rate", 0),
            "Profit Factor": m.get("profit_factor", 0),
            "Pairs": m["n_pairs"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Averages
    print(f"\n{'─'*70}")
    print(f"AVERAGES ACROSS ALL WINDOWS:")
    avg_return = summary_df["Return (%)"].mean()
    avg_sharpe = summary_df["Sharpe"].mean()
    avg_dd = summary_df["Max DD (%)"].mean()
    avg_wr = summary_df["Win Rate (%)"].mean()

    print(f"  Avg Return:   {avg_return:+.2f}%")
    print(f"  Avg Sharpe:   {avg_sharpe:.2f}")
    print(f"  Avg Max DD:   {avg_dd:.2f}%")
    print(f"  Avg Win Rate: {avg_wr:.1f}%")

    # Consistency check
    positive_windows = sum(1 for _, row in summary_df.iterrows() if row["Return (%)"] > 0)
    print(f"\n  Profitable windows: {positive_windows}/{len(summary_df)}")

    if positive_windows == len(summary_df):
        print(f"  ✅ Strategy is profitable in ALL windows — strong evidence of a real edge!")
    elif positive_windows >= len(summary_df) * 0.66:
        print(f"  🟡 Strategy works in most windows — edge likely exists but is inconsistent.")
    else:
        print(f"  ❌ Strategy fails in too many windows — the edge may be illusory.")

    # Save
    summary_df.to_csv("data/walk_forward_results.csv", index=False)
    print(f"\n💾 Saved to data/walk_forward_results.csv")
