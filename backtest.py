"""
backtest.py — Rolling Walk-Forward Backtest (Tuned Parameters)
======================================================================
This script runs the final, fully optimized rolling walk-forward strategy:
  1. Rolling Train/Test: 504 days training (~2 years), 126 days trading (~6 months)
  2. Concentrated selection: Top 5 pairs per period (dual-coint + 60% stability)
  3. Signal parameters: Z_ENTRY = 2.0, Z_EXIT = 0.7, Z_STOP = 3.5
  4. Spread velocity: 2-day lookback confirmation for entries
  5. Capital protection: 5% pair-level drawdown stop

It downloads and compares performance against the Nifty 50 (^NSEI) index.
Saves:
  - data/walk_forward_final_results.csv (period breakdown)
  - data/walk_forward_final_equity.csv (cumulative returns compared to Nifty 50)
  - data/walk_forward_final_trades.csv (all trades across all periods)
======================================================================
"""

import sys, os, warnings
import numpy as np, pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import load_data
from src.pair_selector import (
    generate_candidate_pairs, compute_spread, test_cointegration,
    test_cointegration_engle_granger, compute_half_life,
    test_cointegration_stability, score_pair,
)
from src.signal_generator import (
    compute_rolling_beta_fast, compute_rolling_spread,
    compute_rolling_zscore, apply_risk_overlay,
)
from src.backtester import compute_pair_returns, extract_trade_log
from src.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

# ── Optimized Parameters ──
TRAIN_WINDOW_DAYS = 504
REBALANCE_DAYS = 126
WARMUP_DAYS = 270
Z_ENTRY = 2.2
Z_EXIT = 0.5
Z_STOP = 3.5
VELOCITY_LOOKBACK = 2
PAIR_DD_STOP = 0.03
MIN_STAB = 60.0
MAX_PAIRS = 5
BETA_LOOKBACK = 120
Z_LOOKBACK = 30

def generate_signals_with_velocity(zscore, entry_thresh, exit_thresh, vel_lb):
    """Entry with velocity confirmation."""
    position = 0
    signals = np.zeros(len(zscore))
    zs_val = zscore.values
    for i in range(len(zs_val)):
        z = zs_val[i]
        if np.isnan(z):
            signals[i] = 0
            position = 0
            continue
        if position == 0:
            if z < -entry_thresh:
                if vel_lb > 0 and i >= vel_lb:
                    if zs_val[i] > zs_val[i - vel_lb]:
                        position = 1
                else:
                    position = 1
            elif z > entry_thresh:
                if vel_lb > 0 and i >= vel_lb:
                    if zs_val[i] < zs_val[i - vel_lb]:
                        position = -1
                else:
                    position = -1
        else:
            if abs(z) < exit_thresh:
                position = 0
        signals[i] = position
    return pd.Series(signals, index=zscore.index, name="signal")

def select_pairs(prices):
    candidates = generate_candidate_pairs(prices)
    results = []
    for pair in candidates:
        sa, sb, sec = pair["stock_a"], pair["stock_b"], pair["sector"]
        try:
            pa, pb = prices[sa], prices[sb]
            ols = compute_spread(pa, pb)
            adf = test_cointegration(ols["spread"])
            eg = test_cointegration_engle_granger(pa, pb)
            hl = compute_half_life(ols["spread"])
            stab = test_cointegration_stability(pa, pb)["stability_pct"] if adf["is_cointegrated"] else 0.0
            results.append({
                "stock_a": sa, "stock_b": sb, "sector": sec,
                "beta": ols["beta"], "adf_pvalue": adf["adf_pvalue"],
                "is_cointegrated": adf["is_cointegrated"],
                "eg_cointegrated": eg["eg_is_cointegrated"],
                "half_life": hl, "stability_pct": stab,
                "score": score_pair(adf["adf_pvalue"], hl, stab),
            })
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    mask = (df["is_cointegrated"] & df["eg_cointegrated"]
            & (df["half_life"] >= config.HALF_LIFE_MIN)
            & (df["half_life"] <= config.HALF_LIFE_MAX)
            & (df["stability_pct"] >= MIN_STAB))
    return df[mask].head(MAX_PAIRS)

def run_period(prices, selected, pstart, pend):
    wstart = pstart - pd.Timedelta(days=WARMUP_DAYS)
    sp = prices.loc[wstart:pend]
    all_ret, all_tr = {}, []
    for _, pair in selected.iterrows():
        sa, sb, hl = pair["stock_a"], pair["stock_b"], pair["half_life"]
        try:
            pa, pb = sp[sa], sp[sb]
        except KeyError:
            continue
        beta = compute_rolling_beta_fast(pa, pb, BETA_LOOKBACK)
        spread = compute_rolling_spread(pa, pb, beta)
        zs = compute_rolling_zscore(spread, Z_LOOKBACK)
        raw = generate_signals_with_velocity(zs, Z_ENTRY, Z_EXIT, VELOCITY_LOOKBACK)
        sig = apply_risk_overlay(raw, zs, hl, Z_STOP)
        sdf = pd.DataFrame({"price_a": pa, "price_b": pb, "beta": beta,
                            "spread": spread, "zscore": zs, "signal": sig})
        sdf = sdf.loc[pstart:pend]
        if len(sdf) < 10:
            continue
        rdf = compute_pair_returns(sdf)
        
        # Pair drawdown stop (5%)
        cum = (1 + rdf["net_return"].fillna(0)).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        stops = dd[dd < -PAIR_DD_STOP].index
        if len(stops) > 0:
            rdf.loc[stops[0]:, ["net_return", "gross_return"]] = 0
            
        pk = f"{sa.replace('.NS','')}-{sb.replace('.NS','')}"
        all_ret[pk] = rdf
        all_tr.append(extract_trade_log(sdf, pk))
        
    return {"returns": all_ret,
            "trades": pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()}

if __name__ == "__main__":
    print("=" * 60)
    print("WALK-FORWARD FINAL RUN (TUNED PARAMETERS)")
    print("=" * 60)
    
    # Load prices
    prices = load_data(force_refresh=False)
    dates = prices.index
    
    rb_indices = list(range(TRAIN_WINDOW_DAYS, len(dates), REBALANCE_DAYS))
    print(f"  Total periods: {len(rb_indices)}\n")
    
    all_rets, all_trs, summaries = [], [], []
    
    for i, ri in enumerate(rb_indices):
        tp = prices.iloc[max(0, ri - TRAIN_WINDOW_DAYS):ri]
        te = min(ri + REBALANCE_DAYS, len(dates))
        ps, pe = dates[ri], dates[te - 1]
        print(f"  P{i+1}: {ps.date()} → {pe.date()}", end=" | ")
        
        sel = select_pairs(tp)
        if len(sel) == 0:
            print("No qualifying pairs.")
            summaries.append({
                "Period": f"{ps.date()}→{pe.date()}", "Return (%)": 0.0,
                "Sharpe": 0.0, "Max DD (%)": 0.0, "Trades": 0,
                "Win Rate (%)": 0.0, "Pairs": 0
            })
            continue
            
        res = run_period(prices, sel, ps, pe)
        if not res["returns"]:
            print("No returns generated.")
            continue
            
        lev = getattr(config, "LEVERAGE", 1.0)
        rm = pd.DataFrame({n: r["net_return"] for n, r in res["returns"].items()})
        pr = rm.mean(axis=1).fillna(0) * lev
        all_rets.append(pr)
        
        trades = res["trades"].copy()
        if len(trades) > 0 and "trade_return" in trades.columns:
            trades["trade_return"] = trades["trade_return"] * lev
        all_trs.append(trades)
        
        tp_col = "trade_return" if len(trades) > 0 and "trade_return" in trades.columns else None
        m = compute_all_metrics(pr.dropna(), trades[tp_col] if tp_col else None)
        
        summaries.append({
            "Period": f"{ps.date()}→{pe.date()}", "Return (%)": m["total_return"],
            "Sharpe": m["sharpe_ratio"], "Max DD (%)": m["max_drawdown"],
            "Trades": m.get("n_trades", 0), "Win Rate (%)": m.get("win_rate", 0),
            "Pairs": len(res["returns"])
        })
        print(f"{len(sel)} pairs | Ret={m['total_return']:+.2f}% | Sharpe={m['sharpe_ratio']:.2f} | MaxDD={m['max_drawdown']:.2f}%")
        
    print(f"\n{'='*70}")
    print("PER-PERIOD SUMMARY:")
    print(f"{'='*70}")
    sdf = pd.DataFrame(summaries)
    print(sdf.to_string(index=False))
    
    # Combined calculations
    if all_rets:
        combined_returns = pd.concat(all_rets).sort_index()
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]
        
        combined_trades = pd.concat(all_trs, ignore_index=True) if all_trs else pd.DataFrame()
        tp = combined_trades["trade_return"] if len(combined_trades) > 0 and "trade_return" in combined_trades.columns else None
        
        overall = compute_all_metrics(combined_returns.dropna(), tp)
        
        # Download Nifty 50 benchmark for comparison
        print(f"\n{'─'*70}")
        print("Downloading Nifty 50 benchmark data...")
        test_start = combined_returns.index[0]
        test_end = combined_returns.index[-1]
        
        try:
            nifty = yf.download(
                "^NSEI",
                start=test_start - pd.Timedelta(days=10),
                end=test_end + pd.Timedelta(days=1),
                auto_adjust=True,
                progress=False,
            )
            nifty_close = nifty["Close"].squeeze()
            nifty_returns = nifty_close.pct_change().dropna()
            
            # Align with portfolio returns
            common_idx = nifty_returns.index.intersection(combined_returns.index)
            nifty_returns = nifty_returns.loc[common_idx]
            nifty_metrics = compute_all_metrics(nifty_returns)
            
            benchmark_available = True
        except Exception as e:
            print(f"⚠️ Could not download Nifty 50: {e}. Using flat benchmark.")
            nifty_returns = pd.Series(0.0, index=combined_returns.index)
            nifty_metrics = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "volatility": 0.0}
            benchmark_available = False
            
        print(f"\n{'='*70}")
        print("OVERALL PERFORMANCE SUMMARY:")
        print(f"{'='*70}")
        print(f"Metric                 Portfolio        Nifty 50 (Benchmark)")
        print(f"─" * 70)
        print(f"Total Return           {overall['total_return']:>+6.2f}%          {nifty_metrics['total_return']:>+6.2f}%")
        print(f"CAGR                   {overall['cagr']:>6.2f}%          {nifty_metrics['cagr']:>6.2f}%")
        print(f"Sharpe Ratio           {overall['sharpe_ratio']:>6.2f}          {nifty_metrics['sharpe_ratio']:>6.2f}")
        print(f"Max Drawdown           {overall['max_drawdown']:>6.2f}%          {nifty_metrics['max_drawdown']:>6.2f}%")
        print(f"Calmar Ratio           {overall['calmar_ratio']:>6.2f}          {nifty_metrics['calmar_ratio']:>6.2f}")
        print(f"Volatility (Annual)    {overall['volatility']:>6.2f}%          {nifty_metrics['volatility']:>6.2f}%")
        print(f"Total Trades           {overall.get('n_trades', 0):>6d}              -")
        print(f"Win Rate               {overall.get('win_rate', 0.0):>6.1f}%              -")
        print(f"Profit Factor          {overall.get('profit_factor', 0.0):>6.2f}              -")
        
        n_periods = len(sdf)
        n_profitable = (sdf["Return (%)"] > 0).sum()
        print(f"Profitable Periods     {n_profitable}/{n_periods} ({n_profitable/n_periods*100:.0f}%)        -")
        
        # Save files
        sdf.to_csv("data/walk_forward_final_results.csv", index=False)
        combined_trades.to_csv("data/walk_forward_final_trades.csv", index=False)
        
        # Save equity curves
        equity_df = pd.DataFrame({
            "portfolio": (1 + combined_returns).cumprod() - 1,
            "benchmark": (1 + nifty_returns).cumprod() - 1 if benchmark_available else pd.Series(0.0, index=combined_returns.index)
        })
        equity_df.to_csv("data/walk_forward_final_equity.csv")
        
        print(f"\n💾 Saved Walk-Forward results:")
        print("  - data/walk_forward_final_results.csv")
        print("  - data/walk_forward_final_trades.csv")
        print("  - data/walk_forward_final_equity.csv")
        
    print(f"\n{'='*70}")
