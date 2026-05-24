"""
tune_walk_forward.py — Fast Parameter Tuner for Rolling Walk-Forward
====================================================================
This script pre-computes and caches the selected pairs and rolling metrics 
for each period, then performs a rapid grid search over execution parameters 
(Z_ENTRY, Z_EXIT, Z_STOP, VELOCITY_LOOKBACK, PAIR_DD_STOP) to find the 
configuration that maximizes Sharpe ratio and returns while minimizing drawdown.
====================================================================
"""

import sys, os, warnings
import numpy as np, pandas as pd
from itertools import product

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

TRAIN_WINDOW_DAYS = 504
REBALANCE_DAYS = 126
WARMUP_DAYS = 150
MIN_STAB = 60.0
MAX_PAIRS = 5
LOOKBACK = 60

def select_pairs_cached(prices, dates):
    """Run pair selection for all periods and cache the signals_df raw data."""
    rb_indices = list(range(TRAIN_WINDOW_DAYS, len(dates), REBALANCE_DAYS))
    cached_periods = []
    
    print("⏳ Pre-selecting and caching pairs for all periods...")
    for i, ri in enumerate(rb_indices):
        tp = prices.iloc[max(0, ri - TRAIN_WINDOW_DAYS):ri]
        te = min(ri + REBALANCE_DAYS, len(dates))
        ps, pe = dates[ri], dates[te - 1]
        
        # Select pairs
        candidates = generate_candidate_pairs(tp)
        results = []
        for pair in candidates:
            sa, sb, sec = pair["stock_a"], pair["stock_b"], pair["sector"]
            try:
                pa, pb = tp[sa], tp[sb]
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
            print(f"  P{i+1}: No pairs found.")
            cached_periods.append((ps, pe, []))
            continue
            
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        mask = (df["is_cointegrated"] & df["eg_cointegrated"]
                & (df["half_life"] >= config.HALF_LIFE_MIN)
                & (df["half_life"] <= config.HALF_LIFE_MAX)
                & (df["stability_pct"] >= MIN_STAB))
        selected = df[mask].head(MAX_PAIRS)
        
        # Compute and cache rolling OLS and zscore data for selected pairs
        wstart = ps - pd.Timedelta(days=WARMUP_DAYS)
        sp = prices.loc[wstart:pe]
        
        pair_data_list = []
        for _, pair in selected.iterrows():
            sa, sb, hl = pair["stock_a"], pair["stock_b"], pair["half_life"]
            try:
                pa, pb = sp[sa], sp[sb]
            except KeyError:
                continue
            beta = compute_rolling_beta_fast(pa, pb, LOOKBACK)
            spread = compute_rolling_spread(pa, pb, beta)
            zs = compute_rolling_zscore(spread, LOOKBACK)
            
            pair_data_list.append({
                "sa": sa, "sb": sb, "hl": hl,
                "price_a": pa, "price_b": pb, "beta": beta,
                "spread": spread, "zscore": zs,
            })
            
        print(f"  P{i+1} ({ps.date()} → {pe.date()}): Cached {len(pair_data_list)} pairs.")
        cached_periods.append((ps, pe, pair_data_list))
        
    return cached_periods

def generate_signals_fast(zscore, entry_thresh, exit_thresh, vel_lb):
    """Fast signal generator."""
    position = 0
    signals = np.zeros(len(zscore))
    # Convert series to numpy for speed
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

def evaluate_params(cached_periods, z_entry, z_exit, z_stop, vel_lb, pair_dd_stop):
    """Evaluate a specific set of parameters on the cached periods."""
    all_period_returns = []
    all_period_trades = []
    
    for ps, pe, pair_data_list in cached_periods:
        if not pair_data_list:
            continue
            
        all_ret = {}
        all_tr = []
        for pdata in pair_data_list:
            sa, sb, hl = pdata["sa"], pdata["sb"], pdata["hl"]
            raw = generate_signals_fast(pdata["zscore"], z_entry, z_exit, vel_lb)
            sig = apply_risk_overlay(raw, pdata["zscore"], hl, z_stop)
            
            sdf = pd.DataFrame({
                "price_a": pdata["price_a"], "price_b": pdata["price_b"],
                "beta": pdata["beta"], "spread": pdata["spread"],
                "zscore": pdata["zscore"], "signal": sig
            })
            sdf = sdf.loc[ps:pe]
            if len(sdf) < 10:
                continue
                
            rdf = compute_pair_returns(sdf)
            
            # Pair drawdown stop
            if pair_dd_stop < 1.0:
                cum = (1 + rdf["net_return"].fillna(0)).cumprod()
                dd = (cum - cum.cummax()) / cum.cummax()
                stops = dd[dd < -pair_dd_stop].index
                if len(stops) > 0:
                    rdf.loc[stops[0]:, ["net_return", "gross_return"]] = 0
                    
            pk = f"{sa.replace('.NS','')}-{sb.replace('.NS','')}"
            all_ret[pk] = rdf
            all_tr.append(extract_trade_log(sdf, pk))
            
        if not all_ret:
            continue
            
        rm = pd.DataFrame({n: r["net_return"] for n, r in all_ret.items()})
        pr = rm.mean(axis=1).fillna(0)
        all_period_returns.append(pr)
        all_period_trades.append(pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame())
        
    if not all_period_returns:
        return None
        
    cr = pd.concat(all_period_returns).sort_index()
    cr = cr[~cr.index.duplicated(keep='first')]
    ct = pd.concat(all_period_trades, ignore_index=True) if all_period_trades else pd.DataFrame()
    tp = ct["trade_return"] if len(ct) > 0 and "trade_return" in ct.columns else None
    
    metrics = compute_all_metrics(cr.dropna(), tp)
    return metrics

if __name__ == "__main__":
    print("=" * 60)
    print("WALK-FORWARD PARAMETER SWEEP")
    print("=" * 60)
    
    # Load prices
    prices = load_data(force_refresh=False)
    dates = prices.index
    
    # Pre-select and cache pairs
    cached_periods = select_pairs_cached(prices, dates)
    
    # Search grid
    z_entries = [1.2, 1.5, 1.8, 2.0, 2.2]
    z_exits = [0.2, 0.3, 0.5, 0.7]
    z_stops = [3.0, 3.5, 4.0, 5.0]
    vel_lbs = [0, 2, 3, 5]
    pair_dd_stops = [0.05, 0.08, 0.12, 999.0] # 999.0 means no stop
    
    total_combinations = len(z_entries) * len(z_exits) * len(z_stops) * len(vel_lbs) * len(pair_dd_stops)
    print(f"\n🚀 Starting grid search over {total_combinations} combinations...")
    
    results = []
    count = 0
    
    for z_entry, z_exit, z_stop, vel_lb, dd_stop in product(z_entries, z_exits, z_stops, vel_lbs, pair_dd_stops):
        count += 1
        if count % 100 == 0:
            print(f"  Progress: {count}/{total_combinations}...")
            
        # Quick skip logic: z_exit should be lower than z_entry
        if z_exit >= z_entry:
            continue
            
        m = evaluate_params(cached_periods, z_entry, z_exit, z_stop, vel_lb, dd_stop)
        if m is None:
            continue
            
        results.append({
            "z_entry": z_entry,
            "z_exit": z_exit,
            "z_stop": z_stop,
            "vel_lb": vel_lb,
            "dd_stop": dd_stop,
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "sharpe": m["sharpe_ratio"],
            "max_dd": m["max_drawdown"],
            "trades": m.get("n_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "profit_factor": m.get("profit_factor", 0),
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("data/tuning_walk_forward_results.csv", index=False)
    print("\n💾 Tuning results saved to data/tuning_walk_forward_results.csv")
    
    # Print top 15 results sorted by Return
    print("\n🏆 TOP 15 BY TOTAL RETURN:")
    print(df_results.sort_values("total_return", ascending=False).head(15).to_string(index=False))
    
    # Print top 15 results sorted by Sharpe Ratio
    print("\n🏆 TOP 15 BY SHARPE RATIO:")
    print(df_results.sort_values("sharpe", ascending=False).head(15).to_string(index=False))
