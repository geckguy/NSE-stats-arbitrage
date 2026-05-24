"""
walk_forward_v3.py — Iteration 2: Less aggressive filters
============================================================
Changes from v2:
  - Velocity lookback 5→3 (less strict — allow entries sooner)
  - Pair drawdown stop 5%→8% (give pairs more room to breathe)
  - Training window 504→378 (~1.5 years, more adaptive to regime shifts)
  - Rebalance every 63 days (~3 months instead of 6)
  - Try z_exit=0.25 (let winners run, our tuning showed this helps)
  - Stability requirement stays at 60%
============================================================
"""

import sys, os, warnings
import numpy as np, pandas as pd

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
    compute_rolling_zscore, generate_raw_signals, apply_risk_overlay,
)
from src.backtester import compute_pair_returns, extract_trade_log
from src.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

# ── Tweaked parameters ──
TRAIN_WINDOW_DAYS = 378    # ~1.5 years (was 504)
REBALANCE_DAYS = 63        # ~3 months (was 126)
WARMUP_DAYS = 150
Z_ENTRY = 2.0
Z_EXIT = 0.25              # Let winners run (was 0.5)
LOOKBACK = 60
Z_STOP = 4.0
VELOCITY_LOOKBACK = 3      # Less strict (was 5)
PAIR_DD_STOP = 0.08        # More room (was 0.05)
MIN_STAB = 60.0
MAX_PAIRS = 8

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
        beta = compute_rolling_beta_fast(pa, pb, LOOKBACK)
        spread = compute_rolling_spread(pa, pb, beta)
        zs = compute_rolling_zscore(spread, LOOKBACK)

        # Override velocity lookback for this run
        orig_vel = getattr(config, 'SPREAD_VELOCITY_LOOKBACK', 0)
        config.SPREAD_VELOCITY_LOOKBACK = VELOCITY_LOOKBACK
        raw = generate_raw_signals(zs, Z_ENTRY, Z_EXIT)
        config.SPREAD_VELOCITY_LOOKBACK = orig_vel

        sig = apply_risk_overlay(raw, zs, hl, Z_STOP)
        sdf = pd.DataFrame({"price_a": pa, "price_b": pb, "beta": beta,
                            "spread": spread, "zscore": zs, "signal": sig})
        sdf = sdf.loc[pstart:pend]
        if len(sdf) < 10:
            continue
        rdf = compute_pair_returns(sdf)
        # Pair drawdown stop
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
    print("WALK-FORWARD v3 — Less Aggressive Filters")
    print("=" * 60)
    print(f"  Train: {TRAIN_WINDOW_DAYS}d | Rebalance: {REBALANCE_DAYS}d")
    print(f"  z_entry={Z_ENTRY}, z_exit={Z_EXIT}, velocity={VELOCITY_LOOKBACK}d")
    print(f"  Pair DD stop: {PAIR_DD_STOP*100:.0f}% | Min stability: {MIN_STAB}%\n")

    prices = load_data(force_refresh=False)
    dates = prices.index
    print(f"  Data: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")

    rb_indices = list(range(TRAIN_WINDOW_DAYS, len(dates), REBALANCE_DAYS))
    print(f"  Periods: {len(rb_indices)}\n")

    all_rets, all_trs, summaries = [], [], []

    for i, ri in enumerate(rb_indices):
        tp = prices.iloc[max(0, ri - TRAIN_WINDOW_DAYS):ri]
        te = min(ri + REBALANCE_DAYS, len(dates))
        ps, pe = dates[ri], dates[te - 1]
        print(f"  P{i+1}: {ps.date()}→{pe.date()}", end=" | ")

        sel = select_pairs(tp)
        if len(sel) == 0:
            print("No pairs")
            summaries.append({"Period": f"{ps.date()}→{pe.date()}", "Return (%)": 0,
                              "Sharpe": 0, "Max DD (%)": 0, "Trades": 0,
                              "Win Rate (%)": 0, "Pairs": 0})
            continue

        res = run_period(prices, sel, ps, pe)
        if not res["returns"]:
            print("No returns")
            continue

        rm = pd.DataFrame({n: r["net_return"] for n, r in res["returns"].items()})
        pr = rm.mean(axis=1).fillna(0)
        all_rets.append(pr)
        all_trs.append(res["trades"])
        tp_col = "trade_return" if len(res["trades"]) > 0 and "trade_return" in res["trades"].columns else None
        m = compute_all_metrics(pr.dropna(), res["trades"][tp_col] if tp_col else None)

        summaries.append({"Period": f"{ps.date()}→{pe.date()}", "Return (%)": m["total_return"],
                          "Sharpe": m["sharpe_ratio"], "Max DD (%)": m["max_drawdown"],
                          "Trades": m.get("n_trades", 0), "Win Rate (%)": m.get("win_rate", 0),
                          "Pairs": len(res["returns"])})
        print(f"{len(sel)}p | Ret={m['total_return']:+.1f}% | Sh={m['sharpe_ratio']:.2f} | DD={m['max_drawdown']:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    sdf = pd.DataFrame(summaries)
    print(sdf.to_string(index=False))

    if all_rets:
        cr = pd.concat(all_rets).sort_index()
        cr = cr[~cr.index.duplicated(keep='first')]
        ct = pd.concat(all_trs, ignore_index=True) if all_trs else pd.DataFrame()
        tp = ct["trade_return"] if len(ct) > 0 and "trade_return" in ct.columns else None
        ov = compute_all_metrics(cr.dropna(), tp)
        np_ = len(sdf)
        nprof = (sdf["Return (%)"] > 0).sum()
        print(f"\nOVERALL: Return={ov['total_return']:+.2f}% | CAGR={ov['cagr']:+.2f}% | "
              f"Sharpe={ov['sharpe_ratio']:.2f} | MaxDD={ov['max_drawdown']:.2f}% | "
              f"Calmar={ov['calmar_ratio']:.2f}")
        print(f"Trades={ov.get('n_trades',0)} | WinRate={ov.get('win_rate',0):.1f}% | "
              f"PF={ov.get('profit_factor',0):.2f}")
        print(f"Profitable: {nprof}/{np_} ({nprof/np_*100:.0f}%)")

    sdf.to_csv("data/walk_forward_v3_results.csv", index=False)
    print(f"\n💾 Saved to data/walk_forward_v3_results.csv")
