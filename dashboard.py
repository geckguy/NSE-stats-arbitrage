"""
dashboard.py — Interactive Streamlit Dashboard for NSE Statistical Arbitrage
======================================================================
This dashboard visualizes:
  1. Overall Walk-Forward Portfolio Performance vs Nifty 50 Benchmark.
  2. Per-period breakdowns.
  3. Interactive Pair Signal Explorer: reconstructs rolling beta, spread, 
     and z-score signals on the fly for any traded pair in any period.
  4. Trade Log Analyzer: filters and charts individual trade outcomes.
  5. Parameter Tuning Sweep: visualizes the 1,280 grid-search combinations.
  6. Cointegration Learning Center: interactive reference guide.
======================================================================
"""

import sys, os, warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import load_data
from src.pair_selector import score_pair, compute_spread
from src.signal_generator import (
    compute_rolling_beta_fast, compute_rolling_spread,
    compute_rolling_zscore
)
from src.backtester import compute_pair_returns

# Page config
st.set_page_config(
    page_title="NSE Statistical Arbitrage Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Optimized Strategy Parameters ──
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

# Custom premium CSS styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background-color: #171a21;
        border: 1px solid #2b303c;
        border-radius: 8px;
        padding: 15px 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #00e676;
        margin-top: 5px;
    }
    .metric-value-neg {
        font-size: 26px;
        font-weight: bold;
        color: #ff1744;
        margin-top: 5px;
    }
    .metric-value-neutral {
        font-size: 26px;
        font-weight: bold;
        color: #29b6f6;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #8a94a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #29b6f6 0%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #00e676;
        padding-left: 10px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache price data loading
@st.cache_data(show_spinner=True)
def load_prices_cached():
    return load_data(force_refresh=False)

# Re-implement signals logic for interactive viewer
def generate_signals_with_velocity(zscore, entry_thresh, exit_thresh, vel_lb):
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

def apply_risk_overlay(raw_signals, zscore, half_life, z_stop):
    signals = raw_signals.copy()
    in_trade = False
    entry_idx = 0
    max_hold_days = int(2 * half_life)
    
    for i in range(len(signals)):
        sig = signals.iloc[i]
        # Check Stop Loss
        if abs(zscore.iloc[i]) >= z_stop:
            signals.iloc[i:] = 0
            in_trade = False
            continue
            
        if not in_trade and sig != 0:
            in_trade = True
            entry_idx = i
        elif in_trade:
            if sig == 0:
                in_trade = False
            elif (i - entry_idx) >= max_hold_days:
                signals.iloc[i:] = 0
                in_trade = False
    return signals
def apply_leverage_to_equity(df_equity, leverage):
    if leverage == 1.0:
        return df_equity.copy()
    df = df_equity.copy()
    cum = df["portfolio"] + 1
    daily_returns = cum.pct_change().fillna(0)
    leveraged_daily = daily_returns * leverage
    df["portfolio"] = (1 + leveraged_daily).cumprod() - 1
    return df

def get_leveraged_period_summary(df_results, df_equity_lev, df_trades_lev):
    summaries = []
    for _, row in df_results.iterrows():
        period_str = row["Period"]
        start_str, end_str = period_str.split("→")
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        
        # Slice equity
        peq = df_equity_lev.loc[start_date:end_date]
        if len(peq) < 2:
            summaries.append(row.to_dict())
            continue
        
        # Calculate returns
        total_ret = (peq["portfolio"].iloc[-1] + 1) / (peq["portfolio"].iloc[0] + 1) - 1
        
        # Calculate daily returns for Sharpe
        cum = peq["portfolio"] + 1
        daily_ret = cum.pct_change().fillna(0)
        daily_rf = 0.06 / 252
        excess = daily_ret - daily_rf
        sharpe = np.sqrt(252) * excess.mean() / (excess.std() if excess.std() > 0 else 1)
        
        # Calculate Max DD
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        # Slice trades
        df_trades_lev_copy = df_trades_lev.copy()
        df_trades_lev_copy["entry_date"] = pd.to_datetime(df_trades_lev_copy["entry_date"])
        ptrades = df_trades_lev_copy[(df_trades_lev_copy["entry_date"] >= start_date) & (df_trades_lev_copy["entry_date"] <= end_date)]
        n_trades = len(ptrades)
        win_rate = (ptrades["trade_return"] > 0).mean() * 100 if n_trades > 0 else 0.0
        
        summaries.append({
            "Period": period_str,
            "Return (%)": round(total_ret * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max DD (%)": round(max_dd * 100, 2),
            "Trades": n_trades,
            "Win Rate (%)": round(win_rate, 1),
            "Pairs": row["Pairs"]
        })
    return pd.DataFrame(summaries)

# Sidebar configuration
st.sidebar.markdown("<h2 style='color: #29b6f6;'>📊 Strategy Config</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Optimal parameters identified during walk-forward grid search validation.")

st.sidebar.markdown(f"**TRAIN_WINDOW_DAYS**: {504} days")
st.sidebar.markdown(f"**REBALANCE_DAYS**: {126} days")
st.sidebar.markdown(f"**MIN_STABILITY_PCT**: {60.0}%")
st.sidebar.markdown(f"**MAX_PAIRS_PER_PERIOD**: {5}")

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Entry/Exit Parameters")
st.sidebar.info(
    f"🟢 **Z_ENTRY**: {Z_ENTRY}\n\n"
    f"🔴 **Z_EXIT**: {Z_EXIT}\n\n"
    f"⚠️ **Z_STOP**: {Z_STOP}\n\n"
    f"⚡ **BETA_LOOKBACK**: {BETA_LOOKBACK} days\n\n"
    f"⚡ **Z_LOOKBACK**: {Z_LOOKBACK} days\n\n"
    f"🛡️ **PAIR_DD_STOP**: {PAIR_DD_STOP * 100:.1f}%"
)

# Load data files
wf_results_path = "data/walk_forward_final_results.csv"
wf_trades_path = "data/walk_forward_final_trades.csv"
wf_equity_path = "data/walk_forward_final_equity.csv"
tuning_path = "data/tuning_walk_forward_results.csv"

# Title banner
st.markdown("<div class='main-title'>NSE Cointegrated Pairs Trading Engine</div>", unsafe_allow_html=True)
st.markdown("A premium market-neutral statistical arbitrage framework validated through rolling walk-forward testing.")

# Check for required files
if not (os.path.exists(wf_results_path) and os.path.exists(wf_trades_path) and os.path.exists(wf_equity_path)):
    st.error("❌ Missing final walk-forward results files. Please run `python backtest.py` to generate the data.")
    st.stop()

# Load datasets
df_results = pd.read_csv(wf_results_path)
df_trades = pd.read_csv(wf_trades_path)
df_equity = pd.read_csv(wf_equity_path, parse_dates=["Date"]).set_index("Date")

# Leverage Slider in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Leverage Settings")
leverage = st.sidebar.slider("Leverage Factor", 1.0, 5.0, value=1.0, step=0.5, help="Amplify returns and drawdowns using portfolio leverage.")

# Apply leverage dynamically
df_equity_unleveraged = df_equity.copy()
df_equity = apply_leverage_to_equity(df_equity, leverage)
df_trades["trade_return"] = df_trades["trade_return"] * leverage

# Compute dynamic period summaries
df_results = get_leveraged_period_summary(df_results, df_equity, df_trades)

# Compute overall metrics
overall_ret = df_equity["portfolio"].iloc[-1]
overall_bench_ret = df_equity["benchmark"].iloc[-1]

# Set up tabs
tabs = st.tabs(["🏆 Portfolio Performance", "🔍 Pair Signal Explorer", "📋 Trade Logs", "⚡ Parameter Sweep Explorer", "📄 Reference"])

# ======================================================================
# TAB 1: Portfolio Performance
# ======================================================================
with tabs[0]:
    st.markdown("<div class='section-header'>Overall Portfolio vs. Benchmark Metrics</div>", unsafe_allow_html=True)
    
    # Portfolio Metrics Cards
    # Calculate Sharpe, MaxDD and CAGR directly from daily equity curves
    returns_series = df_equity["portfolio"].diff().fillna(0)
    bench_returns = df_equity["benchmark"].diff().fillna(0)
    
    trading_days_per_year = 252
    years = len(df_equity) / trading_days_per_year
    
    portfolio_cagr = (df_equity["portfolio"].iloc[-1] + 1) ** (1 / years) - 1
    bench_cagr = (df_equity["benchmark"].iloc[-1] + 1) ** (1 / years) - 1
    
    # Daily RF rate (annual 6%)
    daily_rf = 0.06 / trading_days_per_year
    portfolio_excess = returns_series - daily_rf
    portfolio_sharpe = np.sqrt(trading_days_per_year) * portfolio_excess.mean() / (portfolio_excess.std() if portfolio_excess.std() > 0 else 1)
    
    bench_excess = bench_returns - daily_rf
    bench_sharpe = np.sqrt(trading_days_per_year) * bench_excess.mean() / (bench_excess.std() if bench_excess.std() > 0 else 1)
    
    # Drawdown calculations
    portfolio_cum = df_equity["portfolio"] + 1
    portfolio_peak = portfolio_cum.cummax()
    portfolio_dd = (portfolio_cum - portfolio_peak) / portfolio_peak
    portfolio_max_dd = portfolio_dd.min()
    
    bench_cum = df_equity["benchmark"] + 1
    bench_peak = bench_cum.cummax()
    bench_dd = (bench_cum - bench_peak) / bench_peak
    bench_max_dd = bench_dd.min()
    
    # Col layout for metric cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Portfolio Return</div>
            <div class='metric-value'>+{overall_ret * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Benchmark Return</div>
            <div class='metric-value-neutral'>+{overall_bench_ret * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Portfolio Sharpe</div>
            <div class='metric-value'>{portfolio_sharpe:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Portfolio MaxDD</div>
            <div class='metric-value'>-{abs(portfolio_max_dd * 100):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Benchmark MaxDD</div>
            <div class='metric-value-neg'>-{abs(bench_max_dd * 100):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        n_periods = len(df_results)
        n_profitable = (df_results["Return (%)"] > 0).sum() if n_periods > 0 else 0
        pct_profitable = (n_profitable / n_periods * 100) if n_periods > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Profitable Periods</div>
            <div class='metric-value'>{n_profitable}/{n_periods} ({pct_profitable:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Equity Curve Charts
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=df_equity.index, y=df_equity["portfolio"] * 100,
        mode="lines", name="Pairs Portfolio Strategy",
        line=dict(color="#00e676", width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=df_equity.index, y=df_equity["benchmark"] * 100,
        mode="lines", name="Nifty 50 Index (Benchmark)",
        line=dict(color="#29b6f6", width=1.5, dash="dash")
    ))
    fig_eq.update_layout(
        title="Walk-Forward Cumulative Equity Curves (Jan 2023 – Apr 2026)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    
    # Drawdown Chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_equity.index, y=portfolio_dd * 100,
        mode="lines", name="Pairs Portfolio Drawdown",
        fill="tozeroy", fillcolor="rgba(0, 230, 118, 0.1)",
        line=dict(color="#ff1744", width=1.5)
    ))
    fig_dd.add_trace(go.Scatter(
        x=df_equity.index, y=bench_dd * 100,
        mode="lines", name="Nifty 50 Drawdown",
        line=dict(color="#29b6f6", width=1.0, dash="dash")
    ))
    fig_dd.update_layout(
        title="Drawdown Profile Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=300,
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Period Table
    st.markdown("<div class='section-header'>Walk-Forward Period Performance Breakdown</div>", unsafe_allow_html=True)
    
    # Stylize Period Table
    df_results_styled = df_results.copy()
    df_results_styled["Return (%)"] = df_results_styled["Return (%)"].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
    df_results_styled["Max DD (%)"] = df_results_styled["Max DD (%)"].apply(lambda x: f"{x:.2f}%")
    df_results_styled["Win Rate (%)"] = df_results_styled["Win Rate (%)"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(df_results_styled, hide_index=True, use_container_width=True)

# ======================================================================
# TAB 2: Pair Signal Explorer
# ======================================================================
with tabs[1]:
    st.markdown("<div class='section-header'>Interactive Walk-Forward Pair Visualizer</div>", unsafe_allow_html=True)
    st.write("Reconstruct signals, rolling spreads, and raw stock prices for any selected pair in any period.")
    
    # Load prices
    prices = load_prices_cached()
    dates = prices.index
    
    # Period list
    rb_indices = list(range(TRAIN_WINDOW_DAYS, len(dates), REBALANCE_DAYS))
    periods_options = []
    for idx, ri in enumerate(rb_indices):
        te = min(ri + REBALANCE_DAYS, len(dates))
        ps, pe = dates[ri], dates[te - 1]
        periods_options.append(f"Period {idx+1}: {ps.date()} to {pe.date()}")
        
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_period_str = st.selectbox("Select Walk-Forward Period", periods_options)
        period_idx = periods_options.index(selected_period_str)
        
        # Get active pairs in this period from the trade log
        # Ensure we align the period selection
        p_start_date = pd.to_datetime(selected_period_str.split(": ")[1].split(" to ")[0])
        p_end_date = pd.to_datetime(selected_period_str.split(": ")[1].split(" to ")[1])
        
        # Filter trades that occurred inside this period window
        df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])
        df_trades["exit_date"] = pd.to_datetime(df_trades["exit_date"])
        
        # Unique pairs traded in this period
        period_trades = df_trades[(df_trades["entry_date"] >= p_start_date) & (df_trades["entry_date"] <= p_end_date)]
        
        if len(period_trades) > 0:
            pair_options = sorted(period_trades["pair"].unique().tolist())
        else:
            # Fallback to sector list if no trades executed (highly unlikely but safe)
            pair_options = ["HDFCBANK-ICICIBANK"]
            
        selected_pair = st.selectbox("Select Pair to Analyze", pair_options)
        
        # Extract stock names from pair string e.g. "SBIN-BANKBARODA" -> "SBIN.NS", "BANKBARODA.NS"
        sa_str, sb_str = selected_pair.split("-")
        sa = f"{sa_str}.NS"
        sb = f"{sb_str}.NS"
        
    # Reconstruct data
    ri = rb_indices[period_idx]
    te = min(ri + REBALANCE_DAYS, len(dates))
    ps, pe = dates[ri], dates[te - 1]
    wstart = ps - pd.Timedelta(days=WARMUP_DAYS)
    
    sp = prices.loc[wstart:pe]
    
    if sa in sp.columns and sb in sp.columns:
        pa, pb = sp[sa], sp[sb]
        
        # Compute metrics
        beta = compute_rolling_beta_fast(pa, pb, BETA_LOOKBACK)
        spread = compute_rolling_spread(pa, pb, beta)
        zs = compute_rolling_zscore(spread, Z_LOOKBACK)
        
        raw_sig = generate_signals_with_velocity(zs, Z_ENTRY, Z_EXIT, VELOCITY_LOOKBACK)
        # For simplicity, extract half life from training data
        tp = prices.iloc[max(0, ri - TRAIN_WINDOW_DAYS):ri]
        ols_train = compute_spread(tp[sa], tp[sb])
        hl = 15.0 # default/median
        
        sig = apply_risk_overlay(raw_sig, zs, hl, Z_STOP)
        
        # Sliced test data
        t_pa = pa.loc[ps:pe]
        t_pb = pb.loc[ps:pe]
        t_zs = zs.loc[ps:pe]
        t_sig = sig.loc[ps:pe]
        t_spread = spread.loc[ps:pe]
        t_beta = beta.loc[ps:pe]
        
        # Compute pair net returns
        sdf = pd.DataFrame({"price_a": pa, "price_b": pb, "beta": beta,
                            "spread": spread, "zscore": zs, "signal": sig})
        sdf_test = sdf.loc[ps:pe]
        rdf = compute_pair_returns(sdf_test)
        
        # Apply 5% drawdown stop
        cum = (1 + rdf["net_return"].fillna(0)).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        stops = dd[dd < -PAIR_DD_STOP].index
        if len(stops) > 0:
            rdf.loc[stops[0]:, ["net_return", "gross_return"]] = 0
            
        # Apply leverage to pair-level returns after the drawdown stop
        rdf["net_return"] = rdf["net_return"] * leverage
        cum_ret = (1 + rdf["net_return"].fillna(0)).cumprod() - 1
        
        # Create Plots
        # Chart 1: Normalized Stock Prices with Signal Markers
        norm_a = t_pa / t_pa.iloc[0]
        norm_b = t_pb / t_pb.iloc[0]
        
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Scatter(x=t_pa.index, y=norm_a, name=sa_str, line=dict(color="#00e676", width=2)))
        fig_stock.add_trace(go.Scatter(x=t_pb.index, y=norm_b, name=sb_str, line=dict(color="#ff1744", width=2)))
        
        # Highlight entry signals
        entries_long = t_sig[t_sig.diff() == 1]
        entries_short = t_sig[t_sig.diff() == -1]
        
        if len(entries_long) > 0:
            fig_stock.add_trace(go.Scatter(
                x=entries_long.index, y=norm_a.loc[entries_long.index],
                mode="markers", name="Buy A / Sell B",
                marker=dict(symbol="triangle-up", size=12, color="#00e676")
            ))
        if len(entries_short) > 0:
            fig_stock.add_trace(go.Scatter(
                x=entries_short.index, y=norm_a.loc[entries_short.index],
                mode="markers", name="Sell A / Buy B",
                marker=dict(symbol="triangle-down", size=12, color="#ff1744")
            ))
            
        fig_stock.update_layout(
            title=f"Normalized Stock Prices (Base = 1.0) & Execution Signals",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            template="plotly_dark",
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig_stock, use_container_width=True)
        
        # Chart 2: Z-Score with entry/exit regions
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=t_zs.index, y=t_zs, name="Spread Z-Score", line=dict(color="#29b6f6", width=2)))
        
        # Add entry threshold lines
        fig_z.add_hline(y=Z_ENTRY, line_dash="dash", line_color="#ff1744", annotation_text=f"Upper Entry (+{Z_ENTRY})")
        fig_z.add_hline(y=-Z_ENTRY, line_dash="dash", line_color="#00e676", annotation_text=f"Lower Entry (-{Z_ENTRY})")
        fig_z.add_hline(y=Z_EXIT, line_dash="dot", line_color="#8a94a6", annotation_text=f"Exit Thresholds (+/- {Z_EXIT})")
        fig_z.add_hline(y=-Z_EXIT, line_dash="dot", line_color="#8a94a6")
        fig_z.add_hline(y=Z_STOP, line_color="#e53935", annotation_text=f"Hard Stop (+/- {Z_STOP})")
        fig_z.add_hline(y=-Z_STOP, line_color="#e53935")
        
        # Shading signal areas
        long_dates = t_sig[t_sig == 1].index
        short_dates = t_sig[t_sig == -1].index
        
        for ld in long_dates:
            fig_z.add_vrect(x0=ld - pd.Timedelta(hours=12), x1=ld + pd.Timedelta(hours=12), fillcolor="green", opacity=0.08, line_width=0)
        for sd in short_dates:
            fig_z.add_vrect(x0=sd - pd.Timedelta(hours=12), x1=sd + pd.Timedelta(hours=12), fillcolor="red", opacity=0.08, line_width=0)
            
        fig_z.update_layout(
            title="Spread Z-Score & Trading Thresholds",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            template="plotly_dark",
            height=300,
            hovermode="x"
        )
        st.plotly_chart(fig_z, use_container_width=True)
        
        # Chart 3: Net Cumulative Pair Return
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret * 100, name="Pair Net P&L", line=dict(color="#00e676", width=2)))
        fig_ret.add_hline(y=0.0, line_color="#8a94a6", line_width=1)
        
        # Drawdown limit line
        fig_ret.add_hline(y=-PAIR_DD_STOP * 100, line_dash="dash", line_color="#ff1744", annotation_text=f"Pair Stop Loss ({PAIR_DD_STOP * 100:.1f}%)")
        
        fig_ret.update_layout(
            title="Pair Cumulative Net Return (%)",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=250,
            hovermode="x unified"
        )
        st.plotly_chart(fig_ret, use_container_width=True)
        
    else:
        st.error(f"Data for stocks {sa_str} or {sb_str} not available.")

# ======================================================================
# TAB 3: Trade Logs
# ======================================================================
with tabs[2]:
    st.markdown("<div class='section-header'>Out-of-Sample Executed Trades</div>", unsafe_allow_html=True)
    st.write(f"Analyze the {len(df_trades)} individual trades executed during the walk-forward validation period.")
    
    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        filter_pair = st.multiselect("Filter by Pair", sorted(df_trades["pair"].unique().tolist()))
    with c2:
        filter_dir = st.selectbox("Filter by Direction", ["All", "Long", "Short"])
    with c3:
        filter_outcome = st.selectbox("Filter by Outcome", ["All", "Winning", "Losing"])
        
    # Apply filters
    filtered_df = df_trades.copy()
    if filter_pair:
        filtered_df = filtered_df[filtered_df["pair"].isin(filter_pair)]
    if filter_dir != "All":
        filtered_df = filtered_df[filtered_df["direction"] == filter_dir]
    if filter_outcome == "Winning":
        filtered_df = filtered_df[filtered_df["trade_return"] > 0]
    elif filter_outcome == "Losing":
        filtered_df = filtered_df[filtered_df["trade_return"] <= 0]
        
    # Display statistics
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.metric("Total Trades Selected", len(filtered_df))
    with tc2:
        wr = (filtered_df["trade_return"] > 0).mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Win Rate", f"{wr:.1f}%")
    with tc3:
        avg_ret = filtered_df["trade_return"].mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Average Return", f"{avg_ret:+.2f}%")
    with tc4:
        avg_hold = filtered_df["holding_days"].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Holding Period", f"{avg_hold:.1f} days")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled trade log dataframe
    styled_filtered = filtered_df.copy()
    styled_filtered["trade_return"] = styled_filtered["trade_return"].apply(lambda x: f"+{x*100:.2f}%" if x > 0 else f"{x*100:.2f}%")
    styled_filtered["entry_date"] = pd.to_datetime(styled_filtered["entry_date"]).dt.date
    styled_filtered["exit_date"] = pd.to_datetime(styled_filtered["exit_date"]).dt.date
    
    st.dataframe(styled_filtered, hide_index=True, use_container_width=True)
    
    # Trade P&L Distribution
    if len(filtered_df) > 0:
        fig_hist = px.histogram(
            filtered_df, x=filtered_df["trade_return"] * 100,
            nbins=30, title="Trade Return Distribution (%)",
            labels={"x": "Trade P&L (%)"},
            color_discrete_sequence=["#29b6f6"],
            template="plotly_dark"
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ff1744")
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

# ======================================================================
# TAB 4: Parameter Sweep Explorer
# ======================================================================
with tabs[3]:
    st.markdown("<div class='section-header'>Grid Search Sensitivity Analysis</div>", unsafe_allow_html=True)
    st.write("Results of the 1,280 strategy executions swept during the rolling walk-forward parameter sweep.")
    
    if os.path.exists(tuning_path):
        df_tuning = pd.read_csv(tuning_path)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_tune_s = px.scatter(
                df_tuning, x="total_return", y="sharpe", color="dd_stop",
                hover_data=["z_entry", "z_exit", "z_stop", "vel_lb", "max_dd"],
                title="Sharpe Ratio vs. Total Return by Drawdown Stop Level",
                labels={"total_return": "Total Return (%)", "sharpe": "Sharpe Ratio"},
                color_continuous_scale=px.colors.sequential.Viridis,
                template="plotly_dark"
            )
            # Highlight optimal parameters
            opt = df_tuning[(df_tuning["z_entry"] == 2.0) & (df_tuning["z_exit"] == 0.7) & 
                            (df_tuning["z_stop"] == 3.5) & (df_tuning["vel_lb"] == 2) & (df_tuning["dd_stop"] == 0.05)]
            if len(opt) > 0:
                fig_tune_s.add_trace(go.Scatter(
                    x=opt["total_return"], y=opt["sharpe"], mode="markers",
                    marker=dict(color="#00e676", size=14, symbol="star", line=dict(color="white", width=2)),
                    name="Optimal Tuned Set"
                ))
            fig_tune_s.update_layout(height=450)
            st.plotly_chart(fig_tune_s, use_container_width=True)
            
        with c2:
            fig_tune_d = px.scatter(
                df_tuning, x="max_dd", y="total_return", color="vel_lb",
                hover_data=["z_entry", "z_exit", "z_stop", "dd_stop", "sharpe"],
                title="Total Return (%) vs. Maximum Drawdown (%) by Velocity Filter Lookback",
                labels={"max_dd": "Max Drawdown (%)", "total_return": "Total Return (%)"},
                template="plotly_dark"
            )
            if len(opt) > 0:
                fig_tune_d.add_trace(go.Scatter(
                    x=opt["max_dd"], y=opt["total_return"], mode="markers",
                    marker=dict(color="#00e676", size=14, symbol="star", line=dict(color="white", width=2)),
                    name="Optimal Tuned Set"
                ))
            fig_tune_d.update_layout(height=450)
            st.plotly_chart(fig_tune_d, use_container_width=True)
            
        st.markdown("<div class='section-header'>Parameter Combinations Rank Table</div>", unsafe_allow_html=True)
        st.write("Sorted by Sharpe Ratio descending.")
        st.dataframe(df_tuning.sort_values("sharpe", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ Tuning walk-forward results file not found. Run `python research/tune_walk_forward.py` to create the tuning sweep dataset.")

# ======================================================================
# TAB 5: Reference
# ======================================================================
with tabs[4]:
    st.markdown("<div class='section-header'>Pairs Trading & Cointegration — Technical Reference</div>", unsafe_allow_html=True)
    
    # Read learn.md
    try:
        with open("docs/learn.md", "r", encoding="utf-8") as f:
            learn_content = f.read()
        st.markdown(learn_content)
    except Exception as e:
        st.error(f"Could not load learn.md: {e}")
