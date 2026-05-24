# ============================================================
# metrics.py — Performance Metrics for Strategy Evaluation
# ============================================================
# All metrics take a returns series or trade log as input.
# Pure functions, no side effects — easy to test and reuse.
#
# Usage:
#   from src.metrics import compute_all_metrics
#   stats = compute_all_metrics(daily_returns)
# ============================================================

import numpy as np
import pandas as pd

# Trading days per year (used for annualization)
TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = (mean_return - risk_free) / std_return × √252

    Parameters
    ----------
    returns : pd.Series
        Daily returns (not cumulative).
    risk_free_rate : float
        Annual risk-free rate. Default 6% (India 10Y govt bond ~6-7%).

    Returns
    -------
    float
        Annualized Sharpe ratio. > 1.0 is good, > 2.0 is excellent.
    """
    if returns.std() == 0 or len(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf
    return float(np.sqrt(TRADING_DAYS) * excess_returns.mean() / excess_returns.std())


def max_drawdown(returns: pd.Series) -> dict:
    """
    Maximum drawdown — the worst peak-to-trough decline.

    Returns
    -------
    dict with:
        - max_dd: maximum drawdown as a negative fraction (e.g., -0.15 = 15%)
        - max_dd_duration: number of trading days in the worst drawdown
        - drawdown_series: full drawdown time series (for plotting)
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    max_dd = float(drawdown.min())

    # Duration of max drawdown
    # Find the peak before the trough and the recovery after
    trough_idx = drawdown.idxmin()
    peak_before = cumulative.loc[:trough_idx].idxmax()

    # Recovery: when cumulative exceeds the peak again
    after_trough = cumulative.loc[trough_idx:]
    recovery_mask = after_trough >= peak.loc[trough_idx]
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        dd_duration = len(cumulative.loc[peak_before:recovery_idx])
    else:
        dd_duration = len(cumulative.loc[peak_before:])  # Never recovered

    return {
        "max_dd": max_dd,
        "max_dd_duration": dd_duration,
        "drawdown_series": drawdown,
    }


def cagr(returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (final_value / initial_value)^(1/years) - 1
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    final = cumulative.iloc[-1]
    years = len(returns) / TRADING_DAYS

    if years <= 0 or final <= 0:
        return 0.0

    return float(final ** (1 / years) - 1)


def win_rate(trade_pnls: pd.Series) -> float:
    """
    Percentage of trades that were profitable.

    Parameters
    ----------
    trade_pnls : pd.Series
        P&L for each individual trade.
    """
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    """
    Gross profit / gross loss. > 1.0 means profitable overall.

    profit_factor = sum(winning_trades) / abs(sum(losing_trades))
    """
    wins = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())

    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar ratio = CAGR / |max_drawdown|

    Measures return per unit of drawdown risk.
    """
    annual_return = cagr(returns)
    dd = max_drawdown(returns)["max_dd"]

    if dd == 0:
        return 0.0
    return float(annual_return / abs(dd))


def total_return(returns: pd.Series) -> float:
    """Total cumulative return over the period."""
    if len(returns) == 0:
        return 0.0
    return float((1 + returns).cumprod().iloc[-1] - 1)


def volatility(returns: pd.Series) -> float:
    """Annualized volatility (standard deviation of returns)."""
    if len(returns) == 0:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def compute_all_metrics(
    daily_returns: pd.Series,
    trade_pnls: pd.Series | None = None,
) -> dict:
    """
    Compute all performance metrics at once.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily strategy returns.
    trade_pnls : pd.Series, optional
        Per-trade P&L (for win rate, profit factor).

    Returns
    -------
    dict with all metrics.
    """
    dd_info = max_drawdown(daily_returns)

    metrics = {
        "total_return": round(total_return(daily_returns) * 100, 2),
        "cagr": round(cagr(daily_returns) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio(daily_returns), 2),
        "max_drawdown": round(dd_info["max_dd"] * 100, 2),
        "max_dd_duration_days": dd_info["max_dd_duration"],
        "calmar_ratio": round(calmar_ratio(daily_returns), 2),
        "volatility": round(volatility(daily_returns) * 100, 2),
        "trading_days": len(daily_returns),
    }

    if trade_pnls is not None and len(trade_pnls) > 0:
        metrics["n_trades"] = len(trade_pnls)
        metrics["win_rate"] = round(win_rate(trade_pnls) * 100, 1)
        metrics["profit_factor"] = round(profit_factor(trade_pnls), 2)
        metrics["avg_trade_pnl"] = round(float(trade_pnls.mean()) * 100, 4)
        metrics["best_trade"] = round(float(trade_pnls.max()) * 100, 2)
        metrics["worst_trade"] = round(float(trade_pnls.min()) * 100, 2)

    return metrics
