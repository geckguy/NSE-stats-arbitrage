# ============================================================
# risk_manager.py — Position Sizing & Exposure Tracking
# ============================================================
# Handles portfolio-level risk management:
#   1. Position sizing — how much capital per pair
#   2. Net market exposure — are we truly market-neutral?
#   3. Portfolio drawdown breaker — halt if drawdown too large
#
# Usage:
#   from src.risk_manager import compute_position_sizes
#   sizes = compute_position_sizes(selected_pairs)
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_position_sizes(
    n_pairs: int,
    initial_capital: float = None,
    per_pair_pct: float = None,
) -> dict:
    """
    Compute the rupee allocation per pair.

    Parameters
    ----------
    n_pairs : int
        Number of pairs being traded.
    initial_capital : float
        Starting portfolio value. Default from config.
    per_pair_pct : float
        Fraction of capital per pair. Default from config.

    Returns
    -------
    dict with:
        - capital_per_pair: rupees allocated to each pair
        - total_deployed: total capital deployed
        - cash_buffer: remaining cash
        - cash_buffer_pct: buffer as percentage
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL
    if per_pair_pct is None:
        per_pair_pct = config.CAPITAL_PER_PAIR_PCT

    capital_per_pair = initial_capital * per_pair_pct
    total_deployed = capital_per_pair * n_pairs
    cash_buffer = initial_capital - total_deployed

    return {
        "initial_capital": initial_capital,
        "capital_per_pair": round(capital_per_pair, 2),
        "total_deployed": round(total_deployed, 2),
        "cash_buffer": round(max(0, cash_buffer), 2),
        "cash_buffer_pct": round(max(0, cash_buffer) / initial_capital * 100, 1),
        "n_pairs": n_pairs,
    }


def compute_net_exposure(
    all_signals: dict[str, pd.DataFrame],
    capital_per_pair: float,
) -> pd.DataFrame:
    """
    Track net market exposure across all pairs over time.

    A truly market-neutral strategy should have net exposure near 0.
    Positive = net long, negative = net short.

    Parameters
    ----------
    all_signals : dict
        Signal DataFrames for each pair (from signal_generator).
    capital_per_pair : float
        Capital allocated to each pair.

    Returns
    -------
    pd.DataFrame with columns:
        - long_exposure: total capital in long positions
        - short_exposure: total capital in short positions
        - net_exposure: long - short (should be near 0)
        - gross_exposure: long + short (total capital at risk)
    """
    # Collect all signals into one DataFrame
    all_sigs = pd.DataFrame()

    for pair_name, df in all_signals.items():
        all_sigs[pair_name] = df["signal"]

    # Align indices
    all_sigs = all_sigs.dropna()

    # Compute exposures
    long_exposure = (all_sigs == 1).sum(axis=1) * capital_per_pair
    short_exposure = (all_sigs == -1).sum(axis=1) * capital_per_pair

    exposure = pd.DataFrame({
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "net_exposure": long_exposure - short_exposure,
        "gross_exposure": long_exposure + short_exposure,
    })

    return exposure


def check_portfolio_drawdown(
    portfolio_returns: pd.Series,
    max_allowed_dd: float = 0.20,
) -> dict:
    """
    Check if portfolio drawdown exceeds the maximum allowed.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    max_allowed_dd : float
        Maximum allowed drawdown (0.20 = 20%).

    Returns
    -------
    dict with:
        - breached: True if drawdown exceeded limit
        - current_dd: current drawdown level
        - breach_date: date when limit was first breached (or None)
    """
    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    breached = (drawdown < -max_allowed_dd).any()
    current_dd = float(drawdown.iloc[-1])

    breach_date = None
    if breached:
        breach_date = drawdown[drawdown < -max_allowed_dd].index[0]

    return {
        "breached": bool(breached),
        "current_dd": round(current_dd * 100, 2),
        "max_dd": round(float(drawdown.min()) * 100, 2),
        "breach_date": breach_date,
    }
