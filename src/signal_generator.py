# ============================================================
# signal_generator.py — Rolling Z-Score & Trade Signals
# ============================================================
# This module generates buy/sell signals for selected pairs.
#
# Key principle: EVERYTHING is computed on a rolling window.
# At each point in time, we only use data from the past N days.
# This prevents lookahead bias (see learn.md Section 9).
#
# The pipeline for each pair on each day:
#   1. Compute hedge ratio β using rolling OLS (past 60 days)
#   2. Compute spread: log(A) - β × log(B)
#   3. Compute z-score: (spread - rolling_mean) / rolling_std
#   4. Generate signal based on z-score thresholds
#   5. Apply risk overlays (stop-loss, max holding period)
#
# Usage:
#   from src.signal_generator import generate_signals
#   signals_df = generate_signals(prices, selected_pairs)
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# Step 1: Rolling Hedge Ratio (β)
# ============================================================

def compute_rolling_beta(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Compute the hedge ratio β on a rolling window using OLS.

    At each day t, we run:
        log(A)[t-window:t] = α + β × log(B)[t-window:t] + ε

    This gives us a β that adapts over time as the relationship
    between the stocks evolves.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for the two stocks.
    window : int
        Lookback window in trading days. Defaults to config.LOOKBACK_WINDOW.

    Returns
    -------
    pd.Series
        Rolling β values. First (window-1) values will be NaN.
    """
    if window is None:
        window = getattr(config, "BETA_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)

    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # We'll compute rolling OLS manually using pandas rolling
    # (statsmodels RollingOLS can be slow for large datasets)
    betas = pd.Series(index=price_a.index, dtype=float)

    for i in range(window, len(price_a) + 1):
        y = log_a.iloc[i - window:i]
        x = log_b.iloc[i - window:i]
        x_with_const = sm.add_constant(x)

        try:
            model = sm.OLS(y, x_with_const).fit()
            betas.iloc[i - 1] = model.params.iloc[1]
        except Exception:
            betas.iloc[i - 1] = np.nan

    return betas


def compute_rolling_beta_fast(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Fast vectorized rolling beta using the rolling covariance formula.

    β = Cov(log_a, log_b) / Var(log_b)

    This is mathematically equivalent to OLS slope but much faster
    because it uses pandas rolling operations instead of a loop.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for the two stocks.
    window : int
        Lookback window in trading days. Defaults to config.LOOKBACK_WINDOW.

    Returns
    -------
    pd.Series
        Rolling β values. First (window-1) values will be NaN.
    """
    if window is None:
        window = getattr(config, "BETA_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)

    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # Rolling covariance and variance
    rolling_cov = log_a.rolling(window=window).cov(log_b)
    rolling_var = log_b.rolling(window=window).var()

    # β = Cov(A, B) / Var(B)
    beta = rolling_cov / rolling_var

    return beta


# ============================================================
# Step 2: Rolling Spread & Z-Score
# ============================================================

def compute_rolling_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    rolling_beta: pd.Series,
) -> pd.Series:
    """
    Compute the spread using the rolling hedge ratio.

    spread_t = log(A_t) - β_t × log(B_t)

    Unlike the static spread from Phase 2 (OLS residuals over the
    full period), this spread adapts as β changes over time.
    """
    log_a = np.log(price_a)
    log_b = np.log(price_b)

    spread = log_a - rolling_beta * log_b
    return spread


def compute_rolling_zscore(
    spread: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Compute the z-score of the spread using a rolling window.

    z_t = (spread_t - rolling_mean_t) / rolling_std_t

    This is the KEY signal — it tells us how many standard deviations
    the spread is from its recent mean.

    CRITICAL: We use .rolling() which only looks backward.
    This prevents lookahead bias.
    """
    if window is None:
        window = getattr(config, "Z_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (spread - rolling_mean) / rolling_std
    return zscore


# ============================================================
# Step 3: Signal Generation
# ============================================================

def generate_raw_signals(
    zscore: pd.Series,
    entry_threshold: float = None,
    exit_threshold: float = None,
) -> pd.Series:
    """
    Generate trading signals from z-score (vectorized, no loops).

    Signal values:
        +1 = Long spread (buy A, short B) → z was very negative
        -1 = Short spread (short A, buy B) → z was very positive
         0 = Flat (no position)

    The logic:
    - Enter LONG when z < -entry_threshold (spread too low → buy)
    - Enter SHORT when z > +entry_threshold (spread too high → sell)
    - EXIT when |z| < exit_threshold (spread returned to normal)
    - HOLD current position between thresholds

    Parameters
    ----------
    zscore : pd.Series
        Rolling z-score series.
    entry_threshold : float
        Absolute z-score to enter a trade. Default: config.Z_ENTRY_THRESHOLD
    exit_threshold : float
        Absolute z-score to exit a trade. Default: config.Z_EXIT_THRESHOLD

    Returns
    -------
    pd.Series
        Signal series: +1, -1, or 0.
    """
    if entry_threshold is None:
        entry_threshold = config.Z_ENTRY_THRESHOLD
    if exit_threshold is None:
        exit_threshold = config.Z_EXIT_THRESHOLD

    # Start with all zeros (no position)
    signal = pd.Series(0, index=zscore.index, dtype=float)

    # Mark entry and exit points
    # Entry: z crosses beyond ±entry_threshold
    long_entry = zscore < -entry_threshold   # Spread too low → go long
    short_entry = zscore > entry_threshold    # Spread too high → go short
    exit_signal = zscore.abs() < exit_threshold  # Back to normal → exit

    # We need to forward-fill between entry and exit.
    # The logic: once you enter, you stay in until exit.
    #
    # Implementation: set entry signals, then forward-fill,
    # but reset to 0 on exit.
    #
    # SPREAD VELOCITY FILTER: Before entering, check if the z-score
    # is already reverting (moving back toward 0). This avoids
    # entering when the spread is still diverging ("catching knives").

    velocity_lookback = getattr(config, 'SPREAD_VELOCITY_LOOKBACK', 0)

    position = 0
    signals = np.zeros(len(zscore))

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals[i] = 0
            position = 0  # Reset position — can't hold with no signal
            continue

        if position == 0:
            # Not in a trade — look for entry with velocity confirmation
            if z < -entry_threshold:
                # Want to go long — check if z is rising (reverting)
                if velocity_lookback > 0 and i >= velocity_lookback:
                    recent_z = zscore.iloc[i - velocity_lookback:i + 1]
                    if recent_z.iloc[-1] > recent_z.iloc[0]:
                        position = 1  # Confirmed: z is rising → enter long
                else:
                    position = 1    # No velocity check configured
            elif z > entry_threshold:
                # Want to go short — check if z is falling (reverting)
                if velocity_lookback > 0 and i >= velocity_lookback:
                    recent_z = zscore.iloc[i - velocity_lookback:i + 1]
                    if recent_z.iloc[-1] < recent_z.iloc[0]:
                        position = -1  # Confirmed: z is falling → enter short
                else:
                    position = -1   # No velocity check configured
        else:
            # In a trade — look for exit
            if abs(z) < exit_threshold:
                position = 0    # Exit

        signals[i] = position

    return pd.Series(signals, index=zscore.index, name="signal")


# ============================================================
# Step 4: Risk Overlay
# ============================================================

def apply_risk_overlay(
    signal: pd.Series,
    zscore: pd.Series,
    half_life: float,
    stop_loss_z: float = None,
    max_holding_multiplier: float = None,
) -> pd.Series:
    """
    Apply risk management rules on top of raw signals.

    Rules:
    1. STOP-LOSS: If |z| > stop_loss_z, force exit.
       Rationale: extreme z-scores suggest cointegration breakdown.

    2. MAX HOLDING PERIOD: If a trade has been open for longer than
       max_holding_multiplier × half_life days, force exit.
       Rationale: if the spread hasn't reverted in 2× half-life,
       the relationship may have changed.

    Parameters
    ----------
    signal : pd.Series
        Raw signal from generate_raw_signals().
    zscore : pd.Series
        Z-score series (for stop-loss check).
    half_life : float
        Half-life of the pair in trading days.
    stop_loss_z : float
        Z-score threshold for stop-loss. Default: config.Z_STOP_LOSS
    max_holding_multiplier : float
        Multiple of half-life for max holding. Default from config.

    Returns
    -------
    pd.Series
        Risk-adjusted signal series.
    """
    if stop_loss_z is None:
        stop_loss_z = config.Z_STOP_LOSS
    if max_holding_multiplier is None:
        max_holding_multiplier = config.MAX_HOLDING_PERIOD_MULTIPLIER

    max_hold_days = int(half_life * max_holding_multiplier)
    adjusted = signal.copy()

    # Track how long the current trade has been open
    days_in_trade = 0
    prev_signal = 0

    for i in range(len(adjusted)):
        current = adjusted.iloc[i]
        z = zscore.iloc[i]

        # Rule 1: Stop-loss — z-score blowout
        if current != 0 and not np.isnan(z) and abs(z) > stop_loss_z:
            adjusted.iloc[i] = 0
            days_in_trade = 0
            prev_signal = 0
            continue

        # Track holding period
        if current != 0 and current == prev_signal:
            days_in_trade += 1
        elif current != 0:
            days_in_trade = 1  # New trade started
        else:
            days_in_trade = 0

        # Rule 2: Max holding period
        if current != 0 and max_hold_days > 0 and days_in_trade > max_hold_days:
            adjusted.iloc[i] = 0
            days_in_trade = 0
            prev_signal = 0
            continue

        prev_signal = current

    return adjusted


# ============================================================
# Main: Generate signals for all selected pairs
# ============================================================

def generate_signals_for_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    half_life: float,
    beta_window: int = None,
    z_window: int = None,
) -> pd.DataFrame:
    """
    Generate complete signal data for a single pair.

    Returns a DataFrame with columns:
        - price_a, price_b: raw prices
        - beta: rolling hedge ratio
        - spread: rolling spread
        - zscore: rolling z-score
        - raw_signal: signal before risk overlay
        - signal: final signal after risk management

    Parameters
    ----------
    Generate rolling beta, spread, z-score, and signals for a pair.
    """
    if beta_window is None:
        beta_window = getattr(config, "BETA_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)
    if z_window is None:
        z_window = getattr(config, "Z_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)

    # Step 1: Rolling beta (fast vectorized method)
    beta = compute_rolling_beta_fast(price_a, price_b, beta_window)

    # Step 2: Rolling spread
    spread = compute_rolling_spread(price_a, price_b, beta)

    # Step 3: Rolling z-score
    zscore = compute_rolling_zscore(spread, z_window)

    # Step 4: Raw signals
    raw_signal = generate_raw_signals(zscore)

    # Step 5: Risk-adjusted signals
    signal = apply_risk_overlay(raw_signal, zscore, half_life)

    # Combine into a DataFrame
    result = pd.DataFrame({
        "price_a": price_a,
        "price_b": price_b,
        "beta": beta,
        "spread": spread,
        "zscore": zscore,
        "raw_signal": raw_signal,
        "signal": signal,
    })

    return result


def generate_signals(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    use_test_period: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Generate signals for ALL selected pairs.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price data (train + test).
    selected_pairs : pd.DataFrame
        Output from pair_selector.get_selected_pairs().
    use_test_period : bool
        If True, generate signals on the TEST period only
        (last TEST_PERIOD_DAYS). If False, generate on full period
        (useful for visualization, but NOT for backtesting).
    verbose : bool
        Print progress updates.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are pair names (e.g., "HDFCBANK-ICICIBANK"),
        values are DataFrames from generate_signals_for_pair().
    """
    if use_test_period:
        # Use test data — last TEST_PERIOD_DAYS
        # But we need enough days before the test period
        # to warm up the rolling calculations
        max_lookback = max(
            getattr(config, "BETA_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW),
            getattr(config, "Z_LOOKBACK_WINDOW", config.LOOKBACK_WINDOW)
        )
        warmup = max_lookback * 2  # Extra buffer
        start_idx = max(0, len(prices) - config.TEST_PERIOD_DAYS - warmup)
        signal_prices = prices.iloc[start_idx:]

        if verbose:
            test_start = prices.index[-config.TEST_PERIOD_DAYS]
            print(f"📅 Signal generation period:")
            print(f"   Warmup: {signal_prices.index[0].date()} to {test_start.date()}")
            print(f"   Test:   {test_start.date()} to {signal_prices.index[-1].date()}")
    else:
        signal_prices = prices
        if verbose:
            print(f"📅 Signal generation: full period")
            print(f"   {signal_prices.index[0].date()} to {signal_prices.index[-1].date()}")

    all_signals = {}

    for i, (_, pair) in enumerate(selected_pairs.iterrows()):
        stock_a = pair["stock_a"]
        stock_b = pair["stock_b"]
        half_life = pair["half_life"]

        pair_name = f"{stock_a.replace('.NS','')}-{stock_b.replace('.NS','')}"

        if verbose:
            print(f"\n   [{i+1}/{len(selected_pairs)}] {pair_name} "
                  f"(HL={half_life:.0f}d, β={pair['beta']:.3f})")

        try:
            price_a = signal_prices[stock_a]
            price_b = signal_prices[stock_b]

            signals_df = generate_signals_for_pair(
                price_a, price_b, half_life
            )

            # Trim warmup period if using test data
            if use_test_period:
                test_start = prices.index[-config.TEST_PERIOD_DAYS]
                signals_df = signals_df.loc[test_start:]

            # Count trades
            signal_changes = signals_df["signal"].diff().ne(0)
            entries = ((signals_df["signal"] != 0) & signal_changes).sum()

            if verbose:
                long_days = (signals_df["signal"] == 1).sum()
                short_days = (signals_df["signal"] == -1).sum()
                flat_days = (signals_df["signal"] == 0).sum()
                print(f"      Signals: {entries} entries | "
                      f"Long {long_days}d | Short {short_days}d | Flat {flat_days}d")

            all_signals[pair_name] = signals_df

        except Exception as e:
            if verbose:
                print(f"      ⚠️ Error: {e}")

    if verbose:
        print(f"\n✅ Generated signals for {len(all_signals)} pairs")

    return all_signals


# ============================================================
# Run directly for quick testing
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.pair_selector import run_pair_selection, get_selected_pairs

    prices = load_data()
    results = run_pair_selection(prices, verbose=False)
    selected = get_selected_pairs(results)

    all_signals = generate_signals(prices, selected)

    for pair_name, signals_df in all_signals.items():
        print(f"\n{pair_name}:")
        print(signals_df[["zscore", "signal"]].describe())
