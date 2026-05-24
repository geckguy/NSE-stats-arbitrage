# ============================================================
# pair_selector.py — Cointegration Testing & Pair Selection
# ============================================================
# This module answers the question: "Which stock pairs are
# suitable for pairs trading?"
#
# A good pair must satisfy THREE criteria:
#   1. Cointegrated (ADF p-value < 0.05) — spread is stationary
#   2. Good half-life (5-30 days) — reverts fast enough to trade
#   3. Stable cointegration — relationship holds across time
#
# Usage:
#   from src.pair_selector import run_pair_selection
#   results = run_pair_selection(prices)
# ============================================================

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# Step 1: Generate candidate pairs
# ============================================================

def generate_candidate_pairs(prices: pd.DataFrame) -> list[dict]:
    """
    Generate all intra-sector pairs from the available price data.

    For each sector, we create C(n, 2) combinations. For example,
    4 banking stocks → 6 pairs. We only pair stocks within the
    same sector because cross-sector pairs rarely have economic
    justification for cointegration.

    Returns a list of dicts with keys: stock_a, stock_b, sector
    """
    candidates = []

    for sector, tickers in config.SECTOR_TICKERS.items():
        # Only include tickers we actually have data for
        available = [t for t in tickers if t in prices.columns]

        if len(available) < 2:
            print(f"⚠️  {sector}: only {len(available)} tickers available, need at least 2. Skipping.")
            continue

        # Generate all unique pairs within this sector
        for stock_a, stock_b in combinations(available, 2):
            candidates.append({
                "stock_a": stock_a,
                "stock_b": stock_b,
                "sector": sector,
            })

    print(f"📊 Generated {len(candidates)} candidate pairs across {len(config.SECTOR_TICKERS)} sectors")
    return candidates


# ============================================================
# Step 2: Compute spread using OLS regression
# ============================================================

def compute_spread(price_a: pd.Series, price_b: pd.Series) -> dict:
    """
    Compute the spread between two stocks using OLS regression.

    The regression:  log(A) = α + β × log(B) + ε

    Where:
    - β (beta) is the hedge ratio
    - ε (residuals) is the spread
    - α (alpha) is the intercept

    Returns dict with: beta, alpha, spread (Series), r_squared
    """
    # Use log prices (see learn.md Section 6 for why)
    y = np.log(price_a)  # Dependent variable
    x = np.log(price_b)  # Independent variable

    # Add constant (intercept) to the regression
    x_with_const = sm.add_constant(x)

    # Fit OLS regression
    model = sm.OLS(y, x_with_const).fit()

    return {
        "beta": model.params.iloc[1],       # Hedge ratio
        "alpha": model.params.iloc[0],       # Intercept
        "spread": model.resid,         # The spread (residuals)
        "r_squared": model.rsquared,   # How well B explains A
    }


# ============================================================
# Step 3: Test for cointegration (ADF test)
# ============================================================

def test_cointegration(spread: pd.Series) -> dict:
    """
    Run the Augmented Dickey-Fuller test on the spread.

    Null hypothesis: spread has a unit root (non-stationary) → BAD
    Alternative: spread is stationary → GOOD

    We want p-value < 0.05 to reject H₀ and confirm stationarity.

    Also runs the Engle-Granger cointegration test as a cross-check.
    """
    # ADF test on the spread (residuals from OLS)
    adf_result = adfuller(spread, autolag="AIC")

    return {
        "adf_statistic": adf_result[0],  # More negative = more stationary
        "adf_pvalue": adf_result[1],      # < 0.05 = stationary
        "adf_critical_values": adf_result[4],  # 1%, 5%, 10% thresholds
        "is_cointegrated": adf_result[1] < config.ADF_P_VALUE_CUTOFF,
    }


def test_cointegration_engle_granger(price_a: pd.Series, price_b: pd.Series) -> dict:
    """
    Engle-Granger cointegration test — a second opinion.

    This is a convenience wrapper around statsmodels.coint() which
    does the OLS + ADF internally. We use it as a cross-check
    against our manual OLS → ADF approach.
    """
    score, pvalue, critical_values = coint(price_a, price_b)

    return {
        "eg_statistic": score,
        "eg_pvalue": pvalue,
        "eg_critical_values": dict(zip(["1%", "5%", "10%"], critical_values)),
        "eg_is_cointegrated": pvalue < config.ADF_P_VALUE_CUTOFF,
    }


# ============================================================
# Step 4: Compute half-life of mean reversion
# ============================================================

def compute_half_life(spread: pd.Series) -> float:
    """
    Compute the half-life of mean reversion using an AR(1) model.

    The idea: model the spread as mean-reverting:
        spread_t - spread_{t-1} = λ × spread_{t-1} + ε

    If λ < 0, the spread reverts. The half-life tells you
    how many days it takes to revert halfway back to the mean:
        half_life = -log(2) / log(1 + λ)

    Returns:
        float: half-life in trading days. Negative means no reversion.
    """
    # Create lagged spread and spread difference
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()

    # Drop NaN from differencing/lagging
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()

    # Align indices (both start from the second observation)
    common_idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[common_idx]
    spread_diff = spread_diff.loc[common_idx]

    # Regress: Δspread = λ × spread_lag + ε
    model = sm.OLS(spread_diff, spread_lag).fit()
    lambda_param = model.params.iloc[0]

    # Compute half-life
    if lambda_param >= 0:
        # λ ≥ 0 means no mean reversion — spread drifts or is explosive
        return -1.0  # Sentinel value: not mean-reverting

    half_life = -np.log(2) / np.log(1 + lambda_param)

    return half_life


# ============================================================
# Step 5: Cointegration stability (rolling ADF)
# ============================================================

def test_cointegration_stability(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 252,
    step: int = 63,
) -> dict:
    """
    Test whether cointegration holds consistently across time.

    We run the ADF test on rolling sub-windows of the spread.
    If cointegration breaks in too many windows, the pair is unreliable.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for the two stocks.
    window : int
        Rolling window size in trading days (default: 252 = 1 year).
    step : int
        How many days to slide the window forward (default: 63 = 1 quarter).

    Returns
    -------
    dict with:
        - n_windows: total windows tested
        - n_cointegrated: windows where ADF p < 0.05
        - stability_pct: percentage of windows that are cointegrated
        - pvalues: list of p-values per window (for plotting)
    """
    n = len(price_a)
    pvalues = []

    for start in range(0, n - window, step):
        end = start + window
        window_a = price_a.iloc[start:end]
        window_b = price_b.iloc[start:end]

        # Compute spread on this window
        ols_result = compute_spread(window_a, window_b)
        spread = ols_result["spread"]

        # ADF test
        try:
            adf_result = adfuller(spread, autolag="AIC")
            pvalues.append(adf_result[1])
        except Exception:
            pvalues.append(1.0)  # Failed = assume non-stationary

    n_windows = len(pvalues)
    n_cointegrated = sum(1 for p in pvalues if p < config.ADF_P_VALUE_CUTOFF)

    return {
        "n_windows": n_windows,
        "n_cointegrated": n_cointegrated,
        "stability_pct": (n_cointegrated / n_windows * 100) if n_windows > 0 else 0,
        "pvalues": pvalues,
    }


# ============================================================
# Step 6: Score and rank pairs
# ============================================================

def score_pair(adf_pvalue: float, half_life: float, stability_pct: float) -> float:
    """
    Compute a composite score for ranking pairs.

    Higher score = better pair. Components:
    1. ADF p-value (lower = better) → inverted so lower p = higher score
    2. Half-life proximity to 15 days (sweet spot)
    3. Stability percentage (higher = better)

    Each component is normalized to [0, 1] and weighted.
    """
    # Component 1: ADF p-value score (0 to 1, higher is better)
    # p=0 → score=1, p=0.05 → score=0
    adf_score = max(0, 1 - (adf_pvalue / config.ADF_P_VALUE_CUTOFF))

    # Component 2: Half-life score (0 to 1)
    # Best at 15 days, decays as you move away
    ideal_hl = 15.0
    if half_life < 0:
        hl_score = 0.0  # Not mean-reverting
    elif config.HALF_LIFE_MIN <= half_life <= config.HALF_LIFE_MAX:
        # Distance from ideal, normalized by range
        hl_range = config.HALF_LIFE_MAX - config.HALF_LIFE_MIN
        distance = abs(half_life - ideal_hl) / hl_range
        hl_score = 1 - distance
    else:
        hl_score = 0.0  # Outside acceptable range

    # Component 3: Stability score (0 to 1)
    stability_score = stability_pct / 100.0

    # Weighted combination
    # ADF gets highest weight because it's the fundamental test
    score = (0.40 * adf_score) + (0.30 * hl_score) + (0.30 * stability_score)

    return round(score, 4)


# ============================================================
# Main: Run full pair selection pipeline
# ============================================================

def run_pair_selection(prices: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run the complete pair selection pipeline.

    Steps:
    1. Generate all intra-sector pair combinations
    2. For each pair: compute spread, run ADF, compute half-life
    3. Test cointegration stability
    4. Score and rank all pairs
    5. Return top MAX_PAIRS pairs

    Parameters
    ----------
    prices : pd.DataFrame
        Clean price data (from data_loader.load_data()).
    verbose : bool
        If True, print progress updates.

    Returns
    -------
    pd.DataFrame
        All tested pairs with their metrics, sorted by score.
        Columns: stock_a, stock_b, sector, beta, r_squared,
                 adf_statistic, adf_pvalue, is_cointegrated,
                 eg_pvalue, half_life, stability_pct, score
    """
    # Use only training data (exclude test period)
    train_prices = prices.iloc[:-config.TEST_PERIOD_DAYS]

    if verbose:
        print(f"📅 Training period: {train_prices.index[0].date()} to {train_prices.index[-1].date()}")
        print(f"   ({len(train_prices)} trading days)")
        print(f"🔒 Test period held out: last {config.TEST_PERIOD_DAYS} days")
        print()

    # Step 1: Generate candidates
    candidates = generate_candidate_pairs(train_prices)

    # Step 2-5: Analyze each pair
    results = []
    total = len(candidates)

    for i, pair in enumerate(candidates):
        stock_a = pair["stock_a"]
        stock_b = pair["stock_b"]
        sector = pair["sector"]

        if verbose and (i + 1) % 25 == 0:
            print(f"   Analyzing pair {i+1}/{total}...")

        try:
            price_a = train_prices[stock_a]
            price_b = train_prices[stock_b]

            # Compute spread via OLS
            ols = compute_spread(price_a, price_b)

            # ADF test on spread
            adf = test_cointegration(ols["spread"])

            # Engle-Granger cross-check
            eg = test_cointegration_engle_granger(price_a, price_b)

            # Half-life
            hl = compute_half_life(ols["spread"])

            # Stability (only for pairs that pass initial ADF)
            # Skip stability test for clearly non-cointegrated pairs (saves time)
            if adf["is_cointegrated"]:
                stab = test_cointegration_stability(price_a, price_b)
                stability_pct = stab["stability_pct"]
            else:
                stability_pct = 0.0

            # Score
            pair_score = score_pair(adf["adf_pvalue"], hl, stability_pct)

            results.append({
                "stock_a": stock_a,
                "stock_b": stock_b,
                "sector": sector,
                "beta": round(ols["beta"], 4),
                "r_squared": round(ols["r_squared"], 4),
                "adf_statistic": round(adf["adf_statistic"], 4),
                "adf_pvalue": round(adf["adf_pvalue"], 4),
                "is_cointegrated": adf["is_cointegrated"],
                "eg_pvalue": round(eg["eg_pvalue"], 4),
                "half_life": round(hl, 1),
                "stability_pct": round(stability_pct, 1),
                "score": pair_score,
            })

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Error analyzing {stock_a} ↔ {stock_b}: {e}")
            continue

    # Convert to DataFrame and sort by score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"PAIR SELECTION RESULTS")
        print(f"{'='*70}")
        n_coint = results_df["is_cointegrated"].sum()
        print(f"Total pairs tested: {len(results_df)}")
        print(f"Cointegrated (ADF p < {config.ADF_P_VALUE_CUTOFF}): {n_coint}")

        # Good half-life among cointegrated
        good_hl = results_df[
            results_df["is_cointegrated"]
            & (results_df["half_life"] >= config.HALF_LIFE_MIN)
            & (results_df["half_life"] <= config.HALF_LIFE_MAX)
        ]
        print(f"Good half-life ({config.HALF_LIFE_MIN}-{config.HALF_LIFE_MAX} days): {len(good_hl)}")

        print(f"\n🏆 Top {min(config.MAX_PAIRS, len(results_df))} pairs:")
        top = results_df.head(config.MAX_PAIRS)
        for _, row in top.iterrows():
            a = row['stock_a'].replace('.NS', '')
            b = row['stock_b'].replace('.NS', '')
            coint_mark = "✅" if row['is_cointegrated'] else "❌"
            print(f"   {coint_mark} {a} ↔ {b} ({row['sector']}) | "
                  f"ADF p={row['adf_pvalue']:.4f} | "
                  f"HL={row['half_life']:.0f}d | "
                  f"Stability={row['stability_pct']:.0f}% | "
                  f"Score={row['score']:.3f}")

    return results_df


def get_selected_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the results to only the top tradeable pairs.

    Selection criteria:
    1. Must be cointegrated (ADF p < 0.05)
    2. If REQUIRE_BOTH_COINT_TESTS: Engle-Granger must also pass
    3. Must have acceptable half-life (5-30 days)
    4. Must meet minimum stability threshold
    5. Take top MAX_PAIRS by score
    """
    mask = (
        (results_df["is_cointegrated"])
        & (results_df["half_life"] >= config.HALF_LIFE_MIN)
        & (results_df["half_life"] <= config.HALF_LIFE_MAX)
        & (results_df["stability_pct"] >= config.MIN_STABILITY_PCT)
    )

    # Require both cointegration tests if configured
    if config.REQUIRE_BOTH_COINT_TESTS and "eg_pvalue" in results_df.columns:
        mask = mask & (results_df["eg_pvalue"] < config.ADF_P_VALUE_CUTOFF)

    selected = results_df[mask].head(config.MAX_PAIRS)

    print(f"\n✅ Selected {len(selected)} pairs for trading:")
    for _, row in selected.iterrows():
        a = row['stock_a'].replace('.NS', '')
        b = row['stock_b'].replace('.NS', '')
        stab = row.get('stability_pct', 0)
        print(f"   {a} ↔ {b} ({row['sector']}) [stability={stab:.0f}%]")

    return selected


# ============================================================
# Run directly for quick testing
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_data

    prices = load_data()
    results = run_pair_selection(prices)
    selected = get_selected_pairs(results)

    # Save results for the notebook
    results.to_csv("data/pair_selection_results.csv", index=False)
    print(f"\n💾 Full results saved to data/pair_selection_results.csv")
