# ============================================================
# data_loader.py — Download, clean, and cache NSE price data
# ============================================================
# This module handles everything related to getting price data:
#   1. Download adjusted close prices from yfinance
#   2. Clean missing values (forward-fill, then drop)
#   3. Cache to disk as a parquet file (fast & compact)
#   4. Validate data quality
#
# Usage:
#   from src.data_loader import load_data
#   prices = load_data()  # Returns a clean DataFrame
# ============================================================

import os
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_all_tickers() -> list[str]:
    """
    Flatten all tickers from SECTOR_TICKERS into a single list.
    Removes duplicates (in case a stock appears in multiple sectors).
    """
    all_tickers = []
    for sector, tickers in config.SECTOR_TICKERS.items():
        all_tickers.extend(tickers)
    return list(set(all_tickers))  # set() removes duplicates


def download_prices(tickers: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.

    Parameters
    ----------
    tickers : list of str, optional
        Which tickers to download. If None, downloads all from config.
    force_refresh : bool
        If True, re-download even if cached data exists.

    Returns
    -------
    pd.DataFrame
        Columns = ticker symbols, Index = dates, Values = adjusted close prices.
    """
    if tickers is None:
        tickers = get_all_tickers()

    cache_path = os.path.join(config.DATA_DIR, config.CACHE_FILENAME)

    # Check cache first (unless force_refresh)
    if not force_refresh and os.path.exists(cache_path):
        print(f"📂 Loading cached data from {cache_path}")
        cached = pd.read_parquet(cache_path)

        # Check if all requested tickers are in cache
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            return cached[tickers]
        
        # If we have some columns but not all, download only the missing ones to save time/bandwidth
        available_tickers = [t for t in tickers if t in cached.columns]
        if available_tickers:
            print(f"⚠️  Missing {len(missing)} tickers in cache: {missing}. Attempting to download missing only.")
            try:
                raw_missing = yf.download(
                    tickers=missing,
                    start=config.DATA_START,
                    end=config.DATA_END,
                    auto_adjust=True,
                    progress=False,
                )
                if not raw_missing.empty:
                    if isinstance(raw_missing.columns, pd.MultiIndex):
                        prices_missing = raw_missing["Close"]
                    else:
                        prices_missing = raw_missing[["Close"]]
                        prices_missing.columns = [m for m in missing if m in raw_missing.columns] or missing
                    
                    # Merge missing with cached
                    for col in prices_missing.columns:
                        cached[col] = prices_missing[col]
                    
                    # Save updated cache
                    cached.to_parquet(cache_path)
                return cached[[t for t in tickers if t in cached.columns]]
            except Exception as e:
                print(f"⚠️ Failed to download missing tickers: {e}. Returning available cached columns.")
                return cached[available_tickers]
        else:
            print("⚠️ Cache exists but has no matching tickers. Re-downloading all.")

    # Download from yfinance
    print(f"📡 Downloading {len(tickers)} tickers from yfinance...")
    print(f"   Period: {config.DATA_START} to {config.DATA_END}")

    raw = yf.download(
        tickers=tickers,
        start=config.DATA_START,
        end=config.DATA_END,
        auto_adjust=True,   # Use adjusted prices (handles splits/dividends)
        progress=True,
    )

    # yfinance returns multi-level columns when downloading multiple tickers.
    # We only want the "Close" prices (which are adjusted close since auto_adjust=True).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker case
        prices = raw[["Close"]]
        prices.columns = tickers

    return prices


def clean_prices(prices: pd.DataFrame, max_ffill_days: int = 5) -> pd.DataFrame:
    """
    Clean the raw price data.

    Steps:
    1. Forward-fill gaps up to max_ffill_days (handles weekends, short holidays).
    2. Drop any rows that still have NaN (long gaps = suspicious).
    3. Drop tickers that have too much missing data (> 10% missing).
    4. Sort by date.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price DataFrame from download_prices().
    max_ffill_days : int
        Maximum consecutive days to forward-fill. Beyond this, we treat
        the gap as suspicious (stock halt, data error).

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    print(f"\n🧹 Cleaning data...")
    print(f"   Raw shape: {prices.shape}")

    # Track issues
    issues = []

    # Step 1: Check for tickers with too much missing data
    missing_pct = prices.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.10].index.tolist()
    if bad_tickers:
        issues.append(f"Dropped tickers with >10% missing data: {bad_tickers}")
        prices = prices.drop(columns=bad_tickers)

    # Step 2: Forward-fill small gaps (weekends, 1-2 day holidays)
    prices = prices.ffill(limit=max_ffill_days)

    # Step 3: Drop remaining NaN rows
    # (start of the dataset might have NaN before all tickers have data)
    rows_before = len(prices)
    prices = prices.dropna()
    rows_dropped = rows_before - len(prices)
    if rows_dropped > 0:
        issues.append(f"Dropped {rows_dropped} rows with remaining NaN values")

    # Step 4: Sort by date
    prices = prices.sort_index()

    # Report
    print(f"   Clean shape: {prices.shape}")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Trading days: {len(prices)}")
    if issues:
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print(f"   ✅ No issues found")

    return prices


def validate_data(prices: pd.DataFrame) -> dict:
    """
    Run quality checks on the cleaned data.

    Returns a dict with validation results so you can inspect them
    in the notebook.
    """
    report = {
        "shape": prices.shape,
        "date_range": (prices.index[0].date(), prices.index[-1].date()),
        "trading_days": len(prices),
        "tickers": list(prices.columns),
        "missing_values": prices.isna().sum().sum(),
        "issues": [],
    }

    # Check 1: Enough data?
    if len(prices) < 500:
        report["issues"].append(
            f"Only {len(prices)} trading days. Need at least 500 for meaningful analysis."
        )

    # Check 2: Any suspiciously flat periods?
    # (price doesn't change for 10+ consecutive days = possible data error)
    for ticker in prices.columns:
        daily_changes = prices[ticker].diff()
        # Count max consecutive zeros
        is_zero = daily_changes == 0
        max_flat = 0
        current_flat = 0
        for val in is_zero:
            if val:
                current_flat += 1
                max_flat = max(max_flat, current_flat)
            else:
                current_flat = 0
        if max_flat >= 10:
            report["issues"].append(
                f"{ticker}: price unchanged for {max_flat} consecutive days"
            )

    # Check 3: Any extreme daily moves? (> 20% in a single day)
    daily_returns = prices.pct_change()
    for ticker in prices.columns:
        extreme = daily_returns[ticker].abs() > 0.20
        n_extreme = extreme.sum()
        if n_extreme > 0:
            report["issues"].append(
                f"{ticker}: {n_extreme} days with >20% daily move (possible split/error)"
            )

    return report


def save_to_cache(prices: pd.DataFrame) -> str:
    """Save cleaned prices to parquet cache. Returns the file path."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    cache_path = os.path.join(config.DATA_DIR, config.CACHE_FILENAME)
    prices.to_parquet(cache_path)
    print(f"\n💾 Saved to {cache_path}")
    return cache_path


def load_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point: download, clean, cache, and return price data.

    This is the function you'll call from other modules:
        from src.data_loader import load_data
        prices = load_data()

    Parameters
    ----------
    force_refresh : bool
        If True, re-download from yfinance even if cache exists.

    Returns
    -------
    pd.DataFrame
        Clean adjusted close prices. Columns = tickers, Index = dates.
    """
    prices = download_prices(force_refresh=force_refresh)
    prices = clean_prices(prices)
    save_to_cache(prices)
    return prices


def get_sector_for_ticker(ticker: str) -> str | None:
    """Look up which sector a ticker belongs to."""
    for sector, tickers in config.SECTOR_TICKERS.items():
        if ticker in tickers:
            return sector
    return None


def get_tickers_by_sector(sector: str) -> list[str]:
    """Get all tickers for a given sector."""
    return config.SECTOR_TICKERS.get(sector, [])


# ============================================================
# If you run this file directly, it downloads and validates data
# ============================================================
if __name__ == "__main__":
    prices = load_data(force_refresh=True)

    print("\n" + "=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    report = validate_data(prices)

    print(f"\nShape: {report['shape']}")
    print(f"Date range: {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"Trading days: {report['trading_days']}")
    print(f"Tickers ({len(report['tickers'])}): {report['tickers']}")
    print(f"Missing values: {report['missing_values']}")

    if report["issues"]:
        print(f"\n⚠️  Issues found:")
        for issue in report["issues"]:
            print(f"   - {issue}")
    else:
        print(f"\n✅ All checks passed!")
