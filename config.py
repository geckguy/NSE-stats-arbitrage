# ============================================================
# config.py — Central Configuration for Pairs Trading Engine
# ============================================================
# All strategy parameters are defined here. Modify this file
# to adjust thresholds, universes, and cost assumptions.
# ============================================================

# --- Sector universe ---
# Pairs are restricted to within-sector candidates to ensure
# an economically motivated cointegration relationship.
# Tickers use the ".NS" suffix (NSE via yfinance).

SECTOR_TICKERS = {
    # ── BANKING ──────────────────────────────────────────────
    "Private Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
        "AUBANK.NS", "RBLBANK.NS", "CUB.NS",
    ],
    "PSU Banking": [
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
        "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS", "IOB.NS",
    ],

    # ── FINANCIAL SERVICES (non-bank) ────────────────────────
    "NBFCs & Financial Services": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCAMC.NS", "SBILIFE.NS",
        "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "MUTHOOTFIN.NS",
        "CHOLAFIN.NS", "M&MFIN.NS", "SHRIRAMFIN.NS", "PFC.NS",
        "RECLTD.NS", "LICHSGFIN.NS", "MANAPPURAM.NS",
    ],

    # ── TECHNOLOGY ───────────────────────────────────────────
    "IT Services": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
        "LTTS.NS", "OFSS.NS", "TATAELXSI.NS",
    ],

    # ── ENERGY ───────────────────────────────────────────────
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",
        "GAIL.NS", "HINDPETRO.NS", "PETRONET.NS", "OIL.NS",
        "IGL.NS", "MGL.NS", "GUJGASLTD.NS",
    ],

    # ── AUTOMOBILES ──────────────────────────────────────────
    "Automobiles & Auto Components": [
        "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS",
        "BALKRISIND.NS", "MOTHERSON.NS", "BHARATFORG.NS", "BOSCHLTD.NS",
        "MRF.NS", "APOLLOTYRE.NS", "EXIDEIND.NS",
    ],

    # ── CONSUMER ─────────────────────────────────────────────
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",
        "TATACONSUM.NS", "EMAMILTD.NS", "VBL.NS", "UBL.NS",
        "PGHH.NS",
    ],
    "Consumer Durables & Retail": [
        "TITAN.NS", "HAVELLS.NS", "VOLTAS.NS", "WHIRLPOOL.NS",
        "CROMPTON.NS", "BLUESTARCO.NS", "PAGEIND.NS", "TRENT.NS",
        "DMART.NS",
    ],

    # ── METALS & MINING ──────────────────────────────────────
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
        "NATIONALUM.NS", "SAIL.NS", "NMDC.NS", "JINDALSTEL.NS",
        "COALINDIA.NS", "MOIL.NS",
    ],

    # ── PHARMA & HEALTHCARE ──────────────────────────────────
    "Pharma & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
        "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "TORNTPHARM.NS",
        "ALKEM.NS", "IPCALAB.NS", "LAURUSLABS.NS", "APOLLOHOSP.NS",
        "MAXHEALTH.NS", "FORTIS.NS",
    ],

    # ── CEMENT & BUILDING ────────────────────────────────────
    "Cement": [
        "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS",
        "RAMCOCEM.NS", "DALBHARAT.NS", "JKCEMENT.NS", "STARCEMENT.NS",
        "JKLAKSHMI.NS",
    ],

    # ── POWER & UTILITIES ────────────────────────────────────
    "Power & Utilities": [
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "NHPC.NS",
        "JSWENERGY.NS", "TORNTPOWER.NS", "ADANIGREEN.NS", "CESC.NS",
        "SJVN.NS",
    ],

    # ── INFRASTRUCTURE & CAPITAL GOODS ───────────────────────
    "Infrastructure & Capital Goods": [
        "LT.NS", "ADANIPORTS.NS", "SIEMENS.NS", "ABB.NS",
        "CUMMINSIND.NS", "BEL.NS", "HAL.NS",
        "IRCTC.NS", "CONCOR.NS",
    ],

    # ── TELECOM & MEDIA ──────────────────────────────────────
    "Telecom & Media": [
        "BHARTIARTL.NS", "IDEA.NS", "TTML.NS",
        "ZEEL.NS",
    ],

    # ── CHEMICALS ────────────────────────────────────────────
    "Chemicals": [
        "PIDILITIND.NS", "UPL.NS", "SRF.NS", "ATUL.NS",
        "DEEPAKNTR.NS", "NAVINFLUOR.NS", "PIIND.NS",
    ],
}

# --- Leverage ---
# Multiply daily portfolio net returns by this factor.
# For market-neutral strategies, leverage is commonly used to amplify returns.
LEVERAGE = 1.0

# --- Time period ---
DATA_START = "2021-01-01"
DATA_END = "2026-04-30"

# Out-of-sample test window in trading days (252 ≈ 1 year).
TEST_PERIOD_DAYS = 252

# --- Strategy parameters ---
# BETA_LOOKBACK_WINDOW: rolling OLS window for hedge ratio β (120 days ≈ 6 months)
# Z_LOOKBACK_WINDOW: rolling window for z-score normalisation (30 days ≈ 1.5 months)
BETA_LOOKBACK_WINDOW = 120
Z_LOOKBACK_WINDOW = 30
LOOKBACK_WINDOW = 60  # Kept for backward compatibility

# Z_ENTRY_THRESHOLD: Enter a trade when the z-score exceeds this.
Z_ENTRY_THRESHOLD = 2.2

# Z_EXIT_THRESHOLD: Close the trade when z-score returns within this.
Z_EXIT_THRESHOLD = 0.5

# Z_STOP_LOSS: Hard stop — if z exceeds this, the cointegration may have broken.
Z_STOP_LOSS = 3.5

# --- Pair selection criteria ---
# ADF_P_VALUE_CUTOFF: Maximum p-value from the ADF test.
ADF_P_VALUE_CUTOFF = 0.05

# HALF_LIFE_MIN / HALF_LIFE_MAX: The spread's mean-reversion speed.
HALF_LIFE_MIN = 5
HALF_LIFE_MAX = 30

# Maximum number of pairs to trade simultaneously.
MAX_PAIRS = 5

# MIN_STABILITY_PCT: Minimum % of rolling windows where pair must be cointegrated.
MIN_STABILITY_PCT = 60.0

# REQUIRE_BOTH_COINT_TESTS: If True, both ADF and Engle-Granger must pass.
REQUIRE_BOTH_COINT_TESTS = True

# SPREAD_VELOCITY_LOOKBACK: Number of days to check whether the spread is reverting.
SPREAD_VELOCITY_LOOKBACK = 2

# PAIR_DRAWDOWN_STOP: Stop trading a pair if it loses more than this amount during a period.
PAIR_DRAWDOWN_STOP = 0.03  # 3%

# --- Cost assumptions ---
# COST_PER_LEG: Total cost for one side of a trade (buy OR sell).
# Includes brokerage, STT, exchange fees, GST.
# 0.05% is realistic for NSE equity delivery.
COST_PER_LEG = 0.0005  # 0.05%

# SLIPPAGE: Price impact — the price moves slightly between
# your decision to trade and the actual execution.
SLIPPAGE = 0.0003  # 0.03%

# --- Risk management ---
# INITIAL_CAPITAL: Starting portfolio value in ₹.
INITIAL_CAPITAL = 100_000  # ₹1,00,000

# What fraction of total capital to allocate to each pair.
# With 6-8 pairs at 12-15%, you keep some cash buffer.
CAPITAL_PER_PAIR_PCT = 0.12  # 12% per pair

# If a trade hasn't reverted in this many × half-life days,
# force-close it. Something fundamental may have changed.
MAX_HOLDING_PERIOD_MULTIPLIER = 2

# --- Data caching ---
DATA_DIR = "data"
CACHE_FILENAME = "nse_prices.parquet"
