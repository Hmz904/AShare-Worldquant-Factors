"""
config.py — Central configuration for the A-share backtest system
"""

# ── Tushare ───────────────────────────────────────────────────────────────────
TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN_HERE"

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE        = "full"          # "full" | "csi300" | "csi500"
INDEX_BENCHMARK = "000300.SH"    # CSI 300 used as benchmark + short hedge
INDEX_HEDGE     = "000300.SH"    # index ETF leg for long-short PnL

# ── Backtest window ───────────────────────────────────────────────────────────
START_DATE = "20180101"
END_DATE   = "20241231"

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N            = 50       # number of long positions
REBAL_FREQ       = "M"      # "D"=daily, "W"=weekly, "M"=monthly
LONG_WEIGHT      = "equal"  # "equal" | "value"  (value = circ_mv weighted)

# ── Execution simulation ──────────────────────────────────────────────────────
COMMISSION_RATE  = 0.0003   # one-way (买卖各 0.03%)
STAMP_DUTY       = 0.0005   # sell-side only (印花税 0.05%)
SLIPPAGE         = 0.002    # assumed price impact (0.2% one-way)

# ── Universe filters (applied every rebal day) ────────────────────────────────
EXCLUDE_ST       = True     # drop *ST / ST stocks
EXCLUDE_SUSPEND  = True     # drop suspended stocks (vol == 0)
EXCLUDE_LIMIT    = True     # drop stocks hitting ±10% limit (can't fill)
MIN_LIST_DAYS    = 60       # drop stocks listed < 60 days
MIN_PRICE        = 1.0      # drop penny stocks (< 1 RMB)

# ── Risk / neutralization ─────────────────────────────────────────────────────
SECTOR_NEUTRAL   = False    # if True, rank within each industry separately
WINSORIZE_STD    = 3.0      # clip factor values at ±3σ before ranking

# ── Data cache ────────────────────────────────────────────────────────────────
CACHE_DIR        = "./cache"   # local parquet cache to avoid repeated API calls
USE_CACHE        = True

# ── Robustness ────────────────────────────────────────────────────────────────
SUBPERIOD_YEARS  = 2        # split backtest into N-year sub-periods
DECAY_MAX_LAG    = 10       # max lag (days) for IC decay test
