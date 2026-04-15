"""
data_loader.py — Tushare data pipeline with local parquet cache
================================================================
Pulls and caches:
  • Daily OHLCV + adj_factor  →  price panel
  • daily_basic               →  cap, turnover, valuation panel
  • stock_basic + namechange  →  universe meta (ST flag, list_date, industry)
  • index_daily               →  benchmark returns
  • Financial statements      →  income / balance / cashflow (quarterly, backfilled)

All panels are indexed as  (trade_date × ts_code)  DataFrames.
"""

import os
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── init tushare ──────────────────────────────────────────────────────────────
ts.set_token(config.TUSHARE_TOKEN)
pro = ts.pro_api()

Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    return Path(config.CACHE_DIR) / f"{name}.parquet"


def _load_or_fetch(name: str, fetch_fn, force_refresh=False):
    """Return cached parquet if exists, else call fetch_fn() and cache result."""
    p = _cache_path(name)
    if config.USE_CACHE and p.exists() and not force_refresh:
        log.info(f"Cache hit  → {name}")
        return pd.read_parquet(p)
    log.info(f"Fetching   → {name}")
    df = fetch_fn()
    df.to_parquet(p, index=True)
    return df


def _api_call(fn, max_retries=5, sleep=0.4, **kwargs):
    """Tushare API call with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            result = fn(**kwargs)
            if result is not None and len(result) > 0:
                return result
            return pd.DataFrame()
        except Exception as e:
            if "每分钟" in str(e) or "limit" in str(e).lower() or attempt < max_retries - 1:
                log.warning(f"API rate limit / error ({e}), retry {attempt+1}/{max_retries}")
                time.sleep(sleep * (attempt + 1))
            else:
                raise
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Stock universe metadata
# ══════════════════════════════════════════════════════════════════════════════

def load_stock_meta(force_refresh=False) -> pd.DataFrame:
    """
    Returns DataFrame indexed by ts_code with columns:
      name, area, industry, market, list_date, delist_date (None if still listed)
    """
    def _fetch():
        listed   = _api_call(pro.stock_basic, exchange="", list_status="L",
                             fields="ts_code,name,area,industry,market,list_date")
        delisted = _api_call(pro.stock_basic, exchange="", list_status="D",
                             fields="ts_code,name,area,industry,market,list_date,delist_date")
        df = pd.concat([listed, delisted], ignore_index=True)
        df["list_date"]   = pd.to_datetime(df["list_date"],   format="%Y%m%d", errors="coerce")
        df["delist_date"] = pd.to_datetime(df.get("delist_date"), format="%Y%m%d", errors="coerce") if "delist_date" in df.columns else pd.NaT
        df = df.drop_duplicates("ts_code").set_index("ts_code")
        return df

    return _load_or_fetch("stock_meta", _fetch, force_refresh)


def load_st_history(force_refresh=False) -> pd.DataFrame:
    """
    Returns long-format DataFrame: ts_code, start_date, end_date, is_st
    Used to flag ST stocks on any given date.
    """
    def _fetch():
        meta   = load_stock_meta()
        codes  = meta.index.tolist()
        chunks = [codes[i:i+100] for i in range(0, len(codes), 100)]
        rows   = []
        for chunk in chunks:
            for code in chunk:
                df = _api_call(pro.namechange, ts_code=code,
                               fields="ts_code,name,start_date,end_date")
                if not df.empty:
                    df["is_st"] = df["name"].str.contains(r"\*?ST", na=False)
                    rows.append(df[df["is_st"]])
                time.sleep(0.05)
        if not rows:
            return pd.DataFrame(columns=["ts_code","name","start_date","end_date","is_st"])
        out = pd.concat(rows, ignore_index=True)
        out["start_date"] = pd.to_datetime(out["start_date"], format="%Y%m%d", errors="coerce")
        out["end_date"]   = pd.to_datetime(out["end_date"],   format="%Y%m%d", errors="coerce")
        return out

    return _load_or_fetch("st_history", _fetch, force_refresh)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Trade calendar
# ══════════════════════════════════════════════════════════════════════════════

def load_trade_cal(start=config.START_DATE, end=config.END_DATE) -> pd.DatetimeIndex:
    def _fetch():
        df = _api_call(pro.trade_cal, exchange="SSE",
                       start_date=start, end_date=end, is_open="1")
        df["cal_date"] = pd.to_datetime(df["cal_date"], format="%Y%m%d")
        return df[["cal_date"]].sort_values("cal_date").reset_index(drop=True)

    key = f"trade_cal_{start}_{end}"
    df  = _load_or_fetch(key, _fetch)
    return pd.DatetimeIndex(df["cal_date"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. Daily price panel  (trade_date × ts_code)
# ══════════════════════════════════════════════════════════════════════════════

def load_price_panel(start=config.START_DATE, end=config.END_DATE,
                     force_refresh=False) -> dict[str, pd.DataFrame]:
    """
    Returns dict of DataFrames, each shaped (trade_date × ts_code):
      open, high, low, close, adj_close, volume, amount, vwap, returns,
      pct_chg, adj_factor, pre_close
    """
    key = f"price_{start}_{end}"

    def _fetch():
        cal   = load_trade_cal(start, end)
        dates = [d.strftime("%Y%m%d") for d in cal]
        all_daily = []
        all_adj   = []
        all_basic = []

        for i, d in enumerate(dates):
            if i % 50 == 0:
                log.info(f"  price panel  {i}/{len(dates)}  {d}")
            daily = _api_call(pro.daily,       trade_date=d)
            adj   = _api_call(pro.adj_factor,  trade_date=d)
            basic = _api_call(pro.daily_basic, trade_date=d)
            if not daily.empty:
                all_daily.append(daily)
            if not adj.empty:
                all_adj.append(adj)
            if not basic.empty:
                all_basic.append(basic)
            time.sleep(0.06)   # ~16 calls/sec, safe under Pro limit

        daily_df = pd.concat(all_daily, ignore_index=True)
        adj_df   = pd.concat(all_adj,   ignore_index=True)
        basic_df = pd.concat(all_basic, ignore_index=True)

        df = (daily_df
              .merge(adj_df,   on=["ts_code","trade_date"], how="left")
              .merge(basic_df[["ts_code","trade_date","circ_mv","total_mv",
                                "turnover_rate","volume_ratio",
                                "pe_ttm","pb","ps_ttm"]],
                     on=["ts_code","trade_date"], how="left"))

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df["adj_factor"] = df["adj_factor"].fillna(1.0)
        df["adj_close"]  = df["close"] * df["adj_factor"]
        df["vwap"]       = df["amount"] / df["vol"].replace(0, np.nan) * 100
        return df

    raw = _load_or_fetch(key, _fetch, force_refresh)

    # pivot to (date × code) panels
    raw = raw.sort_values(["trade_date","ts_code"])
    idx = "trade_date"

    panels = {}
    for col in ["open","high","low","close","adj_close","vol",
                "amount","vwap","pct_chg","adj_factor","pre_close",
                "circ_mv","total_mv","turnover_rate","volume_ratio",
                "pe_ttm","pb","ps_ttm"]:
        if col in raw.columns:
            panels[col] = raw.pivot(index=idx, columns="ts_code", values=col)

    # derived
    panels["returns"] = panels["adj_close"].pct_change()
    panels["volume"]  = panels["vol"]
    panels["adv20"]   = panels["vol"].rolling(20, min_periods=5).mean()
    panels["cap"]     = panels["circ_mv"]   # WQ 'cap' = float mktcap (万元)

    return panels


# ══════════════════════════════════════════════════════════════════════════════
# 4. Financial statement panel (quarterly → daily backfilled)
# ══════════════════════════════════════════════════════════════════════════════

def load_financial_panel(start=config.START_DATE, end=config.END_DATE,
                         force_refresh=False) -> dict[str, pd.DataFrame]:
    """
    Pulls income / balance / cashflow statements, forward-fills to daily.
    Returns dict of (trade_date × ts_code) panels for fundamental fields.
    """
    key = f"financial_{start}_{end}"

    def _fetch():
        # periods covered: give extra buffer (2 years before start)
        s = str(int(start[:4]) - 2) + "0101"
        periods = []
        y = int(s[:4])
        while y <= int(end[:4]):
            for q in ["0331","0630","0930","1231"]:
                periods.append(f"{y}{q}")
            y += 1
        periods = [p for p in periods if p <= end.replace("-","")]

        inc_rows, bal_rows, cf_rows = [], [], []
        for period in periods:
            log.info(f"  financials  {period}")
            inc = _api_call(pro.income,
                fields="ts_code,ann_date,end_date,revenue,operate_profit,n_income,oper_cost",
                period=period)
            bal = _api_call(pro.balancesheet,
                fields="ts_code,ann_date,end_date,total_assets,total_liab,"
                       "total_cur_liab,total_cur_assets,goodwill,lt_borr,st_borr,"
                       "accounts_receiv,inventories",
                period=period)
            cf  = _api_call(pro.cashflow,
                fields="ts_code,ann_date,end_date,n_cashflow_act,capex",
                period=period)
            for rows, df in [(inc_rows, inc),(bal_rows, bal),(cf_rows, cf)]:
                if not df.empty:
                    rows.append(df)
            time.sleep(0.15)

        def _combine(rows, date_col="ann_date"):
            if not rows:
                return pd.DataFrame()
            df = pd.concat(rows, ignore_index=True)
            df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.sort_values([date_col, "ts_code"]).drop_duplicates(
                subset=["ts_code", "end_date"], keep="last")
            return df

        return {"income": _combine(inc_rows),
                "balance": _combine(bal_rows),
                "cashflow": _combine(cf_rows)}

    raw = _load_or_fetch(key, _fetch, force_refresh)

    trade_cal = load_trade_cal(start, end)
    trade_dates = pd.DatetimeIndex(trade_cal)

    def _to_panel(df, value_col, date_col="ann_date"):
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(index=trade_dates)
        piv = df.pivot_table(index=date_col, columns="ts_code",
                             values=value_col, aggfunc="last")
        piv = piv.reindex(trade_dates, method="ffill")
        return piv

    fin = {}
    inc = raw["income"]
    bal = raw["balance"]
    cf  = raw["cashflow"]

    # income
    for col in ["revenue","operate_profit","n_income","oper_cost"]:
        fin[col] = _to_panel(inc, col)
    # balance
    for col in ["total_assets","total_liab","total_cur_liab","total_cur_assets",
                "goodwill","lt_borr","st_borr","accounts_receiv","inventories"]:
        fin[col] = _to_panel(bal, col)
    # cashflow
    for col in ["n_cashflow_act","capex"]:
        fin[col] = _to_panel(cf, col)

    # derived combos
    fin["assets"]      = fin["total_assets"]
    fin["liabilities"] = fin["total_liab"]
    fin["debt"]        = fin["lt_borr"].add(fin["st_borr"], fill_value=0)
    fin["sales"]       = fin["revenue"]
    fin["cogs"]        = fin["oper_cost"]

    return fin


# ══════════════════════════════════════════════════════════════════════════════
# 5. Index daily (benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def load_index(ts_code=config.INDEX_BENCHMARK,
               start=config.START_DATE, end=config.END_DATE) -> pd.DataFrame:
    def _fetch():
        df = _api_call(pro.index_daily, ts_code=ts_code,
                       start_date=start, end_date=end)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").set_index("trade_date")
        df["returns"] = df["close"].pct_change()
        return df

    return _load_or_fetch(f"index_{ts_code}_{start}_{end}", _fetch)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Universe filter mask  (trade_date × ts_code  →  bool)
# ══════════════════════════════════════════════════════════════════════════════

def build_universe_mask(panels: dict, meta: pd.DataFrame,
                        st_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Returns boolean DataFrame (True = stock is tradeable that day).
    Filters: not ST, not suspended, not limit-hit, listed >= MIN_LIST_DAYS,
             price >= MIN_PRICE.
    """
    ref  = panels["close"]
    dates = ref.index
    codes = ref.columns

    mask = pd.DataFrame(True, index=dates, columns=codes)

    # 1. price > MIN_PRICE
    mask &= ref >= config.MIN_PRICE

    # 2. not suspended (vol == 0)
    if config.EXCLUDE_SUSPEND:
        mask &= panels["volume"] > 0

    # 3. not hitting limit up/down  (pct_chg ≈ ±10%)
    if config.EXCLUDE_LIMIT:
        pct = panels["pct_chg"].abs()
        mask &= pct < 9.8

    # 4. listed at least MIN_LIST_DAYS
    for code in codes:
        if code in meta.index:
            list_dt = meta.loc[code, "list_date"]
            if pd.notna(list_dt):
                mask.loc[dates < (list_dt + pd.Timedelta(days=config.MIN_LIST_DAYS)), code] = False

    # 5. ST flag
    if config.EXCLUDE_ST and not st_hist.empty:
        for _, row in st_hist.iterrows():
            code = row["ts_code"]
            if code not in codes:
                continue
            s = row["start_date"]
            e = row["end_date"] if pd.notna(row["end_date"]) else dates[-1]
            mask.loc[(dates >= s) & (dates <= e), code] = False

    return mask
