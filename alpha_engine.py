"""
alpha_engine.py — Translates all 25 WorldQuant alphas to pandas
================================================================
Each alpha_XX() function receives a `data` dict containing all panels
(trade_date × ts_code DataFrames) and returns a single signal DataFrame.

Panels available in data{}:
  Price   : open, high, low, close, adj_close, pre_close, returns, pct_chg
  Volume  : volume, adv20, amount, vwap, turnover_rate
  Cap     : cap (circ_mv), total_mv
  Factor  : pe_ttm, pb, ps_ttm
  Fin     : revenue, operate_profit, n_income, oper_cost, total_assets,
            total_liab, total_cur_liab, total_cur_assets, goodwill,
            lt_borr, st_borr, capex, n_cashflow_act
            (derived) assets, liabilities, debt, sales, cogs
  Meta    : industry_map (pd.Series: ts_code → industry string)

Alphas with unavailable data are marked SKIP and return None.
"""

import numpy as np
import pandas as pd
from wq_operators import *


def _d(data, key, default=None):
    """Safe panel getter."""
    v = data.get(key, default)
    if v is None and default is not None:
        return default
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Alpha catalogue
# ══════════════════════════════════════════════════════════════════════════════

def alpha_01(data: dict) -> pd.DataFrame | None:
    """
    rank(ts_sum(vec_avg(nws12_afterhsz_sl), 25)) > 0.4 ? 1 :
        0.7 * rank(-ts_delta(close, 10)) + 0.3 * rank(ts_mean(cap, 1800))
    SKIP: nws12_afterhsz_sl (WQ proprietary news NLP vector) not in Tushare.
    APPROXIMATION: replace news sentiment with rank of ts_mean(turnover_rate,25)
    as a momentum-of-attention proxy.
    """
    close  = _d(data, "adj_close")
    cap    = _d(data, "cap")
    tover  = _d(data, "turnover_rate")

    news_proxy = rank(ts_mean(tover, 25))
    cond       = news_proxy > 0.4
    leg_a      = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    leg_b      = 0.7 * rank(-delta(close, 10)) + 0.3 * rank(ts_mean(cap, 1800))
    return if_else(cond, leg_a, leg_b)


def alpha_02(data: dict) -> pd.DataFrame | None:
    """
    alpha_raw = winsorize(ts_backfill(liabilities/assets, 60))
                  * ts_arg_min(debt/equity, 250)
    alpha_neut = group_neutralize(alpha_raw, industry)
    final_signal = trade_when(1, alpha_neut, 0)
    Note: equity ≈ assets - liabilities
    """
    assets  = _d(data, "assets")
    liab    = _d(data, "liabilities")
    debt    = _d(data, "debt")
    ind     = _d(data, "industry_map")
    if assets is None or liab is None:
        return None

    equity     = assets - liab
    lev_ratio  = ts_backfill(liab / assets.replace(0, np.nan), 60)
    d_e        = debt / equity.replace(0, np.nan)
    alpha_raw  = winsorize(lev_ratio) * ts_arg_min(d_e, 250)
    alpha_neut = group_neutralize(alpha_raw, ind)
    return trade_when(1, alpha_neut, 0)


def alpha_03(data: dict) -> pd.DataFrame | None:
    """
    rank(anl4_cff_flag) - rank(assets/liabilities_curr)
    anl4_cff_flag: analyst earnings revision flag — approx with n_income change YoY
    """
    assets      = _d(data, "assets")
    cur_liab    = _d(data, "total_cur_liab")
    n_income    = _d(data, "n_income")
    if assets is None or cur_liab is None:
        return None

    # analyst revision proxy: YoY net income change direction
    anl_proxy = sign(n_income - ts_backfill(n_income.shift(252), 63)) if n_income is not None else None
    if anl_proxy is None:
        return None

    return rank(anl_proxy) - rank(assets / cur_liab.replace(0, np.nan))


def alpha_04(data: dict) -> pd.DataFrame | None:
    """
    trade_when(pcr_oi_180 < 1,
               implied_volatility_call_180 - implied_volatility_put_180, -1)
    SKIP: options data (pcr_oi, IV) not available in Tushare for full A-share.
    """
    return None   # options data unavailable


def alpha_05(data: dict) -> pd.DataFrame | None:
    """
    rank(-ts_sum(group_mean(returns, volume, sector), 22))
    * rank(-ts_delta(ts_decay_linear(close, 5), 1))
    """
    returns = _d(data, "returns")
    volume  = _d(data, "volume")
    close   = _d(data, "adj_close")
    ind     = _d(data, "industry_map")

    gm   = group_mean(returns, volume, ind)
    leg1 = rank(-ts_sum(gm, 22))
    leg2 = rank(-delta(ts_decay_linear(close, 5), 1))
    return leg1 * leg2


def alpha_06(data: dict) -> pd.DataFrame | None:
    """
    a = ts_backfill(fnd6_acqgdwl, 63) / ts_backfill(sales, 63)
      ≈ goodwill / sales  (fnd6_acqgdwl ≈ goodwill from acquisitions)
    aa = -ts_regression(a, ts_product(returns/close+1, 63)-1, 612, lag=1, rettype=3)
    vol_regime = ts_std_dev(returns,20) / ts_std_dev(returns,252)
    zscore1 = group_zscore(winsorize(ts_backfill(aa,63), std=3), industry)
    weight1 = (1 - 0.25*max(0, vol_regime-0.6)) * abs(zscore1)
    weight  = sign(zscore1) * min(weight1, 3)
    trade_when(volume < adv20 && ts_mean(returns,5) < 0 && vol_regime < 0.85,
               weight, 0)
    """
    goodwill = _d(data, "goodwill")
    sales    = _d(data, "sales")
    returns  = _d(data, "returns")
    close    = _d(data, "adj_close")
    volume   = _d(data, "volume")
    adv20    = _d(data, "adv20")
    ind      = _d(data, "industry_map")
    if goodwill is None or sales is None:
        return None

    a   = ts_backfill(goodwill, 63) / ts_backfill(sales, 63).replace(0, np.nan)
    rp  = ts_product(returns / close.replace(0, np.nan) + 1, 63) - 1
    aa  = -ts_regression(a, rp, 612, lag=1, rettype=3)

    vol_regime = ts_std_dev(returns, 20) / ts_std_dev(returns, 252).replace(0, np.nan)
    zscore1    = group_zscore(winsorize(ts_backfill(aa, 63), std=3), ind)
    weight1    = (1 - 0.25 * max_val(pd.DataFrame(0, index=aa.index, columns=aa.columns),
                                     vol_regime - 0.6)) * abs_val(zscore1)
    weight     = sign(zscore1) * min_val(weight1, 3)

    cond = (volume < adv20) & (ts_mean(returns, 5) < 0) & (vol_regime < 0.85)
    return trade_when(cond, weight, 0)


def alpha_07(data: dict) -> pd.DataFrame | None:
    """
    scale((ts_sum(close,7)/7 - close))
    + 20 * scale(ts_corr(vwap, ts_delay(close,5), 230))
    """
    close = _d(data, "adj_close")
    vwap  = _d(data, "vwap")

    leg1 = scale(ts_mean(close, 7) - close)
    leg2 = 20 * scale(ts_corr(vwap, delay(close, 5), 230))
    return leg1 + leg2


def alpha_08(data: dict) -> pd.DataFrame | None:
    """
    r = ts_regression(close, cap, 220, lag=25, rettype=6)
    s = sign(group_neutralize(ts_sum(returns, 4), industry))
    -r * s
    """
    close   = _d(data, "adj_close")
    cap     = _d(data, "cap")
    returns = _d(data, "returns")
    ind     = _d(data, "industry_map")

    r = ts_regression(close, cap, 220, lag=25, rettype=6)
    s = sign(group_neutralize(ts_sum(returns, 4), ind))
    return -r * s


def alpha_09(data: dict) -> pd.DataFrame | None:
    """
    decay_days = 1
    rel_days_since_max = rank(ts_arg_max(close, 30))
    decline_pct = (vwap - close) / close
    decline_pct / min(ts_decay_linear(rel_days_since_max, 1), 0.15)
    """
    close = _d(data, "adj_close")
    vwap  = _d(data, "vwap")

    rel_days_since_max = rank(ts_arg_max(close, 30))
    decline_pct        = (vwap - close) / close.replace(0, np.nan)
    denom              = min_val(ts_decay_linear(rel_days_since_max, 1), 0.15)
    return decline_pct / denom.replace(0, np.nan)


def alpha_10(data: dict) -> pd.DataFrame | None:
    """
    Bollinger Band signal:
    tp = (high+low+close)/3
    MA = ts_mean(tp, 20); BOLU = MA + std; BOLD = MA - std
    signal = ts_sum(close<BOLD,20) - ts_sum(close>BOLU,20)
    ts_zscore(signal, 240)
    """
    high  = _d(data, "high")
    low   = _d(data, "low")
    close = _d(data, "adj_close")
    window = 20

    tp    = (high + low + close) / 3
    ma    = ts_mean(tp, window)
    std   = ts_std_dev(tp, window)
    bolu  = ma + std
    bold  = ma - std
    sig   = ts_sum((close < bold).astype(float), window) - \
            ts_sum((close > bolu).astype(float), window)
    return ts_zscore(sig, 240)


def alpha_11(data: dict) -> pd.DataFrame | None:
    """
    intra_ret = close/open - 1
    mean_returns = group_mean(intra_ret, rank(ts_mean(cap,20)), market)
    horro = abs(intra_ret - mean_returns) / (abs(intra_ret)+abs(mean_returns)+0.1)
    horro_day = ts_mean(horro, 22)
    ret_std = ts_std_dev(intra_ret, 22)
    adj_ret = horro_day * ret_std * intra_ret
    horro_std_bonus = zscore(ts_mean(adj_ret,22)) + zscore(ts_std_dev(adj_ret,22))
    -horro_std_bonus
    Note: 'market' group = treat all stocks as one group (full universe mean)
    """
    close = _d(data, "adj_close")
    open_ = _d(data, "open")
    cap   = _d(data, "cap")
    ind   = _d(data, "industry_map")

    intra_ret    = close / open_.replace(0, np.nan) - 1
    cap_bucket   = rank(ts_mean(cap, 20))
    mean_returns = group_mean(intra_ret, cap_bucket, ind)   # market-level approx

    horro     = abs_val(intra_ret - mean_returns) / \
                (abs_val(intra_ret) + abs_val(mean_returns) + 0.1)
    horro_day = ts_mean(horro, 22)
    ret_std   = ts_std_dev(intra_ret, 22)
    adj_ret   = horro_day * ret_std * intra_ret

    horro_std_bonus = zscore(ts_mean(adj_ret, 22)) + zscore(ts_std_dev(adj_ret, 22))
    return -horro_std_bonus


def alpha_12(data: dict) -> pd.DataFrame | None:
    """
    night = open/delay(close,1) - 1
    drop_hc = high/close - 1
    rise_lc = low/close - 1
    rise_ho = high/open - 1
    drop_lo = low/open - 1
    over_drop = |drop_hc| + |drop_lo|
    over_rise = |rise_lc| + |rise_ho|
    if_else(sign(night) > 0, over_drop, -over_rise)
    """
    close = _d(data, "adj_close")
    open_ = _d(data, "open")
    high  = _d(data, "high")
    low   = _d(data, "low")

    night     = open_ / delay(close, 1).replace(0, np.nan) - 1
    drop_hc   = high / close.replace(0, np.nan) - 1
    rise_lc   = low  / close.replace(0, np.nan) - 1
    rise_ho   = high / open_.replace(0, np.nan) - 1
    drop_lo   = low  / open_.replace(0, np.nan) - 1
    over_drop = abs_val(drop_hc) + abs_val(drop_lo)
    over_rise = abs_val(rise_lc) + abs_val(rise_ho)
    return if_else(sign(night) > 0, over_drop, -over_rise)


def alpha_13(data: dict) -> pd.DataFrame | None:
    """
    group_rank(fnd6_newa2v1300_rdipeps / (cap * return_equity), subindustry)
    fnd6_newa2v1300_rdipeps ≈ net income / shares (EPS from financial data)
    return_equity ≈ n_income / equity (ROE proxy)
    subindustry → use industry (SW L1) as approximation
    """
    n_income = _d(data, "n_income")
    cap      = _d(data, "cap")
    assets   = _d(data, "assets")
    liab     = _d(data, "liabilities")
    ind      = _d(data, "industry_map")
    if n_income is None or cap is None:
        return None

    equity  = (assets - liab).replace(0, np.nan)
    roe     = n_income / equity
    eps_cap = n_income / (cap.replace(0, np.nan) * roe.replace(0, np.nan))
    return group_rank(eps_cap, ind)


def alpha_14(data: dict) -> pd.DataFrame | None:
    """
    ts_rank(-rank(enterprise_value)/rank(ebitda), 5)
    EV ≈ total_mv + debt - cash (cash not in Tushare daily; use total_mv + debt)
    EBITDA ≈ operate_profit + depreciation (depreciation not available → use operate_profit)
    """
    total_mv  = _d(data, "total_mv")
    debt      = _d(data, "debt")
    op_profit = _d(data, "operate_profit")
    if total_mv is None or op_profit is None:
        return None

    ev    = total_mv + debt.fillna(0)
    ebitda = op_profit   # approximation
    return ts_rank(-rank(ev) / rank(ebitda.replace(0, np.nan)), 5)


def alpha_15(data: dict) -> pd.DataFrame | None:
    """
    compound_return = power(1 + returns, 12) - 1
    Annualised monthly return approximation.
    """
    returns = _d(data, "returns")
    return power(1 + returns, 12) - 1


def alpha_16(data: dict) -> pd.DataFrame | None:
    """
    a = ts_mean(returns / ts_delay(group_mean(returns, if_else(rank(ts_sum(returns,66))<0.25,1,0), industry), 66), 1000)
    b = ts_regression(returns, a, 1000)
    c = -b / ts_std_dev(b, 1000)
    c
    Note: very long lookback (1000 days ≈ 4 years). Only meaningful with 5+ years of data.
    """
    returns = _d(data, "returns")
    ind     = _d(data, "industry_map")

    loser_flag = if_else(rank(ts_sum(returns, 66)) < 0.25,
                         pd.DataFrame(1.0, index=returns.index, columns=returns.columns),
                         pd.DataFrame(0.0, index=returns.index, columns=returns.columns))
    gm_lagged  = delay(group_mean(returns, loser_flag, ind), 66)
    a          = ts_mean(returns / gm_lagged.replace(0, np.nan), 1000)
    b          = ts_regression(returns, a, 1000)
    return -b / ts_std_dev(b, 1000).replace(0, np.nan)


def alpha_17(data: dict) -> pd.DataFrame | None:
    """
    ts_regression(close, cap, 220, lag=5, rettype=6)
    Residual of price regressed on market cap (value-size decomposition).
    """
    close = _d(data, "adj_close")
    cap   = _d(data, "cap")
    return ts_regression(close, cap, 220, lag=5, rettype=6)


def alpha_18(data: dict) -> pd.DataFrame | None:
    """
    -ts_backfill(zscore(goodwill/sales), 65)
     * (rank(accrued_liab)*rank(capex)*rank(dividend/sharesout) + rank(debt_st))
    Approximation: accrued_liab ≈ total_cur_liab - st_borr
                   dividend/sharesout → dv_ratio proxy (pb/pe)
                   debt_st → st_borr
    """
    goodwill = _d(data, "goodwill")
    sales    = _d(data, "sales")
    cur_liab = _d(data, "total_cur_liab")
    st_borr  = _d(data, "st_borr")
    capex    = _d(data, "capex")
    pb       = _d(data, "pb")
    pe       = _d(data, "pe_ttm")
    if goodwill is None or sales is None:
        return None

    accrued_liab = (cur_liab - st_borr.fillna(0)).clip(lower=0)
    div_proxy    = pb / pe.replace(0, np.nan)   # approx dividend yield signal
    debt_st      = st_borr.fillna(0)

    base = -ts_backfill(zscore(goodwill / sales.replace(0, np.nan)), 65)
    combo = (rank(accrued_liab) * rank(capex.abs()) * rank(div_proxy) + rank(debt_st))
    return base * combo


def alpha_19(data: dict) -> pd.DataFrame | None:
    """
    rt = power(fnd6_intc, 2.5)     (interest cost — approx with operate_profit)
    b = ts_mean(rt, 22)
    o = ts_std_dev(rt, 12)
    drt = -(ts_delta(b-rt, 20)/22) + o*ts_delta(sqrt(rt), 8)/8
          * quantile(returns, driver='gaussian', sigma=1)
    alpha = group_backfill(group_rank(drt, sector), market, 350)
    trade_when(volume > 0.3*adv20, alpha, -1)
    Note: fnd6_intc ≈ operate_profit (interest / financing cost proxy)
    """
    op_profit = _d(data, "operate_profit")
    returns   = _d(data, "returns")
    volume    = _d(data, "volume")
    adv20     = _d(data, "adv20")
    ind       = _d(data, "industry_map")
    if op_profit is None:
        return None

    rt   = power(abs_val(op_profit), 2.5)
    b    = ts_mean(rt, 22)
    o    = ts_std_dev(rt, 12)
    drt  = -(delta(b - rt, 20) / 22) + \
            o * delta(sqrt(rt), 8) / 8 * quantile(returns, driver="gaussian", sigma=1)

    alpha_raw = group_backfill(group_rank(drt, ind), None, 350)
    cond      = volume > 0.3 * adv20
    return trade_when(cond, alpha_raw, -1)


def alpha_20(data: dict) -> pd.DataFrame | None:
    """
    avg_news = vec_avg(nws12_afterhsz_sl)
    rank(ts_sum(avg_news, 60)) > 0.5 ? 1 : rank(-ts_delta(close, 2))
    APPROXIMATION: same news proxy as alpha_01 (turnover_rate momentum)
    """
    close = _d(data, "adj_close")
    tover = _d(data, "turnover_rate")

    news_proxy = rank(ts_sum(tover, 60))
    cond       = news_proxy > 0.5
    leg_a      = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    leg_b      = rank(-delta(close, 2))
    return if_else(cond, leg_a, leg_b)


def alpha_21(data: dict) -> pd.DataFrame | None:
    """
    rank(volume / (ts_sum(volume,60)/60) * rank(ts_sum(close,5)/5 / close))
    Volume ratio × price momentum.
    """
    close  = _d(data, "adj_close")
    volume = _d(data, "volume")

    vol_ratio = volume / ts_mean(volume, 60).replace(0, np.nan)
    price_mom = rank(ts_mean(close, 5) / close.replace(0, np.nan))
    return rank(vol_ratio * price_mom)


def alpha_22(data: dict) -> pd.DataFrame | None:
    """
    positive_days = ts_sum(if_else(volume > adv20, 1, 0), 250)
    trade_when(returns > ts_std_dev(returns, 20), positive_days, -1)
    """
    returns = _d(data, "returns")
    volume  = _d(data, "volume")
    adv20   = _d(data, "adv20")

    ones = pd.DataFrame(1.0, index=volume.index, columns=volume.columns)
    zeros = pd.DataFrame(0.0, index=volume.index, columns=volume.columns)
    pos_days = ts_sum(if_else(volume > adv20, ones, zeros), 250)
    cond     = returns > ts_std_dev(returns, 20)
    return trade_when(cond, pos_days, -1)


def alpha_23(data: dict) -> pd.DataFrame | None:
    """
    ts_rank(operating_income/cap, 250)
    """
    op_income = _d(data, "operate_profit")
    cap       = _d(data, "cap")
    if op_income is None:
        return None
    return ts_rank(op_income / cap.replace(0, np.nan), 250)


def alpha_24(data: dict) -> pd.DataFrame | None:
    """
    positive_days = ts_sum(volume/vwap > ts_mean(volume/vwap, 20) ? 1 : 0, 250)
    trade_when(returns > 0, positive_days, -1)
    """
    returns = _d(data, "returns")
    volume  = _d(data, "volume")
    vwap    = _d(data, "vwap")

    ratio    = volume / vwap.replace(0, np.nan)
    ones     = pd.DataFrame(1.0, index=volume.index, columns=volume.columns)
    zeros    = pd.DataFrame(0.0, index=volume.index, columns=volume.columns)
    pos_days = ts_sum(if_else(ratio > ts_mean(ratio, 20), ones, zeros), 250)
    return trade_when(returns > 0, pos_days, -1)


def alpha_25(data: dict) -> pd.DataFrame | None:
    """
    gpa = (revenue - cogs) / assets
    gpa_diff = ts_av_diff(gpa, 60)
    group_neutralize(gpa_diff, bucket(rank(cap), range='0.1,1,0.1'))
    Note: bucket by cap decile, then neutralize within cap bucket
    """
    revenue = _d(data, "revenue")
    cogs    = _d(data, "cogs")
    assets  = _d(data, "assets")
    cap     = _d(data, "cap")
    if revenue is None or cogs is None or assets is None:
        return None

    gpa      = (revenue - cogs) / assets.replace(0, np.nan)
    gpa_diff = ts_av_diff(gpa, 60)
    cap_bkt  = bucket(rank(cap), n=10)

    # neutralize within cap decile bucket
    cap_map  = cap_bkt.iloc[-1].dropna().astype(int).astype(str)   # use latest bucket as map
    return group_neutralize(gpa_diff, cap_map)


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

ALPHA_REGISTRY = {
    "alpha_01": (alpha_01, "News momentum proxy + price/cap combo"),
    "alpha_02": (alpha_02, "Leverage ratio × debt/equity min — industry neutral"),
    "alpha_03": (alpha_03, "Analyst revision proxy - current ratio"),
    "alpha_04": (alpha_04, "SKIP — options PCR/IV data unavailable"),
    "alpha_05": (alpha_05, "Sector return mean reversion × price decay"),
    "alpha_06": (alpha_06, "Goodwill/sales regression + vol regime filter"),
    "alpha_07": (alpha_07, "MA deviation + VWAP-price correlation"),
    "alpha_08": (alpha_08, "Price-cap regression residual × sector momentum"),
    "alpha_09": (alpha_09, "Intraday decline / decay of days-from-high"),
    "alpha_10": (alpha_10, "Bollinger Band signal z-scored"),
    "alpha_11": (alpha_11, "Heteroskedastic return anomaly (HORRO)"),
    "alpha_12": (alpha_12, "Overnight gap × intraday range asymmetry"),
    "alpha_13": (alpha_13, "EPS/cap/ROE group rank"),
    "alpha_14": (alpha_14, "EV/EBITDA ts_rank"),
    "alpha_15": (alpha_15, "Compounded return (annualised)"),
    "alpha_16": (alpha_16, "Loser-stock return beta (1000-day)"),
    "alpha_17": (alpha_17, "Price-cap regression residual (lag 5)"),
    "alpha_18": (alpha_18, "Goodwill/sales z-score × accruals combo"),
    "alpha_19": (alpha_19, "Profit volatility dynamics + vol filter"),
    "alpha_20": (alpha_20, "News momentum proxy + short-term reversal"),
    "alpha_21": (alpha_21, "Volume ratio × price momentum rank"),
    "alpha_22": (alpha_22, "High-volume day count, returns filter"),
    "alpha_23": (alpha_23, "Operating income yield ts_rank"),
    "alpha_24": (alpha_24, "Positive buy-pressure days, trend filter"),
    "alpha_25": (alpha_25, "GPA momentum, cap-bucket neutral"),
}


def compute_alpha(name: str, data: dict) -> pd.DataFrame | None:
    """Compute a single alpha by name. Returns None if unavailable."""
    if name not in ALPHA_REGISTRY:
        raise ValueError(f"Unknown alpha: {name}")
    fn, desc = ALPHA_REGISTRY[name]
    try:
        sig = fn(data)
        if sig is not None:
            sig = winsorize(sig)   # always winsorize raw signal
        return sig
    except Exception as e:
        import traceback
        print(f"[{name}] ERROR: {e}")
        traceback.print_exc()
        return None


def compute_all_alphas(data: dict) -> dict[str, pd.DataFrame]:
    """Compute all available alphas. Returns dict of name → signal DataFrame."""
    results = {}
    for name, (fn, desc) in ALPHA_REGISTRY.items():
        print(f"  Computing {name}: {desc}")
        sig = compute_alpha(name, data)
        if sig is not None:
            results[name] = sig
        else:
            print(f"    → SKIPPED (data unavailable)")
    return results
