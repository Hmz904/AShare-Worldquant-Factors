"""
stats.py — Performance metrics + robustness testing
====================================================
Functions:
  performance_stats()  — full tearsheet metrics
  ic_series()          — daily IC (rank correlation of signal vs next-day return)
  ic_summary()         — IC mean, ICIR, IC>0 hit rate
  ic_decay()           — IC at lag 1..N
  quintile_returns()   — mean return by signal quintile
  subperiod_stats()    — rolling sub-period Sharpe / IC
  param_sensitivity()  — sweep a parameter and compare Sharpe
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import config


# ══════════════════════════════════════════════════════════════════════════════
# Core performance metrics
# ══════════════════════════════════════════════════════════════════════════════

def performance_stats(returns: pd.Series, benchmark: pd.Series = None,
                      ann_factor: int = 252) -> dict:
    """
    Full set of performance statistics.
    Returns a dict of metric → value.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # annualised return
    total_ret  = (1 + r).prod() - 1
    n_years    = len(r) / ann_factor
    ann_ret    = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # volatility
    ann_vol    = r.std() * np.sqrt(ann_factor)

    # Sharpe (risk-free = 0 for simplicity; can subtract 3% / 252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Sortino (downside vol only)
    neg        = r[r < 0]
    down_vol   = neg.std() * np.sqrt(ann_factor) if len(neg) > 1 else np.nan
    sortino    = ann_ret / down_vol if (down_vol and down_vol > 0) else np.nan

    # max drawdown
    nav        = (1 + r).cumprod()
    roll_max   = nav.cummax()
    dd         = (nav - roll_max) / roll_max
    max_dd     = dd.min()

    # Calmar
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # win rate
    win_rate   = (r > 0).mean()

    # skewness / kurtosis
    skew       = scipy_stats.skew(r)
    kurt       = scipy_stats.kurtosis(r)

    # VaR 95%
    var_95     = np.percentile(r, 5)
    cvar_95    = r[r <= var_95].mean()

    out = {
        "ann_return"   : round(ann_ret * 100, 2),    # %
        "ann_vol"      : round(ann_vol * 100, 2),
        "sharpe"       : round(sharpe, 3),
        "sortino"      : round(sortino, 3),
        "calmar"       : round(calmar, 3),
        "max_drawdown" : round(max_dd * 100, 2),      # %
        "win_rate"     : round(win_rate * 100, 2),    # %
        "skewness"     : round(skew, 3),
        "kurtosis"     : round(kurt, 3),
        "var_95"       : round(var_95 * 100, 2),      # %
        "cvar_95"      : round(cvar_95 * 100, 2),
        "total_return" : round(total_ret * 100, 2),
        "n_days"       : len(r),
    }

    # alpha / beta vs benchmark
    if benchmark is not None:
        bm = benchmark.reindex(r.index).dropna()
        common = r.index.intersection(bm.index)
        if len(common) > 10:
            slope, intercept, rval, pval, se = scipy_stats.linregress(bm[common], r[common])
            out["beta"]     = round(slope, 3)
            out["alpha_ann"]= round((intercept * ann_factor) * 100, 2)
            out["corr_bm"]  = round(rval, 3)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# IC / factor stats
# ══════════════════════════════════════════════════════════════════════════════

def ic_series(signal: pd.DataFrame, forward_returns: pd.DataFrame,
              lag: int = 1) -> pd.Series:
    """
    Daily IC: cross-sectional rank correlation between today's signal
    and forward returns `lag` days ahead.
    """
    fwd = forward_returns.shift(-lag)
    ics = {}
    for date in signal.index:
        s = signal.loc[date].dropna()
        f = fwd.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(f.index)
        if len(common) < 10:
            continue
        ic, _ = scipy_stats.spearmanr(s[common], f[common])
        ics[date] = ic
    return pd.Series(ics)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) == 0:
        return {}
    return {
        "IC_mean"  : round(ic.mean(), 4),
        "IC_std"   : round(ic.std(), 4),
        "ICIR"     : round(ic.mean() / ic.std(), 3) if ic.std() > 0 else np.nan,
        "IC_pos_%" : round((ic > 0).mean() * 100, 1),
        "IC_t_stat": round(ic.mean() / (ic.std() / np.sqrt(len(ic))), 3),
    }


def ic_decay(signal: pd.DataFrame, forward_returns: pd.DataFrame,
             max_lag: int = config.DECAY_MAX_LAG) -> pd.DataFrame:
    """IC at each lag from 1 to max_lag."""
    rows = []
    for lag in range(1, max_lag + 1):
        ic = ic_series(signal, forward_returns, lag=lag)
        summary = ic_summary(ic)
        summary["lag"] = lag
        rows.append(summary)
    return pd.DataFrame(rows).set_index("lag")


def quintile_returns(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                     n: int = 5, lag: int = 1) -> pd.DataFrame:
    """
    Sort stocks into n quantiles each day by signal,
    compute mean forward return per quantile.
    """
    fwd = forward_returns.shift(-lag)
    bins = list(range(1, n + 1))
    qret = {b: [] for b in bins}

    for date in signal.index:
        s = signal.loc[date].dropna()
        f = fwd.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(f.index)
        if len(common) < n * 2:
            continue
        sc = s[common]
        fc = f[common]
        labels = pd.qcut(sc, n, labels=bins, duplicates="drop")
        for b in bins:
            idx = labels[labels == b].index
            if len(idx) > 0:
                qret[b].append(fc[idx].mean())

    return pd.DataFrame({
        f"Q{b}": pd.Series(qret[b]).mean() * 252 * 100  # annualised %
        for b in bins
    }, index=["ann_return_%"])


# ══════════════════════════════════════════════════════════════════════════════
# Robustness tests
# ══════════════════════════════════════════════════════════════════════════════

def subperiod_stats(returns: pd.Series, benchmark: pd.Series = None,
                    years: int = config.SUBPERIOD_YEARS) -> pd.DataFrame:
    """
    Split return series into sub-periods of `years` years each.
    Compute Sharpe for each sub-period.
    """
    rows = []
    period_days = years * 252
    for start in range(0, len(returns), period_days):
        sub = returns.iloc[start: start + period_days]
        if len(sub) < 60:
            continue
        bm_sub = benchmark.reindex(sub.index) if benchmark is not None else None
        stats  = performance_stats(sub, bm_sub)
        stats["period_start"] = sub.index[0].strftime("%Y-%m-%d")
        stats["period_end"]   = sub.index[-1].strftime("%Y-%m-%d")
        rows.append(stats)
    return pd.DataFrame(rows).set_index("period_start")


def turnover_stats(turnover: pd.Series) -> dict:
    """Annualised turnover and holding period estimate."""
    ann_to = turnover.sum() / (len(turnover) / 252)
    return {
        "ann_turnover_%"   : round(ann_to * 100, 1),
        "avg_hold_days"    : round(1 / (turnover[turnover > 0].mean() + 1e-9), 1)
        if (turnover > 0).any() else np.nan,
    }


def full_report(result, forward_returns: pd.DataFrame,
                signal: pd.DataFrame = None) -> dict:
    """
    Compile complete analytics for a BacktestResult.
    Returns nested dict with all stats.
    """
    sig = signal if signal is not None else result.signal

    perf = performance_stats(result.returns, result.benchmark)
    to   = turnover_stats(result.turnover)
    ic   = ic_summary(ic_series(sig, forward_returns))
    qret = quintile_returns(sig, forward_returns)
    sub  = subperiod_stats(result.returns, result.benchmark)
    decay = ic_decay(sig, forward_returns)

    return {
        "performance" : perf,
        "turnover"    : to,
        "ic"          : ic,
        "quintiles"   : qret,
        "subperiods"  : sub,
        "ic_decay"    : decay,
    }
