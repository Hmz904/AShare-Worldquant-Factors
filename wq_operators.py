"""
wq_operators.py — WorldQuant alpha operator library (pandas / numpy)
=====================================================================
All functions operate on DataFrames shaped (trade_date × ts_code).
Cross-sectional operators work row-wise; time-series operators work col-wise.

Naming follows WQ Brain conventions exactly so alphas can be translated
almost symbol-for-symbol.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _cs(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Apply fn cross-sectionally (row-by-row)."""
    return df.apply(fn, axis=1)


def _require_df(x, like: pd.DataFrame) -> pd.DataFrame:
    """Broadcast scalar or Series to DataFrame."""
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return pd.DataFrame({c: x for c in like.columns}, index=like.index)
    return pd.DataFrame(x, index=like.index, columns=like.columns)


# ══════════════════════════════════════════════════════════════════════════════
# Time-series operators
# ══════════════════════════════════════════════════════════════════════════════

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """ts_delay(x, d) — value d days ago."""
    return x.shift(d)

ts_delay = delay


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """ts_delta(x, d) — x - delay(x, d)."""
    return x - x.shift(d)

ts_delta = delta


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).mean()


def ts_std_dev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).std()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Percentile rank of the current value within the past d observations,
    computed per stock (column-wise). Returns value in [0, 1].
    """
    return x.rolling(d, min_periods=max(1, d // 2)).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling Pearson correlation between two panels (column-wise)."""
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        if col in y.columns:
            result[col] = x[col].rolling(d, min_periods=max(1, d // 2)).corr(y[col])
    return result


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        if col in y.columns:
            result[col] = x[col].rolling(d, min_periods=max(1, d // 2)).cov(y[col])
    return result


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).max()


def ts_arg_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Days since rolling minimum (0 = today is the min)."""
    return x.rolling(d, min_periods=max(1, d // 2)).apply(
        lambda v: len(v) - 1 - np.argmin(v), raw=True
    )


def ts_arg_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Days since rolling maximum."""
    return x.rolling(d, min_periods=max(1, d // 2)).apply(
        lambda v: len(v) - 1 - np.argmax(v), raw=True
    )


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    ts_decay_linear: linearly weighted moving average.
    weights = [1, 2, ..., d] (most recent = d), normalized.
    """
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d, min_periods=max(1, d // 2)).apply(
        lambda v: np.dot(v[-len(w):], w[-len(v):] / w[-len(v):].sum()), raw=True
    )

ts_decay_linear = decay_linear


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(1, d // 2)).apply(np.prod, raw=True)


def ts_backfill(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Forward-fill NaNs, looking back up to d periods."""
    return x.fillna(method="ffill", limit=d)


def ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    mu  = x.rolling(d, min_periods=max(1, d // 2)).mean()
    sig = x.rolling(d, min_periods=max(1, d // 2)).std()
    return (x - mu) / sig.replace(0, np.nan)


def ts_av_diff(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """x - ts_mean(x, d)."""
    return x - ts_mean(x, d)


def ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int,
                  lag: int = 0, rettype: int = 0) -> pd.DataFrame:
    """
    Rolling OLS regression of y on x over d periods.
    rettype:
      0 = fitted value (y_hat at last observation)
      1 = residual  (y - y_hat at last observation)
      2 = beta (slope)
      3 = alpha (intercept)
      4 = R²
      5 = t-stat of beta
      6 = residual (same as 1, WQ convention)
    lag: shift x by lag periods before regression.
    """
    if lag:
        x = x.shift(lag)

    result = pd.DataFrame(np.nan, index=y.index, columns=y.columns)

    for col in y.columns:
        if col not in x.columns:
            continue
        yc = y[col]
        xc = x[col]
        for i in range(d - 1, len(y)):
            ys = yc.iloc[i - d + 1: i + 1].values
            xs = xc.iloc[i - d + 1: i + 1].values
            valid = ~(np.isnan(ys) | np.isnan(xs))
            if valid.sum() < max(5, d // 4):
                continue
            ys, xs = ys[valid], xs[valid]
            slope, intercept, r, p, se = scipy_stats.linregress(xs, ys)
            if rettype in (1, 6):
                result.iloc[i][col] = ys[-1] - (slope * xs[-1] + intercept)
            elif rettype == 2:
                result.iloc[i][col] = slope
            elif rettype == 3:
                result.iloc[i][col] = intercept
            elif rettype == 4:
                result.iloc[i][col] = r ** 2
            elif rettype == 5:
                result.iloc[i][col] = slope / se if se > 0 else np.nan
            else:
                result.iloc[i][col] = slope * xs[-1] + intercept

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Cross-sectional operators
# ══════════════════════════════════════════════════════════════════════════════

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank each day. Returns [0, 1]."""
    return x.rank(axis=1, pct=True, na_option="keep")


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each day."""
    mu  = x.mean(axis=1)
    sig = x.std(axis=1)
    return x.sub(mu, axis=0).div(sig, axis=0)


def scale(x: pd.DataFrame, scale_val: float = 1.0) -> pd.DataFrame:
    """Cross-sectional scaling so sum of abs values = scale_val."""
    abs_sum = x.abs().sum(axis=1)
    return x.div(abs_sum, axis=0).mul(scale_val)


def winsorize(x: pd.DataFrame, std: float = 3.0) -> pd.DataFrame:
    """Clip outliers at ±std standard deviations cross-sectionally."""
    mu  = x.mean(axis=1)
    sig = x.std(axis=1)
    lo  = mu - std * sig
    hi  = mu + std * sig
    return x.clip(lower=lo, upper=hi, axis=0)


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def abs_val(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def power(x: pd.DataFrame, exp: float) -> pd.DataFrame:
    return x ** exp


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.replace(0, np.nan))


def sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(x.clip(lower=0))


def min_val(x: pd.DataFrame, y) -> pd.DataFrame:
    if isinstance(y, (int, float)):
        return x.clip(upper=y)
    return pd.DataFrame(np.minimum(x.values, y.values), index=x.index, columns=x.columns)


def max_val(x: pd.DataFrame, y) -> pd.DataFrame:
    if isinstance(y, (int, float)):
        return x.clip(lower=y)
    return pd.DataFrame(np.maximum(x.values, y.values), index=x.index, columns=x.columns)


def if_else(cond: pd.DataFrame, a, b) -> pd.DataFrame:
    """Element-wise if_else."""
    a = _require_df(a, cond) if not isinstance(a, pd.DataFrame) else a
    b = _require_df(b, cond) if not isinstance(b, pd.DataFrame) else b
    return pd.DataFrame(np.where(cond.values, a.values, b.values),
                        index=cond.index, columns=cond.columns)


def trade_when(condition, signal: pd.DataFrame, otherwise) -> pd.DataFrame:
    """Output signal when condition is True, otherwise output 'otherwise'."""
    if isinstance(condition, (int, float, bool)):
        condition = pd.DataFrame(bool(condition), index=signal.index, columns=signal.columns)
    return if_else(condition.astype(bool), signal, otherwise)


# ══════════════════════════════════════════════════════════════════════════════
# Group (sector) operators
# ══════════════════════════════════════════════════════════════════════════════

def _get_groups(industry_map: pd.Series, cols: pd.Index) -> dict:
    """Returns dict of industry → list of ts_codes."""
    groups = {}
    for code in cols:
        ind = industry_map.get(code, "Unknown")
        groups.setdefault(ind, []).append(code)
    return groups


def group_neutralize(x: pd.DataFrame, industry_map: pd.Series) -> pd.DataFrame:
    """Subtract industry mean from each stock's value (cross-sectionally)."""
    result = x.copy()
    groups = _get_groups(industry_map, x.columns)
    for date in x.index:
        row = x.loc[date]
        for ind, codes in groups.items():
            valid = [c for c in codes if c in row.index and pd.notna(row[c])]
            if not valid:
                continue
            mean = row[valid].mean()
            result.loc[date, valid] = row[valid] - mean
    return result


def group_rank(x: pd.DataFrame, industry_map: pd.Series) -> pd.DataFrame:
    """Percentile rank within each industry group."""
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    groups = _get_groups(industry_map, x.columns)
    for date in x.index:
        row = x.loc[date]
        for ind, codes in groups.items():
            valid = [c for c in codes if c in row.index and pd.notna(row[c])]
            if not valid:
                continue
            sub = row[valid]
            result.loc[date, valid] = sub.rank(pct=True).values
    return result


def group_mean(x: pd.DataFrame, weight: pd.DataFrame,
               industry_map: pd.Series) -> pd.DataFrame:
    """
    Weighted mean of x within each industry group.
    If weight is a DataFrame of equal shape, compute weighted avg;
    if weight is a rank/bucket series, group by that instead.
    Returns a DataFrame same shape as x with the group mean broadcast back.
    """
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    groups = _get_groups(industry_map, x.columns)
    for date in x.index:
        rx = x.loc[date]
        rw = weight.loc[date] if isinstance(weight, pd.DataFrame) else None
        for ind, codes in groups.items():
            valid = [c for c in codes if c in rx.index and pd.notna(rx[c])]
            if not valid:
                continue
            if rw is not None:
                w = rw[valid].fillna(0)
                ws = w.sum()
                gm = (rx[valid] * w).sum() / ws if ws != 0 else rx[valid].mean()
            else:
                gm = rx[valid].mean()
            result.loc[date, valid] = gm
    return result


def group_zscore(x: pd.DataFrame, industry_map: pd.Series) -> pd.DataFrame:
    """Z-score within each industry group cross-sectionally."""
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    groups = _get_groups(industry_map, x.columns)
    for date in x.index:
        row = x.loc[date]
        for ind, codes in groups.items():
            valid = [c for c in codes if c in row.index and pd.notna(row[c])]
            if len(valid) < 2:
                continue
            sub = row[valid]
            mu, sig = sub.mean(), sub.std()
            if sig > 0:
                result.loc[date, valid] = (sub - mu) / sig
    return result


def group_backfill(x: pd.DataFrame, market_map, d: int) -> pd.DataFrame:
    """Forward-fill NaN values within each cross-section, up to d days."""
    return ts_backfill(x, d)


# ══════════════════════════════════════════════════════════════════════════════
# Bucketing / quantile helpers
# ══════════════════════════════════════════════════════════════════════════════

def bucket(x: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Cross-sectional quantile bucket (1..n)."""
    def _row_bucket(row):
        valid = row.dropna()
        if valid.empty:
            return row
        labels = pd.qcut(valid, n, labels=False, duplicates="drop") + 1
        return labels.reindex(row.index)
    return x.apply(_row_bucket, axis=1)


def quantile(x: pd.DataFrame, q: float = 0.5, driver: str = "uniform",
             sigma: float = 1.0) -> pd.DataFrame:
    """
    WQ quantile(x, driver='gaussian', sigma=1):
    Apply Gaussian CDF transform to cross-sectional ranks.
    Returns a value between 0 and 1 approximately.
    """
    r = rank(x)
    if driver == "gaussian":
        from scipy.stats import norm
        return r.apply(lambda row: pd.Series(
            norm.ppf(row.clip(0.01, 0.99)) * sigma, index=row.index
        ), axis=1)
    return r
