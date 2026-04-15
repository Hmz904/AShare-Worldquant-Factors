"""
backtest.py — Long-only portfolio + index short hedge
=====================================================
Strategy:
  • Long the top-N stocks by alpha signal each rebalance date
  • Weight = equal or circ_mv weighted
  • Short the index (CSI 300 / 500) at a beta-matched notional
  • Simulate commission, stamp duty, slippage on each turnover
  • Track daily NAV, positions, turnover
"""

import numpy as np
import pandas as pd
from typing import Optional
import config


class BacktestResult:
    """Container for all backtest output."""
    def __init__(self):
        self.nav          : pd.Series = None   # daily NAV (starts at 1.0)
        self.returns      : pd.Series = None   # daily strategy returns
        self.benchmark    : pd.Series = None   # daily benchmark returns
        self.holdings     : pd.DataFrame = None  # date × stock weights
        self.turnover     : pd.Series = None   # daily one-way turnover
        self.signal       : pd.DataFrame = None  # raw alpha signal
        self.alpha_name   : str = ""


def _rebal_dates(trade_dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return subset of trade_dates at the given rebalance frequency."""
    s = pd.Series(trade_dates, index=trade_dates)
    if freq == "D":
        return trade_dates
    elif freq == "W":
        return pd.DatetimeIndex(s.resample("W-FRI").last().dropna().values)
    elif freq == "M":
        return pd.DatetimeIndex(s.resample("ME").last().dropna().values)
    else:
        raise ValueError(f"Unknown freq: {freq}. Use 'D','W','M'.")


def run_backtest(
    signal       : pd.DataFrame,
    panels       : dict,
    universe_mask: pd.DataFrame,
    benchmark_ret: pd.Series,
    alpha_name   : str = "alpha",
    top_n        : int = config.TOP_N,
    rebal_freq   : str = config.REBAL_FREQ,
    long_weight  : str = config.LONG_WEIGHT,
    commission   : float = config.COMMISSION_RATE,
    stamp_duty   : float = config.STAMP_DUTY,
    slippage     : float = config.SLIPPAGE,
) -> BacktestResult:

    result = BacktestResult()
    result.signal     = signal
    result.alpha_name = alpha_name

    # align all panels to signal index/columns
    trade_dates = signal.index
    codes       = signal.columns

    price_ret = panels["returns"].reindex(index=trade_dates, columns=codes)
    cap_panel = panels.get("cap", pd.DataFrame()).reindex(index=trade_dates, columns=codes)
    mask      = universe_mask.reindex(index=trade_dates, columns=codes).fillna(False)

    # masked signal: only tradeable stocks
    sig_masked = signal.where(mask, other=np.nan)

    rebal_days = _rebal_dates(trade_dates, rebal_freq)
    rebal_set  = set(rebal_days)

    # ── portfolio state ───────────────────────────────────────────────────────
    current_weights = pd.Series(0.0, index=codes)   # current long weights
    nav             = 1.0
    nav_series      = {}
    ret_series      = {}
    holdings_list   = {}
    turnover_list   = {}

    for i, date in enumerate(trade_dates):
        # ── rebalance ─────────────────────────────────────────────────────────
        if date in rebal_set or i == 0:
            row = sig_masked.loc[date].dropna()
            if len(row) == 0:
                new_weights = pd.Series(0.0, index=codes)
            else:
                # select top-N
                top = row.nlargest(top_n)
                top_codes = top.index

                # weighting
                if long_weight == "equal":
                    w = pd.Series(1.0 / len(top_codes), index=top_codes)
                else:  # value weighted by cap
                    caps = cap_panel.loc[date, top_codes].fillna(0)
                    total_cap = caps.sum()
                    w = caps / total_cap if total_cap > 0 else pd.Series(1.0/len(top_codes), index=top_codes)
                new_weights = w.reindex(codes).fillna(0.0)

            # transaction cost
            turnover   = (new_weights - current_weights).abs().sum() / 2
            cost_buy   = (new_weights - current_weights).clip(lower=0).sum() * (commission + slippage)
            cost_sell  = (current_weights - new_weights).clip(lower=0).sum() * (commission + slippage + stamp_duty)
            total_cost = cost_buy + cost_sell

            current_weights = new_weights
            turnover_list[date] = turnover
        else:
            turnover_list[date] = 0.0
            total_cost = 0.0

        # ── daily return ──────────────────────────────────────────────────────
        day_ret_stocks = price_ret.loc[date]
        long_ret       = (current_weights * day_ret_stocks.fillna(0)).sum()

        # index hedge: short benchmark at 1:1 notional (market-neutral overlay)
        bm_ret   = benchmark_ret.get(date, 0.0)
        net_ret  = long_ret - bm_ret - total_cost

        nav      = nav * (1 + net_ret)
        nav_series[date] = nav
        ret_series[date] = net_ret
        holdings_list[date] = current_weights.copy()

    result.nav       = pd.Series(nav_series)
    result.returns   = pd.Series(ret_series)
    result.benchmark = benchmark_ret.reindex(trade_dates).fillna(0)
    result.holdings  = pd.DataFrame(holdings_list).T
    result.turnover  = pd.Series(turnover_list)

    return result
