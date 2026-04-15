"""
main.py — Entry point for the A-share WQ factor backtest system
===============================================================
Usage:
  python main.py                         # run all available alphas
  python main.py --alpha alpha_10        # run a single alpha
  python main.py --alpha alpha_07 alpha_10 alpha_21  # run multiple
  python main.py --refresh               # force re-download all data

Output:
  ./reports/<alpha_name>_report.html     # full HTML tearsheet per alpha
  ./reports/summary.csv                  # comparison table across alphas
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import config
from data_loader import (load_stock_meta, load_st_history, load_trade_cal,
                          load_price_panel, load_financial_panel,
                          load_index, build_universe_mask)
from alpha_engine import ALPHA_REGISTRY, compute_alpha
from backtest import run_backtest
from stats import full_report, performance_stats
from report import generate_report


def build_data_dict(panels: dict, fin: dict, meta: pd.DataFrame) -> dict:
    """Merge all panels into the single `data` dict passed to alpha functions."""
    data = dict(panels)
    data.update(fin)

    # industry map: ts_code → industry string (SW L1 approximation)
    data["industry_map"] = meta["industry"].fillna("Unknown")

    # ensure adj_close is used as 'close' everywhere in alphas
    if "adj_close" in data:
        data["close"] = data["adj_close"]

    return data


def run(alpha_names: list, force_refresh: bool = False):
    Path("./reports").mkdir(exist_ok=True)

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    meta    = load_stock_meta(force_refresh)
    st_hist = load_st_history(force_refresh)
    panels  = load_price_panel(force_refresh=force_refresh)
    fin     = load_financial_panel(force_refresh=force_refresh)
    bm      = load_index()

    bm_ret  = bm["returns"].fillna(0)
    data    = build_data_dict(panels, fin, meta)

    # ── 2. universe mask ──────────────────────────────────────────────────────
    print("\n[2/5] Building universe filter mask...")
    mask = build_universe_mask(panels, meta, st_hist)

    # ── 3. compute alphas ────────────────────────────────────────────────────
    print(f"\n[3/5] Computing {len(alpha_names)} alpha(s)...")
    summary_rows = []

    for alpha_name in alpha_names:
        print(f"\n  ── {alpha_name} ──")
        signal = compute_alpha(alpha_name, data)
        if signal is None:
            print(f"  SKIP {alpha_name} (data unavailable)")
            continue

        # align to common index
        common_dates = panels["returns"].index.intersection(signal.index)
        signal_aligned = signal.reindex(index=common_dates,
                                        columns=panels["returns"].columns)

        # ── 4. backtest ───────────────────────────────────────────────────────
        print(f"  Running backtest...")
        result = run_backtest(
            signal        = signal_aligned,
            panels        = panels,
            universe_mask = mask,
            benchmark_ret = bm_ret,
            alpha_name    = alpha_name,
        )

        # ── 5. stats + report ─────────────────────────────────────────────────
        print(f"  Computing stats...")
        fwd_ret = panels["returns"]
        rpt     = full_report(result, fwd_ret, signal=signal_aligned)

        perf  = rpt["performance"]
        ic    = rpt["ic"]
        to    = rpt["turnover"]

        print(f"  Sharpe  : {perf.get('sharpe', '—')}")
        print(f"  Ann Ret : {perf.get('ann_return', '—')}%")
        print(f"  Max DD  : {perf.get('max_drawdown', '—')}%")
        print(f"  IC mean : {ic.get('IC_mean', '—')}  ICIR: {ic.get('ICIR', '—')}")

        html_path = f"./reports/{alpha_name}_report.html"
        generate_report(result, rpt, alpha_name, html_path)

        summary_rows.append({
            "alpha"       : alpha_name,
            "ann_return%" : perf.get("ann_return"),
            "ann_vol%"    : perf.get("ann_vol"),
            "sharpe"      : perf.get("sharpe"),
            "sortino"     : perf.get("sortino"),
            "max_dd%"     : perf.get("max_drawdown"),
            "calmar"      : perf.get("calmar"),
            "win_rate%"   : perf.get("win_rate"),
            "IC_mean"     : ic.get("IC_mean"),
            "ICIR"        : ic.get("ICIR"),
            "IC_pos%"     : ic.get("IC_pos_%"),
            "ann_turnover%": to.get("ann_turnover_%"),
        })

    # ── summary CSV ───────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index("alpha")
        summary_df.to_csv("./reports/summary.csv")
        print(f"\n[Summary saved] → ./reports/summary.csv")
        print(summary_df.to_string())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-share WQ Factor Backtest")
    parser.add_argument("--alpha", nargs="+", default=None,
                        help="Alpha name(s) to run, e.g. --alpha alpha_10 alpha_21")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download all data (ignore cache)")
    args = parser.parse_args()

    if args.alpha:
        names = args.alpha
    else:
        # run all alphas that aren't explicitly skipped
        names = [n for n, (fn, desc) in ALPHA_REGISTRY.items()
                 if "SKIP" not in desc]

    run(names, force_refresh=args.refresh)
