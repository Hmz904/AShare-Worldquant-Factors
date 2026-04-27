# A-Share WQ Factor Backtest System

A complete Python backtesting framework for translating **WorldQuant Brain alpha expressions** into executable strategies on **A-share (China) equity data** via [Tushare Pro](https://tushare.pro/).

---

## 📊 Featured Result — Idio-Vol Composite (Sharpe 0.78)

After translating 25 WorldQuant alphas, I went one level deeper and ran a full
research cycle to build my own long-only factor on this universe. Started at
-0.21 Sharpe; ended at **+0.78 Sharpe, 10.9% ann. return, -22% max DD** after
six iterations of failure, diagnosis, and refinement.

📄 **[Read the full research log (PDF)](docs/research_notes.pdf)** —
six-iteration journey from failed factors to a working composite, including
the two diagnostic breakthroughs that changed everything.

📈 **[View the sample tearsheet (HTML)](reports/alpha_34_report.html)** —
interactive Chart.js report with NAV curve, drawdown, IC decay, quintile
returns, and turnover analysis.

| Metric              | Value     |
|---------------------|-----------|
| Annualized return   | **+10.9%**|
| Sharpe              | **0.78**  |
| Sortino             | 0.89      |
| Max drawdown        | -22.1%    |
| Calmar              | 0.49      |
| ICIR                | 0.42      |
| Annual turnover     | 586%      |

> Universe: top 1500 A-share by 20d ADV. Monthly rebalance, top 100 equal-weight, 3 bps commission per side. Period 2021-01-04 to 2025-12-31.

---

## Features

- **25 alphas translated** from WorldQuant Brain syntax to pandas/numpy
- **Full WQ operator library** — `ts_rank`, `ts_corr`, `decay_linear`, `group_neutralize`, `winsorize`, `ts_regression`, `trade_when`, and more
- **Long-only + index short hedge** portfolio construction (CSI 300 / 500)
- **Realistic execution simulation** — commission, stamp duty, slippage
- **Universe filtering** — removes ST stocks, suspended stocks, limit-hit stocks, newly listed stocks
- **Rich analytics** — Sharpe, Sortino, Calmar, Max Drawdown, IC, ICIR, turnover, quintile returns
- **Robustness testing** — sub-period splits, IC decay curves, parameter sensitivity
- **HTML tearsheet** — self-contained report with interactive Chart.js charts
- **Local parquet cache** — avoids redundant API calls across runs

---

## File Structure

```
.
├── main.py            # CLI entry point — run everything from here
├── config.py          # All settings (token, dates, universe, costs)
├── data_loader.py     # Tushare data pipeline + caching + universe mask
├── wq_operators.py    # WorldQuant operator library (pandas translations)
├── alpha_engine.py    # All 25 alphas + registry
├── backtest.py        # Portfolio construction + PnL engine
├── stats.py           # Performance metrics + robustness tests
├── report.py          # HTML tearsheet generator
├── docs/
│   └── research_notes.pdf   # 📄 Full research log — six iterations
├── reports/
│   └── alpha_34_report.html # 📈 Sample interactive tearsheet
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.py` and paste your [Tushare Pro token](https://tushare.pro/register):

```python
TUSHARE_TOKEN = "your_token_here"
START_DATE    = "20180101"
END_DATE      = "20241231"
TOP_N         = 50          # number of long positions
REBAL_FREQ    = "M"         # D / W / M
```

### 3. Run

```bash
# Single alpha (fast, good for testing)
python main.py --alpha alpha_10

# Multiple alphas
python main.py --alpha alpha_07 alpha_10 alpha_21

# All 24 available alphas
python main.py

# Force re-download data (first run, or after a long gap)
python main.py --refresh
```

### 4. View results

Reports are saved to `./reports/`:
- `<alpha_name>_report.html` — full tearsheet per alpha ([sample](reports/alpha_34_report.html))
- `summary.csv` — side-by-side comparison across all alphas

---

## Alpha Library

| Alpha | Description | Data Required |
|-------|-------------|---------------|
| alpha_01 | News momentum proxy + price/cap combo | Price, Cap |
| alpha_02 | Leverage ratio × debt/equity min — industry neutral | Financials |
| alpha_03 | Analyst revision proxy − current ratio | Financials |
| alpha_04 | ⚠️ SKIP — options PCR/IV data unavailable in Tushare | Options |
| alpha_05 | Sector return mean reversion × price decay | Price, Vol |
| alpha_06 | Goodwill/sales regression + vol regime filter | Financials |
| alpha_07 | MA deviation + VWAP-price correlation | Price |
| alpha_08 | Price-cap regression residual × sector momentum | Price, Cap |
| alpha_09 | Intraday decline / decay of days-from-high | Price |
| alpha_10 | Bollinger Band signal z-scored | Price |
| alpha_11 | Heteroskedastic return anomaly (HORRO) | Price, Cap |
| alpha_12 | Overnight gap × intraday range asymmetry | Price |
| alpha_13 | EPS/cap/ROE group rank | Financials |
| alpha_14 | EV/EBITDA ts_rank | Financials, Cap |
| alpha_15 | Compounded return (annualised) | Price |
| alpha_16 | Loser-stock return beta (1000-day lookback) | Price |
| alpha_17 | Price-cap regression residual (lag 5) | Price, Cap |
| alpha_18 | Goodwill/sales z-score × accruals combo | Financials |
| alpha_19 | Profit volatility dynamics + volume filter | Financials, Vol |
| alpha_20 | News momentum proxy + short-term reversal | Price |
| alpha_21 | Volume ratio × price momentum rank | Price, Vol |
| alpha_22 | High-volume day count with returns filter | Price, Vol |
| alpha_23 | Operating income yield ts_rank | Financials, Cap |
| alpha_24 | Positive buy-pressure days with trend filter | Price, Vol |
| alpha_25 | GPA momentum, cap-bucket neutralised | Financials, Cap |
| **alpha_34** | **Idio-vol + idio-trend composite — see [research log](docs/research_notes.pdf)** | **Price** |

---

## WorldQuant Operator Mapping

| WQ Operator | This Library | Notes |
|-------------|-------------|-------|
| `rank(x)` | `rank(x)` | Cross-sectional pct rank |
| `ts_rank(x, d)` | `ts_rank(x, d)` | Time-series pct rank over d days |
| `delay(x, d)` | `delay(x, d)` | x shifted d days back |
| `delta(x, d)` | `delta(x, d)` | x − delay(x, d) |
| `ts_sum(x, d)` | `ts_sum(x, d)` | Rolling sum |
| `ts_mean(x, d)` | `ts_mean(x, d)` | Rolling mean |
| `ts_std_dev(x, d)` | `ts_std_dev(x, d)` | Rolling std dev |
| `ts_corr(x, y, d)` | `ts_corr(x, y, d)` | Rolling Pearson correlation |
| `decay_linear(x, d)` | `decay_linear(x, d)` | Linearly weighted MA |
| `ts_regression(y,x,d,lag,rettype)` | `ts_regression(...)` | OLS, rettype 0–6 |
| `ts_backfill(x, d)` | `ts_backfill(x, d)` | Forward-fill NaN up to d days |
| `ts_zscore(x, d)` | `ts_zscore(x, d)` | Rolling z-score |
| `winsorize(x)` | `winsorize(x, std=3)` | Clip at ±3σ cross-sectionally |
| `scale(x)` | `scale(x)` | Cross-sectional normalise |
| `group_neutralize(x, g)` | `group_neutralize(x, industry_map)` | Subtract group mean |
| `group_rank(x, g)` | `group_rank(x, industry_map)` | Rank within group |
| `group_zscore(x, g)` | `group_zscore(x, industry_map)` | Z-score within group |
| `trade_when(cond, x, y)` | `trade_when(cond, x, y)` | Conditional output |
| `if_else(cond, a, b)` | `if_else(cond, a, b)` | Element-wise ternary |

---

## Data Field Mapping (WQ → Tushare)

| WQ Field | Tushare Field | Notes |
|----------|--------------|-------|
| `close` | `adj_close` = `close × adj_factor` | Always use adj prices |
| `open`, `high`, `low` | `open`, `high`, `low` | Unadjusted (intraday ratios) |
| `returns` | `pct_chg / 100` or `adj_close.pct_change()` | |
| `volume` | `vol` (手 × 100 = shares) | |
| `vwap` | `amount / vol × 100` | Approximation |
| `cap` | `circ_mv` (float mktcap, 万元) | |
| `adv20` | `vol.rolling(20).mean()` | |
| `industry` | `stock_basic.industry` | SW Level-1 |
| `assets` | `balancesheet.total_assets` | Quarterly, forward-filled |
| `liabilities` | `balancesheet.total_liab` | |
| `debt` | `lt_borr + st_borr` | |
| `sales` / `revenue` | `income.revenue` | |
| `goodwill` | `balancesheet.goodwill` | |
| `capex` | `cashflow.capex` | |
| `operate_profit` | `income.operate_profit` | |

---

## Performance Notes

- **First data pull** (6+ years, full A-share): ~30–60 min due to Tushare rate limits. All subsequent runs use cached parquet files and are near-instant.
- **Alphas with long lookbacks** (alpha_16 needs 1000 days, alpha_06 needs 612 days): output will be NaN until enough history is loaded — start date of 2018 gives sufficient warmup by ~2021.
- **ts_regression** is the slowest operator (~O(n×d) per stock). Alphas 6, 8, 16, 17 will be slower to compute.

---

## Limitations & Known Approximations

| Alpha | What's Approximated |
|-------|---------------------|
| 01, 20 | `nws12_afterhsz_sl` (WQ news NLP) replaced with `turnover_rate` momentum |
| 04 | Skipped — requires SSE options PCR / implied volatility data |
| 13 | `fnd6_newa2v1300_rdipeps` approximated via `n_income / equity` |
| 19 | `fnd6_intc` (interest cost) approximated via `operate_profit` |
| 18 | Dividend/sharesout approximated via `pb/pe` ratio |

---

## License

MIT — use freely, no warranty.
