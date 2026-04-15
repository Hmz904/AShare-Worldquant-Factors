"""
report.py — Full HTML tearsheet generator
==========================================
Generates a self-contained HTML file with:
  • NAV chart vs benchmark
  • Drawdown chart
  • IC / ICIR bar chart
  • Quintile return chart
  • IC decay chart
  • Sub-period stats table
  • Full metrics table
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def _s(v):
    """Safe JSON-serialisable scalar."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _series_to_json(s: pd.Series) -> str:
    dates = [str(d)[:10] for d in s.index]
    vals  = [_s(v) for v in s.values]
    return json.dumps({"dates": dates, "values": vals})


def _df_to_html(df: pd.DataFrame, title: str = "") -> str:
    styled = df.fillna("—").to_html(classes="stats-table", border=0)
    return f"<h3>{title}</h3>{styled}" if title else styled


def generate_report(result, report: dict, alpha_name: str,
                    output_path: str = "./output_report.html"):
    """
    result  : BacktestResult
    report  : dict from stats.full_report()
    """
    nav_json   = _series_to_json(result.nav)
    bm_nav     = (1 + result.benchmark.reindex(result.returns.index).fillna(0)).cumprod()
    bm_json    = _series_to_json(bm_nav)
    dd         = (result.nav / result.nav.cummax() - 1)
    dd_json    = _series_to_json(dd)

    # IC decay
    ic_decay_df = report.get("ic_decay", pd.DataFrame())
    decay_lags  = list(ic_decay_df.index) if not ic_decay_df.empty else []
    decay_ic    = [_s(v) for v in ic_decay_df.get("IC_mean", pd.Series()).values]

    # Quintile
    qdf     = report.get("quintiles", pd.DataFrame())
    q_labels = list(qdf.columns) if not qdf.empty else []
    q_vals   = [_s(v) for v in qdf.iloc[0].values] if not qdf.empty else []

    # Sub-period table
    sub_html = ""
    if "subperiods" in report and not report["subperiods"].empty:
        sub_df   = report["subperiods"][["period_end","ann_return","ann_vol",
                                         "sharpe","max_drawdown"]].copy()
        sub_html = _df_to_html(sub_df, "Sub-period analysis")

    # Performance table
    perf = report.get("performance", {})
    ic   = report.get("ic", {})
    to   = report.get("turnover", {})
    all_stats = {**perf, **ic, **to}
    stats_rows = "".join(
        f"<tr><td>{k}</td><td>{_s(v) if v is not None else '—'}</td></tr>"
        for k, v in all_stats.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{alpha_name} — Backtest Tearsheet</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f9fa; color: #212529; padding: 20px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 500; color: #495057; margin: 24px 0 12px; }}
  h3 {{ font-size: 14px; font-weight: 500; margin: 16px 0 8px; color: #343a40; }}
  .subtitle {{ color: #6c757d; font-size: 13px; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card.full {{ grid-column: 1 / -1; }}
  canvas {{ max-height: 260px; }}
  .stats-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .stats-table th {{ background: #f1f3f5; padding: 8px 12px; text-align: left;
                     font-weight: 500; border-bottom: 1px solid #dee2e6; }}
  .stats-table td {{ padding: 6px 12px; border-bottom: 1px solid #f1f3f5; }}
  .stats-table tr:hover td {{ background: #f8f9fa; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: white; border-radius: 8px; padding: 14px 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.07); }}
  .kpi-label {{ font-size: 11px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .pos {{ color: #2d6a4f; }} .neg {{ color: #c0392b; }}
</style>
</head>
<body>
<h1>📊 {alpha_name}</h1>
<p class="subtitle">A-share Backtest Tearsheet &nbsp;|&nbsp;
  {result.returns.index[0].strftime('%Y-%m-%d')} → {result.returns.index[-1].strftime('%Y-%m-%d')}
  &nbsp;|&nbsp; Top-{result.holdings.astype(bool).sum(axis=1).median():.0f} long + CSI 300 short hedge
</p>

<!-- KPI row -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Annual Return</div>
    <div class="kpi-value {'pos' if perf.get('ann_return',0)>0 else 'neg'}">{perf.get('ann_return','—')}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value {'pos' if perf.get('sharpe',0)>0 else 'neg'}">{perf.get('sharpe','—')}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value neg">{perf.get('max_drawdown','—')}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">IC / ICIR</div>
    <div class="kpi-value">{ic.get('IC_mean','—')} / {ic.get('ICIR','—')}</div>
  </div>
</div>

<div class="grid">
  <!-- NAV chart -->
  <div class="card full">
    <h2>Cumulative NAV vs Benchmark (CSI 300)</h2>
    <canvas id="navChart"></canvas>
  </div>

  <!-- Drawdown -->
  <div class="card">
    <h2>Drawdown</h2>
    <canvas id="ddChart"></canvas>
  </div>

  <!-- IC decay -->
  <div class="card">
    <h2>IC Decay (lag 1 → {len(decay_lags)} days)</h2>
    <canvas id="decayChart"></canvas>
  </div>

  <!-- Quintile returns -->
  <div class="card">
    <h2>Quintile Annual Returns</h2>
    <canvas id="quintileChart"></canvas>
  </div>

  <!-- Stats table -->
  <div class="card">
    <h2>Full Metrics</h2>
    <table class="stats-table">
      <tr><th>Metric</th><th>Value</th></tr>
      {stats_rows}
    </table>
  </div>
</div>

<!-- Sub-period -->
{f'<div class="card">{sub_html}</div>' if sub_html else ''}

<script>
const navData  = {nav_json};
const bmData   = {bm_json};
const ddData   = {dd_json};
const decayLags = {json.dumps(decay_lags)};
const decayIC   = {json.dumps(decay_ic)};
const qLabels   = {json.dumps(q_labels)};
const qVals     = {json.dumps(q_vals)};

const palette = {{
  blue  : 'rgba(53,162,235,1)',
  blueL : 'rgba(53,162,235,0.15)',
  gray  : 'rgba(150,150,150,0.7)',
  red   : 'rgba(220,53,69,0.8)',
  green : 'rgba(40,167,69,0.8)',
  amber : 'rgba(255,193,7,0.9)',
}};

// NAV chart
new Chart(document.getElementById('navChart'), {{
  type: 'line',
  data: {{
    labels: navData.dates,
    datasets: [
      {{ label: 'Strategy', data: navData.values, borderColor: palette.blue,
         backgroundColor: palette.blueL, fill: true, pointRadius: 0, borderWidth: 2 }},
      {{ label: 'CSI 300', data: bmData.values, borderColor: palette.gray,
         fill: false, pointRadius: 0, borderWidth: 1.5, borderDash: [4,3] }},
    ]
  }},
  options: {{ responsive: true, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }} }}
}});

// Drawdown chart
new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    labels: ddData.dates,
    datasets: [{{ label: 'Drawdown', data: ddData.values.map(v=>v*100),
       borderColor: palette.red, backgroundColor: 'rgba(220,53,69,0.1)',
       fill: true, pointRadius: 0, borderWidth: 1.5 }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ maxTicksLimit: 6 }} }},
               y: {{ ticks: {{ callback: v => v+'%' }} }} }} }}
}});

// IC decay
new Chart(document.getElementById('decayChart'), {{
  type: 'bar',
  data: {{
    labels: decayLags.map(l=>'Lag '+l),
    datasets: [{{ label: 'IC', data: decayIC,
      backgroundColor: decayIC.map(v => v > 0 ? palette.green : palette.red) }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'IC' }} }} }} }}
}});

// Quintile
new Chart(document.getElementById('quintileChart'), {{
  type: 'bar',
  data: {{
    labels: qLabels,
    datasets: [{{ label: 'Ann. Return %', data: qVals,
      backgroundColor: qVals.map(v => v > 0 ? palette.green : palette.red) }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Ann. Return (%)' }} }} }} }}
}});
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved → {output_path}")
    return output_path
