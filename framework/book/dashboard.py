"""Static reports/index.html from ledger.csv. No server; open the file.

Colours and type follow DESIGN.md: near-black ground, two text tiers, one
teal accent, uppercase tracked section labels over 1px rules.
"""
import html
import json
import os

import pandas as pd

from framework.book.allocate import REPORTS
from framework.book.broker import KILL, killed

INDEX = os.path.join(REPORTS, "index.html")
CSS = """
:root{--bg:#05080a;--text:#e6efee;--muted:#5d7a7e;--green:hsl(190 85% 58%);--green-dark:hsl(190 75% 44%);
--border:rgba(140,190,200,.14);--red:#ff5f56}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:14px 32px 24px}
header{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:8px}
h1{font-size:13px;font-weight:500;letter-spacing:.22em;text-transform:uppercase;margin:0}
.meta{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:26px 34px;margin-top:22px}
.wide{grid-column:1/-1}
h2{font-size:10px;font-weight:500;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:4px;margin:0 0 9px}
.stats{display:flex;gap:34px;flex-wrap:wrap}
.stat{font-size:26px;font-weight:400;line-height:1.1;font-variant-numeric:tabular-nums}
.stat .q{display:block;font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:6px 0;border-bottom:1px solid var(--border);text-align:right;font-weight:400}
td:first-child,th:first-child{text-align:left}
th{color:var(--muted);font-size:10px;letter-spacing:.14em;text-transform:uppercase}
.neg{color:var(--red)}.pos{color:var(--green)}
svg text{fill:var(--muted);font-size:10px}
.legend{font-size:11px;color:var(--muted);margin-top:6px}.legend b{font-weight:500;color:var(--text)}
"""


def load_json(name):
    p = os.path.join(REPORTS, name)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def chart(series, w=900, h=240, pad=36):
    """Inline SVG of one or more equity paths on a shared date axis."""
    series = {k: s.dropna() for k, s in series.items() if len(s.dropna())}
    if not series:
        return "<p class=meta>no rows yet</p>"
    dates = sorted({d for s in series.values() for d in s.index})
    lo = min(s.min() for s in series.values())
    hi = max(s.max() for s in series.values())
    lo, hi = lo - (hi - lo or lo * 0.02) * 0.1, hi + (hi - lo or lo * 0.02) * 0.1
    x = lambda d: pad + (dates.index(d) / max(len(dates) - 1, 1)) * (w - 2 * pad)
    y = lambda v: h - pad + (v - lo) / (hi - lo) * (2 * pad - h)
    styles = {"shadow": "stroke:var(--green);stroke-width:1.6", "alpaca": "stroke:var(--text);stroke-width:1.2",
              "expected": "stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4"}
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    for k, s in series.items():
        pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in s.items())
        parts.append(f'<polyline fill="none" style="{styles.get(k, styles["alpaca"])}" points="{pts}"/>')
        if len(s) == 1:
            parts.append(f'<circle cx="{x(s.index[0]):.1f}" cy="{y(s.iloc[0]):.1f}" r="3" style="fill:var(--green)"/>')
    for v in (lo + (hi - lo) * 0.1, hi - (hi - lo) * 0.1):
        parts.append(f'<text x="2" y="{y(v):.1f}">{v:,.0f}</text>')
    parts.append(f'<text x="{pad}" y="{h - 8}">{dates[0]:%Y-%m-%d}</text>')
    parts.append(f'<text x="{w - pad}" y="{h - 8}" text-anchor="end">{dates[-1]:%Y-%m-%d}</text>')
    parts.append("</svg>")
    return "".join(parts)


def pct(v, cls=True):
    c = ' class="neg"' if cls and v < 0 else ' class="pos"' if cls and v > 0 else ""
    return f"<td{c}>{v:+.2%}</td>"


def render(path=INDEX):
    led_path = os.path.join(REPORTS, "ledger.csv")
    led = pd.read_csv(led_path, parse_dates=["date"]) if os.path.exists(led_path) else pd.DataFrame()
    val, alloc = load_json("validation.json"), load_json("allocations.json")
    head = val.get("headline", {})
    shadow = led[led.book == "shadow"].set_index("date") if len(led) else pd.DataFrame()
    alpaca = led[led.book == "alpaca"].set_index("date") if len(led) else pd.DataFrame()
    now = pd.Timestamp.now(tz="America/New_York")
    last_run = pd.Timestamp(led.run_at.max()) if len(led) else None
    since = f"{(now - last_run).days}" if last_run is not None else "never"
    mode = str(alpaca.note.iloc[-1]).split(":")[0] if len(alpaca) else "no keys"
    kill = "KILL PRESENT" if killed() else "armed"

    series = {}
    if len(shadow):
        series["shadow"] = shadow.equity
        if head.get("annual_return") is not None:
            t = (shadow.index - shadow.index[0]).days / 365.25
            series["expected"] = pd.Series(shadow.equity.iloc[0] * (1 + head["annual_return"]) ** t, shadow.index)
    if len(alpaca):
        series["alpaca"] = alpaca.equity
    dd = (shadow.equity / shadow.equity.cummax() - 1).iloc[-1] if len(shadow) else 0.0

    last = shadow.iloc[-1] if len(shadow) else None
    pos_rows = ""
    if last is not None:
        held = json.loads(last.positions)
        target = json.loads(last.targets)
        held_a = json.loads(alpaca.iloc[-1].positions) if len(alpaca) else {}
        for n in sorted(set(held) | set(target) | set(held_a)):
            pos_rows += (f"<tr><td>{html.escape(n)}</td>{pct(held.get(n, 0.0), False)}"
                         f"{pct(target.get(n, 0.0), False)}"
                         f"<td>{pct(held_a[n], False)[4:-5] if n in held_a else '<span class=meta>no keys</span>'}</td></tr>")
    fill_rows = ""
    for _, r in led.sort_values("run_at").tail(2).iterrows() if len(led) else []:
        for f in json.loads(r.fills):
            fill_rows += (f"<tr><td>{r.date:%Y-%m-%d} {r.book}</td><td>{html.escape(str(f.get('instrument', f.get('symbol'))))}</td>"
                          f"<td>{f.get('quantity', f.get('qty'))}</td><td>{f.get('fill', f.get('notional'))}</td></tr>")
    if not fill_rows:
        fill_rows = "<tr><td class=meta colspan=4>no fills yet</td></tr>"

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Premia book</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>
<header><h1>Premia book</h1><span class="meta">last run {html.escape(str(last_run)[:19]) if last_run is not None else 'never'}
 · panel {html.escape(str(led.panel_hash.iloc[-1])) if len(led) else '-'} · {alloc.get('allocator', '-')} allocator</span></header>
<div class="grid">
<section class="wide"><h2>Book</h2><div class="stats">
<div class="stat">{f"{shadow.equity.iloc[-1]:,.0f}" if len(shadow) else '-'}<span class="q">shadow equity</span></div>
<div class="stat">{f"{alpaca.equity.iloc[-1]:,.0f}" if len(alpaca) else '<span class=meta>no keys</span>'}<span class="q">alpaca equity · {html.escape(mode)}</span></div>
<div class="stat{' neg' if dd < -1e-9 else ''}">{dd:.2%}<span class="q">shadow drawdown</span></div>
<div class="stat{' neg' if killed() else ''}">{kill}<span class="q">kill switch · {since} days since last run</span></div>
<div class="stat">{head.get('sharpe', float('nan')):.2f}<span class="q">backtest sharpe · dsr {head.get('dsr', float('nan')):.2f}</span></div>
</div></section>
<section class="wide"><h2>Equity</h2>{chart(series)}
<div class="legend"><b>shadow</b> engine fills at the shadow cost model · <b>alpaca</b> {html.escape(mode)} account · <b>expected</b> backtest drift {head.get('annual_return', 0):+.2%}/yr</div></section>
<section><h2>Positions, weight of equity</h2><table><tr><th>instrument</th><th>shadow</th><th>target</th><th>alpaca ({html.escape(mode)})</th></tr>{pos_rows or '<tr><td class=meta colspan=4>none</td></tr>'}</table></section>
<section><h2>Last orders</h2><table><tr><th>run</th><th>instrument</th><th>qty</th><th>fill / notional</th></tr>{fill_rows}</table></section>
<section class="wide"><h2>Allocator</h2><p class="meta">{html.escape(json.dumps(alloc.get('weights', {})))} · ERC vs 1/N out of sample:
Sharpe {alloc.get('oos', {}).get('erc', {}).get('sharpe', '-')} vs {alloc.get('oos', {}).get('equal', {}).get('sharpe', '-')} · full report in validation.md</p></section>
</div></body></html>"""
    os.makedirs(REPORTS, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(doc)
    return path


if __name__ == "__main__":
    print(render())
