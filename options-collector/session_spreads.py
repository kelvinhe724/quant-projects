"""On-demand table of quoted option spreads by session (open, midday, close). No launchd; run it when you want it.

Run: python3 session_spreads.py            reads source-material/options-collector, writes reports/session_spreads.csv

One row per ticker x moneyness bucket (ATM |k|<2%, OTM 5-10%) x session, on quotes 21-45 days to expiry.
Quotes with bid 0 or ask 0 (0/0 included) are dropped and counted. The close/midday columns are the per-day
ratio of the close median to the midday median, repeated on each session row of the bucket.
"""
import sys

import pandas as pd

import data

HEADER = ("quoted Yahoo spreads, ~15 min delayed, not fills; signed VRP cost reads stay at 15:35 per "
          "vol-risk-premium/PREREG.md; reinterpreting its §6 kill line needs a signed amendment")
BUCKETS = ("ATM", "OTM5-10")
COLS = ["ticker", "moneyness", "dte", "session", "median_rel_spread", "close_over_midday_median",
        "close_over_midday_min", "close_over_midday_max", "n_days", "n_quotes", "first_date", "last_date"]


def bucket(k):
    """Moneyness bucket of |strike/spot - 1|: ATM under 2%, 5-10% OTM, else None."""
    return "ATM" if k < 0.02 else "OTM5-10" if 0.05 < k < 0.10 else None


def read_quotes(root, date, session, ticker):
    """In-window quotes of one ticker/session/day, tagged with bucket; (frame, number dropped for a zero side)."""
    d = root / date if session == "close" else root / date / session
    f = d / f"{ticker}.csv"
    if not f.exists():
        return None, 0
    q = pd.read_csv(f, usecols=["expiry", "strike", "bid", "ask", "spot"])
    q = q[((pd.to_datetime(q["expiry"]) - pd.Timestamp(date)).dt.days.between(21, 45))]
    k = (q["strike"] / q["spot"] - 1).abs()
    q = q.assign(bucket=k.map(bucket)).dropna(subset=["bucket"])
    ok = (q["bid"] > 0) & (q["ask"] > 0)
    q = q[ok].assign(rel=(q["ask"] - q["bid"]) / ((q["ask"] + q["bid"]) / 2))
    return q, int((~ok).sum())


def build(root=None):
    """Return (table, excluded count)."""
    root = data.ROOT if root is None else root
    days = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[:2] == "20")
    rows, excluded = [], 0
    for t in data.UNIVERSE:
        per = {}  # (bucket, session) -> {date: median}, and pooled quote counts
        for date in days:
            for s in data.SESSIONS:
                q, ex = read_quotes(root, date, s, t)
                excluded += ex
                if q is None:
                    continue
                for b, g in q.groupby("bucket"):
                    e = per.setdefault((b, s), {"med": {}, "rel": []})
                    e["med"][date] = g["rel"].median()
                    e["rel"].append(g["rel"])
        for b in BUCKETS:
            mid, clo = per.get((b, "midday"), {"med": {}})["med"], per.get((b, "close"), {"med": {}})["med"]
            ratio = pd.Series({d: clo[d] / mid[d] for d in clo if d in mid})
            for s in data.SESSIONS:
                if (b, s) not in per:
                    continue
                e = per[(b, s)]
                pooled = pd.concat(e["rel"])
                rows.append({"ticker": t, "moneyness": b, "dte": "21-45", "session": s,
                             "median_rel_spread": pooled.median(),
                             "close_over_midday_median": ratio.median() if len(ratio) else float("nan"),
                             "close_over_midday_min": ratio.min() if len(ratio) else float("nan"),
                             "close_over_midday_max": ratio.max() if len(ratio) else float("nan"),
                             "n_days": len(e["med"]), "n_quotes": len(pooled),
                             "first_date": min(e["med"]), "last_date": max(e["med"])})
    return pd.DataFrame(rows, columns=COLS), excluded


def main(root=None, out=None):
    table, excluded = build(root)
    out = data.REPORTS / "session_spreads.csv" if out is None else out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# {HEADER}\n")  # read with pd.read_csv(path, comment="#")
        table.to_csv(f, index=False)
    print(HEADER)
    print(f"excluded {excluded} in-window quotes with a zero bid or ask; wrote {len(table)} rows to {out}")
    return table


if __name__ == "__main__":
    sys.exit(0 if len(main()) else 1)
