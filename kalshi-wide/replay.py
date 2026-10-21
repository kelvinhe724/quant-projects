"""The prior layer of PREREG.md: the wide-book maker rule replayed against the lake's books, nothing refit.

Every number the pre-registration freezes is a constant below. Candidates come
from kalshi_markets, the model price from the frozen "chosen" calibrator, fills
only from touches recorded in kalshi_books (the conservative side the pre-reg
chose), settlement from the public API. Series and settlement lookups are cached
under reports/ so a rerun is offline. One configuration, one registry row.

Run:  ../.venv/bin/python3 replay.py               the pre-registered test
      ../.venv/bin/python3 replay.py --window 75   a second look with a longer cancel window; logged as its own
                                                   trial and tagged second_look (REFUSALS.md §4 if ever cited)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("", "data-lake", "kalshi-desk", "kalshi-model"):
    sys.path.insert(0, os.path.join(ROOT, d))
import candles  # noqa: E402  kalshi-model/candles.py: the trained category list
import lake  # noqa: E402
import model as km  # noqa: E402
import shadow  # noqa: E402  kalshi-desk/shadow.py: the desk's fill rule
from research.registry import Registry, code_hash  # noqa: E402

kelly = candles.research_module("kelly")
REPORTS = os.path.join(HERE, "reports")

# PREREG.md, frozen
MARGIN = 0.02
CANCEL_MIN = 60
ADVERSE = 0.03
MIN_SPREAD = 0.10
MIN_HOURS = 24
SIZE = 1                       # the per-unit track; Kelly is reported, not applied
MIN_FILLS, T_MIN = 300, 2.0    # §5: clears if net P&L per $ staked > 0 with t > 2 on >= 300 fills
START = "2026-09-05"           # the lake's first Kalshi snapshot; PREREG says 2026-08-25 but no book exists before this
CATEGORIES = set(candles.CATEGORIES)
# Kalshi fee schedule effective July 2026 (kalshi.com/docs/kalshi-fee-schedule.pdf): maker fee =
# round up(M x 0.0175 x C x P x (1-P)) to the centicent, charged only on series whose /series fee_type
# names maker fees (159 of 14,714 series on 2026-10-07); every other resting order pays nothing.
MAKER_FEE = 0.0175
API = "https://api.elections.kalshi.com/trade-api/v2"
SLEEP = 0.15


def floor_cent(x):
    """Round a price down to the cent: a lower bid pays less and fills less, the conservative tick."""
    return np.floor(np.asarray(x, float) * 100 + 1e-9) / 100


def maker_fee(price, fee_type, multiplier=1.0):
    """Maker fee a contract in dollars, rounded up to the centicent; zero on series without maker fees."""
    price, mult = np.broadcast_arrays(np.asarray(price, float), np.asarray(multiplier, float))
    has = np.broadcast_to(np.char.find(np.asarray(fee_type, str), "maker") >= 0, price.shape)
    raw = MAKER_FEE * mult * price * (1 - price)
    return np.where(has, np.ceil(raw * 1e4 - 1e-9) / 1e4, 0.0)


def series_table():
    """Every Kalshi series with its category and fee type: GET /series once, cached in reports/series.csv."""
    path = os.path.join(REPORTS, "series.csv")
    if not os.path.exists(path):
        r = requests.get(f"{API}/series", timeout=120)
        r.raise_for_status()
        pd.DataFrame([{"series": s["ticker"], "category": s.get("category", ""), "fee_type": s.get("fee_type", ""),
                       "fee_multiplier": s.get("fee_multiplier", 1)} for s in r.json()["series"]]).to_csv(path, index=False)
    return pd.read_csv(path)


def touches(books):
    """Best YES bid and YES ask (1 - best NO bid) with their sizes per (ts, ticker); two-sided uncrossed books only."""
    best = books.loc[books.groupby(["ts", "ticker", "side"])["price"].idxmax()]
    w = best.pivot(index=["ts", "ticker"], columns="side", values=["price", "size"])
    w = w.reindex(columns=pd.MultiIndex.from_product([["price", "size"], ["yes", "no"]]))
    t = pd.DataFrame({"bid": w[("price", "yes")], "bid_size": w[("size", "yes")],
                      "ask": (1 - w[("price", "no")]).round(4), "ask_size": w[("size", "no")]}).reset_index()
    return t[(t.bid > 0) & (t.ask < 1) & (t.bid < t.ask)].reset_index(drop=True)


def candidates(markets, series, predict, now):
    """The PREREG universe and rule per (snapshot, market): side and resting price, plus the funnel of drops."""
    m = markets
    f = {"rows": len(m)}
    m = m[(m.market_type == "binary") & (m.status == "active") & (m.yes_bid_dollars > 0) & (m.yes_ask_dollars < 1)
          & (m.yes_bid_dollars < m.yes_ask_dollars)].copy()
    f["binary_active_two_sided"] = len(m)
    m["spread"] = (m.yes_ask_dollars - m.yes_bid_dollars).round(4)
    m = m[m.spread >= MIN_SPREAD - 1e-9]
    f["spread_ge_10c"] = len(m)
    m["close"] = pd.to_datetime(m.close_time, utc=True)
    m["hours"] = (m.close - m.ts).dt.total_seconds() / 3600
    m = m[m.hours >= MIN_HOURS]
    f["ge_24h_to_close"] = len(m)
    m["series"] = m.ticker.str.split("-").str[0]
    m = m.merge(series, on="series", how="left")
    m = m[m.category.isin(CATEGORIES)]
    f["trained_category"] = len(m)
    m = m[m.close <= now]
    f["closed_by_now"] = len(m)
    m["mid"] = ((m.yes_bid_dollars + m.yes_ask_dollars) / 2).round(4)
    m["P"] = predict(m.mid.to_numpy())
    m["side"] = np.where(m.P > m.mid, "yes", np.where(m.P < m.mid, "no", ""))
    m = m[m.side != ""]
    f["model_off_mid"] = len(m)
    m["price"] = floor_cent(np.where(m.side == "yes", m.P, 1 - m.P) - MARGIN)
    m = m[m.price >= 0.01]
    f["price_ge_1c"] = len(m)
    # a resting order must rest: a price at or through the far touch would take, and the pre-reg is maker only
    takes = np.where(m.side == "yes", m.price >= m.yes_ask_dollars - 1e-9, m.price >= (1 - m.yes_bid_dollars) - 1e-9)
    m = m[~takes]
    f["rests_inside_book"] = len(m)
    m["yes_leg"] = np.where(m.side == "yes", m.price, 1 - m.price).round(4)   # the exchange quotes NO as a YES price
    return m.reset_index(drop=True), f


def simulate(cands, markets, touch, window_min=CANCEL_MIN, adverse=ADVERSE):
    """Walk each resting order through the later snapshots inside its window.

    At each snapshot: filled if the recorded touch is at or through the resting
    price (shadow.fill, size SIZE); else cancelled if the listing mid has moved
    `adverse` or more from the placement mid; else it keeps resting. Nothing in
    the window means it expires unfilled.
    """
    c = cands.reset_index(drop=True).copy()
    c["cid"] = c.index
    snaps = pd.DatetimeIndex(markets.ts.unique()).sort_values()
    win = pd.Timedelta(minutes=window_min)
    steps = pd.DataFrame([(t0, t, k) for t0 in c.ts.unique()
                          for k, t in enumerate(snaps[(snaps > t0) & (snaps <= t0 + win)])],
                         columns=["ts", "ts_k", "k"]).astype({"ts": c.ts.dtype, "ts_k": c.ts.dtype})
    ev = c[["cid", "ticker", "ts", "side", "yes_leg", "mid"]].merge(steps, on="ts")
    mk = markets.assign(mid_k=(markets.yes_bid_dollars + markets.yes_ask_dollars) / 2)
    ev = ev.merge(mk[["ticker", "ts", "mid_k"]].rename(columns={"ts": "ts_k"}), on=["ticker", "ts_k"], how="left")
    ev = ev.merge(touch.rename(columns={"ts": "ts_k"}), on=["ticker", "ts_k"], how="left")
    has = ev.bid.notna()
    ev["fill"] = False
    ev.loc[has, "fill"] = [shadow.fill("bid" if s == "yes" else "ask", p, SIZE, t) is not None
                           for s, p, t in zip(ev.side[has], ev.yes_leg[has],
                                              ev.loc[has, ["bid", "ask", "bid_size", "ask_size"]].to_dict("records"))]
    ev["cancel"] = (ev.mid_k - ev.mid).abs() >= adverse - 1e-9
    ev["event"] = np.select([ev.fill, ev.cancel], ["fill", "cancel"], "")
    first = ev[ev.event != ""].sort_values(["cid", "k"]).drop_duplicates("cid").set_index("cid")
    c["end"] = c.cid.map(first.event).fillna("expired")
    c["filled"] = c.end == "fill"
    c["fill_ts"] = pd.to_datetime(c.cid.map(first.ts_k.where(first.event == "fill")), utc=True)
    c["n_snaps"] = c.cid.map(ev.groupby("cid").size()).fillna(0).astype(int)
    c["n_books"] = c.cid.map(ev.groupby("cid").bid.count()).fillna(0).astype(int)
    return c.drop(columns="cid")


def next_touch(cands, touch):
    """Coverage diagnostic, no P&L: the first recorded book after placement at any lag, and whether it crossed."""
    t = touch.sort_values("ts").rename(columns={"ts": "ts_book"}).assign(ts=lambda d: d.ts_book)
    c = cands[["ticker", "ts", "side", "yes_leg"]].sort_values("ts")
    j = pd.merge_asof(c, t, on="ts", by="ticker", direction="forward", allow_exact_matches=False)
    j["lag_min"] = (j.ts_book - j.ts).dt.total_seconds() / 60
    has = j.ts_book.notna()
    j["crossed"] = False
    j.loc[has, "crossed"] = [shadow.fill("bid" if s == "yes" else "ask", p, SIZE, tt) is not None
                             for s, p, tt in zip(j.side[has], j.yes_leg[has],
                                                 j.loc[has, ["bid", "ask", "bid_size", "ask_size"]].to_dict("records"))]
    return {"with_later_book": int(has.sum()), "median_lag_min": float(j.lag_min.median()) if has.any() else None,
            "crossed_any_lag": int(j.crossed.sum()), "crossed_within_75min": int((j.crossed & (j.lag_min <= 75)).sum()),
            "crossed_within_120min": int((j.crossed & (j.lag_min <= 120)).sum())}


def results(tickers):
    """Settlement result per ticker from GET /markets/{ticker}, cached in reports/results.csv."""
    path = os.path.join(REPORTS, "results.csv")
    cols = ["ticker", "status", "result", "settlement_ts"]
    have = pd.read_csv(path, dtype=str).fillna("") if os.path.exists(path) else pd.DataFrame(columns=cols)
    done = set(have.loc[have.result.isin(["yes", "no"]), "ticker"])
    rows = []
    for t in [t for t in dict.fromkeys(tickers) if t not in done]:
        for attempt in range(4):
            try:
                r = requests.get(f"{API}/markets/{t}", timeout=25)
            except requests.RequestException:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.0 * (attempt + 1))
                continue
            m = r.json().get("market", {}) if r.ok else {}
            rows.append({"ticker": t, "status": m.get("status", ""), "result": m.get("result") or "",
                         "settlement_ts": m.get("settlement_ts", "")})
            break
        time.sleep(SLEEP)
    if rows:
        have = pd.concat([have[~have.ticker.isin({r["ticker"] for r in rows})], pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(path, index=False)
    return have.set_index("ticker")["result"]


def settle(c, res):
    """Cost, maker fee, stake and settled P&L per filled contract; payoff is P&L per dollar staked."""
    c = c.copy()
    c["result"] = c.ticker.map(res).where(c.filled, "").fillna("")
    c["cost"] = np.where(c.filled, c.price, np.nan)
    c["fee"] = np.where(c.filled, maker_fee(c.price, c.fee_type.fillna(""), c.fee_multiplier.fillna(1)), np.nan)
    c["stake"] = c.cost + c.fee
    settled = c.filled & c.result.isin(["yes", "no"])
    c["pnl"] = np.where(settled, np.where(c.result == c.side, 1 - c.stake, -c.stake), np.nan)
    c["payoff"] = c.pnl / c.stake
    c["kelly_f"] = np.where(c.filled, kelly.kelly_fraction(np.where(c.side == "yes", c.P, 1 - c.P), c.stake), np.nan)
    c["bucket"] = np.where(c.spread <= 0.20 + 1e-9, "10-20c", ">20c")
    return c


def summarise(c):
    """The PREREG §5 numbers, the daily series, and the spread-bucket and category tables."""
    f, s = c[c.filled], c[c.pnl.notna()]
    edge = kelly.per_bet_edge(s)
    daily = s.groupby(s.fill_ts.dt.date).agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    daily["ret"] = daily.pnl / daily.stake
    sd = daily.ret.std(ddof=1) if len(daily) > 1 else np.nan
    out = {"candidates": int(len(c)), "observable": int((c.n_books > 0).sum()), "fills": int(len(f)),
           "fill_rate": float(len(f) / len(c)) if len(c) else np.nan,
           "fill_rate_observable": float(len(f) / (c.n_books > 0).sum()) if (c.n_books > 0).any() else np.nan,
           "unsettled_fills": int(len(f) - len(s)), "pnl_usd": float(s.pnl.sum()), "staked_usd": float(s.stake.sum()),
           "net_per_dollar": float(s.pnl.sum() / s.stake.sum()) if len(s) else np.nan,
           "mean_payoff": float(edge["mean_payoff"]), "t": float(edge["t"]), "n_settled": int(edge["n"]),
           "days": int(len(daily)),
           "daily_sharpe": float(np.sqrt(kelly.DAYS_PER_YEAR) * daily.ret.mean() / sd) if sd and sd > 0 else np.nan,
           "kelly_f_mean": float(f.kelly_f.mean()) if len(f) else np.nan}

    def table(key):
        g = c.groupby(key)
        t = pd.DataFrame({"candidates": g.size(), "observable": g.n_books.apply(lambda x: int((x > 0).sum())),
                          "fills": g.filled.sum().astype(int)})
        t["mean_payoff"] = s.groupby(key).payoff.mean()
        t["t"] = s.groupby(key).payoff.apply(
            lambda p: p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 and p.std() > 0 else np.nan)
        return t.sort_values("candidates", ascending=False)
    return out, daily, table("bucket"), table("category")


def verdict(out):
    if out["fills"] < MIN_FILLS:
        return f"INSUFFICIENT FILLS: {out['fills']} fills < {MIN_FILLS} required (PREREG §5)"
    ok = out["net_per_dollar"] > 0 and out["t"] > T_MIN
    return (f"{'CLEARS' if ok else 'DOES NOT CLEAR'}: net {out['net_per_dollar']:+.4f} per $ staked, "
            f"t = {out['t']:.2f} on {out['fills']} fills (needs > 0 and t > {T_MIN})")


def model_id(entry):
    return os.path.splitext(entry["file"])[0]


def main(window_min=CANCEL_MIN):
    os.makedirs(REPORTS, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    model, entry = km.registered("chosen")
    mid = model_id(entry)
    if not isinstance(model, km.BiasCurve):
        raise SystemExit(f"chosen model {mid} is {type(model).__name__}, which reads path features; this replay "
                         "prices only the mid-only BiasCurve (candles would be needed; see README)")
    markets = lake.load("kalshi_markets", START, None)
    books = lake.load("kalshi_books", START, None)
    for df in (markets, books):
        df["ts"] = df.ts.astype("datetime64[ns, UTC]")
    series = series_table()
    touch = touches(books)
    snaps = pd.DatetimeIndex(markets.ts.unique()).sort_values()
    gap = pd.Series(snaps[1:] - snaps[:-1]).dt.total_seconds().div(60)

    cands, funnel = candidates(markets, series, lambda m: model.predict(pd.DataFrame({"price": m})), now)
    sim = simulate(cands, markets, touch, window_min)
    res = results(sim.loc[sim.filled, "ticker"].tolist())
    c = settle(sim, res)
    out, daily, by_bucket, by_cat = summarise(c)
    cover = next_touch(c, touch)
    second = window_min != CANCEL_MIN
    label = "" if not second else f"-window{window_min}"

    cfg = {"margin": MARGIN, "cancel_min": window_min, "adverse": ADVERSE, "min_spread": MIN_SPREAD,
           "min_hours": MIN_HOURS, "size": SIZE, "model": mid, "fill": "recorded touches only, shadow.fill",
           "fee": f"maker {MAKER_FEE} x series multiplier, July 2026 schedule, 0 on series without maker fees"}
    reg = Registry(os.path.join(REPORTS, "registry"))
    row = reg.record("KALSHI-WIDE", cfg, ["kalshi-lake-wide-books"], (snaps[0], snaps[-1]), daily.ret,
                     metrics={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()},
                     tags={"stage": "prior", "second_look": second},
                     code=code_hash(sys.modules[__name__], shadow))  # the replay and the desk's fill rule

    cols = ["ts", "ticker", "series", "category", "spread", "P", "mid", "side", "price", "filled", "fill_ts", "end",
            "n_snaps", "n_books", "cost", "fee", "stake", "result", "pnl", "payoff", "kelly_f", "bucket"]
    c[cols].to_csv(os.path.join(REPORTS, f"replay{label}.csv"), index=False)
    summary = {"prereg": cfg, "second_look": second, "as_of": str(now), "model": {"id": mid, "class": type(model).__name__},
               "lake": {"start": str(snaps[0]), "end": str(snaps[-1]), "snapshots": int(len(snaps)),
                        "median_gap_min": float(gap.median()), "gaps_within_window": int((gap <= window_min).sum()),
                        "books_per_snapshot": float(touch.groupby("ts").size().median())},
               "funnel": funnel, "results": out, "coverage_next_book": cover,
               "by_bucket": by_bucket.reset_index().to_dict("records"),
               "by_category": by_cat.reset_index().to_dict("records"),
               "registry": {"id": row["id"], "n_trials": row["n_trials"]}, "verdict": verdict(out)}
    with open(os.path.join(REPORTS, f"summary{label}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    pd.set_option("display.width", 160)
    print(f"KALSHI-WIDE prior layer{' SECOND LOOK (not the pre-registered test)' if second else ''}: "
          f"margin {MARGIN:.2f}, cancel {window_min} min, adverse {ADVERSE:.2f}, books >= {MIN_SPREAD:.2f}, "
          f">= {MIN_HOURS}h to close, {SIZE} contract")
    print(f"lake {snaps[0]:%Y-%m-%d %H:%M} -> {snaps[-1]:%Y-%m-%d %H:%M} UTC, {len(snaps)} snapshots, median gap "
          f"{gap.median():.1f} min, {int((gap <= window_min).sum())} gaps inside the {window_min}-min window; "
          f"{summary['lake']['books_per_snapshot']:.0f} two-sided books a snapshot")
    print(f"model {mid} {type(model).__name__} (role chosen); maker fee {MAKER_FEE} x M on "
          f"{int(c.fee_type.fillna('').str.contains('maker').sum())} candidate rows, 0 on the rest")
    print("funnel: " + " -> ".join(f"{k} {v:,}" for k, v in funnel.items()))
    print(f"\ncandidates {out['candidates']:,} | observable (a book in the window) {out['observable']:,} | "
          f"fills {out['fills']:,} | fill rate {out['fill_rate']:.4%} ({out['fill_rate_observable']:.2%} of observable)")
    print(f"net P&L ${out['pnl_usd']:+.2f} on ${out['staked_usd']:.2f} staked = {out['net_per_dollar']:+.4f} per $ | "
          f"mean payoff per $ {out['mean_payoff']:+.4f}, t = {out['t']:.2f} on {out['n_settled']} settled "
          f"({out['unsettled_fills']} unsettled) | daily Sharpe {out['daily_sharpe']:.2f} over {out['days']} days | "
          f"mean Kelly f {out['kelly_f_mean']:.3f}")
    print("\nby spread bucket\n" + by_bucket.round(4).to_string())
    print("\nby category\n" + by_cat.round(4).to_string())
    print(f"\ncoverage diagnostic (no P&L): {cover['with_later_book']:,} candidates have any later recorded book "
          f"(median lag {cover['median_lag_min']} min); the touch there is at or through the resting price for "
          f"{cover['crossed_any_lag']:,} ({cover['crossed_within_75min']:,} within 75 min, "
          f"{cover['crossed_within_120min']:,} within 120 min)")
    print(f"registry {row['id']} ({row['n_trials']} trials logged)")
    print(f"\nVERDICT: {summary['verdict']}")
    return summary


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else CANCEL_MIN)
