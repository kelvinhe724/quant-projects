"""Offline checks on planted frames: fill only on a crossing touch inside the window, the 3c cancel, the 60-minute
expiry, the maker fee and payoff arithmetic, the >= 10c and 24h filters, and the frozen model id.

Run: ../.venv/bin/python3 check.py   (exit 1 on any failure)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay  # noqa: E402  puts ROOT, kalshi-model and kalshi-desk on sys.path
import candles  # noqa: E402
import model as km  # noqa: E402
from research.models import ModelRegistry  # noqa: E402

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


T0 = pd.Timestamp("2026-09-10 12:00", tz="UTC")
NOW = pd.Timestamp("2026-10-01", tz="UTC")


def at(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def book(ts, ticker, yes_bid, no_bid, yes_size=10.0, no_size=10.0):
    """Lake book rows: side yes = YES bids, side no = NO bids (a NO bid at y is a YES ask at 1 - y)."""
    return [(ts, ticker, "yes", yes_bid, yes_size), (ts, ticker, "no", no_bid, no_size)]


def listing(ts, ticker, bid, ask, close=None):
    return {"ts": ts, "ticker": ticker, "market_type": "binary", "status": "active", "yes_bid_dollars": bid,
            "yes_ask_dollars": ask, "close_time": (close or T0 + pd.Timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")}


# touches: best bid, 1 - best NO bid, sizes; one-sided and crossed books dropped
rows = book(T0, "A", 0.30, 0.60, 5, 7) + [(T0, "A", "yes", 0.25, 99), (T0, "A", "no", 0.55, 99)]
rows += [(T0, "B", "yes", 0.4, 1)] + book(T0, "C", 0.50, 0.60)
t = replay.touches(pd.DataFrame(rows, columns=["ts", "ticker", "side", "price", "size"]))
a = t.set_index("ticker").loc["A"]
check("touch: best YES bid 0.30 x 5, YES ask 1 - 0.60 = 0.40 x 7",
      a.bid == 0.30 and a.bid_size == 5 and abs(a.ask - 0.40) < 1e-9 and a.ask_size == 7)
check("touch: a one-sided book and a crossed book are not touches", set(t.ticker) == {"A"})

# the rule: side, resting price, filters
SER = pd.DataFrame({"series": ["KXS", "KXZ"], "category": ["Sports", "Science and Technology"],
                    "fee_type": ["quadratic", "quadratic"], "fee_multiplier": [1, 1]})
mk = pd.DataFrame([listing(T0, "KXS-YES", 0.40, 0.60), listing(T0, "KXS-NO", 0.40, 0.60), listing(T0, "KXS-TAKE", 0.40, 0.60),
                   listing(T0, "KXS-TEN", 0.20, 0.30), listing(T0, "KXS-NINE", 0.20, 0.29),
                   listing(T0, "KXS-23H", 0.40, 0.60, T0 + pd.Timedelta(hours=23)),
                   listing(T0, "KXS-24H", 0.40, 0.60, T0 + pd.Timedelta(hours=24)),
                   listing(T0, "KXZ-CAT", 0.40, 0.60), listing(T0, "KXS-OPEN", 0.40, 0.60, NOW + pd.Timedelta(days=2))])
P = {"KXS-YES": 0.60, "KXS-NO": 0.40, "KXS-TAKE": 0.63, "KXS-TEN": 0.30, "KXS-NINE": 0.30, "KXS-23H": 0.6,
     "KXS-24H": 0.6, "KXZ-CAT": 0.6, "KXS-OPEN": 0.6}
# predict is called after the filters, on the survivors in listing order
kept = [t for t in mk.ticker if t not in ("KXS-NINE", "KXS-23H", "KXZ-CAT", "KXS-OPEN")]
c, funnel = replay.candidates(mk, SER, lambda mid: np.array([P[t] for t in kept]), NOW)
c = c.set_index("ticker")
check("rule: P 0.60 > mid 0.50 rests YES at floor(0.58) = 0.58",
      c.loc["KXS-YES", "side"] == "yes" and c.loc["KXS-YES", "price"] == 0.58 and c.loc["KXS-YES", "yes_leg"] == 0.58)
check("rule: P 0.40 < mid rests NO at 0.58, quoted as YES-leg 0.42",
      c.loc["KXS-NO", "side"] == "no" and c.loc["KXS-NO", "price"] == 0.58 and abs(c.loc["KXS-NO", "yes_leg"] - 0.42) < 1e-9)
check("rule: P 0.63 puts the bid at 0.61 >= ask 0.60, a taker, not a candidate", "KXS-TAKE" not in c.index)
check("filter: a 0.20/0.30 book (spread 0.10 in floating point) is a candidate, 0.20/0.29 is not",
      "KXS-TEN" in c.index and "KXS-NINE" not in c.index)
check("filter: 23h to close is out, 24h is in", "KXS-23H" not in c.index and "KXS-24H" in c.index)
check("filter: a category the model never saw is out", "KXZ-CAT" not in c.index)
check("filter: a market not yet closed is out", "KXS-OPEN" not in c.index)
check("funnel counts every drop", funnel["rows"] == 9 and funnel["spread_ge_10c"] == 8 and funnel["ge_24h_to_close"] == 7
      and funnel["trained_category"] == 6 and funnel["closed_by_now"] == 5 and funnel["rests_inside_book"] == 4)

# the resting order's life: fill, cancel, expiry on planted snapshots at +30, +45, +61 minutes
tick = ["FILL", "LATE", "CANCEL", "SIZE", "SAME", "NOFILL", "ADV2", "ADV3", "NOBOOK"]
mrows = [listing(at(m), t, 0.40, 0.60) for m in (0, 30, 45, 61) for t in tick]
mk2 = pd.DataFrame(mrows)
mk2.loc[(mk2.ticker == "CANCEL") & (mk2.ts == at(30)), ["yes_bid_dollars", "yes_ask_dollars"]] = [0.44, 0.64]  # mid +4c
mk2.loc[(mk2.ticker == "SAME") & (mk2.ts == at(30)), ["yes_bid_dollars", "yes_ask_dollars"]] = [0.37, 0.57]    # mid -3c
mk2.loc[(mk2.ticker == "ADV2") & (mk2.ts == at(30)), ["yes_bid_dollars", "yes_ask_dollars"]] = [0.42, 0.62]    # mid +2c
mk2.loc[(mk2.ticker == "ADV3") & (mk2.ts == at(30)), ["yes_bid_dollars", "yes_ask_dollars"]] = [0.43, 0.63]    # mid +3c
brows = (book(at(30), "FILL", 0.40, 0.38) + book(at(45), "FILL", 0.40, 0.42)        # ask 0.62 then 0.58: fills at +45
         + book(at(61), "LATE", 0.40, 0.42)                                          # crosses only at +61: expired
         + book(at(30), "CANCEL", 0.44, 0.38) + book(at(45), "CANCEL", 0.44, 0.42)  # would cross at +45, cancelled at +30
         + book(at(30), "SIZE", 0.40, 0.42, 10, 0.5)                                 # ask 0.58 but 0.5 contracts
         + book(at(30), "SAME", 0.37, 0.43)                                          # ask 0.57 crosses and mid -3c: fill wins
         + book(at(30), "NOFILL", 0.41, 0.42) + book(at(45), "NOFILL", 0.42, 0.42))  # NO order: YES bid reaches 0.42 at +45
cands = pd.DataFrame({"ts": at(0), "ticker": tick, "side": ["yes"] * 5 + ["no"] + ["yes"] * 3,
                      "yes_leg": [0.58] * 5 + [0.42] + [0.58] * 3, "mid": 0.50})
cands["ts"] = cands.ts.astype("datetime64[ns, UTC]")
tb = replay.touches(pd.DataFrame(brows, columns=["ts", "ticker", "side", "price", "size"]))
s = replay.simulate(cands, mk2, tb).set_index("ticker")
check("fill: a touch 4c above the bid does not fill, the touch at the bid 45 minutes later does",
      s.loc["FILL", "filled"] and s.loc["FILL", "fill_ts"] == at(45) and s.loc["FILL", "n_books"] == 2)
check("expiry: a crossing touch 61 minutes after placement is outside the window",
      not s.loc["LATE", "filled"] and s.loc["LATE", "end"] == "expired" and s.loc["LATE", "n_snaps"] == 2)
check("cancel: a 4c mid move at +30 cancels before the +45 touch would have filled",
      not s.loc["CANCEL", "filled"] and s.loc["CANCEL", "end"] == "cancel")
check("size: a touch at the price with half a contract behind it is not a fill (shadow.fill)",
      not s.loc["SIZE", "filled"])
check("same snapshot: a fill and a 3c move together count as the fill", s.loc["SAME", "filled"])
check("NO order: fills when the YES bid rises to the YES-leg price, not before",
      s.loc["NOFILL", "filled"] and s.loc["NOFILL", "fill_ts"] == at(45))
check("adverse: a 2c move keeps resting, a 3c move cancels",
      s.loc["ADV2", "end"] == "expired" and s.loc["ADV3", "end"] == "cancel")
check("no recorded book in the window: not observable, not filled",
      s.loc["NOBOOK", "n_books"] == 0 and not s.loc["NOBOOK", "filled"])

# fee and payoff arithmetic
check("maker fee: 0.0175 x 0.5 x 0.5 = 0.004375 rounds up to 0.0044 on a maker-fee series, 0 elsewhere",
      replay.maker_fee(0.5, "quadratic_with_maker_fees", 1)[()] == 0.0044 and replay.maker_fee(0.5, "quadratic", 1)[()] == 0)
check("maker fee: multiplier 0.5 at 0.30 gives 0.0175 x 0.5 x 0.21 = 0.0018375 -> 0.0019",
      replay.maker_fee(0.30, "quadratic_with_maker_fees", 0.5)[()] == 0.0019)
f = pd.DataFrame({"ticker": ["W", "L", "N", "U", "X"], "filled": [True, True, True, True, False],
                  "side": ["yes", "yes", "no", "yes", "yes"], "price": [0.30, 0.30, 0.58, 0.30, 0.30],
                  "fee_type": ["quadratic", "quadratic", "quadratic_with_maker_fees", "quadratic", "quadratic"],
                  "fee_multiplier": [1.0] * 5, "P": [0.4, 0.4, 0.3, 0.4, 0.4], "spread": [0.2, 0.2, 0.3, 0.2, 0.2]})
f = replay.settle(f, pd.Series({"W": "yes", "L": "no", "N": "no", "U": ""})).set_index("ticker")
check("payoff: a YES win at 0.30 with no fee earns 0.70 = 2.3333 per $ staked",
      abs(f.loc["W", "pnl"] - 0.70) < 1e-9 and abs(f.loc["W", "payoff"] - 0.70 / 0.30) < 1e-9)
check("payoff: a loss costs the stake, -1 per $", abs(f.loc["L", "pnl"] + 0.30) < 1e-9 and f.loc["L", "payoff"] == -1)
check("payoff: a NO win at 0.58 on a maker-fee series pays 1 - 0.58 - 0.0043 = 0.4157",
      abs(f.loc["N", "fee"] - 0.0043) < 1e-9 and abs(f.loc["N", "pnl"] - 0.4157) < 1e-9)
check("payoff: an unsettled fill and an unfilled order have no P&L",
      np.isnan(f.loc["U", "pnl"]) and np.isnan(f.loc["X", "pnl"]) and np.isnan(f.loc["X", "stake"]))
check("bucket: 0.20 is 10-20c, 0.30 is >20c", f.loc["W", "bucket"] == "10-20c" and f.loc["N", "bucket"] == ">20c")
check("verdict: fewer than 300 fills is INSUFFICIENT whatever the t",
      replay.verdict({"fills": 299, "net_per_dollar": 0.1, "t": 9.0}).startswith("INSUFFICIENT"))
check("verdict: 300 fills clear only with net > 0 and t > 2",
      replay.verdict({"fills": 300, "net_per_dollar": 0.01, "t": 2.01}).startswith("CLEARS")
      and replay.verdict({"fills": 300, "net_per_dollar": 0.01, "t": 1.99}).startswith("DOES NOT")
      and replay.verdict({"fills": 300, "net_per_dollar": -0.01, "t": 2.5}).startswith("DOES NOT"))

# the frozen model: the id the replay uses is the registry's "chosen", and it prices from the mid alone
model, entry = km.registered("chosen")
e = ModelRegistry(os.path.join(km.REPORTS, "models")).entries()
chosen = e[e["meta"].apply(lambda m: m.get("role") == "chosen")].iloc[-1]["id"]
check(f"model: replay uses {replay.model_id(entry)} = registry chosen {chosen}", replay.model_id(entry) == chosen)
mids = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
rng = np.random.default_rng(0)
full = pd.DataFrame({k: rng.normal(size=len(mids)) for k in candles.FEATURES}).assign(price=mids)
check("model: the chosen calibrator is the mid-only BiasCurve (path features change nothing, so no candles are needed)",
      isinstance(model, km.BiasCurve) and np.allclose(model.predict(full), model.predict(pd.DataFrame({"price": mids}))))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
