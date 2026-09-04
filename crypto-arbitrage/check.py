"""Offline checks on synthetic quote feeds: alignment, a planted gap, fee arithmetic, no trades on no gap.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from arb import align, best_gap, executable_gap_bps, gap_persistence, mid_spread_bps, runs, simulate

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


T0 = pd.Timestamp("2026-09-04 00:00:00", tz="UTC")


def feed(rows):
    """Build a long quote frame from (offset_s, exchange, bid, ask) tuples for one symbol."""
    return pd.DataFrame([dict(t_local=T0 + pd.Timedelta(seconds=s), exchange=ex, symbol="BTC",
                              bid=b, ask=a) for s, ex, b, a in rows])


# Venue A quotes at 0.1s past each 3s boundary, venue B at 0.9s past. On the 3s
# grid the quote for time 3 must be the one stamped before 3, never the one at 3.1.
stagger = feed([(0.1, "A", 100.0, 100.2), (3.1, "A", 101.0, 101.2), (6.1, "A", 102.0, 102.2),
                (0.9, "B", 200.0, 200.2), (3.9, "B", 201.0, 201.2), (6.9, "B", 202.0, 202.2)])
p = align(stagger, freq="3s", tolerance="6s")["BTC"]
check("grid starts on the first whole boundary after the first quote",
      p.index[0] == T0 + pd.Timedelta(seconds=3))
check("alignment uses the last quote at or before the grid time, not the next one",
      p.loc[T0 + pd.Timedelta(seconds=3), ("A", "bid")] == 100.0
      and p.loc[T0 + pd.Timedelta(seconds=3), ("B", "bid")] == 200.0
      and p.loc[T0 + pd.Timedelta(seconds=6), ("A", "bid")] == 101.0)
check("quote age is measured from its own timestamp",
      np.isclose(p.loc[T0 + pd.Timedelta(seconds=3), ("A", "age_s")], 2.9)
      and np.isclose(p.loc[T0 + pd.Timedelta(seconds=3), ("B", "age_s")], 2.1))

silent = feed([(s, "A", 100.0, 100.2) for s in range(0, 60, 3)]
              + [(s, "B", 100.0, 100.2) for s in range(0, 60, 3) if s < 21 or s > 45])
ps = align(silent, freq="3s", tolerance="6s")["BTC"]
check("rows where a venue is silent longer than the tolerance are dropped",
      len(ps) < 19 and (ps[("B", "age_s")] <= 6).all() and (ps[("A", "age_s")] <= 6).all())

# Planted gap: three venues tick together; venue X is 20bps rich from sample 100
# through 150 inclusive, otherwise all three quote the same 2bps-wide book.
N, DT = 400, 3
px = 50_000.0
rows = []
for i in range(N):
    for ex in ("X", "Y", "Z"):
        lvl = px * (1 + 20e-4) if ex == "X" and 100 <= i <= 150 else px
        rows.append((i * DT, ex, lvl * (1 - 1e-4), lvl * (1 + 1e-4)))
pp = align(feed(rows), freq="3s", tolerance="6s")["BTC"]
ms = mid_spread_bps(pp)
bg = best_gap(pp)
check("mid spread X-Y is +20bps inside the planted window and 0 outside",
      np.allclose(ms["X-Y"].iloc[100:151], 20.0, atol=0.05) and np.allclose(ms["X-Y"].iloc[:100], 0.0))
check("best executable gap names the right pair and direction (sell rich X, buy cheap Y or Z)",
      set(bg["pair"].iloc[100:151]) <= {"sell@X/buy@Y", "sell@X/buy@Z"})
check("executable gap equals the planted 20bps less the two half-spreads (about 18bps)",
      np.allclose(bg["bps"].iloc[100:151], (px * 1.002 * 0.9999 - px * 1.0001) / (px * 1.0001) * 1e4, atol=0.05))
check("outside the window the best executable gap is negative (books overlap)",
      (bg["bps"].iloc[:100] < 0).all() and (bg["bps"].iloc[151:] < 0).all())
check("run lengths count consecutive samples correctly",
      list(runs([0, 1, 1, 0, 1, 1, 1, 0, 0, 1])) == [2, 3, 1] and len(runs([0, 0])) == 0)
pers = gap_persistence(bg["bps"], [10.0], DT)
check(f"the planted gap is one episode lasting 51 samples = {51 * DT}s",
      pers["episodes"].iloc[0] == 1 and pers["max_s"].iloc[0] == 51 * DT
      and np.isclose(pers["share_of_samples"].iloc[0], 51 / len(pp)))

# Fee arithmetic by hand. One venue pair, buy at ask 100, sell at bid 100.5, 10bps
# each leg, $10,000 lot: qty 100, cost 10,010.00, proceeds 10,039.95, net 29.95.
hand = feed([(0, "A", 100.5, 100.6), (0, "B", 99.9, 100.0),
             (3, "A", 100.5, 100.6), (3, "B", 99.9, 100.0),
             (6, "A", 100.0, 100.1), (6, "B", 100.0, 100.1)])
ph = align(hand, freq="3s", tolerance="6s")["BTC"]
log, s = simulate(ph, {"A": 10.0, "B": 10.0}, notional=10_000.0, units=1)
check("hand case: one trade, net $29.95",
      s["trades"] == 1 and np.isclose(log["net_usd"].iloc[0], 29.95, atol=1e-6))
check("hand case: fills come from the sample after the signal",
      log["signal_time"].iloc[0] == ph.index[0] and log["fill_time"].iloc[0] == ph.index[1])
check("a gap that beats fees before but not after the fee is ignored",
      simulate(ph, {"A": 30.0, "B": 30.0}, units=1)[1]["trades"] == 0)

# The same 50bps gap seen for exactly one sample is gone by the time we fill. The
# signal fires on sample 0 but the fill on sample 1 should lose the spread.
blink = feed([(0, "A", 100.5, 100.6), (0, "B", 99.9, 100.0),
              (3, "A", 100.0, 100.1), (3, "B", 100.0, 100.1),
              (6, "A", 100.0, 100.1), (6, "B", 100.0, 100.1)])
lb, sb = simulate(align(blink, freq="3s", tolerance="6s")["BTC"], {"A": 0.0, "B": 0.0}, units=1)
check("no look-ahead: a one-sample gap fills at the next sample's prices and loses money",
      sb["trades"] == 1 and lb["net_usd"].iloc[0] < 0)

# Inventory: a gap that never closes can only be taken as many times as there is
# coin at the rich venue and cash at the cheap one.
lp, sp = simulate(pp, {"X": 0.0, "Y": 0.0, "Z": 0.0}, units=2)
check("prepositioned inventory caps the trade count; the rest are blocked",
      sp["trades"] == 2 and sp["blocked"] > 0 and sp["signals"] == sp["trades"] + sp["blocked"])

# Transfer mode: buy at sample t+1, sell at the first sample at least transfer_s later.
lt, st = simulate(pp, {"X": 0.0, "Y": 0.0, "Z": 0.0}, units=1, mode="transfer",
                  transfer_s=30, withdraw_fee=0.0)
check("transfer mode sells exactly transfer_s after the fill",
      st["trades"] == 1 and (lt["sell_time"].iloc[0] - lt["fill_time"].iloc[0]).total_seconds() == 30)
lt2, st2 = simulate(pp, {"X": 0.0, "Y": 0.0, "Z": 0.0}, units=1, mode="transfer",
                    transfer_s=DT * (N + 5), withdraw_fee=0.0)
check("transfers that land after the window are reported as unresolved, not traded",
      st2["trades"] == 0 and st2["unresolved"] == st2["signals"] == 51)

# Every venue quoting the identical book: no gap, no trades, at any fee.
flat = feed([(i * DT, ex, 100.0, 100.02) for i in range(50) for ex in ("A", "B", "C")])
pf = align(flat, freq="3s", tolerance="6s")["BTC"]
check("zero-gap feed produces zero trades even with zero fees",
      simulate(pf, {"A": 0.0, "B": 0.0, "C": 0.0})[1]["trades"] == 0
      and (executable_gap_bps(pf) < 0).all().all())

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
