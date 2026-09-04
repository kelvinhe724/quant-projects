"""Offline checks on a synthetic book with a planted imbalance-drift relationship.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from imbalance import (backtest, binned_expectation, forward_index, forward_move_bps,
                       half_spread_bps, hit_rate, imbalance, mid_price, regress, summarise)

rng = np.random.default_rng(3)
N = 6000
LEVELS = 20
TICK = 0.1
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def synthetic_book(slope_bps, spread_ticks=1, noise_bps=0.3, price0=80_000.0):
    """Build N one-second snapshots where next-second mid drift is slope_bps * imbalance."""
    # imbalance persists (AR(1)), so a signal read at t still says something about
    # the move after the fill at t+1
    imb = np.zeros(N)
    for t in range(1, N):
        imb[t] = 0.6 * imb[t - 1] + rng.normal(0, 0.4)
    imb = np.clip(imb, -0.95, 0.95)
    drift = slope_bps * imb / 1e4
    ret = np.zeros(N)
    ret[1:] = drift[:-1] + rng.normal(0, noise_bps / 1e4, N - 1)
    mid = price0 * np.exp(np.cumsum(ret))
    mid = np.round(mid / TICK) * TICK + (spread_ticks % 2) * TICK / 2
    half = spread_ticks * TICK / 2
    ladder = TICK * np.arange(LEVELS)
    bid_px = mid[:, None] - half - ladder
    ask_px = mid[:, None] + half + ladder
    total = 10.0
    bid_sz = np.tile(((1 + imb) * total / 2 / LEVELS)[:, None], (1, LEVELS))
    ask_sz = np.tile(((1 - imb) * total / 2 / LEVELS)[:, None], (1, LEVELS))
    ts = np.arange(N) * 1000
    return bid_px, bid_sz, ask_px, ask_sz, ts, imb


print("BOOK ARITHMETIC\n")

bp = np.array([[100.0, 99.9, 99.8]])
ap = np.array([[100.2, 100.3, 100.4]])
check("mid of 100.0 / 100.2 is 100.1", np.isclose(mid_price(bp, ap)[0], 100.1))
check("half-spread of a 0.2 spread on mid 100.1 is 9.99 bps",
      np.isclose(half_spread_bps(bp, ap)[0], 0.1 / 100.1 * 1e4))

bs = np.array([[3.0, 1.0, 1.0]])
as_ = np.array([[1.0, 1.0, 1.0]])
check("depth-1 imbalance of 3 vs 1 is +0.5", np.isclose(imbalance(bs, as_, 1)[0], 0.5))
check("depth-3 imbalance of 5 vs 3 is +0.25", np.isclose(imbalance(bs, as_, 3)[0], 0.25))
check("a symmetric book has imbalance 0 at every depth",
      all(imbalance(bs, bs, d)[0] == 0 for d in (1, 2, 3)))
check("imbalance flips sign when the sides swap",
      np.isclose(imbalance(as_, bs, 3)[0], -0.25))
check("an empty top level gives 0, not a division error",
      imbalance(np.zeros((1, 2)), np.zeros((1, 2)), 2)[0] == 0)
check("imbalance lives in [-1, 1]",
      imbalance(np.array([[5.0]]), np.array([[0.0]]), 1)[0] == 1
      and imbalance(np.array([[0.0]]), np.array([[5.0]]), 1)[0] == -1)

ts = np.array([0, 1000, 2100, 3000, 5000])
check("forward index at 2s from t=0 lands on the first snapshot >= 2000ms (index 2)",
      list(forward_index(ts, 2)) == [2, 3, 4, 4, -1])
px = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
fwd = forward_move_bps(px, ts, 2)
check("forward move of 100 -> 102 is 200 bps, and past the end is nan",
      np.isclose(fwd[0], 200.0) and np.isnan(fwd[-1]))

print("\nPLANTED RELATIONSHIP\n")

SLOPE = 2.0
bid_px, bid_sz, ask_px, ask_sz, ts, true_imb = synthetic_book(SLOPE)
mid = mid_price(bid_px, ask_px)
for depth in (1, 5, 20):
    check(f"depth-{depth} imbalance recovers the planted imbalance",
          np.allclose(imbalance(bid_sz, ask_sz, depth), true_imb, atol=1e-9))

imb = imbalance(bid_sz, ask_sz, 5)
y1 = forward_move_bps(mid, ts, 1)
fit = regress(imb, y1, 1)
check(f"1s regression recovers the planted slope of {SLOPE} bps (got {fit['slope']:.2f})",
      abs(fit["slope"] - SLOPE) < 0.15)
check(f"the planted slope is highly significant (t = {fit['t_stat']:.1f})", fit["t_stat"] > 10)
check(f"R2 is positive and bounded ({fit['r2']:.3f})", 0 < fit["r2"] < 1)

fit5 = regress(imb, forward_move_bps(mid, ts, 5), 5)
check(f"a 5s horizon accumulates the drift while the imbalance persists (got {fit5['slope']:.2f})",
      SLOPE < fit5["slope"] < 3 * SLOPE)

table = binned_expectation(imb, y1)
check("binned conditional expectation rises monotonically in imbalance",
      table["move_bps"].is_monotonic_increasing)
check(f"hit rate on planted data is well above a coin flip ({hit_rate(imb, y1):.2f})",
      hit_rate(imb, y1) > 0.6)

shuffled = rng.permutation(imb)
null = regress(shuffled, y1, 1)
check(f"shuffling the signal kills the slope (got {null['slope']:.2f}, t = {null['t_stat']:.1f})",
      abs(null["t_stat"]) < 3)

flat_bid, flat_bsz, flat_ask, flat_asz, _, _ = synthetic_book(0.0)
zero = regress(imbalance(flat_bsz, flat_asz, 5), forward_move_bps(mid_price(flat_bid, flat_ask), ts, 1), 1)
check(f"a book with no planted drift gives no slope (t = {zero['t_stat']:.1f})",
      abs(zero["t_stat"]) < 3)

print("\nEXECUTION AND SPREAD\n")

trades = backtest(imb, bid_px, ask_px, ts, horizon_s=1, threshold=0.5)
check("every fill happens one snapshot after its signal", (trades["entry"] == trades["signal_t"] + 1).all())
check("every trade closes at least the horizon after entry",
      (ts[trades["exit"]] - ts[trades["entry"]] >= 1000).all())
check("positions never overlap", (trades["entry"].iloc[1:].to_numpy() >= trades["exit"].iloc[:-1].to_numpy()).all())
check("the strategy only fires on extreme imbalance", (trades["imbalance"].abs() > 0.5).all())
check("side follows the sign of the imbalance", (trades["side"] == np.sign(trades["imbalance"])).all())

first = trades.iloc[0].astype(object)
first[["entry", "exit"]] = first[["entry", "exit"]].astype(int)
manual_gross = first["side"] * (mid[first["exit"]] - mid[first["entry"]]) / mid[first["entry"]] * 1e4
check("gross P&L of the first trade matches the hand calculation",
      np.isclose(first["gross_bps"], manual_gross))
hs = half_spread_bps(bid_px, ask_px)
check("net-of-half-spread subtracts exactly the half-spread at entry",
      np.isclose(first["net_half_bps"], first["gross_bps"] - hs[first["entry"]]))
check("net-of-full-spread subtracts the half-spread at both ends",
      np.isclose(first["net_full_bps"], first["gross_bps"] - hs[first["entry"]] - hs[first["exit"]]))
check("fees reduce net-of-full P&L by twice the fee",
      np.isclose(backtest(imb, bid_px, ask_px, ts, 1, 0.5, fee_bps=1.0)["net_full_bps"].iloc[0],
                 first["net_full_bps"] - 2.0))

toy = pd.DataFrame({"gross_bps": [1.0, -1.0, 0.0, 2.0], "net_half_bps": 0.0, "net_full_bps": 0.0})
check("hit rate ignores trades where the mid did not move (2 of 3 = 0.667)",
      np.isclose(summarise(toy)["hit_rate"], 2 / 3) and np.isclose(summarise(toy)["flat_share"], 0.25))

s = summarise(trades)
check(f"the strategy makes money gross on planted data ({s['gross_bps']:.2f} bps/trade)", s["gross_bps"] > 0)
check(f"hit rate is above a coin flip ({s['hit_rate']:.2f})", s["hit_rate"] > 0.55)
check(f"with a 1-tick spread the planted edge survives the half-spread ({s['net_half_bps']:.2f} bps)",
      s["net_half_bps"] > 0)

# same signal, spread widened to 300 ticks so the half-spread (about 1.9 bps) exceeds
# the roughly 0.8 bps of edge available one second after an extreme imbalance
wide = synthetic_book(SLOPE, spread_ticks=300)
wb_px, wb_sz, wa_px, wa_sz, wts, _ = wide
wide_trades = backtest(imbalance(wb_sz, wa_sz, 5), wb_px, wa_px, wts, 1, 0.5)
ws = summarise(wide_trades)
wide_half = half_spread_bps(wb_px, wa_px).mean()
check(f"widening the spread past the edge ({wide_half:.2f} bps half-spread vs "
      f"{ws['gross_bps']:.2f} bps gross) makes the strategy lose net", ws["net_half_bps"] < 0)
check("gross P&L is unaffected by the spread", ws["gross_bps"] > 0)

# a signal that carries no information should not survive either
noise_trades = backtest(shuffled, bid_px, ask_px, ts, 1, 0.5)
check(f"a shuffled signal earns nothing gross ({summarise(noise_trades)['gross_bps']:.2f} bps/trade)",
      abs(summarise(noise_trades)["gross_bps"]) < 0.5)

# peek: use the imbalance at t+1 (the fill snapshot) instead of t, and gross should jump
# because in the synthetic book the imbalance at the fill drives the very next move
peek = backtest(np.roll(imb, -1), bid_px, ask_px, ts, 1, 0.5)
check("a signal read one snapshot late (a leak) scores higher than the honest one",
      summarise(peek)["gross_bps"] > s["gross_bps"] * 1.3)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
