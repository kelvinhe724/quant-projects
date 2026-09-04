"""Offline tests on simulated markets. No network, no cached data required."""

import numpy as np
import pandas as pd

from calibration import EDGES, bias_curve, bin_calibration, brier, wilson
from kelly import build_bets, kelly_fraction, simulate, stats


def simulate_markets(n, rng, distortion=None, days=400):
    """Make n binary markets. `distortion` maps market price to true probability."""
    price = rng.uniform(0.01, 0.99, n)
    true_p = price if distortion is None else distortion(price)
    outcome = (rng.uniform(size=n) < true_p).astype(int)
    close = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, days, n), unit="D"
    )
    return pd.DataFrame({
        "price": price,
        "outcome": outcome,
        "true_p": true_p,
        "close_time": close,
    }).sort_values("close_time").reset_index(drop=True)


def longshot_distortion(price):
    """Longshots overpriced, favourites underpriced: a Prelec-style shift."""
    gamma = 1.3
    p = np.clip(price, 1e-6, 1 - 1e-6)
    return p ** gamma / (p ** gamma + (1 - p) ** gamma)


def test_wilson():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61, (lo, hi)
    narrow = wilson(5000, 10000)
    assert (narrow[1] - narrow[0]) < (hi - lo) / 5
    assert wilson(0, 0) == (np.nan, np.nan) or np.isnan(wilson(0, 0)[0])


def test_recovers_planted_bias():
    rng = np.random.default_rng(1)
    df = simulate_markets(400_000, rng, distortion=longshot_distortion)
    table = bin_calibration(df["price"], df["outcome"])

    expected = [
        longshot_distortion(np.array([m])).item() for m in table["mean_price"]
    ]
    err = np.abs(table["freq"].values - np.array(expected))
    assert err.max() < 0.02, table

    low = table[table["bin_hi"] <= 0.2]
    high = table[table["bin_lo"] >= 0.8]
    assert (low["bias"] > 0).all(), low
    assert (high["bias"] < 0).all(), high
    assert (low["p_value"] < 0.01).all()
    assert (high["p_value"] < 0.01).all()


def test_null_is_null():
    """A perfectly calibrated market must not look biased.

    The strong version: across many independent replications the rate of
    significant bins has to sit near the 5% nominal level, and the sign of the
    bias must not lean one way.
    """
    flagged = 0
    total = 0
    signed = []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        df = simulate_markets(20_000, rng, distortion=None)
        table = bin_calibration(df["price"], df["outcome"])
        flagged += int((table["p_value"] < 0.05).sum())
        total += len(table)
        signed.append(np.average(table["bias"], weights=table["n"]))

    rate = flagged / total
    assert rate < 0.09, f"false positive rate {rate:.3f} on calibrated markets"

    signed = np.array(signed)
    t = signed.mean() / (signed.std(ddof=1) / np.sqrt(len(signed)))
    assert abs(t) < 3, f"calibrated markets show a directional bias, t={t:.2f}"


def test_null_strategy_makes_no_money():
    """Fitting the bias on half a calibrated sample must not produce an edge."""
    rng = np.random.default_rng(7)
    df = simulate_markets(200_000, rng, distortion=None)
    half = len(df) // 2
    fit, test = df.iloc[:half], df.iloc[half:].reset_index(drop=True)

    f = bias_curve(bin_calibration(fit["price"], fit["outcome"]))
    bets = build_bets(test, f(test["price"].values), spread=0.0)
    r, e = simulate(bets, k=0.25)
    s = stats(r, e)
    assert abs(s["sharpe"]) < 2.0, s
    assert s["total_return"] < 2.0, s


def test_kelly_matches_analytic():
    # Coin lands heads 60% of the time, contract costs 0.50, pays 1.
    # b = 1, f* = (0.6*1 - 0.4)/1 = 0.2.
    assert abs(kelly_fraction(0.6, 0.5) - 0.2) < 1e-12

    # Cost 0.25 gives b = 3. f* = (0.4*3 - 0.6)/3 = 0.2.
    assert abs(kelly_fraction(0.4, 0.25) - 0.2) < 1e-12

    # No edge, no bet.
    assert abs(kelly_fraction(0.3, 0.3)) < 1e-12
    assert kelly_fraction(0.2, 0.3) < 0

    # Kelly is the argmax of expected log growth.
    p, cost = 0.55, 0.4
    f_star = kelly_fraction(p, cost)
    b = (1 - cost) / cost
    grid = np.linspace(0.001, 0.9, 20000)
    growth = p * np.log(1 + grid * b) + (1 - p) * np.log(1 - grid)
    assert abs(grid[growth.argmax()] - f_star) < 1e-3, (grid[growth.argmax()], f_star)


def test_spread_reduces_returns():
    rng = np.random.default_rng(11)
    df = simulate_markets(200_000, rng, distortion=longshot_distortion)
    half = len(df) // 2
    fit, test = df.iloc[:half], df.iloc[half:].reset_index(drop=True)
    f = bias_curve(bin_calibration(fit["price"], fit["outcome"]))
    p_model = f(test["price"].values)

    results = {}
    for spread in (0.0, 0.02, 0.05):
        bets = build_bets(test, p_model, spread=spread)
        r, e = simulate(bets, k=0.25)
        results[spread] = stats(r, e)

    assert results[0.0]["total_return"] > 0, results[0.0]
    assert results[0.0]["total_return"] > results[0.02]["total_return"]
    assert results[0.02]["total_return"] > results[0.05]["total_return"]
    assert results[0.0]["sharpe"] > results[0.05]["sharpe"]


def test_kelly_fraction_scales_exposure():
    """k must actually change the bet size, including on crowded days."""
    bets = pd.DataFrame({
        "date": ["2024-01-01"] * 50,
        "f": np.full(50, 0.30),
        "payoff": np.r_[np.full(30, 1.0), np.full(20, -1.0)],
    })
    r_quarter, _ = simulate(bets, k=0.25)
    r_full, _ = simulate(bets, k=1.0)
    assert abs(r_full.iloc[0] - 4 * r_quarter.iloc[0]) < 1e-12
    assert abs(r_full.iloc[0] - 0.2) < 1e-12  # capped at 100% of bankroll

    # Two bets, total Kelly stake 0.6 < 1, so the cap does not bind and each
    # stake is exactly k * f.
    thin = pd.DataFrame({
        "date": ["2024-01-01"] * 2,
        "f": [0.30, 0.30],
        "payoff": [2.0, -1.0],
    })
    r, _ = simulate(thin, k=0.5)
    assert abs(r.iloc[0] - 0.5 * (0.30 * 2.0 + 0.30 * -1.0)) < 1e-12


def test_brier_decomposition():
    rng = np.random.default_rng(3)
    df = simulate_markets(200_000, rng, distortion=longshot_distortion)
    d = brier(df["price"], df["outcome"])
    assert abs(d["decomposition_residual"]) < 1e-3, d
    assert d["brier"] < d["baseline_brier"]
    assert d["reliability"] > 0

    perfect = simulate_markets(200_000, np.random.default_rng(4))
    dp = brier(perfect["price"], perfect["outcome"])
    assert dp["reliability"] < d["reliability"]


def test_bins_cover_unit_interval():
    assert EDGES[0] == 0.0 and EDGES[-1] == 1.0
    assert (np.diff(EDGES) > 0).all()
    rng = np.random.default_rng(5)
    df = simulate_markets(50_000, rng)
    table = bin_calibration(df["price"], df["outcome"])
    assert table["n"].sum() == len(df)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
