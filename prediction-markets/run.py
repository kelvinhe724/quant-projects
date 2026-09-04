"""Run the full study: calibration, bias test, Kelly simulation, charts."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from calibration import bias_curve, bin_calibration, brier
from kelly import build_bets, per_bet_edge, sharpe_ci, simulate, sweep

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
ASSUMED_SPREAD = 0.02
TIGHT_SPREAD = 0.05
MIN_BIN_TO_PLOT = 20
HORIZON = pd.Timedelta(hours=data.HORIZON_HOURS)
SPREAD_TIERS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01))
PRICE_TIERS = (0.0, 0.05, 0.20, 0.80, 0.95, 1.0)


def fmt(table):
    t = table.copy()
    t["bin"] = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(t["bin_lo"], t["bin_hi"])]
    t["freq_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t["freq_lo"], t["freq_hi"])]
    cols = ["bin", "n", "mean_price", "freq", "freq_ci", "bias", "z", "p_value",
            "outside_ci"]
    return t[cols].to_string(
        index=False,
        formatters={
            "mean_price": "{:.4f}".format,
            "freq": "{:.4f}".format,
            "bias": "{:+.4f}".format,
            "z": "{:+.1f}".format,
            "p_value": "{:.2e}".format,
        },
    )


def calibration_report(price, outcome, label):
    print(f"\n{label}")
    print(f"  markets          {len(price):,}")
    print(f"  yes resolution   {np.mean(outcome):.4f}")
    print(f"  mean price       {np.mean(price):.4f}")
    table = bin_calibration(price, outcome)
    d = brier(price, outcome)
    print(f"  Brier {d['brier']:.4f}  (base-rate baseline {d['baseline_brier']:.4f})")
    print(f"  reliability {d['reliability']:.5f}  resolution {d['resolution']:.5f}  "
          f"uncertainty {d['uncertainty']:.5f}  residual "
          f"{d['decomposition_residual']:.1e}")
    print()
    print(fmt(table))
    return table, d


def stale_price_diagnostic(reports):
    """Show why the last trade price cannot be used as the market's price.

    The /markets listing gives last_price_dollars, the last trade at an unknown
    time. On thin markets that trade is stale and sits far above where the
    contract died, which fabricates a large overpricing bias in every bin.
    """
    df = data.load()
    print("\nDIAGNOSTIC: LAST TRADE PRICE FROM THE MARKET LISTING")
    print(f"  {len(df):,} settled binary markets with at least one trade")
    mid = df[(df["price"] > 0.4) & (df["price"] < 0.5)]
    print(f"  markets whose last trade was 0.40-0.50: {len(mid):,}, "
          f"of which {mid['outcome'].mean():.1%} resolved YES")
    print(f"  median volume of those: {mid['volume'].median():,.0f} contracts "
          f"vs {df['volume'].median():,.0f} for the full sample")
    table, _ = calibration_report(
        df["price"], df["outcome"], "  calibration on the stale last trade price"
    )
    table.to_csv(os.path.join(reports, "calibration_last_trade.csv"), index=False)
    return table


def pooled_bias(snap, label, mask):
    """Mean of (mid minus outcome) over the masked markets, with a t against zero."""
    d = (snap["price"] - snap["outcome"])[mask]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  {label:<24} n={int(mask.sum()):>6,}  mean bias {d.mean():+.4f}  t={t:+.2f}")


def bias_report(snap):
    print("\nPOOLED BIAS, mid minus realised")
    pooled_bias(snap, "below 0.50", snap["price"] < 0.5)
    pooled_bias(snap, "0.50 and above", snap["price"] >= 0.5)
    pooled_bias(snap, "below 0.05", snap["price"] < 0.05)

    # A mid computed from a 0.05 / 0.95 book is not a price. Split the longshot
    # half by how wide the book was when the mid was read.
    print("\nBIAS BELOW 0.50 BY QUOTED SPREAD")
    for lo, hi in SPREAD_TIERS:
        m = (snap["spread"] >= lo) & (snap["spread"] < hi) & (snap["price"] < 0.5)
        pooled_bias(snap, f"spread [{lo:.2f}, {hi:.2f})", m)

    print("\nQUOTED SPREAD BY PRICE LEVEL")
    print(f"  {'price':<12} {'n':>6}  {'median spread':>13}  {'median spread/price':>19}")
    for lo, hi in zip(PRICE_TIERS[:-1], PRICE_TIERS[1:]):
        s = snap[(snap["price"] >= lo) & (snap["price"] < hi)]
        print(f"  {lo:.2f}-{hi:.2f}    {len(s):>6,}  {s['spread'].median():>13.3f}  "
              f"{(s['spread'] / s['price']).median():>19.2f}")


def split(df):
    """Split by close time. The bias curve is estimated on the first half only.

    A test market is traded 24 hours before it closes, so a market closing
    within 24 hours of the fit window's end would be traded on a curve built
    from fit markets that had not yet closed. Those markets are dropped.
    """
    cut = len(df) // 2
    fit = df.iloc[:cut].copy()
    test = df.iloc[cut:]
    test = test[test["close_time"] >= fit["close_time"].max() + HORIZON]
    return fit, test.copy().reset_index(drop=True)


def strategy_report(fit, test):
    print("\nFADE-THE-BIAS STRATEGY")
    print(f"  fit    {len(fit):,} markets, "
          f"{fit['close_time'].min().date()} to {fit['close_time'].max().date()}")
    print(f"  test   {len(test):,} markets, "
          f"{test['close_time'].min().date()} to {test['close_time'].max().date()}")

    curve = bias_curve(bin_calibration(fit["price"], fit["outcome"]))
    p_model = curve(test["price"].values)

    scenarios = {
        "mid to mid, no cost": np.zeros(len(test)),
        f"assumed {ASSUMED_SPREAD:.2f} spread": np.full(len(test), ASSUMED_SPREAD),
        "quoted spread": test["spread"].values,
    }
    out = {}
    for label, spread in scenarios.items():
        bets = build_bets(test, p_model, spread=spread)
        table = sweep(bets)
        table.insert(0, "scenario", label)
        out[label] = (bets, table)
        edge = per_bet_edge(bets)
        r, _ = simulate(bets, k=0.25)
        lo, hi = sharpe_ci(r)
        print(f"\n  {label}: {len(bets):,} of {len(test):,} markets pass the "
              f"Kelly filter ({len(bets) / len(test):.1%})")
        print(f"    mean payoff per dollar staked {edge['mean_payoff']:+.4f} "
              f"(t = {edge['t']:+.2f})")
        print(f"    bootstrap 95% CI on the k=0.25 Sharpe: "
              f"[{lo:+.2f}, {hi:+.2f}]")
        print(table.to_string(
            index=False,
            formatters={
                "k": "{:.2f}".format,
                "sharpe": "{:+.2f}".format,
                "log_growth": "{:+.3f}".format,
                "total_return": "{:+.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "mean_daily": "{:+.5f}".format,
            },
        ))

    # Same curve, but only the test markets whose book was tight when the mid
    # was read. If the gross edge lives in wide books, it vanishes here.
    tight = (test["spread"] <= TIGHT_SPREAD).values
    bets = build_bets(test[tight].reset_index(drop=True), p_model[tight], spread=0.0)
    edge = per_bet_edge(bets)
    print(f"\n  tight books only (spread <= {TIGHT_SPREAD:.2f}), mid to mid, no cost: "
          f"{len(bets):,} of {int(tight.sum()):,} markets, mean payoff "
          f"{edge['mean_payoff']:+.4f} (t = {edge['t']:+.2f})")
    return out


def chart_calibration(tables, path):
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for (label, t), colour in zip(tables.items(), ["#1f77b4", "#d62728"]):
        t = t[t["n"] >= MIN_BIN_TO_PLOT]
        err = np.clip(
            np.vstack([t["freq"] - t["freq_lo"], t["freq_hi"] - t["freq"]]), 0, None
        )
        ax.errorbar(t["mean_price"], t["freq"], yerr=err, fmt="o-", ms=4, lw=1.2,
                    capsize=2.5, color=colour, label=label)
    ax.set_xlabel("implied probability")
    ax.set_ylabel("realised frequency of YES")
    ax.set_title("Kalshi settled binary markets: calibration")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_bias(table, path):
    table = table[table["n"] >= MIN_BIN_TO_PLOT]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="k", lw=1)
    ax.errorbar(table["mean_price"], table["bias"], yerr=1.96 * table["bias_se"],
                fmt="o", ms=5, capsize=3, color="#1f77b4")
    for _, r in table.iterrows():
        ax.annotate(f"{int(r['n'])}", (r["mean_price"], r["bias"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6, color="#555555")
    ax.set_xlabel("quoted mid 24h before close")
    ax.set_ylabel("mid minus realised frequency")
    ax.set_title("Per-bin bias with 95% intervals (positive = longshot overpriced)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_equity(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.8))
    for ax, (label, (bets, _)) in zip(np.atleast_1d(axes), results.items()):
        for k in (0.25, 0.5, 1.0):
            r, e = simulate(bets, k=k)
            if len(e):
                ax.plot(e.index, e.values, lw=1.3, label=f"k = {k}")
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("bankroll")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("Out-of-sample equity, fractional Kelly")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_sensitivity(results, path):
    ks = np.arange(0.05, 1.01, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, (bets, _) in results.items():
        s = sweep(bets, ks=ks)
        axes[0].plot(s["k"], s["log_growth"], "o-", ms=3, label=label)
        axes[1].plot(s["k"], s["max_drawdown"], "o-", ms=3, label=label)
    axes[0].set_ylabel("annualised log growth")
    axes[1].set_ylabel("max drawdown")
    for ax in axes:
        ax.set_xlabel("Kelly fraction k")
        ax.axhline(0, color="k", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Kelly fraction sensitivity, out of sample")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_spread(snap, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    edges = np.linspace(0, 1, 21)
    idx = np.digitize(snap["price"], edges[1:-1])
    mids, med, p25, p75 = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 20:
            continue
        mids.append((edges[b] + edges[b + 1]) / 2)
        med.append(snap.loc[m, "spread"].median())
        p25.append(snap.loc[m, "spread"].quantile(0.25))
        p75.append(snap.loc[m, "spread"].quantile(0.75))
    ax.fill_between(mids, p25, p75, alpha=0.25, color="#1f77b4", label="25-75 pct")
    ax.plot(mids, med, "o-", color="#1f77b4", ms=4, label="median quoted spread")
    ax.axhline(ASSUMED_SPREAD, color="#d62728", ls="--", lw=1,
               label=f"{ASSUMED_SPREAD:.2f} assumption")
    ax.set_xlabel("quoted mid")
    ax.set_ylabel("bid-ask spread (dollars)")
    ax.set_title("Quoted spread by price level, 24h before close")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    stale_table = stale_price_diagnostic(REPORTS)

    snap = data.load_snapshot()
    print(f"\n{'=' * 78}\nQUOTED MID 24H BEFORE CLOSE\n{'=' * 78}")
    print(f"  median quoted spread {snap['spread'].median():.3f}, "
          f"mean {snap['spread'].mean():.3f}")
    print(f"  markets with spread > {ASSUMED_SPREAD:.2f}: "
          f"{(snap['spread'] > ASSUMED_SPREAD).mean():.1%}")
    print(snap.groupby("category").agg(
        n=("outcome", "size"), yes=("outcome", "mean"),
        med_spread=("spread", "median")).sort_values("n", ascending=False).to_string())

    table, _ = calibration_report(
        snap["price"], snap["outcome"], "CALIBRATION ON THE QUOTED MID"
    )
    bias_report(snap)

    chart_calibration(
        {
            f"quoted mid 24h before close (n={len(snap):,})": table,
            f"last trade price (n={stale_table['n'].sum():,})": stale_table,
        },
        os.path.join(REPORTS, "calibration.png"),
    )
    chart_bias(table, os.path.join(REPORTS, "bias.png"))
    chart_spread(snap, os.path.join(REPORTS, "spread.png"))

    fit, test = split(snap)
    results = strategy_report(fit, test)
    chart_equity(results, os.path.join(REPORTS, "equity.png"))
    chart_sensitivity(results, os.path.join(REPORTS, "kelly_sensitivity.png"))

    table.to_csv(os.path.join(REPORTS, "calibration_quoted_mid.csv"), index=False)
    pd.concat([t for _, t in results.values()]).to_csv(
        os.path.join(REPORTS, "kelly_sweep.csv"), index=False
    )
    print(f"\ncharts and tables written to {REPORTS}")


if __name__ == "__main__":
    main()
