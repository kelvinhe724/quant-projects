"""Full pipeline on real prices: baseline, crossover comparison, walk-forward, final test.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from backtest import BORROW_BPS, COST_BPS, TRADING_DAYS, beta_alpha, by_year, run, summary
from momentum import (LOOKBACK, MA_LONG, MA_SHORT, SKIP, TAIL, cross_sectional_weights,
                      crossover_weights, information_coefficient, long_only_weights,
                      ma_signal, momentum_score, quantile_returns)

REPORTS = data.REPORTS
IS = (data.IS_START, data.IS_END)
TEST = (data.TEST_START, data.TEST_END)

# The frozen baseline, written down before any result was looked at: Appendix A of
# the spec, tightened from 20% tails to deciles because the brief asks for deciles.
BASELINE = {"lookback": LOOKBACK, "skip": SKIP, "tail": TAIL}

# A small, economically motivated grid. Three horizons that are all recognisable
# momentum specifications, two tail widths. No threshold, filter or weighting
# scheme is searched; that is what keeps the multiple-testing burden small.
GRID = [{"lookback": lb, "skip": SKIP, "tail": t}
        for lb in (126, 189, 252) for t in (0.10, 0.20)]

WF_FIRST_YEAR = 2010


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def label(cfg):
    return f"{cfg['lookback'] // 21}-{cfg['skip'] // 21} / {cfg['tail']:.0%} tails"


def fmt(m):
    """Format one metrics dict into the reporting column."""
    return pd.Series({
        "annual return (gross)": f"{m['gross_annual_return']:+.2%}",
        "annual return (net)": f"{m['annual_return']:+.2%}",
        "annual vol": f"{m['annual_vol']:.2%}",
        "Sharpe (gross)": f"{m['gross_sharpe']:.2f}",
        "Sharpe (net)": f"{m['sharpe']:.2f}",
        "Sortino (net)": f"{m['sortino']:.2f}",
        "max drawdown (net)": f"{m['max_drawdown']:.2%}",
        "drawdown trough": f"{m['dd_trough']:%Y-%m-%d}",
        "Calmar": f"{m['calmar']:.2f}",
        "hit rate (monthly)": f"{m['hit_rate_monthly']:.1%}",
        "hit rate (daily)": f"{m['hit_rate_daily']:.1%}",
        "annual turnover": f"{m['annual_turnover']:.1f}x",
        "cost drag (annual)": f"{m['cost_drag']:.2%}",
        "long leg return": f"{m['long_annual_return']:+.2%}",
        "short leg return": f"{m['short_annual_return']:+.2%}",
        "avg gross exposure": f"{m['avg_gross_exposure']:.2f}",
        "avg net exposure": f"{m['avg_net_exposure']:+.3f}",
    })


def score(book, span):
    """Summarise a full-sample book over one window."""
    return summary(book.loc[span[0]:span[1]])


def main():
    px, returns, eligible, sectors, bench = data.get_panel()
    dates = data.rebalance_dates(px.index, data.IS_START, data.TEST_END)
    print(f"universe: {len(px.columns)} names, {px.index[0].date()} to {px.index[-1].date()}")
    print(f"in-sample {IS[0]} to {IS[1]}  |  final test {TEST[0]} to {TEST[1]}")
    print(f"rebalance dates: {len(dates)}  |  costs {COST_BPS:.0f}bps one-way, "
          f"{BORROW_BPS:.0f}bps/yr borrow")

    n_elig = eligible.loc[data.IS_START:].sum(axis=1)
    print(f"eligible names per date: min {n_elig.min()}, median {int(n_elig.median())}, "
          f"max {n_elig.max()}")

    books = {}

    def book_for(cfg):
        """Backtest one cross-sectional configuration over the whole sample, once."""
        key = tuple(sorted(cfg.items()))
        if key not in books:
            w = cross_sectional_weights(
                momentum_score(px, cfg["lookback"], cfg["skip"]),
                eligible, dates, tail=cfg["tail"])
            books[key] = run(w, returns)
        return books[key]

    baseline = book_for(BASELINE)
    signal = ma_signal(px, MA_SHORT, MA_LONG)
    crossover = run(crossover_weights(signal, eligible), returns)
    crossover_monthly = run(crossover_weights(signal, eligible, dates), returns)
    long_only = run(long_only_weights(signal, eligible), returns)

    missing = baseline.attrs["held_returns_missing"]
    print(f"stock-days held with a missing return: {missing}")

    signal_diagnostics(px, eligible, dates)

    show("Frozen baseline vs the moving-average crossover, in-sample "
         f"({IS[0]} to {IS[1]}, net of costs)",
         pd.DataFrame({
             f"12-1 momentum, {TAIL:.0%} tails": fmt(score(baseline, IS)),
             f"{MA_SHORT}/{MA_LONG} crossover, daily": fmt(score(crossover, IS)),
             f"{MA_SHORT}/{MA_LONG} crossover, monthly": fmt(score(crossover_monthly, IS)),
             f"{MA_SHORT}/{MA_LONG} long only": fmt(score(long_only, IS)),
         }))

    chosen, wf_book, wf_log = walk_forward(book_for, dates)

    show("Walk-forward validation: configuration chosen for each block from prior data "
         "only", wf_log)

    print(f"\nchained walk-forward result ({wf_log.index[0]}-{wf_log.index[-1]}): "
          f"net Sharpe {summary(wf_book)['sharpe']:.2f}, "
          f"net annual return {summary(wf_book)['annual_return']:+.2%}")
    switches = (wf_log["config"] != wf_log["config"].shift()).sum() - 1
    print(f"configuration changed in {switches} of {len(wf_log) - 1} handovers; "
          f"most often selected: {label(chosen)}")

    print("\n" + "=" * 78)
    print("FINAL TEST: run once, on data untouched by every step above")
    print("=" * 78)

    final = pd.DataFrame({
        f"frozen baseline ({label(BASELINE)})": fmt(score(baseline, TEST)),
        f"walk-forward pick ({label(chosen)})": fmt(score(book_for(chosen), TEST)),
        f"{MA_SHORT}/{MA_LONG} crossover, daily": fmt(score(crossover, TEST)),
        f"{MA_SHORT}/{MA_LONG} long only": fmt(score(long_only, TEST)),
    })
    show(f"Out-of-sample performance ({TEST[0]} to {TEST[1]}, net of costs)", final)
    final.to_csv(os.path.join(REPORTS, "final_test.csv"))

    for name, book in (("12-1 momentum", baseline), ("crossover", crossover)):
        ba = beta_alpha(book.loc[TEST[0]:TEST[1]], bench)
        print(f"{name} vs SPY out-of-sample: beta {ba['beta']:+.2f}, "
              f"annualised alpha {ba['alpha']:+.2%}")

    years = by_year(baseline.loc[IS[0]:TEST[1]])
    years["in/out"] = np.where(years.index >= 2021, "test", "in-sample")
    show("12-1 momentum by calendar year (net of costs)",
         years.assign(**{"gross": years["gross"].map("{:+.2%}".format),
                         "net": years["net"].map("{:+.2%}".format),
                         "sharpe": years["sharpe"].round(2),
                         "max_dd": years["max_dd"].map("{:.1%}".format),
                         "turnover": years["turnover"].round(1)}))
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))

    crash_2020(baseline, crossover, bench)
    cost_sensitivity(px, eligible, dates, returns, signal)
    survivorship(px, returns, eligible, bench)
    charts(baseline, crossover, long_only, wf_book, bench, years, px, eligible, dates)
    print(f"\ncharts and tables written to {REPORTS}/")


def signal_diagnostics(px, eligible, dates):
    """Report the information coefficient and quantile returns of the raw signal."""
    score_panel = momentum_score(px).loc[dates]
    forward = (px.shift(-SKIP) / px - 1).loc[dates]
    is_dates = dates[dates <= IS[1]]

    ic = information_coefficient(score_panel.loc[is_dates], forward.loc[is_dates],
                                 eligible.loc[is_dates])
    t_stat = ic.mean() / ic.std() * np.sqrt(len(ic))
    print(f"\nin-sample information coefficient: mean {ic.mean():+.4f}, "
          f"t-stat {t_stat:+.2f} over {len(ic)} months, "
          f"{(ic > 0).mean():.0%} of months positive")

    buckets = quantile_returns(score_panel.loc[is_dates], forward.loc[is_dates],
                               eligible.loc[is_dates], n_buckets=5)
    show("In-sample average next-month return by momentum quintile "
         "(1 = worst momentum, diagnostic only: this uses forward returns)",
         buckets.map("{:+.2%}".format).rename("mean next-month return").to_frame())


def walk_forward(book_for, dates):
    """Select a configuration for each year using only the years before it."""
    rows = {}
    for year in range(WF_FIRST_YEAR, int(IS[1][:4]) + 1):
        train_end = f"{year - 1}-12-31"
        scored = {}
        for cfg in GRID:
            m = summary(book_for(cfg).loc[IS[0]:train_end])
            scored[label(cfg)] = (m["sharpe"], cfg)
        best_label = max(scored, key=lambda k: scored[k][0])
        best_sharpe, best_cfg = scored[best_label]
        block = book_for(best_cfg).loc[f"{year}-01-01":f"{year}-12-31"]
        rows[year] = {
            "config": best_label,
            "cfg": best_cfg,
            "train Sharpe": round(best_sharpe, 2),
            "block net return": f"{(1 + block['net']).prod() - 1:+.2%}",
            "block Sharpe": round(block["net"].mean() / block["net"].std()
                                  * np.sqrt(TRADING_DAYS), 2),
        }
    log = pd.DataFrame(rows).T
    book = pd.concat([book_for(r["cfg"]).loc[f"{y}-01-01":f"{y}-12-31"]
                      for y, r in log.iterrows()])
    modal = log["config"].mode().iloc[0]
    chosen = next(c for c in GRID if label(c) == modal)
    return chosen, book, log.drop(columns="cfg")


def crash_2020(baseline, crossover, bench):
    """Report what momentum did through the 2020 reversal."""
    print("\n" + "=" * 78)
    print("2020: THE MOMENTUM CRASH")
    print("=" * 78)

    year = baseline.loc["2020-01-01":"2020-12-31"]
    monthly = pd.DataFrame({
        "momentum net": year["net"].resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "long leg": year["long"].resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "short leg": year["short"].resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "crossover net": crossover.loc["2020", "net"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1),
        "SPY": bench.loc["2020"].resample("ME").apply(lambda x: (1 + x).prod() - 1),
    })
    monthly.index = monthly.index.strftime("%Y-%m")
    show("2020 monthly returns", monthly.map("{:+.2%}".format))
    monthly.to_csv(os.path.join(REPORTS, "crash_2020.csv"))

    worst = baseline.loc["2020", "net"].nsmallest(5)
    show("Five worst days for 12-1 momentum in 2020",
         pd.DataFrame({"momentum net": worst.map("{:+.2%}".format),
                       "long leg": baseline.loc[worst.index, "long"].map("{:+.2%}".format),
                       "short leg": baseline.loc[worst.index, "short"].map("{:+.2%}".format),
                       "SPY": bench.loc[worst.index].map("{:+.2%}".format)}))

    dd = (1 + baseline.loc["2020-01-01":"2021-06-30", "net"]).cumprod()
    underwater = dd / dd.cummax() - 1
    print(f"\npeak-to-trough over 2020 into mid-2021: {underwater.min():.2%} "
          f"(trough {underwater.idxmin():%Y-%m-%d})")
    print(f"full-year 2020: gross {(1 + year['gross']).prod() - 1:+.2%}, "
          f"net {(1 + year['net']).prod() - 1:+.2%}, "
          f"SPY {(1 + bench.loc['2020']).prod() - 1:+.2%}")


def cost_sensitivity(px, eligible, dates, returns, signal):
    """Re-score both strategies across the required cost scenarios."""
    w = cross_sectional_weights(momentum_score(px), eligible, dates, tail=TAIL)
    ma_w = crossover_weights(signal, eligible)
    scenarios = [("frictionless", 0.0, 0.0), ("low", 5.0, 25.0),
                 ("base", COST_BPS, BORROW_BPS), ("stress", 20.0, 100.0),
                 ("severe", 40.0, 200.0)]
    rows = {}
    for name, cost, borrow in scenarios:
        mom, ma = run(w, returns, cost, borrow), run(ma_w, returns, cost, borrow)
        rows[f"{name} ({cost:.0f}bps / {borrow:.0f}bps)"] = {
            "momentum in-sample": round(summary(mom.loc[IS[0]:IS[1]])["sharpe"], 2),
            "momentum out-of-sample": round(summary(mom.loc[TEST[0]:TEST[1]])["sharpe"], 2),
            "crossover in-sample": round(summary(ma.loc[IS[0]:IS[1]])["sharpe"], 2),
            "crossover out-of-sample": round(summary(ma.loc[TEST[0]:TEST[1]])["sharpe"], 2),
        }
    table = pd.DataFrame(rows).T
    show("Net Sharpe by cost scenario (one-way trading bps / annual borrow bps)", table)
    table.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))


def survivorship(px, returns, eligible, bench):
    """Quantify the direction and rough size of the survivorship bias."""
    print("\n" + "=" * 78)
    print("SURVIVORSHIP BIAS")
    print("=" * 78)

    span = slice(data.IS_START, data.TEST_END)
    # membership decided at the close of t-1 earns the return of t, same clock as the backtest
    tradable = eligible.shift(1).fillna(False).astype(bool)
    equal_weight = returns.loc[span].where(tradable.loc[span]).mean(axis=1)
    years = len(equal_weight) / TRADING_DAYS
    ew_annual = (1 + equal_weight.fillna(0)).prod() ** (1 / years) - 1
    spy_annual = (1 + bench.loc[span]).prod() ** (1 / len(bench.loc[span]) * TRADING_DAYS) - 1

    late = (px.notna().idxmax() > pd.Timestamp(data.IS_START)).sum()
    print(f"universe is today's S&P 500 membership, applied back to {data.IS_START}. "
          f"{len(px.columns)} names survived the download; four more "
          f"(BK, HES, IPG, MMC) no longer resolve on Yahoo at all.")
    print(f"names with no price on {data.IS_START}: {late} of {len(px.columns)} "
          f"(they enter as they list, which the eligibility flag handles correctly)")
    print(f"\nequal-weight universe: {ew_annual:+.2%}/yr   SPY: {spy_annual:+.2%}/yr   "
          f"gap: {ew_annual - spy_annual:+.2%}/yr")
    print("""
That gap is the level of the bias: a basket picked for being in the index today
beats the index the index actually was. It hits the long-short book differently
from a long-only one. Every name that fell far enough to be deleted is missing,
and those deletions are exactly the names 12-1 momentum would have been short.
The short leg is therefore the biased leg, and it is biased against the strategy:
the worst losers were removed from the sample before they could be shorted. The
long leg is biased the other way, since a stock in the index today is one whose
past winning streak did not later reverse into deletion. Net direction for a
long-short book is genuinely ambiguous, which is why the honest claim is only
that the sign of the bias is unknown and the magnitude is on the order of the
2-4%/yr gap above, not that the results are conservative.""")


def charts(baseline, crossover, long_only, wf_book, bench, years, px, eligible, dates):
    span = slice(data.IS_START, data.TEST_END)
    split = pd.Timestamp(data.TEST_START)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, series, style in (
            ("12-1 momentum, gross", baseline.loc[span, "gross"], "--"),
            ("12-1 momentum, net", baseline.loc[span, "net"], "-"),
            ("50/200 crossover, net", crossover.loc[span, "net"], "-"),
            ("50/200 long only, net", long_only.loc[span, "net"], "-"),
            ("SPY", bench.loc[span], ":")):
        ax.plot((1 + series.fillna(0)).cumprod(), style, linewidth=1.1, label=name)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.text(split, ax.get_ylim()[1], " final test", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title("Equity curves, gross and net of costs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    equity = (1 + baseline.loc[span, "net"]).cumprod()
    underwater = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(underwater.index, underwater, 0, color="firebrick", alpha=0.55)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.set_title("12-1 momentum drawdown, net of costs")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "drawdown.png"), dpi=140)
    plt.close(fig)

    crash = slice("2020-01-01", "2021-06-30")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, series in (("12-1 momentum, net", baseline.loc[crash, "net"]),
                         ("long leg", baseline.loc[crash, "long"]),
                         ("short leg", baseline.loc[crash, "short"]),
                         ("SPY", bench.loc[crash])):
        ax.plot((1 + series.fillna(0)).cumprod(), linewidth=1.2, label=name)
    ax.axhline(1.0, color="grey", linewidth=0.7)
    ax.set_title("The 2020 reversal, rebased to 1 on 2020-01-01")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "crash_2020.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["steelblue" if y < 2021 else "darkorange" for y in years.index]
    ax.bar(years.index, years["net"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("12-1 momentum net return by year (orange = untouched final test)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "by_year.png"), dpi=140)
    plt.close(fig)

    forward = (px.shift(-SKIP) / px - 1).loc[dates]
    score_panel = momentum_score(px).loc[dates]
    is_dates = dates[dates <= IS[1]]
    test_dates = dates[dates >= TEST[0]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    for offset, (name, sub) in zip((-width / 2, width / 2),
                                   (("in-sample", is_dates), ("final test", test_dates))):
        b = quantile_returns(score_panel.loc[sub], forward.loc[sub],
                             eligible.loc[sub], n_buckets=5)
        ax.bar(np.arange(1, 6) + offset, b, width, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("momentum quintile (1 = worst)")
    ax.set_ylabel("mean next-month return")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.set_title("Does the signal order future returns?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "quantiles.png"), dpi=140)
    plt.close(fig)

    rolling = baseline.loc[span, "net"].rolling(TRADING_DAYS)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rolling.mean() / rolling.std() * np.sqrt(TRADING_DAYS), linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.set_title("Rolling 12-month net Sharpe, 12-1 momentum")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "rolling_sharpe.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
