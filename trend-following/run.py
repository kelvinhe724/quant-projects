"""Full pipeline on real prices: lookbacks, headline strategy, crises, decay, costs, ETF cross-check.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from backtest import COST_BPS, MONTHS, beta_alpha, by_year, metrics, run, summary
from trend import LOOKBACK, SKIP, TARGET_VOL, build

REPORTS = data.REPORTS
IS = (data.START, data.IS_END)
TEST = (data.TEST_START, data.DATA_END)

# Written down before any result was looked at: 12-1 sign, 40% per-asset vol
# target, class-balanced book, monthly rebalance, 5bps one-way.
HEADLINE = dict(lookback=LOOKBACK, skip=SKIP)
LOOKBACKS = [(1, 0), (3, 0), (6, 0), (12, 0), (12, 1)]

CRISES = [
    ("2008 full year", "2008-01-31", "2008-12-31"),
    ("Lehman, Sep 2008 to Feb 2009", "2008-09-30", "2009-02-27"),
    ("2020 Feb to Mar (the crash)", "2020-02-28", "2020-03-31"),
    ("2020 Apr to Dec (the reversal)", "2020-04-30", "2020-12-31"),
    ("2020 full year", "2020-01-31", "2020-12-31"),
    ("2022 full year", "2022-01-31", "2022-12-30"),
]
SUBPERIODS = [("2005-2009", "2005", "2009"), ("2010-2014", "2010", "2014"),
              ("2015-2019", "2015", "2019"), ("2020-2026", "2020", "2026"),
              ("post-2010", "2010", "2026")]


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def label(lookback, skip):
    return f"{lookback}-{skip}"


def fmt(m):
    """Format one metrics dict into a reporting column."""
    return pd.Series({
        "annual return (gross)": f"{m['gross_annual_return']:+.2%}",
        "annual return (net)": f"{m['annual_return']:+.2%}",
        "annual vol": f"{m['annual_vol']:.2%}",
        "Sharpe (gross)": f"{m['gross_sharpe']:.2f}",
        "Sharpe (net)": f"{m['sharpe']:.2f}",
        "max drawdown (net)": f"{m['max_drawdown']:.2%}",
        "drawdown trough": f"{m['dd_trough']:%Y-%m}",
        "hit rate (monthly)": f"{m['hit_rate']:.1%}",
        "annual turnover": f"{m['annual_turnover']:.1f}x",
        "avg gross leverage": f"{m['avg_leverage']:.2f}x",
        "months": m["n_months"],
    })


def period_return(r, start, end):
    return (1 + r.loc[start:end]).prod() - 1


def main():
    px, rets, classes, _ = data.get_panel()
    monthly = data.monthly_returns(px)
    spy = data.monthly_returns(data.clean(data.download()))[data.BENCHMARK]
    print(f"universe: {len(px.columns)} assets in {classes.nunique()} classes, "
          f"{px.index[0].date()} to {px.index[-1].date()}")
    print(f"scored from {IS[0]}  |  in-sample to {IS[1]}  |  final test from {TEST[0]}")
    print(f"signal {label(**HEADLINE)}, {TARGET_VOL:.0%} per-asset vol target, "
          f"class-balanced, {COST_BPS:.0f}bps one-way on traded notional")

    books = {}

    def book_for(lookback, skip, scheme="class", universe=None):
        key = (lookback, skip, scheme, universe is not None)
        if key not in books:
            if universe is None:
                w = build(px, rets, classes, lookback, skip, scheme=scheme)
                books[key] = run(w, monthly).loc[IS[0]:]
            else:
                p, r, c, _ = data.get_panel(universe)
                books[key] = run(build(p, r, c, lookback, skip, scheme=scheme),
                                 data.monthly_returns(p)).loc[IS[0]:]
        return books[key]

    headline = book_for(**HEADLINE)
    print(f"asset-months held with a missing return: {headline.attrs['held_returns_missing']}")

    lookbacks(book_for)

    show(f"Headline {label(**HEADLINE)} strategy, full sample {IS[0]} to {px.index[-1].date()}",
         pd.DataFrame({
             "class-balanced": fmt(summary(headline)),
             "equal-weight assets": fmt(summary(book_for(LOOKBACK, SKIP, "equal"))),
             "12-0 (no skip)": fmt(summary(book_for(12, 0))),
         }))
    perf = pd.DataFrame({"full": summary(headline), "in-sample": summary(headline.loc[IS[0]:IS[1]]),
                         "test": summary(headline.loc[TEST[0]:])})
    perf.to_csv(os.path.join(REPORTS, "performance.csv"))

    ba = beta_alpha(headline["net"], spy)
    spy_m = metrics(spy.loc[IS[0]:])
    print(f"\nvs SPY over the full sample: beta {ba['beta']:+.2f}, annualised alpha "
          f"{ba['alpha']:+.2%}; SPY itself {spy_m['annual_return']:+.2%}/yr, "
          f"Sharpe {spy_m['sharpe']:.2f}, max drawdown {spy_m['max_drawdown']:.1%}")

    crises(headline, spy, classes)
    decay(headline, spy)
    attribution(headline, classes)
    costs(book_for)
    etf_check(book_for, spy)

    years = by_year(headline)
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))
    show("By calendar year, net",
         years.assign(gross=years["gross"].map("{:+.1%}".format),
                      net=years["net"].map("{:+.1%}".format),
                      sharpe=years["sharpe"].round(2),
                      SPY=spy.groupby(spy.index.year).apply(lambda r: (1 + r).prod() - 1)
                      .reindex(years.index).map("{:+.1%}".format)))

    charts(headline, spy, classes, px, rets, monthly, years)
    print(f"\ncharts and tables written to {REPORTS}/")


def lookbacks(book_for):
    """Compare lookbacks in-sample, then show the same table on the untouched test window."""
    rows = {}
    for lb, skip in LOOKBACKS:
        b = book_for(lb, skip)
        for name, span in (("in-sample", IS), ("test", TEST)):
            m = summary(b.loc[span[0]:span[1]])
            rows[(label(lb, skip), name)] = {
                "Sharpe gross": round(m["gross_sharpe"], 2),
                "Sharpe net": round(m["sharpe"], 2),
                "return net": f"{m['annual_return']:+.1%}",
                "max dd": f"{m['max_drawdown']:.0%}",
                "turnover": f"{m['annual_turnover']:.0f}x",
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["lookback", "window"]
    show(f"Lookback comparison (in-sample {IS[0]} to {IS[1]}, test {TEST[0]} on)",
         table)
    table.to_csv(os.path.join(REPORTS, "lookbacks.csv"))


def crises(book, spy, classes):
    """Report the strategy against SPY through the periods a trend follower is bought for."""
    print("\n" + "=" * 78)
    print("CRISIS ALPHA")
    print("=" * 78)
    contrib = book.attrs["contrib"]
    by_class = contrib.T.groupby(classes).sum().T
    rows = {}
    for name, start, end in CRISES:
        rows[name] = {"trend net": f"{period_return(book['net'], start, end):+.1%}",
                      "SPY": f"{period_return(spy, start, end):+.1%}",
                      **{c: f"{period_return(by_class[c], start, end):+.1%}"
                         for c in by_class.columns}}
    table = pd.DataFrame(rows).T
    show("Crisis periods: strategy net vs SPY, with the gross contribution of each class", table)
    table.to_csv(os.path.join(REPORTS, "crisis.csv"))

    year = pd.concat([by_class.loc["2020"], book.loc["2020", ["gross", "net"]],
                      spy.loc["2020"].rename("SPY")], axis=1)
    year.index = year.index.strftime("%Y-%m")
    show("2020 month by month (class columns are gross contributions)",
         year.map("{:+.1%}".format))
    year.to_csv(os.path.join(REPORTS, "year_2020.csv"))

    down = spy.loc[book.index][spy.loc[book.index] < -0.05].index
    print(f"\nmonths SPY fell more than 5%: {len(down)}; strategy net in those months averaged "
          f"{book.loc[down, 'net'].mean():+.2%} vs SPY {spy.loc[down].mean():+.2%}; "
          f"strategy positive in {(book.loc[down, 'net'] > 0).mean():.0%} of them")
    for name, start, end in CRISES[:1] + CRISES[4:5]:
        ba = beta_alpha(book.loc[start:end, "net"], spy.loc[start:end])
        print(f"beta to SPY, {name}: {ba['beta']:+.2f}")


def decay(book, spy):
    """Sub-period performance, to see whether the effect faded after 2010."""
    print("\n" + "=" * 78)
    print("DECAY")
    print("=" * 78)
    rows = {}
    for name, start, end in SUBPERIODS:
        m = summary(book.loc[start:end])
        s = metrics(spy.loc[start:end])
        rows[name] = {"Sharpe gross": round(m["gross_sharpe"], 2),
                      "Sharpe net": round(m["sharpe"], 2),
                      "return net": f"{m['annual_return']:+.1%}",
                      "max dd": f"{m['max_drawdown']:.0%}",
                      "hit rate": f"{m['hit_rate']:.0%}",
                      "SPY Sharpe": round(s["sharpe"], 2),
                      "months": m["n_months"]}
    table = pd.DataFrame(rows).T
    show("Sub-periods, headline strategy", table)
    table.to_csv(os.path.join(REPORTS, "subperiods.csv"))

    pre, post = book.loc[:"2009", "net"], book.loc["2010":, "net"]
    diff = pre.mean() - post.mean()
    se = np.sqrt(pre.var() / len(pre) + post.var() / len(post))
    print(f"\nmean monthly net return 2005-2009 {pre.mean():+.2%} vs 2010 on {post.mean():+.2%}; "
          f"difference {diff:+.2%}/month, t-stat {diff / se:.2f}")


def attribution(book, classes):
    """Annualised gross contribution and stand-alone Sharpe of each asset and class."""
    contrib = book.attrs["contrib"]
    years = len(contrib) / MONTHS
    per_asset = pd.DataFrame({
        "class": classes,
        "annual contribution": contrib.sum() / years,
        "stand-alone Sharpe": contrib.mean() / contrib.std() * np.sqrt(MONTHS),
    }).sort_values("annual contribution", ascending=False)
    show("Attribution by asset (gross, full sample)",
         per_asset.assign(**{"annual contribution": per_asset["annual contribution"]
                             .map("{:+.2%}".format),
                             "stand-alone Sharpe": per_asset["stand-alone Sharpe"].round(2)}))
    by_class = contrib.T.groupby(classes).sum().T
    cls = pd.DataFrame({"annual contribution": by_class.sum() / years,
                        "stand-alone Sharpe": by_class.mean() / by_class.std() * np.sqrt(MONTHS)})
    show("Attribution by class", cls.assign(**{
        "annual contribution": cls["annual contribution"].map("{:+.2%}".format),
        "stand-alone Sharpe": cls["stand-alone Sharpe"].round(2)}))
    per_asset.to_csv(os.path.join(REPORTS, "attribution.csv"))


def costs(book_for):
    """Net Sharpe of the headline strategy under several cost assumptions."""
    b = book_for(**HEADLINE)
    rows = {}
    for bps in (0, 5, 10, 20, 40):
        net = b["gross"] - b["traded"] * bps / 1e4
        rows[f"{bps}bps"] = {
            "full-sample Sharpe": round(metrics(net)["sharpe"], 2),
            "in-sample Sharpe": round(metrics(net.loc[IS[0]:IS[1]])["sharpe"], 2),
            "test Sharpe": round(metrics(net.loc[TEST[0]:])["sharpe"], 2),
            "annual drag": f"{(b['traded'] * bps / 1e4).sum() / (len(b) / MONTHS):.2%}",
        }
    table = pd.DataFrame(rows).T
    show("Cost sensitivity (one-way bps on traded notional)", table)
    table.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))


def etf_check(book_for, spy):
    """Rerun the headline on an all-ETF universe with no roll splices."""
    print("\n" + "=" * 78)
    print("ETF CROSS-CHECK (no futures splices, total-return series)")
    print("=" * 78)
    etf = book_for(LOOKBACK, SKIP, universe=data.ETFS)
    fut = book_for(**HEADLINE)
    table = pd.DataFrame({"futures universe": fmt(summary(fut)), "ETF universe": fmt(summary(etf))})
    show(f"Full sample {IS[0]} on", table)
    corr = fut["net"].corr(etf["net"])
    print(f"\nmonthly net return correlation between the two universes: {corr:.2f}")
    for name, start, end in CRISES:
        print(f"{name}: futures {period_return(fut['net'], start, end):+.1%}, "
              f"ETF {period_return(etf['net'], start, end):+.1%}, "
              f"SPY {period_return(spy, start, end):+.1%}")
    table.to_csv(os.path.join(REPORTS, "etf_check.csv"))


def charts(book, spy, classes, px, rets, monthly, years):
    split = pd.Timestamp(TEST[0])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, series, style in (("trend, gross", book["gross"], "--"),
                                ("trend, net", book["net"], "-"),
                                ("SPY", spy.loc[book.index], ":")):
        ax.plot((1 + series.fillna(0)).cumprod(), style, linewidth=1.2, label=name)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.text(split, ax.get_ylim()[1], " final test", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title(f"Time-series momentum {label(**HEADLINE)}, class-balanced, vs SPY")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for name, r, color in (("trend, net", book["net"], "firebrick"), ("SPY", spy.loc[book.index], "grey")):
        equity = (1 + r).cumprod()
        ax.fill_between(equity.index, equity / equity.cummax() - 1, 0, color=color, alpha=0.45,
                        label=name)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.set_title("Drawdowns")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "drawdown.png"), dpi=140)
    plt.close(fig)

    # Daily view of 2020 with the month-end weights held fixed inside each month,
    # which is exactly what the monthly engine assumes.
    w = build(px, rets, classes, **HEADLINE)
    held = w.reindex(px.index).ffill().shift(1).fillna(0.0)
    daily = (held * rets.fillna(0.0)).loc["2020-01-01":"2020-12-31"]
    spy_daily = data.clean(data.download())[data.BENCHMARK].pct_change().loc["2020"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((1 + daily.sum(axis=1)).cumprod(), linewidth=1.6, color="black", label="trend, gross")
    for cls, names in classes.groupby(classes).groups.items():
        ax.plot(1 + daily[list(names)].sum(axis=1).cumsum(), linewidth=1.0, label=cls)
    ax.plot((1 + spy_daily).cumprod(), ":", linewidth=1.4, color="grey", label="SPY")
    ax.axhline(1.0, color="grey", linewidth=0.7)
    ax.axvline(pd.Timestamp("2020-03-23"), color="firebrick", linewidth=0.8)
    ax.text(pd.Timestamp("2020-03-23"), ax.get_ylim()[0], " SPY low, 23 Mar", fontsize=8, va="bottom")
    ax.set_title("2020: the crash and the reversal, rebased to 1 on 2020-01-01 "
                 "(class lines are cumulative gross contributions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "crisis_2020.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    spy_years = spy.groupby(spy.index.year).apply(lambda r: (1 + r).prod() - 1).reindex(years.index)
    x = np.arange(len(years))
    ax.bar(x - 0.2, years["net"], 0.4, color=["steelblue" if y < 2016 else "darkorange"
                                              for y in years.index], label="trend, net")
    ax.bar(x + 0.2, spy_years, 0.4, color="lightgrey", label="SPY")
    ax.set_xticks(x, [str(y) for y in years.index], rotation=45, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Net return by year (orange = untouched final test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "by_year.png"), dpi=140)
    plt.close(fig)

    table = pd.read_csv(os.path.join(REPORTS, "lookbacks.csv"), index_col=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [label(lb, s) for lb, s in LOOKBACKS]
    for offset, window in ((-0.2, "in-sample"), (0.2, "test")):
        ax.bar(np.arange(len(labels)) + offset,
               [table.loc[(l, window), "Sharpe net"] for l in labels], 0.4, label=window)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("net Sharpe")
    ax.set_title("Net Sharpe by lookback (months-skip)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "lookbacks.png"), dpi=140)
    plt.close(fig)

    rolling = book["net"].rolling(36)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rolling.mean() / rolling.std() * np.sqrt(MONTHS), linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.set_title("Rolling 36-month net Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "rolling_sharpe.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
