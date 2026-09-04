"""Full pipeline on real data: UIP test, G10 carry backtest, crisis autopsies, charts.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from backtest import COST_BPS, TRADING_DAYS, by_year, max_drawdown, monthly, run, summary
from carry import N_LEG, carry_weights, differential, excess_returns, uip_regression

REPORTS = data.REPORTS
IS = (data.IS_START, data.IS_END)
TEST = (data.TEST_START, None)

CRISES = {
    "2008 global financial crisis": ("2008-07-01", "2009-03-31"),
    "January 2015, SNB drops the CHF floor": ("2015-01-01", "2015-01-31"),
    "August 2015, China devaluation": ("2015-08-01", "2015-09-30"),
    "March 2020, covid": ("2020-02-15", "2020-04-30"),
}


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def fmt(m):
    """Format one metrics dict into the reporting column."""
    return pd.Series({
        "annual return (gross)": f"{m['gross_annual_return']:+.2%}",
        "annual return (net)": f"{m['annual_return']:+.2%}",
        "annual vol": f"{m['annual_vol']:.2%}",
        "Sharpe (gross)": f"{m['gross_sharpe']:.2f}",
        "Sharpe (net)": f"{m['sharpe']:.2f}",
        "skew (monthly)": f"{m['skew_monthly']:+.2f}",
        "skew (daily)": f"{m['skew_daily']:+.2f}",
        "excess kurtosis (monthly)": f"{m['kurtosis_monthly']:.2f}",
        "max drawdown (net)": f"{m['max_drawdown']:.2%}",
        "drawdown peak": f"{m['dd_peak']:%Y-%m-%d}",
        "drawdown trough": f"{m['dd_trough']:%Y-%m-%d}",
        "drawdown recovered": ("never" if pd.isna(m["dd_recovery"])
                               else f"{m['dd_recovery']:%Y-%m-%d}"),
        "worst month": f"{m['worst_month']:+.2%} ({m['worst_month_date']:%Y-%m})",
        "hit rate (monthly)": f"{m['hit_rate_monthly']:.1%}",
        "annual turnover": f"{m['annual_turnover']:.2f}x",
        "cost drag (annual)": f"{m['cost_drag']:.3%}",
        "months": m["n_months"],
    })


def score(book, span):
    return summary(book.loc[span[0]:span[1]])


def main():
    px, rates, ends = data.get_panel()
    rets = excess_returns(px, rates)
    print("series used")
    for ccy, (sid, quote) in data.SPOT.items():
        print(f"  {ccy} spot  {sid} ({quote}), rate {data.RATE[ccy]}")
    print(f"  USD rate  {data.RATE['USD']}")
    print(f"\nsample {px.index[0].date()} to {px.index[-1].date()}, {len(ends)} month ends")
    print(f"in-sample {IS[0]} to {IS[1]}  |  final test {TEST[0]} to {px.index[-1].date()}")
    print(f"long top {N_LEG} / short bottom {N_LEG} by 3m rate, {COST_BPS:.0f}bps one-way cost")

    last_actual = {c: data.fetch(s).dropna().index[-1].strftime("%Y-%m")
                   for c, s in data.RATE.items()}
    print("last published rate observation: " + ", ".join(f"{c} {d}" for c, d in last_actual.items()))

    show("Average 3m rate by currency, percent (in-sample / final test)",
         pd.DataFrame({"in-sample": rates.loc[IS[0]:IS[1], :].mean(),
                       "final test": rates.loc[TEST[0]:, :].mean()}).round(2).T)

    print("\n" + "=" * 78)
    print("UNCOVERED INTEREST PARITY")
    print("=" * 78)
    print("regress next month's log spot change on (i_usd - i_fx)/12; UIP says slope = 1")
    is_ends, test_ends = ends[ends <= IS[1]], ends[ends >= TEST[0]]
    uip = {}
    for name, span in (("in-sample", is_ends), ("final test", test_ends), ("full", ends)):
        table = uip_regression(px, rates, span)
        uip[name] = table
        show(f"UIP regression, {name} ({len(span)} months)", table.round(3))
    uip["full"].to_csv(os.path.join(REPORTS, "uip_regression.csv"))

    w = carry_weights(rates, ends)
    w_dn = carry_weights(rates, ends, dollar_neutral=True)
    book = run(w, rets)
    book_dn = run(w_dn, rets)

    print("\n" + "=" * 78)
    print("CARRY BACKTEST")
    print("=" * 78)
    for name, span in (("in-sample", IS), ("final test", TEST)):
        show(f"G10 carry, {name} ({span[0]} to {span[1] or px.index[-1].date()})",
             pd.DataFrame({"USD ranked": fmt(score(book, span)),
                           "dollar neutral": fmt(score(book_dn, span))}))
    full = pd.DataFrame({"USD ranked": fmt(summary(book)),
                         "dollar neutral": fmt(summary(book_dn))})
    show("G10 carry, full sample", full)
    full.to_csv(os.path.join(REPORTS, "performance.csv"))

    accrual = rets - px.pct_change().reindex(columns=rets.columns).fillna(0.0)
    held = w.reindex(rets.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    decomposition = {}
    for name, span in (("in-sample", IS), ("final test", TEST), ("full", (None, None))):
        sl = slice(span[0], span[1])
        yrs = len(held.loc[sl]) / TRADING_DAYS
        decomposition[name] = {
            "interest accrual": (held * accrual).loc[sl].sum().sum() / yrs,
            "spot": (held * (rets - accrual)).loc[sl].sum().sum() / yrs,
            "gross (arithmetic)": book.loc[sl, "gross"].sum() / yrs,
        }
    show("Where the gross return comes from, per year (USD ranked)",
         pd.DataFrame(decomposition).map("{:+.2%}".format))
    n_years = len(book) / TRADING_DAYS
    print(f"\nstandard error of a full-sample Sharpe ratio, roughly 1/sqrt(years): "
          f"{1 / np.sqrt(n_years):.2f} over {n_years:.1f} years")

    usd_held = w["USD"].ne(0).mean()
    print(f"\nUSD is in one of the tails on {usd_held:.0%} of rebalance dates; "
          f"average net dollar exposure {book['net_dollar'].mean():+.2f}")
    show("Share of rebalance dates each currency is held long / short (USD ranked)",
         pd.DataFrame({"long": (w > 0).mean(), "short": (w < 0).mean()}).T.map("{:.0%}".format))

    years = by_year(book)
    show("G10 carry by calendar year (USD ranked)",
         years.assign(gross=years["gross"].map("{:+.2%}".format),
                      net=years["net"].map("{:+.2%}".format),
                      sharpe=years["sharpe"].round(2),
                      max_dd=years["max_dd"].map("{:.1%}".format),
                      turnover=years["turnover"].round(2)))
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))

    m = monthly(book["net"])
    print(f"\nmonthly returns: mean {m.mean():+.2%}, sd {m.std():.2%}, "
          f"worst {m.min():+.2%} ({m.idxmin():%Y-%m}), best {m.max():+.2%} ({m.idxmax():%Y-%m})")
    print(f"months below -5%: {(m < -0.05).sum()}   months above +5%: {(m > 0.05).sum()}")

    print("\n" + "=" * 78)
    print("CRISIS AUTOPSIES")
    print("=" * 78)
    for name, span in CRISES.items():
        autopsy(name, span, book, w, rets, px, rates)

    cost_sensitivity(w, w_dn, rets)
    charts(book, book_dn, w, px, rates, ends, years, uip["full"])
    print(f"\ncharts and tables written to {REPORTS}/")


def autopsy(name, span, book, w, rets, px, rates):
    """Report what the book held going into a crisis and where the losses came from."""
    print(f"\n{name}: {span[0]} to {span[1]}")
    chunk = book.loc[span[0]:span[1]]
    held = w.reindex(rets.index).ffill().shift(1).loc[span[0]:span[1]]
    contrib = (held * rets.loc[span[0]:span[1]]).sum()
    entering = w.loc[:span[0]].iloc[-1]
    print(f"  net return over the window {(1 + chunk['net']).prod() - 1:+.2%}, "
          f"gross {(1 + chunk['gross']).prod() - 1:+.2%}")
    dd = max_drawdown(chunk["net"])
    print(f"  worst drawdown inside the window {dd['max_drawdown']:.2%} "
          f"({dd['dd_peak']:%Y-%m-%d} to {dd['dd_trough']:%Y-%m-%d})")
    longs = [c for c in entering.index if entering[c] > 0]
    shorts = [c for c in entering.index if entering[c] < 0]
    print(f"  entering: long {', '.join(longs)}  short {', '.join(shorts)}")
    spot_move = (px.loc[:span[1]].iloc[-1] / px.loc[:span[0]].iloc[-1] - 1)
    table = pd.DataFrame({
        "weight entering": entering.map("{:+.2f}".format),
        "spot move vs USD": spot_move.reindex(entering.index).map(
            lambda v: "" if pd.isna(v) else f"{v:+.1%}"),
        "contribution": contrib.map("{:+.2%}".format),
    })
    show("  by currency", table[(entering != 0) | (contrib.abs() > 1e-6)])
    worst = chunk["net"].nsmallest(3)
    show("  worst days", pd.DataFrame({
        "book": worst.map("{:+.2%}".format),
        "long leg": book.loc[worst.index, "long"].map("{:+.2%}".format),
        "short leg": book.loc[worst.index, "short"].map("{:+.2%}".format)}))


def cost_sensitivity(w, w_dn, rets):
    rows = {}
    for bps in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        a, b = run(w, rets, bps), run(w_dn, rets, bps)
        rows[f"{bps:.0f}bps"] = {
            "USD ranked, in-sample": round(score(a, IS)["sharpe"], 2),
            "USD ranked, final test": round(score(a, TEST)["sharpe"], 2),
            "dollar neutral, in-sample": round(score(b, IS)["sharpe"], 2),
            "dollar neutral, final test": round(score(b, TEST)["sharpe"], 2),
            "USD ranked, annual cost": f"{summary(a)['cost_drag']:.3%}",
        }
    table = pd.DataFrame(rows).T
    show("Net Sharpe by one-way transaction cost", table)
    table.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))


def charts(book, book_dn, w, px, rates, ends, years, uip):
    split = pd.Timestamp(data.TEST_START)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, s, style in (("G10 carry, gross", book["gross"], "--"),
                           ("G10 carry, net", book["net"], "-"),
                           ("dollar neutral, net", book_dn["net"], "-")):
        ax.plot((1 + s).cumprod(), style, linewidth=1.1, label=name)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.text(split, ax.get_ylim()[1], " final test", va="top", fontsize=8)
    ax.set_ylabel("growth of 1")
    ax.set_title("G10 carry, long top 3 / short bottom 3 by 3m rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    equity = (1 + book["net"]).cumprod()
    underwater = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(underwater.index, underwater, 0, color="firebrick", alpha=0.55)
    ax.axvline(split, color="black", linewidth=0.9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Drawdown, net of costs")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "drawdown.png"), dpi=140)
    plt.close(fig)

    m = monthly(book["net"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(m, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"Monthly net returns, skew {m.skew():+.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "monthly_histogram.png"), dpi=140)
    plt.close(fig)

    crash = slice("2008-01-01", "2009-12-31")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, s in (("carry, net", book.loc[crash, "net"]),
                    ("long leg", book.loc[crash, "long"]),
                    ("short leg", book.loc[crash, "short"])):
        ax.plot((1 + s).cumprod(), linewidth=1.2, label=name)
    ax.axhline(1.0, color="grey", linewidth=0.7)
    ax.set_title("2008-2009, rebased to 1 on 2008-01-01")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "crisis_2008.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["steelblue" if y < split.year else "darkorange" for y in years.index]
    ax.bar(years.index, years["net"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Net return by year (orange = final test)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "by_year.png"), dpi=140)
    plt.close(fig)

    held = w.T
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.imshow(held.values, aspect="auto", cmap="RdBu", vmin=-0.4, vmax=0.4,
              extent=[0, len(held.columns), len(held.index), 0])
    ax.set_yticks(np.arange(len(held.index)) + 0.5)
    ax.set_yticklabels(held.index)
    ticks = [i for i, d in enumerate(held.columns) if d.month == 12 and d.year % 4 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([held.columns[i].year for i in ticks])
    ax.set_title("Holdings by month: blue long, red short")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "holdings.png"), dpi=140)
    plt.close(fig)

    s = np.log(px.loc[ends])
    ds = (s.shift(-1) - s).stack()
    fp = (-differential(rates.loc[ends]).drop(columns="USD") / 100.0 / 12.0).stack()
    both = pd.concat([fp, ds], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(both.iloc[:, 0], both.iloc[:, 1], s=6, alpha=0.4)
    grid = np.linspace(both.iloc[:, 0].min(), both.iloc[:, 0].max(), 2)
    ax.plot(grid, grid, color="grey", linewidth=1, label="UIP: slope 1")
    ax.plot(grid, uip.loc["pooled", "alpha_monthly"] + uip.loc["pooled", "slope"] * grid,
            color="firebrick", linewidth=1.2, label=f"fitted, slope {uip.loc['pooled', 'slope']:.2f}")
    ax.set_xlabel("(i_usd - i_fx) / 12, monthly")
    ax.set_ylabel("next month log spot change (USD per fx)")
    ax.set_title("Fama regression, all currencies pooled")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "uip_scatter.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)
    main()
