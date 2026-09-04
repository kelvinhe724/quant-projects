"""Full study on real Binance data: funding, basis, the carry trade, and its tails.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import carry
import data
from carry import (PERIODS_PER_YEAR, annualise, autocorrelation, basis, drivers,
                   funding_filter, implied_funding, metrics, simulate,
                   yearly_funding)

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
MARGIN_FRACS = (0.1, 0.2, 0.3, 0.5, 1.0)
COST_GRID = ((0.0, 0.0), (5.0, 2.0), (10.0, 5.0), (20.0, 10.0), (40.0, 20.0))
DELEVERAGING = {
    "Mar 2020 covid crash": ("2020-03-08", "2020-03-31"),
    "May 2021 China ban": ("2021-05-12", "2021-07-31"),
    "Jun 2022 3AC and Celsius": ("2022-06-01", "2022-07-31"),
    "Nov 2022 FTX": ("2022-11-05", "2022-12-31"),
}


def show(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(rows.to_string())


def sharpe_over(excess):
    """Annualised Sharpe of a per-period excess return series."""
    return excess.mean() / excess.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def funding_section(panels, bills):
    """Distribution, annualised carry by year, persistence, and what drives it."""
    stats = pd.DataFrame({
        sym: {
            "periods": len(p),
            "mean 8h": p["rate"].mean(),
            "median 8h": p["rate"].median(),
            "std 8h": p["rate"].std(),
            "annualised (simple)": annualise(p["rate"].mean()),
            "annualised (compound)": (1 + p["rate"]).prod()
            ** (PERIODS_PER_YEAR / len(p)) - 1,
            "share negative": (p["rate"] < 0).mean(),
            "1st pctile": p["rate"].quantile(0.01),
            "99th pctile": p["rate"].quantile(0.99),
            "at the 0.75% cap": (p["rate"].abs() >= carry.RATE_CAP - 1e-9).mean(),
            "at the 0.01% floor": np.isclose(p["rate"], carry.INTEREST).mean(),
        } for sym, p in panels.items()})
    show("Funding rate distribution, 8h payments", stats.round(6))
    stats.to_csv(os.path.join(REPORTS, "funding_stats.csv"))

    years = pd.DataFrame({
        f"{sym} annualised": yearly_funding(p)["annualised_compound"]
        for sym, p in panels.items()})
    for sym, p in panels.items():
        years[f"{sym} % negative"] = yearly_funding(p)["pct_negative"]
    years["3m T-bill"] = bills.groupby(bills.index.year).mean()
    show("Annualised funding by year (2026 is to 31 July)",
         years.apply(lambda c: c.map("{:.2%}".format)))
    years.to_csv(os.path.join(REPORTS, "funding_by_year.csv"))

    # Binance sets funding to a fixed 0.01% per 8h whenever the premium sits
    # inside the clamp, so most of the average carry is that constant rather
    # than anything the market is paying.
    floor = annualise(carry.INTEREST)
    decomp = pd.DataFrame({"interest component": floor}, index=years.index)
    for sym, p in panels.items():
        total = yearly_funding(p)["annualised_simple"]
        decomp[f"{sym} total"] = total
        decomp[f"{sym} premium"] = total - floor
    show("Where the carry comes from: the exchange's fixed interest term vs the "
         "market premium", decomp.apply(lambda c: c.map("{:.2%}".format)))
    decomp.to_csv(os.path.join(REPORTS, "funding_decomposition.csv"))

    ac = pd.DataFrame({sym: autocorrelation(p["rate"])
                       for sym, p in panels.items()})
    ac.index = [f"{k} periods ({k * 8 / 24:.0f}d)" for k in ac.index]
    show("Funding autocorrelation", ac.round(3))

    rows = []
    for sym, p in panels.items():
        d = drivers(p)
        x = sm.add_constant(d[["momentum", "vol"]])
        fit = sm.OLS(d["rate"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 90})
        rows.append({"symbol": sym, "beta momentum": fit.params["momentum"],
                     "t momentum": fit.tvalues["momentum"],
                     "beta vol": fit.params["vol"], "t vol": fit.tvalues["vol"],
                     "R2": fit.rsquared, "n": int(fit.nobs)})
    show("Funding on 30d trailing spot momentum and realised vol "
         "(Newey-West, 90 lags)", pd.DataFrame(rows).set_index("symbol").round(5))


def basis_section(panels):
    """Check the observed premium reproduces the funding the exchange charged."""
    rows = []
    for sym, p in panels.items():
        prem = basis(p)
        pred = pd.Series(implied_funding(prem), index=p.index)
        fit = sm.OLS(p["rate"], sm.add_constant(pred.rename("implied"))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 90})
        rows.append({
            "symbol": sym,
            "mean basis": prem.mean(),
            "median basis": prem.median(),
            "std basis": prem.std(),
            "share positive": (prem > 0).mean(),
            "corr(funding, implied)": np.corrcoef(p["rate"], pred)[0, 1],
            "slope on implied": fit.params["implied"],
            "R2": fit.rsquared,
        })
    show("Perp-spot basis at settlement vs the funding it implies",
         pd.DataFrame(rows).set_index("symbol").round(5))


def strategy_section(panels, bills):
    """Headline carry results, cost sensitivity and the margin buffer sweep."""
    books = {}
    for sym, p in panels.items():
        gross = simulate(p, spot_fee_bps=0.0, perp_fee_bps=0.0, slip_bps=0.0)
        net = simulate(p)
        books[sym] = net
        net[["equity", "funding", "fees"]].to_csv(
            os.path.join(REPORTS, f"equity_{sym}.csv"))
        table = pd.DataFrame({"gross": metrics(gross), "net": metrics(net)})
        table = table.drop(index=["liquidated_at", "equity_at_liquidation"])
        # The plain Sharpe above is on raw returns. The position ties up cash,
        # so the honest ratio subtracts the T-bill rate the cash would have earned.
        cash = bills.reindex(p.index.tz_localize(None), method="ffill").to_numpy()
        cash = cash / PERIODS_PER_YEAR
        table.loc["sharpe_over_cash"] = [
            sharpe_over(b["ret"].to_numpy() - cash) for b in (gross, net)]
        show(f"{sym} delta-neutral carry, 50% margin buffer, "
             "30-day rebalance", table.astype(float).round(4))
        table.to_csv(os.path.join(REPORTS, f"performance_{sym}.csv"))

    yearly = pd.DataFrame({
        sym: b["ret"].groupby(b.index.year).apply(lambda r: (1 + r).prod() - 1)
        for sym, b in books.items()})
    yearly["3m T-bill"] = bills.groupby(bills.index.year).mean()
    for sym in books:
        yearly[f"{sym} over cash"] = yearly[sym] - yearly["3m T-bill"]
    show("Carry return by year, net of costs, against the cash it ties up",
         yearly.apply(lambda c: c.map("{:+.2%}".format)))
    yearly.to_csv(os.path.join(REPORTS, "carry_by_year.csv"))

    rows = []
    for sym, p in panels.items():
        for spot_bps, perp_bps in COST_GRID:
            m = metrics(simulate(p, spot_fee_bps=spot_bps, perp_fee_bps=perp_bps))
            rows.append({"symbol": sym, "spot bps": spot_bps, "perp bps": perp_bps,
                         "annual return": m["annual_return"], "sharpe": m["sharpe"],
                         "fees paid": m["fees_paid"]})
    costs = pd.DataFrame(rows).set_index(["symbol", "spot bps"])
    show("Sensitivity to the fee assumption (1bp slippage per fill on top)",
         costs.round(4))
    costs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))

    rows = []
    for sym, p in panels.items():
        for mark in ("spot", "perp"):
            for mf in MARGIN_FRACS:
                m = metrics(simulate(p, margin_frac=mf, mark=mark))
                rows.append({
                    "symbol": sym, "mark": mark, "margin buffer": mf,
                    "annual return": m["annual_return"], "sharpe": m["sharpe"],
                    "max drawdown": m["max_drawdown"],
                    "liquidated": "" if m["liquidated_at"] is None
                    else str(m["liquidated_at"].date()),
                    "equity then": m["equity_at_liquidation"]})
    buffers = pd.DataFrame(rows).set_index(["symbol", "mark", "margin buffer"])
    show("Margin buffer sweep. `mark=spot` is how Binance actually liquidates; "
         "`mark=perp` prices the wick", buffers.round(4))
    buffers.to_csv(os.path.join(REPORTS, "margin_buffer.csv"))

    rows = []
    for sym, p in panels.items():
        base = metrics(simulate(p))
        filt = metrics(simulate(p, hold=funding_filter(p)))
        rows.append({"symbol": sym, "always on": base["annual_return"],
                     "always on sharpe": base["sharpe"],
                     "filtered": filt["annual_return"],
                     "filtered sharpe": filt["sharpe"],
                     "filtered fees": filt["fees_paid"]})
    show("Switching off while the trailing 7-day funding mean is negative",
         pd.DataFrame(rows).set_index("symbol").round(4))
    return books


def risk_section(panels, books):
    """Negative funding runs, deleveraging episodes, tails, and stablecoin risk."""
    rows = []
    for sym, p in panels.items():
        neg = p["rate"] < 0
        runs = neg.ne(neg.shift()).cumsum()[neg]
        lengths = runs.value_counts()
        worst = runs.value_counts().idxmax()
        window = p.loc[runs[runs == worst].index]
        rows.append({
            "symbol": sym,
            "longest negative run (periods)": int(lengths.max()),
            "in days": lengths.max() * 8 / 24,
            "run started": str(window.index[0].date()),
            "cost of that run": window["rate"].sum(),
            "worst 30d funding, annualised": annualise(
                p["rate"].rolling(90).mean().min()),
            "worst 90d funding, annualised": annualise(
                p["rate"].rolling(270).mean().min()),
        })
    show("Extended negative funding", pd.DataFrame(rows).set_index("symbol").round(4))

    rows = []
    for name, (lo, hi) in DELEVERAGING.items():
        row = {"episode": name}
        for sym, p in panels.items():
            w = p.loc[lo:hi]
            row[f"{sym} funding, ann."] = annualise(w["rate"].mean())
            row[f"{sym} % negative"] = (w["rate"] < 0).mean()
            b = books[sym].loc[lo:hi]
            row[f"{sym} carry P&L"] = (1 + b["ret"]).prod() - 1
        rows.append(row)
    episodes = pd.DataFrame(rows).set_index("episode")
    show("Deleveraging episodes", episodes.round(4))
    episodes.to_csv(os.path.join(REPORTS, "deleveraging.csv"))

    rows = []
    for sym, b in books.items():
        r = b["ret"]
        m = metrics(b)
        rows.append({
            "symbol": sym, "skew": m["skew"], "excess kurtosis": m["excess_kurtosis"],
            "worst 8h": r.min(), "worst day": r.rolling(3).sum().min(),
            "worst month": r.rolling(90).sum().min(),
            "1% quantile": m["q01"], "1% under a normal": m["normal_q01"],
            "ratio": m["q01"] / m["normal_q01"],
            "share of periods beyond 5 sd":
                (r.sub(r.mean()).abs() > 5 * r.std()).mean(),
        })
    show("Return tails, 8h periods", pd.DataFrame(rows).set_index("symbol").round(6))

    usdc = data.stablecoin()
    liquid = usdc[usdc["volume"] > usdc["volume"].median() * 0.05]
    off = (liquid["close"] - 1).abs()
    show("USDC/USDT on Binance, a direct read on stablecoin risk to the quote leg",
         pd.Series({
             "median close": f"{liquid['close'].median():.4f}",
             "max close": f"{liquid['close'].max():.4f}",
             "min close": f"{liquid['close'].min():.4f}",
             "worst deviation from 1.00": f"{off.max():.2%}",
             "date of that": str(off.idxmax().date()),
             "share of 8h bars more than 0.5% off": f"{(off > 0.005).mean():.2%}",
             "share more than 2% off": f"{(off > 0.02).mean():.2%}",
         }))


def charts(panels, books, bills):
    roll = 90  # 30 days of 8h periods

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for sym, p in panels.items():
        ax.plot(annualise(p["rate"].rolling(roll).mean()), linewidth=1, label=sym)
    ax.plot(bills, color="black", linewidth=1, linestyle="--", label="3m T-bill")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("annualised")
    ax.set_title("Funding rate, trailing 30-day mean, annualised")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "funding_history.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sym, p in panels.items():
        axes[0].hist(p["rate"] * 1e4, bins=200, histtype="step", label=sym)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("8h funding, bps")
    axes[0].set_xlim(-30, 40)
    axes[0].axvline(0, color="grey", linewidth=0.8)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distribution of 8h funding")
    btc = panels["BTCUSDT"]
    axes[1].boxplot([g["rate"].to_numpy() * 1e4 for _, g in btc.groupby(btc.index.year)],
                    tick_labels=sorted(btc.index.year.unique()), showfliers=False)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_ylabel("8h funding, bps")
    axes[1].set_title("BTCUSDT funding by year")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "funding_distribution.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    p = panels["BTCUSDT"]
    prem = basis(p)
    ax.scatter(implied_funding(prem) * 1e4, p["rate"] * 1e4, s=2, alpha=0.2)
    lim = [-40, 60]
    ax.plot(lim, lim, color="firebrick", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("funding implied by the observed basis, bps")
    ax.set_ylabel("funding actually charged, bps")
    ax.set_title("BTCUSDT: basis against realised funding")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "basis_vs_funding.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for sym, b in books.items():
        axes[0].plot(b["equity"], linewidth=1.2, label=f"{sym} carry, net")
    cash = (1 + bills.reindex(books["BTCUSDT"].index.tz_localize(None),
                              method="ffill").to_numpy() / 365 / 3).cumprod()
    axes[0].plot(books["BTCUSDT"].index, cash, color="black", linestyle="--",
                 linewidth=1, label="3m T-bill, rolled")
    axes[0].set_ylabel("growth of 1")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Delta-neutral carry against cash")
    for sym, b in books.items():
        eq = b["equity"]
        axes[1].fill_between(eq.index, eq / eq.cummax() - 1, 0, alpha=0.5, label=sym)
    axes[1].set_ylabel("drawdown")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for mf in MARGIN_FRACS:
        b = simulate(panels["BTCUSDT"], margin_frac=mf)
        ax.plot(b["equity"], linewidth=1, label=f"buffer {mf:.0%}")
        when = b.attrs["liquidated_at"]
        if when is not None:
            ax.scatter([when], [b["equity"].loc[when]], marker="x", color="black", s=40)
    ax.set_title("BTCUSDT carry by margin buffer, crosses mark liquidation")
    ax.set_ylabel("growth of 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "margin_buffer.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (sym, b) in zip(axes, books.items()):
        z = ((b["ret"] - b["ret"].mean()) / b["ret"].std()).sort_values()
        theory = norm.ppf((np.arange(len(z)) + 0.5) / len(z))
        ax.plot(theory, z, ".", markersize=2)
        ax.plot(theory[[0, -1]], theory[[0, -1]], color="firebrick", linewidth=1)
        ax.set_title(f"{sym}: 8h returns vs normal")
        ax.set_xlabel("normal quantile")
        ax.set_ylabel("observed, standardised")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "tails.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    panels = data.load_all()
    bills = data.tbill()
    for sym, p in panels.items():
        print(f"{sym}: {len(p)} funding periods, {p.index[0]} to {p.index[-1]}, "
              f"all at {int(p['interval_hours'].iloc[0])}h intervals")

    funding_section(panels, bills)
    basis_section(panels)
    books = strategy_section(panels, bills)
    risk_section(panels, books)
    charts(panels, books, bills)
    print(f"\ncharts and tables written to {REPORTS}/")


if __name__ == "__main__":
    main()
