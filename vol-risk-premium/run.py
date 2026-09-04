"""Measure the volatility risk premium on SPY, then trade it and count the tail.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import data
from vrp import (HORIZON, TRADING_DAYS, drawdown, forward_realised_vol, garch_forecast,
                 hac_mean, metrics, simulate, straddle_price)

REPORTS = data.REPORTS
COST_BPS = 1.0
OPTION_SPREAD = 0.01

EPISODES = {
    "Aug 2011 US downgrade": ("2011-07-25", "2011-08-31"),
    "Aug 2015 China devaluation": ("2015-08-17", "2015-09-30"),
    "Feb 2018 Volmageddon": ("2018-01-26", "2018-02-28"),
    "Dec 2018 selloff": ("2018-12-03", "2018-12-31"),
    "Mar 2020 Covid": ("2020-02-19", "2020-04-30"),
    "Aug 2024 yen carry unwind": ("2024-07-31", "2024-08-15"),
    "Apr 2025 tariffs": ("2025-03-28", "2025-04-30"),
}


def show(title, obj):
    print(f"\n{title}")
    print("-" * len(title))
    print(obj if isinstance(obj, str) else obj.to_string())


def chain_check():
    """Compare the VIX against an ATM implied vol backed out of live SPY option mids."""
    try:
        q, spot, snap = data.get_chain()
    except Exception as exc:
        print(f"\noption chain unavailable ({exc}); skipping the cross-check")
        return
    if q is None or q.empty:
        print("\nno listed expiry in the 20-45 day window; skipping the cross-check")
        return

    rows = []
    for (exp, T), g in q.groupby(["expiry", "T"]):
        piv = g.pivot_table(index="strike", columns="cp", values="mid").dropna()
        if len(piv) < 5:
            continue
        near = piv.index[np.argmin(np.abs(piv.index - spot))]
        K = float(near)
        straddle = float(piv.loc[near, "C"] + piv.loc[near, "P"])

        # Put-call parity on the strikes around spot gives the forward without
        # assuming a rate or a dividend yield, which is what the simulator does
        # assume. The gap between F and spot is the size of that approximation.
        band = piv[(piv.index > 0.95 * spot) & (piv.index < 1.05 * spot)]
        slope, intercept = np.polyfit(band.index.to_numpy(float),
                                      (band["C"] - band["P"]).to_numpy(float), 1)
        fwd = intercept / -slope

        lo, hi = 1e-4, 5.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if straddle_price(spot, K, T, mid) < straddle:
                lo = mid
            else:
                hi = mid
        rows.append({"expiry": exp.date(), "days": round(T * 365),
                     "strike": K, "straddle mid": round(straddle, 2),
                     "ATM IV": f"{0.5 * (lo + hi):.2%}",
                     "F/S - 1": f"{fwd / spot - 1:+.3%}",
                     "market IV (yahoo)": f"{g.impliedVolatility.median():.2%}"})

    daily = data.get_daily()
    vix_now = daily["vix"].iloc[-1]
    show(f"Live SPY chain vs VIX  (spot {spot:.2f}, snapshot {snap:%Y-%m-%d}, "
         f"last VIX close {vix_now:.2f})", pd.DataFrame(rows))
    print("The straddle IV backed out of real mids is what the simulator would have\n"
          "sold. The VIX is a variance-swap rate over the whole strike ladder, so it\n"
          "sits above the ATM point by the skew premium.")


def measure(df):
    """Compare implied against subsequently realised volatility."""
    iv = df["vix"] / 100.0
    rv = forward_realised_vol(df["ret"], HORIZON)
    gap = (iv - rv).dropna()

    stats = pd.Series({
        "mean implied": f"{iv.reindex(gap.index).mean():.2%}",
        "mean realised (next 21d)": f"{rv.reindex(gap.index).mean():.2%}",
        "mean gap": f"{gap.mean():.2%}",
        "median gap": f"{gap.median():.2%}",
        "gap std": f"{gap.std():.2%}",
        "gap 5th pct": f"{gap.quantile(0.05):.2%}",
        "gap 95th pct": f"{gap.quantile(0.95):.2%}",
        "gap min": f"{gap.min():.2%}",
        "days positive": f"{(gap > 0).mean():.1%}",
        "implied / realised ratio": f"{(iv.reindex(gap.index) / rv.reindex(gap.index)).median():.2f}x",
    })
    show("Implied minus subsequently realised volatility, SPY 2010-2026", stats)

    h = hac_mean(gap)
    print(f"\nNewey-West mean gap {h['mean']:+.4f} ({h['mean']:.2%}), "
          f"se {h['se']:.4f}, t = {h['t']:.2f} on {h['n']} overlapping days.")
    print("The windows overlap 20 of 21 days, so the naive t-stat "
          f"({gap.mean() / gap.std() * np.sqrt(len(gap)):.1f}) is meaningless; "
          f"{int(1.5 * HORIZON)} Newey-West lags is the honest version.")

    lags = [1, 5, 21, 63, 126, 252]
    ac = pd.Series({f"{l}d": gap.autocorr(l) for l in lags}, name="autocorrelation")
    show("Persistence of the gap", ac.round(3).to_frame())

    fwd_spy = np.exp(np.log(df["spy"]).diff().rolling(HORIZON).sum().shift(-HORIZON)) - 1
    cond = pd.DataFrame({
        "gap": gap,
        "VIX level": iv.reindex(gap.index),
        "forward 21d SPY return": fwd_spy.reindex(gap.index),
        "trailing 21d realised vol": (np.sqrt(TRADING_DAYS * (df["ret"] ** 2)
                                              .rolling(HORIZON).mean())).reindex(gap.index),
    }).dropna()
    show("Correlation of the gap with market conditions", cond.corr()["gap"].round(3).to_frame())

    buckets = cond.groupby(pd.qcut(cond["VIX level"], 5,
                                   labels=["VIX q1 (calm)", "q2", "q3", "q4", "q5 (panic)"]),
                           observed=True)["gap"].agg(
        **{"mean gap": "mean", "worst gap": "min", "share positive": lambda g: (g > 0).mean(),
           "days": "size"})
    buckets["mean gap"] = buckets["mean gap"].map("{:.2%}".format)
    buckets["worst gap"] = buckets["worst gap"].map("{:.2%}".format)
    buckets["share positive"] = buckets["share positive"].map("{:.1%}".format)
    show("Gap by starting VIX quintile", buckets)
    return iv, rv, gap


def forecast_comparison(df, iv, rv):
    """Is the premium a bad forecast or a risk premium? Race GARCH against the VIX."""
    cache = REPORTS / "garch_forecast.csv"
    if cache.exists():
        g = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        print("\nfitting rolling GARCH(1,1), a few hundred fits, this takes a minute")
        g = garch_forecast(df["ret"], HORIZON)
        g.to_csv(cache)

    panel = pd.DataFrame({"iv": iv, "garch": g, "rv": rv}).dropna()
    rows = []
    for name, f in (("VIX implied", panel["iv"]), ("GARCH(1,1) forecast", panel["garch"])):
        err = f - panel["rv"]
        fit = sm.OLS(panel["rv"], sm.add_constant(f.rename("x"))).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(1.5 * HORIZON)})
        rows.append({
            "forecast": name,
            "mean level": f"{f.mean():.2%}",
            "mean error vs realised": f"{err.mean():+.2%}",
            "RMSE": f"{np.sqrt((err ** 2).mean()):.2%}",
            "corr with realised": round(f.corr(panel["rv"]), 3),
            "MZ intercept": round(fit.params.iloc[0], 4),
            "MZ slope": round(fit.params.iloc[1], 3),
            "MZ R2": round(fit.rsquared, 3),
        })
    show("Forecasting the next 21 days of realised vol", pd.DataFrame(rows).set_index("forecast"))
    print("A Mincer-Zarnowitz regression rv = a + b*forecast is unbiased when a = 0\n"
          "and b = 1. A forecast that is right about the shape but sits above the\n"
          "level shows up as slope near 1 with a negative intercept.")

    both = sm.OLS(panel["rv"], sm.add_constant(panel[["iv", "garch"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(1.5 * HORIZON)})
    print(f"\nEncompassing regression rv ~ iv + garch: "
          f"iv {both.params['iv']:.3f} (t {both.tvalues['iv']:.2f}), "
          f"garch {both.params['garch']:.3f} (t {both.tvalues['garch']:.2f}), "
          f"R2 {both.rsquared:.3f}")
    return panel


def strategy(df):
    """Roll the short delta-hedged straddle and score it."""
    sim = simulate(df, cost_bps=COST_BPS, option_spread=OPTION_SPREAD)
    gross = simulate(df, cost_bps=0.0, option_spread=0.0)["pnl"]
    m = metrics(sim["pnl"])
    gm = metrics(gross)

    prem = sim["premium"][sim["premium"] > 0]
    print(f"\n{len(prem)} monthly cycles, {len(sim)} trading days. "
          f"Average straddle premium sold: {prem.mean():.2%} of spot "
          f"(min {prem.min():.2%}, max {prem.max():.2%}).")
    print(f"Assumptions: {COST_BPS:.1f}bp one-way slippage on every hedge share, "
          f"{OPTION_SPREAD:.1%} of mid given up on the option, options marked at the "
          "VIX, zero rate, zero dividend, no margin financing.")

    table = pd.DataFrame({
        "gross of costs": fmt(gm),
        "net of costs": fmt(m),
    })
    show("Short 1-month ATM straddle, delta-hedged daily, per $1 of straddle notional",
         table)
    table.to_csv(REPORTS / "performance.csv")
    return sim, m


def fmt(m):
    return pd.Series({
        "total P&L (notional)": f"{m['total_pnl']:+.3f}",
        "annual P&L (notional)": f"{m['annual_pnl']:+.2%}",
        "annual vol": f"{m['annual_vol']:.2%}",
        "Sharpe": f"{m['sharpe']:.2f}",
        "Sharpe excl. worst 1% of days": f"{m['sharpe_ex_worst_1pct']:.2f}",
        "skew": f"{m['skew']:.2f}",
        "excess kurtosis": f"{m['excess_kurtosis']:.1f}",
        "worst day": f"{m['worst_day']:.2%}",
        "best day": f"{m['best_day']:.2%}",
        "max drawdown": f"{m['max_drawdown']:.2%}",
        "hit rate": f"{m['hit_rate']:.1%}",
        "days": f"{m['days']:.0f}",
    })


def tails(df, sim, m):
    """The part that decides whether the Sharpe means anything."""
    p = sim["pnl"]
    normal_kurt = pd.Series(np.random.default_rng(1).normal(0, p.std(), len(p))).kurtosis()
    worst = p.nsmallest(10)
    tbl = pd.DataFrame({"P&L (notional)": worst.map("{:.2%}".format),
                        "SPY return": np.log(df["spy"]).diff().reindex(
                            worst.index).map("{:+.2%}".format),
                        "VIX": df["vix"].reindex(worst.index).round(1),
                        "as sigma": (worst / p.std()).round(1)})
    show("Ten worst days", tbl)
    print(f"\nA normal series of the same standard deviation returns excess kurtosis "
          f"{normal_kurt:.2f}; this one returns {m['excess_kurtosis']:.1f}.")

    cuts = []
    for k in (0, 1, 3, 5, 10, 21):
        trimmed = p.drop(p.nsmallest(k).index) if k else p
        cuts.append({"worst days removed": k,
                     "annual P&L": f"{trimmed.sum() / (len(trimmed) / TRADING_DAYS):.2%}",
                     "Sharpe": round(trimmed.mean() / trimmed.std() * np.sqrt(TRADING_DAYS), 2),
                     "share of total P&L removed":
                         f"{-p.nsmallest(k).sum() / p.sum():.1%}" if k else "0.0%"})
    show("Sharpe after deleting the worst days", pd.DataFrame(cuts).set_index("worst days removed"))

    rows = []
    for name, (a, b) in EPISODES.items():
        w = p.loc[a:b]
        if w.empty:
            continue
        spy = df["spy"].loc[a:b]
        rows.append({"episode": name, "days": len(w),
                     "P&L (notional)": f"{w.sum():+.2%}",
                     "worst day": f"{w.min():.2%}",
                     "SPY": f"{spy.iloc[-1] / spy.iloc[0] - 1:+.1%}",
                     "VIX high": round(df["vix"].loc[a:b].max(), 1),
                     "months of average P&L": round(w.sum() / (p.sum() / (len(p) / 21)), 1)})
    show("Crisis episodes", pd.DataFrame(rows).set_index("episode"))

    dd = drawdown(p)
    trough = dd.idxmin()
    peak = p.cumsum().loc[:trough].idxmax()
    rec = dd.loc[trough:]
    recovered = rec[rec >= -1e-12]
    print(f"\nDeepest drawdown {dd.min():.2%} of notional, from {peak.date()} to "
          f"{trough.date()} ({(trough - peak).days} calendar days), "
          + (f"recovered {recovered.index[0].date()}"
             if len(recovered) else "never recovered in sample") + ".")

    cyc = sim.groupby("cycle")["pnl"].sum()
    print(f"\nPer cycle: {(cyc > 0).mean():.1%} of {len(cyc)} months positive, "
          f"median {cyc.median():+.2%}, worst {cyc.min():.2%} "
          f"({sim.index[sim['cycle'] == cyc.idxmin()][0].date()}), "
          f"best {cyc.max():+.2%}.")

    spy_ret = np.log(df["spy"]).diff().reindex(p.index)
    down = spy_ret < spy_ret.quantile(0.05)
    print(f"Correlation with SPY daily returns: {p.corr(spy_ret):.2f} overall, "
          f"{p[down].corr(spy_ret[down]):.2f} on the worst 5% of equity days.")
    print(f"Mean P&L on those days: {p[down].mean():.3%} against "
          f"{p[~down].mean():.3%} on the rest.")

    joint = pd.DataFrame({"pnl": p, "spy": spy_ret}).dropna()
    beta_fit = sm.OLS(joint["pnl"], sm.add_constant(joint["spy"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    stripped = joint["pnl"] - beta_fit.params["spy"] * joint["spy"]
    print(f"P&L ~ SPY: beta {beta_fit.params['spy']:.3f} "
          f"(t {beta_fit.tvalues['spy']:.1f}), alpha "
          f"{beta_fit.params['const'] * TRADING_DAYS:.2%} a year "
          f"(t {beta_fit.tvalues['const']:.2f}), R2 {beta_fit.rsquared:.2f}.")
    print(f"Selling the beta away leaves {stripped.sum() / (len(stripped) / TRADING_DAYS):.2%} "
          f"a year at Sharpe {stripped.mean() / stripped.std() * np.sqrt(TRADING_DAYS):.2f}, "
          f"skew {stripped.skew():.2f}, worst day {stripped.min():.2%}.")


def sensitivity(df):
    rows = []
    for cb, osp in ((0.0, 0.0), (0.5, 0.005), (1.0, 0.01), (2.0, 0.02), (5.0, 0.05)):
        m = metrics(simulate(df, cost_bps=cb, option_spread=osp)["pnl"])
        rows.append({"hedge bps (one way)": cb, "option spread": f"{osp:.1%}",
                     "annual P&L": f"{m['annual_pnl']:.2%}", "Sharpe": round(m["sharpe"], 2),
                     "max drawdown": f"{m['max_drawdown']:.2%}"})
    show("Sensitivity to the cost assumption", pd.DataFrame(rows).set_index("hedge bps (one way)"))

    rows = []
    for off in (0.0, 0.01, 0.02, 0.03, 0.04):
        m = metrics(simulate(df, cost_bps=COST_BPS, option_spread=OPTION_SPREAD,
                             iv_offset=off)["pnl"])
        rows.append({"vol points below VIX": f"{off:.0%}",
                     "annual P&L": f"{m['annual_pnl']:.2%}",
                     "Sharpe": round(m["sharpe"], 2),
                     "max drawdown": f"{m['max_drawdown']:.2%}"})
    cal_rows = []
    for off in range(HORIZON):
        pnl = simulate(df, cost_bps=COST_BPS, option_spread=OPTION_SPREAD,
                       offset=off)["pnl"]
        m = metrics(pnl)
        cal_rows.append({"offset": off, "annual P&L": m["annual_pnl"],
                         "max drawdown": m["max_drawdown"],
                         "Feb 2018": pnl.loc["2018-01-26":"2018-02-28"].sum(),
                         "Mar 2020": pnl.loc["2020-02-19":"2020-04-30"].sum(),
                         "Sharpe": m["sharpe"]})
    cal = pd.DataFrame(cal_rows).set_index("offset")
    agg = cal.agg(["min", "median", "max"]).T
    summ = agg.drop(index="Sharpe").map("{:.2%}".format)
    summ.loc["Sharpe"] = agg.loc["Sharpe"].map("{:.2f}".format)
    show("The same strategy started on each of the 21 possible cycle calendars", summ)

    show("Sensitivity to selling below the VIX, which is where an ATM straddle "
         "actually trades", pd.DataFrame(rows).set_index("vol points below VIX"))


def charts(df, iv, rv, gap, panel, sim):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(iv, lw=0.7, label="VIX / 100 (implied)")
    axes[0].plot(rv, lw=0.7, color="firebrick", label="realised over the next 21 days")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("annualised vol")
    axes[0].set_title("Implied volatility against what actually followed")
    axes[1].fill_between(gap.index, gap, 0, where=gap >= 0, color="seagreen", alpha=0.7)
    axes[1].fill_between(gap.index, gap, 0, where=gap < 0, color="firebrick", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("implied minus realised")
    fig.tight_layout()
    fig.savefig(REPORTS / "premium.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(gap, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(gap.mean(), color="firebrick", lw=1.2,
               label=f"mean {gap.mean():.2%}")
    ax.set_title("Distribution of the implied-minus-realised gap")
    ax.set_xlabel("annualised vol points")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "gap_distribution.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(sim["pnl"].cumsum(), color="darkslategray", lw=1.1)
    axes[0].set_ylabel("cumulative P&L per $1 notional")
    axes[0].set_title("Short delta-hedged straddle, net of costs")
    dd = drawdown(sim["pnl"])
    axes[1].fill_between(dd.index, dd, 0, color="firebrick", alpha=0.6)
    axes[1].set_ylabel("drawdown")
    for name, (a, b) in EPISODES.items():
        for ax in axes:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.18)
    fig.tight_layout()
    fig.savefig(REPORTS / "equity_curve.png", dpi=140)
    plt.close(fig)

    p = sim["pnl"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(p, bins=140, color="steelblue", edgecolor="none", log=True)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(p.quantile(0.01), color="firebrick", ls="--", lw=1,
               label=f"1st percentile {p.quantile(0.01):.2%}")
    ax.set_title("Daily P&L distribution, log count. The left tail is the product")
    ax.set_xlabel("P&L per $1 notional")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "pnl_distribution.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(panel["iv"], panel["rv"], s=4, alpha=0.3, label="VIX implied")
    ax.scatter(panel["garch"], panel["rv"], s=4, alpha=0.3, color="darkorange",
               label="GARCH(1,1)")
    lim = [0, max(panel.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xlabel("forecast")
    ax.set_ylabel("realised over the next 21 days")
    ax.set_title("Forecast against outcome. Points below the line are forecasts too high")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "forecast_vs_realised.png", dpi=140)
    plt.close(fig)


def main():
    df = data.get_daily()
    print(f"SPY and VIX, {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")

    iv, rv, gap = measure(df)
    chain_check()
    panel = forecast_comparison(df, iv, rv)
    sim, m = strategy(df)
    tails(df, sim, m)
    sensitivity(df)
    charts(df, iv, rv, gap, panel, sim)
    print(f"\ncharts and tables written to {REPORTS}{os.sep}")


if __name__ == "__main__":
    main()
