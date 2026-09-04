"""Full pipeline on FRED and ETF data: the curve study since 1962, the predictive
regression, 2022, the ETF map, and the sleeve through the book's engine.

Writes tables and charts to reports/ and a copy of the console to reports/run_log.txt.

Run: ../.venv/bin/python3 run.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import carry
import data
from framework.book.strategies import ETFBeta, book_config
from framework.book.validate import attribution, gross_returns
from framework.engine import drawdown, run, sharpe

REPORTS = data.REPORTS
HOLD_BENCH = 10.0


class Tee:
    """Write stdout to the terminal and to reports/run_log.txt at the same time."""

    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def monthly_stats(r):
    """Annualised statistics of a monthly decimal return series."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    under = eq / eq.cummax() - 1
    return {"annual_return": float(eq.iloc[-1] ** (12 / len(r)) - 1), "annual_vol": float(r.std() * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)), "max_drawdown": float(under.min()),
            "trough": under.idxmin().date(), "months_held": int((r != 0).sum()), "months": len(r)}


def decade_table(realised):
    """Mean annualised excess return by tenor and decade, with a t-statistic on the monthly means."""
    dec = realised.index.year // 10 * 10
    mean = realised.groupby(dec).mean()
    t = realised.groupby(dec).apply(lambda g: g.mean() / g.std() * np.sqrt(g.notna().sum()))
    full = pd.DataFrame({"mean": realised.mean(), "t": realised.mean() / realised.std() * np.sqrt(realised.notna().sum())}).T
    out = pd.concat({"mean": mean, "t": t}, axis=1)
    out.loc["1962-2026", ("mean", slice(None))] = full.loc["mean"].to_numpy()
    out.loc["1962-2026", ("t", slice(None))] = full.loc["t"].to_numpy()
    return out


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))

    wide = data.curve_wide(data.load_curve())
    funding = data.load_funding()
    etfs = data.load_etfs()
    tenors = carry.SLEEVE_TENORS
    print(f"curve: {wide.index[0].date()} to {wide.index[-1].date()}, {len(wide)} days, "
          f"tenors {[f'{t:g}' for t in wide.columns]}")
    print(f"first print: " + ", ".join(f"{t:g}y {wide[t].first_valid_index().year}" for t in wide.columns))
    print(f"funding: DGS3MO from 1982-01-04, DTB3 before; {funding.index[0].date()} to {funding.index[-1].date()}")
    print(f"ETFs: {list(etfs.close.columns)}, {etfs.index[0].date()} to {etfs.asof.date()}")
    print(f"sleeve tenors {[f'{t:g}' for t in tenors]}, top {carry.N_LEG} by expected excess return, "
          f"target duration {carry.TARGET_DURATION} years, holding {carry.HOLD * 12:.0f} month")

    print("\n" + "=" * 78 + "\nCURVE STUDY, 1962-2026, MONTH ENDS\n" + "=" * 78)
    panel = carry.monthly_panel(wide, funding, tenors)
    rmse = panel["rmse"] * 100
    print(f"\nNelson-Siegel fit at {len(rmse)} month ends: rmse mean {rmse.mean():.1f} bp, median {rmse.median():.1f} bp, "
          f"max {rmse.max():.0f} bp ({rmse.idxmax().date()})")
    show("fit rmse by decade (bp)", rmse.groupby(rmse.index.year // 10 * 10).agg(["mean", "max"]).round(1))

    exp = panel["expected"]
    show("expected excess return by tenor, percent a year (carry + rolldown)",
         exp.describe().loc[["mean", "std", "min", "50%", "max"]].round(2))
    show("of which rolldown, percent a year", panel["rolldown"].describe().loc[["mean", "std", "min", "max"]].round(2))

    dec = decade_table(panel["realised"])
    show("TERM PREMIUM BY DECADE: realised excess return over 3m funding, percent a year, "
         "one-month holds of par bonds on the fitted curve", dec.round(2))
    dec.to_csv(os.path.join(REPORTS, "term_premium_by_decade.csv"))

    print("\n" + "=" * 78 + "\n2022\n" + "=" * 78)
    # realised on row d0 is earned over (d0, d1]; shift it to d1 so a calendar
    # year is Dec-to-Dec, not Jan-to-Jan
    earned = (panel["realised"].mul(panel["dt"], axis=0) / 100).shift(1)
    monthly22 = earned.loc["2022"]
    year22 = (1 + monthly22).prod() - 1
    going_in = exp.loc["2021-12-31"]
    table22 = pd.DataFrame({"expected at end 2021 (%/yr)": going_in, "realised 2022 (%)": year22 * 100,
                            "duration at end 2021": panel["duration"].loc["2021-12-31"]}).round(2)
    table22.index = [f"{t:g}y" for t in table22.index]
    show("what carry plus rolldown said going in, and what a one-month-rolled par bond then did", table22)
    table22.to_csv(os.path.join(REPORTS, "year_2022.csv"))
    m22 = monthly22.copy()
    m22.index = m22.index.strftime("%Y-%m")
    m22.columns = [f"{t:g}y" for t in m22.columns]
    show("2022 month by month, excess return in percent", (m22 * 100).round(1))
    yr = earned.groupby(earned.index.year).apply(lambda g: (1 + g).prod(min_count=1) - 1)
    show("worst calendar years for the 10y and 30y since 1962, excess return in percent",
         pd.DataFrame({"10y": yr[10.0].nsmallest(5), "30y": yr[30.0].reindex(yr[10.0].nsmallest(5).index)}).mul(100).round(1))

    print("\n" + "=" * 78 + "\nDOES CARRY PLUS ROLLDOWN PREDICT THE REALISED EXCESS RETURN?\n" + "=" * 78)
    reg = carry.predictive_regression(exp, panel["realised"])
    show("realised (%/yr, next month) on expected (%/yr), Newey-West t per tenor, pooled clustered by month",
         reg.round(3))
    reg.to_csv(os.path.join(REPORTS, "predictive_regression.csv"))
    reg_c = carry.predictive_regression(panel["carry"], panel["realised"])
    reg_r = carry.predictive_regression(panel["rolldown"], panel["realised"])
    show("the two pieces separately, pooled rows",
         pd.DataFrame({"carry alone": reg_c.loc["pooled"], "rolldown alone": reg_r.loc["pooled"]}).T.round(3))
    halves = {}
    for name, sub in (("1962-1994", exp.index < "1995"), ("1995-2026", exp.index >= "1995")):
        halves[name] = carry.predictive_regression(exp[sub], panel["realised"][sub]).loc["pooled"]
    show("pooled, by half of the sample", pd.DataFrame(halves).T.round(3))

    print("\n" + "=" * 78 + "\nTHE RULE ON THE CURVE ITSELF, NO COSTS, 1962-2026\n" + "=" * 78)
    r_long, w_long = carry.sleeve_returns(panel)
    r_ls, w_ls = carry.sleeve_returns(panel, long_short=True)
    bench_w = pd.DataFrame(0.0, index=panel["duration"].index, columns=panel["duration"].columns)
    bench_w[HOLD_BENCH] = carry.TARGET_DURATION / panel["duration"][HOLD_BENCH]
    r_bench = (bench_w * panel["realised"]).sum(axis=1, min_count=1) / 100 * panel["dt"]
    study = pd.DataFrame({
        f"sleeve: top {carry.N_LEG} long only": monthly_stats(r_long),
        f"top {carry.N_LEG} vs bottom {carry.N_LEG}, duration neutral": monthly_stats(r_ls),
        f"always long {HOLD_BENCH:g}y at the same duration": monthly_stats(r_bench),
    }).T
    show("monthly excess returns of a book carrying 7 years of duration", study.round(3))
    study.to_csv(os.path.join(REPORTS, "study_performance.csv"))
    flat_months = (w_long.abs().sum(axis=1) == 0).mean()
    print(f"\nthe long-only rule is flat in {flat_months:.0%} of months (every expected excess return negative)")
    by_dec = pd.DataFrame({"sleeve": r_long, "long/short": r_ls, f"long {HOLD_BENCH:g}y": r_bench})
    by_dec = by_dec.groupby(by_dec.index.year // 10 * 10).apply(lambda g: g.mean() / g.std() * np.sqrt(12))
    show("Sharpe by decade", by_dec.round(2))
    held = w_long.gt(0).mean().rename("share of months held")
    held.index = [f"{t:g}y" for t in held.index]
    show("what the long-only rule holds", held.map("{:.0%}".format).to_frame())
    exp_signal = pd.DataFrame({"sleeve": r_long, f"long {HOLD_BENCH:g}y": r_bench}).dropna()
    fit = np.polyfit(exp_signal.iloc[:, 1], exp_signal.iloc[:, 0], 1)
    print(f"\nsleeve on the 10y benchmark: beta {fit[0]:.2f}, alpha {fit[1] * 12:+.2%} a year")

    print("\n" + "=" * 78 + "\nETF MAP\n" + "=" * 78)
    rows = {}
    for etf, T in data.ETF_TENOR.items():
        if etf not in etfs.close.columns:
            continue
        D, n = carry.empirical_duration(etfs.close[etf].pct_change(), wide[T].reindex(etfs.calendar))
        par_now = float(carry.par_duration(wide[T].dropna().iloc[-1], T))
        rows[etf] = {"tenor": f"{T:g}y", "fact sheet duration": data.ETF_DURATION[etf],
                     "empirical duration": round(D, 2), "days": n,
                     "par duration at today's yield": round(par_now, 2),
                     "par duration at 4%": round(float(carry.par_duration(4.0, T)), 2)}
    etf_table = pd.DataFrame(rows).T
    show("fund duration against the tenor it stands in for (empirical = minus the slope of daily fund return on "
         "daily yield change)", etf_table)
    etf_table.to_csv(os.path.join(REPORTS, "etf_durations.csv"))

    print("\n" + "=" * 78 + "\nTHROUGH THE BOOK'S ENGINE, 2002-2026\n" + "=" * 78)
    print("shadow cost model: 5 bps commission, 2 bps half spread, sqrt impact, 50 bps borrow; overlay 10% vol "
          "target, 3x gross cap, half size past 15% drawdown, kill off, 10% position buffer; next-open fills; "
          "cash earns nothing")
    bars = data.build_bars(etfs, wide, funding)
    runs = {
        "RatesCarry (sleeve)": run(carry.RatesCarryETF(), bars, config=book_config(kill=False)),
        "RatesCarry long/short": run(carry.RatesCarryETF(long_short=True), bars, config=book_config(kill=False)),
        "hold IEF+TLT 1/N": run(ETFBeta(["IEF", "TLT"]), bars, config=book_config(kill=False)),
        "hold all five 1/N": run(ETFBeta(list(etfs.close.columns)), bars, config=book_config(kill=False)),
        "RatesCarry, 2x costs": run(carry.RatesCarryETF(), bars, config=book_config(kill=False, cost_scale=2.0)),
        "RatesCarry, 4x costs": run(carry.RatesCarryETF(), bars, config=book_config(kill=False, cost_scale=4.0)),
    }
    perf = {}
    for name, res in runs.items():
        r = res.returns[res.returns != 0]
        g = gross_returns(res)[res.returns != 0]
        m = res.metrics
        y22 = res.returns.loc["2022"]
        perf[name] = {"sharpe gross": sharpe(g), "sharpe net": sharpe(r), "annual return": m["annual_return"],
                      "annual vol": m["annual_vol"], "max drawdown": m["max_drawdown"],
                      "trough": m["dd_trough"].date(), "2022 return": float((1 + y22).prod() - 1),
                      "2022 drawdown": drawdown(y22)[1]["max_drawdown"],
                      "turnover/yr": m["annual_turnover"], "trades": len(res.trades),
                      "live from": r.index[0].date()}
    perf = pd.DataFrame(perf).T
    show("net of the shadow cost model, daily returns, days before live excluded", perf.round(3))
    perf.to_csv(os.path.join(REPORTS, "engine_performance.csv"))

    sleeve = runs["RatesCarry (sleeve)"]
    years = pd.DataFrame({"sleeve": sleeve.by_year()["return"], "hold IEF+TLT": runs["hold IEF+TLT 1/N"].by_year()["return"],
                          "sleeve sharpe": sleeve.by_year()["sharpe"],
                          "sleeve max dd": sleeve.by_year()["max_drawdown"]})
    show("by calendar year", years.round(3))
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))

    attr = {}
    for label, bench in (("IEF+TLT", ["IEF", "TLT"]), ("all five funds", list(etfs.close.columns))):
        for b, s in attribution(sleeve.returns, bars, bench).items():
            attr[f"{label}, {b}"] = s
    attr = pd.DataFrame(attr).T
    show("ATTRIBUTION: sleeve net returns on owning its benchmark (framework/book/validate.py::attribution)",
         attr.round(3))
    attr.to_csv(os.path.join(REPORTS, "attribution.csv"))
    sleeve.report(os.path.join(REPORTS, "engine"))

    charts(panel, dec, reg, r_long, r_ls, r_bench, w_long, runs, exp)
    print(f"\ntables and charts written to {REPORTS}/ in {time.time() - t0:.0f}s")


def charts(panel, dec, reg, r_long, r_ls, r_bench, w_long, runs, exp):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    decades = [d for d in dec.index if d != "1962-2026"]
    width = 0.16
    for i, T in enumerate(carry.SLEEVE_TENORS):
        ax.bar(np.arange(len(decades)) + (i - 2) * width, dec.loc[decades, ("mean", T)], width, label=f"{T:g}y")
    ax.set_xticks(np.arange(len(decades)))
    ax.set_xticklabels([f"{d}s" for d in decades])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("excess return, % a year")
    ax.set_title("Term premium by decade: realised excess return of one-month-rolled par bonds over 3m funding")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "term_premium_by_decade.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, T in zip(axes, (10.0, 30.0)):
        d = pd.DataFrame({"x": exp[T], "y": panel["realised"][T]}).dropna()
        ax.scatter(d["x"], d["y"], s=6, alpha=0.5)
        row = reg.loc[f"{T:g}y"]
        xs = np.linspace(d["x"].min(), d["x"].max(), 50)
        ax.plot(xs, row["alpha"] + row["slope"] * xs, color="firebrick",
                label=f"slope {row['slope']:.2f} (t {row['t']:.1f}), R2 {row['r2']:.3f}")
        ax.plot(xs, xs, color="grey", linestyle="--", linewidth=0.8, label="slope 1")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("expected excess return, % a year (carry + rolldown)")
        ax.set_ylabel("realised next month, % a year")
        ax.set_title(f"{T:g}y, {len(d)} months")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "predictive_regression.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for name, r in (("sleeve, top 2 long only", r_long), ("top 2 vs bottom 2, duration neutral", r_ls),
                    ("always long 10y, same duration", r_bench)):
        axes[0].plot((1 + r.fillna(0)).cumprod(), linewidth=1.1, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of 1, excess of funding (log)")
    axes[0].set_title("The rule on the curve itself, 7 years of duration, no costs, 1962-2026")
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    axes[0].yaxis.set_minor_formatter(lambda v, _: "")
    axes[1].stackplot(w_long.index, [(w_long[T] * panel["duration"][T]).fillna(0.0) for T in w_long.columns],
                      labels=[f"{T:g}y" for T in w_long.columns], alpha=0.8)
    axes[1].set_ylabel("duration held, years")
    axes[1].legend(fontsize=7, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "study_equity.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for name in ("RatesCarry (sleeve)", "RatesCarry long/short", "hold IEF+TLT 1/N"):
        res = runs[name]
        axes[0].plot(res.equity / res.config.capital, linewidth=1.1, label=name)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    axes[0].yaxis.set_minor_formatter(lambda v, _: "")
    axes[0].set_ylabel("growth of 1 (log)")
    axes[0].set_title("Through the book's engine, net of the shadow cost model, 10% vol target")
    axes[0].legend(fontsize=8)
    under, _ = drawdown(runs["RatesCarry (sleeve)"].returns)
    axes[1].fill_between(under.index, under, 0, color="firebrick", alpha=0.5)
    axes[1].set_ylabel("sleeve drawdown")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "engine_equity.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
