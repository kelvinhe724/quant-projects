"""Full pipeline on real returns: rolling out-of-sample comparison, costs, break-even window.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from backtest import COST_BPS, MONTHS, metrics, run, sharpe_diff_pvalue
from portfolio import RULES, equal_weight, estimate, tangency

REPORTS = data.REPORTS
WINDOWS = (60, 120)
GRID = (24, 36, 48, 60, 84, 120, 150, 180, 240)
SIM_GRID = (24, 36, 48, 60, 72, 84, 90, 96, 108, 120, 180, 240, 360, 480, 720, 1200, 2400, 6000)
SIM_DRAWS = 2000
COST_GRID = (0.0, 10.0, 25.0, 50.0)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "font.size": 9})


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def fmt(m, g, ruined=False):
    return pd.Series({
        "excess return (gross)": f"{g['mean_excess']:+.2%}",
        "excess return (net)": f"{m['mean_excess']:+.2%}",
        "vol": f"{m['vol']:.2%}",
        "Sharpe (gross)": f"{g['sharpe']:.2f}",
        "Sharpe (net)": f"{m['sharpe']:.2f}",
        "max drawdown (net)": "ruin" if ruined else f"{m['max_drawdown']:.1%}",
        "monthly turnover": f"{m['turnover']:.3f}",
        "avg gross exposure": f"{m['gross_exposure']:.2f}",
        "largest |weight|": f"{m['max_weight']:.2f}",
    })


def true_sharpe(w, mu, sigma):
    return (w @ mu) / np.sqrt(w @ sigma @ w) * np.sqrt(MONTHS)


def simulated_break_even(mu, sigma, rng):
    """Expected true Sharpe of plug-in tangency weights against the window length, moments held fixed."""
    n = len(mu)
    rows = []
    for m in SIM_GRID:
        draws = rng.multivariate_normal(mu, sigma, size=(SIM_DRAWS, m))
        sr = [true_sharpe(tangency(*estimate(d)), mu, sigma) for d in draws]
        rows.append({"window": m, "expected_sharpe": np.mean(sr), "median_sharpe": np.median(sr),
                     "share_beating_1/N": np.mean(np.array(sr) > true_sharpe(equal_weight(n), mu, sigma))})
    return pd.DataFrame(rows).set_index("window")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    rets, rf, excess = data.monthly_panel()
    print(f"{excess.shape[1]} assets, {len(excess)} months, "
          f"{excess.index[0].date()} to {excess.index[-1].date()}, costs {COST_BPS:.0f}bps one-way")

    mu_full, sigma_full = estimate(excess)
    n = len(mu_full)
    s_tan = true_sharpe(tangency(mu_full, sigma_full), mu_full, sigma_full)
    s_ew = true_sharpe(equal_weight(n), mu_full, sigma_full)
    print(f"\nfull-sample (in-sample, look-ahead) Sharpe: tangency {s_tan:.2f}, 1/N {s_ew:.2f}. "
          f"That gap is the prize an estimator has to capture out of sample.")

    books = {}
    for window in WINDOWS:
        table, pvals = {}, {}
        for rule in RULES:
            book, path = run(rule, excess, window, total=rets)
            books[(rule, window)] = book
            # a monthly return below -100% is bankruptcy, after which a drawdown figure means nothing
            table[rule] = fmt(metrics(book), metrics(book, "gross"), ruined=book["net"].min() < -1)
            if rule != "1/N":
                pvals[rule] = sharpe_diff_pvalue(book["net"], books[("1/N", window)]["net"])
            path.to_csv(os.path.join(REPORTS, f"weights_{rule.replace(' ', '_').replace('/', '')}_{window}.csv"))
        table = pd.DataFrame(table)
        table.loc["p-value vs 1/N (net Sharpe)"] = pd.Series(pvals).map("{:.2f}".format).reindex(table.columns).fillna("")
        first = books[("1/N", window)].index[0].date()
        show(f"OUT OF SAMPLE, {window}-month rolling window, {first} onward "
             f"({len(books[('1/N', window)])} months)", table)
        table.to_csv(os.path.join(REPORTS, f"oos_{window}.csv"))

    rows = {}
    for bps in COST_GRID:
        rows[f"{bps:.0f}bps"] = {rule: metrics(run(rule, excess, 60, cost_bps=bps, total=rets)[0])["sharpe"]
                                 for rule in RULES}
    cost_table = pd.DataFrame(rows).round(2)
    show("NET SHARPE BY ONE-WAY COST, 60-month window", cost_table)
    cost_table.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))

    grid_rules = ["tangency", "tangency long-only", "min variance", "shrunk tangency"]
    rows = []
    for m in GRID:
        base = run("1/N", excess, m, total=rets)[0]
        row = {"window": m, "months": len(base), "1/N": metrics(base)["sharpe"]}
        for rule in grid_rules:
            row[rule] = metrics(run(rule, excess, m, total=rets)[0])["sharpe"]
        rows.append(row)
    empirical = pd.DataFrame(rows).set_index("window")
    show("EMPIRICAL BREAK-EVEN: net OOS Sharpe by estimation window, each row on its own OOS period",
         empirical.round(2))
    empirical.to_csv(os.path.join(REPORTS, "break_even_empirical.csv"))

    rng = np.random.default_rng(0)
    sim = simulated_break_even(mu_full, sigma_full, rng)
    sim["1/N"] = s_ew
    show(f"SIMULATED BREAK-EVEN: true Sharpe of plug-in tangency, moments fixed at full-sample "
         f"values (N={n}, {SIM_DRAWS} draws per window)", sim.round(3))
    sim.to_csv(os.path.join(REPORTS, "break_even_simulated.csv"))
    crossing = sim.index[sim["expected_sharpe"] > s_ew]
    be = int(crossing[0]) if len(crossing) else None
    print(f"\nbreak-even window (expected Sharpe of sample tangency first exceeds 1/N): "
          f"{be if be else 'beyond ' + str(SIM_GRID[-1])} months, i.e. "
          f"{be / 12 if be else SIM_GRID[-1] / 12:.1f}{'' if be else '+'} years of monthly data")

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, rule in zip(COLORS, RULES):
        r = books[(rule, 60)]["net"]
        if rule == "tangency":
            continue
        ax.plot((1 + r).cumprod(), color=color, lw=1.6, label=rule)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("Growth of $1 in excess of T-bills, 60-month window, net of 10bps\n"
                 "(unconstrained tangency omitted: it goes bankrupt)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, rule in zip(COLORS, ["1/N"] + grid_rules):
        ax.plot(empirical.index, empirical[rule], color=color, lw=1.6, marker="o", ms=4, label=rule)
    ax.set_ylim(-1, 1.5)
    ax.set_xlabel("estimation window (months)")
    ax.set_ylabel("net OOS Sharpe")
    ax.set_title("Real data: OOS Sharpe by estimation window (each window on its own OOS period)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "break_even_empirical.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sim.index, sim["expected_sharpe"], color=COLORS[1], lw=1.6, marker="o", ms=4,
            label="sample tangency, expected true Sharpe")
    ax.axhline(s_ew, color=COLORS[0], lw=1.6, label=f"1/N true Sharpe {s_ew:.2f}")
    ax.axhline(s_tan, color="#898781", lw=1, ls="--", label=f"true tangency Sharpe {s_tan:.2f}")
    if be:
        ax.axvline(be, color="#898781", lw=0.8, ls=":")
        ax.annotate(f"break-even {be} months", (be, s_ew), xytext=(be * 1.3, s_ew - 0.35),
                    color="#52514e", arrowprops={"arrowstyle": "-", "color": "#898781"})
    ax.set_xscale("log")
    ax.set_xlabel("estimation window (months, log scale)")
    ax.set_ylabel("annualised Sharpe on the true moments")
    ax.set_title("Simulation: how much history the sample tangency portfolio needs to beat 1/N")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "break_even_simulated.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(9, 4))
    for color, rule in zip(COLORS[1:], ["tangency", "min variance", "shrunk tangency", "shrunk min variance"]):
        ax.plot(books[(rule, 60)]["gross_exposure"], color=color, lw=1.4, label=rule)
    ax.set_yscale("log")
    ax.set_ylabel("gross exposure (sum of |weights|, log)")
    ax.set_title("How far the optimisers stray from fully invested, 60-month window")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "gross_exposure.png"), dpi=140)
    print(f"\ncharts and tables written to {REPORTS}")


if __name__ == "__main__":
    main()
