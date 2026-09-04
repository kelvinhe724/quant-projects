"""Static versus Kalman hedge ratios on the pairs project's 35 pairs and windows.

Run: python3 run.py
Writes tables and charts to reports/ and a copy of the console to reports/run_log.txt.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from kalman import kalman_spread
from strategy import COST_BPS, NOISE_GRID, TRADING_DAYS, evaluate, metrics, trades, tune_noise

REPORTS = data.REPORTS
FORMATION = (data.FORMATION_START, data.FORMATION_END)
OOS = (data.OOS_START, data.OOS_END)


class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.file.write(s)
        self.stdout.write(s)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def fmt(m):
    return pd.Series({
        "annual return (gross)": f"{m['gross_annual_return']:+.2%}",
        "annual return (net)": f"{m['annual_return']:+.2%}",
        "annual vol": f"{m['annual_vol']:.2%}",
        "Sharpe (gross)": f"{m['gross_sharpe']:.2f}",
        "Sharpe (net)": f"{m['sharpe']:.2f}",
        "max drawdown": f"{m['max_drawdown']:.2%}",
        "trades": f"{m['n_trades']:.0f}",
        "hit rate": f"{m['hit_rate']:.1%}",
        "avg hold (days)": f"{m['avg_hold_days']:.0f}",
        "annual turnover": f"{m['annual_turnover']:.1f}x",
    })


def by_year(port):
    return port["net"].groupby(port.index.year).agg(
        **{"net return": lambda r: f"{(1 + r).prod() - 1:+.2%}",
           "Sharpe": lambda r: f"{r.mean() / r.std() * np.sqrt(TRADING_DAYS):.2f}"})


def main():
    pairs, px, log_px = data.get_prices()
    print(f"{len(pairs)} pairs from ../pairs-trading, {len(px.columns)} names, "
          f"{px.index[0].date()} to {px.index[-1].date()}")
    print(f"formation {FORMATION[0]} to {FORMATION[1]}  |  out-of-sample {OOS[0]} to {OOS[1]}")
    print(f"rules fixed from the pairs project; {COST_BPS:.0f}bps round trip")

    q_star, grid = tune_noise(log_px, pairs, FORMATION)
    print(f"\nstate-noise ratio chosen on formation Sharpe: {q_star:g}")

    static = {w: evaluate(log_px, pairs, win, FORMATION[1]) for w, win in
              (("formation", FORMATION), ("oos", OOS))}
    kalman = {w: evaluate(log_px, pairs, win, FORMATION[1], q_star) for w, win in
              (("formation", FORMATION), ("oos", OOS))}

    table = pd.DataFrame({
        "static, formation": fmt(static["formation"][0]),
        "static, out-of-sample": fmt(static["oos"][0]),
        f"Kalman q={q_star:g}, formation": fmt(kalman["formation"][0]),
        f"Kalman q={q_star:g}, out-of-sample": fmt(kalman["oos"][0]),
    })
    show("Performance, same 35 pairs, same rules, same windows", table)
    table.to_csv(os.path.join(REPORTS, "performance.csv"))

    # Every ratio is scored out of sample here for the reader; only the formation
    # column was used to pick q_star.
    rows = []
    for q in NOISE_GRID:
        m_o, port_o, _ = evaluate(log_px, pairs, OOS, FORMATION[1], q)
        rows.append({"noise_ratio": f"{q:g}",
                     "formation Sharpe": grid.set_index("noise_ratio").loc[q, "formation_sharpe"],
                     "oos Sharpe (net)": m_o["sharpe"],
                     "oos Sharpe (gross)": m_o["gross_sharpe"],
                     "oos net return/yr": m_o["annual_return"],
                     "oos turnover/yr": m_o["annual_turnover"],
                     "oos trades": m_o["n_trades"]})
    sens = pd.DataFrame(rows).set_index("noise_ratio")
    sens.loc["static"] = [static["formation"][0]["sharpe"], static["oos"][0]["sharpe"],
                          static["oos"][0]["gross_sharpe"], static["oos"][0]["annual_return"],
                          static["oos"][0]["annual_turnover"], static["oos"][0]["n_trades"]]
    show("State-noise sensitivity (q chosen on the formation column only)", sens.round(3))
    sens.to_csv(os.path.join(REPORTS, "noise_sensitivity.csv"))

    years = pd.concat({"static": by_year(static["oos"][1]),
                       "Kalman": by_year(kalman["oos"][1])}, axis=1)
    show("Out-of-sample by year", years)
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))

    costs = pd.DataFrame([
        {"round trip bps": c,
         "static oos Sharpe": round(evaluate(log_px, pairs, OOS, FORMATION[1], None, c)[0]["sharpe"], 2),
         "Kalman oos Sharpe": round(evaluate(log_px, pairs, OOS, FORMATION[1], q_star, c)[0]["sharpe"], 2)}
        for c in (0, 5, 10, 20, 30)]).set_index("round trip bps")
    show("Sensitivity to the transaction cost assumption", costs)
    costs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))

    per_pair = pd.DataFrame({
        "static": {f"{a}/{b}": metrics(v["net"], trades(v))["sharpe"]
                   for (a, b), v in static["oos"][2].items()},
        "Kalman": {f"{a}/{b}": metrics(v["net"], trades(v))["sharpe"]
                   for (a, b), v in kalman["oos"][2].items()},
    })
    per_pair["diff"] = per_pair["Kalman"] - per_pair["static"]
    per_pair.to_csv(os.path.join(REPORTS, "per_pair_oos.csv"))
    print(f"\nout-of-sample per-pair net Sharpe: static median {per_pair['static'].median():.2f} "
          f"({(per_pair['static'] > 0).sum()} of {len(per_pair)} positive), "
          f"Kalman median {per_pair['Kalman'].median():.2f} "
          f"({(per_pair['Kalman'] > 0).sum()} positive); Kalman beats static on "
          f"{(per_pair['diff'] > 0).sum()} pairs")

    drift = pd.DataFrame([
        {"pair": f"{p.a}/{p.b}", "static beta": p.beta,
         "Kalman beta 2019-12-31": kalman["formation"][2][(p.a, p.b)]["beta"].iloc[-1],
         "Kalman beta 2026-08": kalman["oos"][2][(p.a, p.b)]["beta"].iloc[-1]}
        for p in pairs.itertuples()]).set_index("pair")
    drift["oos drift"] = drift["Kalman beta 2026-08"] - drift["Kalman beta 2019-12-31"]
    drift.to_csv(os.path.join(REPORTS, "beta_drift.csv"))
    print(f"median |beta change| over the out-of-sample window: "
          f"{drift['oos drift'].abs().median():.2f}; "
          f"largest {drift['oos drift'].abs().idxmax()} at {drift['oos drift'].abs().max():+.2f}")

    charts(static, kalman, sens, per_pair, log_px, pairs, q_star)
    print(f"\ncharts and tables written to {REPORTS}/")


def charts(static, kalman, sens, per_pair, log_px, pairs, q_star):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res, color in (("static", static, "steelblue"), ("Kalman", kalman, "darkorange")):
        for w, style in (("formation", "--"), ("oos", "-")):
            ax.plot((1 + res[w][1]["net"]).cumprod(), style, color=color, linewidth=1.2,
                    label=f"{name} {w}, net")
    ax.axvline(pd.Timestamp(data.OOS_START), color="black", linewidth=0.8)
    ax.set_title(f"Equity curves, net of {COST_BPS:.0f}bps, static vs Kalman q={q_star:g}")
    ax.set_ylabel("growth of 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    s = sens.drop(index="static")
    xs = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, s["formation Sharpe"], "o-", label="formation (used to choose q)")
    ax.plot(xs, s["oos Sharpe (net)"], "s-", label="out-of-sample (not used)")
    ax.axhline(sens.loc["static", "formation Sharpe"], color="steelblue", linestyle=":",
               label="static, formation")
    ax.axhline(sens.loc["static", "oos Sharpe (net)"], color="darkorange", linestyle=":",
               label="static, out-of-sample")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(s.index))
    ax.set_xlabel("state noise / observation noise")
    ax.set_ylabel("net Sharpe")
    ax.set_title("State-noise sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "noise_sensitivity.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(per_pair["static"], per_pair["Kalman"], s=18, color="steelblue")
    lim = per_pair[["static", "Kalman"]].abs().max().max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("static, out-of-sample Sharpe")
    ax.set_ylabel("Kalman, out-of-sample Sharpe")
    ax.set_title("Per-pair out-of-sample Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "pair_sharpes.png"), dpi=140)
    plt.close(fig)

    p = pairs.iloc[0]
    form = log_px.loc[:data.FORMATION_END]
    obs_var = float((form[p.a] - p.beta * form[p.b] - p.intercept).var())
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for q, color in ((1e-3, "lightgrey"), (q_star, "darkorange")):
        sp, beta, _ = kalman_spread(log_px[p.a], log_px[p.b], q, obs_var)
        axes[0].plot(beta, color=color, linewidth=1, label=f"Kalman q={q:g}")
        if q == q_star:
            axes[2].plot(sp.iloc[60:], color=color, linewidth=0.8, label="Kalman innovation")
    axes[0].axhline(p.beta, color="steelblue", linestyle="--", label="static (formation OLS)")
    axes[0].set_ylabel("hedge ratio")
    axes[0].set_ylim(p.beta - 1.0, p.beta + 1.0)
    axes[0].legend(fontsize=8)
    axes[1].plot(log_px[p.a] - p.beta * log_px[p.b] - p.intercept, color="steelblue",
                 linewidth=0.8, label="static spread")
    axes[1].set_ylabel("static spread")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel("Kalman spread")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(pd.Timestamp(data.OOS_START), color="black", linewidth=0.8)
    axes[0].set_title(f"{p.a} / {p.b}: hedge ratio paths and the two spreads")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "sample_pair.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))
    main()
