"""Full pipeline on real prices: screen, tune on formation, trade out of sample.

Run: python3 run.py
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from backtest import DEFAULT_COST_BPS, TRADING_DAYS, evaluate, metrics, trades
from pairs import bonferroni, screen, select

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
FORMATION = (data.FORMATION_START, data.FORMATION_END)
OOS = (data.OOS_START, data.OOS_END)

GRID = {
    "pvalue_max": None,
    "lookback": [30, 60, 90],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_z": [0.0, 0.25, 0.5],
}
STOP_Z = 4.0
MAX_HOLD = 60
MAX_PAIRS = 40


def show(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(rows.to_string())


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


def main():
    px, log_px, sectors = data.get_prices()
    form_log, _ = data.split(log_px)
    print(f"universe: {len(px.columns)} names, {px.index[0].date()} to {px.index[-1].date()}")
    print(f"formation {FORMATION[0]} to {FORMATION[1]}  |  out-of-sample {OOS[0]} to {OOS[1]}")

    screened = screen(form_log, sectors)
    n_tests = len(screened)
    n_all_pairs = len(px.columns) * (len(px.columns) - 1) // 2
    bonf = bonferroni(n_tests)

    counts = pd.DataFrame({
        "threshold": ["p < 0.10", "p < 0.05", "p < 0.01",
                      f"Bonferroni p < {bonf:.2e}"],
        "pairs passing": [int((screened.pvalue < t).sum())
                          for t in (0.10, 0.05, 0.01, bonf)],
        "expected by chance": [f"{t * n_tests:.1f}"
                               for t in (0.10, 0.05, 0.01, bonf)],
    })
    print(f"\nwithin-sector pairs tested: {n_tests} (all-pairs would be {n_all_pairs}; "
          f"the full 500-name index would be {500 * 499 // 2})")
    show("Cointegration screening vs chance", counts)
    screened.head(20).to_csv(os.path.join(REPORTS, "top_pairs.csv"), index=False)

    GRID["pvalue_max"] = [0.01, 0.05, bonf]
    keys = list(GRID)
    results = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        chosen = select(screened, cfg["pvalue_max"], max_pairs=MAX_PAIRS)
        if len(chosen) < 5:
            continue
        rules = dict(lookback=cfg["lookback"], entry_z=cfg["entry_z"],
                     exit_z=cfg["exit_z"], stop_z=STOP_Z, max_hold=MAX_HOLD,
                     cost_bps=DEFAULT_COST_BPS)
        m, _, _ = evaluate(log_px, chosen, window=FORMATION, **rules)
        results.append({**cfg, "n_pairs": len(chosen), "sharpe": m["sharpe"],
                        "annual_return": m["annual_return"]})

    grid = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    grid.to_csv(os.path.join(REPORTS, "parameter_grid.csv"), index=False)
    show(f"Formation parameter search, top 10 of {len(grid)} configurations "
         "(net of costs)", grid.head(10).round(3))

    best = grid.iloc[0]
    rules = dict(lookback=int(best.lookback), entry_z=float(best.entry_z),
                 exit_z=float(best.exit_z), stop_z=STOP_Z, max_hold=MAX_HOLD,
                 cost_bps=DEFAULT_COST_BPS)
    chosen = select(screened, float(best.pvalue_max), max_pairs=MAX_PAIRS)
    print(f"\nchosen: p <= {best.pvalue_max:.2e}, lookback {rules['lookback']}, "
          f"entry {rules['entry_z']}, exit {rules['exit_z']}, "
          f"stop {STOP_Z}, max hold {MAX_HOLD}d, {len(chosen)} pairs, "
          f"{DEFAULT_COST_BPS:.0f}bps round trip")
    chosen.to_csv(os.path.join(REPORTS, "selected_pairs.csv"), index=False)

    in_m, in_port, _ = evaluate(log_px, chosen, window=FORMATION, **rules)
    out_m, out_port, out_legs = evaluate(log_px, chosen, window=OOS, **rules)
    table = pd.DataFrame({"in-sample (formation)": fmt(in_m),
                          "out-of-sample": fmt(out_m)})
    show("Performance", table)
    table.to_csv(os.path.join(REPORTS, "performance.csv"))

    by_year = out_port["net"].groupby(out_port.index.year).agg(
        **{"net return": lambda r: f"{(1 + r).prod() - 1:+.2%}",
           "Sharpe": lambda r: f"{r.mean() / r.std() * np.sqrt(TRADING_DAYS):.2f}"})
    show("Out-of-sample by year", by_year)

    costs = pd.DataFrame([
        {"round trip bps": c,
         "in-sample Sharpe": round(evaluate(log_px, chosen, window=FORMATION,
                                            **{**rules, "cost_bps": c})[0]["sharpe"], 2),
         "out-of-sample Sharpe": round(evaluate(log_px, chosen, window=OOS,
                                                **{**rules, "cost_bps": c})[0]["sharpe"], 2)}
        for c in (0, 5, 10, 20, 30)]).set_index("round trip bps")
    show("Sensitivity to the transaction cost assumption", costs)
    costs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))

    sens = grid[(grid.pvalue_max == best.pvalue_max) &
                (grid.lookback == best.lookback)].pivot(
        index="entry_z", columns="exit_z", values="sharpe").round(2)
    show("Formation Sharpe by entry / exit threshold (lookback "
         f"{rules['lookback']}, p <= {best.pvalue_max:.2e})", sens)

    per_pair = pd.DataFrame([
        {"pair": f"{a}/{b}", **metrics(v["net"], trades(v))}
        for (a, b), v in out_legs.items()]).set_index("pair")
    per_pair.to_csv(os.path.join(REPORTS, "per_pair_oos.csv"))
    print(f"\nout-of-sample per-pair Sharpe: median {per_pair.sharpe.median():.2f}, "
          f"{(per_pair.sharpe > 0).sum()} of {len(per_pair)} positive")

    charts(in_port, out_port, out_legs, per_pair, log_px, chosen, rules)
    print(f"\ncharts and tables written to {REPORTS}/")


def charts(in_port, out_port, out_legs, per_pair, log_px, chosen, rules):
    from pairs import positions, spread, zscore

    fig, ax = plt.subplots(figsize=(10, 5))
    for window, port in (("formation", in_port), ("out-of-sample", out_port)):
        for col, style in (("gross", "--"), ("net", "-")):
            ax.plot((1 + port[col]).cumprod(), style, linewidth=1.2,
                    label=f"{window} {col}")
    ax.axvline(pd.Timestamp(data.OOS_START), color="black", linewidth=0.8)
    ax.set_title("Equity curve, gross and net of costs")
    ax.set_ylabel("growth of 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    equity = (1 + out_port["net"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dd = equity / equity.cummax() - 1
    ax.fill_between(dd.index, dd, 0, color="firebrick", alpha=0.5)
    ax.set_title("Out-of-sample drawdown, net of costs")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "drawdown.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_pair.sharpe.dropna(), bins=15, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Out-of-sample Sharpe by pair")
    ax.set_xlabel("Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "pair_sharpes.png"), dpi=140)
    plt.close(fig)

    p = chosen.iloc[0]
    sp = spread(log_px[p.a], log_px[p.b], p.beta, p.intercept)
    z = zscore(sp, rules["lookback"])
    pos = positions(z, rules["entry_z"], rules["exit_z"], STOP_Z, MAX_HOLD).shift(1)
    entries = z.index[(pos != 0) & (pos.shift(1).fillna(0) == 0)]
    exits = z.index[(pos == 0) & (pos.shift(1).fillna(0) != 0)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(np.exp(log_px[p.a]), label=p.a, linewidth=1)
    axes[0].plot(np.exp(log_px[p.b]), label=p.b, linewidth=1)
    axes[0].set_ylabel("price")
    axes[0].legend(fontsize=8)
    axes[1].plot(sp, color="darkslategray", linewidth=1)
    axes[1].set_ylabel("spread")
    axes[2].plot(z, color="steelblue", linewidth=0.8)
    for level in (rules["entry_z"], -rules["entry_z"]):
        axes[2].axhline(level, color="grey", linestyle=":", linewidth=0.8)
    axes[2].scatter(entries, z[entries], marker="^", color="green", s=18, label="entry")
    axes[2].scatter(exits, z[exits], marker="v", color="red", s=18, label="exit")
    axes[2].set_ylabel("z-score")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(pd.Timestamp(data.OOS_START), color="black", linewidth=0.8)
    axes[0].set_title(f"{p.a} / {p.b}: prices, spread and trailing z-score "
                      f"(p = {p.pvalue:.4f}, beta = {p.beta:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "sample_pair.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
