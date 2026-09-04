"""Run every strategy over many simulated sessions, print the tables, write the charts.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sim
from maker import AvellanedaStoikov, Naive, Skewed

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
SESSIONS = 1000
SWEEP_SESSIONS = 400
SEED = 1
SESSIONS_PER_YEAR = 252

BASE = sim.Market(informed_frac=0.30, informed_lag=100, position_limit=20)
JUMPY = sim.Market(informed_frac=0.30, informed_lag=100, position_limit=20,
                   jump_rate=4.0, jump_sd=0.015)

COLOURS = {"naive": "#c44e52", "skewed": "#4c72b0", "avellaneda-stoikov": "#55a868"}


def makers():
    return [Naive(), Skewed(), AvellanedaStoikov()]


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def frame(results):
    """Stack the per-session summary dicts into a DataFrame."""
    return pd.DataFrame(results)


def metrics(df):
    """Reduce one strategy's session-level results to the headline row."""
    pnl = df["pnl"]
    return {
        "mean P&L": pnl.mean(),
        "P&L sd": pnl.std(),
        "Sharpe (session)": pnl.mean() / pnl.std(),
        "Sharpe (annual)": pnl.mean() / pnl.std() * np.sqrt(SESSIONS_PER_YEAR),
        "5th pct P&L": pnl.quantile(0.05),
        "losing sessions": (pnl < 0).mean(),
        "mean max drawdown": df["max_drawdown"].mean(),
        "worst drawdown": df["max_drawdown"].min(),
        "inventory sd": df["inventory_std"].mean(),
        "mean max |inv|": df["max_abs_inventory"].mean(),
        "worst |inv|": df["max_abs_inventory"].max(),
        "fills": df["fills"].mean(),
        "limit-bound steps": df["limit_steps"].mean(),
    }


def parts(df):
    """Reduce one strategy's results to the P&L decomposition row."""
    n = df["fills"].mean()
    return {
        "spread captured": df["spread"].mean(),
        "adverse selection": -df["adverse_selection"].mean(),
        "inventory P&L": df["inventory"].mean(),
        "total": df["total"].mean(),
        "spread per fill": df["spread"].mean() / n,
        "adverse per fill": -df["adverse_selection"].mean() / n,
    }


def run_market(market, n_sessions=SESSIONS, seed=SEED):
    return {m.name: frame(sim.run_sessions(m, market, n_sessions, seed=seed))
            for m in makers()}


def paired(results, base="naive"):
    """Compare each strategy to the baseline on matched sessions."""
    rows = []
    for name, df in results.items():
        if name == base:
            continue
        diff = df["pnl"].values - results[base]["pnl"].values
        rows.append({"strategy": name, "mean P&L vs naive": diff.mean(),
                     "t-stat": diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))})
    return pd.DataFrame(rows).set_index("strategy")


def main():
    print(f"market: s0={BASE.s0}, sigma={BASE.sigma} per session, {BASE.n_steps} steps, "
          f"A={BASE.arrival_rate}, kappa={BASE.decay}, informed {BASE.informed_frac:.0%}, "
          f"informed horizon {BASE.informed_lag} steps, position limit {BASE.position_limit}")
    print(f"informed edge sigma*sqrt(h) = "
          f"{BASE.sigma * np.sqrt(BASE.informed_lag / BASE.n_steps):.2f} "
          f"against a naive half-spread of {Naive().half_spread}")
    print(f"{SESSIONS} sessions per strategy, same price paths and same order flow for all three")

    base = run_market(BASE)
    head = pd.DataFrame({k: metrics(v) for k, v in base.items()}).round(2)
    show("Base market: headline", head)
    head.to_csv(os.path.join(REPORTS, "headline.csv"))

    dec = pd.DataFrame({k: parts(v) for k, v in base.items()}).round(3)
    show("Base market: P&L decomposition, mean dollars per session", dec)
    dec.to_csv(os.path.join(REPORTS, "decomposition.csv"))

    show("Base market: paired difference against the naive maker", paired(base).round(2))

    jumpy = run_market(JUMPY)
    jhead = pd.DataFrame({k: {**metrics(v), **parts(v)} for k, v in jumpy.items()}).round(2)
    show(f"Jump market: {JUMPY.jump_rate} jumps per session of {JUMPY.jump_sd:.1%} log size",
         jhead)
    jhead.to_csv(os.path.join(REPORTS, "jump_market.csv"))

    show("Naive half-spread sweep, base market (the baseline is tuned in its own favour)",
         sweep_param([(h, Naive(h)) for h in (0.5, 0.65, 0.85, 1.0, 1.3, 1.7)],
                     "half-spread").round(2))
    show("Skew sweep at a fixed 0.85 half-spread",
         sweep_param([(s, Skewed(0.85, s)) for s in (0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6)],
                     "skew").round(2))
    show("Avellaneda-Stoikov risk aversion sweep",
         sweep_param([(g, AvellanedaStoikov(g)) for g in (0.02, 0.05, 0.1, 0.2, 0.4)],
                     "gamma").round(2))

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    sweep = informed_sweep(fractions)
    sweep.to_csv(os.path.join(REPORTS, "informed_sweep.csv"), index=False)
    show("Performance against the informed fraction of order flow",
         sweep.pivot(index="informed_frac", columns="strategy",
                     values=["mean_pnl", "sharpe", "adverse_selection", "inventory_std"]).round(2))

    charts(base, jumpy, sweep)
    print(f"\ncharts and tables written to {REPORTS}/")


def sweep_param(pairs, label, market=None, n=SWEEP_SESSIONS):
    market = market or BASE
    rows = []
    for value, maker in pairs:
        df = frame(sim.run_sessions(maker, market, n, seed=SEED))
        rows.append({label: value, "mean P&L": df["pnl"].mean(),
                     "Sharpe": df["pnl"].mean() / df["pnl"].std(),
                     "inventory sd": df["inventory_std"].mean(),
                     "adverse selection": -df["adverse_selection"].mean(),
                     "fills": df["fills"].mean()})
    return pd.DataFrame(rows).set_index(label)


def informed_sweep(fractions, n=SWEEP_SESSIONS):
    rows = []
    for phi in fractions:
        market = sim.Market(informed_frac=phi, informed_lag=BASE.informed_lag,
                            position_limit=BASE.position_limit)
        for maker in makers():
            df = frame(sim.run_sessions(maker, market, n, seed=SEED))
            rows.append({"informed_frac": phi, "strategy": maker.name,
                         "mean_pnl": df["pnl"].mean(),
                         "sharpe": df["pnl"].mean() / df["pnl"].std(),
                         "adverse_selection": -df["adverse_selection"].mean(),
                         "inventory_std": df["inventory_std"].mean(),
                         "fills": df["fills"].mean()})
    return pd.DataFrame(rows)


def charts(base, jumpy, sweep):
    sample_session(jumpy["naive"]["pnl"])
    inventory_paths()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.histogram_bin_edges(np.concatenate([d["pnl"] for d in base.values()]), 45)
    for name, df in base.items():
        ax.hist(df["pnl"], bins=bins, histtype="step", linewidth=1.5,
                color=COLOURS[name], label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("session P&L ($)")
    ax.set_ylabel("sessions")
    ax.set_title(f"P&L across {SESSIONS} sessions, {BASE.informed_frac:.0%} informed flow")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "pnl_distribution.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["spread captured", "adverse selection", "inventory P&L", "total"]
    width = 0.26
    x = np.arange(len(labels))
    for k, (name, df) in enumerate(base.items()):
        p = parts(df)
        ax.bar(x + (k - 1) * width, [p[l] for l in labels], width,
               color=COLOURS[name], label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean dollars per session")
    ax.set_title("P&L decomposition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "decomposition.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [("mean_pnl", "mean P&L ($)"), ("sharpe", "Sharpe per session"),
              ("adverse_selection", "adverse selection ($)"),
              ("inventory_std", "inventory sd")]
    for ax, (col, label) in zip(axes.ravel(), panels):
        for name in COLOURS:
            sub = sweep[sweep.strategy == name]
            ax.plot(sub.informed_frac, sub[col], marker="o", markersize=3.5,
                    color=COLOURS[name], label=name)
        ax.set_xlabel("informed fraction of order flow")
        ax.set_ylabel(label)
        ax.axhline(0, color="black", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Comparative static: performance against informed flow")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "informed_sweep.png"), dpi=140)
    plt.close(fig)


def sample_session(naive_pnl, seed=268):
    """Draw one session picked to show the naive maker's failure mode, not a typical one."""
    draws = sim.draw_session(JUMPY, seed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for row, maker in enumerate((Naive(), AvellanedaStoikov())):
        session = sim.simulate(maker, JUMPY, draws)
        t = np.arange(JUMPY.n_steps)
        ax = axes[row, 0]
        ax.plot(session.price, color="black", linewidth=1.0, label="efficient price")
        ax.plot(t, session.bids, color="#4c72b0", linewidth=0.7, label="bid")
        ax.plot(t, session.asks, color="#c44e52", linewidth=0.7, label="ask")
        buys = [(i, p) for i, q, p, _ in session.fills if q == 1]
        sells = [(i, p) for i, q, p, _ in session.fills if q == -1]
        if buys:
            ax.scatter(*zip(*buys), marker="^", s=16, color="#4c72b0", zorder=3)
        if sells:
            ax.scatter(*zip(*sells), marker="v", s=16, color="#c44e52", zorder=3)
        ax.set_ylabel("price")
        ax.set_title(f"{maker.name}: quotes and fills")
        if row == 0:
            ax.legend(fontsize=7)
        axes[row, 1].plot(session.inventory, color=COLOURS[maker.name], linewidth=1.0)
        axes[row, 1].axhline(0, color="black", linewidth=0.6)
        for level in (JUMPY.position_limit, -JUMPY.position_limit):
            axes[row, 1].axhline(level, color="grey", linestyle=":", linewidth=0.8)
        axes[row, 1].set_ylabel("inventory")
        axes[row, 1].set_title(f"{maker.name}: inventory, session P&L {session.total_pnl:+.1f}")
        if maker.name == "naive":
            pct = (naive_pnl < session.total_pnl).mean()
            axes[row, 1].set_title(f"{maker.name}: inventory, session P&L "
                                   f"{session.total_pnl:+.1f} ({pct:.1%} of sessions)")
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle("One bad session for the naive maker, same prices and same order flow "
                 "for both (jump market)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "sample_session.png"), dpi=140)
    plt.close(fig)


def inventory_paths():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, maker in zip(axes, makers()):
        for k in range(30):
            session = sim.simulate(maker, BASE, sim.draw_session(BASE, SEED + k))
            ax.plot(session.inventory, linewidth=0.7, alpha=0.55,
                    color=COLOURS[maker.name])
        ax.axhline(0, color="black", linewidth=0.6)
        for level in (BASE.position_limit, -BASE.position_limit):
            ax.axhline(level, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(maker.name)
        ax.set_xlabel("step")
    axes[0].set_ylabel("inventory")
    fig.suptitle(f"Inventory paths, 30 sessions each, {BASE.informed_frac:.0%} informed flow")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "inventory_paths.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
