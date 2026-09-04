"""Full pipeline on SPY: state tables, the in-sample trap, rolling OOS strategy vs benchmarks.

Run: python3 run.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import regime
from backtest import (COST_BPS, FIRST_TEST_YEAR, by_year, hmm_exposure_insample,
                      hmm_exposure_oos, run, summary, vol_target_exposure)

REPORTS = data.REPORTS
os.makedirs(REPORTS, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.f.write(s)
        self.stdout.write(s)

    def flush(self):
        self.f.flush()
        self.stdout.flush()


sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def fmt(perf):
    out = perf.copy()
    for c in ["annual_return_gross", "annual_return_net", "annual_vol", "max_drawdown_net", "avg_exposure"]:
        out[c] = out[c].map("{:+.2%}".format)
    for c in ["sharpe_gross", "sharpe_net"]:
        out[c] = out[c].map("{:.2f}".format)
    out["annual_turnover"] = out["annual_turnover"].map("{:.1f}x".format)
    out["dd_trough"] = out["dd_trough"].map(lambda d: f"{d:%Y-%m}")
    return out


px = data.download()
ret = px.pct_change().dropna()
ret_pct = 100 * data.log_returns(px)
print(f"SPY {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}, {len(ret)} daily returns")

# 1. Full-sample fits, for characterisation only.
full = {k: regime.fit(ret_pct, k) for k in (2, 3)}
for k, m in full.items():
    d = regime.describe(m)
    d.to_csv(os.path.join(REPORTS, f"states_{k}.csv"))
    show(f"{k}-state fit on the full sample 2000-2026 (log-likelihood {m.score(ret_pct.to_numpy().reshape(-1, 1)):.1f})", d.round(4))
    show(f"{k}-state transition matrix", pd.DataFrame(m.transmat_, index=d.index, columns=d.index).round(4))

# 2. Smoothed vs filtered on the full-sample 2-state fit.
m2 = full[2]
bear = regime.bear_state(m2)
p_smooth = pd.Series(regime.smoothed(m2, ret_pct)[:, bear], index=ret_pct.index)
p_filt = pd.Series(regime.filtered(m2, ret_pct)[:, bear], index=ret_pct.index)
pd.DataFrame({"smoothed_bear": p_smooth, "filtered_bear": p_filt}).to_csv(
    os.path.join(REPORTS, "probabilities_full_sample.csv"))
print(f"\nbear-probability, full-sample fit: smoothed > 0.5 on {(p_smooth > 0.5).mean():.1%} of days, "
      f"filtered > 0.5 on {(p_filt > 0.5).mean():.1%}; correlation {p_smooth.corr(p_filt):.2f}")

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(px[ret.index], color="black", lw=0.8)
ax[0].fill_between(ret.index, px.min(), px.max(), where=p_smooth > 0.5, color="red", alpha=0.15,
                   label="smoothed P(bear) > 0.5")
ax[0].set_yscale("log")
ax[0].set_title("SPY with full-sample smoothed bear regime (this is the in-sample view, not tradeable)")
ax[0].legend(loc="upper left")
ax[1].plot(p_smooth, lw=0.7, label="smoothed (uses future)")
ax[1].plot(p_filt, lw=0.7, alpha=0.7, label="filtered (real time)")
ax[1].set_ylabel("P(bear)")
ax[1].legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "regimes_full_sample.png"), dpi=130)
plt.close(fig)

# 3. Out-of-sample exposures. Expanding window, refit each January.
oos2, fits2 = hmm_exposure_oos(ret_pct, 2)
oos3, fits3 = hmm_exposure_oos(ret_pct, 3)
fits2.to_csv(os.path.join(REPORTS, "refits_2state.csv"))
fits3.to_csv(os.path.join(REPORTS, "refits_3state.csv"))
show("2-state parameters by refit year (fit on data before that year)",
     fits2.reset_index().pivot(index="fit_year", columns="index",
                               values=["mean_ann", "vol_ann", "expected_duration"]).round(3))

test = ret[ret.index.year >= FIRST_TEST_YEAR]
exposures = {
    "buy_and_hold": pd.Series(1.0, index=test.index),
    "vol_target_15": vol_target_exposure(ret).reindex(test.index),
    "hmm_2state": oos2.reindex(test.index),
    "hmm_3state": oos3.reindex(test.index),
}
# The trap, shown on the same window: full-sample fit, smoothed probabilities.
exposures["trap_insample_smoothed"] = hmm_exposure_insample(ret_pct, 2, use_smoothed=True).reindex(test.index)
exposures["trap_insample_filtered"] = hmm_exposure_insample(ret_pct, 2, use_smoothed=False).reindex(test.index)

perf = summary(test, exposures)
perf.to_csv(os.path.join(REPORTS, "performance.csv"))
show(f"performance {FIRST_TEST_YEAR}-{test.index[-1].year}, {COST_BPS:.0f}bps one-way "
     "(hmm_* rows are out-of-sample, trap_* rows are the in-sample fit for contrast)", fmt(perf))

pd.DataFrame(exposures).to_csv(os.path.join(REPORTS, "exposures.csv"))

# 4. Sharpe at matched vol: exposure rules that sit in cash have less vol, so also
# report each rule levered to buy-and-hold's realised vol (costs scale with it).
bh_vol = run(test, exposures["buy_and_hold"])["net"].std()
matched = {}
for name in ["vol_target_15", "hmm_2state", "hmm_3state"]:
    raw = run(test, exposures[name], cost_bps=0)["gross"].std()
    matched[name] = exposures[name] * bh_vol / raw
matched["buy_and_hold"] = exposures["buy_and_hold"]
perf_matched = summary(test, matched)
perf_matched.to_csv(os.path.join(REPORTS, "performance_vol_matched.csv"))
show("same rules levered to buy-and-hold's realised vol (leverage allowed, same 5bps)", fmt(perf_matched))

# 5. By year.
yearly = by_year(test, {k: exposures[k] for k in ["buy_and_hold", "vol_target_15", "hmm_2state", "hmm_3state"]})
yearly.to_csv(os.path.join(REPORTS, "by_year.csv"))
show("net return by year", yearly["return"].map("{:+.1%}".format))
show("net Sharpe by year", yearly["sharpe"].round(2))

# 6. Cost sensitivity.
rows = []
for bps in [0, 5, 10, 20, 40]:
    s = summary(test, {k: exposures[k] for k in ["vol_target_15", "hmm_2state", "hmm_3state"]}, cost_bps=bps)
    rows.append(s["sharpe_net"].rename(f"{bps}bps"))
costs = pd.concat(rows, axis=1).T
costs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))
show("net Sharpe by one-way cost", costs.round(2))

# 7. Crisis windows: what did each rule hold going in.
crises = [("2011 Aug debt-ceiling", "2011-07-22", "2011-10-03"),
          ("2015 Aug", "2015-08-17", "2015-08-25"),
          ("2018 Q4", "2018-10-01", "2018-12-24"),
          ("2020 Covid crash", "2020-02-19", "2020-03-23"),
          ("2020 recovery", "2020-03-24", "2020-12-31"),
          ("2022 full year", "2022-01-03", "2022-12-30")]
rows = []
for name, a, b in crises:
    row = {"window": name}
    for k in ["buy_and_hold", "vol_target_15", "hmm_2state", "hmm_3state"]:
        bt = run(test, exposures[k])
        row[k] = (1 + bt["net"][a:b]).prod() - 1
        row[f"{k}_exp"] = bt["position"][a:b].mean()
    rows.append(row)
crisis = pd.DataFrame(rows).set_index("window")
crisis.to_csv(os.path.join(REPORTS, "crises.csv"))
show("net return over stress windows", crisis[[c for c in crisis.columns if not c.endswith("_exp")]].map("{:+.1%}".format))
show("average exposure over stress windows", crisis[[c for c in crisis.columns if c.endswith("_exp")]].round(2))

# 8. Charts.
fig, ax = plt.subplots(figsize=(12, 5))
for k in ["buy_and_hold", "vol_target_15", "hmm_2state", "hmm_3state", "trap_insample_smoothed"]:
    eq = (1 + run(test, exposures[k])["net"]).cumprod()
    ax.plot(eq, lw=1.0 if k != "trap_insample_smoothed" else 0.8,
            ls="--" if k.startswith("trap") else "-", label=k)
ax.set_yscale("log")
ax.set_title(f"net equity curves, {FIRST_TEST_YEAR} onward, {COST_BPS:.0f}bps one-way")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
for k in ["buy_and_hold", "vol_target_15", "hmm_2state"]:
    eq = (1 + run(test, exposures[k])["net"]).cumprod()
    ax.plot(eq / eq.cummax() - 1, lw=0.8, label=k)
ax.set_title("drawdown, net")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "drawdown.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(px[test.index[0]:], color="black", lw=0.8)
ax[0].set_yscale("log")
ax[0].set_title("SPY and out-of-sample exposures")
ax[1].plot(exposures["hmm_2state"], lw=0.6, label="hmm_2state")
ax[1].plot(exposures["vol_target_15"], lw=0.6, alpha=0.7, label="vol_target_15")
ax[1].set_ylabel("exposure")
ax[1].legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "exposure.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for i, (col, title) in enumerate([("mean_ann", "annualised mean"), ("vol_ann", "annualised vol"),
                                  ("expected_duration", "expected duration (days)")]):
    for s in fits2.index.unique():
        ax[i].plot(fits2.loc[s, "fit_year"], fits2.loc[s, col], marker="o", ms=3, label=s)
    ax[i].set_title(title)
ax[0].legend()
fig.suptitle("2-state parameters by refit year")
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "parameter_drift.png"), dpi=130)
plt.close(fig)

print(f"\nwrote reports to {REPORTS}")
