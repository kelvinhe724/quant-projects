"""Full pipeline on FRED data: in-sample probit, expanding-window forecasts, scoring, the 2022-24 episode.

Run: python3 run.py
"""
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from model import (auc, brier, calibration, expanding_forecasts, fit_probit,
                   inversion_episodes, predict, roc_curve, scorecard)

REPORTS = data.REPORTS
OOS_START = "1980-01-01"
ANNOUNCE_LAG = 12
MODELS = {
    "spread": ["spread_10y3m"],
    "sahm": ["sahm"],
    "both": ["spread_10y3m", "sahm"],
}


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


def base_rate_forecasts(label, start, announce_lag):
    """Expanding mean of the labels that were knowable at each date."""
    known = label.shift(data.HORIZON + announce_lag)
    out = known.expanding().mean()
    out[out.index < pd.Timestamp(start)] = np.nan
    return out.rename("base_rate")


def shade_recessions(ax, usrec):
    for _, row in inversion_episodes(-usrec + 0.5).iterrows():
        ax.axvspan(row["start"], row["end"], color="grey", alpha=0.25, lw=0)


def main():
    m = data.monthly()
    d = data.daily()
    scored = m.dropna(subset=["recession_ahead"])
    recessions = inversion_episodes(-m["usrec"] + 0.5)
    print(f"sample {m.index[0].date()} to {m.index[-1].date()}, {len(m)} months")
    print(f"label observable through {scored.index[-1].date()}, {len(scored)} months")
    print(f"NBER recessions in sample: {len(recessions)}, "
          f"{int(m['usrec'].sum())} recession months, "
          f"label rate {scored['recession_ahead'].mean():.3f}")
    print(f"USREC since 2022-01: {int(m.loc['2022':, 'usrec'].sum())} recession months")

    print("\nIN-SAMPLE PROBIT (full sample, HAC standard errors)")
    fits = {}
    rows = []
    for name, feats in [("spread 10y-3m", ["spread_10y3m"]), ("spread 10y-2y", ["spread_10y2y"]),
                        ("sahm", ["sahm"]), ("spread + sahm", ["spread_10y3m", "sahm"])]:
        fit = fit_probit(scored, feats)
        fits[name] = fit
        p = predict(fit, scored.dropna(subset=feats), feats)
        y = scored.loc[p.index, "recession_ahead"]
        row = {"model": name, "n": int(fit.nobs), "pseudo_r2": fit.prsquared,
               "auc_in_sample": auc(y, p), "brier_in_sample": brier(y, p)}
        for f in ["const"] + feats:
            row[f"coef_{f}"] = fit.params[f]
            row[f"t_{f}"] = fit.tvalues[f]
        rows.append(row)
    insample = pd.DataFrame(rows).set_index("model")
    show("in-sample fits", insample.round(3))
    insample.to_csv(os.path.join(REPORTS, "insample_fit.csv"))

    spread_fit = fits["spread 10y-3m"]
    for s in [1.0, 0.0, -0.5, -1.0]:
        p = spread_fit.predict(np.array([[1.0, s]]))[0]
        print(f"  P(recession within 12m | spread = {s:+.1f}) = {p:.2f}")

    print("\nINVERSION EPISODES (monthly average 10y-3m below zero)")
    ep = inversion_episodes(m["spread_10y3m"])
    ep["recession_within_12m"] = [
        bool(m.loc[a: a + pd.DateOffset(months=data.HORIZON), "usrec"].max() == 1) for a in ep["start"]]
    nxt = []
    for a in ep["start"]:
        later = recessions[recessions["start"] > a]
        nxt.append((later["start"].iloc[0] - a).days // 30 if len(later) else np.nan)
    ep["months_to_next_recession"] = nxt
    show("episodes", ep)
    ep.to_csv(os.path.join(REPORTS, "inversions.csv"), index=False)
    missed = [r["start"].date() for _, r in recessions.iterrows()
              if not (m.loc[r["start"] - pd.DateOffset(months=18): r["start"], "spread_10y3m"] < 0).any()]
    print(f"recessions with no inversion in the prior 18 months: {missed or 'none'}")

    print(f"\nEXPANDING-WINDOW FORECASTS from {OOS_START}, refit every month")
    fc = pd.DataFrame(index=m.index)
    fc["recession_ahead"] = m["recession_ahead"]
    fc["spread_10y3m"] = m["spread_10y3m"]
    fc["usrec"] = m["usrec"]
    for lag in [0, ANNOUNCE_LAG]:
        fc[f"base_rate_lag{lag}"] = base_rate_forecasts(m["recession_ahead"], OOS_START, lag)
        for name, feats in MODELS.items():
            fc[f"{name}_lag{lag}"] = expanding_forecasts(m, feats, OOS_START, announce_lag=lag)
    fc.to_csv(os.path.join(REPORTS, "forecasts.csv"))

    cards = []
    ev = fc.dropna(subset=["recession_ahead"]).loc[OOS_START:]
    for col in [c for c in fc.columns if "_lag" in c]:
        ok = ev[col].notna()
        card = scorecard(ev.loc[ok, "recession_ahead"], ev.loc[ok, col])
        card["model"], card["announce_lag"] = col.rsplit("_lag", 1)
        base = ev.loc[ok, f"base_rate_lag{card['announce_lag']}"]
        card["brier_base_rate"] = brier(ev.loc[ok, "recession_ahead"], base)
        card["skill_vs_base_rate"] = 1 - card["brier"] / card["brier_base_rate"]
        cards.append(card)
    cards = pd.DataFrame(cards).set_index(["announce_lag", "model"])[
        ["n", "positives", "auc", "brier", "brier_base_rate", "skill_vs_base_rate"]]
    show(f"out-of-sample scorecard, {ev.index[0].date()} to {ev.index[-1].date()}", cards.round(3))
    cards.to_csv(os.path.join(REPORTS, "scorecard.csv"))

    cal = calibration(ev["recession_ahead"], ev["spread_lag0"], bins=10)
    show("calibration of the spread model, out of sample, by forecast decile", cal.round(3))
    cal.to_csv(os.path.join(REPORTS, "calibration.csv"), index=False)

    print("\nSUBPERIOD AUC, spread model vs sahm model (lag 0)")
    sub = []
    for a, b in [("1980", "1999"), ("2000", "2019"), ("1980", "2007"), ("2008", str(ev.index[-1].year))]:
        w = ev.loc[a:b]
        sub.append({"window": f"{a}-{b}", "n": len(w), "positives": int(w["recession_ahead"].sum()),
                    "auc_spread": auc(w["recession_ahead"], w["spread_lag0"]),
                    "auc_sahm": auc(w["recession_ahead"], w["sahm_lag0"]),
                    "auc_both": auc(w["recession_ahead"], w["both_lag0"])})
    show("subperiods", pd.DataFrame(sub).set_index("window").round(3))

    print("\nTHE 2022-2024 INVERSION, what the model said in real time")
    recent = fc.loc["2022-01-01":, ["spread_10y3m", "spread_lag0", "spread_lag12", "sahm_lag0",
                                     "both_lag0", "base_rate_lag0", "usrec", "recession_ahead"]]
    recent["sahm"] = m.loc[recent.index, "sahm"]
    show("monthly", recent.round(2))
    recent.to_csv(os.path.join(REPORTS, "episode_2022.csv"))
    daily_neg = d.loc["2022-01-01":, "T10Y3M"].dropna() < 0
    inv_days = daily_neg[daily_neg].index
    runs = [(len(days := [ts for ts, _ in grp]), days[0], days[-1])
            for neg, grp in itertools.groupby(daily_neg.items(), key=lambda kv: kv[1]) if neg]
    n_run, run_start, run_end = max(runs)
    peak = recent["spread_lag0"].idxmax()
    print(f"daily 10y-3m first inverted {inv_days[0].date()}, last inverted {inv_days[-1].date()}, "
          f"{int(daily_neg.sum())} inverted trading days in total; longest unbroken run "
          f"{run_start.date()} to {run_end.date()}, {n_run} trading days")
    print(f"deepest monthly inversion {m['spread_10y3m'].loc['2022':].min():+.2f} "
          f"in {m['spread_10y3m'].loc['2022':].idxmin():%Y-%m}")
    print(f"peak real-time probability {recent['spread_lag0'].max():.2f} in {peak:%Y-%m}; "
          f"months with P >= 0.5: {int((recent['spread_lag0'] >= 0.5).sum())}")
    print(f"NBER recession months since 2022-01 as of data end ({m.index[-1]:%Y-%m}): "
          f"{int(m.loc['2022':, 'usrec'].sum())}")
    print(f"Sahm gap since 2022: peak {recent['sahm'].max():+.2f} in {recent['sahm'].idxmax():%Y-%m}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    shade_recessions(axes[0], m["usrec"])
    axes[0].plot(m.index, m["spread_10y3m"], lw=0.9, color="C0")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("10y - 3m spread (pp)")
    axes[0].set_title("Term spread, NBER recessions shaded")
    shade_recessions(axes[1], m["usrec"])
    axes[1].plot(fc.index, fc["spread_lag0"], lw=0.9, color="C3", label="spread probit, expanding window")
    axes[1].plot(fc.index, fc["base_rate_lag0"], lw=0.8, color="grey", ls="--", label="base rate")
    axes[1].set_ylabel("P(recession within 12m)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "history.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for col, lbl in [("spread_lag0", "spread"), ("sahm_lag0", "sahm"), ("both_lag0", "spread + sahm")]:
        fp, tp = roc_curve(ev["recession_ahead"], ev[col])
        axes[0].plot(fp, tp, label=f"{lbl}  AUC {auc(ev['recession_ahead'], ev[col]):.2f}")
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"Out-of-sample ROC, {ev.index[0].year}-{ev.index[-1].year}")
    axes[0].legend(loc="lower right")
    axes[1].plot(cal["forecast"], cal["observed"], marker="o", color="C3")
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("mean forecast in decile")
    axes[1].set_ylabel("observed recession rate")
    axes[1].set_title("Calibration, spread model, out of sample")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "roc_calibration.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(13, 5))
    dd = d.loc["2021-06-01":, "T10Y3M"].dropna()
    ax.plot(dd.index, dd, lw=0.8, color="C0", label="daily 10y - 3m")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("spread (pp)")
    ax2 = ax.twinx()
    ax2.plot(recent.index, recent["spread_lag0"], color="C3", lw=1.5, drawstyle="steps-post",
             label="real-time P(recession within 12m)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("probability")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("2022-2024 inversion: no NBER recession as of data end")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "episode_2022.png"), dpi=130)
    print("\ncharts and tables written to reports/")


if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))
    main()
