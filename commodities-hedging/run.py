"""Full cross-hedge pipeline on the thirteen continuous futures. Tables and charts to reports/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import SPECS, TEST_START, TRAIN_END, get_panel, notional, split
from hedging import (block_bootstrap, contracts_from_return_beta, effectiveness,
                     hedged_pnl, integer_hedge, min_var_ratio, ols_hac, r2_matrix,
                     risk_report, rolling_ratio, screen, spurious_demo,
                     subperiod_table, walk_forward)

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)
pd.set_option("display.width", 160)

GROUPS = {t: SPECS[t]["group"] for t in SPECS}
LABEL = {t: SPECS[t]["name"] for t in SPECS}


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


def save(fig, name):
    path = os.path.join(REPORTS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


heading("1. CONTRACT SPECIFICATION AUDIT")

px, returns, pnl, cov = get_panel()
specs = pd.DataFrame(SPECS).T[["name", "venue", "quote", "lot", "quote_conversion",
                               "multiplier", "tick", "tick_value"]]
print(specs.to_string())
specs.to_csv(os.path.join(REPORTS, "contract_specs.csv"))

heading("2. DATA COVERAGE")

print(cov.to_string())
print(f"\ncommon window {px.index[0].date()} to {px.index[-1].date()}, {len(px)} trading days")
train_r, test_r = split(returns)
train_p, test_p = split(pnl)
print(f"train {train_r.index[0].date()} to {train_r.index[-1].date()} ({len(train_r)} days)")
print(f"test  {test_r.index[0].date()} to {test_r.index[-1].date()} ({len(test_r)} days)")
cov.to_csv(os.path.join(REPORTS, "coverage.csv"))

heading("3. RETURN AND CONTRACT-P&L SUMMARY")

summary = pd.DataFrame({
    "ann_vol_%": returns.std() * np.sqrt(252) * 100,
    "skew": returns.skew(),
    "kurtosis": returns.kurtosis(),
    "min_%": returns.min() * 100,
    "max_%": returns.max() * 100,
    "notional_$": notional(px.iloc[-1]),
    "daily_sd_$": pnl.std(),
    "tick_$": [SPECS[t]["tick_value"] for t in returns.columns],
})
summary.index = [LABEL[t] for t in summary.index]
print(summary.round(2).to_string())
summary.to_csv(os.path.join(REPORTS, "return_summary.csv"))

heading("4. THE SPURIOUS-REGRESSION TRAP")

print("Regressing price LEVELS on price levels, all 78 pairs, training window:\n")
level_rows, return_rows = [], []
train_px = px.loc[:TRAIN_END]
for i, a in enumerate(px.columns):
    for b in px.columns[i + 1:]:
        lev = ols_hac(train_px[a], train_px[b])
        ret = ols_hac(train_r[a], train_r[b])
        level_rows.append({"a": a, "b": b, "r2": lev["r2"], "dw": lev["dw"],
                           "t": abs(lev["beta"] / lev["beta_se"])})
        return_rows.append({"a": a, "b": b, "r2": ret["r2"], "dw": ret["dw"],
                            "t": abs(ret["beta"] / ret["beta_se"])})
levels = pd.DataFrame(level_rows)
rets_reg = pd.DataFrame(return_rows)

compare = pd.DataFrame({
    "mean R2": [levels["r2"].mean(), rets_reg["r2"].mean()],
    "median R2": [levels["r2"].median(), rets_reg["r2"].median()],
    "max R2": [levels["r2"].max(), rets_reg["r2"].max()],
    "pairs with R2 > 0.5": [(levels["r2"] > 0.5).sum(), (rets_reg["r2"] > 0.5).sum()],
    "mean Durbin-Watson": [levels["dw"].mean(), rets_reg["dw"].mean()],
    "pairs with |t| > 1.96": [(levels["t"] > 1.96).sum(), (rets_reg["t"] > 1.96).sum()],
}, index=["price levels", "daily returns"]).T
print(compare.round(3).to_string())
compare.to_csv(os.path.join(REPORTS, "spurious_comparison.csv"))

print("\nHighest level R2 among pairs whose return R2 is under 0.05:")
merged = levels.merge(rets_reg, on=["a", "b"], suffixes=("_lev", "_ret"))
junk = merged[merged["r2_ret"] < 0.05].sort_values("r2_lev", ascending=False).head(6)
for _, row in junk.iterrows():
    print(f"  {LABEL[row['a']]:<16} vs {LABEL[row['b']]:<16} "
          f"levels R2 {row['r2_lev']:.3f} (DW {row['dw_lev']:.2f})   "
          f"returns R2 {row['r2_ret']:.4f}")

print("\nControl: 200 pairs of independent simulated random walks put through the same code.")
demo = spurious_demo(n=len(train_px), n_pairs=200, seed=1)
print(f"  levels     mean R2 {demo['r2_levels'].mean():.3f}, "
      f"mean Durbin-Watson {demo['dw_levels'].mean():.3f}, "
      f"|t| > 1.96 in {(demo['t_levels'] > 1.96).mean():.0%} of pairs")
print(f"  differences mean R2 {demo['r2_diffs'].mean():.5f}, "
      f"|t| > 1.96 in {(demo['t_diffs'] > 1.96).mean():.0%} of pairs")
print("\nEvery regression from here on uses returns or per-contract dollar P&L.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(levels["r2"], bins=20, color="#8c6d3f", edgecolor="white")
ax[0].set_title("real pairs, price-level regressions")
ax[0].set_xlabel("R-squared")
ax[1].hist(demo["r2_levels"], bins=20, color="#7a7a7a", edgecolor="white")
ax[1].set_title("independent random walks, price-level regressions")
ax[1].set_xlabel("R-squared")
for a in ax:
    a.set_xlim(0, 1)
    a.set_ylabel("pairs")
fig.suptitle("A high R-squared on price levels means very little")
save(fig, "spurious_regression.png")

heading("5. PAIRWISE SCREEN ON RETURNS, TRAINING WINDOW")

ranked = screen(train_r)
ranked["target"] = [LABEL[a] for a in ranked["a"]]
ranked["hedge"] = [LABEL[b] for b in ranked["b"]]
print(f"{len(ranked)} unordered pairs, {len(train_r)} common observations each\n")
print("Top 15 by R-squared:")
print(ranked.head(15)[["target", "hedge", "corr", "r2"]].to_string(index=False,
                                                                  float_format="%.4f"))
print("\nBottom 5:")
print(ranked.tail(5)[["target", "hedge", "corr", "r2"]].to_string(index=False,
                                                                 float_format="%.4f"))
ranked.to_csv(os.path.join(REPORTS, "pairwise_screen.csv"), index=False)

r2m = r2_matrix(train_r)
corr = train_r.corr()
order = sorted(px.columns, key=lambda t: (GROUPS[t], t))
names = [LABEL[t] for t in order]

fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
for axis, mat, title, cmap, lim in [
    (ax[0], r2m.loc[order, order], "R-squared (screening statistic)", "YlOrBr", (0, 0.8)),
    (ax[1], corr.loc[order, order], "correlation (gives the hedge direction)", "RdBu_r", (-1, 1))]:
    im = axis.imshow(mat.to_numpy(), cmap=cmap, vmin=lim[0], vmax=lim[1])
    axis.set_xticks(range(len(order)), names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(order)), names, fontsize=8)
    axis.set_title(title)
    for i in range(len(order)):
        for j in range(len(order)):
            axis.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axis, fraction=0.046)
fig.suptitle(f"Daily returns, {train_r.index[0].date()} to {train_r.index[-1].date()}")
save(fig, "heatmaps.png")

heading("6. UNIT-AWARE HEDGE RATIOS")

ref_px = px.loc[:TRAIN_END].iloc[-1]
ref_notional = notional(ref_px)
print(f"Reference date for notionals: {px.loc[:TRAIN_END].index[-1].date()}\n")

rows = []
for _, pair in ranked.iterrows():
    for target, hedge in [(pair["a"], pair["b"]), (pair["b"], pair["a"])]:
        beta = ols_hac(train_r[target], train_r[hedge])
        h_pnl = min_var_ratio(train_p[target], train_p[hedge])
        h_ret = contracts_from_return_beta(beta["beta"], ref_px[target], ref_px[hedge],
                                           target, hedge)
        eff_is = effectiveness(train_p[target], train_p[hedge], h_pnl)
        eff_oos = effectiveness(test_p[target], test_p[hedge], h_pnl)
        rows.append({
            "target": target, "hedge": hedge,
            "target_name": LABEL[target], "hedge_name": LABEL[hedge],
            "n": beta["n"], "corr": pair["corr"], "r2": beta["r2"],
            "beta": beta["beta"], "beta_hac_se": beta["beta_se"],
            "alpha": beta["alpha"], "alpha_hac_se": beta["alpha_se"],
            "resid_sd": beta["resid_sd"], "dw": beta["dw"],
            "h_from_beta": h_ret, "h_from_pnl": h_pnl,
            "h_gap_%": 100 * (h_ret - h_pnl) / h_pnl if h_pnl else np.nan,
            "eff_train": eff_is, "eff_test": eff_oos,
        })
ratios = pd.DataFrame(rows)
ratios.to_csv(os.path.join(REPORTS, "hedge_ratios.csv"), index=False)

print("The two routes to a contract ratio, both directions of the ten tightest pairs:")
top = ratios.sort_values("r2", ascending=False).head(20)
print(top[["target_name", "hedge_name", "r2", "beta", "beta_hac_se",
           "h_from_beta", "h_from_pnl", "h_gap_%"]].to_string(index=False,
                                                              float_format="%.4f"))

print("\nWorked example, one long WTI contract hedged with Brent:")
row = ratios[(ratios["target"] == "CL=F") & (ratios["hedge"] == "BZ=F")].iloc[0]
print(f"  WTI notional   {ref_notional['CL=F']:>12,.0f}  ({ref_px['CL=F']:.2f} USD/bbl x 1,000 bbl)")
print(f"  Brent notional {ref_notional['BZ=F']:>12,.0f}  ({ref_px['BZ=F']:.2f} USD/bbl x 1,000 bbl)")
print(f"  return beta    {row['beta']:>12.4f}  (HAC se {row['beta_hac_se']:.4f})")
print(f"  contract ratio via notionals {row['h_from_beta']:.4f} Brent per WTI")
print(f"  contract ratio via dollar P&L regression {row['h_from_pnl']:.4f}")
print(f"  training-window variance reduction {row['eff_train']:.1%}")

heading("7. TOP THREE HEDGE CANDIDATES PER TARGET")

candidates = []
for target in px.columns:
    block = ratios[ratios["target"] == target].sort_values("eff_train", ascending=False)
    print(f"\n{LABEL[target]} ({GROUPS[target]}), one long contract, "
          f"daily P&L sd ${train_p[target].std():,.0f}")
    for rank, (_, r) in enumerate(block.head(3).iterrows(), 1):
        direction = "short" if r["h_from_pnl"] > 0 else "long"
        print(f"  {rank}. {r['hedge_name']:<16} R2 {r['r2']:.3f}  "
              f"corr {r['corr']:+.3f}  h {r['h_from_pnl']:+.3f} "
              f"({direction} {abs(r['h_from_pnl']):.2f} contracts)  "
              f"train {r['eff_train']:+.1%}  test {r['eff_test']:+.1%}")
        candidates.append({**r.to_dict(), "rank": rank})
pd.DataFrame(candidates).to_csv(os.path.join(REPORTS, "top_candidates.csv"), index=False)

heading("8. INTEGER CONTRACTS")

best = ratios.sort_values("eff_train", ascending=False).iloc[0]
target, hedge = best["target"], best["hedge"]
print(f"{LABEL[target]} hedged with {LABEL[hedge]}, continuous ratio "
      f"{best['h_from_pnl']:.4f}\n")
int_rows = []
for n_target in (1, 2, 5, 10, 25, 50, 100):
    continuous = best["h_from_pnl"] * n_target
    eff_cont = effectiveness(test_p[target], test_p[hedge], continuous, n_target=n_target)
    for candidate in integer_hedge(best["h_from_pnl"], n_target):
        int_rows.append({
            "target_contracts": n_target,
            "continuous": continuous,
            "whole_contracts": candidate,
            "eff_continuous": eff_cont,
            "eff_integer": effectiveness(test_p[target], test_p[hedge], candidate,
                                         n_target=n_target),
            "residual_notional_$": (candidate - continuous) * ref_notional[hedge],
        })
integers = pd.DataFrame(int_rows)
integers["cost_of_rounding"] = integers["eff_continuous"] - integers["eff_integer"]
print(integers.to_string(index=False, float_format="%.4f"))
integers.to_csv(os.path.join(REPORTS, "integer_contracts.csv"), index=False)
print("\nThe integer is chosen from the training ratio, not by searching the test window.")

heading("9. STABILITY")

tracked = ratios.sort_values("eff_train", ascending=False).copy()
tracked["unordered"] = [frozenset(p) for p in zip(tracked["target"], tracked["hedge"])]
pairs_to_track = tracked.drop_duplicates("unordered").head(4)
stab_rows = []
for _, r in ratios.iterrows():
    roll = rolling_ratio(pnl[r["target"]], pnl[r["hedge"]], window=60)
    stab_rows.append({
        "target_name": r["target_name"], "hedge_name": r["hedge_name"],
        "h_full": r["h_from_pnl"],
        "h_median": roll["h"].median(),
        "h_iqr": roll["h"].quantile(0.75) - roll["h"].quantile(0.25),
        "h_cv": roll["h"].std() / abs(roll["h"].mean()),
        "sign_flips_%": 100 * (np.sign(roll["h"]) != np.sign(r["h_from_pnl"])).mean(),
        "r2_median": roll["r2"].median(),
        "r2_iqr": roll["r2"].quantile(0.75) - roll["r2"].quantile(0.25),
        "r2_p05": roll["r2"].quantile(0.05),
        "windows_r2_above_25%": 100 * (roll["r2"] > 0.25).mean(),
    })
stability = pd.DataFrame(stab_rows)
stability.to_csv(os.path.join(REPORTS, "stability.csv"), index=False)
print("60-day rolling stability, the ten pairs with the highest training effectiveness:\n")
key = ratios.sort_values("eff_train", ascending=False).head(10)[["target_name", "hedge_name"]]
print(stability.merge(key, on=["target_name", "hedge_name"]).to_string(index=False,
                                                                      float_format="%.3f"))

print("\nSubperiod re-estimation, three equal blocks, both directions of the four leading pairs:")
for _, r in pairs_to_track.iterrows():
    for target, hedge in [(r["target"], r["hedge"]), (r["hedge"], r["target"])]:
        print(f"\n{LABEL[target]} on {LABEL[hedge]}:")
        print(subperiod_table(pnl, target, hedge, n_periods=3)
              .to_string(index=False, float_format="%.4f"))

print("\nBlock bootstrap, 10-day blocks, 1000 replications, training window:")
boot_rows = []
for _, r in pairs_to_track.iterrows():
    boot = block_bootstrap(train_p[r["target"]], train_p[r["hedge"]],
                           block=10, n_reps=1000, seed=5)
    lo, mid, hi = boot["h"].quantile([0.025, 0.5, 0.975])
    r2lo, r2mid, r2hi = boot["r2"].quantile([0.025, 0.5, 0.975])
    boot_rows.append({"target": r["target_name"], "hedge": r["hedge_name"],
                      "h_2.5%": lo, "h_50%": mid, "h_97.5%": hi,
                      "r2_2.5%": r2lo, "r2_50%": r2mid, "r2_97.5%": r2hi})
bootstrap = pd.DataFrame(boot_rows)
print(bootstrap.to_string(index=False, float_format="%.4f"))
bootstrap.to_csv(os.path.join(REPORTS, "bootstrap.csv"), index=False)

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for axis, (_, r) in zip(axes.ravel(), pairs_to_track.iterrows()):
    roll = rolling_ratio(pnl[r["target"]], pnl[r["hedge"]], window=60)
    axis.plot(roll.index, roll["h"], lw=0.9, color="#2f5d50")
    axis.axhline(r["h_from_pnl"], color="#b03a2e", lw=1.0, ls="--",
                 label=f"full-training h = {r['h_from_pnl']:.2f}")
    axis.axvline(pd.Timestamp(TEST_START), color="#555", lw=0.8)
    axis.set_title(f"{r['target_name']} hedged with {r['hedge_name']}", fontsize=10)
    axis.set_ylabel("contracts of hedge")
    axis.legend(fontsize=7)
fig.suptitle("60-day rolling minimum-variance ratio (vertical line = start of test window)")
save(fig, "rolling_ratios.png")

heading("10. OUT-OF-SAMPLE HEDGE VALIDATION")

print(f"Ratios estimated on {train_p.index[0].date()} to {train_p.index[-1].date()}, "
      f"frozen, applied to {test_p.index[0].date()} to {test_p.index[-1].date()}.\n")
oos = ratios.sort_values("eff_train", ascending=False).head(15)
print(oos[["target_name", "hedge_name", "r2", "h_from_pnl",
           "eff_train", "eff_test"]].to_string(index=False, float_format="%.4f"))
decay = oos["eff_train"] - oos["eff_test"]
print(f"\nmean train effectiveness {oos['eff_train'].mean():.4f}, "
      f"mean test {oos['eff_test'].mean():.4f}, mean decay {decay.mean():+.4f}")
print(f"pairs whose test effectiveness is negative: "
      f"{(ratios['eff_test'] < 0).sum()} of {len(ratios)} ordered pairs")

top_pair = oos.iloc[0]
t, h = top_pair["target"], top_pair["hedge"]
book = pd.DataFrame({
    "unhedged": test_p[t],
    "hedged_continuous": hedged_pnl(test_p[t], test_p[h], top_pair["h_from_pnl"]),
    "hedged_whole": hedged_pnl(test_p[t], test_p[h], round(top_pair["h_from_pnl"])),
})
print(f"\nRisk report, one long {LABEL[t]} contract hedged with "
      f"{top_pair['h_from_pnl']:.3f} {LABEL[h]}, test window, dollars per day:\n")
report = pd.DataFrame({
    "continuous": risk_report(book["unhedged"], book["hedged_continuous"]),
    f"whole ({round(top_pair['h_from_pnl'])} contract)":
        risk_report(book["unhedged"], book["hedged_whole"]),
})
print(report.to_string(float_format="%.4f"))
report.to_csv(os.path.join(REPORTS, "risk_report.csv"))

wf = walk_forward(test_p[t], test_p[h], window=60)
wf_eff = 1 - wf["hedged"].var() / wf["unhedged"].var()
# Compare on the same days: the walk-forward loses its first 60 test days to the window.
frozen_eff = effectiveness(test_p[t].loc[wf.index], test_p[h].loc[wf.index], top_pair["h_from_pnl"])
turnover = wf["h"].diff().abs().mean()
print(f"\nWalk-forward on the test window, 60-day trailing estimate re-fitted daily "
      f"({len(wf)} days scored):")
print(f"  effectiveness {wf_eff:.4f} against {frozen_eff:.4f} for the frozen ratio on the same days")
print(f"  mean daily change in the ratio {turnover:.4f} contracts")

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(book.index, book["unhedged"].cumsum(), color="#b03a2e", lw=1.1,
           label=f"unhedged {LABEL[t]}")
ax[0].plot(book.index, book["hedged_continuous"].cumsum(), color="#2f5d50", lw=1.1,
           label=f"hedged with {top_pair['h_from_pnl']:.2f} {LABEL[h]}")
ax[0].set_ylabel("cumulative P&L, $ per contract")
ax[0].legend(fontsize=8)
ax[0].set_title(f"Out-of-sample, ratio frozen at {TRAIN_END}")
ax[1].plot(book.index, book["unhedged"].rolling(21).std(), color="#b03a2e", lw=1.0,
           label="unhedged")
ax[1].plot(book.index, book["hedged_continuous"].rolling(21).std(), color="#2f5d50", lw=1.0,
           label="hedged")
ax[1].set_ylabel("21-day rolling sd, $/day")
ax[1].legend(fontsize=8)
save(fig, "out_of_sample.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ratios["eff_train"], ratios["eff_test"], s=14, alpha=0.6, color="#2f5d50")
lim = [-0.4, 1.0]
ax.plot(lim, lim, color="#888", lw=0.8, ls="--")
ax.axhline(0, color="#b03a2e", lw=0.7)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("training-window variance reduction")
ax.set_ylabel("test-window variance reduction")
ax.set_title("Hedge effectiveness in and out of sample, all 156 ordered pairs")
save(fig, "effectiveness_scatter.png")

heading("SUMMARY")

print(f"instruments      {len(px.columns)} continuous front-month futures")
print(f"window           {px.index[0].date()} to {px.index[-1].date()} ({len(px)} days)")
print(f"pairs            {len(ranked)} unordered, {len(ratios)} ordered target-hedge directions")
print(f"best in training {oos.iloc[0]['target_name']} on {oos.iloc[0]['hedge_name']}, "
      f"{oos.iloc[0]['eff_train']:.1%} train, {oos.iloc[0]['eff_test']:.1%} test")
print(f"spurious trap    mean level R2 {levels['r2'].mean():.3f} vs "
      f"mean return R2 {rets_reg['r2'].mean():.3f}")
