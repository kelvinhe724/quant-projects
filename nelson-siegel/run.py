"""FRED Treasury curve pipeline: static benchmark vs sequential ridge/L1,
chronological tuning, charts to reports/. The Brent path sits behind
run_brent() for when the Bloomberg CSV lands."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (load_treasury_curve, curve_slices, load_panel, daily_slices,
                  FRED_SERIES)
from ns import (fit_all_static, fit_sequential, change_scale, path_roughness,
                _predict)

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
SCALE_FALLBACK = np.array([0.02, 0.02, 0.05, 0.1])
START, END = "2015-01-01", "2025-12-31"
SHOWCASE = ["2019-01-02", "2020-04-01", "2023-07-03"]


def to_frame(results):
    return pd.DataFrame([{k: r[k] for k in ("date", "beta0", "beta1", "beta2",
                                            "lam", "rmse", "success")}
                         for r in results]).set_index("date")


def fit_everything(slices, grids=None):
    n = len(slices)
    burn_in, tune_end = 20, int(n * 0.75)
    bench = fit_all_static(slices, mode="random")
    # scale comes from the tuning segment only; the eval period stays untouched
    scale = np.maximum(change_scale(bench[:tune_end]), SCALE_FALLBACK / 10)
    grids = grids or {"ridge": [1e-5, 1e-4, 1e-3], "l1": [1e-5, 1e-4, 1e-3]}

    fitted, chosen = {"static": bench}, {}
    for pen, grid in grids.items():
        best = None
        for s in grid:
            res = fit_sequential(slices[:tune_end], pen, strength=s,
                                 scale=scale, burn_in=burn_in)
            rmse = np.mean([r["rmse"] for r in res[burn_in:]])
            rough = path_roughness(res, skip=burn_in)
            score = rmse + 0.1 * rough
            print(f"  {pen} strength {s:g}: val rmse {rmse:.5f} rough {rough:.5f}")
            if best is None or score < best[0]:
                best = (score, s)
        chosen[pen] = best[1]
        fitted[pen] = fit_sequential(slices, pen, strength=best[1],
                                     scale=scale, burn_in=burn_in)
        print(f"  chosen {pen} strength: {best[1]:g}")
    return fitted, chosen, burn_in, tune_end


def main():
    os.makedirs(REPORTS, exist_ok=True)
    curve = load_treasury_curve(START, END)
    slices = curve_slices(curve)
    n = len(slices)
    print(f"FRED Treasury CMT curve: {len(curve)} observations, {n} trading days, "
          f"{curve['date'].min().date()} to {curve['date'].max().date()}")
    print(f"maturities: {sorted(FRED_SERIES.values())}")
    print(f"{sum(s[4] for s in slices)} days missing at least one maturity\n")

    fitted, chosen, burn_in, tune_end = fit_everything(slices)
    frames = {k: to_frame(v) for k, v in fitted.items()}
    split = frames["ridge"].index[tune_end]

    print("\nfull sample (bp of yield):")
    for k, f in frames.items():
        d = f[["beta0", "beta1", "beta2"]].diff().abs().iloc[burn_in:]
        print(f"  {k:7s} rmse {f['rmse'].iloc[burn_in:].mean()*100:7.2f} bp   "
              f"mean |daily param change| {d.mean().mean()*100:7.2f} bp   "
              f"lambda sd {f['lam'].iloc[burn_in:].std():.3f}")

    print(f"\nuntouched evaluation period (from {split.date()}):")
    for k, f in frames.items():
        seg = f.iloc[tune_end:]
        d = f[["beta0", "beta1", "beta2"]].diff().abs().iloc[tune_end:]
        print(f"  {k:7s} rmse {seg['rmse'].mean()*100:7.2f} bp   "
              f"mean |daily param change| {d.mean().mean()*100:7.2f} bp")

    # does the level factor track the long yield, as it should?
    y10 = curve[curve["maturity_years"] == 10].set_index("date")["yield"]
    y10 = y10.reindex(frames["ridge"].index)
    spread = (curve[curve["maturity_years"] == 10].set_index("date")["yield"]
              - curve[curve["maturity_years"] == 0.25].set_index("date")["yield"]
              ).reindex(frames["ridge"].index)
    print("\nfactor interpretation:")
    for k, f in frames.items():
        print(f"  {k:7s} corr(beta0, 10y) {f['beta0'].corr(y10):+.4f}   "
              f"corr(-beta1, 10y-3m) {(-f['beta1']).corr(spread):+.4f}")

    styles = [("static", "0.7"), ("ridge", "C0"), ("l1", "C1")]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, p, lab in zip(axes, ["beta0", "beta1", "beta2", "lam"],
                          ["level (beta0)", "slope (beta1)", "curvature (beta2)",
                           "lambda (yrs)"]):
        for k, style in styles:
            ax.plot(frames[k].index, frames[k][p], style, label=k, lw=0.8)
        ax.set_ylabel(lab)
    axes[0].plot(y10.index, y10, "k--", lw=0.8, label="actual 10y")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title("Nelson-Siegel factor paths, US Treasury curve 2015-2025")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "parameter_paths.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(11, 4))
    for k, style in styles:
        ax.plot(frames[k].index, frames[k]["rmse"] * 100, style, label=k, lw=0.7)
    ax.axvline(split, color="k", ls="--", lw=0.8)
    ax.set_ylabel("bp")
    ax.set_title("daily cross-sectional fit RMSE; dashed = start of untouched eval")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "rmse.png"), dpi=120)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for k, style in styles:
        d = frames[k][["beta0", "beta1", "beta2"]].diff().abs().sum(axis=1) * 100
        axes[0].plot(d.index, d, style, label=k, lw=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("bp/day, log scale")
    axes[0].set_title("stability: total daily factor movement, static vs ridge vs L1")
    axes[0].legend()
    means = {k: frames[k][["beta0", "beta1", "beta2"]].diff().abs()
             .iloc[burn_in:].mean() * 100 for k in frames}
    idx = np.arange(3)
    for j, (k, style) in enumerate(styles):
        axes[1].bar(idx + j * 0.27, means[k].values, 0.27, label=k,
                    color=style if style != "0.7" else "0.7")
    axes[1].set_xticks(idx + 0.27)
    axes[1].set_xticklabels(["beta0", "beta1", "beta2"])
    axes[1].set_ylabel("mean |daily change|, bp")
    axes[1].set_title("mean daily factor change by method (lower = more stable)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "parameter_changes.png"), dpi=120)

    dates = pd.DatetimeIndex(frames["ridge"].index)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, target in zip(axes, SHOWCASE):
        i = int(dates.get_indexer([pd.Timestamp(target)], method="nearest")[0])
        d, tau, y, _, _ = slices[i]
        ax.plot(tau, y, "o", ms=5, color="k", label="observed CMT")
        grid = np.geomspace(tau.min(), tau.max(), 200)
        ax.plot(grid, _predict(fitted["ridge"][i]["phi"], grid), "C0", lw=1.5,
                label="NS fit (ridge)")
        ax.set_xscale("log")
        ax.set_xlabel("maturity (years)")
        ax.set_title(f"{d.date()}  rmse {fitted['ridge'][i]['rmse']*100:.1f} bp")
    axes[0].set_ylabel("yield (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("fitted vs actual: normal curve, 2020 zero floor, 2023 inversion")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "fitted_curves.png"), dpi=120)

    print(f"\ncharts saved to {REPORTS}/")


def run_brent():
    """Brent futures path. Real if ../source-material/brent_settles.csv exists,
    synthetic otherwise."""
    panel = load_panel()
    print("panel:", len(panel), "rows,",
          "SYNTHETIC" if panel["synthetic"].iloc[0] else "real CSV")
    slices = daily_slices(panel)
    fitted, chosen, burn_in, tune_end = fit_everything(slices)
    frames = {k: to_frame(v) for k, v in fitted.items()}
    for k, f in frames.items():
        print(f"  {k:7s} rmse {f['rmse'].iloc[burn_in:].mean():.5f} (log price)")
    return frames


if __name__ == "__main__":
    if "--brent" in sys.argv:
        run_brent()
    else:
        main()
