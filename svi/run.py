"""Fit SVI to a cached or live SPY option chain and write the diagnostics to reports/.

Reruns use the newest cached chain so the numbers reproduce; pass --refresh to
pull a new snapshot.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import clean, load_or_fetch
from svi import (ARB_GRID, add_implied_vols, butterfly_check, calendar_check,
                 durrleman_g, fit_surface, market_butterfly_check,
                 market_calendar_check, params_table, raw_svi)

TICKER = "SPY"
ETA, LAM = 0.0, 0.01

raw, spot, snapshot = load_or_fetch(TICKER, refresh="--refresh" in sys.argv)
r = float(raw["r"].iloc[0])
q, fwd, counts = clean(raw)
q = add_implied_vols(q)

print(f"{TICKER} spot {spot:.2f}   snapshot {snapshot:%Y-%m-%d %H:%M} UTC   "
      f"bill rate {r:.2%}")
print("\nCleaning funnel:")
for k, v in counts.items():
    print(f"  {k:26s} {v:5d}")
print(f"  {'with an invertible vol':26s} {len(q):5d}")

print("\nForwards from put-call parity:")
print(fwd.assign(expiry=fwd["expiry"].dt.date).round(4).to_string(index=False))

d = q["iv_mid"] - q["impliedVolatility"]
print(f"\nMy implied vol vs the yfinance column, {len(q)} quotes: "
      f"mean {d.mean()*100:+.2f} vol pts, RMSE {np.sqrt((d**2).mean())*100:.2f}, "
      f"max |diff| {d.abs().max()*100:.2f}")

print("\nStatic arbitrage in the raw mid quotes, before any fitting:")
for T, g in q.groupby("T"):
    b = market_butterfly_check(g)
    print(f"  T={T:.3f}  {b['n_negative']:3d}/{b['n_butterflies']:3d} butterflies "
          f"quoted negative, worst {b['worst']:.4f}")
mcal = market_calendar_check(q)
print(f"  calendar: {sum(c['violates'] for c in mcal)}/{len(mcal)} adjacent pairs "
      "have total variance falling with maturity")

free = fit_surface(q, eta=ETA, enforce_calendar=False, enforce_butterfly=False)
fits = fit_surface(q, eta=ETA, lam=LAM, enforce_calendar=True, enforce_butterfly=True)


def arb_summary(f):
    nb = sum(butterfly_check(x["params"])["violates"] for x in f)
    nc = sum(c["violates"] for c in calendar_check(f))
    return nb, nc


nb, nc = arb_summary(free)
print(f"\nUnconstrained fit: {nb}/{len(free)} slices break the butterfly condition, "
      f"{nc}/{len(free)-1} adjacent pairs break calendar")
nb, nc = arb_summary(fits)
print(f"Constrained fit:   {nb}/{len(fits)} slices break the butterfly condition, "
      f"{nc}/{len(fits)-1} adjacent pairs break calendar")

table = params_table(fits)
free_table = params_table(free)
print("\nFitted SVI parameters (constrained, eta=%.1f lambda=%.2f):" % (ETA, LAM))
out = table.assign(expiry=table["expiry"].dt.date,
                   rmse_vol_pts=table["rmse_vol"] * 100).drop(columns="rmse_vol")
print(out.round(4).to_string(index=False))
print(f"\nRMSE in vol points, unconstrained vs constrained, by slice:")
for a, b in zip(free_table.itertuples(), table.itertuples()):
    print(f"  T={a.T:.3f}  {a.rmse_vol*100:5.3f}  ->  {b.rmse_vol*100:5.3f}")
print(f"  {'mean':9s} {free_table.rmse_vol.mean()*100:5.3f}  ->  "
      f"{table.rmse_vol.mean()*100:5.3f}")

print("\nBid-ask penalty sweep (lambda=%.2f), with and without inverse-spread "
      "weighting of the mid errors:" % LAM)
for wbs in (True, False):
    for eta in (0.0, 1.0, 10.0, 100.0):
        t = params_table(fit_surface(q, eta=eta, lam=LAM, weight_by_spread=wbs))
        print(f"  weighted={str(wbs):5s} eta={eta:6.1f}  RMSE {t.rmse_vol.mean()*100:.3f} "
              f"vol pts   {t.outside_band.sum():3d}/{t.n.sum()} fitted vols outside the band")

print("\nRidge sweep across the expiry ladder (eta=%.1f):" % ETA)
for lam in (0.0, 0.01, 0.1, 1.0):
    t = params_table(fit_surface(q, eta=ETA, lam=lam))
    churn = np.abs(np.diff(t[["a", "b", "rho", "m", "sigma"]].to_numpy(), axis=0)).sum()
    print(f"  lambda={lam:5.2f}  RMSE {t.rmse_vol.mean()*100:.3f} vol pts   "
          f"parameter churn {churn:.3f}")

n = len(fits)
fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.2 * ((n + 2) // 3)))
for ax, f in zip(axes.flat, fits):
    g = q[q["T"] == f["T"]].sort_values("k")
    ax.fill_between(g["k"], g["iv_bid"] * 100, g["iv_ask"] * 100,
                    color="0.8", label="bid-ask")
    ax.plot(g["k"], g["iv_mid"] * 100, ".", ms=4, color="k", label="mid")
    kk = np.linspace(g["k"].min(), g["k"].max(), 200)
    ax.plot(kk, np.sqrt(raw_svi(kk, f["params"]) / f["T"]) * 100, "r-", lw=1.2,
            label="SVI")
    ax.set_title(f"{f['expiry']:%Y-%m-%d}  T={f['T']:.3f}  "
                 f"RMSE {f['rmse_vol']*100:.2f} vp", fontsize=9)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("implied vol (%)")
for ax in axes.flat[n:]:
    ax.axis("off")
axes.flat[0].legend(fontsize=7)
fig.suptitle(f"{TICKER} SVI smiles, {snapshot:%Y-%m-%d}")
fig.tight_layout()
fig.savefig("reports/smiles.png", dpi=120)

fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(121, projection="3d")
kk = np.linspace(-0.35, 0.25, 60)
Ts = np.array([f["T"] for f in fits])
K, TT = np.meshgrid(kk, Ts)
Z = np.array([np.sqrt(np.maximum(raw_svi(kk, f["params"]), 1e-12) / f["T"]) * 100
              for f in fits])
ax.plot_surface(K, TT, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax.scatter(q["k"], q["T"], q["iv_mid"] * 100, s=3, color="k")
ax.set_xlabel("k")
ax.set_ylabel("T (years)")
ax.set_zlabel("implied vol (%)")
ax.set_title("Fitted surface and market mids")
ax2 = fig.add_subplot(122)
for f in fits:
    ax2.plot(ARB_GRID, raw_svi(ARB_GRID, f["params"]), lw=1,
             label=f"T={f['T']:.2f}")
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(0, None)
ax2.set_xlabel("log-moneyness k")
ax2.set_ylabel("total variance w")
ax2.set_title("Total variance slices (no crossings = no calendar arbitrage)")
ax2.legend(fontsize=7)
fig.tight_layout()
fig.savefig("reports/surface.png", dpi=120)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for f in fits:
    ax1.plot(f["k"], f["resid"] * 100, ".", ms=5, label=f"T={f['T']:.2f}")
ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel("log-moneyness k")
ax1.set_ylabel("fitted minus market (vol points)")
ax1.set_title("Fit residuals")
ax1.legend(fontsize=7)
for f in fits:
    ax2.plot(ARB_GRID, durrleman_g(ARB_GRID, f["params"]), lw=1, label=f"T={f['T']:.2f}")
ax2.axhline(0, color="r", lw=0.8, ls="--")
ax2.set_xlim(-1.0, 1.0)
ax2.set_xlabel("log-moneyness k")
ax2.set_ylabel("g(k)")
ax2.set_title("Durrleman function (negative anywhere = butterfly arbitrage)")
fig.tight_layout()
fig.savefig("reports/residuals.png", dpi=120)

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, name in zip(axes, ["a", "b", "rho", "m", "sigma"]):
    ax.plot(table["T"], table[name], "o-", label="constrained")
    ax.plot(free_table["T"], free_table[name], "x--", color="0.6", label="free")
    ax.set_title(name)
    ax.set_xlabel("T (years)")
axes[0].legend(fontsize=7)
fig.suptitle("SVI parameter term structure")
fig.tight_layout()
fig.savefig("reports/params.png", dpi=120)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(q["impliedVolatility"] * 100, q["iv_mid"] * 100, s=6, c=q["T"], cmap="viridis")
lims = [q["iv_mid"].min() * 100, q["iv_mid"].max() * 100]
ax.plot(lims, lims, "r-", lw=0.8)
ax.set_xlabel("yfinance impliedVolatility (%)")
ax.set_ylabel("my mid implied vol (%)")
ax.set_title("Implied vol: my Black-76 inversion vs the Yahoo column")
fig.tight_layout()
fig.savefig("reports/iv_comparison.png", dpi=120)

table.assign(expiry=table["expiry"].dt.date).to_csv("reports/svi_params.csv", index=False)
q.to_csv("reports/clean_quotes.csv", index=False)
print("\nCharts in reports/: smiles.png, surface.png, residuals.png, params.png, "
      "iv_comparison.png")
print("Tables in reports/: svi_params.csv, clean_quotes.csv")
