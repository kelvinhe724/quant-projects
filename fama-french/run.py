"""Replicate the 3- and 5-factor models, then backtest a blended factor portfolio."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import FACTORS_3, FACTORS_5, STOCKS, excess, get_factors, get_stock_returns, get_test_portfolios
from factors import blend, fit_panel, grs, performance, premia

pd.set_option("display.width", 200)

SPLIT = pd.Period("2010-01", freq="M")

factors = get_factors()
ports = get_test_portfolios()
idx = factors.index.intersection(ports.index)
factors, ports = factors.loc[idx], ports.loc[idx]
ex_ports = ports.sub(factors["RF"], axis=0)

print(f"Sample: {idx[0]} to {idx[-1]}, {len(idx)} months")

models = {
    "CAPM": factors[["Mkt-RF"]],
    "FF3": factors[FACTORS_3],
    "FF5": factors[FACTORS_5],
    "FF5+MOM": factors[FACTORS_5 + ["Mom"]],
}

print("\n25 size/value portfolios, model comparison:")
rows = []
fits = {}
for name, X in models.items():
    table, resid = fit_panel(ex_ports, X)
    fits[name] = table
    g = grs(table["alpha"], resid, X)
    rows.append({"model": name, "k": X.shape[1], "mean_r2": round(table["r2"].mean(), 4),
                 "mean_abs_alpha": round(g["mean_abs_alpha"], 4),
                 "alphas_t>2": int((table["t_alpha"].abs() > 2).sum()),
                 "grs": round(g["grs"], 3), "grs_p": f"{g['p_value']:.2e}"})
print(pd.DataFrame(rows).to_string(index=False))

print("\nFF3 alphas across the size/value grid (monthly %, t-stat in brackets):")
grid = fits["FF3"]
for i, size in enumerate(["Small", "2", "3", "4", "Big"]):
    cells = grid.index[i * 5:(i + 1) * 5]
    print("  " + size.ljust(6) + " ".join(
        f"{grid.loc[c,'alpha']:6.2f}[{grid.loc[c,'t_alpha']:5.2f}]" for c in cells))

print("\nSame test on the original 1963-07 to 1991-12 window of Fama-French (1993):")
orig = factors.index <= pd.Period("1991-12", freq="M")
for name in ["CAPM", "FF3"]:
    X = models[name].loc[orig]
    t, e = fit_panel(ex_ports.loc[orig], X)
    g = grs(t["alpha"], e, X)
    print(f"  {name:5s} GRS {g['grs']:.3f} (p {g['p_value']:.4f}), mean |alpha| "
          f"{g['mean_abs_alpha']:.3f}, mean R2 {t['r2'].mean():.3f}, T={g['T']}")

print("\nSmall-growth vs small-value corner, FF3 vs FF5:")
for corner in ["SMALL LoBM", "SMALL HiBM", "BIG LoBM", "BIG HiBM"]:
    a3, a5 = fits["FF3"].loc[corner], fits["FF5"].loc[corner]
    print(f"  {corner:12s} FF3 alpha {a3['alpha']:6.3f} (t {a3['t_alpha']:5.2f}) "
          f"R2 {a3['r2']:.3f} | FF5 alpha {a5['alpha']:6.3f} (t {a5['t_alpha']:5.2f}) R2 {a5['r2']:.3f}")

stock_px = get_stock_returns(STOCKS)
ex_stocks = excess(stock_px, factors["RF"]).dropna()
print(f"\nIndividual stocks, {ex_stocks.index[0]} to {ex_stocks.index[-1]} "
      f"({len(ex_stocks)} months):")
stock_rows = []
for name, X in models.items():
    t, _ = fit_panel(ex_stocks, X.loc[ex_stocks.index])
    for s in t.index:
        stock_rows.append({"stock": s, "model": name, "alpha": t.loc[s, "alpha"],
                           "t_alpha": t.loc[s, "t_alpha"], "r2": t.loc[s, "r2"]})
sr = pd.DataFrame(stock_rows)
print(sr.pivot(index="stock", columns="model", values="r2")[list(models)].round(3).to_string())
print("\nFF5 loadings and alphas per stock:")
t5, resid5 = fit_panel(ex_stocks, factors[FACTORS_5].loc[ex_stocks.index])
print(t5[["alpha", "t_alpha"] + FACTORS_5 + ["r2"]].round(3).to_string())
g_stocks = grs(t5["alpha"], resid5, factors[FACTORS_5].loc[ex_stocks.index])
print(f"GRS on the 10 stocks (FF5): {g_stocks['grs']:.3f}, p = {g_stocks['p_value']:.4f}")

print(f"\nFactor premia, full sample vs the {SPLIT} split:")
prem = premia(factors[FACTORS_5 + ["Mom"]], split=SPLIT)
print(prem.round(3).to_string())

print("\nBlended factor portfolio (inverse-vol, 60-month lookback):")
mix = ["SMB", "HML", "RMW", "CMA", "Mom"]
bl = blend(factors, mix)
mkt = factors["Mkt-RF"].loc[bl.index]
half = blend(factors, ["Mkt-RF"] + mix)
comp = pd.DataFrame({
    "market (Mkt-RF)": performance(mkt),
    "blend (5 long-short)": performance(bl),
    "blend incl. market": performance(half),
}).T
print(comp.round(3).to_string())

print(f"\nSame comparison, {SPLIT} onward:")
late = bl.index >= SPLIT
comp_late = pd.DataFrame({
    "market (Mkt-RF)": performance(mkt[late]),
    "blend (5 long-short)": performance(bl[late]),
    "blend incl. market": performance(half[late]),
}).T
print(comp_late.round(3).to_string())

cum = (1 + factors[FACTORS_5 + ["Mom"]] / 100).cumprod()
fig, ax = plt.subplots(figsize=(11, 5))
cum.plot(ax=ax, lw=1)
ax.set_yscale("log")
ax.set_title("Cumulative factor returns, 1963-2026 (log scale)")
fig.tight_layout()
fig.savefig("reports/factor_cumulative.png", dpi=120)

fig, ax = plt.subplots(figsize=(9, 5))
for label, s in [("market", mkt), ("blend", bl), ("blend incl. market", half)]:
    ax.plot(s.index.to_timestamp(), (1 + s / 100).cumprod(), lw=1, label=label)
ax.set_yscale("log")
ax.legend()
ax.set_title("Blended factor portfolio vs the market")
fig.tight_layout()
fig.savefig("reports/blend_vs_market.png", dpi=120)

fig, ax = plt.subplots(figsize=(9, 5))
for label, s in [("market", mkt), ("blend", bl)]:
    c = (1 + s / 100).cumprod()
    ax.plot(c.index.to_timestamp(), (c / c.cummax() - 1) * 100, lw=0.9, label=label)
ax.legend()
ax.set_title("Drawdown (%)")
fig.tight_layout()
fig.savefig("reports/drawdown.png", dpi=120)

roll = factors[FACTORS_5 + ["Mom"]].rolling(120).mean() * 12
fig, ax = plt.subplots(figsize=(11, 5))
roll.dropna().plot(ax=ax, lw=1)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("10-year rolling annualized factor premium (%)")
fig.tight_layout()
fig.savefig("reports/rolling_premia.png", dpi=120)

alpha_grid = fits["FF5"]["alpha"].to_numpy().reshape(5, 5)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(alpha_grid, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(5), ["LoBM", "2", "3", "4", "HiBM"])
ax.set_yticks(range(5), ["Small", "2", "3", "4", "Big"])
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{alpha_grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("FF5 alpha, monthly %")
fig.tight_layout()
fig.savefig("reports/alpha_grid.png", dpi=120)

print("\nCharts in reports/: factor_cumulative.png, blend_vs_market.png, "
      "drawdown.png, rolling_premia.png, alpha_grid.png")
