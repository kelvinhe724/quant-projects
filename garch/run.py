"""Fit GARCH(1,1) to four equity indices and write charts to reports/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import INDICES, get_prices, log_returns
from garch import fit_all, var_backtest

px = get_prices()
rets = log_returns(px)

print("Summary statistics (daily log returns x100):")
print(rets.describe().T[["count", "mean", "std", "min", "max"]].round(3))

table, fits = fit_all(rets)
print("\nGARCH(1,1) parameters:")
print(table.round(4).to_string(index=False))

table_t, fits_t = fit_all(rets, dist="t")
print("\nAIC/BIC, normal vs student-t:")
cmp = table[["index", "aic", "bic"]].merge(
    table_t[["index", "aic", "bic"]], on="index", suffixes=("_norm", "_t"))
print(cmp.round(1).to_string(index=False))

print("\n5% VaR backtest (student-t fits):")
for name, res in fits_t.items():
    vb = var_backtest(rets[name], res)
    print(f"  {name:10s} expected {vb['expected']:.1%}  observed {vb['observed']:.2%}  over {vb['n_days']} days")

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, name in zip(axes.flat, rets.columns):
    ax.plot(rets[name], lw=0.4)
    ax.set_title(f"{name} ({INDICES.get(name, '')})")
fig.suptitle("Daily log returns x100")
fig.tight_layout()
fig.savefig("reports/returns.png", dpi=120)

fig, ax = plt.subplots(figsize=(12, 5))
for name, res in fits.items():
    ax.plot(res.conditional_volatility, lw=0.7, label=name)
ax.set_title("GARCH(1,1) conditional volatility (daily, %)")
ax.legend()
fig.tight_layout()
fig.savefig("reports/conditional_vol.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(table["index"], table["persistence"])
ax.axhline(1.0, color="red", ls="--", lw=0.8)
ax.set_ylim(0.9, 1.005)
ax.set_title("Volatility persistence (alpha + beta)")
fig.tight_layout()
fig.savefig("reports/persistence.png", dpi=120)

print("\nCharts saved to reports/: returns.png, conditional_vol.png, persistence.png")
