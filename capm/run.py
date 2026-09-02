"""Download prices, fit CAPM per stock, write table and charts to reports/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

from data import build_excess_returns
from capm import fit_all, rolling_beta

# one liquid name per GICS sector
TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "PG", "CAT", "AMZN", "NEE", "DIS", "LIN"]
WINDOW = 60

os.makedirs("reports", exist_ok=True)

excess, market = build_excess_returns(TICKERS)
print(f"{len(excess)} trading days, {excess.index[0].date()} to {excess.index[-1].date()}")

table = fit_all(excess, market)
table = table[["ticker", "alpha_annual", "beta", "r2", "alpha_pvalue"]]
print("\nCAPM results (sorted by beta):")
print(table.round(4).to_string(index=False))
table.to_csv("reports/capm_table.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
for ax, name in zip(axes.flat, TICKERS):
    df = pd.concat([excess[name], market], axis=1).dropna()
    ax.scatter(df.iloc[:, 1], df.iloc[:, 0], s=3, alpha=0.4)
    fit = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    xs = pd.Series([df.iloc[:, 1].min(), df.iloc[:, 1].max()])
    ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, color="red", lw=1)
    ax.set_title(f"{name}  beta={fit.params.iloc[1]:.2f}")
fig.suptitle("Daily excess returns vs market, with fitted CAPM line")
fig.tight_layout()
fig.savefig("reports/scatter_grid.png", dpi=120)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
for ax, name in zip(axes.flat, TICKERS):
    df = pd.concat([excess[name], market], axis=1).dropna()
    fit = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    ax.plot(fit.resid, lw=0.4)
    ax.axhline(0, color="red", lw=0.6)
    ax.set_title(name)
fig.suptitle("CAPM regression residuals")
fig.tight_layout()
fig.savefig("reports/residuals.png", dpi=120)

fig, ax = plt.subplots(figsize=(12, 6))
for name in TICKERS:
    ax.plot(rolling_beta(excess[name], market, WINDOW), lw=0.8, label=name)
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.set_title(f"{WINDOW}-day rolling beta")
ax.legend(ncol=5, fontsize=8)
fig.tight_layout()
fig.savefig("reports/rolling_beta.png", dpi=120)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(table["ticker"], table["beta"])
ax.axhline(1.0, color="red", ls="--", lw=0.8)
ax.set_title("Full-sample beta by stock")
fig.tight_layout()
fig.savefig("reports/beta_bars.png", dpi=120)

print("\nCharts saved to reports/: scatter_grid.png, residuals.png, rolling_beta.png, beta_bars.png")
print("Table saved to reports/capm_table.csv")
