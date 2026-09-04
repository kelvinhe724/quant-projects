"""The loop: as-of view -> strategy -> risk overlay -> next-bar fill -> mark.

Each strategy runs in its own book with an equal share of capital, so
per-strategy attribution is exact and books never net against each other.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .execution import CostModel, Executor
from .metrics import TRADING_DAYS, by_year, drawdown, rolling_sharpe, summary
from .portfolio import Portfolio
from .risk import RiskConfig, RiskManager


@dataclass
class Config:
    capital: float = 1_000_000.0
    fill: str = "open"
    costs: CostModel = field(default_factory=CostModel)
    borrow_bps: float = 0.0
    financing: str = None
    financing_spread_bps: float = 0.0
    earn_on_cash: bool = True
    risk: RiskConfig = None
    allocations: dict = None


class Results:
    def __init__(self, books, trades, config):
        self.books = books
        self.config = config
        self.trades = pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame(
            columns=["date", "instrument", "quantity", "price", "fill", "notional", "commission",
                     "slippage", "slippage_bps", "participation", "capped", "sigma", "strategy"])
        self.equity_by_strategy = pd.DataFrame({k: b["equity"] for k, b in books.items()})
        self.equity = self.equity_by_strategy.sum(axis=1).rename("equity")
        self.returns = self.equity.pct_change().fillna(0.0).rename("return")
        self.returns_by_strategy = self.equity_by_strategy.pct_change().fillna(0.0)
        self.attribution = sum(b["pnl"].reindex(columns=self.instruments, fill_value=0.0)
                               for b in books.values())
        self.attribution_by_strategy = {k: b["pnl"] for k, b in books.items()}
        self.carry = sum(b["carry"] for b in books.values())
        self.weights = sum(b["weights"].reindex(columns=self.instruments, fill_value=0.0)
                           * (b["equity"] / self.equity).to_numpy()[:, None] for b in books.values())
        traded = self.trades.groupby("date")["notional"].sum() if len(self.trades) else pd.Series(dtype=float)
        self.turnover = (traded.reindex(self.equity.index).fillna(0.0) / self.equity).rename("turnover")
        self.metrics = summary(self.returns, turnover=self.turnover)
        if len(self.trades):
            self.metrics["total_cost"] = float(self.trades["commission"].sum() + self.trades["slippage"].sum())
            self.metrics["avg_slippage_bps"] = float(
                (self.trades["slippage_bps"] * self.trades["notional"]).sum() / self.trades["notional"].sum())
        self.metrics["avg_gross_exposure"] = float(self.weights.abs().sum(axis=1).mean())

    @property
    def instruments(self):
        return sorted({c for b in self.books.values() for c in b["pnl"].columns})

    def by_year(self):
        return by_year(self.returns)

    def report(self, out_dir):
        """Write CSV tables and PNG charts to out_dir."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        p = lambda name: os.path.join(out_dir, name)
        self.equity.to_frame().join(self.equity_by_strategy, rsuffix="_book").to_csv(p("equity.csv"))
        self.returns.to_csv(p("returns.csv"))
        self.trades.to_csv(p("trades.csv"), index=False)
        self.weights.to_csv(p("weights.csv"))
        self.attribution.to_csv(p("attribution_daily.csv"))
        years = len(self.returns) / TRADING_DAYS
        summ = pd.DataFrame({
            "pnl": self.attribution.sum(),
            "annual_pnl_pct": self.attribution.sum() / self.config.capital / years,
            "stand_alone_sharpe": self.attribution.mean() / self.attribution.std().replace(0, np.nan)
                                  * np.sqrt(TRADING_DAYS),
        }).sort_values("pnl", ascending=False)
        summ.to_csv(p("attribution.csv"))
        pd.Series(self.metrics).to_csv(p("metrics.csv"), header=False)
        self.by_year().to_csv(p("by_year.csv"))

        under, _ = drawdown(self.returns)
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
        axes[0].plot(self.equity / self.config.capital, color="black", linewidth=1.2, label="total")
        if len(self.books) > 1:
            for k in self.equity_by_strategy:
                axes[0].plot(self.equity_by_strategy[k] / self.equity_by_strategy[k].iloc[0], linewidth=0.9,
                             label=k)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("growth of 1 (log)")
        axes[0].legend(fontsize=8)
        axes[1].fill_between(under.index, under, 0, color="firebrick", alpha=0.5)
        axes[1].set_ylabel("drawdown")
        axes[2].plot(rolling_sharpe(self.returns, TRADING_DAYS), linewidth=1.0)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_ylabel("rolling 1y Sharpe")
        fig.tight_layout()
        fig.savefig(p("equity.png"), dpi=140)
        plt.close(fig)

        yrs = self.by_year()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(yrs.index.astype(str), yrs["return"], color=["steelblue" if v >= 0 else "firebrick"
                                                            for v in yrs["return"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title("Return by year")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        fig.tight_layout()
        fig.savefig(p("by_year.png"), dpi=140)
        plt.close(fig)
        return out_dir


def run(strategies, bars, start=None, end=None, config=None):
    """Backtest one or more strategies on `bars` between start and end inclusive."""
    from .strategy import Strategy

    if isinstance(strategies, Strategy):
        strategies = [strategies]
    config = config or Config()
    cal = bars.calendar
    start = cal[0] if start is None else pd.Timestamp(start)
    end = bars.asof if end is None else pd.Timestamp(end)
    days = cal[(cal >= start) & (cal <= end)]
    if len(days) == 0:
        raise ValueError("no bars in range")
    rate = bars.series(config.financing) if config.financing else None
    executor = Executor(config.costs, config.fill)
    alloc = config.allocations or {str(s): 1 / len(strategies) for s in strategies}

    books, all_trades = {}, []
    for strat in strategies:
        key = str(strat)
        book = Portfolio(config.capital * alloc[key], config.borrow_bps,
                         config.financing_spread_bps, config.earn_on_cash)
        risk = RiskManager(config.risk)
        pending, weights_path = None, {}
        for day in days:
            view = bars.upto(day)
            if pending:
                trades, residual = executor.execute(day, view, pending, book, key)
                all_trades.extend(trades)
                pending = residual or None
            book.accrue(day, float(rate.loc[day]) if rate is not None else None)
            book.mark(day, view.close.loc[day])
            weights_path[day] = book.weights()
            targets = strat.on_bar(day, view)
            if targets is not None:
                targets = {n: float(w) for n, w in targets.items() if n in view.instruments}
                pending = risk.apply(targets, view, book.equity_history)
            else:
                fix = risk.drift(book.weights(), view, book.equity_history)
                if fix is not None:
                    pending = fix
        equity, pnl, carry = book.frames()
        books[key] = {"equity": equity, "pnl": pnl, "carry": carry,
                      "weights": pd.DataFrame(weights_path).T.reindex(equity.index).fillna(0.0),
                      "risk": risk, "pending": pending}
    return Results(books, all_trades, config)
