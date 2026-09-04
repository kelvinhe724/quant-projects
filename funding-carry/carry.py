"""Funding statistics, the perp-spot basis, and the delta-neutral carry simulator."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365 * 24 / 8

# Binance USD-M retail fee schedule, no BNB discount, no VIP tier.
SPOT_FEE_BPS = 10.0
PERP_FEE_BPS = 5.0
SLIP_BPS = 1.0
MAINTENANCE_MARGIN = 0.005

# Funding formula constants: the interest component is 0.01% per 8h and the
# premium is only allowed to pull the rate 0.05% away from it.
INTEREST = 0.0001
PREMIUM_CLAMP = 0.0005
RATE_CAP = 0.0075


def annualise(rate, hours=8.0):
    """Scale one funding payment to a simple annual rate."""
    return rate * (365 * 24 / hours)


def yearly_funding(panel):
    """Annualised funding, hit rate and dispersion, one row per calendar year."""
    rows = []
    for year, g in panel.groupby(panel.index.year):
        r = g["rate"]
        n = len(r)
        rows.append({
            "periods": n,
            "mean_8h": r.mean(),
            "annualised_simple": annualise(r.mean()),
            "annualised_compound": (1 + r).prod() ** (PERIODS_PER_YEAR / n) - 1,
            "pct_negative": (r < 0).mean(),
            "min_8h": r.min(),
            "max_8h": r.max(),
        })
    return pd.DataFrame(rows, index=sorted(panel.index.year.unique()))


def autocorrelation(rate, lags=(1, 3, 21, 90)):
    """Autocorrelation of the funding series at the given period lags."""
    return pd.Series({lag: rate.autocorr(lag) for lag in lags}, name="autocorr")


def basis(panel):
    """Perp premium over spot at the settlement snapshot, in fraction of spot."""
    return panel["perp"] / panel["spot"] - 1.0


def implied_funding(prem):
    """Binance's funding formula applied to an observed premium."""
    return np.clip(prem + np.clip(INTEREST - prem, -PREMIUM_CLAMP, PREMIUM_CLAMP),
                   -RATE_CAP, RATE_CAP)


def drivers(panel, window=90):
    """Funding alongside trailing spot momentum and realised vol, 90 periods = 30d."""
    ret = np.log(panel["spot"]).diff()
    return pd.DataFrame({
        "rate": panel["rate"],
        "momentum": ret.rolling(window).sum(),
        "vol": ret.rolling(window).std() * np.sqrt(PERIODS_PER_YEAR),
    }).dropna()


def simulate(panel, margin_frac=0.5, rebalance_every=90, topup_at=0.4,
             mmr=MAINTENANCE_MARGIN, mark="spot", spot_fee_bps=SPOT_FEE_BPS,
             perp_fee_bps=PERP_FEE_BPS, slip_bps=SLIP_BPS, hold=None):
    """Run long spot / short perp on one panel and return the per-period book.

    Capital is 1. It splits into spot notional and perp margin so that margin is
    `margin_frac` of the spot leg. Both legs are resized to current equity every
    `rebalance_every` periods, and sooner if the margin account has burned through
    `topup_at` of the buffer it started the cycle with. Liquidation is tested
    against the high inside each period, not its close.

    Binance liquidates on a mark price built from a spot index, so `mark="spot"`
    uses the spot high. `mark="perp"` triggers on the perp's own traded high,
    which during a squeeze wicks far above the index and gives a worst case.
    """
    spot = panel["spot"].to_numpy()
    perp = panel["perp"].to_numpy()
    high = panel[f"{mark}_high"].to_numpy()
    rate = panel["rate"].to_numpy()
    n = len(panel)
    want = np.ones(n, bool) if hold is None else np.asarray(hold, bool)

    spot_cost = (spot_fee_bps + slip_bps) / 1e4
    perp_cost = (perp_fee_bps + slip_bps) / 1e4

    equity = np.ones(n)
    qty = np.zeros(n)
    funding_pnl = np.zeros(n)
    price_pnl = np.zeros(n)
    fee_paid = np.zeros(n)
    margin_ratio = np.full(n, np.nan)

    eq = 1.0
    held = 0.0
    posted = 0.0
    perp_ref = np.nan
    banked = 0.0
    liquidated_at = None
    since_rebalance = 0

    def resize(i, target_qty):
        nonlocal held, posted, perp_ref, banked, eq
        traded = abs(target_qty - held)
        fee = traded * spot[i] * spot_cost + traded * perp[i] * perp_cost
        eq -= fee
        held = target_qty
        posted = eq - held * spot[i]
        perp_ref = perp[i]
        banked = 0.0
        return fee

    for i in range(n):
        fee = 0.0
        if i > 0 and held > 0:
            price = held * ((spot[i] - spot[i - 1]) - (perp[i] - perp[i - 1]))
            fund = rate[i] * held * perp[i]
            eq += price + fund
            banked += fund
            price_pnl[i] = price
            funding_pnl[i] = fund

            margin_equity = posted + held * (perp_ref - high[i]) + banked
            margin_ratio[i] = margin_equity / (held * high[i])
            if margin_equity < topup_at * posted:
                since_rebalance = rebalance_every
            if margin_equity < mmr * held * high[i] and liquidated_at is None:
                # Short is force-closed at the liquidation price and the margin
                # account is wiped. The spot leg is now unhedged and gets sold at
                # the period close, so whatever the price gave back is a real loss.
                eq = held * spot[i]
                fee = held * spot[i] * spot_cost
                eq -= fee
                held, posted, banked = 0.0, 0.0, 0.0
                liquidated_at = panel.index[i]
                want = want.copy()
                want[i:] = False

        if liquidated_at is None:
            since_rebalance += 1
            if want[i] and held == 0.0:
                fee += resize(i, (eq / (1 + margin_frac)) / spot[i])
                since_rebalance = 0
            elif not want[i] and held > 0:
                fee += resize(i, 0.0)
                since_rebalance = 0
            elif held > 0 and since_rebalance >= rebalance_every:
                fee += resize(i, (eq / (1 + margin_frac)) / spot[i])
                since_rebalance = 0

        equity[i] = eq
        qty[i] = held
        fee_paid[i] = fee

    if held > 0:
        closing = held * spot[-1] * spot_cost + held * perp[-1] * perp_cost
        eq -= closing
        equity[-1] = eq
        fee_paid[-1] += closing

    book = pd.DataFrame({"equity": equity, "qty": qty, "funding": funding_pnl,
                         "price": price_pnl, "fees": fee_paid,
                         "margin_ratio": margin_ratio}, index=panel.index)
    book["ret"] = book["equity"].pct_change().fillna(0.0)
    book.attrs["liquidated_at"] = liquidated_at
    return book


def metrics(book):
    """Summarise a book into the usual performance table plus tail statistics."""
    r = book["ret"]
    equity = book["equity"]
    years = len(r) / PERIODS_PER_YEAR
    sd = r.std()
    out = {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "annual_vol": sd * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": r.mean() / sd * np.sqrt(PERIODS_PER_YEAR) if sd else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "funding_collected": book["funding"].sum(),
        "price_pnl": book["price"].sum(),
        "fees_paid": book["fees"].sum(),
        "skew": r.skew(),
        "excess_kurtosis": r.kurtosis(),
        "worst_8h": r.min(),
        "q01": r.quantile(0.01),
        "normal_q01": r.mean() + sd * -2.326,
        "liquidated_at": book.attrs.get("liquidated_at"),
    }
    when = out["liquidated_at"]
    out["equity_at_liquidation"] = equity.loc[when] if when is not None else np.nan
    return out


def funding_filter(panel, window=21, threshold=0.0):
    """Hold only while the trailing mean funding rate is above `threshold`.

    The mean is computed through the previous period and shifted, so the decision
    at t uses nothing from t.
    """
    return (panel["rate"].rolling(window).mean().shift(1) > threshold).fillna(False)
