"""Two curve sources.

Default is the US Treasury constant-maturity yield curve from FRED. The second
is a Brent futures panel, real if ../source-material/brent_settles.csv exists
and synthetic otherwise; the synthetic panel also backs check.py.
"""
import os
import numpy as np
import pandas as pd

FRED_SERIES = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
    "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0,
    "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}
MIN_MATURITIES = 6
CACHE = os.path.join(os.path.dirname(__file__), "reports", "treasury_curve.csv")


def load_treasury_curve(start="2015-01-01", end="2025-12-31", refresh=False):
    """Daily panel of (date, maturity_years, yield), CMT yields in percent."""
    if os.path.exists(CACHE) and not refresh:
        df = pd.read_csv(CACHE, parse_dates=["date"])
        sel = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sel):
            return sel.reset_index(drop=True)

    from pandas_datareader import data as pdr
    wide = pdr.DataReader(list(FRED_SERIES), "fred", start, end)
    # drop days FRED never published (federal holidays come back as NaN rows)
    # before filling, otherwise ffill clones the previous day's curve into them
    wide = wide[wide.count(axis=1) >= MIN_MATURITIES].ffill()
    long = wide.stack().rename("yield").reset_index()
    long.columns = ["date", "series", "yield"]
    long["maturity_years"] = long["series"].map(FRED_SERIES)
    counts = long.groupby("date")["yield"].count()
    keep = counts[counts >= MIN_MATURITIES].index
    long = long[long["date"].isin(keep)]
    out = long[["date", "maturity_years", "yield"]].sort_values(
        ["date", "maturity_years"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def curve_slices(curve):
    """Curve rows regrouped into the (date, tau, y, labels, incomplete) tuples
    the NS fitters consume."""
    full = len(FRED_SERIES)
    out = []
    for d, g in curve.groupby("date"):
        g = g.sort_values("maturity_years")
        out.append((d, g["maturity_years"].to_numpy(), g["yield"].to_numpy(),
                    [f"{m}y" for m in g["maturity_years"]], len(g) < full))
    return out

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

# COH6..CON7. SCOPE=13 keeps the curve through COH7 (Mar 2027); 17 = all of them.
TICKERS = ["COH6", "COJ6", "COK6", "COM6", "CON6", "COQ6", "COU6", "COV6",
           "COX6", "COZ6", "COF7", "COG7", "COH7", "COJ7", "COK7", "COM7", "CON7"]
SCOPE = 13

DAYS_PER_MONTH = 30.4375


def contract_month(ticker):
    code, digit = ticker[2], ticker[3]
    year = 2020 + int(digit)
    return pd.Timestamp(year, MONTH_CODES[code], 1)


def last_trading_date(ticker):
    # ICE rule: trading ceases on the last business day of the second month
    # before the contract month. The Christmas/New Year tweaks are ignored.
    cm = contract_month(ticker)
    end_of_prev = cm - pd.offsets.MonthBegin(2) + pd.offsets.MonthEnd(0)
    return end_of_prev if end_of_prev.dayofweek < 5 else end_of_prev - pd.offsets.BDay(1)


def contract_table(scope=SCOPE):
    rows = []
    for t in TICKERS[:scope]:
        rows.append({"ticker": t, "contract_month": contract_month(t),
                     "last_trade": last_trading_date(t)})
    return pd.DataFrame(rows)


def synthetic_panel(scope=SCOPE, start="2026-01-02", end="2026-06-01", seed=42):
    """Synthetic settle panel: an NS curve whose parameters random-walk, drifting
    from contango into backwardation over the sample."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    contracts = contract_table(scope)

    b0, b1, b2, lam = 4.35, -0.06, 0.02, 6.0   # log-price scale; ~77 USD long end
    rows = []
    for i, d in enumerate(dates):
        b0 += rng.normal(0, 0.004)
        b1 += rng.normal(0, 0.003) + 0.0012      # slow flip toward backwardation
        b2 += rng.normal(0, 0.002)
        lam = float(np.clip(lam * np.exp(rng.normal(0, 0.02)), 2, 18))
        for _, c in contracts.iterrows():
            if d > c["last_trade"]:
                continue                          # expired: no price, no forward fill
            tau = (c["contract_month"] - d).days / DAYS_PER_MONTH
            x = tau / lam
            f = (1 - np.exp(-x)) / x
            logp = b0 + b1 * f + b2 * (f - np.exp(-x)) + rng.normal(0, 0.0015)
            rows.append({"date": d, "ticker": c["ticker"], "price": np.exp(logp),
                         "contract_month": c["contract_month"],
                         "last_trade": c["last_trade"]})
    return pd.DataFrame(rows)


def load_panel(scope=SCOPE):
    csv = os.path.join(os.path.dirname(__file__), "..", "source-material", "brent_settles.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, parse_dates=["date"])
        ct = contract_table(17).set_index("ticker")
        df["contract_month"] = df["ticker"].map(ct["contract_month"])
        df["last_trade"] = df["ticker"].map(ct["last_trade"])
        df = df[df["date"] <= df["last_trade"]]
        df["synthetic"] = False
        return df
    df = synthetic_panel(scope)
    df["synthetic"] = True
    return df


def daily_slices(panel, k=12, buffer_days=3):
    """Per date, the nearest k contracts still trading with buffer_days of room
    before last trade. Returns (date, tau_months, log_prices, tickers,
    incomplete) tuples."""
    out = []
    for d, g in panel.groupby("date"):
        g = g[g["last_trade"] >= d + pd.offsets.BDay(buffer_days)].copy()
        g["tau"] = (g["contract_month"] - d).dt.days / DAYS_PER_MONTH
        g = g.sort_values("tau").head(k)
        out.append((d, g["tau"].to_numpy(), np.log(g["price"].to_numpy()),
                    list(g["ticker"]), len(g) < k))
    return out
