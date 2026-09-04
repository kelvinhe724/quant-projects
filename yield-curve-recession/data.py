"""Monthly term spreads, the NBER recession indicator and unemployment from FRED,
cached under source-material/yield-curve-recession/ so every rerun is offline.

The 10y-3m spread on FRED (T10Y3M) starts in 1982. To reach 1962 the monthly
series is built from GS10 (10-year constant maturity, monthly average) and TB3MS
(3-month bill, secondary market, monthly average), which is the pair Estrella and
Mishkin used. The daily constant-maturity series are kept for the 2022-24 chart.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "source-material", "yield-curve-recession")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

MONTHLY = ["GS10", "GS2", "TB3MS", "USREC", "UNRATE", "SAHMREALTIME", "SAHMCURRENT"]
DAILY = ["DGS10", "DGS2", "DGS3MO", "T10Y3M", "T10Y2Y"]

START = "1962-01-01"
HORIZON = 12


def fetch(series_id):
    """Return one FRED series as a float Series, downloading only if uncached."""
    path = os.path.join(CACHE_DIR, f"{series_id}.csv")
    if not os.path.exists(path):
        import certifi
        import requests

        response = requests.get(FRED.format(series_id), verify=certifi.where(), timeout=60)
        response.raise_for_status()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(response.content)
    raw = pd.read_csv(path, na_values=".", index_col=0, parse_dates=True)
    return raw.iloc[:, 0].astype(float).rename(series_id)


def sahm(unrate):
    """Real-time Sahm gap: 3-month average unemployment minus its low over the prior 12 months.

    The rule fires at 0.5. Computed from the final-vintage UNRATE, so it is
    the Sahm rule as it looks today, not as it printed at the time.
    """
    # BLS skipped the October 2025 release during the shutdown; one interpolated
    # month keeps the rolling windows alive; limit_area="inside" stops pandas from
    # carrying the last print forward into months BLS has not published yet
    avg3 = unrate.interpolate(limit=1, limit_area="inside").rolling(3).mean()
    return avg3 - avg3.shift(1).rolling(12).min()


def recession_ahead(usrec, horizon=HORIZON):
    """1 if any of months t+1 .. t+horizon is an NBER recession month, NaN when the window is not fully observed."""
    fwd = pd.concat([usrec.shift(-k) for k in range(1, horizon + 1)], axis=1)
    label = fwd.max(axis=1)
    label[fwd.isna().any(axis=1)] = float("nan")
    return label.rename("recession_ahead")


def monthly():
    """Monthly panel from 1962: spreads, recession label, Sahm gap."""
    raw = pd.concat([fetch(s) for s in MONTHLY], axis=1, sort=True)
    raw.index = raw.index.to_period("M").to_timestamp()
    raw = raw.loc[START:]
    out = pd.DataFrame(index=raw.index)
    out["spread_10y3m"] = raw["GS10"] - raw["TB3MS"]
    out["spread_10y2y"] = raw["GS10"] - raw["GS2"]
    out["usrec"] = raw["USREC"]
    out["unrate"] = raw["UNRATE"]
    out["sahm"] = sahm(raw["UNRATE"])
    out["sahm_fred_realtime"] = raw["SAHMREALTIME"]
    out["recession_ahead"] = recession_ahead(raw["USREC"])
    return out


def daily():
    """Daily constant-maturity yields and spreads, for the recent-episode chart."""
    return pd.concat([fetch(s) for s in DAILY], axis=1, sort=True)


if __name__ == "__main__":
    m = monthly()
    print(f"{m.index[0].date()} to {m.index[-1].date()}, {len(m)} months")
    print(f"recession months: {int(m['usrec'].sum())}, "
          f"label rate: {m['recession_ahead'].mean():.3f}")
    print(m.tail(15).round(2).to_string())
