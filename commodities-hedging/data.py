"""Continuous front-month commodity futures and the contract specs that turn prices into dollars.

The project brief calls for thirteen specific December-2025 contracts from Bloomberg.
I do not have Bloomberg, so this pulls the front-month continuous series for the same
underlyings from Yahoo Finance. What that changes is written out in the README; the
short version is that the series carry roll returns and there is no single expiry.
"""
import os

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "source-material", "commodities")
PRICE_CACHE = os.path.join(CACHE_DIR, "continuous_close.csv")
VOLUME_CACHE = os.path.join(CACHE_DIR, "continuous_volume.csv")

# Bloomberg-style .csv drop-in. If a licensed export of the actual December-2025
# contracts ever arrives, put it here with the same column names as SPECS and set
# USE_BLOOMBERG; nothing downstream needs to change except the expiry handling,
# which the README describes.
BLOOMBERG_FILE = os.path.join(CACHE_DIR, "bloomberg_dec2025.csv")
USE_BLOOMBERG = False

START = "2015-01-02"
END = "2026-09-01"

# lot: exchange contract size. quote_conversion: dollars per one unit of quoted price
# (1.0 when the quote is already dollars, 0.01 when it is US cents). multiplier is the
# product, i.e. the dollar P&L of a 1.00 move in the quoted price.
# Sources: CME Group contract specifications for CL, NG, HO, RB, GC, SI, HG, PL, ZC,
# ZW, ZS; ICE Futures Europe for Brent; ICE Futures U.S. for Coffee C. Checked
# 2026-09-03. Ticks are in quoted price units, not dollars.
SPECS = {
    "CL=F": dict(name="WTI crude",      group="energy",   venue="NYMEX",
                 quote="USD/barrel",    lot=1_000,  quote_conversion=1.0,   tick=0.01),
    "BZ=F": dict(name="Brent crude",    group="energy",   venue="ICE Europe",
                 quote="USD/barrel",    lot=1_000,  quote_conversion=1.0,   tick=0.01),
    "NG=F": dict(name="Henry Hub gas",  group="energy",   venue="NYMEX",
                 quote="USD/MMBtu",     lot=10_000, quote_conversion=1.0,   tick=0.001),
    "HO=F": dict(name="NY Harbor ULSD", group="energy",   venue="NYMEX",
                 quote="USD/gallon",    lot=42_000, quote_conversion=1.0,   tick=0.0001),
    "RB=F": dict(name="RBOB gasoline",  group="energy",   venue="NYMEX",
                 quote="USD/gallon",    lot=42_000, quote_conversion=1.0,   tick=0.0001),
    "GC=F": dict(name="Gold",           group="metals",   venue="COMEX",
                 quote="USD/troy oz",   lot=100,    quote_conversion=1.0,   tick=0.10),
    "SI=F": dict(name="Silver",         group="metals",   venue="COMEX",
                 quote="USD/troy oz",   lot=5_000,  quote_conversion=1.0,   tick=0.005),
    "HG=F": dict(name="Copper",         group="metals",   venue="COMEX",
                 quote="USD/pound",     lot=25_000, quote_conversion=1.0,   tick=0.0005),
    "PL=F": dict(name="Platinum",       group="metals",   venue="NYMEX",
                 quote="USD/troy oz",   lot=50,     quote_conversion=1.0,   tick=0.10),
    "KC=F": dict(name="Coffee C",       group="agriculture", venue="ICE US",
                 quote="cents/pound",   lot=37_500, quote_conversion=0.01,  tick=0.05),
    "ZC=F": dict(name="Corn",           group="agriculture", venue="CBOT",
                 quote="cents/bushel",  lot=5_000,  quote_conversion=0.01,  tick=0.25),
    "ZW=F": dict(name="SRW wheat",      group="agriculture", venue="CBOT",
                 quote="cents/bushel",  lot=5_000,  quote_conversion=0.01,  tick=0.25),
    "ZS=F": dict(name="Soybeans",       group="agriculture", venue="CBOT",
                 quote="cents/bushel",  lot=5_000,  quote_conversion=0.01,  tick=0.25),
}

for _spec in SPECS.values():
    _spec["multiplier"] = _spec["lot"] * _spec["quote_conversion"]
    _spec["tick_value"] = _spec["tick"] * _spec["multiplier"]

TICKERS = list(SPECS)

TRAIN_END = "2023-12-29"
TEST_START = "2024-01-02"

# A settlement that repeats exactly is usually no trade rather than no news. Runs
# longer than this get flagged in the coverage table and the day is dropped from
# the return series for that contract.
MAX_STALE_RUN = 3


def download(start=START, end=END):
    """Fetch continuous front-month closes and volumes, using the CSV cache if present."""
    if USE_BLOOMBERG:
        px = pd.read_csv(BLOOMBERG_FILE, index_col=0, parse_dates=True).sort_index()
        return px, pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    if os.path.exists(PRICE_CACHE) and os.path.exists(VOLUME_CACHE):
        read = dict(index_col=0, parse_dates=True)
        return pd.read_csv(PRICE_CACHE, **read), pd.read_csv(VOLUME_CACHE, **read)
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column")
    if raw.empty:
        raise RuntimeError("yfinance returned nothing; no cache to fall back on")
    px, vol = raw["Close"][TICKERS].sort_index(), raw["Volume"][TICKERS].sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    px.to_csv(PRICE_CACHE)
    vol.to_csv(VOLUME_CACHE)
    return px, vol


def stale_runs(px):
    """Flag each price that is the 4th or later repeat of an unchanged settlement."""
    unchanged = px.diff() == 0
    run = unchanged.copy().astype(float)
    for col in px.columns:
        count = 0
        values = []
        for flag in unchanged[col]:
            count = count + 1 if flag else 0
            values.append(count)
        run[col] = values
    return run > MAX_STALE_RUN


def coverage(px, vol, stale):
    """Per-contract first date, last date, observation count and stale-print rate."""
    rows = []
    for t in px.columns:
        s = px[t]
        rows.append({
            "contract": SPECS[t]["name"],
            "first": s.first_valid_index().date(),
            "last": s.last_valid_index().date(),
            "obs": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "stale_flagged": int(stale[t].sum()),
            "median_volume": float(vol[t].median()) if vol[t].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows, index=px.columns)


def get_panel():
    """Return aligned prices, simple returns, per-contract dollar P&L and the coverage table.

    Prices are kept on the intersection of dates where every contract printed, which is
    the common-window design in section 3 of the brief. Stale settlement runs are set to
    missing before that intersection is taken.
    """
    px, vol = download()
    stale = stale_runs(px)
    clean = px.where(~stale).where(px > 0)
    common = clean.dropna(how="any")
    returns = common.pct_change().dropna()
    pnl = contract_pnl(common).loc[returns.index]
    return common, returns, pnl, coverage(px, vol, stale)


def contract_pnl(px):
    """Daily dollar P&L of one long contract, price change times the contract multiplier."""
    mult = pd.Series({t: SPECS[t]["multiplier"] for t in px.columns})
    return px.diff() * mult


def notional(px_row):
    """Dollar notional of one contract at the given prices."""
    return pd.Series({t: px_row[t] * SPECS[t]["multiplier"] for t in px_row.index})


def split(frame):
    """Chronological train/test split. The test window is never used to choose anything."""
    return frame.loc[:TRAIN_END], frame.loc[TEST_START:]


if __name__ == "__main__":
    px, rets, pnl, cov = get_panel()
    print(cov.to_string())
    print(f"\ncommon window: {px.index[0].date()} to {px.index[-1].date()}, {len(px)} days")
    tr, te = split(rets)
    print(f"train {tr.index[0].date()}-{tr.index[-1].date()} ({len(tr)}), "
          f"test {te.index[0].date()}-{te.index[-1].date()} ({len(te)})")
    print("\ncontract notionals on the last date:")
    print(notional(px.iloc[-1]).round(0).to_string())
