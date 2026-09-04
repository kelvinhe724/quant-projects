"""Treasury curve, funding rate and ETF proxies for the rates carry sleeve.

The curve comes through the Nelson-Siegel project's FRED loader (constant
maturity yields, 1962 on). The funding leg is the 3-month CMT from 1982 and
the 3-month bill secondary rate before that. ETFs come through the framework's
yfinance loader so the sleeve can run in the same engine as the other sleeves.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from framework.engine import Bars, load_fred, load_yfinance  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ns_data = _module("ns_data", os.path.join(ROOT, "nelson-siegel", "data.py"))
ns = _module("ns", os.path.join(ROOT, "nelson-siegel", "ns.py"))

REPORTS = os.path.join(HERE, "reports")
CURVE_CACHE = os.path.join(REPORTS, "treasury_curve.csv")
START = "1962-01-01"
END = "2026-09-01"
ETF_START = "2002-07-01"

# Curve tenors in years, in FRED's order.
TENORS = np.array(sorted(ns_data.FRED_SERIES.values()))

# ETF -> the curve tenor it stands in for, and its effective duration in years
# (iShares fact sheets, mid-2025). run.py measures the empirical duration of
# each fund against its tenor's yield changes; check.py holds the two within
# tolerance of the tenor's par duration.
ETF_TENOR = {"SHY": 2.0, "IEI": 5.0, "IEF": 10.0, "TLH": 20.0, "TLT": 30.0}
ETF_DURATION = {"SHY": 1.85, "IEI": 4.3, "IEF": 7.2, "TLH": 12.6, "TLT": 15.9}
FUNDING = "funding"


def load_curve(refresh=False):
    """Daily long panel (date, maturity_years, yield) in percent, 1962 on."""
    # The NS project's loader caches into its own reports/ and drops any day
    # with fewer than six maturities, which is every day before 1969. Both are
    # module globals; repoint them for this project.
    ns_data.CACHE = CURVE_CACHE
    ns_data.MIN_MATURITIES = 4
    return ns_data.load_treasury_curve(START, END, refresh=refresh)


def curve_wide(curve):
    """Date x tenor table of yields, with the loader's forward fill undone across FRED's gaps.

    The NS loader forward-fills within a day's row. DGS20 was not published
    from 1987 to 1993 and DGS30 from 2002 to 2006; filled through, those would
    be a constant yield for years. Mask each tenor back to the raw series.
    """
    wide = curve.pivot(index="date", columns="maturity_years", values="yield").sort_index()
    # 1/12 does not survive the CSV round trip exactly
    wide.columns = [min(TENORS, key=lambda t: abs(t - c)) for c in wide.columns]
    for sid, tau in ns_data.FRED_SERIES.items():
        raw = load_fred(sid).reindex(wide.index)
        wide[tau] = wide[tau].where(raw.notna())
    return wide


def load_funding():
    """3-month funding rate in percent: DGS3MO from 1982, DTB3 before it."""
    cmt = load_fred("DGS3MO").dropna()
    bill = load_fred("DTB3").dropna()
    out = pd.concat([bill[bill.index < cmt.index[0]], cmt]).rename(FUNDING)
    return out[out.index >= START]


def load_etfs(start=ETF_START, end=END):
    """Bars of the ETF proxies; a ticker that will not download is dropped with a message."""
    names = []
    for t in ETF_TENOR:
        try:
            load_yfinance([t], start, end)
            names.append(t)
        except Exception as exc:
            print(f"{t} did not load ({exc}); running without it")
    return load_yfinance(names, start, end)


def month_ends(index):
    """Last date of each month in `index`; a trailing partial month is dropped.

    A sample ending on the 1st would otherwise make a one-day "month" whose
    annualised return is a few hundred percent, and that point sits in the
    regression at 365x weight.
    """
    days = pd.Series(index, index=index)
    ends = pd.DatetimeIndex(sorted(days.groupby([index.year, index.month]).max()))
    return ends[:-1] if ends[-1].day < 28 else ends


def build_bars(etfs, wide, funding):
    """Attach the curve and funding series to the ETF bars under tenor keys.

    Bars drops NaN and forward-fills, which would carry a stale yield across
    a FRED gap; a missing print is attached as inf so the strategy can see it.
    """
    series = {f"y{t:g}": wide[t].where(wide[t].notna(), np.inf) for t in wide.columns}
    series[FUNDING] = funding
    frames = {f: etfs.field(f) for f in ("open", "high", "low", "close", "volume")}
    return Bars(frames, series)


if __name__ == "__main__":
    curve = load_curve()
    wide = curve_wide(curve)
    print(f"curve: {wide.index[0].date()} to {wide.index[-1].date()}, {len(wide)} days")
    print(wide.notna().sum().rename("days with a print").to_string())
    print(pd.DataFrame({"first": wide.apply(lambda s: s.first_valid_index().date())}).T.to_string())
    f = load_funding()
    print(f"funding: {f.index[0].date()} to {f.index[-1].date()}, {len(f)} days")
    etfs = load_etfs()
    px = etfs.close
    print(f"etfs: {list(px.columns)}, {px.index[0].date()} to {px.index[-1].date()}")
    print(px.apply(lambda s: s.first_valid_index().date()).to_string())
