"""Starter feature sets. Price features use the close through t - 1; the lag is declared, not implied."""
import re

import numpy as np
import pandas as pd

from .store import Feature


def _close(raw):
    return raw.frames["close"]


def _ret(raw):
    return _close(raw).pct_change()


def momentum(raw, lookback, skip=21):
    px = _close(raw)
    return px.shift(skip) / px.shift(lookback) - 1


def realised_vol(raw, window):
    return np.log(_close(raw)).diff().rolling(window).std() * np.sqrt(252)


def dollar_volume(raw, window=21):
    return (_close(raw) * raw.frames["volume"]).rolling(window).mean()


def amihud(raw, window=21):
    dv = _close(raw) * raw.frames["volume"]
    return (_ret(raw).abs() / dv.where(dv > 0)).rolling(window).mean() * 1e6


def hl_range(raw, window=21):
    return ((raw.frames["high"] - raw.frames["low"]) / _close(raw)).rolling(window).mean()


PRICE = [
    Feature("ret_1", _ret, 1),
    Feature("ret_21", lambda r: _close(r).pct_change(21), 1),
    Feature("mom_3_1", lambda r: momentum(r, 63), 1),
    Feature("mom_6_1", lambda r: momentum(r, 126), 1),
    Feature("mom_12_1", lambda r: momentum(r, 252), 1),
    Feature("vol_21", lambda r: realised_vol(r, 21), 1),
    Feature("vol_63", lambda r: realised_vol(r, 63), 1),
    Feature("dollar_vol_21", dollar_volume, 1),
    Feature("amihud_21", amihud, 1),
    Feature("hl_range_21", hl_range, 1),
]


def macro(lags):
    """One feature per attached series, broadcast to every instrument; `lags` maps series name to its lag.

    The lag is the publication delay in sessions. A FRED monthly value is
    dated the first of its month and released weeks later, so declare that.
    """
    return [Feature(name, lambda r, n=name: r.series[n], lag) for name, lag in lags.items()]


def edgar_counts(words=("risk", "uncertain", "loss", "impairment", "going concern"), lag=1):
    """Fundamentals-lite: per-filing word counts, carried forward until the next filing.

    raw.filings is long: date, ticker, text. Each word becomes a feature
    holding the count in the ticker's latest filing on or before t - lag,
    plus the filing's length and the sessions since it. Dated by filing
    date, so lag=1 means a filing landing on t is usable from t + 1.
    """
    def counts(word):
        pat = re.compile(r"\b" + re.escape(word) + r"\b", re.I)

        def fn(raw):
            f = raw.filings
            if f.empty:
                return pd.DataFrame(index=raw.calendar, columns=raw.instruments, dtype=float)
            n = f["text"].map(lambda t: len(pat.findall(str(t))))
            wide = n.groupby([f["date"], f["ticker"]]).sum().unstack("ticker")
            return wide.ffill().reindex(raw.calendar, method="ffill")
        return fn

    def length(raw):
        f = raw.filings
        if f.empty:
            return pd.DataFrame(index=raw.calendar, columns=raw.instruments, dtype=float)
        n = f["text"].map(lambda t: len(str(t).split()))
        return n.groupby([f["date"], f["ticker"]]).sum().unstack("ticker").ffill().reindex(raw.calendar, method="ffill")

    def age(raw):
        f = raw.filings
        pos = pd.Series(np.arange(len(raw.calendar)), index=raw.calendar)
        out = pd.DataFrame(np.nan, index=raw.calendar, columns=raw.instruments)
        for tkr, g in f.groupby("ticker"):
            if tkr not in out.columns:
                continue
            last = pd.Series(1.0, index=pd.DatetimeIndex(g["date"])).groupby(level=0).sum()
            filed = pd.Series(pos.reindex(last.index, method="bfill").to_numpy(), index=last.index)
            filed = filed.reindex(raw.calendar, method="ffill")
            out[tkr] = pos - filed
        return out

    feats = [Feature(f"edgar_{w.replace(' ', '_')}", counts(w), lag) for w in words]
    return feats + [Feature("edgar_len", length, lag), Feature("edgar_age", age, lag)]
