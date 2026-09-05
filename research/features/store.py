"""Point-in-time feature store. Every feature declares the lag it is published at.

Raw data is a set of wide frames on one calendar (close, volume, ...), any
number of daily series (FRED) and an optional long frame of filings. A
Feature maps Raw to a dates x instruments frame; the store then shifts it
by the feature's lag, so the row dated t holds a value computed from data
through t - lag. That shift is the only place a lag is applied.

The shift cannot stop a feature function from reading rows after its own
date, so audit() perturbs one raw row and recomputes: every feature row
dated before the perturbation plus the lag must be byte-identical. A
feature that fails is a peek, and build(audit=True) refuses it.
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


class PeekError(Exception):
    """A feature row changed when only later raw data was perturbed."""


class Raw:
    """Frames (field -> dates x instruments), daily series, and filings on one calendar."""

    def __init__(self, frames, series=None, filings=None):
        idx = frames["close"].index
        self.frames = {k: v.reindex(idx) for k, v in frames.items()}
        self.series = {k: v.dropna().sort_index().reindex(idx, method="ffill") for k, v in (series or {}).items()}
        self.filings = filings if filings is not None else pd.DataFrame(columns=["date", "ticker", "text"])

    @property
    def calendar(self):
        return self.frames["close"].index

    @property
    def instruments(self):
        return list(self.frames["close"].columns)

    def upto(self, date):
        """Everything dated at or before `date`."""
        date = pd.Timestamp(date)
        frames = {k: v.loc[:date] for k, v in self.frames.items()}
        series = {k: v.loc[:date] for k, v in self.series.items()}
        return Raw(frames, series, self.filings[self.filings["date"] <= date])

    def perturbed(self, date):
        """A copy with every raw observation dated `date` changed, for the peek audit."""
        frames = {k: v.copy() for k, v in self.frames.items()}
        for v in frames.values():
            v.loc[date] = v.loc[date].fillna(1.0) * 1.5 + 1.0
        series = {k: v.copy() for k, v in self.series.items()}
        for v in series.values():
            v.loc[date] = (v.loc[date] if pd.notna(v.loc[date]) else 0.0) + 1.0
        rows = pd.DataFrame({"date": [date] * len(self.instruments), "ticker": self.instruments,
                             "text": ["risk uncertain loss impairment going concern"] * len(self.instruments)})
        return Raw(frames, series, pd.concat([self.filings, rows], ignore_index=True))

    @classmethod
    def from_bars(cls, bars, series=None):
        """From a framework Bars view; the view's own cut is the as-of date."""
        frames = {f: bars.field(f) for f in FIELDS}
        series = {k: bars.series(k) for k in (bars.series_names if series is None else series)}
        return cls(frames, series)

    @classmethod
    def from_lake(cls, load, instruments, start, end, fred=(), filings=False):
        """Build from the data-lake loader, written against the plan's one-line contract.

        `load(dataset, start, end)` returns a DataFrame. Datasets assumed:
        "equities/daily" long with date, ticker, open, high, low, close,
        volume; "fred/<id>" with date, value; "edgar/filings" long with
        date, ticker, text. The lake did not exist when this was written,
        so check.py stubs `load` and run.py reads the framework's panel.
        """
        eq = load("equities/daily", start, end)
        eq = eq[eq["ticker"].isin(instruments)]
        frames = {f: eq.pivot(index="date", columns="ticker", values=f).sort_index().reindex(columns=instruments)
                  for f in FIELDS}
        series = {}
        for sid in fred:
            s = load(f"fred/{sid}", start, end)
            series[sid] = pd.Series(s["value"].to_numpy(), index=pd.DatetimeIndex(s["date"]), name=sid)
        fil = load("edgar/filings", start, end) if filings else None
        return cls(frames, series, fil)


@dataclass(frozen=True)
class Feature:
    """name, fn(Raw) -> frame (dates x instruments) or series (dates), lag in sessions."""
    name: str
    fn: callable
    lag: int

    def __post_init__(self):
        if not isinstance(self.lag, (int, np.integer)) or self.lag < 0:
            raise ValueError(f"{self.name}: lag must be a non-negative integer, got {self.lag!r}")

    def compute(self, raw):
        out = self.fn(raw)
        if isinstance(out, pd.Series):
            out = pd.DataFrame({k: out for k in raw.instruments})
        out = out.reindex(index=raw.calendar, columns=raw.instruments).astype(float)
        return out.shift(self.lag)


class FeatureStore:
    def __init__(self, features):
        self.features = {}
        for f in features:
            if f.name in self.features:
                raise ValueError(f"duplicate feature {f.name}")
            self.features[f.name] = f

    @property
    def names(self):
        return list(self.features)

    @property
    def lags(self):
        return {k: f.lag for k, f in self.features.items()}

    def build(self, raw, audit=False, at=None):
        """dates x (feature, instrument) frame. audit=True runs the peek audit first."""
        if audit:
            self.audit(raw, at)
        out = {f.name: f.compute(raw) for f in self.features.values()}
        return pd.concat(out, axis=1, names=["feature", "instrument"])

    def audit(self, raw, at=None):
        """Perturb the raw row dated `at` (default: three quarters through) and check nothing earlier moves.

        For a feature with lag L, rows dated before `at` + L sessions may
        not depend on the row at `at`. Anything that does is a peek.
        """
        cal = raw.calendar
        pos = len(cal) * 3 // 4 if at is None else cal.get_loc(pd.Timestamp(at))
        bumped = raw.perturbed(cal[pos])
        bad = []
        for f in self.features.values():
            a, b = f.compute(raw), f.compute(bumped)
            stop = min(pos + f.lag, len(cal))
            if not a.iloc[:stop].equals(b.iloc[:stop]):
                first = a.iloc[:stop].ne(b.iloc[:stop]).any(axis=1).idxmax()
                bad.append(f"{f.name} (lag {f.lag}) at {first.date()} sees {cal[pos].date()}")
        if bad:
            raise PeekError("; ".join(bad))
        return True


def cross_section(panel, date):
    """instruments x features at one date, from a build() panel."""
    return panel.loc[pd.Timestamp(date)].unstack("feature")


def long_panel(panel):
    """(date, instrument) x features."""
    return panel.stack("instrument", future_stack=True).sort_index()


def panel_hash(frame):
    """Short content fingerprint of a frame: values, index and columns."""
    h = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    h.update(",".join(map(str, frame.columns)).encode())
    return h.hexdigest()[:16]
