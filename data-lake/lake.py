"""The desk's data layer: one Parquet store, one loader.

Every dataset is a directory of Parquet parts under store/. Parts are keyed by
symbol (equities, FRED) or by date (everything collected on a clock), so a load
touches only the files it needs. `load` is the only read path the other projects
use; `write` is the only write path the collectors use.

    import lake
    px = lake.load("equities_daily", "2020-01-01", "2020-12-31", universe="sp500")
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE = os.path.join(HERE, "store")
REPORTS = os.path.join(HERE, "reports")
INTERVALS = os.path.join(ROOT, "pit-universe", "reports", "membership_intervals.csv")
OPTIONS_DIR = os.path.join(ROOT, "options-collector")

# dataset -> (time column, key column, partition kind, expected cadence)
# partition "key": one part per key value; "date": one part per calendar day;
# "hour": store/<dataset>/<date>/<HH>.parquet; "month": one part per month.
DATASETS = {
    "equities_daily": ("date", "ticker", "key", "1B"),
    "fred": ("date", "series", "key", "1B"),
    "kalshi_markets": ("ts", "ticker", "date", "1h"),
    "kalshi_books": ("ts", "ticker", "date", "1h"),
    "edgar_8k": ("filed", "ticker", "date", "1B"),
    "crypto_l2": ("ts", "symbol", "hour", "1s"),
    "crypto_klines_1m": ("ts", "symbol", "month", "1min"),
    "crypto_bookdepth": ("ts", "symbol", "month", "1min"),
    "options": ("snapshot", "ticker", "external", "1B"),
}


def datasets():
    """Names the loader knows, in store order."""
    return list(DATASETS)


def part_path(dataset, part):
    """Absolute path of one part. Hour parts are '<date>/<HH>'."""
    return os.path.join(STORE, dataset, f"{part}.parquet")


def parts(dataset):
    """Sorted list of part names present on disk."""
    d = os.path.join(STORE, dataset)
    if not os.path.isdir(d):
        return []
    out = []
    for dirpath, _, files in os.walk(d):
        rel = os.path.relpath(dirpath, d)
        for f in files:
            if f.endswith(".parquet"):
                name = f[:-8] if rel == "." else f"{rel}/{f[:-8]}"
                out.append(name)
    return sorted(out)


def write(dataset, part, df, keys=None):
    """Merge df into one part: existing rows with the same keys are replaced. Returns rows written."""
    if dataset not in DATASETS:
        raise KeyError(dataset)
    tcol, kcol, _, _ = DATASETS[dataset]
    keys = keys or [tcol, kcol]
    path = part_path(dataset, part)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(df)


def _select(dataset, start, end, universe):
    """Parts worth opening for the request, by partition kind."""
    _, _, kind, _ = DATASETS[dataset]
    names = parts(dataset)
    if kind == "key" and isinstance(universe, (list, tuple, set)):
        names = [n for n in names if n in set(universe)]
    elif kind in ("date", "hour", "month"):
        lo = None if start is None else _naive(start)
        hi = None if end is None else _naive(end)

        def keep(n):
            day = n.split("/")[0] if kind != "month" else n + "-01"
            d = pd.Timestamp(day)
            if lo is not None and (d < lo.normalize() if kind != "month" else d < lo.normalize().replace(day=1)):
                return False
            if hi is not None and d > hi.normalize():
                return False
            return True
        names = [n for n in names if keep(n)]
    return names


def sp500_intervals(path=None):
    """Membership spells from the pit-universe project: ticker, start, end."""
    return pd.read_csv(path or INTERVALS, parse_dates=["start", "end"])


def in_universe(df, tcol, kcol, intervals):
    """Rows whose key was in the S&P 500 on their date, by the pit-universe intervals."""
    d = df.reset_index(drop=True)
    t = pd.to_datetime(d[tcol])
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    m = pd.DataFrame({"_i": d.index, "k": d[kcol], "t": t}).merge(intervals, left_on="k", right_on="ticker")
    hit = m.loc[(m["t"] >= m["start"]) & (m["t"] <= m["end"]), "_i"].unique()
    return d.loc[sorted(hit)]


def _load_options(start, end, universe):
    """Chains from the options collector, through its own loader."""
    sys.path.insert(0, OPTIONS_DIR)
    import data as opt  # noqa: E402
    frames = []
    for day in opt.list_days():
        if (start and day < str(pd.Timestamp(start).date())) or (end and day > str(pd.Timestamp(end).date())):
            continue
        chains, _ = opt.load_day(day)
        for t, df in chains.items():
            if universe is None or t in universe:
                frames.append(df.assign(ticker=t))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "snapshot"])


def _naive(bound):
    """Bound as a naive timestamp (tz-aware bounds become UTC wall time), for matching part names."""
    b = pd.Timestamp(bound)
    return b.tz_convert(None) if b.tz is not None else b


def _align(bound, tz):
    """Coerce a user bound to the column's tz-awareness (naive bounds are read as UTC)."""
    if bound is None:
        return None
    b = pd.Timestamp(bound)
    if tz is None:
        return b.tz_convert(None) if b.tz is not None else b
    return b.tz_localize("UTC") if b.tz is None else b.tz_convert("UTC")


def load(dataset, start=None, end=None, universe=None, as_of=None):
    """Return rows of `dataset` with start <= time <= end.

    universe: list of keys, or "sp500" for the point-in-time S&P 500 on each row's date.
    as_of: no row with a time after this instant is returned, whatever end says.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {datasets()}")
    tcol, kcol, kind, _ = DATASETS[dataset]
    # parts to open: the tighter of end and as_of (as naive so mixed tz-awareness cannot raise)
    sel_end = end if as_of is None else (as_of if end is None else min(_naive(end), _naive(as_of)))
    if kind == "external":
        df = _load_options(start, sel_end, None if universe == "sp500" else universe)
    else:
        names = _select(dataset, start, sel_end, universe)
        frames = [pd.read_parquet(part_path(dataset, n)) for n in names]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[tcol, kcol])
    if len(df) == 0:
        return df
    t = pd.to_datetime(df[tcol])
    lo, hi = _align(start, t.dt.tz), _align(end, t.dt.tz)
    if hi is not None and hi == hi.normalize() and tcol in ("ts", "snapshot"):
        hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    # as_of is an instant, never widened to the end of its day: no row after it, whatever end says
    cap = _align(as_of, t.dt.tz)
    if cap is not None and (hi is None or cap < hi):
        hi = cap
    mask = pd.Series(True, index=df.index)
    if lo is not None:
        mask &= t >= lo
    if hi is not None:
        mask &= t <= hi
    df = df[mask]
    if universe == "sp500":
        df = in_universe(df, tcol, kcol, sp500_intervals())
    elif universe is not None:
        df = df[df[kcol].isin(set(universe))]
    return df.reset_index(drop=True)


def gaps(times, expected, tolerance=1.5):
    """Intervals where consecutive sorted times are more than tolerance * expected apart.

    expected: pandas offset alias ('1s', '1h', '1B'). For '1B' the calendar is
    business days, so weekends are not gaps. Returns a frame with `after`,
    `before` and `missing` (count of expected points not seen).
    """
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(times)).dropna().unique()).sort_values()
    if len(t) < 2:
        return pd.DataFrame(columns=["after", "before", "missing"])
    if expected == "1B":
        full = pd.bdate_range(t[0], t[-1])
        missing = full.difference(t.normalize())
        rows, run = [], []
        for d in missing:
            if run and (d - run[-1]).days > 3:
                rows.append(run)
                run = []
            run.append(d)
        if run:
            rows.append(run)
        return pd.DataFrame([{"after": r[0] - pd.offsets.BDay(1), "before": r[-1] + pd.offsets.BDay(1),
                              "missing": len(r)} for r in rows])
    step = pd.Timedelta(expected)
    diff = pd.Series(t[1:] - t[:-1])
    hit = diff > tolerance * step
    return pd.DataFrame({"after": t[:-1][hit.to_numpy()], "before": t[1:][hit.to_numpy()],
                         "missing": (diff[hit] / step - 1).round().astype(int).to_numpy()})


def status(path=None):
    """The latest quality report, or None if it has not run."""
    p = path or os.path.join(REPORTS, "status.json")
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    for name in datasets():
        n = parts(name) if DATASETS[name][2] != "external" else []
        print(f"{name:18s} {len(n):5d} parts")
