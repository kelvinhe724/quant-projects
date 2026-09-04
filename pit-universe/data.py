"""Point-in-time S&P 500 membership from Wikipedia, and prices for every name that was in it.

Two public sources. The "Historical components of the S&P 500" article carries a
table of dated additions and removals. Old revisions of "List of S&P 500
companies" carry the full member list as it stood on the revision date, which
anchors the walk so gaps in the change table cannot compound. Everything fetched
is cached under source-material/pit-universe/ so reruns are offline.
"""
import io
import json
import logging
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.normpath(os.path.join(HERE, "..", "source-material", "pit-universe"))
REPORTS = os.path.join(HERE, "reports")

API = "https://en.wikipedia.org/w/api.php"
LIST_PAGE = "List of S&P 500 companies"
CHANGES_PAGE = "Historical components of the S&P 500"
HEADERS = {"User-Agent": "pit-universe research script (kelvinhe724@uchicago.edu)"}

# Same span as the momentum project: warm-up from mid-2003, scoring from 2005.
PANEL_START = "2003-06-01"
PANEL_END = "2026-08-31"
BENCHMARK = "SPY"

# The first list revision with ticker symbols is from 2007-11-29. Two anchors a
# year after that; the change table alone is sparse before 2011.
SNAPSHOT_YEARS = range(2008, 2027)

# If a change row sits within this many days of an anchor, a disagreement between
# the walked set and the snapshot is an editing lag, and the change row wins.
ANCHOR_LAG_DAYS = 45

# Renames where the security name changed too, so name matching cannot pair them.
# old ticker -> ticker yfinance serves the same history under today.
RENAMES = {
    "FB": ("META", "2022-06-09"), "WLP": ("ANTM", "2014-12-03"), "ANTM": ("ELV", "2022-06-28"),
    "PCLN": ("BKNG", "2018-02-27"), "TSO": ("ANDV", "2017-08-01"), "MHFI": ("SPGI", "2016-04-28"),
    "LUK": ("JEF", "2018-05-24"), "KORS": ("CPRI", "2019-01-02"), "CBS": ("VIAC", "2019-12-05"),
    "VIAC": ("PARA", "2022-02-17"), "PARA": ("PSKY", "2025-08-07"), "HRS": ("LHX", "2019-07-01"),
    "SYMC": ("NLOK", "2019-11-05"), "NLOK": ("GEN", "2022-11-08"), "TMK": ("GL", "2019-08-08"),
    "HCP": ("PEAK", "2019-11-05"), "PEAK": ("DOC", "2024-03-04"), "BBT": ("TFC", "2019-12-09"),
    "UTX": ("RTX", "2020-04-03"), "IR": ("TT", "2020-03-02"), "CTL": ("LUMN", "2020-09-18"),
    "MYL": ("VTRS", "2020-11-17"), "COG": ("CTRA", "2021-10-01"), "LB": ("BBWI", "2021-08-03"),
    "PKI": ("RVTY", "2023-05-16"), "ABC": ("COR", "2023-08-30"), "WRK": ("SW", "2024-07-08"),
    "MWV": ("WRK", "2015-07-01"), "WAG": ("WBA", "2014-12-31"), "ZMH": ("ZBH", "2015-06-25"),
    "NU": ("ES", "2015-02-19"), "GCI": ("TGNA", "2015-06-29"), "FO": ("BEAM", "2011-10-04"),
    "MOT": ("MSI", "2011-01-04"), "WPI": ("ACT", "2013-01-24"), "ACT": ("AGN", "2015-06-15"),
    "WPO": ("GHC", "2013-11-29"), "COH": ("TPR", "2017-10-31"), "DLPH": ("APTV", "2017-12-05"),
    "BHGE": ("BKR", "2019-10-18"), "MXB": ("MSCI", "2010-05-08"), "FPL": ("NEE", "2010-05-24"),
    "EQR": ("VMRK", "2026-07-01"), "WMI": ("WM", "2008-11-14"), "ACE": ("CB", "2016-01-15"),
    "FI": ("FISV", "2025-12-23"), "DWDP": ("DD", "2019-06-03"),
    # the change table records these as a removal plus an addition, but Yahoo
    # serves the old company's history under the new symbol
    "KFT": ("MDLZ", "2012-10-02"), "DISCA": ("WBD", "2022-04-11"), "WLTW": ("WTW", "2022-01-10"),
    "PX": ("LIN", "2018-10-31"), "FLT": ("CPAY", "2024-03-25"), "ARNC": ("HWM", "2020-04-01"),
    "BHI": ("BKR", "2017-07-07"),
}

# Wikipedia spellings of the same symbol.
SPELLING = {"BRKB": "BRK-B", "BFB": "BF-B", "SGPPRB": "SGP"}

# Change rows the Wikipedia table lacks. Yahoo backfills GOOG (class C) with the
# class A history, so GOOG is treated as continuous and GOOGL as the new line.
EXTRA_CHANGES = [
    {"date": "2014-04-03", "add": "GOOGL", "rem": None, "add_name": "Google Class A"},
]


def fetch(url, params=None, path=None):
    """GET a URL, returning cached text when `path` already exists."""
    if path and os.path.exists(path):
        return open(path).read()
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(r.text)
        time.sleep(0.5)
    return r.text


def clean_ticker(raw):
    """Normalise a ticker cell to the form yfinance uses (BRK.B -> BRK-B)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = re.sub(r"\[.*?\]", "", str(raw)).strip().upper()
    s = s.split(":")[-1].strip()
    s = re.sub(r"[^A-Z0-9.\-]", "", s).replace(".", "-")
    return SPELLING.get(s, s) or None


def clean_name(raw):
    """Reduce a security name to a comparable key."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(raw).lower())
    drop = {"inc", "incorporated", "corp", "corporation", "co", "company", "companies",
            "ltd", "plc", "the", "group", "holdings", "holding", "class", "a", "b", "c",
            "common", "stock", "shares", "and", "of", "international", "intl", "limited",
            "sa", "nv", "ag", "llc", "lp", "trust"}
    return " ".join(w for w in s.split() if w not in drop)


def revision_before(date):
    """Return (revid, timestamp) of the last revision of the list page before `date`."""
    path = os.path.join(CACHE, "revisions.json")
    known = json.load(open(path)) if os.path.exists(path) else {}
    if date not in known:
        params = dict(action="query", prop="revisions", titles=LIST_PAGE, rvlimit=1,
                      rvdir="older", rvstart=f"{date}T00:00:00Z", rvprop="ids|timestamp",
                      format="json", formatversion=2)
        r = json.loads(fetch(API, params))
        rev = r["query"]["pages"][0]["revisions"][0]
        known[date] = [rev["revid"], rev["timestamp"][:10]]
        os.makedirs(CACHE, exist_ok=True)
        json.dump(known, open(path, "w"), indent=1)
    return known[date]


def parse_members(html):
    """Read the member table of one revision into ticker, name, sector."""
    t = max(pd.read_html(io.StringIO(html)), key=len)
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[0] for c in t.columns]
    low = {c: str(c).lower() for c in t.columns}
    tick = next(c for c in t.columns if "symbol" in low[c] or "ticker" in low[c])
    name = next(c for c in t.columns if low[c] in ("security", "company"))
    sect = next((c for c in t.columns if "gics sector" in low[c]), None)
    out = pd.DataFrame({
        "ticker": t[tick].map(clean_ticker),
        "name": t[name].astype(str).str.strip(),
        "sector": t[sect].astype(str).str.strip() if sect is not None else None,
    })
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def snapshots():
    """Return {date: members frame} for every anchor revision plus today's page."""
    out = {}
    for year in SNAPSHOT_YEARS:
        for day in (f"{year}-01-01", f"{year}-07-01"):
            revid, stamp = revision_before(day)
            html = fetch("https://en.wikipedia.org/w/index.php", {"oldid": revid},
                         os.path.join(CACHE, "revisions", f"rev_{revid}.html"))
            out[pd.Timestamp(stamp)] = parse_members(html)
    path = os.path.join(CACHE, "current_list.html")
    stamp_path = os.path.join(CACHE, "current_list_date.txt")
    if not os.path.exists(path):
        open(stamp_path, "w").write(pd.Timestamp.today().strftime("%Y-%m-%d"))
    html = fetch(f"https://en.wikipedia.org/wiki/{LIST_PAGE.replace(' ', '_').replace('&', '%26')}",
                 path=path)
    out[pd.Timestamp(open(stamp_path).read().strip())] = parse_members(html)
    return dict(sorted(out.items()))


def changes():
    """Read the dated addition/removal table into one row per change."""
    html = fetch(f"https://en.wikipedia.org/wiki/{CHANGES_PAGE.replace(' ', '_').replace('&', '%26')}",
                 path=os.path.join(CACHE, "historical_components.html"))
    t = max(pd.read_html(io.StringIO(html)), key=len)
    t.columns = ["date", "add", "add_name", "rem", "rem_name", "reason", "refs"][:len(t.columns)]
    t = t[["date", "add", "add_name", "rem", "rem_name"]].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["add"] = t["add"].map(clean_ticker)
    t["rem"] = t["rem"].map(clean_ticker)
    t = pd.concat([t, pd.DataFrame(EXTRA_CHANGES).assign(date=lambda d: pd.to_datetime(d["date"]))],
                  ignore_index=True)
    t = t.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return t.astype({"add": object, "rem": object}).where(t.notna(), None)


def canonical(ticker, date, renames):
    """Map a symbol as written on `date` to the symbol its history trades under now.

    renames: {old: (new, effective)}. A mention dated before `effective` refers
    to the old company and is mapped; a later mention is a reused symbol and
    stays as written. Chains (CBS -> VIAC -> PARA -> PSKY) are followed.
    """
    if not ticker:
        return None
    seen = set()
    while ticker in renames and ticker not in seen:
        new, effective = renames[ticker]
        if pd.Timestamp(date) >= pd.Timestamp(effective):
            break
        seen.add(ticker)
        ticker = new
    return ticker


def apply_renames(snaps, chg, renames):
    """Rewrite every symbol in the snapshots and change list to its canonical form."""
    snaps = {d: s.assign(ticker=s["ticker"].map(lambda t: canonical(t, d, renames)))
              .drop_duplicates("ticker") for d, s in snaps.items()}
    chg = chg.assign(add=[canonical(t, d, renames) for t, d in zip(chg["add"], chg["date"])],
                     rem=[canonical(t, d, renames) for t, d in zip(chg["rem"], chg["date"])])
    return snaps, chg


def reconcile(snaps, chg):
    """Find symbols that changed between snapshots without a change row.

    Returns (renames, unresolved). Renames pairs a vanished symbol with an
    appeared one when the security names match, dated at the snapshot where the
    new symbol first shows. Unresolved rows are changes the table missed; the
    panel snaps to the snapshot for those.
    """
    renames, rows = {}, []
    lag = pd.Timedelta(days=ANCHOR_LAG_DAYS)
    dates = list(snaps)
    for d0, d1 in zip(dates, dates[1:]):
        s0, s1 = snaps[d0].set_index("ticker"), snaps[d1].set_index("ticker")
        window = chg[(chg["date"] > d0 - lag) & (chg["date"] <= d1 + lag)]
        gone = [t for t in s0.index if t not in s1.index and t not in set(window["rem"].dropna())]
        new = [t for t in s1.index if t not in s0.index and t not in set(window["add"].dropna())]
        by_name = {clean_name(s1.loc[t, "name"]): t for t in new}
        for t in list(gone):
            key = clean_name(s0.loc[t, "name"])
            if key in by_name:
                renames[t] = (by_name[key], d1.strftime("%Y-%m-%d"))
                new.remove(by_name.pop(key))
                gone.remove(t)
        rows += [(d0, d1, "gone", t, s0.loc[t, "name"]) for t in gone]
        rows += [(d0, d1, "new", t, s1.loc[t, "name"]) for t in new]
    unresolved = pd.DataFrame(rows, columns=["from", "to", "kind", "ticker", "name"])
    return renames, unresolved


def membership_panel(chg, anchors, dates, lag_days=ANCHOR_LAG_DAYS):
    """Build the daily membership table from a change list and dated snapshots.

    chg: frame with date, add, rem. A change effective on d puts the added name
    in from d and takes the removed name out from d. anchors: {date: set}. The
    walk runs forward from each anchor and snaps to the next, except where a
    change row within lag_days of the anchor explains the disagreement. Before
    the first anchor it runs backward. Returns (panel, snaps) where snaps lists
    what each anchor corrected.
    """
    dates = pd.DatetimeIndex(dates)
    adds = chg.dropna(subset=["add"]).groupby("date")["add"].apply(set).to_dict()
    rems = chg.dropna(subset=["rem"]).groupby("date")["rem"].apply(set).to_dict()
    events = sorted(set(adds) | set(rems))
    anchor_dates = sorted(anchors)
    first = anchor_dates[0]
    lag = pd.Timedelta(days=lag_days)

    def nearby(sets, date):
        return set().union(*(v for k, v in sets.items() if abs(k - date) <= lag))

    state, snaps = {}, []
    cur = set(anchors[first])
    state[first] = frozenset(cur)
    for date in sorted(set(events) | set(anchor_dates)):
        if date <= first:
            continue
        if date in events:
            cur = (cur - rems.get(date, set())) | adds.get(date, set())
        if date in anchors:
            target = set(anchors[date])
            drop = (cur - target) - nearby(adds, date)
            put = (target - cur) - nearby(rems, date)
            snaps += [(date, t, "missed removal") for t in sorted(drop)]
            snaps += [(date, t, "missed addition") for t in sorted(put)]
            cur = (cur - drop) | put
        state[date] = frozenset(cur)

    cur = set(anchors[first])
    for date in sorted((d for d in events if dates[0] <= d < first), reverse=True):
        state[date] = frozenset(cur)
        cur = (cur - adds.get(date, set())) | rems.get(date, set())
    state.setdefault(dates[0], frozenset(cur))

    order = sorted(state)
    names = sorted(set().union(*state.values()))
    panel = pd.DataFrame(False, index=dates, columns=names)
    for start, nxt in zip(order, order[1:] + [None]):
        rows = (dates >= start) & ((dates < nxt) if nxt is not None else True)
        panel.loc[rows, list(state[start])] = True
    return panel, pd.DataFrame(snaps, columns=["anchor", "ticker", "kind"])


def build_panel():
    """Fetch, reconcile and walk; return the panel and everything used to build it."""
    snaps, chg = apply_renames(snapshots(), changes(), RENAMES)
    auto, unresolved = reconcile(snaps, chg)
    snaps, chg = apply_renames(snaps, chg, auto)
    anchors = {d: set(s["ticker"]) for d, s in snaps.items()}
    dates = pd.bdate_range(PANEL_START, PANEL_END)
    panel, corrections = membership_panel(chg, anchors, dates)
    renames = {**RENAMES, **auto}
    return dict(panel=panel, snapshots=snaps, changes=chg, renames=renames,
                unresolved=unresolved, corrections=corrections,
                today=set(snaps[max(snaps)]["ticker"]))


def intervals(panel):
    """Collapse a bool panel into one row per continuous membership spell."""
    rows = []
    for t in panel.columns:
        m = panel[t].to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            rows.append((t, panel.index[s], panel.index[e - 1]))
    return pd.DataFrame(rows, columns=["ticker", "start", "end"])


def sectors_at(snaps, date):
    """Sector labels from the most recent snapshot on or before `date`."""
    date = pd.Timestamp(date)
    prior = [d for d in snaps if d <= date and snaps[d]["sector"].notna().any()]
    s = snaps[max(prior)]
    return s.set_index("ticker")["sector"].rename("sector")


def download(tickers, start=PANEL_START, end=PANEL_END, batch=100):
    """Fetch adjusted closes and volumes for `tickers`, using the CSV cache if present."""
    px_path, vol_path = os.path.join(CACHE, "prices.csv"), os.path.join(CACHE, "volumes.csv")
    if os.path.exists(px_path) and os.path.exists(vol_path):
        read = dict(index_col=0, parse_dates=True)
        return pd.read_csv(px_path, **read), pd.read_csv(vol_path, **read)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    tickers = sorted(set(tickers) | {BENCHMARK})
    px, vol = [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                          auto_adjust=True, progress=False, group_by="column", threads=True)
        if raw.empty:
            continue
        px.append(raw["Close"].reindex(columns=chunk))
        vol.append(raw["Volume"].reindex(columns=chunk))
        print(f"  downloaded {min(i + batch, len(tickers))}/{len(tickers)}", flush=True)
    px = pd.concat(px, axis=1).sort_index().dropna(how="all")
    vol = pd.concat(vol, axis=1).sort_index().reindex(px.index)
    px = px.where(px > 0)
    px.to_csv(px_path)
    vol.to_csv(vol_path)
    return px, vol


def coverage(panel, px, today):
    """Per-ticker table of membership span, price span and whether the two overlap."""
    rows = []
    for t in panel.columns:
        m = panel[t]
        days = m[m]
        s = px[t] if t in px.columns else pd.Series(dtype=float)
        have = s.reindex(days.index).notna() if len(s) else pd.Series(False, index=days.index)
        rows.append({
            "ticker": t,
            "in_today": t in today,
            "first_member": days.index[0], "last_member": days.index[-1],
            "member_days": int(m.sum()),
            "first_price": s.first_valid_index(), "last_price": s.last_valid_index(),
            "priced_member_days": int(have.sum()),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out["price_share"] = out["priced_member_days"] / out["member_days"]
    return out


def load():
    """Return everything the comparisons need, building and caching on first call."""
    d = build_panel()
    px, vol = download(sorted(set(d["panel"].columns) | d["today"]))
    # trading days only: the walk used business days, which include holidays
    d["panel"] = d["panel"].reindex(px.index)
    d["px"], d["vol"] = px, vol
    d["coverage"] = coverage(d["panel"], d["px"], d["today"])
    return d


if __name__ == "__main__":
    d = load()
    panel, cov = d["panel"], d["coverage"]
    print(f"panel: {panel.shape[1]} names, {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"snapshots: {len(d['snapshots'])}, changes: {len(d['changes'])}, "
          f"renames: {len(d['renames'])}, unresolved: {len(d['unresolved'])}, "
          f"anchor corrections: {len(d['corrections'])}")
    removed = cov[~cov["in_today"]]
    print(f"names ever removed: {len(removed)}; with any price {int((removed['price_share'] > 0).sum())}; "
          f"with >=90% of member days priced {int((removed['price_share'] >= 0.9).sum())}")
