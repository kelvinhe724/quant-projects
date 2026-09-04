"""Download football-data.co.uk match odds and reshape them into one quote per row.

Each source CSV is one league-season. The 1X2 odds appear twice: columns like
B365H are the price the site collected mid-week, columns like B365CH are the
price shortly before kickoff. This module calls those two phases open and close.
"""
import io
import os
import re

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
TIDY = os.path.join(HERE, "data", "quotes.csv")
BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "D1": "Germany Bundesliga",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
}

SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2010, 2026)]

BOOKS = {
    "B365": "Bet365", "BW": "Bwin", "IW": "Interwetten", "LB": "Ladbrokes",
    "PS": "Pinnacle", "WH": "William Hill", "VC": "VC Bet", "GB": "Gamebookers",
    "SB": "Sportingbet", "SJ": "Stan James", "BS": "Blue Square",
}

# Max is the best price quoted by any book the site tracks, Avg is their mean.
# Neither is a book you can bet with, so they are excluded from per-book stats.
AGGREGATES = {"Max": "best of all tracked books", "Avg": "mean of tracked books"}

# Seasons before 2019-20 label the aggregate columns with a BetBrain prefix, and
# Pinnacle appears as PS from 2012-13 but as PSC only from that season onward.
RENAME = {"BbMx": "Max", "BbAv": "Avg", "BbMxC": "MaxC", "BbAvC": "AvgC"}


def url(season, league):
    return f"{BASE}/{season}/{league}.csv"


def fetch(season, league, refresh=False):
    """Download one league-season CSV to the raw cache and return its path."""
    path = os.path.join(RAW, f"{season}_{league}.csv")
    if os.path.exists(path) and not refresh:
        return path
    os.makedirs(RAW, exist_ok=True)
    r = requests.get(url(season, league), timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def read_raw(path):
    """Parse one cached CSV, dropping the blank filler rows these files carry."""
    with open(path, "rb") as f:
        blob = f.read()
    df = pd.read_csv(io.BytesIO(blob), encoding="latin-1", on_bad_lines="skip")
    df.columns = [RENAME.get(c[:-1], c[:-1]) + c[-1] if c[:-1] in RENAME else c
                  for c in df.columns]
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df


def melt_quotes(df, season, league):
    """Turn the wide bookmaker columns into one row per match, book and phase."""
    keys = ["match_id", "league", "season", "date", "home", "away", "result"]
    df = df.assign(league=league, season=season,
                   home=df["HomeTeam"], away=df["AwayTeam"], result=df["FTR"])
    df["match_id"] = (season + "_" + league + "_" +
                      df.index.astype(str).str.zfill(4))
    rows = []
    for prefix in list(BOOKS) + list(AGGREGATES):
        for phase, tag in (("open", ""), ("close", "C")):
            cols = [f"{prefix}{tag}{o}" for o in "HDA"]
            if not all(c in df.columns for c in cols):
                continue
            block = df[keys + cols].copy()
            block.columns = keys + ["oh", "od", "oa"]
            block["book"] = prefix
            block["phase"] = phase
            rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["oh", "od", "oa"])


def build(leagues=None, seasons=None, refresh=False):
    """Download every league-season and return the tidy quote table."""
    leagues = leagues or list(LEAGUES)
    seasons = seasons or SEASONS
    frames, missing = [], []
    for season in seasons:
        for league in leagues:
            try:
                raw = read_raw(fetch(season, league, refresh))
            except Exception as exc:
                missing.append((season, league, str(exc)[:60]))
                continue
            frames.append(melt_quotes(raw, season, league))
    if missing:
        print(f"skipped {len(missing)} league-seasons: {missing}")
    quotes = pd.concat(frames, ignore_index=True)
    return sanity_filter(quotes)


def sanity_filter(q):
    """Drop quotes with impossible prices or an overround outside 0.5% to 40%."""
    ok = (q[["oh", "od", "oa"]] > 1.0).all(axis=1)
    book_sum = (1 / q[["oh", "od", "oa"]]).sum(axis=1)
    # Max is the best price across books and can legitimately sum below 1, which
    # is the arbitrage case, so it gets a floor of 0.90 rather than 1.005.
    floor = q["book"].map(lambda b: 0.90 if b == "Max" else 1.005)
    ok &= (book_sum >= floor) & (book_sum <= 1.40)
    return q[ok].reset_index(drop=True)


def load(refresh=False):
    """Return the tidy quote table, building and caching it on first call."""
    if os.path.exists(TIDY) and not refresh:
        return pd.read_csv(TIDY, parse_dates=["date"])
    q = build(refresh=refresh)
    os.makedirs(os.path.dirname(TIDY), exist_ok=True)
    q.to_csv(TIDY, index=False)
    return q


def pivot_books(q, phase, books):
    """Return a match-indexed frame of home/draw/away odds, one column set per book."""
    sub = q[(q["phase"] == phase) & (q["book"].isin(books))]
    wide = sub.pivot_table(index="match_id", columns="book",
                           values=["oh", "od", "oa"], aggfunc="first")
    wide.columns = [f"{b}_{c}" for c, b in wide.columns]
    return wide


def match_table(q):
    """One row per match with league, date and result, no odds."""
    cols = ["match_id", "league", "season", "date", "home", "away", "result"]
    return q[cols].drop_duplicates("match_id").set_index("match_id")


if __name__ == "__main__":
    quotes = load()
    print(quotes.shape)
    print(quotes.groupby(["league"])["match_id"].nunique())
    print(quotes.groupby(["book", "phase"]).size().unstack(fill_value=0))
