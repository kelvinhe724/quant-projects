"""Align quotes across venues, measure the gaps, and simulate taking them after fees."""
import itertools

import numpy as np
import pandas as pd

# Published taker fee at the lowest volume tier, in bps, as I read the fee pages in
# September 2026. Treat as approximate: schedules change and OKX in particular is
# far cheaper than the US venues.
TAKER_BPS = {"coinbase": 60.0, "kraken": 40.0, "binance_us": 40.0, "okx": 10.0}
PRO_BPS = {ex: 10.0 for ex in TAKER_BPS}

# Chain transfer assumptions for the "capital must move" case: seconds until an
# exchange credits a deposit, and the withdrawal fee in coin.
TRANSFER = {"BTC": dict(seconds=30 * 60, withdraw_fee=0.0002),
            "ETH": dict(seconds=5 * 60, withdraw_fee=0.001)}


def align(quotes, freq="3s", tolerance="6s"):
    """Put every venue's last-known quote on one clock.

    Only quotes stamped at or before each grid time are used, so a row never sees a
    quote that had not arrived yet. Rows where any venue is silent for longer than
    `tolerance` are dropped. Returns {symbol: frame} with columns (exchange, field).
    """
    quotes = quotes.sort_values("t_local")
    grid = pd.date_range(quotes["t_local"].min().ceil(freq), quotes["t_local"].max(), freq=freq)
    base = pd.DataFrame({"t": grid})
    out = {}
    for sym, qs in quotes.groupby("symbol"):
        cols = {}
        for ex, qe in qs.groupby("exchange"):
            m = pd.merge_asof(base, qe[["t_local", "bid", "ask"]], left_on="t", right_on="t_local",
                              direction="backward", tolerance=pd.Timedelta(tolerance))
            cols[(ex, "bid")] = m["bid"].to_numpy()
            cols[(ex, "ask")] = m["ask"].to_numpy()
            cols[(ex, "mid")] = ((m["bid"] + m["ask"]) / 2).to_numpy()
            cols[(ex, "age_s")] = (m["t"] - m["t_local"]).dt.total_seconds().to_numpy()
        panel = pd.DataFrame(cols, index=grid)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["exchange", "field"])
        out[sym] = panel.dropna()
    return out


def venues(panel):
    return sorted(panel.columns.get_level_values(0).unique())


def mid_spread_bps(panel):
    """Signed mid-to-mid spread for each unordered venue pair, in bps of the average mid."""
    out = {}
    for a, b in itertools.combinations(venues(panel), 2):
        ma, mb = panel[(a, "mid")], panel[(b, "mid")]
        out[f"{a}-{b}"] = (ma - mb) / ((ma + mb) / 2) * 1e4
    return pd.DataFrame(out)


def executable_gap_bps(panel):
    """Gap you could actually cross: sell at one venue's bid, buy at another's ask.

    Column "sell@a/buy@b" is (bid_a - ask_b) / ask_b in bps. Negative means the
    books overlap the normal way and there is nothing to take.
    """
    out = {}
    for a, b in itertools.permutations(venues(panel), 2):
        out[f"sell@{a}/buy@{b}"] = (panel[(a, "bid")] - panel[(b, "ask")]) / panel[(b, "ask")] * 1e4
    return pd.DataFrame(out)


def best_gap(panel):
    """Largest executable gap per sample and which pair it is."""
    g = executable_gap_bps(panel)
    return pd.DataFrame({"bps": g.max(axis=1), "pair": g.idxmax(axis=1)})


def runs(mask):
    """Lengths, in samples, of each unbroken run of True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.array([], dtype=int)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def gap_persistence(bps, thresholds, dt_s):
    """For each threshold: how often the best gap exceeds it and how long those episodes last."""
    rows = []
    for x in thresholds:
        r = runs(bps > x)
        rows.append(dict(threshold_bps=x, share_of_samples=(bps > x).mean(), episodes=len(r),
                         median_s=np.median(r) * dt_s if len(r) else 0.0,
                         p90_s=np.percentile(r, 90) * dt_s if len(r) else 0.0,
                         max_s=r.max() * dt_s if len(r) else 0.0))
    return pd.DataFrame(rows)


def parse_pair(name):
    sell, buy = name.split("/")
    return sell[len("sell@"):], buy[len("buy@"):]


def simulate(panel, fees_bps, notional=10_000.0, min_edge_bps=0.0, units=5, mode="prepositioned",
             transfer_s=0, withdraw_fee=0.0):
    """Trade the best executable gap whenever it beats both taker fees plus `min_edge_bps`.

    Signal on sample t, fills at sample t+1. `units` is the number of `notional`-sized
    lots of both cash and coin parked at every venue in prepositioned mode; in transfer
    mode each venue starts with cash only, coin bought at the cheap venue is withdrawn,
    and the sale happens at the first sample at least `transfer_s` after the buy.
    A lot that cannot be funded is counted as blocked, not traded.
    """
    gap = executable_gap_bps(panel)
    fee_total = {c: fees_bps[parse_pair(c)[0]] + fees_bps[parse_pair(c)[1]] for c in gap.columns}
    edge = gap - pd.Series(fee_total)
    times = panel.index
    cash = {v: units for v in venues(panel)}
    coin = {v: units if mode == "prepositioned" else 0 for v in venues(panel)}
    trades, blocked, unresolved = [], 0, 0

    for t in range(len(panel) - 1):
        col = edge.iloc[t].idxmax()
        if edge.iloc[t][col] <= min_edge_bps:
            continue
        sell_v, buy_v = parse_pair(col)
        fill = panel.iloc[t + 1]
        ask = fill[(buy_v, "ask")]
        qty = notional / ask
        cost = qty * ask * (1 + fees_bps[buy_v] / 1e4)

        if mode == "prepositioned":
            if cash[buy_v] < 1 or coin[sell_v] < 1:
                blocked += 1
                continue
            bid = fill[(sell_v, "bid")]
            sold_qty, sell_time = qty, times[t + 1]
            cash[buy_v] -= 1
            coin[buy_v] += 1
            coin[sell_v] -= 1
            cash[sell_v] += 1
        else:
            if cash[buy_v] < 1:
                blocked += 1
                continue
            arrive = times[t + 1] + pd.Timedelta(seconds=transfer_s)
            k = times.searchsorted(arrive)
            if k >= len(panel):
                unresolved += 1
                continue
            bid = panel.iloc[k][(sell_v, "bid")]
            sold_qty, sell_time = qty - withdraw_fee, times[k]
            cash[buy_v] -= 1
            cash[sell_v] += 1

        proceeds = sold_qty * bid * (1 - fees_bps[sell_v] / 1e4)
        trades.append(dict(signal_time=times[t], fill_time=times[t + 1], sell_time=sell_time,
                           sell=sell_v, buy=buy_v, signal_bps=gap.iloc[t][col],
                           gross_bps=(bid - ask) / ask * 1e4, net_usd=proceeds - cost,
                           net_bps=(proceeds - cost) / notional * 1e4))
    log = pd.DataFrame(trades)
    # the last sample has no t+1 to fill at, so it is never a signal
    summary = dict(mode=mode, signals=int((edge.iloc[:-1].max(axis=1) > min_edge_bps).sum()),
                   trades=len(log), blocked=blocked, unresolved=unresolved,
                   gross_bps_mean=log["gross_bps"].mean() if len(log) else np.nan,
                   net_usd_total=log["net_usd"].sum() if len(log) else 0.0,
                   net_bps_mean=log["net_bps"].mean() if len(log) else np.nan,
                   win_rate=(log["net_usd"] > 0).mean() if len(log) else np.nan)
    return log, summary


def close_gap_bps(hist):
    """Signed Coinbase-minus-Kraken close gap in bps on a shared candle clock."""
    a, b = hist["coinbase"]["close"], hist["kraken"]["close"]
    j = pd.concat([a, b], axis=1, keys=["coinbase", "kraken"]).dropna()
    return (j["coinbase"] - j["kraken"]) / j.mean(axis=1) * 1e4


def half_life(x):
    """AR(1) half-life in periods of a mean-reverting series; inf if it does not revert."""
    x = pd.Series(x).dropna()
    phi = np.corrcoef(x.iloc[1:], x.iloc[:-1])[0, 1]
    return np.inf if phi >= 1 or phi <= 0 else -np.log(2) / np.log(phi)
