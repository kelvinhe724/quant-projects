"""Earnings 8-K events: reaction session, FinBERT score, causal ranks, features and the drift Alpha.

Timing convention. Every filing has an acceptance timestamp. Its reaction
session R is the session whose close-to-close return carries the news:
the same day when accepted before 09:00 ET, otherwise the next session.
A feature dated R holds the sentiment of the filing and the reaction
close(R) / close(R - 1) - 1, both known by the close of R, at lag 0. The
harness prices the strategy on p(t) = open(t + 1), so a position set on
row R is filled at the open after the reaction session and a horizon of
h sessions is open-to-open. Nothing dated R reads a price after close(R).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.alpha import Alpha
from research.features.store import Feature

ET = "America/New_York"
CUTOFF = "09:00"
RANK_WINDOW = 126
MIN_REF = 50
MAX_CHUNKS, CHUNK_TOKENS = 4, 510
SIGNALS = ("sent", "react", "resid")


def reaction_session(accepted, calendar):
    """Session whose close-to-close return carries each filing; NaT when it is past the calendar."""
    t = pd.DatetimeIndex(pd.to_datetime(accepted, utc=True)).tz_convert(ET)
    day = pd.DatetimeIndex(t.tz_localize(None).normalize())
    early = np.array([s < CUTOFF for s in t.strftime("%H:%M")])
    pos = np.where(early, calendar.searchsorted(day, side="left"), calendar.searchsorted(day, side="right"))
    ok = pos < len(calendar)
    vals = calendar.to_numpy()[np.minimum(pos, len(calendar) - 1)]
    return pd.DatetimeIndex(np.where(ok, vals, np.datetime64("NaT")))


def chunks(text, tokenizer, n=MAX_CHUNKS, size=CHUNK_TOKENS):
    """Token id windows over the first n * size tokens of a document."""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return [ids[i:i + size] for i in range(0, min(len(ids), n * size), size)]


def finbert_scorer(device=None, batch=32, name="ProsusAI/finbert"):
    """score(texts) -> P(positive) - P(negative), mean over each text's chunks. Model cached by HuggingFace."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval().to(device)
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    ipos, ineg = labels["positive"], labels["negative"]
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    def score(texts):
        pieces, owner = [], []
        for i, t in enumerate(texts):
            for c in chunks(t, tok):
                pieces.append([cls] + c + [sep])
                owner.append(i)
        out = np.zeros(len(pieces))
        with torch.no_grad():
            for b in range(0, len(pieces), batch):
                block = pieces[b:b + batch]
                width = max(len(x) for x in block)
                ids = torch.tensor([x + [pad] * (width - len(x)) for x in block], device=device)
                mask = (ids != pad).long()
                p = torch.softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1).cpu().numpy()
                out[b:b + batch] = p[:, ipos] - p[:, ineg]
        s = pd.Series(out).groupby(np.array(owner, dtype=int)).mean()
        return s.reindex(range(len(texts))).to_numpy()

    return score


def causal_ranks(ev, calendar, window=RANK_WINDOW, min_ref=MIN_REF):
    """Percentiles of sent, react and of sent residual to react, each against the trailing `window` sessions of events.

    ev has date (reaction session), sent, react. The reference set for an
    event is every event whose reaction session falls in the window ending
    at its own, itself included, so nothing later than its own close is used.
    The residual is from an OLS of sent on react fitted on that same set.
    """
    pos = calendar.get_indexer(pd.DatetimeIndex(ev["date"]))
    s, r = ev["sent"].to_numpy(float), ev["react"].to_numpy(float)
    order = np.argsort(pos, kind="stable")
    ps, ss, rs = pos[order], s[order], r[order]
    out = np.full((len(ev), 3), np.nan)

    def pct(ref, x):
        return ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref)

    for i in range(len(ev)):
        if pos[i] < 0:
            continue
        lo, hi = np.searchsorted(ps, pos[i] - window + 1, "left"), np.searchsorted(ps, pos[i], "right")
        a, b = ss[lo:hi], rs[lo:hi]
        both = ~(np.isnan(a) | np.isnan(b))
        if not np.isnan(s[i]) and (~np.isnan(a)).sum() >= min_ref:
            out[i, 0] = pct(a[~np.isnan(a)], s[i])
        if not np.isnan(r[i]) and (~np.isnan(b)).sum() >= min_ref:
            out[i, 1] = pct(b[~np.isnan(b)], r[i])
        if both.sum() >= min_ref and not np.isnan(s[i]) and not np.isnan(r[i]):
            beta, alpha = np.polyfit(b[both], a[both], 1)
            out[i, 2] = pct(a[both] - alpha - beta * b[both], s[i] - alpha - beta * r[i])
    return pd.DataFrame(out, columns=["sent_pct", "react_pct", "resid_pct"], index=ev.index)


def event_table(raw):
    """One row per (reaction session, ticker) from raw.filings with the reaction return attached."""
    f = raw.filings
    ev = f[f["ticker"].isin(raw.instruments)].copy()
    ev["date"] = pd.DatetimeIndex(ev["date"])
    ev = ev.sort_values("date", kind="stable").drop_duplicates(["date", "ticker"], keep="first")
    ret = raw.frames["close"].pct_change()
    ev["react"] = ret.stack().reindex(pd.MultiIndex.from_frame(ev[["date", "ticker"]])).to_numpy()
    if "sent" not in ev:
        ev["sent"] = np.nan
    ev = ev.reset_index(drop=True)
    return pd.concat([ev, causal_ranks(ev, raw.calendar)], axis=1)


def event_features(raw):
    """Wide dates x instruments frames: the three percentiles carried forward, and the sessions since the event."""
    ev = event_table(raw)
    cal, names = raw.calendar, raw.instruments
    out = {}
    for c in ("sent_pct", "react_pct", "resid_pct"):
        wide = ev.pivot(index="date", columns="ticker", values=c)
        out[c] = wide.reindex(index=cal, columns=names).ffill()
    pos = pd.Series(np.arange(len(cal), dtype=float), index=cal)
    filed = pd.Series(1.0, index=pd.MultiIndex.from_frame(ev[["date", "ticker"]])).unstack("ticker")
    filed = filed.mul(pos, axis=0).reindex(index=cal, columns=names).ffill()
    out["age"] = filed.rsub(pos, axis=0)
    return out


FEATURES = [Feature(c, (lambda r, c=c: event_features(r)[c]), 0) for c in ("sent_pct", "react_pct", "resid_pct", "age")]


class EventDrift(Alpha):
    """Long the top `tail` of a signal's percentile, short the bottom, over the `horizon` sessions after the reaction."""

    def __init__(self, signal="sent", horizon=10, tail=0.2, min_names=5):
        self.sig, self.horizon, self.tail, self.min_names = signal, horizon, tail, min_names
        self.name = f"EventDrift[{signal},{horizon}]"

    def signal(self, xs):
        s = xs[f"{self.sig}_pct"].where(xs["age"] < self.horizon).dropna()
        w = pd.Series(0.0, index=xs.index)
        long, short = s.index[s >= 1 - self.tail], s.index[s <= self.tail]
        if len(long) >= self.min_names:
            w[long] = 1.0 / len(long)
        if len(short) >= self.min_names:
            w[short] = -1.0 / len(short)
        return w


def forward_returns(ev, opens, spy_open, horizons, entry=1):
    """Excess open-to-open return from open(R + entry) to open(R + entry + h) for each event and h."""
    cal = opens.index
    pos = cal.get_indexer(pd.DatetimeIndex(ev["date"]))
    col = opens.columns.get_indexer(ev["ticker"])
    o = opens.to_numpy()
    m = spy_open.reindex(cal).to_numpy()
    out = {}
    for h in horizons:
        a, b = pos + entry, pos + entry + h
        ok = (pos >= 0) & (col >= 0) & (b < len(cal))
        a, b = np.minimum(a, len(cal) - 1), np.minimum(b, len(cal) - 1)
        r = o[b, col] / o[a, col] - 1 - (m[b] / m[a] - 1)
        out[h] = np.where(ok, r, np.nan)
    return pd.DataFrame(out, index=ev.index)


def ic_table(ev, fwd, signals=SIGNALS, min_month=20):
    """Spearman IC per signal and horizon, pooled and as a t from the monthly ICs."""
    rows = []
    month = pd.DatetimeIndex(ev["date"]).to_period("M")
    for sig in signals:
        x = ev[f"{sig}_pct"] if f"{sig}_pct" in ev else ev[sig]
        for h in fwd.columns:
            d = pd.DataFrame({"x": x, "y": fwd[h], "m": month}).dropna()
            pooled = spearmanr(d["x"], d["y"]).correlation if len(d) > 2 else np.nan
            ics = d.groupby("m").apply(lambda g: spearmanr(g["x"], g["y"]).correlation if len(g) >= min_month else np.nan,
                                       include_groups=False).dropna()
            t = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 1 else np.nan
            rows.append({"signal": sig, "horizon": h, "ic": pooled, "ic_monthly_mean": ics.mean(), "t": t,
                         "n_events": len(d), "n_months": len(ics)})
    return pd.DataFrame(rows)
