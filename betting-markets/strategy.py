"""Betting rules, Kelly staking and cross-book arbitrage detection.

Every simulation here returns per-bet profit in units of stake, so the mean is
the return on turnover. A flat-staked bettor with no information loses the
bookmaker's margin, and that is the number every rule has to beat.
"""
import numpy as np
import pandas as pd

from odds import implied


def profit(odds, won):
    """Profit per unit staked: odds minus one if the bet wins, minus one if it loses."""
    return np.where(np.asarray(won, dtype=bool), np.asarray(odds) - 1.0, -1.0)


def kelly_fraction(p, odds, cap=1.0):
    """Fraction of bankroll Kelly stakes given probability p at decimal odds.

    Negative values mean no bet rather than a short, since you cannot lay at the
    back price.
    """
    b = np.asarray(odds) - 1.0
    f = (np.asarray(p) * (b + 1.0) - 1.0) / b
    return np.clip(f, 0.0, cap)


def kelly_growth(f, odds, won):
    """Log growth of a bankroll staking fraction f on each bet in sequence."""
    return np.log1p(f * profit(odds, won))


def pick_column(odds, rule):
    """Choose one of the three outcomes per match. Returns column indices."""
    o = np.asarray(odds, dtype=float)
    if rule == "favourite":
        return o.argmin(axis=1)
    if rule == "longshot":
        return o.argmax(axis=1)
    if rule == "draw":
        return np.full(len(o), 1)
    if rule == "home":
        return np.zeros(len(o), dtype=int)
    raise ValueError(rule)


def flat_bet(odds, y, cols):
    """Stake one unit on the chosen outcome of every match."""
    rows = np.arange(len(cols))
    o = np.asarray(odds, dtype=float)[rows, cols]
    won = np.asarray(y)[rows, cols] > 0
    return o, won, profit(o, won)


def summarise(pnl, label, stake=None):
    """Return on turnover with a t-statistic against zero."""
    pnl = np.asarray(pnl, dtype=float)
    stake = np.ones_like(pnl) if stake is None else np.asarray(stake, dtype=float)
    turnover = stake.sum()
    if len(pnl) == 0 or turnover == 0:
        return {"rule": label, "bets": 0}
    se = pnl.std(ddof=1) / np.sqrt(len(pnl)) if len(pnl) > 1 else np.nan
    return {
        "rule": label,
        "bets": int(len(pnl)),
        "turnover": float(turnover),
        "roi": float(pnl.sum() / turnover),
        "total": float(pnl.sum()),
        "t": float(pnl.mean() / se) if se and se > 0 else np.nan,
    }


def fair_odds(p):
    """Odds a zero-margin book would quote for probabilities p."""
    return 1.0 / np.asarray(p, dtype=float)


def value_bets(p_true, odds, threshold=0.0):
    """Mask of bets where the model probability beats the break-even probability."""
    edge = np.asarray(p_true) * np.asarray(odds) - 1.0
    return edge > threshold, edge


def arbitrage(best_odds):
    """Detect matches where the best price on each outcome sums below one.

    Returns the book sum and the guaranteed return on total stake, which is
    1/booksum - 1 when the stakes are split to pay the same amount whichever
    outcome lands.
    """
    s = implied(best_odds).sum(axis=-1)
    return s, np.where(s < 1.0, 1.0 / s - 1.0, 0.0)


def best_across_books(frames):
    """Element-wise maximum odds over a list of (n, 3) arrays, ignoring gaps."""
    stack = np.stack([np.asarray(f, dtype=float) for f in frames])
    return np.nanmax(stack, axis=0)


def disagreement_bets(book_odds, consensus_odds, y, margin=0.0):
    """Back an outcome whenever one book prices it longer than the consensus.

    The consensus is de-vigged and used as the truth, so this is the standard
    'shop for the outlier price' rule that value-betting services sell.
    """
    p = 1.0 / np.asarray(consensus_odds, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    o = np.asarray(book_odds, dtype=float)
    mask = (p * o - 1.0) > margin
    rows, cols = np.nonzero(mask & np.isfinite(o))
    won = np.asarray(y)[rows, cols] > 0
    return pd.DataFrame({"row": rows, "col": cols, "odds": o[rows, cols],
                         "edge": (p * o - 1.0)[rows, cols], "won": won,
                         "pnl": profit(o[rows, cols], won)})
