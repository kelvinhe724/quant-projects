"""Market efficiency study on real football odds: vig, calibration, closing lines, strategies.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import data
import odds as od
import strategy as st

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

SHARP = "PS"          # Pinnacle, the low-margin book that takes sharp money
RETAIL = "B365"       # Bet365, a mass-market book
CONSENSUS = "Avg"     # mean price across the books the site tracks
BEST = "Max"          # best price across those books
NAMED = ["B365", "BW", "IW", "LB", "PS", "WH", "VC"]
ODDS_BUCKETS = [1.0, 1.5, 2.0, 2.75, 4.0, 6.0, 10.0, 20.0, 1000.0]


def show(title, frame):
    print(f"\n{title}\n" + "-" * len(title))
    print(frame.to_string())


def save(frame, name):
    frame.to_csv(os.path.join(REPORTS, name))


def odds_of(q, book, phase, index):
    """Return an (n, 3) array of H/D/A odds for one book and phase, aligned to index."""
    sub = q[(q["book"] == book) & (q["phase"] == phase)].set_index("match_id")
    return sub.reindex(index)[["oh", "od", "oa"]].to_numpy(dtype=float)


def main():
    q = data.load()
    matches = data.match_table(q)
    print(f"{len(matches):,} matches, {len(q):,} quotes, "
          f"{matches['date'].min().date()} to {matches['date'].max().date()}")

    coverage = q.groupby(["book", "phase"])["match_id"].nunique().unstack(fill_value=0)
    coverage["first season"] = q.groupby("book")["season"].min()
    coverage["last season"] = q.groupby("book")["season"].max()
    show("Bookmaker coverage (matches quoted)", coverage.sort_values("open", ascending=False))
    save(coverage, "coverage.csv")

    vig = section_overround(q, matches)
    section_devig(q, matches)
    cal = section_bias(q, matches)
    section_closing_line(q, matches)
    section_strategies(q, matches)
    arb = section_arbitrage(q, matches)
    charts(vig, cal, arb, q, matches)
    print(f"\ncharts and tables written to {REPORTS}/")


def section_overround(q, matches):
    """Margin per book, per season and per league."""
    q = q.copy()
    q["overround"] = od.overround(q[["oh", "od", "oa"]].to_numpy())
    real = q[q["book"].isin(NAMED)]
    by_book = real.pivot_table(index="book", columns="phase", values="overround",
                               aggfunc="mean") * 100
    by_book["matches"] = real.groupby("book")["match_id"].nunique()
    show("Mean overround by bookmaker (%)", by_book.round(2).sort_values("open"))
    save(by_book.round(3), "overround_by_book.csv")

    by_season = real[real["phase"] == "open"].pivot_table(
        index="season", columns="book", values="overround", aggfunc="mean") * 100
    show("Mean opening overround by season (%)", by_season.round(2))
    save(by_season.round(3), "overround_by_season.csv")

    by_league = real.pivot_table(index="league", columns="book", values="overround",
                                 aggfunc="mean")[[SHARP, RETAIL]] * 100
    by_league["all named books"] = real.groupby("league")["overround"].mean() * 100
    show("Mean overround by league (%)", by_league.round(2))
    save(by_league.round(3), "overround_by_league.csv")

    agg = q[q["book"].isin([CONSENSUS, BEST])].pivot_table(
        index="book", columns="phase", values="overround", aggfunc="mean") * 100
    show("Aggregate lines: consensus and best-of-books overround (%)", agg.round(2))
    return by_season


def section_devig(q, matches):
    """How much the three de-vig methods disagree on real prices."""
    idx = matches.index
    o = odds_of(q, SHARP, "close", idx)
    keep = np.isfinite(o).all(axis=1)
    o = o[keep]
    probs = {m: od.devig(o, m) for m in od.METHODS}
    fav = np.argmin(o, axis=1)
    dog = np.argmax(o, axis=1)
    rows = np.arange(len(o))

    tbl = pd.DataFrame({
        "mean favourite prob": {m: p[rows, fav].mean() for m, p in probs.items()},
        "mean longshot prob": {m: p[rows, dog].mean() for m, p in probs.items()},
    })
    base = probs["proportional"]
    tbl["favourite vs proportional (pp)"] = [
        (p[rows, fav] - base[rows, fav]).mean() * 100 for p in probs.values()]
    tbl["longshot vs proportional (pp)"] = [
        (p[rows, dog] - base[rows, dog]).mean() * 100 for p in probs.values()]
    show(f"De-vigging {SHARP} closing lines, three methods "
         f"({len(o):,} matches)", tbl.round(4))
    save(tbl.round(5), "devig_methods.csv")

    _, z = od.devig_shin(o, return_z=True)
    print(f"\nShin insider share z: median {np.median(z):.4f}, "
          f"mean {z.mean():.4f}, 95th pct {np.percentile(z, 95):.4f}")
    return probs


def section_bias(q, matches):
    """Favourite-longshot bias: calibration of de-vigged probabilities."""
    idx = matches.index
    out = {}
    for label, book, phase in ((f"{SHARP} closing", SHARP, "close"),
                               (f"{RETAIL} closing", RETAIL, "close"),
                               (f"{CONSENSUS} closing", CONSENSUS, "close")):
        o = odds_of(q, book, phase, idx)
        keep = np.isfinite(o).all(axis=1)
        y = od.outcome_matrix(matches["result"])[keep]
        o = o[keep]
        for method in od.METHODS:
            p = od.devig(o, method)
            fit = od.calibration_regression(p, y)
            out[(label, method)] = {
                "matches": len(o), "slope": fit["slope"], "se": fit["slope_se"],
                "t vs 1": fit["slope_t"], "p": fit["slope_p"],
                "intercept": fit["intercept"], "brier": od.brier(p, y).mean(),
            }
    reg = pd.DataFrame(out).T.round(4)
    reg.index.names = ["line", "de-vig"]
    show("Calibration regression: slope above 1 means longshots are overpriced", reg)
    save(reg, "calibration_regression.csv")

    o = odds_of(q, SHARP, "close", idx)
    keep = np.isfinite(o).all(axis=1)
    o, y = o[keep], od.outcome_matrix(matches["result"])[keep]
    p = od.devig(o, "shin")
    cal = od.calibration(p, y, odds=o)
    show(f"Calibration of Shin-de-vigged {SHARP} closing odds, by probability bin",
         cal.round(4))
    save(cal.round(5), "calibration_bins.csv")

    buckets = pd.cut(o.ravel(), ODDS_BUCKETS)
    money = pd.DataFrame({"odds": o.ravel(), "won": y.ravel(),
                          "p": p.ravel(), "bucket": buckets})
    money["pnl"] = st.profit(money["odds"], money["won"])
    money["fair_pnl"] = st.profit(1.0 / money["p"], money["won"])
    by_bucket = money.groupby("bucket", observed=True).agg(
        bets=("pnl", "size"), mean_odds=("odds", "mean"),
        hit_rate=("won", "mean"), model_prob=("p", "mean"),
        roi=("pnl", "mean"), roi_no_vig=("fair_pnl", "mean"))
    by_bucket["t"] = money.groupby("bucket", observed=True)["pnl"].apply(
        lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))
    show(f"Return to backing every outcome in an odds bucket at {SHARP} closing",
         by_bucket.round(4))
    save(by_bucket.round(5), "roi_by_odds_bucket.csv")

    # The bucket table is the one FLB test that never touches a de-vig model:
    # regress realised profit on log odds. A negative slope means longer prices
    # pay worse, which is the bias stated in money rather than in probability.
    gradient = []
    for label, pnl in (("at quoted odds", money["pnl"]),
                       ("at Shin fair odds", money["fair_pnl"])):
        x = sm.add_constant(np.log(money["odds"].to_numpy()))
        groups = np.repeat(np.arange(len(o)), 3)
        fit = sm.OLS(pnl.to_numpy(), x).fit(cov_type="cluster",
                                            cov_kwds={"groups": groups})
        gradient.append({"prices": label, "intercept": fit.params[0],
                         "slope on log odds": fit.params[1], "se": fit.bse[1],
                         "t": fit.tvalues[1], "p": fit.pvalues[1]})
    grad = pd.DataFrame(gradient).set_index("prices")
    show("Favourite-longshot bias without a de-vig model: profit against log odds", grad.round(5))
    save(grad.round(6), "flb_gradient.csv")
    return cal


def section_closing_line(q, matches):
    """Do closing prices forecast better than the opening prices of the same book."""
    idx = matches.index
    y_all = od.outcome_matrix(matches["result"])
    rows = []
    per_season = {}
    for book in [SHARP, RETAIL, CONSENSUS]:
        o_open = odds_of(q, book, "open", idx)
        o_close = odds_of(q, book, "close", idx)
        keep = np.isfinite(o_open).all(axis=1) & np.isfinite(o_close).all(axis=1)
        if keep.sum() < 1000:
            continue
        y = y_all[keep]
        p_open = od.devig(o_open[keep], "shin")
        p_close = od.devig(o_close[keep], "shin")
        b_open, b_close = od.brier(p_open, y), od.brier(p_close, y)
        l_open, l_close = od.log_loss(p_open, y), od.log_loss(p_close, y)
        tb = stats.ttest_rel(b_close, b_open)
        tl = stats.ttest_rel(l_close, l_open)
        rows.append({
            "book": book, "matches": int(keep.sum()),
            "brier open": b_open.mean(), "brier close": b_close.mean(),
            "brier gain": b_open.mean() - b_close.mean(), "brier t": tb.statistic,
            "brier p": tb.pvalue, "logloss open": l_open.mean(),
            "logloss close": l_close.mean(),
            "logloss gain": l_open.mean() - l_close.mean(), "logloss p": tl.pvalue,
        })
        if book == SHARP:
            season = matches.loc[keep, "season"].to_numpy()
            per_season = pd.DataFrame({"season": season, "open": b_open,
                                       "close": b_close}).groupby("season").mean()
    tbl = pd.DataFrame(rows).set_index("book")
    show("Closing versus opening lines (lower Brier and log loss are better)",
         tbl.round(5))
    save(tbl.round(6), "closing_line_value.csv")
    if len(per_season):
        per_season["gain"] = per_season["open"] - per_season["close"]
        show(f"{SHARP} Brier by season, opening versus closing", per_season.round(5))
        save(per_season.round(6), "clv_by_season.csv")

    # Does the closing price of the sharp book beat the retail book's closing price?
    o_sharp = odds_of(q, SHARP, "close", idx)
    o_retail = odds_of(q, RETAIL, "close", idx)
    keep = np.isfinite(o_sharp).all(axis=1) & np.isfinite(o_retail).all(axis=1)
    y = y_all[keep]
    bs = od.brier(od.devig(o_sharp[keep], "shin"), y)
    br = od.brier(od.devig(o_retail[keep], "shin"), y)
    t = stats.ttest_rel(bs, br)
    print(f"\n{SHARP} closing Brier {bs.mean():.5f} vs {RETAIL} closing "
          f"{br.mean():.5f} on {keep.sum():,} shared matches "
          f"(diff {br.mean() - bs.mean():+.5f}, p={t.pvalue:.3g})")


def section_strategies(q, matches):
    """Flat and Kelly betting rules, priced with and without the margin."""
    idx = matches.index
    y_all = od.outcome_matrix(matches["result"])
    rows = []
    for book in [SHARP, RETAIL, BEST]:
        o = odds_of(q, book, "close", idx)
        keep = np.isfinite(o).all(axis=1)
        o, y = o[keep], y_all[keep]
        p_fair = od.devig(o, "shin")
        for rule in ("favourite", "longshot", "draw", "home"):
            cols = st.pick_column(o, rule)
            picked, won, pnl = st.flat_bet(o, y, cols)
            fair = st.profit(1.0 / p_fair[np.arange(len(cols)), cols], won)
            s = st.summarise(pnl, f"back the {rule} ({book} close)")
            s["roi no vig"] = float(fair.mean())
            s["mean odds"] = float(picked.mean())
            s["hit rate"] = float(won.mean())
            rows.append(s)
    flat = pd.DataFrame(rows).set_index("rule")
    show("Flat betting rules, one unit per match", flat.round(4))
    save(flat.round(5), "flat_strategies.csv")

    # Back the best available price whenever it beats the de-vigged consensus.
    o_best = odds_of(q, BEST, "close", idx)
    o_cons = odds_of(q, CONSENSUS, "close", idx)
    keep = np.isfinite(o_best).all(axis=1) & np.isfinite(o_cons).all(axis=1)
    o_best, o_cons, y = o_best[keep], o_cons[keep], y_all[keep]
    rows = []
    for margin in (0.0, 0.02, 0.05, 0.10):
        bets = st.disagreement_bets(o_best, o_cons, y, margin)
        s = st.summarise(bets["pnl"], f"best price beats consensus by >{margin:.0%}")
        s["mean edge"] = float(bets["edge"].mean()) if len(bets) else np.nan
        rows.append(s)
    o_sharp = odds_of(q, SHARP, "close", idx)
    sharp_ok = np.isfinite(o_sharp[keep]).all(axis=1)
    os_ = o_sharp[keep][sharp_ok]
    bets = st.disagreement_bets(o_best[sharp_ok], os_, y[sharp_ok], 0.0)
    rows.append(st.summarise(bets["pnl"], f"best price beats de-vigged {SHARP}"))

    # Control: same selections, but taking the consensus price instead of the
    # best price. If the edge is the best-of-many-quotes artifact rather than
    # real information, it dies here.
    ctrl = st.disagreement_bets(o_cons[sharp_ok], os_, y[sharp_ok], 0.0)
    rows.append(st.summarise(ctrl["pnl"], f"consensus price beats de-vigged {SHARP}"))
    same = bets[["row", "col"]].merge(
        pd.DataFrame({"row": np.arange(len(os_)).repeat(3),
                      "col": np.tile([0, 1, 2], len(os_)),
                      "cons": o_cons[sharp_ok].ravel(),
                      "won": (y[sharp_ok] > 0).ravel()}), on=["row", "col"])
    rows.append(st.summarise(st.profit(same["cons"], same["won"]),
                             f"best-price selections, filled at consensus price"))
    value = pd.DataFrame(rows).set_index("rule")
    show("Shopping for the outlier price", value.round(4))
    save(value.round(5), "value_strategies.csv")

    kelly_rows = []
    p_sharp = od.devig(os_, "shin")
    yb = y[sharp_ok]
    won = yb > 0
    for price_name, prices in (("best price", o_best[sharp_ok]),
                               ("consensus price", o_cons[sharp_ok])):
        mask = (p_sharp * prices - 1.0) > 0
        for frac, label in ((1.0, "full Kelly"), (0.5, "half Kelly"),
                            (0.25, "quarter Kelly")):
            f = np.where(mask, frac * st.kelly_fraction(p_sharp, prices), 0.0)
            s = st.summarise((f * st.profit(prices, won))[mask],
                             f"{label}, {price_name}", stake=f[mask])
            s["mean stake"] = float(f[mask].mean())
            s["log growth"] = float(np.log1p(f * st.profit(prices, won)).sum())
            kelly_rows.append(s)
    ob, mask = o_best[sharp_ok], (p_sharp * o_best[sharp_ok] - 1.0) > 0
    kelly = pd.DataFrame(kelly_rows).set_index("rule")
    show("Kelly staking, sizing off the sharp book and betting the best price", kelly.round(4))
    save(kelly.round(5), "kelly_strategies.csv")

    f = st.kelly_fraction(p_sharp, ob)
    f = np.where(mask, f, 0.0)
    growth = np.cumsum(np.log1p(f * st.profit(ob, yb > 0)))
    np.save(os.path.join(REPORTS, "kelly_growth.npy"), growth)


def section_arbitrage(q, matches):
    """How often the best price on every outcome sums below one, and by how much."""
    idx = matches.index
    o_best = odds_of(q, BEST, "close", idx)
    keep = np.isfinite(o_best).all(axis=1)
    s, ret = st.arbitrage(o_best[keep])
    named = [odds_of(q, b, "close", idx) for b in NAMED]
    stack = np.stack(named)
    n_books = np.isfinite(stack).all(axis=2).sum(axis=0)
    own_best = np.where(np.isfinite(stack), stack, -np.inf).max(axis=0)
    ok2 = (n_books >= 4) & np.isfinite(own_best).all(axis=1)
    s2, ret2 = st.arbitrage(own_best[ok2])

    tbl = pd.DataFrame([
        {"source": f"{BEST} column (all books the site tracks)",
         "matches": int(keep.sum()), "arb rate": float((s < 1).mean()),
         "mean return when arbed": float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0,
         "max return": float(ret.max()), "median book sum": float(np.median(s))},
        {"source": f"best of {len(NAMED)} named books, 4+ quoting",
         "matches": int(ok2.sum()), "arb rate": float((s2 < 1).mean()),
         "mean return when arbed": float(ret2[ret2 > 0].mean()) if (ret2 > 0).any() else 0.0,
         "max return": float(ret2.max()), "median book sum": float(np.median(s2))},
    ]).set_index("source")
    show("Cross-book arbitrage on closing lines", tbl.round(5))
    save(tbl.round(6), "arbitrage.csv")

    frames = []
    for label, mask, bs, r in ((BEST, keep, s, ret), ("named books", ok2, s2, ret2)):
        season = matches.loc[mask, "season"].to_numpy()
        g = pd.DataFrame({"season": season, "arb": bs < 1, "ret": r}).groupby("season")
        frames.append(g.agg(**{
            f"{label} matches": ("arb", "size"),
            f"{label} arb rate": ("arb", "mean"),
            f"{label} mean return": ("ret", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
        }))
    by_season = pd.concat(frames, axis=1)
    show("Arbitrage frequency by season, closing lines", by_season.round(5))
    save(by_season.round(6), "arbitrage_by_season.csv")
    print(f"\nbooks quoting per match (named books, closing): "
          f"median {np.median(n_books[ok2]):.0f}, mean {n_books[ok2].mean():.2f}")
    return by_season


def charts(vig, cal, arb, q, matches):
    fig, ax = plt.subplots(figsize=(10, 5))
    for book in vig.columns:
        if vig[book].notna().sum() > 4:
            ax.plot(vig.index, vig[book], marker="o", markersize=3,
                    linewidth=1.2, label=data.BOOKS.get(book, book))
    ax.set_ylabel("mean overround (%)")
    ax.set_xlabel("season")
    ax.set_title("Bookmaker margin on 1X2 football markets, six European leagues")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "overround_by_season.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.errorbar(cal["mean_p"], cal["realised"],
                yerr=[cal["realised"] - cal["lo"], cal["hi"] - cal["realised"]],
                fmt="o", color="steelblue", capsize=3, markersize=5,
                label=f"{SHARP} closing, Shin de-vig")
    ax.set_xlabel("de-vigged implied probability")
    ax.set_ylabel("realised frequency")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(cal["mean_p"], (cal["realised"] - cal["mean_p"]) * 100, "o-",
            color="firebrick", markersize=5)
    ax.fill_between(cal["mean_p"], (cal["lo"] - cal["mean_p"]) * 100,
                    (cal["hi"] - cal["mean_p"]) * 100, color="firebrick", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("de-vigged implied probability")
    ax.set_ylabel("realised minus predicted (percentage points)")
    ax.set_title("Calibration error with 95% Wilson intervals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "calibration.png"), dpi=140)
    plt.close(fig)

    bucket = pd.read_csv(os.path.join(REPORTS, "roi_by_odds_bucket.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bucket))
    ax.bar(x - 0.2, bucket["roi"] * 100, 0.4, color="firebrick", label="at quoted odds")
    ax.bar(x + 0.2, bucket["roi_no_vig"] * 100, 0.4, color="steelblue",
           label="at de-vigged fair odds")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket["bucket"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("return on turnover (%)")
    ax.set_title(f"Return to flat betting every outcome in an odds bucket ({SHARP} closing)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "roi_by_odds.png"), dpi=140)
    plt.close(fig)

    clv_path = os.path.join(REPORTS, "clv_by_season.csv")
    if os.path.exists(clv_path):
        clv = pd.read_csv(clv_path)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(clv["season"], clv["open"], "o-", color="darkorange", label="opening line")
        ax.plot(clv["season"], clv["close"], "o-", color="seagreen", label="closing line")
        ax.set_ylabel("mean Brier score (lower is better)")
        ax.set_title(f"{SHARP} forecast accuracy, opening versus closing prices")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS, "closing_line_value.png"), dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(arb))
    ax.bar(x - 0.2, arb[f"{BEST} arb rate"] * 100, 0.4, color="lightsteelblue",
           label=f"{BEST} column, every book the site tracks")
    ax.bar(x + 0.2, arb["named books arb rate"] * 100, 0.4, color="steelblue",
           label="best of the seven named books")
    ax.set_xticks(x)
    ax.set_xticklabels(arb.index, rotation=45)
    ax.set_ylabel("share of matches with a cross-book arb (%)")
    ax2 = ax.twinx()
    ax2.plot(x, arb["named books mean return"] * 100, "o-", color="firebrick",
             label="mean return when one exists (named books)")
    ax2.set_ylabel("mean arb return (%)")
    ax.set_title("Cross-book arbitrage on closing 1X2 prices")
    ax.legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="upper center")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "arbitrage.png"), dpi=140)
    plt.close(fig)

    growth = np.load(os.path.join(REPORTS, "kelly_growth.npy"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.exp(growth), color="darkslategray", linewidth=1)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("bankroll, log scale")
    ax.set_xlabel("match")
    ax.set_title(f"Full Kelly on best price whenever it beats the de-vigged {SHARP} line")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "kelly_equity.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
