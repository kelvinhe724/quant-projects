"""Summarise every collected day that lacks a summary, then report the latest one.

Run: python3 run.py
Writes reports/atm_by_expiry.csv, reports/horizons_latest.csv,
reports/implied_corr.csv, reports/term_structure.png and reports/run_log.txt.
"""
import io
import sys
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import data
import surface


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def folder_bytes(path):
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def main():
    days = data.list_days()
    if not days:
        print("nothing collected yet: run collect.py first")
        return 1
    for d in days:
        surface.summarise_day(d)
    last = days[-1]
    summary = pd.read_csv(data.SUMMARY / f"{last}.csv")
    horizons = pd.read_csv(data.SUMMARY / f"{last}_horizons.csv")
    chains, manifest = data.load_day(last)

    raw_mb = folder_bytes(data.day_dir(last)) / 1e6
    print(f"days collected: {len(days)} ({days[0]} to {last})")
    print(f"latest snapshot {manifest['snapshot_utc']} [{manifest['session']}], "
          f"{manifest['rows']} raw quotes across {len(manifest['collected'])} tickers, "
          f"missing: {manifest['missing'] or 'none'}, bill rate {manifest['rate']}")
    print(f"storage: {raw_mb:.1f} MB raw per day, "
          f"{(data.SUMMARY / f'{last}.csv').stat().st_size / 1e3:.0f} KB summary; "
          f"a year of trading days is roughly {raw_mb * 252 / 1e3:.1f} GB raw")
    if manifest["session"] == "intraday":
        print("this snapshot is intraday; the launchd job takes the closing one")

    counts = pd.DataFrame({
        "raw": {t: len(q) for t, q in chains.items()},
        "empty_quotes": {t: surface.stale_stats(q)["zero_quote_frac"] for t, q in chains.items()},
        "clean": summary.groupby("ticker")["n_clean"].sum(),
        "expiries": summary.dropna(subset=["atm_iv"]).groupby("ticker").size(),
    }).fillna(0).astype({"raw": int, "clean": int, "expiries": int})
    counts["stale"] = counts["empty_quotes"] > surface.STALE_ZERO_FRAC
    counts["empty_quotes"] = counts["empty_quotes"].map("{:.0%}".format)
    show("Quotes per ticker: raw, share with no bid or ask, surviving the cleaning pass, "
         "expiries with an ATM vol", counts.loc[[t for t in data.UNIVERSE if t in counts.index]])

    atm = summary.pivot(index="expiry", columns="ticker", values="atm_iv")
    atm = atm[[t for t in data.UNIVERSE if t in atm.columns]]
    show(f"ATM implied vol by expiry, {last}", atm.map(lambda v: f"{v:.1%}" if pd.notna(v) else ""))
    atm.to_csv(data.REPORTS / "atm_by_expiry.csv")

    skew = summary.pivot(index="expiry", columns="ticker", values="skew_25d")
    skew = skew[[t for t in data.UNIVERSE if t in skew.columns]]
    show("25-delta skew (put vol minus call vol) by expiry",
         skew.map(lambda v: f"{v:+.1%}" if pd.notna(v) else ""))

    h = horizons.set_index("ticker").reindex([t for t in data.UNIVERSE if t in set(horizons["ticker"])])
    show("ATM vol at fixed horizons, interpolated in total variance",
         h[["n_expiries", "iv30", "iv60", "iv90"]].assign(
             **{c: h[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "") for c in ("iv30", "iv60", "iv90")}))
    h.to_csv(data.REPORTS / "horizons_latest.csv")

    spy = h.loc["SPY"] if "SPY" in h.index else None
    if spy is not None and pd.notna(spy["rho30"]):
        print(f"\nimplied correlation of the {int(spy['n_names'])}-name basket against SPY: "
              f"30 day {spy['rho30']:.3f}, 60 day {spy['rho60']:.3f} "
              f"(weights from {spy['weights_from']})")
    else:
        print("\nimplied correlation not computable today (SPY or too many names missing)")

    hist = surface.load_horizons()
    corr = hist[hist["ticker"] == "SPY"][["date", "iv30", "iv60", "rho30", "rho60", "n_names"]]
    corr.to_csv(data.REPORTS / "implied_corr.csv", index=False)
    if len(corr) > 1:
        show("Implied correlation history", corr.set_index("date").round(3))

    fig, ax = plt.subplots(figsize=(10, 5))
    for t in atm.columns:
        s = summary[summary["ticker"] == t].dropna(subset=["atm_iv"]).sort_values("T")
        ax.plot(s["T"] * 365, s["atm_iv"], marker="o", ms=3, lw=1,
                label=t, alpha=0.9 if t in data.ETFS else 0.6)
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM implied vol")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"ATM term structure, {last} ({manifest['session']})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(data.REPORTS / "term_structure.png", dpi=140)
    plt.close(fig)
    print(f"\ntables and chart written to {data.REPORTS}/")
    return 0


if __name__ == "__main__":
    data.REPORTS.mkdir(exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main()
    sys.stdout.write(buf.getvalue())
    (data.REPORTS / "run_log.txt").write_text(buf.getvalue())
    raise SystemExit(code)
