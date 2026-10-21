"""Pre-registration checker: what research/PREREG-TEMPLATE.md (v2) asks, verified mechanically.

Three items, each printed PASS or FAIL with its evidence:

  1 tradability  every equity/ETF named in section 2 against Alpaca's own asset flags
                 (tradable, shortable, marginable), with the paper keys read the way
                 framework/book/broker.read_env reads them; only get_asset is called and the
                 keys are never printed. FAIL if the page plans a short on a name Alpaca
                 will not short. A short, sold or written option is flagged, not failed:
                 Alpaca's flags cannot show it is allowed, and Alpaca permits no uncovered
                 short option.
  2 provenance   every date range in README.md, */README.md and REFUSALS.md, in the working
                 tree and in the git history (git log -S on the window's start year), that
                 overlaps the section 5 untouched window in a document naming the same
                 instruments, each dated by the commit that first published it. FAIL if one
                 predates the pre-registration's commit and section 2a names neither its
                 file nor its commit.
  3 power        section 2b's no-edge pass rate and decision rule, beside the pass rate
                 recomputed by normal approximation from the track length and Sharpe bar.
                 FAIL if 2b, its rate, its rule, the length or the bar is missing.

A pre-v2 page without numbered sections is read from its **Instrument.** and
**Test.** paragraphs, and says so.

Run: ../.venv/bin/python3 prereg_check.py ../vol-risk-premium/PREREG.md [more.md ...] [--env path/to/.env]
Exits 1 if any item fails.
"""
import functools
import glob
import math
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENV = None  # framework/.env, unless --env names another file
# ponytail: upper-case abbreviations these pages use that are not instruments; add one when a page trips on it
STOP = {"ATM", "OTM", "ITM", "ETF", "ETFS", "CT", "ET", "UTC", "MA", "PIT", "IV", "NW", "KYC", "API", "USD", "NYSE",
        "OECD", "FRED", "DSR", "PSR", "VRP", "CSV", "SEC", "OIA", "EOD", "GTC", "IOC", "TIF", "PNL", "OK", "NO",
        "YES", "AND", "OR", "NOT", "THE", "DTE", "OI", "IRA", "PDF", "JSON", "GARCH", "HMM", "LOB", "MVO", "ERC"}
SHORT = r"short(?:s|ed|ing)?"
SELL = SHORT + r"|sell(?:s|ing)?|sold|writ(?:e|es|ing|ten)|wrote"
OPTION = r"\b(?:options?|straddles?|strangles?|calls?|puts?|condors?)\b"
NEGATION = {"no", "never", "not", "without", "cannot", "can't", "nor"}
DATE = r"\d{4}(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?)?"
RANGE = re.compile(rf"(?<![\d-])({DATE})\s*(?:→|->|–|—|-|to|through|until)\s*({DATE})(?![\d-])")
CORPUS = ("README.md", "*/README.md", "REFUSALS.md", "*/REFUSALS.md")
DECIDE = r"\bdecid\w*|\bdecision\b|\blonger track\b|\bextend\w*|\bshrunk\b|\bshrink\w*|\bjudge?ment\b"


def sections(text):
    """{'2': body, '2a': ..., '5': ...} by '## 2.'-style headings; a pre-v2 page's bold
    **Instrument.** and **Test.** paragraphs stand in for 2 and 5."""
    parts = re.split(r"^##\s+(.*)$", text, flags=re.M)
    out = {m.group(1): body for head, body in zip(parts[1::2], parts[2::2]) if (m := re.match(r"(\d+[a-z]?)\.", head))}
    for sid, label in (("2", "Instrument"), ("5", "Test")):
        m = re.search(rf"^\*\*{label}\.\*\*(.*)$", text, flags=re.M)
        if sid not in out and m:
            out[sid] = m.group(1)
    return out


def tickers(text):
    """Upper-case words of 2-5 letters, in order of first use, minus STOP."""
    return list(dict.fromkeys(t for t in re.findall(r"(?<![\w-])[A-Z]{2,5}(?![\w-])", text) if t not in STOP))


def term_in(term, text):
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text) is not None


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n", text) if s.strip()]


def planned(sentence, words):
    """`words` occurs in the sentence with none of the three words before it a negation."""
    for m in re.finditer(rf"\b(?:{words})\b", sentence, flags=re.I):
        if not NEGATION & {w.lower() for w in re.findall(r"[\w']+", sentence[:m.start()])[-3:]}:
            return True
    return False


def clip(s, n=110):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


@functools.lru_cache(maxsize=None)
def client():
    from alpaca.trading.client import TradingClient
    from framework.book.broker import PAPER_KEYS, read_env
    keys = read_env(PAPER_KEYS, ENV)
    if not all(keys.get(k) for k in PAPER_KEYS):
        raise RuntimeError(f"no Alpaca paper keys in {ENV or 'framework/.env'}: tradability not checked")
    # raw JSON: alpaca-py's Asset model refuses classes its enum does not list (XSP comes back as one)
    return TradingClient(keys[PAPER_KEYS[0]], keys[PAPER_KEYS[1]], paper=True, raw_data=True)


def asset(symbol):
    """Alpaca's asset record for symbol as a dict; None when Alpaca lists no such asset."""
    from alpaca.common.exceptions import APIError
    try:
        a = client().get_asset(symbol)
    except APIError as e:
        if e.status_code in (404, 422):
            return None
        raise
    return {"asset_class": a.get("class"), **{k: a.get(k) for k in ("name", "tradable", "shortable", "marginable",
                                                                    "easy_to_borrow")}}


def tradability(text):
    sec = sections(text).get("2")
    if not sec:
        return False, ["no section 2 (universe and instrument) to read"]
    sents, lines, ok, equities = sentences(text), [], True, 0
    for t in tickers(sec):
        try:
            a = asset(t)
        except Exception as e:  # no keys, no network: unverified is not a pass
            return False, lines + [f"{t}: Alpaca lookup failed ({type(e).__name__}: {clip(str(e), 90)})"]
        if a is None or a["asset_class"] != "us_equity":
            lines.append(f"{t}: not an Alpaca equity/ETF" + (f" ({a['asset_class']})" if a else ""))
            continue
        equities += 1
        shorts = [s for s in sents if term_in(t, s) and not re.search(OPTION, s, re.I) and planned(s, SHORT)]
        bad = bool(shorts) and not a["shortable"]
        ok &= not bad
        lines.append(f"{t} ({a['name']}): tradable={a['tradable']} shortable={a['shortable']} "
                     f"marginable={a['marginable']}; "
                     + (f'planned short: "{clip(shorts[0], 80)}"' if shorts else "no short planned")
                     + (" -> Alpaca will not short it" if bad else ""))
    if not equities:
        lines.append("no Alpaca equity/ETF named in section 2; other venues are not checked here")
    opts = [s for s in sents if re.search(OPTION, s, re.I) and planned(s, SELL)]
    if opts:
        lines.append(f"FLAG options leg: cannot verify from Alpaca flags; Alpaca permits no uncovered short options "
                     f'({len(opts)} sentences, first: "{clip(opts[0], 80)}")')
    return ok, lines


def git(*args, root=ROOT):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def first_published(path, text, root):
    """(commit, time) of the first commit that put `text` into `path`; (None, None) if git has none."""
    out = git("log", "-S", text, "--reverse", "--format=%h %cI", "--", path, root=root).split()
    return (out[0], pd.Timestamp(out[1])) if out else (None, None)


def committed(path, root):
    """(commit, time) of the commit that added `path`, the pre-registration's lock; (None, None) if none."""
    out = git("log", "--diff-filter=A", "--format=%h %cI", "--", path, root=root).split()
    return (out[-2], pd.Timestamp(out[-1])) if out else (None, None)


def bounds(token, end=False):
    """First (or last) day that a YYYY, YYYY-MM or YYYY-MM-DD token covers."""
    p = pd.Period(token, freq={4: "Y", 7: "M"}.get(len(token), "D"))
    return (p.end_time if end else p.start_time).normalize()


def ranges(text):
    """(first day, last day, text) of every date range in text."""
    out = []
    for m in RANGE.finditer(text):
        a, z = bounds(m.group(1)), bounds(m.group(2), end=True)
        if 1990 <= a.year and z.year <= 2100 and a <= z:
            out.append((a, z, m.group(0)))
    return out


def untouched(sec5):
    """The untouched window: the first date range after the word 'untouched' in section 5, same paragraph."""
    m = re.search(r"untouched", sec5 or "", flags=re.I)
    found = ranges(sec5[m.end():].split("\n\n")[0]) if m else []
    return found[0] if found else None


def blocks(path, text):
    """A project README is about its own instruments, so the file is one block; the root README and REFUSALS
    cover every project, so there a range must share its paragraph (or table row) with an instrument."""
    if "/" in path:
        return [text]
    out = []
    for para in re.split(r"\n\s*\n", text):
        out += para.splitlines() if para.lstrip().startswith("|") else [para]
    return out


def overlaps(path, text, window, terms):
    return {(path, s) for b in blocks(path, text) if any(term_in(t, b) for t in terms)
            for a, z, s in ranges(b) if a <= window[1] and z >= window[0]}


def provenance(text, root=ROOT, prereg=None):
    s = sections(text)
    win = untouched(s.get("5"))
    if win is None:
        return False, ["section 5 names no untouched window with dates"]
    sec2 = s.get("2", "")
    terms = tickers(sec2) + re.findall(r"S&P\s?\d{3,4}", sec2)
    head = f'untouched {win[0].date()} -> {win[1].date()} (section 5: "{clip(win[2])}")'
    if not terms:
        return False, [head, "section 2 names no instrument to match the window's readers against"]
    found = set()
    for f in sorted({os.path.relpath(p, root) for pat in CORPUS for p in glob.glob(os.path.join(root, pat))}):
        with open(os.path.join(root, f)) as fh:
            found |= overlaps(f, fh.read(), win, terms)
    for chunk in git("log", "-S", str(win[0].year), "--format=%x00%h", "--name-only", "--", *CORPUS,
                     root=root).split("\x00")[1:]:
        h, *paths = chunk.split()
        for f in paths:
            found |= overlaps(f, git("show", f"{h}:{f}", root=root), win, terms)
    commit, when = committed(prereg, root) if prereg else (None, None)
    ack, ok = s.get("2a", ""), True
    lines = [f"{head}; instruments {', '.join(terms)}; pre-registration "
             + (f"added in {commit} {when}" if commit else "not committed, so every hit predates it")]
    by_file = {}
    for f, rng in sorted(found):
        h, t = first_published(f, rng, root)
        before = when is None or (t is not None and t < when)
        named = re.search(rf"(?<![\w/.-]){re.escape(f)}", ack) is not None or bool(h and h in ack)
        ok &= not (before and not named)
        by_file.setdefault(f, []).append((t, h, before, named, rng))
    for f, hits in by_file.items():
        dated = [x for x in hits if x[0] is not None]
        t, h = min(dated, key=lambda x: x[0])[:2] if dated else (None, None)
        state = ("after the pre-registration" if not any(x[2] for x in hits) else
                 "before the pre-registration and not in 2a" if any(x[2] and not x[3] for x in hits) else
                 "before the pre-registration, named in 2a")
        lines.append(f"{f}, first published {h} {t.date()}" if h else f"{f}, in no commit")
        lines[-1] += f", {state}: " + ", ".join(f'"{clip(x[4], 40)}"' for x in hits)
    if not found:
        lines.append("no README or REFUSALS range overlaps the window on these instruments")
    return ok, lines


def no_edge_pass_rate(n, bar):
    """P(annualised Sharpe estimate > bar over n daily sessions | true Sharpe 0), normal approximation:
    the daily Sharpe's SE is sqrt((1 + 0.5 S^2) / n), at S = 0, annualised by sqrt(252)."""
    se = math.sqrt((1 + 0.5 * 0.0 ** 2) / n) * math.sqrt(252)
    return 0.5 * math.erfc(bar / se / math.sqrt(2))


def power(text):
    s = sections(text)
    sec = s.get("2b")
    src = sec if sec is not None else s.get("5", "")[max(s.get("5", "").find("Gate"), 0):]
    m = re.search(r"(\d[\d,]*)\s*(?:trading\s+)?(?:sessions|days)\b", src)
    mo = re.search(r"(\d+)[\s-]*months?\b", src)
    n = int(m.group(1).replace(",", "")) if m else int(mo.group(1)) * 21 if mo else None
    b = re.search(r"Sharpe[^.\d]{0,40}?(?:>|≥|above|over|at least|of)?\s*(\d+(?:\.\d+)?)(?![\d/])"
                  r"|(\d+(?:\.\d+)?)\s+(?:net\s+)?Sharpe", src)
    bar = float(b.group(1) or b.group(2)) if b else None
    computed = no_edge_pass_rate(n, bar) if n and bar is not None else None
    stated = None
    for sent in sentences(sec or ""):
        if re.search(r"no.?edge|zero.?edge|\bnull\b", sent, flags=re.I):
            p = re.search(r"(\d+(?:\.\d+)?)\s*%", sent) or re.search(r"(?<![\d.])(0?\.\d+)(?![\d.])", sent)
            if p:
                stated = float(p.group(1)) / (100 if "%" in p.group(0) else 1)
                break
    rule = sec is not None and re.search(DECIDE, sec, flags=re.I) is not None
    where = "section 2b" if sec is not None else "section 5 (no section 2b)"
    if computed is not None:
        comp = (f"computed: P(Sharpe > {bar:g} over {n} sessions | no edge) = {computed:.1%}, normal approximation, "
                "SE sqrt((1 + 0.5 S^2) / N) at S = 0")
    else:
        comp = f"computed: not possible, track length {n or 'absent'}, Sharpe bar {'absent' if bar is None else bar}"
    lines = [f"stated no-edge pass rate: {'absent' if stated is None else f'{stated:.1%}'}"
             + ("" if sec is not None else " (no section 2b)"),
             f"{comp}; length and bar read from {where}",
             f"decision rule: {'stated' if rule else 'absent'}"]
    return bool(sec is not None and stated is not None and rule and computed is not None), lines


def check(path):
    """Print the three items for one PREREG.md; return the number that failed."""
    with open(path) as fh:
        text = fh.read()
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    print(f"\n{rel}")
    if not re.search(r"^##\s+2\.", text, flags=re.M):
        print("      pre-v2 layout: no numbered sections; sections 2 and 5 read from **Instrument.** and **Test.**")
    bad = 0
    for i, (name, (ok, lines)) in enumerate((("tradability", tradability(text)),
                                              ("window provenance", provenance(text, ROOT, rel)),
                                              ("power", power(text))), 1):
        print(f"{'PASS' if ok else 'FAIL'}  {i} {name}")
        for line in lines:
            print(f"      {line}")
        bad += not ok
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--env" in args:
        i = args.index("--env")
        ENV = args[i + 1]
        del args[i:i + 2]
    sys.exit(1 if sum(check(p) for p in args) else 0)
