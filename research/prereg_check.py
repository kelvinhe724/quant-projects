"""Pre-registration checker: what research/PREREG-TEMPLATE.md (v2) asks, verified mechanically.

Three items, each printed PASS or FAIL with its evidence:

  1 tradability  every equity/ETF named in section 2 against Alpaca's own asset flags
                 (tradable, shortable, easy_to_borrow, marginable), with the paper keys read
                 the way framework/book/broker.read_env reads them; only get_asset is called
                 and the keys are never printed. Sections 2 and 3 are read a paragraph at a
                 time for what each leg does: a short (short, sell/sold, a negative weight)
                 unless a negation sits on either side of the verb in its clause, attributed
                 to the name after the verb, else the one before, else every name in the
                 sentence, then the paragraph, then section 2. FAIL on tradable=False, on a
                 short of a name Alpaca will not short or does not list as easy to borrow, on
                 any sold or written option leg ("Alpaca allows no uncovered short options")
                 unless its sentence says covered, and when no leg was verified at all.
                 Upper-case words on the STOP list that are live Alpaca assets are printed
                 as a notice, since they were not read as instruments.
  2 provenance   every date range in README.md, */README.md, REFUSALS.md, */reports/*.json,
                 */reports/*.md and every */reports/registry/runs.jsonl (its window and
                 universe columns), in the working tree and in every commit git log -G finds
                 a date range in, that overlaps the section 5 untouched window in the same
                 paragraph, table row or JSON object as one of the instruments, each dated by
                 the commit that first published it. FAIL if one predates the
                 pre-registration's commit and section 2a names neither its file nor its
                 commit.
  3 power        section 2b's no-edge pass rate and decision rule, beside the pass rate
                 recomputed by normal approximation from the track length (in sessions) and
                 the Sharpe bar ("Sharpe above X", "Sharpe of at least X", "Sharpe >= X",
                 "net Sharpe ... X"). FAIL if 2b, its rate, its rule (a longer track, a shrunk
                 estimate or Kelvin's judgment, the template's three), the length or the bar
                 is missing, or the stated rate is more than 2 points from the computed one.
                 A book that sells options gets a WARNING: the normal formula understates its
                 no-edge pass rate, and the page must state the share of P&L in its worst 1%
                 of days.

A pre-v2 page without numbered sections is read from its **Instrument.** and
**Test.** paragraphs, and says so.

Run: ../.venv/bin/python3 prereg_check.py ../vol-risk-premium/PREREG.md [more.md ...] [--env path/to/.env]
Exits 1 if any item fails.
"""
import functools
import glob
import json
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
# a short: the verb (short-term and friends are not one), or a negative weight
VERB = re.compile(r"(?<![\w-])(?:short(?:s|ed|ing)?|sell(?:s|ing)?|sold|writ(?:e|es|ing|ten)|wrote)(?![\w-])"
                  r"|\bweights?\s*(?:of\s*)?[-−–]\s*\d", re.I)
# ponytail: nouns that make "short" an adjective; add one when a page trips on it
NOT_A_SHORT = {"term", "window", "windows", "history", "sample", "track", "period", "run", "horizon", "list", "span",
               "rate", "rates", "interest", "end", "squeeze", "version", "answer"}
OPTION = re.compile(r"^(?:options?|straddles?|strangles?|calls?|puts?)$", re.I)
NEGATION = {"no", "never", "not", "without", "cannot", "can't", "nor", "none", "impossible", "isn't", "won't",
            "doesn't", "don't", "refuses", "refused", "unavailable"}
STOPWORDS = {"and", "or", "then", "when", "while", "but", "if", "until", "unless", "buy", "buys", "bought", "buying",
             "to", "with", "against", "hedged", "so", "because", "after", "before"}
CLAUSE = r"[,;:()—–]|\s(?:but|so|while|whereas)\s"
DATE = r"\d{4}(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?)?"
RANGE = re.compile(rf"(?<![\d-])({DATE})\s*(?:→|->|–|—|-|to|through|until)\s*({DATE})(?![\d-])")
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}")
CORPUS = ("README.md", "*/README.md", "REFUSALS.md", "*/REFUSALS.md", "*/reports/*.md", "*/*/reports/*.md",
          "*/reports/*.json", "*/*/reports/*.json", "*/reports/registry/runs.jsonl", "*/*/reports/registry/runs.jsonl")
DECIDE = r"\blonger (?:forward |paper )?track\b|\bshrunk estimate\b|\bjudge?ment\b"
LENGTH = r"(\d[\d,]*)\s*(?:trading\s+)?sessions\b"
BAR = (r"\bSharpe\s+(?:above|of at least)\s+(-?\d+(?:\.\d+)?)(?![\d/])"
       r"|\bSharpe\s*(?:≥|>=)\s*(-?\d+(?:\.\d+)?)(?![\d/])"
       r"|\bnet Sharpe\s[^.\d]{0,40}?(-?\d+(?:\.\d+)?)(?![\d/])")


def sections(text):
    """{'2': body, '2a': ..., '5': ...} by '## 2.'-style headings; a pre-v2 page's bold
    **Instrument.** and **Test.** paragraphs stand in for 2 and 5."""
    parts = re.split(r"^##\s+(.*)$", text, flags=re.M)
    out = {}
    for head, body in zip(parts[1::2], parts[2::2]):
        m = re.match(r"(\d+)(?:\.?([a-z]))?\.?(?=\s|$)", head)  # "2.", "2a.", "2.a", "2b" (a "2.a" is not "2")
        if m:
            out.setdefault(m.group(1) + (m.group(2) or ""), body)
    for sid, label in (("2", "Instrument"), ("5", "Test")):
        m = re.search(rf"^\*\*{label}\.\*\*(.*?)(?=\n\s*\n|\n\*\*|\Z)", text, flags=re.M | re.S)
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


def paragraphs(text):
    """Blank-line paragraphs, each list item its own, wrapped lines joined."""
    out = []
    for block in re.split(r"\n\s*\n", text or ""):
        cur = []
        for line in block.splitlines():
            if re.match(r"\s*(?:[-*+]|\d+\.)\s", line) and cur:
                out.append(" ".join(cur))
                cur = []
            if line.strip():
                cur.append(line.strip())
        if cur:
            out.append(" ".join(cur))
    return out


def words(text):
    return [(m.group(0), m.start()) for m in re.finditer(r"[\w'&]+(?:[-/][\w']+)*", text)]


def legs(text, names):
    """What sections 2-3 do with each leg: ({ticker: first sentence that shorts it}, [sentences selling options]).

    `names` are the section 2 tickers, the last resort for a short that names no leg.
    """
    shorts, options = {}, []
    for para in paragraphs(text):
        for sent in re.split(r"(?<=[.;!?])\s+", para):
            for m in VERB.finditer(sent):
                verb = m.group(0).lower()
                start = max([0] + [x.end() for x in re.finditer(CLAUSE, sent[:m.start()])])
                stop = min([len(sent)] + [m.end() + x.start() for x in re.finditer(CLAUSE, sent[m.end():])])
                before = [w.lower() for w, _ in words(sent[start:m.start()])][-3:]
                after_all = [w for w, _ in words(sent[m.end():stop])]
                if NEGATION & set(before) or NEGATION & {w.lower() for w in after_all[:4]}:
                    continue
                if verb.startswith("short") and after_all and after_all[0].lower() in NOT_A_SHORT:
                    continue
                after = []
                for w in after_all[:4]:
                    if w.lower() in STOPWORDS:
                        break
                    after.append(w)
                back = [w for w, _ in words(sent[:m.start()])][-3:]
                if any(OPTION.match(w) for w in after) or (not verb.startswith("weight") and not tickers(" ".join(after))
                                                            and any(OPTION.match(w) for w in back)):
                    if not re.search(r"\bcovered\b", sent, re.I):
                        options.append(sent)
                    continue
                named = (tickers(" ".join(after))[:1] or tickers(" ".join(back))[-1:] or tickers(sent)
                         or tickers(para) or list(names))
                for t in named:
                    shorts.setdefault(t, sent)
    return shorts, options


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
    secs = sections(text)
    sec = secs.get("2")
    if not sec:
        return False, ["no section 2 (universe and instrument) to read"]
    names = tickers(sec)
    shorts, options = legs(sec + "\n\n" + secs.get("3", ""), names)
    lines, ok, verified = [], True, 0
    stop = [w for w in dict.fromkeys(re.findall(r"(?<![\w-])[A-Z]{2,5}(?![\w-])", sec)) if w in STOP]
    for t in list(dict.fromkeys(names + list(shorts))) + stop:
        try:
            a = asset(t)
        except Exception as e:  # no keys, no network: unverified is not a pass
            return False, lines + [f"{t}: Alpaca lookup failed ({type(e).__name__}: {clip(str(e), 90)})"]
        if t in stop:
            if a is not None and a["asset_class"] == "us_equity":
                lines.append(f"notice: {t} is on the STOP list, so it was not read as an instrument, but it is a live "
                             f"Alpaca asset ({a['name']}); if the page trades it, take it off the list")
            continue
        if a is None or a["asset_class"] != "us_equity":
            lines.append(f"{t}: not an Alpaca equity/ETF" + (f" ({a['asset_class']})" if a else "")
                         + (f'; planned short not checked: "{clip(shorts[t], 70)}"' if t in shorts else ""))
            continue
        bad = []
        if not a["tradable"]:
            bad.append("Alpaca lists it tradable=False")
        if t in shorts and not (a["shortable"] and a["easy_to_borrow"]):
            bad.append("Alpaca will not short it" if not a["shortable"] else "Alpaca does not list it easy to borrow")
        ok &= not bad
        verified += not bad
        lines.append(f"{t} ({a['name']}): tradable={a['tradable']} shortable={a['shortable']} "
                     f"easy_to_borrow={a['easy_to_borrow']} marginable={a['marginable']}; "
                     + (f'planned short: "{clip(shorts[t], 80)}"' if t in shorts else "no short planned")
                     + "".join(f" -> {x}" for x in bad))
    if options:
        ok = False
        lines.append(f"options leg sold or written: Alpaca allows no uncovered short options "
                     f'({len(options)} sentences, first: "{clip(options[0], 80)}")')
    if not verified:
        ok = False
        lines.append("no leg verified: no Alpaca equity/ETF in section 2 passes its flags; other venues are not "
                     "checked here")
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


def day(x):
    """A YYYY-MM-DD... string as a naive day; None if it is not one."""
    if not isinstance(x, str) or not DAY.match(x):
        return None
    try:
        t = pd.Timestamp(x)
    except ValueError:
        return None
    return (t.tz_localize(None) if t.tzinfo else t).normalize()


def json_blocks(obj, inherited=""):
    """One block per JSON object: its own scalars and keys plus its ancestors' (never a sibling's), and the
    ranges it states: a start/end pair of keys, a two-date window, or a range written inside a string.
    A range is (first day, last day, what to show, the literal git -S looks for)."""
    if isinstance(obj, list):
        return [b for x in obj for b in json_blocks(x, inherited)]
    if not isinstance(obj, dict):
        return []
    flat = lambda v: isinstance(v, (str, int, float)) or (isinstance(v, list) and all(
        isinstance(x, (str, int, float)) for x in v))
    text = inherited + " " + " ".join(f"{k} {v}" if flat(v) else str(k) for k, v in obj.items())
    rng = []
    for k, v in obj.items():
        if isinstance(v, str):
            rng += [(a, z, t, t) for a, z, t in ranges(v) if not day(v)]
        if isinstance(v, list) and len(v) == 2 and day(v[0]) is not None and day(v[1]) is not None:
            rng.append((day(v[0]), day(v[1]), f"{v[0][:10]} to {v[1][:10]}", v[0]))
        if str(k).endswith("start") and day(v) is not None and day(obj.get(str(k)[:-5] + "end")) is not None:
            z = obj[str(k)[:-5] + "end"]
            rng.append((day(v), day(z), f"{v[:10]} to {z[:10]}", v))
    out = [(text, rng)]
    for k, v in obj.items():
        if not flat(v):
            out += json_blocks(v, text)
    return out


def blocks(path, text):
    """(text, ranges) per block: a JSON object in a .json or .jsonl file, else a paragraph or a table row.
    An instrument and a range count together only inside one block."""
    if path.endswith((".json", ".jsonl")):
        try:
            docs = [json.loads(text)] if path.endswith(".json") else [json.loads(x) for x in text.splitlines()
                                                                      if x.strip()]
            return [b for d in docs for b in json_blocks(d)]
        except ValueError:
            pass
    out = []
    for para in re.split(r"\n\s*\n", text):
        for b in (para.splitlines() if para.lstrip().startswith("|") else [para]):
            out.append((b, [(a, z, t, t) for a, z, t in ranges(b)]))
    return out


def overlaps(path, text, window, terms):
    return {(path, show, needle) for b, rng in blocks(path, text) if any(term_in(t, b) for t in terms)
            for a, z, show, needle in rng if a <= window[1] and z >= window[0]}


def history_pattern(window):
    """git -G (extended regex): a date range, or a quoted date, whose end year is not before the window's."""
    years = "|".join(str(y) for y in range(window[0].year, 2100))
    return (rf"[0-9]{{4}}(-[0-9]{{2}}){{0,2}} *(→|->|–|—|-|to|through|until) *({years})"
            rf'|"({years})-[0-9]{{2}}-[0-9]{{2}}')


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
    for chunk in git("log", "-G", history_pattern(win), "--format=%x00%h", "--name-only", "--", *CORPUS,
                     root=root).split("\x00")[1:]:
        h, *paths = chunk.split()
        for f in paths:
            found |= overlaps(f, git("show", f"{h}:{f}", root=root), win, terms)
    commit, when = committed(prereg, root) if prereg else (None, None)
    ack, ok = s.get("2a", ""), True
    lines = [f"{head}; instruments {', '.join(terms)}; pre-registration "
             + (f"added in {commit} {when}" if commit else "not committed, so every hit predates it")]
    by_file = {}
    for f, show, needle in sorted(found):
        h, t = first_published(f, needle, root)
        before = when is None or (t is not None and t < when)
        named = re.search(rf"(?<![\w/.-]){re.escape(f)}", ack) is not None or bool(h and h in ack)
        ok &= not (before and not named)
        by_file.setdefault(f, []).append((t, h, before, named, show))
    for f, hits in by_file.items():
        dated = [x for x in hits if x[0] is not None]
        t, h = min(dated, key=lambda x: x[0])[:2] if dated else (None, None)
        state = ("after the pre-registration" if not any(x[2] for x in hits) else
                 "before the pre-registration and not in 2a" if any(x[2] and not x[3] for x in hits) else
                 "before the pre-registration, named in 2a")
        shows = sorted({x[4] for x in hits})
        lines.append(f"{f}, first published {h} {t.date()}" if h else f"{f}, in no commit")
        lines[-1] += f", {state}: " + ", ".join(f'"{clip(x, 40)}"' for x in shows[:6]) + (
            f" (+{len(shows) - 6} more)" if len(shows) > 6 else "")
    if not found:
        lines.append("no README, REFUSALS, report or registry range overlaps the window on these instruments")
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
    m = re.search(LENGTH, src)
    n = int(m.group(1).replace(",", "")) if m else None
    b = re.search(BAR, src)
    bar = float(next(g for g in b.groups() if g is not None)) if b else None
    computed = no_edge_pass_rate(n, bar) if n and bar is not None else None
    stated = None
    for sent in sentences(sec or ""):
        if re.search(r"no.?edge|zero.?edge|\bnull\b", sent, flags=re.I):
            p = re.search(r"(\d+(?:\.\d+)?)\s*%", sent) or re.search(r"(?<![\d.])(0?\.\d+)(?![\d.])", sent)
            if p:
                stated = float(p.group(1)) / (100 if "%" in p.group(0) else 1)
                break
    rule = sec is not None and re.search(DECIDE, sec, flags=re.I) is not None
    agree = stated is not None and computed is not None and abs(stated - computed) <= 0.02
    where = "section 2b" if sec is not None else "section 5 (no section 2b)"
    if computed is not None:
        comp = (f"computed: P(Sharpe > {bar:g} over {n} sessions | no edge) = {computed:.1%}, normal approximation, "
                "SE sqrt((1 + 0.5 S^2) / N) at S = 0")
    else:
        comp = (f"computed: not possible, track length {n or 'absent'} (read only as 'N sessions'), Sharpe bar "
                f"{'absent' if bar is None else bar} (read only as 'Sharpe above X', 'Sharpe of at least X', "
                "'Sharpe >= X', 'net Sharpe ... X')")
    lines = [f"stated no-edge pass rate: {'absent' if stated is None else f'{stated:.1%}'}"
             + ("" if sec is not None else " (no section 2b)")
             + (f"; {abs(stated - computed) * 100:.1f} points from the computed rate"
                + ("" if agree else ", more than 2") if stated is not None and computed is not None else ""),
             f"{comp}; length and bar read from {where}",
             f"decision rule: {'stated' if rule else 'absent'} (a longer track, a shrunk estimate or Kelvin's judgment)"]
    if legs(s.get("2", "") + "\n\n" + s.get("3", ""), [])[1]:
        lines.append("WARNING: this book sells options. The normal-formula rate understates its no-edge pass rate "
                     "(a zero-mean short-vol stream clears the bar more often than a normal one), and the page must "
                     "state the share of P&L in its worst 1% of days")
    return bool(sec is not None and agree and rule), lines


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
