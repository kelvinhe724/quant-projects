# Kalshi Desk

A prediction-market desk that forward-tests the favorite-longshot research in
`../prediction-markets` on Kalshi's demo exchange with fake money, and reads a
real account when a key is present. The calibration curve and Kelly maths are
imported from that project, not copied, so the forward test scores markets with
exactly the model the backtest used.

## Why demo first

The backtest (`../prediction-markets/reports/results.txt`) said:

> mid to mid, no cost: mean payoff per dollar staked +0.0241 (t = +3.00)
> quoted spread: mean payoff per dollar staked -0.0120 (t = -1.28)

The bias is real and the textbook sign, and it is smaller than the cost of
taking it. +2.4 cents at the mid becomes -1.2 cents once you pay the spread that
was actually quoted. A strategy with that profile does not get real money; it
gets a forward test that either reproduces the number or tells us the backtest
was wrong about something. Demo fills are free, and the scoring still reads the
production order book, so the signal is real even when the fill is not.

Live trading is off by default. `live.py` refuses to run unless
`LIVE_TRADING=true` is set in `.env` (or the environment), the phrase
`place real money orders` is typed at the prompt, and every order fits under
the dollar caps. There is no flag to skip the prompt, and a closed stdin
counts as a wrong phrase. The open-dollar cap is measured from this ledger,
not from the account, so positions opened elsewhere do not count against it. This is deliberate: the research says the edge does
not survive the spread, so production should only ever be a conscious act.

## The three books

One ledger (`ledger.sqlite`), one `book` column, three values. Same scanner,
same sizing, same caps, same statistics, so the rows are directly comparable.

| book | account | fill | settlement |
|---|---|---|---|
| `shadow` | none | simulated: the public orderbook is re-read at placement and the fill is written at the quoted touch if the order would have crossed it at its full size | public `GET /markets/{ticker}` result |
| `demo` | demo exchange, fake money | real, from the demo matching engine | same |
| `live` | production, real money | real | same |

Why shadow exists: Kalshi will not open an account (demo included) without
identity verification, and the strategy needs contracts to accumulate before
that is done. Public data is enough for that. Shadow is also the control once
a demo key arrives: it runs beside the demo book with no market
impact, and any gap between the two is the cost of actually being in the book
(queue position, partial fills, the demo engine's own quirks).

Shadow is optimistic by construction. It assumes the whole order fills at the
touch the instant it is placed, which is the best case for a taker. Treat its
P&L as an upper bound and its sign as the thing being tested.

If no demo key is configured, `paper.py place` and `paper.py poll` say so in
one line and run the shadow book instead; the dashboard opens on the shadow
book. If a demo key exists, demo is the default and shadow keeps running as the
control. The book selector in the header switches every panel.

## Files

| file | what |
|---|---|
| `kalshi.py` | thin API client: RSA-PSS signing, retry with backoff on 429/5xx, cursor pagination, demo/prod switch |
| `scanner.py` | public data only. Pulls open markets closing in 6h to 72h, keeps tight two-sided books, fetches the live orderbook for the top 40, scores each with the imported bias curve, ranks, caches to `../source-material/kalshi-desk/` |
| `paper.py` | the demo book: place capped orders, poll fills and settlements, SQLite ledger, P&L with the backtest's statistics. Falls through to shadow when there is no key |
| `shadow.py` | the shadow book: same placement code, simulated fill against the public touch, public settlement, `run --every 15m` scheduler. Its client cannot sign a request, so it never touches an account |
| `live.py` | production placement through the same code path, three locks on the door |
| `desk.py` + `index.html` | local dashboard on port 6161 |
| `check.py` | offline tests, exit nonzero on failure |
| `walk.py` | press-Enter walkthrough of one market end to end |

## Setup

```
cd quant-projects
.venv/bin/python -m pip install cryptography    # already in requirements.txt
cd kalshi-desk
cp .env.example .env
```

The scanner, the shadow book and the dashboard work with no key. That is the
whole setup until an account exists: `../.venv/bin/python check.py`, then
`../.venv/bin/python shadow.py place`. Everything that touches an account needs a key, and
demo and production keys are separate accounts with separate keys.

1. Demo account: sign up at https://demo.kalshi.co (a normal Kalshi login does
   not carry over).
2. Click your profile picture, top right, then Account Settings. On the profile
   page find the "API Keys" section and click "Create New API Key".
3. Kalshi shows the Key ID and downloads the private key as a `.txt` once. It
   cannot be retrieved again. Save the file as `keys/kalshi-demo.key` inside
   this folder (`keys/` is gitignored).
4. Put the Key ID in `.env` as `KALSHI_DEMO_KEY_ID=...` and leave
   `KALSHI_DEMO_KEY_PATH=keys/kalshi-demo.key`.
5. `python paper.py poll` should print `ledger is empty` rather than a key error.

For production repeat the same steps at https://kalshi.com/account/profile and
fill `KALSHI_PROD_KEY_ID` / `KALSHI_PROD_KEY_PATH`. That only enables the
account panel and `live.py`; nothing is sent until the locks above are opened.

If a key is missing, every authenticated path fails with one line saying which
key, where to create it, and which `.env` variables to fill.

## Running it

```
../.venv/bin/python check.py            # offline tests
../.venv/bin/python scanner.py          # ranked table, no key needed
../.venv/bin/python shadow.py place 5   # top 5 fades, simulated against the public touch
../.venv/bin/python shadow.py settle    # public settlements and marks for the shadow book
../.venv/bin/python paper.py place 5    # top 5 fades on the demo exchange (shadow if no key)
../.venv/bin/python paper.py poll       # fills, order status, settlements, marks
../.venv/bin/python paper.py stats      # every book vs the backtest
../.venv/bin/python desk.py             # http://127.0.0.1:6161
```

To accumulate shadow contracts unattended, either keep one process running:

```
../.venv/bin/python shadow.py run --every 15m
```

or put the place+settle pair on cron (absolute paths; cron has no cwd):

```
*/15 * * * * cd /path/to/quant-projects/kalshi-desk && (../.venv/bin/python shadow.py place 5 && ../.venv/bin/python shadow.py settle) >> shadow.log 2>&1
```

Both are idempotent. A ticker already filled in the shadow book is skipped on
the next pass, the fill id is the ticker so a second write is ignored, and
settlements are keyed by ticker. Running the cycle twice in a row adds new
tickers, never a second copy of an old one.

`paper.py place` reads the balance from the demo account, runs the scanner
against production books, skips tickers already in the ledger, and sizes each
order as `min(KELLY_FRACTION * f * bankroll, MAX_ORDER_DOLLARS)` with the
running total capped at `MAX_OPEN_DOLLARS` minus what is already at risk.
Orders are GTC limits at the touch (taker) with a ten-minute expiry, which is
the "quoted spread" scenario of the backtest. Run `poll` a few times a day; a
launchd or cron entry every hour is plenty.

`stats` reports mean payoff per dollar staked, its t-statistic and the Brier
score over settled contracts, one line per book, next to the backtest's
-0.0120 / -1.28 / 0.1527. Each book's statistics only ever see that book's
rows; `check.py` guards this.
Fees are recorded per fill and shown as a separate net line; the backtest did
not charge fees, so the gross line is the like-for-like one.

## What the scanner scores

For each market: mid from the live touch, model probability from the fitted
bias curve (`calibration.bias_curve` over `reports/calibration_quoted_mid.csv`),
edge at the mid and edge after crossing the quoted spread, exact Kelly fraction
(`kelly.kelly_fraction`) on the better side, and the capped stake. Ranking is
by edge after the spread. Expect the 0.40 to 0.50 bin to dominate: that is
where the research found the largest bias (+4.9 cents) in tight books.

An optional signal: `SCANNER_MODEL=kalshi-model ../.venv/bin/python scanner.py`
scores each market with the registered calibrator from `../kalshi-model`
instead of the bin curve (it pulls each candidate's candles, so the scan is
slower). The default is unchanged; that project's untouched window found the
market mid better than either curve, so the swap is for comparison, not an
upgrade.

## Caveats carried over from the research

The sample was 77% sports props, the horizon was 24h, and the bias in tight
books was under a cent outside the very cheap end. The scanner filters to
books 10 cents wide or tighter and 6h to 72h from close, which is the closest
match to those conditions the live listing allows. Fills at the touch on a
market trading a few hundred contracts will be worse than the quote, not
better. A few dozen settled contracts prove nothing either way; the backtest
t-stat came from 4,223.

## API notes

Verified against https://docs.kalshi.com on 2026-09-04. Order placement uses
the V2 endpoint `POST /portfolio/events/orders` (side `bid`/`ask` on the YES
leg, fixed-point dollar strings); the legacy `POST /portfolio/orders` is
deprecated. Cancel is `DELETE /portfolio/events/orders/{id}?market_ticker=`.
Signing is `timestamp_ms + METHOD + path` with RSA-PSS SHA-256, path from the
host root and without the query string. Demo host `demo-api.kalshi.co` is the
supported alias of `external-api.demo.kalshi.co`.
