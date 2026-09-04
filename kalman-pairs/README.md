# Kalman-filter hedge ratios for pairs trading

Takes the 35 pairs, the trading rules and the formation/out-of-sample windows
from the [pairs-trading](../pairs-trading) project, and swaps the frozen OLS
hedge ratio for a Kalman filter that lets the ratio (and the intercept) drift as
a random walk. Everything else is held fixed so the only question being asked is
whether an adaptive hedge helps out of sample.

It does not. The static hedge ratio scores a net out-of-sample Sharpe of 0.08;
the filter, with its state noise tuned on the formation window, scores -0.16.
Nothing in the state-noise grid beats the frozen beta out of sample, and none of
the numbers is distinguishable from zero. Pushing the state noise up finds a
little more gross signal (gross Sharpe 0.51 at the top of the grid) and spends
all of it on turnover.

## Data

Same prices as the pairs project: daily adjusted closes from Yahoo via
`yfinance`, 2015-01-02 to 2026-08-28, for the 70 names in the 35 selected
pairs. `data.py` seeds its cache from `../pairs-trading/reports/prices.csv` when
that file exists, so the two projects run on byte-identical inputs. The pair list
with its static beta and intercept is read from
`../pairs-trading/reports/selected_pairs.csv`.

The survivorship problem is inherited unchanged: the universe is current S&P 500
membership filtered to names with continuous history through 2026, so
delisted and acquired names are absent. It biases both strategies the same way
and the comparison between them is unaffected; the level of either number is
not to be trusted on its own.

## Periods

| window | dates | used for |
|---|---|---|
| formation | 2015-01-02 to 2019-12-31 | state-noise choice; the static beta was fit here by the pairs project |
| out-of-sample | 2020-01-02 to 2026-08-28 | the headline comparison, nothing else |

The 35 pairs and the rules (lookback 90, entry 1.5, exit 0.0, stop 4, 60-day
maximum hold, 10bps round trip) were chosen by the pairs project on the
formation window. I do not re-tune them; that would make the comparison a
comparison of two searches rather than two hedge ratios.

## Method

### The filter

`kalman.py` is about thirty lines of numpy, no library. For one pair with
`y = log A` and `x = log B`:

```
state        s_t = [beta_t, alpha_t]
transition   s_t = s_{t-1} + w_t       w_t ~ N(0, Q),   Q = q * R * I
observation  y_t = beta_t x_t + alpha_t + v_t     v_t ~ N(0, R)
```

Each day: inflate the state covariance by Q, form the prediction error
`e_t = y_t - (beta_{t-1} x_t + alpha_{t-1})`, compute the gain, update. The
state at row t uses prices through t and nothing later; `check.py` bumps one
price and confirms every earlier output is unchanged.

`R` is the variance of the static OLS residual on the formation window, per
pair. `q` is the ratio of state noise to observation noise and is the single
parameter this project introduces. At `q = 0` the state is constant and the
filter is recursive least squares: started from a diffuse prior it reproduces
the full-sample OLS fit to 1e-4 (checked), which means the `q = 0` filter at
2019-12-31 holds exactly the pairs project's static beta and thereafter keeps
re-estimating it on an expanding window.

### The spread

The Kalman spread is the innovation `e_t`, the gap between today's price and
where yesterday's hedge ratio said it would be. Using the post-update residual
instead would shrink the spread toward zero by construction as the filter
absorbs each move. The innovation then goes through the same trailing z-score
and the same entry/exit rules as the static spread, imported from
`../pairs-trading/pairs.py` so there is one copy of the logic.

### Trading with a moving hedge

Signal at t, fill at t+1. The hedge ratio applied to day t's return is the
estimate from t-1, the day the position was decided. Turnover counts both legs
separately, so while a position is open a drifting beta pays to rebalance the
short leg every day; the static path pays nothing between entry and exit. The
static path through my `pair_backtest` reproduces the pairs project's
`pair_backtest` to floating-point precision (checked).

### Choosing q

Grid `{0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`, scored on formation-window net
Sharpe with the filter run on formation prices only. `check.py` replaces
everything after the formation window with noise and confirms the choice and
every formation score are unchanged.

## Results

The chosen state noise is `q = 0`, the bottom of the grid.

| | static, formation | static, OOS | Kalman q=0, formation | Kalman q=0, OOS |
|---|---|---|---|---|
| annual return, gross | +6.60% | +0.72% | +4.58% | -0.19% |
| annual return, net | +6.11% | +0.25% | +4.07% | -0.67% |
| annual vol | 1.98% | 3.86% | 2.47% | 3.81% |
| Sharpe, gross | 3.24 | 0.21 | 1.83 | -0.03 |
| Sharpe, net | 3.01 | 0.08 | 1.63 | -0.16 |
| max drawdown | -1.26% | -7.19% | -1.81% | -9.01% |
| trades | 809 | 1,127 | 847 | 1,138 |
| hit rate | 77.4% | 62.9% | 72.1% | 62.0% |
| average hold | 30 days | 34 days | 29 days | 33 days |
| annual turnover | 9.1x | 9.5x | 9.9x | 9.7x |

The formation columns are not comparable and I want to be explicit about why.
The static beta on formation is the full-window OLS fit scored on the same
window: it saw every day of 2015-2019 before being scored on any of them. The
filter at day t has seen days 1 to t. So 3.01 against 1.63 is hindsight against
no hindsight, and the gap is a measure of how much of the pairs project's
in-sample Sharpe was the hedge ratio peeking. The out-of-sample columns are
the fair comparison. There the static ratio is 0.08 and the filter is -0.16.

### State-noise sensitivity

| q | formation Sharpe | OOS Sharpe, net | OOS Sharpe, gross | OOS net return/yr | OOS turnover/yr | OOS trades |
|---|---|---|---|---|---|---|
| 0 | 1.63 | -0.16 | -0.03 | -0.67% | 9.7x | 1,138 |
| 1e-7 | 1.61 | -0.19 | -0.06 | -0.77% | 9.6x | 1,127 |
| 1e-6 | 1.43 | 0.02 | 0.15 | +0.01% | 9.8x | 1,157 |
| 1e-5 | 1.20 | -0.01 | 0.14 | -0.09% | 10.4x | 1,221 |
| 1e-4 | 0.80 | -0.11 | 0.07 | -0.44% | 12.5x | 1,458 |
| 1e-3 | 0.41 | -0.10 | 0.19 | -0.39% | 18.9x | 2,194 |
| 1e-2 | 0.09 | -0.04 | 0.51 | -0.14% | 30.2x | 3,502 |
| static | 3.01 | 0.08 | 0.21 | +0.25% | 9.5x | 1,127 |

Only the formation column was used to pick q; the rest is shown so the reader
can see what the choice cost. Three things in this table:

Formation Sharpe falls monotonically in q, so the tuning lands on the edge of
the grid. Adaptivity looked worse and worse on the formation data at every step.

Out of sample the net column is flat around zero for every q. The best net
number, 0.02 at `q = 1e-6`, would have been picked by nobody and is not
different from -0.16 given seven years of daily data at 3.8% vol (standard
error on Sharpe roughly 0.4).

The gross column does rise with q, to 0.51 at `q = 1e-2`. A fast-moving hedge
does find something: the innovation spread is closer to white noise, the z-score
fires more often, and the trades are slightly better than coin flips before
costs. But turnover triples to 30x a year, 3,502 trades against 1,127, and at
10bps round trip that is about 150bps a year of cost against roughly 140bps
of gross return.
Whatever the filter finds it spends on rebalancing. `reports/noise_sensitivity.png`
shows the two Sharpe lines against q.

### By year, out of sample

| year | static, net | static Sharpe | Kalman q=0, net | Kalman Sharpe |
|---|---|---|---|---|
| 2020 | -0.16% | 0.01 | -1.00% | -0.12 |
| 2021 | -2.73% | -0.97 | -3.06% | -1.17 |
| 2022 | +0.40% | 0.13 | -1.55% | -0.41 |
| 2023 | +4.20% | 1.53 | +3.24% | 1.16 |
| 2024 | -0.70% | -0.25 | +0.74% | 0.30 |
| 2025 | +2.47% | 0.83 | -1.65% | -0.51 |
| 2026 (to Aug) | -1.68% | -0.65 | -1.03% | -0.38 |

Same shape, same bad years. The two are not doing different things; 2023 is
good for both and 2021 is bad for both.

### Per pair

Out-of-sample net Sharpe per pair: static median 0.03 with 19 of 35 positive,
Kalman median -0.04 with 16 positive. The filter beats the frozen ratio on 18
pairs and loses on 17. `reports/pair_sharpes.png` is a scatter of one against
the other and it is a cloud around the diagonal.

The hedge ratios did move. Median absolute change in the `q = 0` beta over the
out-of-sample window is 0.27; NOC/UPS went from 2.31 to 0.91. The question was
never whether the ratios drift, it was whether tracking the drift makes the
spread more tradable, and the answer on this data is no.

`reports/sample_pair.png` is BDX/ISRG, the pair with the lowest formation
p-value. The static spread walks from 0 to -0.8 between 2020 and 2025. The
expanding-window beta follows the relationship down from 0.51 to 0.25 and the
innovation spread stays closer to zero, but it still drifts to -0.5 in 2025.
The relationship broke; a slower or faster hedge ratio does not un-break it.

### Costs

| round trip | static OOS Sharpe | Kalman q=0 OOS Sharpe |
|---|---|---|
| 0bps | 0.21 | -0.03 |
| 5bps | 0.14 | -0.09 |
| 10bps | 0.08 | -0.16 |
| 20bps | -0.04 | -0.28 |
| 30bps | -0.16 | -0.41 |

The filter is below the static ratio at every cost level, including zero, so
this is not a cost story at the chosen q. It is a cost story at high q, per the
sensitivity table.

## What I take from it

The pairs project's in-sample Sharpe of 3.01 falls to 1.63 the moment the hedge
ratio is estimated sequentially instead of with the full window, on the same
pairs and the same dates. That is the cleanest number here and it is a
statement about the static method, not the filter.

Out of sample, the hedge ratio is not the problem. The 35 pairs were chosen by a
screen that passes about 74 pairs by chance out of 1,480, and the relationships
mostly did not persist. No amount of adaptivity in the hedge recovers a
cointegrating relation that stopped existing.

## Look-ahead audit

- The filter is a single forward pass; row t uses `y[:t+1]` and `x[:t+1]`.
  Mutation test in `check.py`.
- The innovation at t uses the state from t-1 (checked against the explicit
  formula).
- Execution lag is the pairs project's `.shift(1)` on the target position;
  the hedge ratio applied to day t's return is `beta.shift(1)` (checked).
- `R` per pair is the formation-window residual variance. `q` is chosen on a
  filter run over formation prices only (mutation test on post-formation data).
- Scoring windows are disjoint and the out-of-sample window starts after
  formation ends (checked).
- Warm-up: the out-of-sample filter starts from its 2019-12-31 state rather
  than from a diffuse prior. That state is a function of past prices only.
- Inherited from the pairs project and not fixed: the universe was filtered on
  history through 2026, and the 35 pairs were chosen on the formation window
  using a screen with a large multiple-testing problem.

## Limitations

**One q for all pairs, and a scalar Q.** Beta and the intercept get the same
state variance even though `x` sits near 4 and the intercept absorbs most of
any level shift. Per-pair q, or a Q estimated by maximum likelihood on the
innovations, would be the standard next step. I did not do it because the grid
already says the answer is at the bottom, and a finer search on formation data
would only find a better-fitting number to fail with out of sample.

**q lands on the grid edge.** The tuning picked 0, the smallest value offered.
A grid that extended lower would pick something lower; this is the same as
saying formation data prefers no adaptivity at all.

**Near-collinearity of beta and alpha.** With log prices around 4 and daily
moves around 1%, a change of 0.1 in beta looks almost identical to a change of
0.4 in alpha over any short window. The filter cannot separate them quickly and
the synthetic check reflects that: it needs x to move a fair amount before the
tracking advantage over OLS shows up. On real pairs this means the beta path
is smoother than the true relationship and the intercept absorbs the fast
part.

**Everything inherited.** Survivorship, one split with no re-formation, a flat
10bps cost with no borrow or impact, no risk overlay. See the pairs README.

**Statistical power.** Seven years at 3.8% vol gives a standard error on
annualised Sharpe of about 0.4. The gap between 0.08 and -0.16 is half of one
standard error. The honest statement is that both are zero, not that the static
ratio won.

## Files

- `data.py` shared pairs and prices from the pairs project, cache under `../source-material/kalman-pairs/`
- `kalman.py` the filter in numpy, plus the OLS it should collapse to
- `strategy.py` static and Kalman spreads through the pairs project's trading rules; q tuning
- `check.py` offline checks on synthetic pairs
- `run.py` full pipeline, tables and charts to `reports/`, console copy in `reports/run_log.txt`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline and exits nonzero on any failure. `run.py` needs
`../pairs-trading/reports/selected_pairs.csv`; it copies the pairs project's
price file if present, otherwise downloads once.
