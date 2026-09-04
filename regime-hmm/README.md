# Market regime detection with a hidden Markov model

Daily SPY returns look like they come from two distributions: a calm one that
drifts up and a violent one that drifts down. A Gaussian hidden Markov model
formalises that. Each day the market is in one of K unobserved states, each
state has its own mean and variance, and the state follows a Markov chain. EM
recovers the parameters, and the forward algorithm gives a probability of being
in each state today from returns up to today.

The question I wanted to answer is not whether the regimes exist (they
obviously do, in the sense that vol clusters) but whether knowing the filtered
regime probability in real time is worth anything as a position signal, after
costs, and against the obvious cheaper alternative of just scaling exposure by
trailing volatility.

Short version: the 2-state HMM strategy beats buy-and-hold on Sharpe out of
sample (0.90 vs 0.86 net) and cuts the worst drawdown from -34% to -14%, but
it does not beat a 15% vol-targeting rule (0.97), and it gives up a third of the
return to do it. The in-sample version of the same strategy shows a Sharpe of
1.88. That number is the trap this project is about.

## Data

SPY adjusted close from yfinance, 2000-01-03 to 2026-08-31, 6,704 daily
returns. `data.py` downloads once and caches to
`../source-material/regime-hmm/spy.csv`. The model is fit on 100 x log return;
the backtest uses simple returns.

Survivorship is not an issue for a single index ETF in the usual sense, but the
choice of asset is itself a survivor: US large-cap equities over 2000-2026 is
one of the best-performing series in the world, and "stay long except in
high-vol periods" is a rule tuned to a series that always recovered. On an
index that had a lost decade, the same rule would look different.

## Method

**Model.** `hmmlearn.GaussianHMM`, diagonal covariance, one feature (the daily
return), EM for up to 500 iterations with tolerance 1e-6, fixed seed. After
fitting I permute the states so state 0 is always the lowest-variance state.
EM is invariant to relabelling, so without that step "state 0" means whatever
the k-means initialisation happened to produce, and a refit next year could
silently swap bull and bear. `check.py` confirms two seeds give the same sorted
parameters.

**Filtered vs smoothed.** `hmmlearn.predict_proba` returns smoothed
probabilities, P(state_t | all returns through the end of the sample). That is
the picture everyone shows, with the shaded bear regimes lining up neatly with
2008 and 2020. It is not tradeable: the probability on 15 Sep 2008 is computed
knowing what happened in October. `regime.filtered` is the forward recursion
only, P(state_t | returns through t). That is the quantity you actually have
at the close of day t.

**Strategy.** Exposure at the close of t is 1 - P(highest-vol state | returns
through t), held over day t+1. Long-only, unlevered, so exposure sits in
[0, 1]. Costs are 5bps one-way on the change in exposure.

**Out of sample.** The first test year is 2010. Each January the model is refit
on every return before that year (expanding window, so the 2010 fit sees
2000-2009 and the 2026 fit sees 2000-2025). The filter is then run with that
frozen model over history up to the end of the year and only that year's rows
are kept. Nothing about a year's exposure depends on data from that year's
parameters or any later year. `check.py` verifies this by truncating the sample
at 2017 and confirming the exposure through 2017 is bit-identical.

**Benchmarks.** Buy-and-hold, and a vol-targeting rule: exposure = 15% /
trailing 20-day EWM realised vol, capped at 1. The cap makes it unlevered like
the HMM rule. This is the honest comparison, because most of what an HMM knows
about regimes is that vol is high.

**The trap, deliberately included.** `trap_insample_smoothed` is what you get by
fitting once on 2000-2026 and trading the smoothed probability. I also include
`trap_insample_filtered`, the same full-sample fit but with the causal filter,
to separate the two leaks: knowing the future through the parameters, and
knowing it through the probabilities.

## State characterisation

2-state fit on the full sample:

| | annual mean | annual vol | P(stay) | expected duration | long-run share |
|---|---|---|---|---|---|
| state 0 (bull) | +22.7% | 11.1% | 0.987 | 79 days | 70% |
| state 1 (bear) | -26.6% | 30.8% | 0.970 | 33 days | 30% |

3-state fit:

| | annual mean | annual vol | P(stay) | expected duration | long-run share |
|---|---|---|---|---|---|
| state 0 | +30.2% | 8.6% | 0.972 | 36 days | 48% |
| state 1 | -4.2% | 18.9% | 0.962 | 26 days | 45% |
| state 2 | -55.6% | 46.9% | 0.952 | 21 days | 8% |

The 3-state transition matrix has one structural feature worth reading off:
the probability of going straight from the calm state to the crisis state is
0.0000, and straight from crisis to calm is 0.0000. The market goes through
the middle state in both directions. That is why the 3-state strategy trades
so much less than the 2-state one (4.3x vs 13.1x annual turnover): it only
de-risks on the crisis state, and the crisis state is rare.

The parameters are stable across the 17 rolling refits. The bear-state vol
sits between 29.6% and 33.1% every year; the bull mean drifts up from 14% to
23% as the post-2009 bull market accumulates. `reports/parameter_drift.png`.

## Results

2010 to August 2026, 5bps one-way, daily rebalanced. The `hmm_*` rows are
out of sample with annual refits. The `trap_*` rows are the full-sample fit
evaluated on the same window and are there to be laughed at.

| | return, gross | return, net | vol | Sharpe, gross | Sharpe, net | max drawdown | turnover | avg exposure |
|---|---|---|---|---|---|---|---|---|
| buy-and-hold | +14.18% | +14.18% | 17.07% | 0.86 | 0.86 | -33.7% | 0.1x | 100% |
| vol-target 15% | +12.43% | +12.25% | 12.72% | 0.99 | 0.97 | -18.7% | 3.2x | 90% |
| HMM 2-state | +9.78% | +9.06% | 10.28% | 0.96 | 0.90 | -13.8% | 13.1x | 78% |
| HMM 3-state | +12.91% | +12.67% | 13.91% | 0.94 | 0.93 | -22.2% | 4.3x | 95% |
| trap: in-sample, smoothed | +18.67% | +18.42% | 9.20% | 1.91 | 1.88 | -9.8% | 4.3x | 77% |
| trap: in-sample, filtered | +9.55% | +8.75% | 10.08% | 0.96 | 0.88 | -12.7% | 14.7x | 76% |

Three things to take from this.

The out-of-sample HMM is a real but small improvement over buy-and-hold on a
risk-adjusted basis, and a large one on drawdown. It is not an improvement on
vol-targeting. Vol-targeting has a higher net Sharpe, a higher return, and a
quarter of the turnover. The 3-state model is closer to vol-targeting on every
column, because it has effectively become a vol-targeting rule with a higher
threshold.

The smoothed in-sample strategy has a Sharpe of 1.88, double anything
achievable. Almost all of that comes from the smoothing, not from the
full-sample parameters: the in-sample fit with the causal filter gets 0.88,
essentially the same as the rolling refit at 0.90. So the parameters are not
where the look-ahead lives. The look-ahead lives in the backward pass, which
knows on the first day of a selloff that it is the first day of a selloff.

The 2-state HMM trades a lot. 13x annual turnover means the filtered bear
probability oscillates. Going from 5bps to 20bps one-way takes its net Sharpe
from 0.90 to 0.70; vol-targeting goes from 0.97 to 0.93 over the same range.

Levering every rule to buy-and-hold's 17% realised vol (leverage allowed, costs
scale with it) gives net Sharpe 0.97 / 0.90 / 0.93 for vol-target / HMM-2 /
HMM-3, unchanged by construction, and net returns of 16.3% / 14.8% / 15.5%
against 14.2% for buy-and-hold. `reports/performance_vol_matched.csv`.

By year, net:

| year | buy-and-hold | vol-target | HMM 2-state | HMM 3-state |
|---|---|---|---|---|
| 2010 | +13.1% | +13.7% | +9.0% | +13.8% |
| 2011 | +1.9% | -1.8% | -4.5% | -8.6% |
| 2012 | +16.0% | +13.8% | +11.7% | +15.8% |
| 2013 | +32.3% | +32.0% | +29.8% | +32.1% |
| 2014 | +13.5% | +12.6% | +11.6% | +13.3% |
| 2015 | +1.2% | -0.8% | -4.1% | -2.2% |
| 2016 | +12.0% | +9.8% | +4.5% | +11.0% |
| 2017 | +21.7% | +21.7% | +21.4% | +21.7% |
| 2018 | -4.6% | -5.0% | -0.8% | -9.6% |
| 2019 | +31.2% | +25.1% | +15.1% | +26.1% |
| 2020 | +18.3% | +13.1% | +8.7% | +19.9% |
| 2021 | +28.7% | +26.1% | +12.9% | +28.3% |
| 2022 | -18.2% | -14.7% | -7.1% | -14.3% |
| 2023 | +26.2% | +23.7% | +16.6% | +25.9% |
| 2024 | +24.9% | +22.4% | +16.9% | +24.4% |
| 2025 | +17.7% | +12.4% | +12.3% | +14.9% |
| 2026 (to Aug, not annualised) | +13.1% | +11.0% | +3.6% | +12.8% |

The 2-state strategy beats buy-and-hold in exactly two years, 2018 and 2022,
the two down years. Everywhere else it lags, sometimes badly (2019: 15% vs
31%). That is the whole trade: pay a few points a year in bull markets for
smaller losses in bear markets. Whether that is worth it depends on the
investor, not on the Sharpe.

Stress windows, net return and average exposure held:

| window | buy-and-hold | vol-target | HMM 2 | HMM 3 | HMM 2 exposure |
|---|---|---|---|---|---|
| 2011 Aug | -17.8% | -14.8% | -10.1% | -19.7% | 0.20 |
| 2018 Q4 | -18.9% | -15.5% | -7.1% | -18.5% | 0.26 |
| 2020 Covid crash (Feb 19 to Mar 23) | -33.4% | -14.6% | -4.6% | -9.9% | 0.17 |
| 2020 recovery (Mar 24 to Dec 31) | +69.8% | +26.5% | +10.5% | +27.4% | 0.43 |
| 2022 | -18.2% | -14.7% | -7.1% | -14.3% | 0.26 |

The 2020 row is the strategy in one line. It lost 4.6% in the crash against
33.4% for the index, then made 10.5% in the recovery against 69.8%. Filtered
probabilities react to a vol spike within days, but a vol spike is also what
the bottom looks like, so the model is at its most defensive at the moment of
maximum opportunity. The 3-state model, which only de-risks on the extreme
state, held 0.79 average exposure through the recovery and captured most of it.

## Look-ahead audit

- Parameters: refit each January on strictly prior years. Verified by the
  truncation test in `check.py`.
- Probabilities: the strategy uses `regime.filtered`, a forward pass only.
  Verified by the mutation test in `check.py`: replacing every return after
  t=4000 with noise leaves the filtered probabilities before t=4000 unchanged
  to machine precision, and changes the smoothed ones (so the test cannot pass
  vacuously).
- Execution: `backtest.run` shifts the exposure by one day before multiplying
  by returns. Nothing reads the unshifted series.
- Vol-targeting: `ewm(span=20).std()` is trailing. Same shift.
- The one soft spot: the vol-target parameters (15%, 20 days) and the HMM
  choices (K=2 or 3, Gaussian, annual refit, expanding window) were fixed before
  I looked at any result, but I did look at the results before deciding what
  else to report. The stress-window table in particular was chosen after seeing
  the yearly numbers.

## Limitations

**It is mostly a vol signal.** Two Gaussians with different variances is a
crude stochastic volatility model. The mean parameters are poorly identified
(a -26.6% annual drift on a state that lasts 33 days is a 0.1% daily mean
against a 1.9% daily standard deviation), so what the filter actually detects is
a variance change. That is why vol-targeting matches or beats it. A fairer
test of whether the HMM adds anything beyond vol would condition on realised
vol and ask whether the regime probability has residual predictive power. I did
not do that.

**Gaussian tails.** Daily SPY returns have a kurtosis around 12; a mixture of
two Gaussians gets some of the way there and not all. Fat-tailed emissions
(Student-t) would give a less jumpy filtered probability and probably cut the
turnover.

**Annual refit is arbitrary.** Monthly refits would be more realistic and
would let the model absorb a crisis into its parameters sooner. Expanding
window also means the 2026 fit has 26 years of history, so it will never adapt
to a genuine structural change. A rolling 10-year window is the obvious
alternative and I would expect it to trade more.

**One asset, one sample.** Seventeen years of test data with two real bear
markets is not a lot of independent regime transitions. The 2-state model's
Sharpe edge over buy-and-hold (0.90 vs 0.86) is well within noise. The drawdown
reduction is more robust because it is mechanical: the rule cannot be fully
long during a vol spike.

**Long-only.** Going short in the bear state would be the natural extension
and would also be a much larger bet on the mean parameters, which are the
parameters I trust least.

**Costs.** 5bps one-way on SPY is realistic for size that does not move the
market. The daily rebalancing at fractional exposures is unrealistic; a real
implementation would rebalance on a threshold and the turnover figures would
drop, mostly for the 2-state model.

## Files

- `data.py` downloads and caches SPY
- `regime.py` fit, state sorting, description, filtered and smoothed probabilities
- `backtest.py` rolling out-of-sample exposure, benchmarks, return engine, metrics
- `check.py` synthetic tests, exits nonzero on failure
- `run.py` the full pipeline; writes charts and CSVs to `reports/` and the console output to `reports/run_log.txt`

Run with the shared venv at `../.venv`:

```
python3 check.py
python3 run.py
```
