# Does the yield curve predict recessions? Probit on the 10y-3m spread, 1962-2026

The Estrella-Mishkin result: a probit of "NBER recession within the next 12
months" on the 10-year minus 3-month Treasury spread has real forecasting power,
and an inverted curve has preceded every US recession since the 1960s. I rebuilt
it from FRED data, scored it properly out of sample with an expanding window,
compared it to the Sahm rule and to a naive base rate, and looked at what the
model said month by month during the 2022-2024 inversion, which was the deepest
and longest since 1981 and has not, as of the August 2026 data, been followed by
a recession.

The short version: the spread carries information, but less than the in-sample
fit suggests, the probabilities it produces are not well calibrated at the top,
and with eight recessions in the sample almost everything about it is a
small-number problem.

## Data

All from FRED, cached under `source-material/yield-curve-recession/` so reruns
are offline.

| Series | Use |
|---|---|
| GS10, TB3MS | Monthly-average 10y constant maturity and 3m bill. Their difference is the 10y-3m spread back to 1962. FRED's own T10Y3M only starts in 1982. |
| GS2 | 10y-2y spread, from 1976 |
| USREC | NBER recession months, 1 or 0 |
| UNRATE | Unemployment rate, for the Sahm rule |
| SAHMREALTIME | FRED's real-time Sahm series, for a sanity check of my own construction |
| DGS10, DGS2, DGS3MO, T10Y3M, T10Y2Y | Daily, for the 2022-24 chart |

Sample is 1962-01 to 2026-08, 776 months, 8 recessions, 85 recession months.
The label for month t is 1 if any of months t+1 to t+12 is a recession month, so
it is defined through 2025-08 and is 1 in 22.6% of months. BLS did not publish
an October 2025 unemployment number because of the shutdown; that one month is
interpolated so the Sahm windows do not break. The August 2026 unemployment
rate was not out when the data was pulled, so the Sahm gap ends in July 2026.

## Method

**Probit.** P(recession within 12m) = Phi(a + b * spread). Fit by maximum
likelihood. Consecutive labels share eleven months of the same future, so the
usual standard errors are wrong; I use HAC with 11 lags.

**Out of sample.** Starting 1980-01, refit every month on the data that was
knowable at that date and forecast the current month. Knowable means the label
for month s requires USREC through s+12, so a forecast at t trains on rows
s <= t-12 only. I also run a version with a further 12-month announcement lag,
since NBER dates business cycle peaks six to twelve months after the fact. The
feature (the spread) is never revised, so there is no vintage problem on that
side.

**Comparisons.** A base rate (expanding mean of the labels known at t), a
probit on the Sahm gap alone, and a probit on spread plus Sahm gap. The Sahm
gap is the 3-month average unemployment rate minus its low over the prior 12
months, from final-vintage UNRATE. It is a rule that says a recession has
already started, not a 12-month-ahead forecast, so it is being asked to do a job
it was not designed for; that is the point of the comparison.

**Scoring.** AUC, Brier score, and Brier relative to the base rate. Calibration
by forecast decile.

**Checks.** `check.py` builds a synthetic spread with a planted probit
relationship and confirms the fit recovers it, that a noise regressor gets a
zero coefficient, that mutating every observation after a cut date leaves every
forecast up to that date bit-identical, that a flipped label at date t first
enters the forecast at t+12 (t+12+lag with an announcement lag), that a random
predictor scores AUC 0.5, and that a perfectly calibrated synthetic forecast sits
on the calibration diagonal. 30 checks, exits nonzero on any failure.

## Results

All numbers from `run.py`, tables in `reports/`.

### In-sample fit, 1962-2025

| Model | n | Coefficient | HAC t | Pseudo R2 | AUC |
|---|---|---|---|---|---|
| 10y-3m spread | 764 | -0.420 | -3.43 | 0.123 | 0.747 |
| 10y-2y spread | 591 | -0.589 | -2.72 | 0.118 | 0.736 |
| Sahm gap | 750 | +0.162 | 1.32 | 0.014 | 0.723 |
| Spread + Sahm | 750 | -0.469 / +0.263 | -3.88 / 1.43 | 0.158 | 0.793 |

Intercept for the spread model is -0.259. What that means in probabilities:

| Spread | P(recession within 12m) |
|---|---|
| +1.0 | 0.25 |
| 0.0 | 0.40 |
| -0.5 | 0.48 |
| -1.0 | 0.56 |

A flat curve is roughly a coin flip and a one-point inversion is only 56%. The
coefficient depends heavily on where the sample ends: -0.607 through 1995,
-0.552 through 2019, -0.420 through 2025. One episode with no recession moved
it by a quarter.

### Inversion episodes

Monthly-average 10y-3m below zero. "Within 12m" counts an ongoing recession.

| Start | Months inverted | Recession within 12m | Months to next recession |
|---|---|---|---|
| 1966-09 | 5 | no | 40 |
| 1969-01 | 1 | yes | 12 |
| 1969-07 | 2 | yes | 6 |
| 1969-11 | 3 | yes | 2 |
| 1973-06 | 13 | yes | 6 |
| 1974-08 | 2 | yes (ongoing) | |
| 1978-12 | 17 | no | 14 |
| 1980-11 | 10 | yes | 9 |
| 2000-08 | 5 | yes | 8 |
| 2006-08 | 9 | no | 17 |
| 2019-06 | 4 | yes | 9 |
| 2020-02 | 1 | yes | 0 |
| 2022-11 | 25 | no | none yet |

The famous claim survives in a weak form: every recession since 1969 except
July 1990 had an inversion somewhere in the prior 18 months. But 1966 was a
false alarm, 1978 and 2006 led by more than 12 months, and 2022 has led by 46
months and counting. The 12-month horizon is a modelling convenience, not
something the data pins down.

### Out of sample, 1980-01 to 2025-08 (548 months, 113 positives)

| Model | AUC | Brier | Skill vs base rate |
|---|---|---|---|
| Base rate | 0.42 | 0.173 | 0 |
| Spread | 0.650 | 0.161 | +6.6% |
| Sahm | 0.598 | 0.177 | -2.3% |
| Spread + Sahm | 0.754 | 0.160 | +7.5% |

With the 12-month announcement lag on the labels: spread 0.649, Sahm 0.584,
both 0.757. The lag barely matters because a year of extra labels changes an
expanding-window fit very little after the first decade.

The base rate's AUC is below 0.5 because the historical recession frequency has
been drifting down while the positives are clustered, so a higher base rate
tends to coincide with calmer years. That is not a bug, it is what a constant
forecast looks like when the constant moves.

By subperiod, spread model AUC is 0.671 in 1980-1999, 0.703 in 2000-2019,
0.731 in 1980-2007 and **0.455 in 2008-2025**. The last window contains the
2019 inversion (followed by the COVID recession, which the curve cannot have
known about) and the 2022 inversion (followed by nothing). The spread has no
out-of-sample skill in the last eighteen years.

Adding the Sahm gap helps in every subperiod (0.789, 0.850, 0.818, 0.651). The
combination works because the two series have different timing: the spread
leads the recession by a year or more, and unemployment starts rising a few
months before the NBER peak, so a probit on both spans the horizon better than
either alone.

### Calibration

Spread model, out of sample, by forecast decile:

| Mean forecast | Observed rate | n |
|---|---|---|
| 0.02 | 0.20 | 55 |
| 0.05 | 0.05 | 55 |
| 0.08 | 0.18 | 55 |
| 0.11 | 0.13 | 54 |
| 0.16 | 0.16 | 55 |
| 0.21 | 0.13 | 55 |
| 0.27 | 0.09 | 54 |
| 0.36 | 0.24 | 55 |
| 0.45 | 0.40 | 55 |
| 0.69 | 0.47 | 55 |

The bottom decile is badly under-forecast, and the reason is the label, not
1990: its eleven positives are 1980-82 and late 2008, months when the curve
had already steepened because the Fed was cutting into a recession that was
still running, and "recession within 12 months" counts the rest of an
ongoing one. The top decile over-forecasts: when the model said 69% on
average, recessions followed 47% of the time. The 2022-24 episode is most of
the top decile's miss.

### The 2022-2024 inversion, in real time

Daily 10y-3m first closed below zero on 2022-10-18 and stayed inverted from
2022-10-25 to 2024-12-12, 534 consecutive trading days, with brief
re-inversions through October 2025. Monthly average was negative from 2022-11
to 2024-11, 25 months, deepest at -1.57 in May 2023.

The expanding-window model, using only data available each month:

| Month | Spread | P(recession within 12m) |
|---|---|---|
| 2022-07 | +0.67 | 0.35 |
| 2022-11 | -0.26 | 0.55 |
| 2023-05 | -1.57 | 0.81 |
| 2024-08 | -1.18 | 0.67 |
| 2024-12 | +0.12 | 0.41 |
| 2026-08 | +0.96 | 0.25 |

The model was above 50% for 24 consecutive months, peaking at 81% in May 2023.
The label for every one of those months is now known and is zero: USREC has no
recession month from 2022-01 through 2026-08. The Sahm gap reached 0.57 in
August 2024, above its 0.50 trigger, and that also did not mark a recession.
Both classic indicators fired and both were wrong, at least on the 12-month
clock. Whether a recession that starts later still counts as "predicted" is a
question about what the indicator is for, not about the data.

I am not going to explain this away. Candidate stories (the Fed's balance sheet
compressing term premium, so inversion measured something other than expected
policy easing; excess household savings; an unusually strong labour market) are
all plausible and all unfalsifiable from one episode.

## Limitations

**Eight recessions.** The probit is fit on 764 months but the information is in
eight events. Every result above has an effective sample size of about eight,
and the 2022 episode alone flips the top calibration decile from
under-forecast (mean forecast 0.61, observed 0.75, through 2021) to
over-forecast (0.69 against 0.47). The 2008-2021 AUC was already only 0.54
before 2022 arrived, so the recent weakness is not one episode either. A t-stat of -3.4
with HAC errors still treats 764 overlapping months as more information than
they carry.

**Publication bias.** The 10y-3m spread with a 12-month horizon is the
specification that survived thirty years of the literature trying maturity
pairs and horizons. Estrella and Mishkin chose it because it fit best on data
through 1995. I did not search, but the choice I inherited already embeds a
search, and the out-of-sample window starting 1980 overlaps their sample, so
"out of sample" here is out of sample for my code, not for the idea.

**Real-time labels.** The label uses final NBER dates, which are announced with
a lag and occasionally revised. The 12-month announcement-lag run addresses the
first problem and makes little difference. The feature is a Treasury yield and
is not revised, which is the main reason the yield curve is a cleaner real-time
indicator than any macro series. The Sahm gap, by contrast, uses final-vintage
unemployment; the real-time SAHMREALTIME series is in the data file for
comparison; the two differ by 0.03 in a typical month and by up to 0.24 since
2000 (0.37 in 1976) around revisions to the unemployment rate.

**Overlapping labels.** A 12-month-ahead label means consecutive observations
are nearly identical. The HAC correction handles the standard errors but not
the fact that AUC and Brier are computed on 548 months that are really about 45
independent years.

**The horizon is arbitrary.** Inversions have led recessions by 2 to 17 months
in the cases where a recession came. A model that scored "recession within 24
months" would call 1978 and 2006 hits and would still be waiting on 2022.

**No costs, no strategy.** This is a forecasting exercise, not a trading one.
Turning the probability into a position would need a rule, a benchmark and a
cost model, and with eight events there is nothing to fit one on.

## Files

- `data.py` fetches and caches the FRED series and builds the monthly panel
- `model.py` probit fit, expanding-window forecasts, AUC, Brier, calibration
- `check.py` offline checks on synthetic data, exits 1 on any failure
- `run.py` the pipeline, writes charts and tables to `reports/` and a log to `reports/run_log.txt`

## References

Estrella, A. and Mishkin, F. (1998). Predicting U.S. recessions: financial
variables as leading indicators. Review of Economics and Statistics 80(1).

Sahm, C. (2019). Direct stimulus payments to individuals. In Recession Ready,
Brookings.
