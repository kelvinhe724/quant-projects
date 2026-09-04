# Multiple linear regression on diamond prices

An 18-term linear model for the price of a diamond, built the long way: a candidate
specification with transformations and interactions, AIC/BIC selection, ridge, lasso
and elastic net on identical folds, repeated-CV and bootstrap stability testing, HC3
robust inference on a held-out split, and one run on an untouched test set.

The three findings that survived all of it:

1. **Carat is almost the whole model.** Log carat alone explains 93.3% of the
   variation in log price. Seventeen more terms, three quality grades and four
   interactions add 4.9 points on top of that.
2. **The log-log transformation matters more than any feature choice.** Moving to
   log price and log carat cuts median absolute error from $643 to $342 with the
   same single predictor, and it fixes the residual funnel that makes the level
   model's standard errors meaningless.
3. **Cut, color and clarity look backwards in raw price.** In the selection sample,
   flawless diamonds average $2,828 and the worst clarity grade averages $3,978. The effect is entirely
   confounding with size: I1 stones average 1.29 carats and IF stones average 0.51.
   Hold carat fixed and the ordering flips to the one a jeweller would predict.

I also found that regularisation buys nothing here, and I have kept that result rather
than presenting the 0.3% lasso improvement as a win.

## Data

The ggplot2/seaborn diamonds table, 53,940 rows and 10 columns, fetched once from the
seaborn-data mirror and cached to `source-material/diamonds/diamonds.csv`. No Kaggle
login is needed. `seaborn.load_dataset` itself fails in this environment on an SSL
certificate error, so `data.py` requests the same CSV directly with a `certifi` bundle;
seaborn is not a dependency of this project.

Data-quality audit on the raw file:

| Rule | Rows | Action |
|---|---|---|
| Non-positive price | 0 | none needed |
| Zero dimension (x, y or z = 0) | 20 | removed, log(volume) is undefined |
| Depth outside 43-79 | 0 | none needed |
| Table outside 43-95 | 0 | none needed |
| Reported depth disagrees with 200z/(x+y) by more than 1pp | 93 | flagged, kept |
| Aspect ratio x/y outside 0.9-1.1 | 16 | flagged, kept |
| y or z above 20mm | 3 | flagged, kept |
| Exact duplicate rows | 289 | 145 dropped, first copy of each kept |

Duplicates are dropped because identical records landing in different folds is leakage,
not because they look untidy. Everything else that is merely extreme keeps its flag and
stays in the sample: the 31.8mm z value and the 58.9mm y value are almost certainly typos,
but deleting rows for being inconvenient is how you end up with a model that only works
on the easy part of the data. **Modelling sample: 53,775 rows, 80 still carrying a flag.**

Split, stratified on log-price deciles so all three samples carry the same price
distribution (median price $2,401 and median carat 0.70 in each):

| Sample | Rows | Use |
|---|---|---|
| Selection | 28,233 | EDA, feature engineering, AIC/BIC, hyperparameter tuning, stability |
| Inference | 9,410 | fit the locked specification once, compute HC3 tests |
| Test | 16,132 | run once, at the end |

Seed 20260903 throughout, set in `data.py`.

## Specification

The 18 candidate terms are the specification from the project brief: `log_carat`,
`log_carat_sq`, `log_volume`, `aspect_xy`, centred `depth`/`table` and their squares,
ordinal scores for cut/color/clarity and their squares, and four interactions of size
with quality. Centring constants for depth and table are learned inside each training
fold and applied outward, which `check.py` verifies.

Target is log price. Predictions come back to dollars through `exp()` with Duan's
smearing factor estimated on the training fold's residuals; the factor is 1.009 on the
full development sample, so the naive back-transformation would have been about 0.9% low.

## Results

Everything below comes from `run.py`; the full console output is in
`reports/run_log.txt`.

### Baseline ladder

Fitted on the selection sample. R2, adjusted R2, AIC and BIC are in-sample on log price;
RMSE, MAE and median AE are 5-fold cross-validated, in dollars.

| Model | Terms | R2 | Adj R2 | AIC | BIC | CV RMSE | CV MAE | CV MedAE |
|---|---|---|---|---|---|---|---|---|
| Mean price | 0 | | | | | $3,979 | $3,025 | $2,772 |
| Carat only, price in dollars | 1 | 0.848 | | | | $1,550 | $1,011 | $643 |
| Carat only, log-log | 1 | 0.933 | 0.933 | 4,673 | 4,689 | $1,673 | $861 | $342 |
| Carat + curvature | 2 | 0.933 | 0.933 | 4,675 | 4,700 | $1,674 | $862 | $342 |
| Carat + quality scores | 5 | 0.979 | 0.979 | -28,404 | -28,355 | $999 | $464 | $184 |
| Full 18-term | 18 | 0.982 | 0.982 | -32,182 | -32,025 | $925 | $424 | $177 |
| AIC selected (17 terms) | 17 | 0.982 | 0.982 | **-32,184** | -32,035 | $924 | $424 | $177 |
| BIC selected (16 terms) | 16 | 0.982 | 0.982 | -32,182 | **-32,042** | $924 | $424 | $177 |
| One-hot quality benchmark | 25 | 0.983 | 0.983 | -33,780 | -33,566 | **$873** | **$411** | **$169** |

Look at what the level model does to the metrics. Carat-only in dollars has the lower RMSE
($1,550 vs $1,673) and the far worse MAE and median error. RMSE rewards getting the
$18,000 stones roughly right and the log model deliberately does not optimise for that;
by the metric that describes a typical diamond, the log model is nearly twice as accurate.

AIC drops `carat_x_cut`. BIC drops `carat_x_cut` and `table_x_cut`. The three criteria
disagree by two terms out of eighteen and by less than $1 of CV RMSE, which is the honest
summary: after `log_carat`, the specification search is not where the accuracy lives.

The one-hot benchmark is the uncomfortable result. Replacing the ordinal quality scores
with 17 dummies improves CV RMSE by $52, about 5.6%. The equal-spacing assumption that
keeps the model inside the 10-20 term budget costs real accuracy. I kept the score
specification as the primary model because the brief asks for it and because it is what
the inference section can interpret, but the ordinal coding is a modelling choice that
loses to the flexible one on this data, not a free simplification.

### Regularisation

Ridge, lasso and elastic net were tuned on the same folds, with the feature build,
standardisation and estimator all inside one pipeline so nothing is fitted before the
split. Nested CV (5 outer, 3 inner) puts alpha selection inside each outer training fold:

| Model | Nested outer RMSE | SD | Alphas chosen across folds |
|---|---|---|---|
| Ridge | $923 | $60 | 0.22, 1.0, 2.15, 4.64 |
| Lasso | $922 | $59 | 1e-6, 5.6e-5, 1.1e-4 |
| Elastic net | $923 | $60 | 1e-4, 3.2e-4 |

The penalised models below use the one-standard-error rule on the 5-fold tuning
grid: the most regularised setting whose mean CV error is within one SE of the
best. That gives ridge alpha 46.4, lasso 8.3e-4 and elastic net 0.001 with
l1_ratio 0.1, against CV-optimal alphas of 1e-4, 5.6e-5 and 3.2e-5. Under repeated
5-fold CV, 25 folds per model, on identical fold indices:

| Model | CV RMSE mean | SD | 5th pct | 95th pct | CV MAE | Gap (val - train) | Min sign consistency |
|---|---|---|---|---|---|---|---|
| OLS full 18-term | $922 | $72 | $832 | $1,033 | $424 | $1.4 | 0.56 |
| OLS BIC 16-term | $921 | $72 | $832 | $1,032 | $424 | $0.8 | 0.76 |
| Ridge | $924 | $68 | $835 | $1,026 | $430 | $1.5 | 0.88 |
| Lasso | **$919** | $72 | $831 | $1,025 | $429 | $0.9 | 0.60 |
| Elastic net | $923 | $70 | $834 | $1,028 | $427 | $1.4 | 0.68 |

**Lasso "won" by $3 against a fold-to-fold standard deviation of $72.** That is a
twenty-fifth of one fold's standard deviation, a fifth of the standard error of the
25-fold mean, and I do not believe it. The reason regularisation has
nothing to do here is arithmetic: 28,233 training rows against 18 terms is about 1,500
observations per parameter, and the generalisation gap for every model is under $2 on a
$920 error. There is no variance to trade bias for. The CV-optimal lasso alpha is 5.6e-5,
which is a rounding error away from unpenalised OLS. The one-SE alpha of 8.3e-4 zeroes
two terms on the full selection fit (`table_c` and `cut_score_sq`, ignoring the
hierarchy) and costs nothing measurable; the lasso path in `reports/lasso_path.csv`
only starts removing terms in bulk above alpha = 0.002.

The one thing regularisation does buy is coefficient stability, and only ridge buys it.
Ridge's worst sign consistency across the 25 folds is 0.88 against 0.56 for OLS, because
`log_carat` and `log_volume` carry VIFs of 633 and 613 and ridge spreads the size effect
across them instead of letting each fold reallocate it.

### Stability

500 bootstrap draws on the locked 18-term specification, standardised coefficients:

| Term | Mean | SD | Sign consistency | 95% bootstrap CI |
|---|---|---|---|---|
| log_carat | 0.873 | 0.063 | 1.00 | [0.745, 0.983] |
| clarity_score | 0.367 | 0.005 | 1.00 | [0.357, 0.377] |
| color_score | 0.272 | 0.004 | 1.00 | [0.263, 0.281] |
| log_volume | 0.190 | 0.062 | 1.00 | [0.079, 0.315] |
| clarity_score_sq | -0.164 | 0.005 | 1.00 | [-0.175, -0.154] |
| color_score_sq | -0.140 | 0.005 | 1.00 | [-0.149, -0.131] |
| cut_score | 0.057 | 0.010 | 1.00 | [0.036, 0.075] |
| carat_x_clarity | 0.032 | 0.004 | 1.00 | [0.025, 0.039] |
| cut_score_sq | -0.033 | 0.010 | 1.00 | [-0.051, -0.014] |
| carat_x_color | 0.019 | 0.003 | 1.00 | [0.013, 0.024] |
| table_x_cut | -0.008 | 0.005 | 0.94 | [-0.018, 0.002] |
| **carat_x_cut** | **0.000** | **0.004** | **0.53** | **[-0.008, 0.009]** |

Fifteen of eighteen terms are directionally rock solid. The interaction of size with cut
is the one term that is genuinely nothing: its sign is a coin flip across draws, its
bootstrap interval straddles zero, and rerunning the AIC search inside 100 bootstrap
draws retains it only **24%** of the time. `table_x_cut` sits in between at 75%
inclusion. Every other term is selected in 99-100% of draws, which is why AIC and BIC
both land on essentially the same 16-17 term model.

The `log_carat` / `log_volume` pair deserves a caveat. Both are individually stable in
sign, but their bootstrap SDs (0.063 and 0.062) are ten times larger than any other
stable term's, and they move against each other: they are two measurements of the same
physical thing, and the model is splitting one effect between them rather than
identifying two.

### Diagnostics

| Check | Result |
|---|---|
| Breusch-Pagan, level target | LM 7,123, p < 1e-300 |
| Breusch-Pagan, log target | LM 1,804, p < 1e-300 |
| RESET, power 2 | F 203.8, p 4.6e-46 |
| Residual skew / excess kurtosis | -0.11 / +2.93 |
| Jarque-Bera | p < 1e-300 |
| Condition number, locked design | 1,079 |
| Max VIF | 633 (`log_carat`), 613 (`log_volume`) |
| Cook's D above 4/n | 1,243 rows, 4.4%, max 1.56 |

The log transformation cuts the Breusch-Pagan statistic by a factor of four but does not
remove heteroskedasticity, which is why every reported p-value below uses HC3. RESET
rejects at any threshold, but with 28,000 observations RESET rejects almost anything;
the residual-vs-fitted plot in `reports/diagnostics.png` shows a roughly even band with
mild curvature at the cheap end rather than the funnel the level model produces.

Residual excess kurtosis of +2.93 is the one diagnostic I cannot wave away with sample
size. The residuals are symmetric but genuinely heavy-tailed, so a normal-theory
confidence interval understates the chance of a large error. This is the second reason
the report leans on HC3 and bootstrap intervals rather than the textbook ones.

Refitting without the 1,243 highest-influence rows moves `aspect_xy` by 112% of its own
value, and moves nothing else materially. `aspect_xy` is a shape ratio hovering around
1.0 with almost no variation, so a handful of malformed stones set it single-handed.
Its standardised coefficient is 0.011. It is not a driver of anything and I would drop it
from a production specification.

### Inference, on the held-out inference split

The BIC specification was locked before this split was touched. 16 terms, 17 estimated
coefficients, n = 9,410. Overall F = 28,557 (p < 1e-300), R2 = 0.982, residual SE = 0.137.

HC3 standard errors run 1.07x to 7.12x the classical ones, median 1.78x. **Using
textbook OLS standard errors here would overstate precision by roughly a factor of two.**

| Term | Estimate | HC3 SE | t | p | Holm p |
|---|---|---|---|---|---|
| log_carat | 1.193 | 0.186 | 6.42 | <1e-9 | <1e-8 |
| log_carat_sq | 0.043 | 0.008 | 5.70 | <1e-7 | <1e-6 |
| log_volume | 0.655 | 0.186 | 3.52 | 0.0004 | 0.0035 |
| aspect_xy | 0.966 | 0.297 | 3.25 | 0.0012 | 0.0081 |
| depth_c | 0.002 | 0.005 | 0.36 | 0.72 | 1.00 |
| depth_c_sq | -0.001 | 0.002 | -0.59 | 0.56 | 1.00 |
| table_c | 0.001 | 0.001 | 0.54 | 0.59 | 1.00 |
| table_c_sq | -0.000 | 0.001 | -0.58 | 0.56 | 1.00 |
| cut_score | 0.064 | 0.036 | 1.78 | 0.075 | 0.45 |
| cut_score_sq | -0.006 | 0.004 | -1.32 | 0.19 | 0.94 |
| color_score | 0.168 | 0.004 | 38.18 | <1e-300 | <1e-300 |
| color_score_sq | -0.010 | 0.001 | -19.61 | <1e-84 | <1e-83 |
| clarity_score | 0.223 | 0.006 | 38.50 | <1e-300 | <1e-300 |
| clarity_score_sq | -0.011 | 0.001 | -15.99 | <1e-56 | <1e-55 |
| carat_x_color | 0.008 | 0.002 | 4.13 | <1e-4 | 0.0004 |
| carat_x_clarity | 0.010 | 0.002 | 4.05 | 0.0001 | 0.0005 |

Joint F-tests on term groups, which is the right unit for a score-plus-square pair:

| Group | Terms | F | p |
|---|---|---|---|
| Clarity | 2 | 3,358.1 | <1e-300 |
| Color | 2 | 2,705.7 | <1e-300 |
| Carat | 2 | 31.9 | <1e-13 |
| Cut | 2 | 26.6 | <1e-11 |
| Size (volume, aspect) | 2 | 18.7 | <1e-8 |
| Interactions | 2 | 14.7 | <1e-6 |
| Depth | 2 | 3.11 | 0.045 |
| Table | 2 | 0.45 | 0.64 |

The cut group is the case for grouped testing. Neither `cut_score` (p = 0.075) nor
`cut_score_sq` (p = 0.19) clears 5% alone, and after Holm correction across the 16
coefficients both are nowhere near it. Jointly, cut is significant at F = 26.6. Reading
the individual rows and concluding "cut doesn't matter" would have been wrong.

Depth and table are the opposite case: individually insignificant, jointly insignificant
(table F = 0.45), and their standardised coefficients are under 0.011. Four of the
eighteen candidate terms are measuring proportions that do not price diamonds.

Practical effects at the median 0.70-carat stone:

- One clarity grade is worth **+12.9%** in price.
- One color grade is worth **+7.7%**.
- One cut grade is worth **+1.3%**, and that estimate is not distinguishable from zero
  individually.
- The price elasticity to carat is **1.16** at the median stone: 10% more weight, 11.7%
  more money. The elasticity rises with size because `log_carat_sq` is positive.

That elasticity is a conditional number and should be read with care, because
`log_volume` is in the model absorbing part of the same size effect at VIF 613. The
carat-only model puts the unconditional elasticity at 1.67.

### Final test, run once

Fitted on selection + inference (37,643 rows), evaluated on the 16,132 held-out rows.

| Model | Test RMSE | Test MAE | Test MedAE | Test R2 | Inside the 5-95% CV band? |
|---|---|---|---|---|---|
| OLS full 18-term | **$849** | $416 | $173 | 0.955 | yes, [$832, $1,033] |
| OLS BIC 16-term | $850 | $417 | $173 | 0.955 | yes, [$832, $1,032] |
| Lasso | $850 | $421 | $175 | 0.955 | yes, [$831, $1,025] |
| Elastic net | $858 | $420 | $173 | 0.954 | yes, [$834, $1,028] |
| Ridge | $860 | $420 | $173 | 0.954 | yes, [$835, $1,026] |

Every model lands inside its own repeated-CV band, near the optimistic end, which is what
you expect when the final fit gets 67% more training data than any CV fold
(37,643 rows against 22,586). The spread
across five very different regularisation choices is $11, or 1.3%.

**Final model: the BIC-selected 16-term OLS.** It is within $1 of the best test RMSE, it
is the most parsimonious specification that survives selection, its generalisation gap is
the smallest of the OLS variants, and it is the one whose coefficients I can defend
individually. The decision rule was fixed in advance: prefer the simplest model whose
performance is within one standard error of the best.

Errors by price decile for that model on the test set:

| Price decile | n | RMSE | MAE | Mean error |
|---|---|---|---|---|
| 1 (cheapest) | 1,620 | $71 | $58 | -$24 |
| 5 | 1,617 | $275 | $194 | -$78 |
| 8 | 1,613 | $770 | $541 | +$24 |
| 10 (most expensive) | 1,614 | $2,143 | $1,500 | +$36 |

Error scales with price almost exactly proportionally, which is the log model working as
intended: it is fitting percentage error, so a $15,000 stone gets a $1,500 error and a
$600 stone gets a $58 one. The subgroup table and `reports/test_fit.png` are for the
BIC model, which was fixed before the test split was opened; `run.py` does not pick
the final model by test RMSE. If you need dollar accuracy on expensive stones specifically,
this is the wrong loss function and you should say so before deploying it.

## Model card

| Field | Content |
|---|---|
| Purpose | Explain and predict diamond price within the range of this dataset |
| Data | ggplot2/seaborn diamonds, 53,940 rows; 53,775 after removing 20 zero-dimension rows and 145 duplicate copies |
| Target | log(price), returned to dollars via exp() with Duan smearing (factor 1.009) |
| Features | 16 terms: log_carat, log_carat_sq, log_volume, aspect_xy, depth_c, depth_c_sq, table_c, table_c_sq, cut_score, cut_score_sq, color_score, color_score_sq, clarity_score, clarity_score_sq, carat_x_color, carat_x_clarity. Quality grades on ordinal 1-n scores, worst to best. Depth and table centred on training means |
| Validation | Repeated 5-fold CV (25 folds), nested CV for penalised models, one evaluation on a 16,132-row test set |
| Performance | Test RMSE $850, MAE $417, median AE $173, R2 0.955 |
| Stability | 15/18 terms at 99.8% or better bootstrap sign consistency; carat_x_cut at 53% and dropped; log_carat/log_volume unstable as a pair at VIF 633/613 |
| Inference | HC3 robust SEs on an independent 9,410-row split; joint F-tests by group; Holm and BH adjusted p-values |
| Limitations | See below |
| Intended use | Educational statistical modelling. Not an appraisal system |

## Limitations

- **The prices are from one 2008 snapshot of one retailer's inventory.** Nothing here
  transfers to today's market, to another dealer, or to lab-grown stones, and no
  certification, fluorescence, origin or date variable exists in the file to condition on.
- **Nothing here is causal.** Cut, color, clarity and carat are jointly chosen by whoever
  cut the rough stone; the clarity coefficient is the price difference between stones that
  differ in clarity and in everything correlated with clarity that the model does not
  observe. "Upgrade a stone's clarity and it is worth 13% more" is not a claim this data
  supports.
- **`log_carat` and `log_volume` are the same variable twice.** VIF 633 and 613. Their
  individual coefficients are not separately identified and should be read as a group.
- **The ordinal coding is a known constraint, not a modelling result.** One-hot quality
  encoding beats it by 5.6% CV RMSE. The 10-20 term budget is what keeps it in.
- **4.4% of rows exceed the 4/n Cook's D threshold.** Sensitivity analysis shows only
  `aspect_xy` is materially affected, but a 5% influence rate means the model is
  supported by a lumpy sample, and the largest stones are extrapolation territory.
- **Selection and inference are separated, but not perfectly.** The candidate 18 terms
  were fixed before any fitting, and inference runs on a split that selection never
  touched. But the transformations themselves come from the brief and from EDA on the
  full literature about this dataset, so the p-values are cleaner than post-selection
  OLS and not as clean as a genuine pre-registration.

## Files

| File | What it does |
|---|---|
| `data.py` | Fetch and cache the CSV, run the quality audit, clean and stratify-split |
| `model.py` | Feature construction, OLS ladder, backward AIC/BIC with hierarchy, penalised pipelines, repeated/nested CV, bootstrap, HC3 inference, VIF |
| `check.py` | 52 offline checks on synthetic data with planted coefficients; exits nonzero on any failure |
| `run.py` | The full pipeline; writes 15 tables and 7 figures to `reports/` |

```
python3 check.py    # 52/52 checks, ~10s
python3 run.py      # full pipeline, ~155s
```
