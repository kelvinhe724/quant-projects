# Gradient boosting on credit default risk

Predicts whether a credit-card customer defaults next month. A logistic
regression sets the floor and LightGBM with early stopping is the real model.
Most of the work is in the feature engineering, which turns six months of
repayment history into delinquency and utilization summaries.

## Data

UCI "Default of Credit Card Clients": 30,000 Taiwanese credit-card customers
from 2005, 23 raw features, 22.1% default rate. Public and downloadable without
a login. `data.py` fetches it once from `archive.ics.uci.edu` and caches it as
CSV at `../source-material/uci-credit/` (2.8 MB).

The Kaggle Home Credit table is used instead if `application_train.csv` is
present at `../source-material/home-credit/`, but that needs a Kaggle login so
it stays opt-in. A synthetic generator exists only to give `check.py` planted
ground truth.

Engineered features use only the six months preceding the target month, so none
of them can leak the label. Payment-to-bill ratios pair `PAY_AMT_t` with
`BILL_AMT_{t+1}`, the bill that payment actually settles, never with the outcome
month. `LIMIT_BAL = 0` and non-positive bills are guarded so no denominator goes
to zero. That leaves 38 features from the 23 raw columns: utilization (latest,
mean, max), payment-to-bill ratios per month and their mean, delinquent month
count, worst delinquency, delinquency trend, bill trend, total paid, and count
of zero-payment months.

Validation is a stratified 75/25 split, 22,500 train and 7,500 valid, plus
5-fold stratified CV. Hyperparameters are sensible defaults, not tuned.

## Files

- `data.py` download, cache, feature engineering, synthetic generator
- `gb.py` `prepare`, `fit_baseline`, `fit_lgbm`, `importance`, `cv_auc`
- `check.py` offline checks on synthetic data with planted signal features
- `run.py` full pipeline, charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

The first `run.py` downloads the dataset; after that it reads the cache.

## Results

| model | validation AUC |
|---|---|
| logistic regression baseline | 0.7469 |
| LightGBM, 92 trees after early stopping | 0.7840 |
| LightGBM, 5-fold stratified CV | 0.7864 ± 0.0080 |

Fold AUCs were 0.7876, 0.7731, 0.7831, 0.7918, 0.7966. That lands inside the
published band for this dataset, roughly 0.77 to 0.79, so nothing here is
suspiciously good.

Top five features by gain: PAY_0 (16012), MAX_DELINQUENCY (10858), N_DELINQUENT
(8897), PAY_AMT_TOTAL (3202), BILL_AMT1 (1619). Repayment history dominates.
The most recent month's status alone carries most of the signal, and two of the
next three are engineered delinquency summaries, which is the case for building
them. Balances and utilization matter far less than I expected going in.

Charts in `reports/`: `roc.png` (baseline against boosted), `importance.png`
(top 15 by gain), `calibration.png` (both models, 10 quantile bins).

## Limitations

No hyperparameter tuning. On this feature set it moves fractions of a point, so
it was not worth the compute; it becomes worth doing once the feature set grows.
The 0.037 AUC gap between logistic and LightGBM is real but modest, which says
the signal here is mostly linear in the engineered features rather than hiding
in interactions. The calibration curve shows LightGBM drifting at the high end,
where predicted default probability runs above the observed rate, so the raw
scores are usable for ranking but not as probabilities without isotonic or
Platt scaling. The data is one bank, one country, 2005, and a model fit on it
should not be pointed at anything else.
