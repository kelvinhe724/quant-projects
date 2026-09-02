"""Data loading for the default-prediction project.

Default source is the UCI "Default of Credit Card Clients" set (Taiwan, 2005):
30,000 customers, 23 raw features, target = default next month. Public, no
login, downloaded once and cached as CSV under source-material/uci-credit/.

The Kaggle Home Credit table is used instead if it is present under
source-material/home-credit/; it needs a Kaggle login, so it is opt-in.

make_synthetic() exists only for check.py, which needs planted ground truth.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(__file__)
UCI_DIR = os.path.join(ROOT, "..", "source-material", "uci-credit")
UCI_CSV = os.path.join(UCI_DIR, "uci_credit_default.csv")
UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls",
]
KAGGLE_PATH = os.path.join(ROOT, "..", "source-material", "home-credit",
                           "application_train.csv")

EDUCATION = ["Lower secondary", "Secondary", "Incomplete higher", "Higher"]
CONTRACT = ["Cash loans", "Revolving loans"]
INCOME_TYPE = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
HOUSING = ["House / apartment", "Rented apartment", "With parents", "Municipal apartment"]

SEX_MAP = {1: "male", 2: "female"}
EDU_MAP = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MAR_MAP = {1: "married", 2: "single", 3: "other"}
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def _download_uci():
    import io
    import ssl
    import zipfile

    try:  # some python builds have no system CA bundle wired up
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None

    os.makedirs(UCI_DIR, exist_ok=True)
    last = None
    for url in UCI_URLS:
        try:
            raw = urllib.request.urlopen(url, timeout=120, context=ctx).read()
        except Exception as e:  # try the next mirror
            last = e
            continue
        if url.endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(raw))
            name = next(n for n in zf.namelist() if n.endswith((".xls", ".xlsx")))
            raw = zf.read(name)
        # the sheet has a junk first row (X1..X23) above the real header
        df = pd.read_excel(io.BytesIO(raw), header=1)
        df.to_csv(UCI_CSV, index=False)
        return df
    raise RuntimeError(f"could not download the UCI credit dataset: {last}")


def load_uci():
    """UCI credit-card default data, cached after the first call."""
    df = pd.read_csv(UCI_CSV) if os.path.exists(UCI_CSV) else _download_uci()
    df = df.rename(columns={"default payment next month": "TARGET",
                            "PAY_1": "PAY_0"})
    df = df.drop(columns=[c for c in ["ID"] if c in df.columns])

    # codes outside the documented range fall back to "other"/"unknown"
    df["SEX"] = df["SEX"].map(SEX_MAP).fillna("unknown")
    df["EDUCATION"] = df["EDUCATION"].map(EDU_MAP).fillna("other")
    df["MARRIAGE"] = df["MARRIAGE"].map(MAR_MAP).fillna("other")

    df = engineer(df)
    return df


def engineer(df):
    """Derive features from the six months preceding the target month.

    Nothing here touches the outcome month, so none of it can leak the label.
    PAY_n is repayment status n months back (-1 paid duly, >=1 months late);
    BILL_AMT1 / PAY_AMT1 are the most recent month.
    """
    limit = df["LIMIT_BAL"].replace(0, np.nan)

    df["UTILIZATION"] = df["BILL_AMT1"] / limit
    df["UTILIZATION_MEAN"] = df[BILL_COLS].mean(axis=1) / limit
    df["UTILIZATION_MAX"] = df[BILL_COLS].max(axis=1) / limit

    # payment made in month t settles the bill billed in month t+1
    for i in range(1, 6):
        prev_bill = df[f"BILL_AMT{i + 1}"].where(df[f"BILL_AMT{i + 1}"] > 0)
        df[f"PAY_RATIO_{i}"] = df[f"PAY_AMT{i}"] / prev_bill
    ratios = [f"PAY_RATIO_{i}" for i in range(1, 6)]
    df["PAY_RATIO_MEAN"] = df[ratios].mean(axis=1)

    df["N_DELINQUENT"] = (df[PAY_COLS] >= 1).sum(axis=1)
    df["MAX_DELINQUENCY"] = df[PAY_COLS].max(axis=1)
    df["DELINQ_TREND"] = df["PAY_0"] - df["PAY_6"]

    df["BILL_TREND"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / limit
    df["PAY_AMT_TOTAL"] = df[AMT_COLS].sum(axis=1)
    df["ZERO_PAY_MONTHS"] = (df[AMT_COLS] == 0).sum(axis=1)
    return df


def make_synthetic(n=20000, seed=7):
    """Fake applications for check.py.

    Risk is driven by ext_score_1/2 (and their product), credit-to-income and
    education. Every other column is noise.
    """
    rng = np.random.default_rng(seed)

    income = np.exp(rng.normal(11.6, 0.5, n))
    credit = income * np.exp(rng.normal(1.0, 0.6, n))
    age = rng.uniform(21, 68, n)
    employed_years = np.clip(rng.exponential(6, n), 0, age - 18)
    ext1 = rng.beta(4, 3, n)
    ext2 = rng.beta(4, 3, n)
    edu = rng.choice(len(EDUCATION), n, p=[0.1, 0.6, 0.1, 0.2])
    contract = rng.choice(CONTRACT, n, p=[0.9, 0.1])
    income_type = rng.choice(INCOME_TYPE, n, p=[0.5, 0.23, 0.18, 0.07, 0.02])
    housing = rng.choice(HOUSING, n, p=[0.88, 0.05, 0.04, 0.03])
    children = rng.poisson(0.5, n)
    car_age = np.where(rng.random(n) < 0.34, rng.uniform(0, 25, n), np.nan)
    phone_change_days = rng.exponential(900, n)

    cti = credit / income
    logit = (-2.4
             - 4.5 * (ext1 - 0.5) - 4.5 * (ext2 - 0.5)
             - 6.0 * (ext1 - 0.5) * (ext2 - 0.5)
             + 0.35 * (cti - 3)
             - 0.35 * (edu - 1)
             - 0.02 * (age - 40)
             + rng.normal(0, 0.3, n))
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "TARGET": target,
        "AMT_INCOME_TOTAL": income.round(0),
        "AMT_CREDIT": credit.round(0),
        "CREDIT_TO_INCOME": cti.round(3),
        "AGE_YEARS": age.round(1),
        "YEARS_EMPLOYED": employed_years.round(1),
        "EXT_SCORE_1": ext1.round(4),
        "EXT_SCORE_2": ext2.round(4),
        "CNT_CHILDREN": children,
        "OWN_CAR_AGE": car_age.round(1),
        "DAYS_LAST_PHONE_CHANGE": phone_change_days.round(0),
        "NAME_EDUCATION_TYPE": np.array(EDUCATION)[edu],
        "NAME_CONTRACT_TYPE": contract,
        "NAME_INCOME_TYPE": income_type,
        "NAME_HOUSING_TYPE": housing,
    })

    df.loc[rng.random(n) < 0.15, "EXT_SCORE_2"] = np.nan
    miss1 = rng.random(n) < np.where(target == 1, 0.35, 0.15)
    df.loc[miss1, "EXT_SCORE_1"] = np.nan
    df.loc[rng.random(n) < 0.05, "YEARS_EMPLOYED"] = np.nan
    return df


def load():
    """Returns (df, source): UCI unless a Home Credit export is present."""
    if os.path.exists(KAGGLE_PATH):
        df = pd.read_csv(KAGGLE_PATH)
        return df.drop(columns=[c for c in ["SK_ID_CURR"] if c in df.columns]), "kaggle"
    return load_uci(), "uci"
