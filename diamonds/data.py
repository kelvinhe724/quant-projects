"""Diamonds dataset: fetch, audit, clean and split.

The file is the ggplot2/seaborn diamonds table (53,940 rows), cached under
source-material/ so every rerun is offline.
"""
import io
import os

import numpy as np
import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "source-material", "diamonds", "diamonds.csv")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

SEED = 20260903
TEST_FRACTION = 0.30
INFERENCE_FRACTION = 0.25


def load_raw():
    """Read the cached CSV, downloading it once if the cache is empty."""
    if not os.path.exists(CACHE):
        import certifi
        import requests

        response = requests.get(SOURCE_URL, verify=certifi.where(), timeout=120)
        response.raise_for_status()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "wb") as handle:
            handle.write(response.content)
        return pd.read_csv(io.BytesIO(response.content))
    return pd.read_csv(CACHE)


def quality_flags(df):
    """Return one boolean column per data-quality rule from the specification."""
    calculated_depth = 200 * df["z"] / (df["x"] + df["y"]).replace(0, np.nan)
    aspect = df["x"] / df["y"].replace(0, np.nan)
    flags = pd.DataFrame(index=df.index)
    flags["bad_price"] = df["price"] <= 0
    flags["zero_dimension"] = (df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0)
    flags["implausible_depth"] = (df["depth"] < 43) | (df["depth"] > 79)
    flags["implausible_table"] = (df["table"] < 43) | (df["table"] > 95)
    flags["depth_inconsistent"] = (calculated_depth - df["depth"]).abs() > 1.0
    flags["extreme_aspect"] = (aspect < 0.9) | (aspect > 1.1)
    flags["extreme_dimension"] = (df["y"] > 20) | (df["z"] > 20)
    flags["duplicate_row"] = df.duplicated(keep=False)
    return flags


def audit(df=None):
    """Build the raw-to-modelling sample flow table."""
    df = load_raw() if df is None else df
    flags = quality_flags(df)
    rows = [{"rule": name, "rows": int(flags[name].sum()),
             "pct": 100 * flags[name].mean()} for name in flags.columns]
    return pd.DataFrame(rows)


def clean(df=None):
    """Drop physically impossible rows and exact duplicates; flag the rest.

    Only two rules delete: a non-positive price makes the target undefined, and a
    zero dimension makes log(volume) undefined. Exact duplicates are dropped
    because keeping them would place identical records in different CV folds.
    Everything else stays in the sample carrying its flag.
    """
    df = load_raw() if df is None else df
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    flags = quality_flags(df)
    impossible = flags["bad_price"] | flags["zero_dimension"]
    out = df.loc[~impossible].copy()
    out = out.drop_duplicates(keep="first")
    out = out.reset_index(drop=True)
    out["flagged"] = quality_flags(out).drop(columns=["duplicate_row"]).any(axis=1)
    return out


def split(df, seed=SEED):
    """Stratified three-way split: selection, inference and untouched test.

    Stratification uses log-price deciles so the three samples carry the same
    price distribution. The bins exist only for splitting and never enter a model.
    """
    rng = np.random.default_rng(seed)
    bins = pd.qcut(np.log(df["price"]), 10, labels=False)
    role = pd.Series("selection", index=df.index)
    for _, idx in df.groupby(bins).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * len(idx)))
        n_inference = int(round(INFERENCE_FRACTION * (len(idx) - n_test)))
        role.loc[idx[:n_test]] = "test"
        role.loc[idx[n_test:n_test + n_inference]] = "inference"
    return role


if __name__ == "__main__":
    raw = load_raw()
    print(f"raw: {raw.shape[0]} rows, {raw.shape[1]} columns")
    print(audit(raw).to_string(index=False))
    df = clean(raw)
    print(f"\nmodelling sample: {len(df)} rows "
          f"({len(raw) - len(df)} removed), {df['flagged'].sum()} still flagged")
    role = split(df)
    print(role.value_counts().to_string())
    for name, sub in df.groupby(role):
        print(f"{name:>10}: median price ${sub['price'].median():.0f}, "
              f"median carat {sub['carat'].median():.2f}")
