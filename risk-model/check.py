"""Offline checks on simulated returns with a planted factor structure. Exits 1 on any failure.

Sixty names load on three planted factors, so the Marchenko-Pastur count
must find three, the observed-factor model must recover the betas, and
every structured estimator must beat the sample covariance against the
truth. Pure noise must give zero factors. VaR coverage is checked on
simulated GARCH-t returns with the weights held fixed.

Run: ../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import risk

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


rng = np.random.default_rng(11)


def planted(n, t, k=3, fvol=(0.012, 0.008, 0.006), scale=(1.0, 0.8, 0.7), idio=0.01):
    B = rng.normal(0, 1, (n, k)) * np.array(scale[:k])
    B[:, 0] = 1 + 0.3 * rng.normal(0, 1, n)
    F = rng.normal(0, 1, (t, k)) * np.array(fvol[:k])
    R = F @ B.T + rng.normal(0, idio, (t, n))
    S = B @ np.diag(np.array(fvol[:k]) ** 2) @ B.T + np.eye(n) * idio ** 2
    return R, F, B, S


N, T = 60, 2000
R, F, B, S_true = planted(N, T)
lam = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))[::-1]
check("MP count finds the three planted factors", risk.n_factors(lam, T) == 3, f"k={risk.n_factors(lam, T)}")

noise = rng.normal(0, 1, (1000, 100))
lam_n = np.linalg.eigvalsh(np.corrcoef(noise, rowvar=False))[::-1]
check("MP count on pure noise (N=100, T=1000) is zero", risk.n_factors(lam_n, 1000) == 0,
      f"top eigenvalue {lam_n[0]:.3f} vs edge {risk.mp_edge(100, 1000):.3f}")
noise2 = rng.standard_t(5, (400, 300))
lam_n2 = np.linalg.eigvalsh(np.corrcoef(noise2, rowvar=False))[::-1]
check("MP count on fat-tailed noise near q=1 (N=300, T=400) is zero", risk.n_factors(lam_n2, 400) == 0,
      f"top {lam_n2[0]:.3f} vs edge {risk.mp_edge(300, 400):.3f}")
R1, _, _, _ = planted(80, 1500, k=1)
lam1 = np.linalg.eigvalsh(np.corrcoef(R1, rowvar=False))[::-1]
check("one planted factor gives k=1", risk.n_factors(lam1, 1500) == 1)
check("MP edge rises with N/T", risk.mp_edge(500, 504) > risk.mp_edge(50, 504) > risk.mp_edge(5, 504))

S_ff, B_hat, D_hat = risk.cov_ff(R, F)
err = np.abs(B_hat - B).mean()
check("observed-factor betas recovered (mean abs error < 0.06)", err < 0.06, f"{err:.4f}")
check("observed-factor residual variance within 10% of planted",
      abs(D_hat.mean() / 1e-4 - 1) < 0.10, f"{D_hat.mean():.3e}")

frob = lambda S: np.linalg.norm(S - S_true)
S_s, _ = risk.estimate("sample", R)
S_lw, meta_lw = risk.estimate("lw", R)
S_p, meta_p = risk.estimate("pca", R)
check("Ledoit-Wolf closer to truth than sample at T=2000", frob(S_lw) < frob(S_s), f"shrinkage {meta_lw['shrinkage']:.3f}")
check("observed-factor model closer to truth than sample at T=2000", frob(S_ff) < frob(S_s))
check("PCA estimator reports k=3", meta_p["k"] == 3)

w_id = risk.gmv(np.eye(5))
check("GMV on identity is 1/N and sums to one", np.allclose(w_id, 0.2))

Rb, Fb, Bb, Sb = planted(200, 250)
Rb_oos = Fb[:0].sum() + (rng.normal(0, 1, (2000, 3)) * np.array((0.012, 0.008, 0.006))) @ Bb.T \
    + rng.normal(0, 0.01, (2000, 200))
est_b = {e: risk.estimate(e, Rb, Fb)[0] for e in ("sample", "lw", "pca", "ff")}
frob_b = {e: np.linalg.norm(S - Sb) for e, S in est_b.items()}
check("N=200, T=250: PCA and observed-factor covariances closer to truth than sample",
      frob_b["pca"] < frob_b["sample"] and frob_b["ff"] < frob_b["sample"],
      ", ".join(f"{e} {v:.2e}" for e, v in frob_b.items()))
oos = {e: (Rb_oos @ risk.gmv(S)).std() for e, S in est_b.items()}
check("N=200, T=250: sample GMV realises more vol out of sample than PCA, LW and FF",
      all(oos["sample"] > oos[e] for e in ("lw", "pca", "ff")),
      ", ".join(f"{e} {v * np.sqrt(252):.3f}" for e, v in oos.items()))

idx = pd.bdate_range("2015-01-01", periods=T)
names = [f"A{i}" for i in range(N)]
ret = pd.DataFrame(R, index=idx, columns=names)
factors = pd.DataFrame(np.column_stack([F, rng.normal(0, 1, (T, 3)) * 0.01, np.zeros(T)]), index=idx,
                       columns=risk.FACTORS + ["RF"])
pos = {n: 1 / 20 for n in names[:20]}
classes = pd.Series({n: ("x" if i % 2 else "y") for i, n in enumerate(names)})
ex = risk.exposures(pos, idx[-1], ret, factors, classes=classes)
wv = np.array([pos[n] for n in ex["assets"].index])
check("asset contributions sum to portfolio vol", abs(ex["assets"]["contribution"].sum() - ex["vol"]) < 1e-9)
check("asset pct shares sum to one", abs(ex["assets"]["pct"].sum() - 1) < 1e-9)
check("class table sums to the asset table", abs(ex["by_class"]["pct"].sum() - 1) < 1e-9 and set(ex["by_class"].index) == {"x", "y"})
x_true = B[:20].T @ wv
check("Fama-French exposures are B'w on the planted betas (max abs error < 0.05)",
      np.abs(ex["ff"].loc[risk.FACTORS[:3], "exposure"].to_numpy() - x_true).max() < 0.05)
check("noise factors carry no exposure", np.abs(ex["ff"].loc[risk.FACTORS[3:], "exposure"]).max() < 0.05)
check("FF factor and residual shares sum to one", abs(ex["ff"]["pct"].sum() - 1) < 1e-9)
check("statistical block finds the market factor on the 20 held names and shares sum to one",
      ex["k"] >= 1 and abs(ex["stat"]["pct"].sum() - 1) < 1e-9, f"k={ex['k']}")
check("the first statistical factor carries most of the planted book's variance", ex["stat"].loc["PC1", "pct"] > 0.5,
      f"PC1 {ex['stat'].loc['PC1', 'pct']:.3f}, residual {ex['stat'].loc['residual', 'pct']:.3f}")
check("Fama-French market share agrees with the statistical block's residual share (within 0.1)",
      abs(ex["ff"].loc["residual", "pct"] - ex["stat"].loc["residual", "pct"]) < 0.1,
      f"ff residual {ex['ff'].loc['residual', 'pct']:.3f}")
try:
    risk.exposures({"A0": 1.0, "ZZZ": 0.5}, idx[-1], ret, factors, classes=classes)
    check("a name without history raises", False)
except ValueError as e:
    check("a name without history raises", "ZZZ" in str(e))

v = risk.var(pos, idx[-1], ret, garch_window=1000)
check("var returns the three methods with ES above VaR",
      all(v[m]["es"] > v[m]["var"] > 0 for m in ("parametric", "historical", "fhs")),
      ", ".join(f"{m} {v[m]['var'] * 100:.2f}%" for m in v))
p_true = float(np.sqrt(wv @ S_true[:20, :20] @ wv))
check("parametric VaR on the planted book is 2.33 sigma of the true vol (within 15%)",
      abs(v["parametric"]["var"] / (2.3263 * p_true) - 1) < 0.15, f"{v['parametric']['var']:.4f} vs {2.3263 * p_true:.4f}")

sd = {"A0": [0.01, -0.02, 0.03], "A1": [0.00, 0.01, -0.04]}
sret = pd.DataFrame(sd, index=pd.bdate_range("2008-09-01", periods=3))
st = risk.stress({"A0": 0.5, "A1": 0.5}, sret, {"w": ("2008-09-01", "2008-09-03")})["w"]
port = np.array([0.005, -0.005, -0.005])
check("stress compounds fixed weights through the window", abs(st["return"] - ((1 + port).prod() - 1)) < 1e-12
      and abs(st["worst_day"] + 0.005) < 1e-12 and abs(st["by_asset"]["A1"] - (-0.015)) < 1e-12)

k_exact = risk.kupiec(np.arange(1000) < 10, 0.99)
k_all = risk.kupiec(np.ones(200, bool), 0.99)
check("Kupiec: exact 1% coverage has p near 1, every-day breach near 0", k_exact["p_value"] > 0.99 and k_all["p_value"] < 1e-6)


def garch_t(t, omega=0.02, alpha=0.06, beta=0.92, nu=5):
    z = rng.standard_t(nu, t) * np.sqrt((nu - 2) / nu)
    s2, e = np.zeros(t), np.zeros(t)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * z[0]
    for i in range(1, t):
        s2[i] = omega + alpha * e[i - 1] ** 2 + beta * s2[i - 1]
        e[i] = np.sqrt(s2[i]) * z[i]
    return e / 100


Tv = 3500
common = garch_t(Tv)
sim = pd.DataFrame({f"S{i}": 0.6 * common + 0.8 * garch_t(Tv) for i in range(4)}, index=pd.bdate_range("2010-01-01", periods=Tv))
bt = risk.var_backtest(sim, {n: 0.25 for n in sim.columns}, garch_window=1000)
cov = risk.coverage(bt)
check("FHS 99% VaR coverage on GARCH-t returns is inside 0.5% to 1.6% with Kupiec p > 0.01",
      0.005 <= cov["fhs"]["rate"] <= 0.016 and cov["fhs"]["p_value"] > 0.01,
      f"rate {cov['fhs']['rate']:.4f} p {cov['fhs']['p_value']:.3f} n {cov['fhs']['n']}")
check("historical 99% VaR coverage is inside 0.5% to 2%", 0.005 <= cov["historical"]["rate"] <= 0.02,
      f"rate {cov['historical']['rate']:.4f}")
check("normal parametric breaches more often than FHS on fat-tailed clustered returns",
      cov["parametric"]["rate"] > cov["fhs"]["rate"], f"{cov['parametric']['rate']:.4f} vs {cov['fhs']['rate']:.4f}")

iid = pd.DataFrame(rng.normal(0, 0.01, (2500, 3)), columns=list("XYZ"), index=pd.bdate_range("2012-01-01", periods=2500))
bt2 = risk.var_backtest(iid, {"X": 0.4, "Y": 0.3, "Z": 0.3}, garch_window=504)
cov2 = risk.coverage(bt2)
check("parametric 99% VaR on iid normal returns covers 0.5% to 1.6%", 0.005 <= cov2["parametric"]["rate"] <= 0.016,
      f"rate {cov2['parametric']['rate']:.4f} n {cov2['parametric']['n']}")

close = pd.DataFrame(100 * np.exp(np.cumsum(R[:, :5], axis=0)), index=idx, columns=names[:5])
r_feat = risk.returns({"close": close, "volume": close * 0 + 1}, audit=True)
check("returns through the feature store pass the peek audit and match close-to-close",
      np.allclose(r_feat.iloc[1:].to_numpy(), (close / close.shift(1) - 1).iloc[1:].to_numpy()))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
