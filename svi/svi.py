"""Black-76 implied vols, raw-SVI slice calibration, and static arbitrage checks."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm

PARAM_NAMES = ["a", "b", "rho", "m", "sigma"]

# arbitrage is enforced and checked on the same log-moneyness grid, wider than
# any quoted strike, so the fitted slice is clean where it gets extrapolated too
ARB_GRID = np.linspace(-1.5, 1.5, 301)
PENALTY = 1e5


def bs_price(F, K, T, vol, D, cp):
    """Black-76 price of a European option on a forward, discounted by D."""
    F, K, T, vol = map(np.asarray, (F, K, T, vol))
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
        d2 = d1 - sd
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(np.asarray(cp) == "C", call, put)


def implied_vol(price, F, K, T, D, cp, lo=1e-4, hi=5.0):
    """Invert Black-76 for one quote. NaN if the price is outside no-arb bounds."""
    intrinsic = D * (F - K if cp == "C" else K - F)
    upper = D * (F if cp == "C" else K)
    if not (max(intrinsic, 0) < price < upper):
        return np.nan
    f = lambda v: float(bs_price(F, K, T, v, D, cp)) - price
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-8)


def add_implied_vols(q):
    """Attach bid/mid/ask implied vols and total variance to a clean panel."""
    q = q.copy()
    for side in ("bid", "mid", "ask"):
        q[f"iv_{side}"] = [
            implied_vol(row[side], row["F"], row["strike"], row["T"], row["D"], row["cp"])
            for _, row in q.iterrows()
        ]
    q = q.dropna(subset=["iv_mid"])
    for side in ("bid", "mid", "ask"):
        q[f"w_{side}"] = q[f"iv_{side}"] ** 2 * q["T"]
    return q.reset_index(drop=True)


def raw_svi(k, p):
    """Total implied variance w(k) under the raw SVI parameterisation."""
    a, b, rho, m, sigma = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x**2 + sigma**2))


def svi_derivs(k, p):
    """Return w, dw/dk, d2w/dk2 for the raw parameterisation."""
    a, b, rho, m, sigma = p
    x = k - m
    root = np.sqrt(x**2 + sigma**2)
    w = a + b * (rho * x + root)
    w1 = b * (rho + x / root)
    w2 = b * sigma**2 / root**3
    return w, w1, w2


def durrleman_g(k, p):
    """Durrleman's function. Non-negative everywhere means no butterfly arbitrage."""
    w, w1, w2 = svi_derivs(k, p)
    return (1 - k * w1 / (2 * w))**2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def butterfly_check(p, k_grid=None):
    """Worst Durrleman value and where it sits, for one slice."""
    k_grid = ARB_GRID if k_grid is None else k_grid
    g = durrleman_g(k_grid, p)
    i = int(np.argmin(g))
    return {"min_g": float(g[i]), "k_at_min": float(k_grid[i]),
            "violates": bool(g[i] < -1e-8)}


def calendar_check(fits, k_grid=None):
    """Pairwise check that total variance never falls as maturity rises."""
    k_grid = ARB_GRID if k_grid is None else k_grid
    fits = sorted(fits, key=lambda f: f["T"])
    out = []
    for lo, hi in zip(fits, fits[1:]):
        gap = raw_svi(k_grid, hi["params"]) - raw_svi(k_grid, lo["params"])
        i = int(np.argmin(gap))
        out.append({"T_short": lo["T"], "T_long": hi["T"],
                    "min_gap": float(gap[i]), "k_at_min": float(k_grid[i]),
                    "violates": bool(gap[i] < -1e-8)})
    return out


def market_butterfly_check(slice_df):
    """Convexity of call prices in strike, straight off the quotes.

    A negative second difference is a butterfly the market is quoting at a
    negative price. Runs on the raw mid quotes, before any SVI is fitted.
    """
    d = slice_df.sort_values("strike")
    F, T, D = d["F"].iloc[0], d["T"].iloc[0], d["D"].iloc[0]
    # put quotes are mapped to calls by parity so the whole strip is comparable
    C = bs_price(F, d["strike"].to_numpy(float), T, d["iv_mid"].to_numpy(float), D, "C")
    K = d["strike"].to_numpy(float)
    if len(K) < 3:
        return {"n_butterflies": 0, "n_negative": 0, "worst": np.nan}
    K1, K2, K3 = K[:-2], K[1:-1], K[2:]
    C1, C2, C3 = C[:-2], C[1:-1], C[2:]
    fly = (K3 - K2) / (K3 - K1) * C1 + (K2 - K1) / (K3 - K1) * C3 - C2
    return {"n_butterflies": len(fly), "n_negative": int((fly < -1e-9).sum()),
            "worst": float(fly.min())}


def market_calendar_check(q):
    """Total variance monotone in T at shared log-moneyness, on market vols.

    Interpolates each slice's market total variance onto a common k grid,
    restricted to the overlap of the two slices' quoted ranges.
    """
    out = []
    slices = [(T, g) for T, g in q.groupby("T")]
    slices.sort()
    for (T1, g1), (T2, g2) in zip(slices, slices[1:]):
        lo = max(g1["k"].min(), g2["k"].min())
        hi = min(g1["k"].max(), g2["k"].max())
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 101)
        w1 = np.interp(grid, g1.sort_values("k")["k"], g1.sort_values("k")["w_mid"])
        w2 = np.interp(grid, g2.sort_values("k")["k"], g2.sort_values("k")["w_mid"])
        gap = w2 - w1
        i = int(np.argmin(gap))
        out.append({"T_short": T1, "T_long": T2, "min_gap": float(gap[i]),
                    "k_at_min": float(grid[i]), "violates": bool(gap[i] < -1e-8)})
    return out


def _starts(k, w):
    """Structured multi-start grid, anchored on the observed variance level."""
    w_min, w_max = float(np.min(w)), float(np.max(w))
    span = max(k.max() - k.min(), 1e-3)
    out = []
    for m in (k[np.argmin(w)], 0.0, -0.05 * span):
        for sig in (0.05, 0.2, 0.5):
            for rho in (-0.8, -0.4, 0.0, 0.4):
                b = max((w_max - w_min) / span, 1e-3)
                out.append(np.array([max(w_min - b * sig, 1e-6), b, rho, m, sig]))
    return out


def fit_slice(k, w_mid, T, weights=None, w_bid=None, w_ask=None,
              eta=0.0, lam=0.0, ref=None, floor=None, enforce_butterfly=True):
    """Calibrate raw SVI to one expiry slice.

    weights scale the fit-to-mid squared error. eta prices quotes whose fitted
    variance escapes the bid-ask band, lam is a ridge pulling the parameters
    towards ref, and floor is a (k_grid, w_grid) pair the fitted slice must not
    dip below, which is how calendar arbitrage against the previous expiry is
    ruled out. enforce_butterfly adds Gatheral's necessary condition as a hard
    constraint and Durrleman's as a penalty over ARB_GRID.
    """
    k = np.asarray(k, float)
    w_mid = np.asarray(w_mid, float)
    weights = np.ones_like(k) if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum() * len(weights)
    g_grid = ARB_GRID

    def objective(p):
        w_fit = raw_svi(k, p)
        err = np.sum(weights * (w_fit - w_mid) ** 2)
        if eta and w_bid is not None:
            over = np.maximum(w_fit - w_ask, 0)
            under = np.maximum(w_bid - w_fit, 0)
            err += eta * np.sum(over**2 + under**2)
        if lam and ref is not None:
            err += lam * np.sum((p - ref) ** 2)
        if enforce_butterfly:
            err += PENALTY * np.sum(np.minimum(durrleman_g(g_grid, p), 0) ** 2)
        if floor is not None:
            fk, fw = floor
            err += PENALTY * np.sum(np.minimum(raw_svi(fk, p) - fw, 0) ** 2)
        return err

    w_max = float(np.max(w_mid))
    bounds = [(-2 * w_max, 2 * w_max), (1e-6, 5.0), (-0.999, 0.999), (-1.5, 1.5), (1e-4, 2.0)]
    cons = [{"type": "ineq", "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2)}]
    if enforce_butterfly:
        # Lee's moment bound: the wing slopes of total variance, b(1 +/- rho),
        # cannot exceed 2. Necessary, not sufficient; Durrleman does the rest.
        cons.append({"type": "ineq", "fun": lambda p: 2 - p[1] * (1 + abs(p[2]))})

    best, best_val = None, np.inf
    for p0 in _starts(k, w_mid):
        try:
            res = minimize(objective, p0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        except Exception:
            continue
        if res.success and res.fun < best_val:
            best, best_val = res.x, res.fun
    if best is None:
        raise RuntimeError("no SVI start converged")
    # the arbitrage penalties make the objective stiff and SLSQP sometimes
    # stalls short of the local minimum; one restart from the winner fixes it
    res = minimize(objective, best, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-14})
    return res.x if res.success and res.fun < best_val else best


def fit_surface(q, eta=0.0, lam=0.0, enforce_calendar=True, enforce_butterfly=True,
                weight_by_spread=True):
    """Fit every expiry in ascending maturity order.

    With enforce_calendar the previous slice's fitted variance becomes a floor
    for the next one, so the surface comes out calendar-arbitrage-free by
    construction rather than by luck.
    """
    fits, prev, ref = [], None, None
    for T, g in sorted(q.groupby("T"), key=lambda x: x[0]):
        g = g.sort_values("k")
        k = g["k"].to_numpy(float)
        band = (g["w_ask"] - g["w_bid"]).to_numpy(float)
        weights = (1.0 / np.clip(band, np.percentile(band, 10), None) ** 2
                   if weight_by_spread else None)
        floor = None
        if enforce_calendar and prev is not None:
            floor = (ARB_GRID, raw_svi(ARB_GRID, prev) + 1e-6)
        p = fit_slice(k, g["w_mid"].to_numpy(float), T, weights=weights,
                      w_bid=g["w_bid"].to_numpy(float), w_ask=g["w_ask"].to_numpy(float),
                      eta=eta, lam=lam, ref=ref, floor=floor,
                      enforce_butterfly=enforce_butterfly)
        iv_fit = np.sqrt(np.maximum(raw_svi(k, p), 1e-12) / T)
        resid = iv_fit - g["iv_mid"].to_numpy(float)
        fits.append({
            "expiry": g["expiry"].iloc[0], "T": T, "params": p, "n": len(g),
            "k": k, "iv_mid": g["iv_mid"].to_numpy(float), "iv_fit": iv_fit,
            "resid": resid,
            "rmse_vol": float(np.sqrt(np.mean(resid**2))),
            "max_abs_resid": float(np.max(np.abs(resid))),
            "outside_band": int(np.sum((raw_svi(k, p) > g["w_ask"].to_numpy(float)) |
                                       (raw_svi(k, p) < g["w_bid"].to_numpy(float)))),
        })
        prev, ref = p, p
    return fits


def params_table(fits):
    rows = [dict(zip(PARAM_NAMES, f["params"]), expiry=f["expiry"], T=f["T"],
                 n=f["n"], rmse_vol=f["rmse_vol"], outside_band=f["outside_band"])
            for f in fits]
    cols = ["expiry", "T", "n"] + PARAM_NAMES + ["rmse_vol", "outside_band"]
    return pd.DataFrame(rows)[cols]
