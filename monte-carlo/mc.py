"""Monte Carlo option pricing under risk-neutral GBM, with variance reduction and Greeks.

Zero dividends throughout. Time is measured in trading days, 252 to the year.
"""
import numpy as np
from scipy import optimize
from scipy.stats import norm, qmc

STEPS_PER_YEAR = 252
CHUNK_ELEMS = 4_000_000


def n_steps_for(expiry_days):
    """Pick the discretisation: half-day steps under 30 days to expiry, whole days at or above."""
    step = 0.5 if expiry_days < 30 else 1.0
    return int(np.ceil(expiry_days / step))


def bs_price(S0, K, T, r, sigma, option_type="call"):
    """Black-Scholes price of a European option."""
    if T <= 0 or sigma <= 0:
        intrinsic = S0 - K if option_type == "call" else K - S0
        return max(intrinsic, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def bs_greeks(S0, K, T, r, sigma, option_type="call"):
    """Closed-form delta and vega. Vega is quoted per one volatility point."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return {"delta": delta, "vega": S0 * norm.pdf(d1) * np.sqrt(T) / 100}


def implied_vol(price, S0, K, T, r, option_type="call", lo=1e-4, hi=5.0):
    """Back out the volatility that reproduces an observed option price."""
    def gap(s):
        return bs_price(S0, K, T, r, s, option_type) - price

    if gap(lo) > 0 or gap(hi) < 0:
        return np.nan
    return optimize.brentq(gap, lo, hi, xtol=1e-10)


def geometric_asian_price(S0, K, T, r, sigma, m, option_type="call"):
    """Closed form for a discretely monitored geometric-average Asian option.

    The geometric average of lognormals is itself lognormal, which is the whole
    reason this option has a formula and the arithmetic one does not. Monitoring
    dates are t = T/m, 2T/m, ..., T.
    """
    dt = T / m
    mu = np.log(S0) + (r - 0.5 * sigma ** 2) * dt * (m + 1) / 2
    var = sigma ** 2 * dt * (m + 1) * (2 * m + 1) / (6 * m)
    sd = np.sqrt(var)
    d1 = (mu + var - np.log(K)) / sd
    d2 = d1 - sd
    fwd = np.exp(mu + 0.5 * var)
    if option_type == "call":
        return np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def _evolve(z, S0, r, sigma, dt):
    """Turn a block of standard normals into the path statistics every payoff needs."""
    log_path = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    S = np.exp(log_path)
    return {
        "ST": S[:, -1],
        "arith": S.mean(axis=1),
        "geom": np.exp(log_path.mean(axis=1)),
        "min": S.min(axis=1),
        "max": S.max(axis=1),
    }


def path_stats(S0, T, r, sigma, m, n_paths, method="plain", seed=0):
    """Simulate n_paths GBM paths over m steps and return per-path summaries.

    method is plain, antithetic, or sobol. Antithetic paths come back interleaved,
    so path 2i and 2i+1 are a mirrored pair. Paths are generated in blocks so peak
    memory stays flat regardless of n_paths.
    """
    dt = T / m
    rng = np.random.default_rng(seed)
    engine = qmc.Sobol(d=m, scramble=True, seed=seed) if method == "sobol" else None
    block = max(2, (CHUNK_ELEMS // m) // 2 * 2)
    if method == "sobol":
        block = 1 << max(1, int(np.log2(block)))  # Sobol is only balanced on powers of two

    parts, done = [], 0
    while done < n_paths:
        c = min(block, n_paths - done)
        if method == "antithetic":
            base = rng.standard_normal((c // 2, m))
            z = np.empty((c, m))
            z[0::2], z[1::2] = base, -base
        elif method == "sobol":
            z = norm.ppf(np.clip(engine.random(c), 1e-12, 1 - 1e-12))
        else:
            z = rng.standard_normal((c, m))
        parts.append(_evolve(z, S0, r, sigma, dt))
        done += c
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def payoff(st, K, option_type="call", option_style="european",
           barrier=None, barrier_type=None, asian_average="arithmetic"):
    """Undiscounted payoff per path, given the path summaries from path_stats."""
    if option_style == "asian":
        underlying = st["geom"] if asian_average == "geometric" else st["arith"]
    else:
        underlying = st["ST"]

    pay = np.maximum(underlying - K, 0) if option_type == "call" else np.maximum(K - underlying, 0)

    if option_style == "barrier":
        if barrier is None or barrier_type is None:
            raise ValueError("barrier options need barrier and barrier_type")
        hit = st["max"] >= barrier if barrier_type.startswith("up") else st["min"] <= barrier
        pay = pay * (hit if barrier_type.endswith("in") else ~hit)
    return pay


def control_variate(y, x, x_mean):
    """Shrink y's variance using a correlated x whose mean is known exactly."""
    b = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - b * (x - x_mean)


def price_option(S0, K, expiry_days, r, sigma, option_type="call", option_style="european",
                 n_paths=100_000, barrier=None, barrier_type=None,
                 asian_average="arithmetic", method="plain", control=False, seed=0):
    """Price one option by simulation and report its standard error.

    control=True subtracts a control variate: the discounted terminal price for
    European payoffs, the geometric-average Asian for arithmetic Asians.
    """
    T = expiry_days / STEPS_PER_YEAR
    m = n_steps_for(expiry_days)
    disc = np.exp(-r * T)
    st = path_stats(S0, T, r, sigma, m, n_paths, method, seed)
    y = disc * payoff(st, K, option_type, option_style, barrier, barrier_type, asian_average)

    if control:
        if option_style == "asian" and asian_average == "arithmetic":
            g = disc * payoff(st, K, option_type, "asian", asian_average="geometric")
            y = control_variate(y, g, geometric_asian_price(S0, K, T, r, sigma, m, option_type))
        elif option_style == "european":
            y = control_variate(y, disc * st["ST"], S0)
        else:
            raise ValueError(f"no control variate defined for {option_style}")

    if method == "antithetic":
        y = y.reshape(-1, 2).mean(axis=1)

    se = y.std(ddof=1) / np.sqrt(len(y))
    price = y.mean()
    return {"price": price, "se": se, "n_paths": n_paths, "n_samples": len(y),
            "ci": (price - 1.96 * se, price + 1.96 * se)}


def price_qmc(S0, K, expiry_days, r, sigma, n_paths=131_072, n_scrambles=16, seed=0, **kw):
    """Price with scrambled Sobol points, using independent scrambles for the error bar.

    A single Sobol sequence gives no honest standard error, since the points are
    not independent. Randomising the scramble and looking at the spread across
    replicates does. n_paths is per replicate; total work is n_paths * n_scrambles.
    """
    means = []
    per = n_paths // n_scrambles
    for i in range(n_scrambles):
        out = price_option(S0, K, expiry_days, r, sigma, n_paths=per,
                           method="sobol", seed=seed + 1000 * i, **kw)
        means.append(out["price"])
    means = np.array(means)
    price = means.mean()
    se = means.std(ddof=1) / np.sqrt(n_scrambles)
    return {"price": price, "se": se, "n_paths": per * n_scrambles, "n_samples": n_scrambles,
            "ci": (price - 1.96 * se, price + 1.96 * se)}


def greeks_fd(S0, K, expiry_days, r, sigma, h_s=None, h_v=0.01, n_paths=200_000, seed=0, **kw):
    """Central-difference delta and vega on common random numbers.

    Reusing the seed across the bumped runs is what makes this usable: with
    independent draws the differencing noise swamps the sensitivity.
    """
    h_s = h_s or 0.01 * S0
    f = lambda s, v: price_option(s, K, expiry_days, r, v, n_paths=n_paths, seed=seed, **kw)["price"]
    return {
        "delta": (f(S0 + h_s, sigma) - f(S0 - h_s, sigma)) / (2 * h_s),
        "vega": (f(S0, sigma + h_v) - f(S0, sigma - h_v)) / (2 * h_v) / 100,
    }


def greeks_pathwise(S0, K, expiry_days, r, sigma, option_type="call",
                    n_paths=200_000, method="plain", seed=0):
    """Differentiate the European payoff along each path, no bumping.

    Only valid because the vanilla payoff is continuous in S0 and sigma. It is
    not available for a knock-out, whose payoff jumps at the barrier.
    """
    T = expiry_days / STEPS_PER_YEAR
    m = n_steps_for(expiry_days)
    st = path_stats(S0, T, r, sigma, m, n_paths, method, seed)
    ST = st["ST"]
    disc = np.exp(-r * T)
    itm = (ST > K) if option_type == "call" else (ST < K)
    sign = 1.0 if option_type == "call" else -1.0

    d = disc * sign * itm * ST / S0
    v = disc * sign * itm * ST * (np.log(ST / S0) - (r + 0.5 * sigma ** 2) * T) / sigma
    return _summarise(d, v, method)


def greeks_lr(S0, K, expiry_days, r, sigma, option_type="call",
              n_paths=200_000, method="plain", seed=0):
    """Weight the payoff by the score of the terminal density instead of differentiating it.

    Works for any payoff, including discontinuous ones, at the cost of much
    noisier estimates.
    """
    T = expiry_days / STEPS_PER_YEAR
    m = n_steps_for(expiry_days)
    st = path_stats(S0, T, r, sigma, m, n_paths, method, seed)
    ST = st["ST"]
    disc = np.exp(-r * T)
    z = (np.log(ST / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pay = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    d = disc * pay * z / (S0 * sigma * np.sqrt(T))
    v = disc * pay * ((z ** 2 - 1) / sigma - z * np.sqrt(T))
    return _summarise(d, v, method)


def _summarise(d, v, method):
    if method == "antithetic":
        d, v = d.reshape(-1, 2).mean(axis=1), v.reshape(-1, 2).mean(axis=1)
    return {"delta": d.mean(), "delta_se": d.std(ddof=1) / np.sqrt(len(d)),
            "vega": v.mean() / 100, "vega_se": v.std(ddof=1) / np.sqrt(len(v)) / 100}
