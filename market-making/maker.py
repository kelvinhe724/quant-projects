"""Quoting strategies. Each returns a (bid, ask) pair given the efficient price and inventory."""
import math

DEFAULT_HALF_SPREAD = 0.85


class Maker:
    """Interface for a quoting strategy."""

    name = "maker"

    def quote(self, price, inventory, time_left, market):
        """Return the bid and ask to post, or None on a side to stand aside."""
        raise NotImplementedError


class Naive(Maker):
    """Post a fixed symmetric spread around the observed price and ignore inventory."""

    name = "naive"

    def __init__(self, half_spread=DEFAULT_HALF_SPREAD):
        self.half_spread = half_spread

    def quote(self, price, inventory, time_left, market):
        return price - self.half_spread, price + self.half_spread


class Skewed(Maker):
    """Shift both quotes against the position so the cheap side is the flattening side."""

    name = "skewed"

    def __init__(self, half_spread=DEFAULT_HALF_SPREAD, skew=0.15):
        self.half_spread = half_spread
        self.skew = skew

    def quote(self, price, inventory, time_left, market):
        centre = price - self.skew * inventory
        return centre - self.half_spread, centre + self.half_spread


class AvellanedaStoikov(Maker):
    """Reservation price and optimal spread from Avellaneda and Stoikov (2008).

        r      = s - q * gamma * sigma^2 * (T - t)
        spread = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

    The paper's price process is arithmetic Brownian motion. This market is GBM, so
    sigma is read as a local dollar volatility and rescaled by price / s0.
    """

    name = "avellaneda-stoikov"

    def __init__(self, gamma=0.1, sigma=None, kappa=None):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa

    def quote(self, price, inventory, time_left, market):
        sigma = self.sigma if self.sigma is not None else market.sigma * price / market.s0
        kappa = self.kappa if self.kappa is not None else market.decay
        variance = sigma ** 2 * max(time_left, 0.0)
        reservation = price - inventory * self.gamma * variance
        spread = self.gamma * variance + (2.0 / self.gamma) * math.log(1.0 + self.gamma / kappa)
        return reservation - spread / 2.0, reservation + spread / 2.0


def default_makers():
    """Build the three strategies compared in run.py."""
    return [Naive(), Skewed(), AvellanedaStoikov()]
