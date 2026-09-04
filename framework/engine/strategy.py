"""The one interface a strategy implements."""


class Strategy:
    """Map an as-of view of the market to target weights.

    on_bar is called once per trading day with `bars` already cut at `asof`.
    Return a dict of instrument -> weight of equity (sum of |w| is the gross
    exposure), or None to leave the current targets unchanged. Weights are
    filled on the next bar, never on this one.
    """
    name = None

    def on_bar(self, asof, bars):
        raise NotImplementedError

    def __str__(self):
        return self.name or type(self).__name__
