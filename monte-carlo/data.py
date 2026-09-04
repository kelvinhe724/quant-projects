"""Live spot, rate and option-chain inputs from Yahoo, for calibrating the engine to a real quote."""
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar

TRADING_DAYS = 252


def trading_days_between(start, end):
    """Business days from start to end, skipping US federal holidays.

    The federal calendar is a stand-in for the NYSE one: it shares Labor Day,
    Thanksgiving and the rest, but adds Columbus and Veterans Day and misses
    Good Friday. Close enough for a one- to two-month window.
    """
    hol = USFederalHolidayCalendar().holidays(start, end).values.astype("datetime64[D]")
    return int(np.busday_count(start, end, holidays=hol))


def risk_free_rate():
    """13-week T-bill discount rate as a decimal, standing in for the short rate."""
    px = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
    return float(px.iloc[-1]) / 100


def realised_vol(ticker="SPY", lookback=252):
    """Annualised standard deviation of daily log returns over the last `lookback` days."""
    px = yf.Ticker(ticker).history(period="2y")["Close"].dropna()
    ret = np.log(px).diff().dropna().iloc[-lookback:]
    return float(ret.std() * np.sqrt(TRADING_DAYS)), len(ret)


def atm_call(ticker="SPY", min_days=25, max_days=75):
    """Pick the nearest-the-money quoted call in the first expiry inside the day window.

    Returns the mid quote and everything the pricer needs. Expiry is converted to
    trading days with a holiday-aware business-day count, since the engine
    measures time in 252-day years. Candidates must sit within 2% of spot: after
    hours Yahoo zeroes the bids near the money and the nearest live quote can be
    a deep in-the-money strike whose mid sits below the discounted intrinsic.
    """
    tk = yf.Ticker(ticker)
    # history() can return NaN for today's close after hours, which would silently
    # pair yesterday's spot with today's chain; the live last price avoids that
    spot = float(tk.fast_info.last_price or tk.history(period="5d")["Close"].dropna().iloc[-1])
    today = pd.Timestamp.today().normalize()

    for expiry in tk.options:
        cal_days = (pd.Timestamp(expiry) - today).days
        if not min_days <= cal_days <= max_days:
            continue
        calls = tk.option_chain(expiry).calls
        calls = calls[(calls.bid > 0) & (calls.ask > calls.bid)
                      & ((calls.strike - spot).abs() < 0.02 * spot)].copy()
        if calls.empty:
            continue
        calls["mid"] = (calls.bid + calls.ask) / 2
        row = calls.iloc[(calls.strike - spot).abs().argsort().iloc[0]]
        return {
            "ticker": ticker,
            "spot": spot,
            "strike": float(row.strike),
            "expiry": expiry,
            "calendar_days": cal_days,
            "trading_days": trading_days_between(today.date(), pd.Timestamp(expiry).date()),
            "bid": float(row.bid),
            "ask": float(row.ask),
            "mid": float(row.mid),
            "volume": float(row.volume) if pd.notna(row.volume) else 0.0,
            "open_interest": float(row.openInterest) if pd.notna(row.openInterest) else 0.0,
            "yahoo_iv": float(row.impliedVolatility),
        }
    raise RuntimeError(f"no {ticker} expiry between {min_days} and {max_days} days out")
