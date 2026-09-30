"""Market data loading and validation.

Responsible for turning a (ticker, start, end) request into a clean
OHLCV DataFrame in the exact shape backtesting.py expects:
columns ``Open, High, Low, Close, Volume`` and a DatetimeIndex.

Design notes
------------
- Errors are never swallowed: every failure mode raises ``DataLoadError``
  with a message safe to show directly in the Streamlit UI.
- No look-ahead is introduced here: this module only fetches and cleans
  historical data, it does not touch anything related to signal timing.
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import yfinance as yf

SUPPORTED_TICKERS = ("SPY", "QQQ", "AAPL", "MSFT")

# Most strategies here use rolling windows; require enough bars that a
# lookback of up to ~60 days still leaves data to actually trade on.
MIN_TRADING_DAYS = 90

_TICKER_RE = re.compile(r"^[A-Z.\-]{1,10}$")


class DataLoadError(Exception):
    """Raised for any problem retrieving or validating market data.

    The message is written to be shown as-is in the Streamlit UI.
    """


def validate_ticker(ticker: str) -> str:
    """Validate and normalize a ticker symbol.

    Accepts any syntactically plausible ticker (not just
    ``SUPPORTED_TICKERS``) so the loader stays reusable, but the
    dashboard's dropdown is restricted to ``SUPPORTED_TICKERS``.
    """
    if not ticker or not isinstance(ticker, str):
        raise DataLoadError("Ticker must be a non-empty string.")
    ticker = ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise DataLoadError(
            f"'{ticker}' does not look like a valid ticker symbol."
        )
    return ticker


def validate_date_range(
    start: dt.date, end: dt.date
) -> tuple[dt.date, dt.date]:
    """Validate a requested date range.

    Raises DataLoadError if the range is empty, inverted, in the
    future, or too short to support the strategies' rolling windows.
    """
    today = dt.date.today()
    if start >= end:
        raise DataLoadError("Start date must be before end date.")
    if end > today:
        raise DataLoadError("End date cannot be in the future.")
    span_days = (end - start).days
    if span_days < MIN_TRADING_DAYS:
        raise DataLoadError(
            f"Date range is too short ({span_days} days). "
            f"Pick a range of at least ~{MIN_TRADING_DAYS} calendar days "
            "so rolling indicators have enough history."
        )
    return start, end


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance can return a MultiIndex column header for a single
    ticker depending on version/config; collapse it to plain names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def load_market_data(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Download and clean historical OHLCV data for one ticker.

    Returns a DataFrame indexed by date with columns
    ``Open, High, Low, Close, Volume`` — the exact shape
    ``backtesting.Backtest`` requires. Raises ``DataLoadError`` on any
    problem (bad input, network/API failure, empty result, unusable
    data) rather than returning a partially-broken frame.
    """
    ticker = validate_ticker(ticker)
    start, end = validate_date_range(start, end)

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # network/API errors from yfinance/curl
        raise DataLoadError(
            f"Failed to download data for '{ticker}': {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise DataLoadError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    raw = _flatten_columns(raw)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise DataLoadError(
            f"Downloaded data for '{ticker}' is missing columns: "
            f"{missing_cols}."
        )

    df = raw[required].copy()

    # Missing values: forward-fill small gaps (e.g. a stale print),
    # then drop any row that is still incomplete (e.g. leading NaNs
    # before the series starts). This is documented because silently
    # interpolating price data would misrepresent what actually
    # happened in the market.
    df = df.ffill()
    df = df.dropna(subset=required)

    if df.empty:
        raise DataLoadError(
            f"Data for '{ticker}' was empty after cleaning missing values."
        )

    if len(df) < MIN_TRADING_DAYS // 2:
        raise DataLoadError(
            f"Only {len(df)} usable trading days for '{ticker}' in this "
            "range — not enough history to backtest reliably."
        )

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df
