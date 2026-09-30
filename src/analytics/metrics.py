"""Performance metrics computed from an equity curve and trade log.

These are implemented independently of backtesting.py (which reports
its own version of most of these numbers) so that:
1. the math is transparent and testable with synthetic data, and
2. the formulas can be explained precisely in an interview.

All functions are defensive about edge cases (flat/constant equity,
too few data points, zero trades) — they return a sane default (0.0)
rather than raising or returning NaN/inf, since these are meant to be
rendered directly on a dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Below this many periods, extrapolating to an annual rate is not
# statistically meaningful (a two-day return "annualized" via a
# geometric formula explodes to an absurd number) — fall back to the
# unannualized total return instead of reporting a misleading figure.
MIN_PERIODS_FOR_ANNUALIZATION = 5


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Simple period-over-period returns implied by an equity curve."""
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    """Total return over the full equity curve, e.g. 0.234 = +23.4%."""
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annualized_return(
    equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """CAGR implied by the equity curve's total return and length.

    Uses geometric annualization: (1 + total_return)^(periods_per_year
    / n_periods) - 1. If the strategy lost everything (or worse),
    fractional-power annualization is undefined, so this reports -1.0
    (-100%) instead of a complex/NaN result. With too few periods to
    extrapolate responsibly (see MIN_PERIODS_FOR_ANNUALIZATION), the
    unannualized total return is returned instead of a blown-up
    number.
    """
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return 0.0
    if n_periods < MIN_PERIODS_FOR_ANNUALIZATION:
        return total_return(equity)
    growth = 1 + total_return(equity)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n_periods) - 1)


def annualized_volatility(
    equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized standard deviation of period returns."""
    returns = returns_from_equity(equity)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of period returns vs. a risk-free rate.

    `risk_free_rate` is annual; it is converted to a per-period rate
    before subtracting. Returns 0.0 if there is no return variance to
    divide by (e.g. a flat equity curve), rather than +/-inf or NaN.
    """
    returns = returns_from_equity(equity)
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline, as a negative fraction.

    E.g. -0.231 means the equity curve fell 23.1% from a prior peak at
    its worst point. 0.0 for a flat or ever-rising curve.
    """
    if len(equity) < 1:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def win_rate(trades: pd.DataFrame, pnl_col: str = "PnL") -> float:
    """Fraction of closed trades with positive PnL. 0.0 if no trades."""
    if trades is None or len(trades) == 0:
        return 0.0
    return float((trades[pnl_col] > 0).mean())


def num_trades(trades: pd.DataFrame) -> int:
    """Count of closed trades. 0 if there are none."""
    return 0 if trades is None else int(len(trades))


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Bundle every headline metric into one dict for the dashboard."""
    return {
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity),
        "annualized_volatility": annualized_volatility(equity),
        "sharpe_ratio": sharpe_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "num_trades": num_trades(trades),
    }
