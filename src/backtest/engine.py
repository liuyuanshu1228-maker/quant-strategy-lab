"""Backtest execution, built on top of backtesting.py.

Execution assumption (this is the part that matters for avoiding
look-ahead bias): a strategy's signal for bar t is computed using only
data available up to and including bar t's close (see
``src/strategies/base.py``). backtesting.py's default order semantics
(``trade_on_close=False``) fill an order placed while processing bar t
at bar **t+1's open** — so the earliest a signal can affect the
portfolio is the bar after it became known. This module relies on that
default rather than overriding it, and relies on backtesting.py for
all order/position/cash accounting rather than reimplementing a
trading engine from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from backtesting import Backtest, Strategy

from src.strategies.base import LONG, SignalStrategy


@dataclass
class BacktestConfig:
    """User-configurable backtest parameters."""

    initial_capital: float = 100_000.0
    commission: float = 0.001  # fraction per trade, e.g. 0.001 = 0.1%


@dataclass
class BacktestResult:
    """Everything the dashboard needs, pulled out of backtesting.py's
    stats object into a small, stable shape.
    """

    equity_curve: pd.Series  # portfolio value indexed by date
    trades: pd.DataFrame  # one row per closed trade
    raw_stats: pd.Series  # backtesting.py's own summary (for reference)


def _make_signal_adapter(signal: pd.Series) -> type[Strategy]:
    """Build a backtesting.py ``Strategy`` that just follows a
    precomputed signal.

    The signal is computed once, vectorized, by the ``SignalStrategy``
    (see strategies/); this adapter's only job is to translate
    "signal == LONG" into buy/close calls one bar at a time, the way
    backtesting.py expects. Wrapping it with ``self.I`` keeps it
    aligned with backtesting.py's internal bar-by-bar indexing.
    """

    class _SignalAdapter(Strategy):
        def init(self):
            self.signal_line = self.I(
                lambda: signal.to_numpy(), name="signal", plot=False
            )

        def next(self):
            current = self.signal_line[-1]
            if current == LONG and not self.position:
                self.buy()
            elif current != LONG and self.position:
                self.position.close()

    return _SignalAdapter


def run_backtest(
    data: pd.DataFrame,
    strategy: SignalStrategy,
    config: BacktestConfig,
) -> BacktestResult:
    """Run one backtest and return a dashboard-ready result.

    Raises ValueError if the strategy produces a signal that doesn't
    line up with the input data (a strategy bug, not a data problem).
    """
    if len(data) < 2:
        raise ValueError("Not enough data to backtest.")

    signal = strategy.generate_signal(data)
    if not signal.index.equals(data.index):
        raise ValueError(
            "Strategy signal index must exactly match the input data "
            "index (same dates, same order)."
        )

    adapter = _make_signal_adapter(signal)
    bt = Backtest(
        data,
        adapter,
        cash=config.initial_capital,
        commission=config.commission,
        exclusive_orders=True,
        # Close any position still open at the last bar so it counts
        # toward # of trades / win rate — otherwise a still-open
        # position affects the equity curve but is invisible to the
        # trade-level stats, which would make win rate and number of
        # trades inconsistent with what the equity curve shows.
        finalize_trades=True,
    )
    stats = bt.run()

    equity_curve = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        raw_stats=stats,
    )


def buy_and_hold_equity(
    data: pd.DataFrame, initial_capital: float
) -> pd.Series:
    """Benchmark equity curve: buy at the first bar's close, hold.

    Ignores commission (a single entry trade's cost is negligible and
    the point of this curve is a clean baseline, not a tradable
    strategy in its own right).
    """
    close = data["Close"]
    return initial_capital * (close / close.iloc[0])
