"""Abstract interface for trading strategies.

Strategies here are pure signal generators: given historical OHLCV
data, they compute a target position (0 = flat, 1 = long) for every
bar, using only information available up to and including that bar.
They know nothing about order execution, cash, or commissions — that
is the backtest engine's job (see ``src/backtest/engine.py``).

This separation is the same interface / abstract-class pattern used
for polymorphic dispatch generally: swapping strategies means passing
a different ``SignalStrategy`` subclass to the backtest engine, with
no changes to the engine itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

FLAT = 0
LONG = 1


class SignalStrategy(ABC):
    """Base class all strategies implement.

    Subclasses must implement ``generate_signal`` only. The returned
    Series must be indexed exactly like the input ``data`` and contain
    only 0 (flat) or 1 (long) — this project only trades long
    positions, never short.
    """

    name: str = "SignalStrategy"

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute a position signal for every row of ``data``.

        Implementations must be causal: the value at row ``i`` may
        only depend on ``data.iloc[: i + 1]`` (e.g. a rolling window
        ending at ``i``), never on future rows. This is what keeps the
        backtest itself free of look-ahead bias in signal computation;
        the backtest engine additionally delays order fills to the
        following bar's open, so a signal known at bar close is never
        assumed tradable before the market had a chance to react.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Human-readable one-line summary for the UI/README."""
        return self.name
