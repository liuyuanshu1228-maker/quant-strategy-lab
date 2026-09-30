"""Momentum (trend-following) strategy.

Rule: go long when the trailing N-day return is positive, flat
otherwise. A long-only, single-asset trend filter — simple enough to
explain in an interview, and deliberately does not try to predict
returns, only to follow an already-established trend.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import FLAT, LONG, SignalStrategy


class MomentumStrategy(SignalStrategy):
    """Long when trailing `lookback`-day return is positive."""

    def __init__(self, lookback: int = 20):
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.lookback = lookback
        self.name = f"Momentum(lookback={lookback})"

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        # Trailing return over `lookback` bars — backward-looking by
        # construction, so this never touches future data.
        momentum = close.pct_change(self.lookback)

        signal = pd.Series(FLAT, index=data.index, dtype=int)
        signal[momentum > 0] = LONG
        # Warm-up period (not enough history yet for a lookback
        # return): stay flat rather than guessing.
        signal[momentum.isna()] = FLAT
        return signal
