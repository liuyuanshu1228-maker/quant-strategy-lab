"""Mean-reversion strategy based on a rolling Z-score.

Rule: go long when price is sufficiently below its rolling mean (a
statistically cheap dip, by this measure) and exit once price has
reverted back toward the mean.

    z = (price - rolling_mean) / rolling_std
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import FLAT, LONG, SignalStrategy


class MeanReversionStrategy(SignalStrategy):
    """Long below `entry_z`, flat again once z recovers to `exit_z`."""

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = -1.0,
        exit_z: float = 0.0,
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if entry_z >= exit_z:
            raise ValueError(
                "entry_z must be below exit_z (enter on a dip, "
                "exit once price has reverted)"
            )
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.name = (
            f"MeanReversion(lookback={lookback}, "
            f"entry_z={entry_z}, exit_z={exit_z})"
        )

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()
        z = (close - rolling_mean) / rolling_std

        # Entry/exit thresholds create hysteresis (a position, once
        # opened, is held until the exit condition — not just
        # re-evaluated as a stateless threshold each bar), so this is
        # a small causal state machine rather than a vectorized
        # comparison. Each step only reads z up to the current index.
        #
        # NaN handling: during warm-up (before the rolling window has
        # `lookback` points) there is no signal yet, so we stay flat.
        # A NaN that appears *after* warm-up (e.g. rolling_std is
        # briefly exactly zero) carries no new information either way
        # — rather than force-closing a real position on a data
        # artifact, we simply hold whatever position we already had.
        z_values = z.to_numpy()
        out = np.full(len(z_values), FLAT, dtype=int)
        position = FLAT
        seen_valid_z = False
        for i, zi in enumerate(z_values):
            if np.isnan(zi):
                if not seen_valid_z:
                    position = FLAT
                # else: no new information this bar — hold position.
            else:
                seen_valid_z = True
                if position == FLAT and zi <= self.entry_z:
                    position = LONG
                elif position == LONG and zi >= self.exit_z:
                    position = FLAT
            out[i] = position

        return pd.Series(out, index=data.index, dtype=int)
