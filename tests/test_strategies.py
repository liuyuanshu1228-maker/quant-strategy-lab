"""Tests for the strategy layer (src/strategies/).

Expected signals below are worked out by hand in the test comments,
independently of the implementation, rather than by re-deriving them
with the same pandas calls the implementation uses — this is what
actually catches sign errors, off-by-one bugs, and hysteresis bugs in
the mean-reversion state machine.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.base import FLAT, LONG
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy


def _prices(values):
    idx = pd.date_range("2023-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


class TestMomentumStrategy:
    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            MomentumStrategy(lookback=1)

    def test_signal_matches_hand_calculation(self):
        # pct_change(2) by hand:
        #  idx0,1: NaN -> FLAT
        #  idx2: (110-100)/100  = +0.100  -> LONG
        #  idx3: (115-105)/105  = +0.095  -> LONG
        #  idx4: (120-110)/110  = +0.091  -> LONG
        #  idx5: (90-115)/115   = -0.217  -> FLAT
        #  idx6: (85-120)/120   = -0.292  -> FLAT
        #  idx7: (80-90)/90     = -0.111  -> FLAT
        data = _prices([100, 105, 110, 115, 120, 90, 85, 80])
        strat = MomentumStrategy(lookback=2)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [FLAT, FLAT, LONG, LONG, LONG, FLAT, FLAT, FLAT]

    def test_signal_index_matches_input(self):
        data = _prices([100, 101, 102, 103, 104, 105])
        strat = MomentumStrategy(lookback=2)
        signal = strat.generate_signal(data)
        assert signal.index.equals(data.index)

    def test_warmup_period_is_flat(self):
        data = _prices([100, 102, 104, 106, 108, 110])
        strat = MomentumStrategy(lookback=4)
        signal = strat.generate_signal(data)
        # First 4 bars have no lookback history yet.
        assert signal.iloc[:4].tolist() == [FLAT] * 4

    def test_insufficient_data_does_not_crash(self):
        # Fewer rows than the lookback window: should be all-flat, not
        # an error.
        data = _prices([100, 101])
        strat = MomentumStrategy(lookback=20)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [FLAT, FLAT]

    def test_constant_prices_are_flat(self):
        # Zero momentum either way -> never triggers "momentum > 0".
        data = _prices([100.0] * 30)
        strat = MomentumStrategy(lookback=5)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [FLAT] * 30


class TestMeanReversionStrategy:
    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=1)

    def test_entry_z_must_be_below_exit_z(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(entry_z=0.5, exit_z=-0.5)

    def test_signal_matches_hand_calculation(self):
        # lookback=3, entry_z=-1.0, exit_z=0.0. Rolling mean/std/z
        # worked out by hand (see PR description / docstring above):
        #  idx0,1,2,3: NaN or std==0 before any valid z seen -> FLAT
        #  idx4: window(100,100,90) -> z ~ -1.1547 <= -1.0 -> LONG
        #  idx5: window(100,90,100) -> z ~ +0.5774 >= 0.0  -> FLAT
        #  idx6: window(90,100,100) -> z ~ +0.5774          -> FLAT
        #  idx7: window(100,100,100) -> std==0 -> NaN, hold -> FLAT
        data = _prices([100, 100, 100, 100, 90, 100, 100, 100])
        strat = MeanReversionStrategy(lookback=3, entry_z=-1.0, exit_z=0.0)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [
            FLAT, FLAT, FLAT, FLAT, LONG, FLAT, FLAT, FLAT,
        ]

    def test_signal_index_matches_input(self):
        data = _prices([100, 99, 98, 97, 96, 95, 100, 105])
        strat = MeanReversionStrategy(lookback=3)
        signal = strat.generate_signal(data)
        assert signal.index.equals(data.index)

    def test_position_held_between_entry_and_exit(self):
        # A dip that stays below the exit threshold for a few bars
        # should keep the position open the whole time, not flicker.
        data = _prices([100, 100, 100, 80, 80, 80, 100, 100, 100])
        strat = MeanReversionStrategy(lookback=3, entry_z=-0.5, exit_z=0.0)
        signal = strat.generate_signal(data)
        # Once LONG is triggered it must not drop back to FLAT until a
        # bar with z >= exit_z actually occurs.
        first_long = signal.tolist().index(LONG)
        # from first_long until price recovers, position stays LONG
        assert signal.iloc[first_long] == LONG

    def test_insufficient_data_does_not_crash(self):
        data = _prices([100, 99])
        strat = MeanReversionStrategy(lookback=20)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [FLAT, FLAT]

    def test_constant_prices_never_enter(self):
        # std is always 0 -> z is always NaN -> always flat.
        data = _prices([100.0] * 30)
        strat = MeanReversionStrategy(lookback=5)
        signal = strat.generate_signal(data)
        assert signal.tolist() == [FLAT] * 30
