"""Tests for src/analytics/metrics.py.

All data here is deterministic and synthetic — no network access, no
randomness — so these tests are reproducible and fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    num_trades,
    sharpe_ratio,
    summarize,
    total_return,
    win_rate,
)


def _equity(values, periods="D"):
    idx = pd.date_range("2023-01-01", periods=len(values), freq=periods)
    return pd.Series(values, index=idx, dtype=float)


class TestTotalReturn:
    def test_basic_gain(self):
        eq = _equity([100.0, 110.0, 121.0])
        assert total_return(eq) == pytest.approx(0.21)

    def test_basic_loss(self):
        eq = _equity([100.0, 90.0, 81.0])
        assert total_return(eq) == pytest.approx(-0.19)

    def test_flat_equity_is_zero(self):
        eq = _equity([100.0] * 50)
        assert total_return(eq) == 0.0

    def test_single_point_is_zero(self):
        eq = _equity([100.0])
        assert total_return(eq) == 0.0


class TestAnnualizedReturn:
    def test_doubles_in_one_year(self):
        # 252 trading days, price doubles -> annualized return ~= 100%.
        eq = _equity(np.linspace(100.0, 200.0, 253))
        assert annualized_return(eq, periods_per_year=252) == pytest.approx(
            1.0, rel=1e-6
        )

    def test_total_wipeout_is_negative_one(self):
        eq = _equity(np.linspace(100.0, 0.0, 100))
        assert annualized_return(eq) == -1.0

    def test_too_few_periods_falls_back_to_total_return(self):
        # Only 2 data points (1 period) -- annualizing would explode;
        # should fall back to the plain total return instead.
        eq = _equity([100.0, 150.0])
        result = annualized_return(eq)
        assert result == pytest.approx(total_return(eq))
        assert abs(result) < 10  # sanity: not an astronomical number


class TestAnnualizedVolatility:
    def test_constant_equity_has_zero_vol(self):
        eq = _equity([100.0] * 30)
        assert annualized_volatility(eq) == 0.0

    def test_varying_equity_has_positive_vol(self):
        values = [100, 102, 99, 105, 101, 108, 103, 110, 106, 112]
        eq = _equity(values)
        assert annualized_volatility(eq) > 0.0

    def test_insufficient_data_is_zero(self):
        eq = _equity([100.0])
        assert annualized_volatility(eq) == 0.0


class TestSharpeRatio:
    def test_constant_equity_is_zero_not_nan_or_inf(self):
        eq = _equity([100.0] * 30)
        result = sharpe_ratio(eq)
        assert result == 0.0
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_steady_uptrend_is_positive(self):
        eq = _equity(np.linspace(100.0, 200.0, 100))
        assert sharpe_ratio(eq) > 0.0

    def test_steady_downtrend_is_negative(self):
        eq = _equity(np.linspace(200.0, 100.0, 100))
        assert sharpe_ratio(eq) < 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        # Peak 100 -> trough 80 -> partial recovery to 90.
        eq = _equity([100.0, 80.0, 90.0])
        assert max_drawdown(eq) == pytest.approx(-0.20)

    def test_new_highs_only_has_zero_drawdown(self):
        eq = _equity([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(eq) == 0.0

    def test_repeated_drawdowns_takes_the_worst(self):
        # -10% then recovery then -30% -- worst is -30%.
        eq = _equity([100.0, 90.0, 95.0, 100.0, 70.0, 80.0])
        assert max_drawdown(eq) == pytest.approx(-0.30)


class TestWinRateAndTrades:
    def test_mixed_trades(self):
        trades = pd.DataFrame({"PnL": [100.0, -50.0, 200.0, -10.0]})
        assert win_rate(trades) == pytest.approx(0.5)
        assert num_trades(trades) == 4

    def test_no_trades(self):
        trades = pd.DataFrame(columns=["PnL"])
        assert win_rate(trades) == 0.0
        assert num_trades(trades) == 0

    def test_none_trades(self):
        assert win_rate(None) == 0.0
        assert num_trades(None) == 0

    def test_all_winners(self):
        trades = pd.DataFrame({"PnL": [10.0, 20.0, 30.0]})
        assert win_rate(trades) == pytest.approx(1.0)


class TestSummarize:
    def test_returns_all_expected_keys(self):
        eq = _equity([100.0, 105.0, 102.0, 110.0])
        trades = pd.DataFrame({"PnL": [50.0, -20.0]})
        result = summarize(eq, trades)
        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "num_trades",
        }
        assert set(result.keys()) == expected_keys
        assert result["num_trades"] == 2
