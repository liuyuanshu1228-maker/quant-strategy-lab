"""Quant Strategy Lab — Streamlit dashboard.

Wires together the data loader, strategies, backtest engine, and
metrics into an interactive research tool. Business logic (data
cleaning, signal generation, backtest execution, metric math) lives in
``src/``; this file is UI wiring only.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.metrics import summarize
from src.backtest.engine import BacktestConfig, buy_and_hold_equity, run_backtest
from src.data.loader import SUPPORTED_TICKERS, DataLoadError, load_market_data
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy

st.set_page_config(page_title="Quant Strategy Lab", layout="wide")


@st.cache_data(show_spinner="Downloading market data...")
def _cached_load_market_data(
    ticker: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    return load_market_data(ticker, start, end)


def _build_strategy(name: str, params: dict):
    if name == "Momentum":
        return MomentumStrategy(lookback=params["lookback"])
    if name == "Mean Reversion":
        return MeanReversionStrategy(
            lookback=params["lookback"],
            entry_z=params["entry_z"],
            exit_z=params["exit_z"],
        )
    raise ValueError(f"Unknown strategy: {name}")


def _render_sidebar() -> dict:
    st.sidebar.header("Configuration")

    ticker = st.sidebar.selectbox("Ticker", SUPPORTED_TICKERS, index=0)
    strategy_name = st.sidebar.selectbox(
        "Strategy", ["Momentum", "Mean Reversion"]
    )

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)
    start = st.sidebar.date_input(
        "Start date", value=default_start, max_value=today
    )
    end = st.sidebar.date_input("End date", value=today, max_value=today)

    initial_capital = st.sidebar.number_input(
        "Initial capital ($)", min_value=1_000.0, value=100_000.0, step=1_000.0
    )
    commission_pct = st.sidebar.number_input(
        "Transaction cost (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.10,
        step=0.01,
        help="Applied as a fraction of trade value on every fill.",
    )

    st.sidebar.subheader("Strategy parameters")
    params: dict = {}
    if strategy_name == "Momentum":
        params["lookback"] = st.sidebar.slider(
            "Lookback (days)", min_value=5, max_value=120, value=20
        )
    else:
        params["lookback"] = st.sidebar.slider(
            "Lookback (days)", min_value=5, max_value=120, value=20
        )
        params["entry_z"] = st.sidebar.slider(
            "Entry Z-score", min_value=-3.0, max_value=0.0, value=-1.0, step=0.1
        )
        params["exit_z"] = st.sidebar.slider(
            "Exit Z-score", min_value=0.0, max_value=3.0, value=0.0, step=0.1
        )

    run_clicked = st.sidebar.button(
        "Run Backtest", type="primary", width="stretch"
    )

    return {
        "ticker": ticker,
        "strategy_name": strategy_name,
        "params": params,
        "start": start,
        "end": end,
        "initial_capital": initial_capital,
        "commission": commission_pct / 100.0,
        "run_clicked": run_clicked,
    }


def _render_metric_cards(metrics: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")
    cols[1].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    cols[2].metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    cols[3].metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")


def _render_equity_chart(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    strategy_label: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            name=strategy_label,
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            name="Buy & Hold",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Strategy vs Buy & Hold — Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


def _render_additional_stats(metrics: dict, benchmark_return: float) -> None:
    cols = st.columns(4)
    cols[0].metric(
        "Annualized Return", f"{metrics['annualized_return'] * 100:.2f}%"
    )
    cols[1].metric(
        "Annualized Volatility",
        f"{metrics['annualized_volatility'] * 100:.2f}%",
    )
    cols[2].metric("Number of Trades", f"{metrics['num_trades']}")
    cols[3].metric("Buy & Hold Return", f"{benchmark_return * 100:.2f}%")


def _render_trade_history(trades: pd.DataFrame) -> None:
    st.subheader("Trade History")
    if trades is None or trades.empty:
        st.info("No trades were opened and closed during this period.")
        return
    display_cols = [
        "EntryTime",
        "ExitTime",
        "EntryPrice",
        "ExitPrice",
        "Size",
        "PnL",
        "ReturnPct",
    ]
    display_cols = [c for c in display_cols if c in trades.columns]
    st.dataframe(trades[display_cols], width="stretch", hide_index=True)


def _render_strategy_explanation(strategy_name: str, params: dict) -> None:
    with st.expander("Strategy explanation & assumptions", expanded=False):
        if strategy_name == "Momentum":
            st.markdown(
                f"**Momentum** — go long when the trailing "
                f"**{params['lookback']}-day** return is positive, flat "
                f"otherwise.\n\n"
                f"- Rule: `signal = LONG if pct_change(Close, "
                f"{params['lookback']}) > 0 else FLAT`\n"
                f"- Long-only, no shorting."
            )
        else:
            st.markdown(
                f"**Mean Reversion** — go long when price is "
                f"`{params['entry_z']}` standard deviations below its "
                f"**{params['lookback']}-day** rolling mean (Z-score), and "
                f"exit once the Z-score recovers to `{params['exit_z']}`.\n\n"
                f"- `z = (price - rolling_mean) / rolling_std`\n"
                f"- Long-only, no shorting."
            )
        st.markdown(
            "\n**Execution & bias notes**\n"
            "- A signal computed from bar *t*'s close is only ever filled "
            "at bar *t+1*'s open — the backtest never assumes a trade "
            "could happen before the market had a chance to react.\n"
            "- Transaction costs are applied on every fill using the "
            "commission rate set in the sidebar.\n"
            "- Strategy parameters here are **not** optimized against "
            "this data — they are fixed, reasonable defaults you can "
            "adjust yourself.\n"
            "- Benchmark is a simple Buy & Hold of the same ticker over "
            "the same period.\n\n"
            "**Disclaimer:** this is an educational/research project, "
            "not financial advice. Backtested performance does not "
            "indicate future results."
        )


def main() -> None:
    st.title("Quant Strategy Lab")
    st.caption(
        "Research, backtest, and evaluate systematic trading strategies."
    )

    cfg = _render_sidebar()

    if not cfg["run_clicked"]:
        st.info("Configure a backtest in the sidebar and click **Run Backtest**.")
        return

    try:
        data = _cached_load_market_data(cfg["ticker"], cfg["start"], cfg["end"])
    except DataLoadError as exc:
        st.error(str(exc))
        return

    strategy = _build_strategy(cfg["strategy_name"], cfg["params"])
    backtest_cfg = BacktestConfig(
        initial_capital=cfg["initial_capital"], commission=cfg["commission"]
    )

    try:
        result = run_backtest(data, strategy, backtest_cfg)
    except ValueError as exc:
        st.error(f"Backtest failed: {exc}")
        return

    benchmark_equity = buy_and_hold_equity(data, cfg["initial_capital"])
    metrics = summarize(result.equity_curve, result.trades)
    benchmark_return = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1

    _render_metric_cards(metrics)
    st.divider()
    _render_equity_chart(result.equity_curve, benchmark_equity, strategy.describe())
    _render_additional_stats(metrics, benchmark_return)
    st.divider()
    _render_trade_history(result.trades)
    _render_strategy_explanation(cfg["strategy_name"], cfg["params"])


if __name__ == "__main__":
    main()
