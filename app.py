"""Streamlit dashboard for US Stock SMA Crossover Backtesting System."""

from __future__ import annotations

import logging

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from backtest.data import fetch_prices
from backtest.engine import run_backtest
from backtest.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from backtest.strategy import generate_sma_signals

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VICI Backtesting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 US Stock SMA Crossover Backtesting")
st.caption("Strategy: Fast/Slow Simple Moving Average crossover · Benchmark: Buy & Hold SPY")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=__import__("datetime").date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=__import__("datetime").date(2024, 1, 1))

    st.subheader("SMA Windows")
    fast_window = st.slider("Fast SMA", min_value=5, max_value=100, value=20, step=1)
    slow_window = st.slider("Slow SMA", min_value=10, max_value=300, value=50, step=1)

    if fast_window >= slow_window:
        st.error("Fast SMA must be less than Slow SMA.")
        st.stop()

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000,
    )

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
    st.stop()

# ── Data fetching ─────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}…"):
    try:
        prices_df = fetch_prices(ticker, str(start_date), str(end_date))
    except ValueError as e:
        st.error(f"Data error: {e}")
        st.stop()

if len(prices_df) < slow_window + 1:
    st.error(
        f"Not enough data ({len(prices_df)} rows) for slow SMA window ({slow_window}). "
        "Widen the date range or reduce the slow SMA window."
    )
    st.stop()

# Fetch SPY benchmark (same date range)
with st.spinner("Fetching SPY benchmark…"):
    try:
        spy_df = fetch_prices("SPY", str(start_date), str(end_date))
    except ValueError:
        spy_df = None

# ── Strategy & backtest ───────────────────────────────────────────────────────
signals = generate_sma_signals(prices_df["close"], fast_window=fast_window, slow_window=slow_window)
result = run_backtest(signals, initial_capital=float(initial_capital))

equity: pl.Series = result["equity"]
trades: pl.DataFrame = result["trades"]

# Daily returns from equity curve (pct change, drop first null)
daily_returns = equity.pct_change().drop_nulls()

# ── Metrics ───────────────────────────────────────────────────────────────────
cagr = calculate_cagr(equity)
vol = calculate_annualized_volatility(daily_returns) if len(daily_returns) > 0 else 0.0
sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0
mdd = calculate_max_drawdown(equity)
win_rate = calculate_win_rate(trades)

st.subheader("📊 Performance Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CAGR", f"{cagr:.1%}")
m2.metric("Ann. Volatility", f"{vol:.1%}")
m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
m4.metric("Max Drawdown", f"{mdd:.1%}")
m5.metric("Win Rate", f"{win_rate:.1%}")

# ── Equity curve chart ────────────────────────────────────────────────────────
st.subheader("📉 Equity Curve")

dates = prices_df["date"].to_list()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates,
    y=equity.to_list(),
    mode="lines",
    name=f"{ticker} Strategy",
    line=dict(color="#1f77b4", width=2),
))

# SPY benchmark — normalized to initial_capital
if spy_df is not None:
    spy_closes = spy_df["close"].to_list()
    spy_dates = spy_df["date"].to_list()
    spy_normalized = [c * (float(initial_capital) / spy_closes[0]) for c in spy_closes]
    fig.add_trace(go.Scatter(
        x=spy_dates,
        y=spy_normalized,
        mode="lines",
        name="SPY (Buy & Hold)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

# Buy/sell markers
buy_indices = [i for i, s in enumerate(signals["signal"].to_list()) if s == 1]
sell_indices = [i for i, s in enumerate(signals["signal"].to_list()) if s == -1]

if buy_indices:
    fig.add_trace(go.Scatter(
        x=[dates[i] for i in buy_indices],
        y=[equity.to_list()[i] for i in buy_indices],
        mode="markers",
        name="BUY",
        marker=dict(symbol="triangle-up", size=10, color="green"),
    ))

if sell_indices:
    fig.add_trace(go.Scatter(
        x=[dates[i] for i in sell_indices],
        y=[equity.to_list()[i] for i in sell_indices],
        mode="markers",
        name="SELL",
        marker=dict(symbol="triangle-down", size=10, color="red"),
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
    margin=dict(l=0, r=0, t=30, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# ── SMA chart ─────────────────────────────────────────────────────────────────
with st.expander("📈 Price & SMA Chart"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=prices_df["close"].to_list(), name="Close", line=dict(color="gray", width=1)))
    fig2.add_trace(go.Scatter(x=dates, y=signals["fast_sma"].to_list(), name=f"Fast SMA ({fast_window})", line=dict(color="blue")))
    fig2.add_trace(go.Scatter(x=dates, y=signals["slow_sma"].to_list(), name=f"Slow SMA ({slow_window})", line=dict(color="orange")))
    fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Trade log ─────────────────────────────────────────────────────────────────
st.subheader("📋 Trade Log")
if trades.is_empty():
    st.info("No completed trades in the selected period.")
else:
    display_trades = trades.with_columns([
        pl.col("entry_price").round(2),
        pl.col("exit_price").round(2),
        pl.col("pnl").round(2),
    ])
    # Map index back to dates for readability
    entry_dates = [dates[i] if 0 <= i < len(dates) else None for i in display_trades["entry_date"].to_list()]
    exit_dates = [dates[i] if 0 <= i < len(dates) else None for i in display_trades["exit_date"].to_list()]
    display_df = display_trades.with_columns([
        pl.Series("entry_date", entry_dates),
        pl.Series("exit_date", exit_dates),
    ])
    st.dataframe(display_df, use_container_width=True)
    st.caption(
        f"Total trades: {trades.height} | "
        f"Total PnL: ${trades['pnl'].sum():,.2f} | "
        f"Win rate: {win_rate:.1%}"
    )
