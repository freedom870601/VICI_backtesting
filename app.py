"""Streamlit dashboard for US Stock Backtesting System."""

from __future__ import annotations

import datetime
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
from backtest.strategy import generate_momentum_signals, generate_sma_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Distinct color palette for multi-ticker plots
_TICKER_COLORS = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def parse_tickers(raw_input: str) -> list[str]:
    """Parse comma-separated ticker string → deduplicated uppercase list."""
    seen: set[str] = set()
    result: list[str] = []
    for part in raw_input.split(","):
        t = part.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def run_ticker_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    strategy: str,
    fast_window: int,
    slow_window: int,
    lookback: int,
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
) -> dict | None:
    """Run full pipeline for one ticker. Returns None on failure (logs error)."""
    try:
        prices_df = fetch_prices(ticker, start_date, end_date)
    except ValueError as exc:
        logger.error("Failed to fetch data for %s: %s", ticker, exc)
        return None

    try:
        if strategy == "SMA Crossover":
            if len(prices_df) < slow_window + 1:
                logger.error(
                    "%s: not enough data (%d rows) for slow_window=%d",
                    ticker, len(prices_df), slow_window,
                )
                return None
            signals = generate_sma_signals(
                prices_df["close"], fast_window=fast_window, slow_window=slow_window
            )
        else:  # Momentum
            if len(prices_df) < lookback + 1:
                logger.error(
                    "%s: not enough data (%d rows) for lookback=%d",
                    ticker, len(prices_df), lookback,
                )
                return None
            signals = generate_momentum_signals(prices_df["close"], lookback=lookback)
    except ValueError as exc:
        logger.error("Strategy error for %s: %s", ticker, exc)
        return None

    result = run_backtest(
        signals,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    equity: pl.Series = result["equity"]
    trades: pl.DataFrame = result["trades"]
    daily_returns = equity.pct_change().drop_nulls()

    metrics = {
        "cagr": calculate_cagr(equity),
        "vol": calculate_annualized_volatility(daily_returns) if len(daily_returns) > 0 else 0.0,
        "sharpe": calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0,
        "mdd": calculate_max_drawdown(equity),
        "win_rate": calculate_win_rate(trades),
    }

    return {
        "ticker": ticker,
        "prices_df": prices_df,
        "signals": signals,
        "result": result,
        "metrics": metrics,
    }


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VICI Backtesting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 US Stock Backtesting System")
st.caption("Strategies: SMA Crossover · Momentum | Benchmark: Buy & Hold SPY")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    raw_tickers = st.text_input("Ticker Symbols (comma-separated)", value="AAPL")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date(2024, 1, 1))

    strategy = st.selectbox("Strategy", ["SMA Crossover", "Momentum"])

    fast_window = slow_window = lookback = 0
    if strategy == "SMA Crossover":
        st.subheader("SMA Windows")
        fast_window = st.slider("Fast SMA", min_value=5, max_value=100, value=20, step=1)
        slow_window = st.slider("Slow SMA", min_value=10, max_value=300, value=50, step=1)
        if fast_window >= slow_window:
            st.error("Fast SMA must be less than Slow SMA.")
            st.stop()
    else:
        st.subheader("Momentum Parameters")
        lookback = st.slider("Lookback Period (days)", min_value=20, max_value=504, value=252, step=1)

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000,
    )

    st.markdown("---")
    st.subheader("Transaction Costs")
    commission_pct = st.slider("Commission Rate (%)", min_value=0.00, max_value=2.00, value=0.00, step=0.01)
    slippage_pct = st.slider("Slippage Rate (%)", min_value=0.00, max_value=2.00, value=0.00, step=0.01)

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
    st.stop()

# ── Parse tickers ──────────────────────────────────────────────────────────────
tickers = parse_tickers(raw_tickers)
if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

commission_rate = commission_pct / 100.0
slippage_rate = slippage_pct / 100.0

# ── Run pipeline for each ticker ───────────────────────────────────────────────
ticker_results: list[dict] = []
failed_tickers: list[str] = []

with st.spinner(f"Running backtest for {', '.join(tickers)}…"):
    for ticker in tickers:
        res = run_ticker_pipeline(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            strategy=strategy,
            fast_window=fast_window,
            slow_window=slow_window,
            lookback=lookback,
            initial_capital=float(initial_capital),
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
        if res is None:
            failed_tickers.append(ticker)
        else:
            ticker_results.append(res)

for bad in failed_tickers:
    st.warning(f"⚠️ Could not fetch or process data for **{bad}** — skipped.")

if not ticker_results:
    st.error("No valid results. Check your ticker symbols and date range.")
    st.stop()

# ── Fetch SPY benchmark once ───────────────────────────────────────────────────
with st.spinner("Fetching SPY benchmark…"):
    try:
        spy_df = fetch_prices("SPY", str(start_date), str(end_date))
    except ValueError:
        spy_df = None

# ── Metrics comparison table ───────────────────────────────────────────────────
st.subheader("📊 Performance Metrics")

if len(ticker_results) == 1:
    # Single ticker: show original 5-column metric cards
    m = ticker_results[0]["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{m['cagr']:.1%}")
    c2.metric("Ann. Volatility", f"{m['vol']:.1%}")
    c3.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{m['mdd']:.1%}")
    c5.metric("Win Rate", f"{m['win_rate']:.1%}")
else:
    # Multi-ticker: show comparison table
    rows = []
    for r in ticker_results:
        m = r["metrics"]
        rows.append({
            "Ticker": r["ticker"],
            "CAGR": f"{m['cagr']:.1%}",
            "Ann. Vol": f"{m['vol']:.1%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Max Drawdown": f"{m['mdd']:.1%}",
            "Win Rate": f"{m['win_rate']:.1%}",
        })
    st.dataframe(pl.DataFrame(rows), use_container_width=True)

# ── Normalized equity curves chart ────────────────────────────────────────────
st.subheader("📉 Equity Curves (Normalized to 100)")

fig = go.Figure()

for idx, r in enumerate(ticker_results):
    equity_list = r["result"]["equity"].to_list()
    dates = r["prices_df"]["date"].to_list()
    first_val = equity_list[0] if equity_list[0] != 0 else 1.0
    normalized = [v / first_val * 100 for v in equity_list]
    color = _TICKER_COLORS[idx % len(_TICKER_COLORS)]
    fig.add_trace(go.Scatter(
        x=dates,
        y=normalized,
        mode="lines",
        name=r["ticker"],
        line=dict(color=color, width=2),
    ))

    # Add buy/sell markers only for single-ticker SMA crossover
    if len(ticker_results) == 1 and strategy == "SMA Crossover":
        signals_col = r["signals"]["signal"].to_list()
        buy_idx = [i for i, s in enumerate(signals_col) if s == 1]
        sell_idx = [i for i, s in enumerate(signals_col) if s == -1]
        if buy_idx:
            fig.add_trace(go.Scatter(
                x=[dates[i] for i in buy_idx],
                y=[normalized[i] for i in buy_idx],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10, color="green"),
            ))
        if sell_idx:
            fig.add_trace(go.Scatter(
                x=[dates[i] for i in sell_idx],
                y=[normalized[i] for i in sell_idx],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            ))

# SPY benchmark (dashed)
if spy_df is not None:
    spy_closes = spy_df["close"].to_list()
    spy_dates = spy_df["date"].to_list()
    spy_norm = [c / spy_closes[0] * 100 for c in spy_closes]
    fig.add_trace(go.Scatter(
        x=spy_dates,
        y=spy_norm,
        mode="lines",
        name="SPY (Buy & Hold)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Normalized Value (base 100)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── SMA chart (single ticker + SMA strategy only) ─────────────────────────────
if strategy == "SMA Crossover" and len(ticker_results) == 1:
    r = ticker_results[0]
    dates = r["prices_df"]["date"].to_list()
    signals = r["signals"]
    with st.expander("📈 Price & SMA Chart"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=r["prices_df"]["close"].to_list(),
            name="Close", line=dict(color="gray", width=1),
        ))
        fig2.add_trace(go.Scatter(
            x=dates, y=signals["fast_sma"].to_list(),
            name=f"Fast SMA ({fast_window})", line=dict(color="blue"),
        ))
        fig2.add_trace(go.Scatter(
            x=dates, y=signals["slow_sma"].to_list(),
            name=f"Slow SMA ({slow_window})", line=dict(color="orange"),
        ))
        fig2.update_layout(
            height=400, xaxis_title="Date", yaxis_title="Price ($)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Trade logs (one expander per ticker) ──────────────────────────────────────
st.subheader("📋 Trade Logs")
for r in ticker_results:
    trades = r["result"]["trades"]
    dates = r["prices_df"]["date"].to_list()
    win_rate = r["metrics"]["win_rate"]
    with st.expander(f"{r['ticker']} — Trade Log"):
        if trades.is_empty():
            st.info("No completed trades in the selected period.")
        else:
            display_trades = trades.with_columns([
                pl.col("entry_price").round(2),
                pl.col("exit_price").round(2),
                pl.col("pnl").round(2),
            ])
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
