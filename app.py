"""Streamlit dashboard for US Stock Backtesting System."""

from __future__ import annotations

import datetime
import logging

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from backtest.data import fetch_prices
from backtest.engine import run_backtest
from backtest.factor import run_capm_regression, run_long_short_backtest
from backtest.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
    holding_period_stats,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
)
from backtest.strategy import generate_sma_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_prices(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Cached wrapper around fetch_prices; results expire after 1 hour."""
    return fetch_prices(ticker, start, end)


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
    fast_window: int,
    slow_window: int,
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    spread_bps: float = 0.0,
    entry_price_type: str = "close",
) -> dict | None:
    """Run full pipeline for one ticker. Returns None on failure (logs error)."""
    try:
        prices_df = cached_fetch_prices(ticker, start_date, end_date)
    except Exception as exc:
        logger.error("Failed to fetch data for %s: %s", ticker, exc)
        return None

    try:
        if len(prices_df) < slow_window + 1:
            logger.error("%s: not enough data (%d rows) for slow_window=%d",
                         ticker, len(prices_df), slow_window)
            return None
        signals = generate_sma_signals(
            prices_df["close"], fast_window=fast_window, slow_window=slow_window
        )
    except ValueError as exc:
        logger.error("Strategy error for %s: %s", ticker, exc)
        return None

    if entry_price_type == "open" and "open" in prices_df.columns:
        signals = signals.with_columns(prices_df["open"])

    result = run_backtest(
        signals,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        spread_bps=spread_bps,
        entry_price_type=entry_price_type,
    )

    equity: pl.Series = result["equity"]
    trades: pl.DataFrame = result["trades"]
    daily_returns = equity.pct_change().drop_nulls()

    return {
        "ticker": ticker,
        "prices_df": prices_df,
        "signals": signals,
        "result": result,
        "metrics": {
            "cagr": calculate_cagr(equity),
            "vol": calculate_annualized_volatility(daily_returns) if len(daily_returns) > 0 else 0.0,
            "sharpe": calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0,
            "mdd": calculate_max_drawdown(equity),
            "win_rate": calculate_win_rate(trades),
        },
    }


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="VICI Backtesting", page_icon="📈", layout="wide")
st.title("📈 US Stock Backtesting System")
st.caption("Strategy: SMA Crossover | Benchmark: Buy & Hold SPY")

# ── Sidebar: mode switcher + context-sensitive controls ──────────────────────
with st.sidebar:
    mode = st.radio(
        "Mode",
        ["📈 Single Stock", "🔬 Factor Analysis"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Single Stock sidebar ─────────────────────────────────────────────────
    if mode == "📈 Single Stock":
        st.header("⚙️ Backtest Parameters")

        raw_tickers = st.text_input("Ticker Symbols (comma-separated)", value="AAPL")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.date(2024, 1, 1))

        st.subheader("SMA Windows")
        fast_window = st.slider("Fast SMA", min_value=5, max_value=100, value=20, step=1, help="Fast Simple Moving Average window (days). A shorter window reacts quickly to price changes.")
        slow_window = st.slider("Slow SMA", min_value=10, max_value=300, value=50, step=1, help="Slow Simple Moving Average window (days). A longer window reflects the broader trend.")
        sma_valid = fast_window < slow_window
        if not sma_valid:
            st.error("⚠️ Fast SMA must be less than Slow SMA.")

        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
            value=10_000, step=1_000,
        )

        st.markdown("---")
        st.subheader("Transaction Costs")
        commission_pct = st.slider("Commission (%)", min_value=0.00, max_value=2.00, value=0.00, step=0.01, help="Broker fee charged per trade, as a percentage of the trade value.")
        slippage_pct   = st.slider("Slippage (%)",   min_value=0.00, max_value=2.00, value=0.00, step=0.01, help="Estimated price impact: the difference between the expected and actual fill price.")
        spread_bps_val = st.slider("Bid-Ask Spread (bps)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, help="Bid-ask spread in basis points (1 bps = 0.01%). Buying costs more; selling nets less.")
        entry_price_type = st.selectbox("Entry Price", ["close", "open"], index=0, help="'close' fills at today's closing price; 'open' fills at the next bar's opening price.")

        run_btn = st.button(
            "▶ Run Backtest", type="primary", use_container_width=True,
            disabled=not sma_valid,
        )
        run_factor_btn = False

    # ── Factor Analysis sidebar ──────────────────────────────────────────────
    else:
        st.header("⚙️ Factor Analysis Parameters")

        factor_raw_tickers = st.text_input(
            "Stock Universe (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM",
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            factor_start = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
        with fc2:
            factor_end   = st.date_input("End Date",   value=datetime.date(2024, 1, 1))

        st.markdown("---")
        st.subheader("Long-Short Settings")
        factor_top_n    = st.number_input("Long Top N", min_value=1, max_value=20, value=3, help="Number of highest-momentum stocks to go long (buy) each rebalance period.")
        factor_bottom_n = st.number_input("Short Bottom N", min_value=1, max_value=20, value=3, help="Number of lowest-momentum stocks to go short (sell) each rebalance period.")

        st.markdown("---")
        st.subheader("Strategy")
        factor_lookback = st.slider("Momentum Lookback (days)", min_value=20, max_value=252, value=63, step=1, help="Look-back window used to rank stocks by past return. Longer = slower-changing rankings.")

        st.markdown("---")
        st.subheader("Transaction Costs")
        factor_commission = st.slider("Commission (%)", min_value=0.00, max_value=2.00, value=0.10, step=0.01, help="Broker fee charged per trade, as a percentage of the trade value.")
        factor_spread     = st.slider("Bid-Ask Spread (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, help="Bid-ask spread in basis points (1 bps = 0.01%). Buying costs more; selling nets less.")

        run_factor_btn = st.button("▶ Run Factor Analysis", type="primary", use_container_width=True)
        run_btn = False


# ============================================================
# Main content: renders based on selected mode
# ============================================================

# ── Single Stock ─────────────────────────────────────────────────────────────
if mode == "📈 Single Stock":
    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
    else:
        tickers = parse_tickers(raw_tickers)
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
        else:
            ticker_results: list[dict] = []
            failed_tickers: list[str] = []

            with st.spinner(f"Running backtest for {', '.join(tickers)}…"):
                for ticker in tickers:
                    res = run_ticker_pipeline(
                        ticker=ticker,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        fast_window=fast_window,
                        slow_window=slow_window,
                        initial_capital=float(initial_capital),
                        commission_rate=commission_pct / 100.0,
                        slippage_rate=slippage_pct / 100.0,
                        spread_bps=spread_bps_val,
                        entry_price_type=entry_price_type,
                    )
                    if res is None:
                        failed_tickers.append(ticker)
                    else:
                        ticker_results.append(res)

            for bad in failed_tickers:
                st.warning(f"⚠️ Could not fetch or process data for **{bad}** — skipped.")

            if not ticker_results:
                st.error("No valid results. Check your ticker symbols and date range.")
            else:
                with st.spinner("Fetching SPY benchmark…"):
                    try:
                        spy_df = cached_fetch_prices("SPY", str(start_date), str(end_date))
                    except Exception:
                        spy_df = None

                tab_overview, tab_charts, tab_trades, tab_advanced = st.tabs([
                    "📊 Overview", "📉 Charts", "📋 Trades", "🔬 Advanced"
                ])

                # ── Tab 1: Overview ──────────────────────────────────────────────
                with tab_overview:
                    if len(ticker_results) == 1:
                        m = ticker_results[0]["metrics"]
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("CAGR", f"{m['cagr']:.1%}", help="Compound Annual Growth Rate: the smoothed annual return over the full period.")
                        c2.metric("Ann. Volatility", f"{m['vol']:.1%}", help="Annualized standard deviation of daily returns — a measure of risk.")
                        c3.metric("Sharpe Ratio", f"{m['sharpe']:.2f}", help="Risk-adjusted return: (portfolio return − risk-free rate) ÷ volatility. Higher is better.")
                        c4.metric("Max Drawdown", f"{m['mdd']:.1%}", help="Largest peak-to-trough decline in portfolio value. Measures worst-case loss.")
                        c5.metric("Win Rate", f"{m['win_rate']:.1%}", help="Proportion of completed trades that were profitable.")
                    else:
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

                # ── Tab 2: Charts ────────────────────────────────────────────────
                with tab_charts:
                    st.subheader("Equity Curves (Normalized to 100)")
                    fig = go.Figure()
                    for idx, r in enumerate(ticker_results):
                        equity_list = r["result"]["equity"].to_list()
                        dates = r["prices_df"]["date"].to_list()
                        first_val = equity_list[0] if equity_list[0] != 0 else 1.0
                        normalized = [v / first_val * 100 for v in equity_list]
                        color = _TICKER_COLORS[idx % len(_TICKER_COLORS)]
                        fig.add_trace(go.Scatter(
                            x=dates, y=normalized, mode="lines",
                            name=r["ticker"], line=dict(color=color, width=2),
                        ))
                        if len(ticker_results) == 1:
                            sigs = r["signals"]["signal"].to_list()
                            buy_idx  = [i for i, s in enumerate(sigs) if s == 1]
                            sell_idx = [i for i, s in enumerate(sigs) if s == -1]
                            if buy_idx:
                                fig.add_trace(go.Scatter(
                                    x=[dates[i] for i in buy_idx],
                                    y=[normalized[i] for i in buy_idx],
                                    mode="markers", name="BUY",
                                    marker=dict(symbol="triangle-up", size=10, color="green"),
                                ))
                            if sell_idx:
                                fig.add_trace(go.Scatter(
                                    x=[dates[i] for i in sell_idx],
                                    y=[normalized[i] for i in sell_idx],
                                    mode="markers", name="SELL",
                                    marker=dict(symbol="triangle-down", size=10, color="red"),
                                ))

                    if spy_df is not None:
                        spy_closes = spy_df["close"].to_list()
                        spy_norm = [c / spy_closes[0] * 100 for c in spy_closes]
                        fig.add_trace(go.Scatter(
                            x=spy_df["date"].to_list(), y=spy_norm, mode="lines",
                            name="SPY (Buy & Hold)",
                            line=dict(color="#ff7f0e", width=2, dash="dash"),
                        ))

                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title="Date", yaxis_title="Normalized Value (base 100)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if len(ticker_results) == 1:
                        st.subheader("Price & SMA Chart")
                        r = ticker_results[0]
                        dates = r["prices_df"]["date"].to_list()
                        sigs = r["signals"]
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=dates, y=r["prices_df"]["close"].to_list(),
                            name="Close", line=dict(color="gray", width=1),
                        ))
                        fig2.add_trace(go.Scatter(
                            x=dates, y=sigs["fast_sma"].to_list(),
                            name=f"Fast SMA ({fast_window})", line=dict(color="blue"),
                        ))
                        fig2.add_trace(go.Scatter(
                            x=dates, y=sigs["slow_sma"].to_list(),
                            name=f"Slow SMA ({slow_window})", line=dict(color="orange"),
                        ))
                        fig2.update_layout(
                            template="plotly_dark",
                            height=400, xaxis_title="Date", yaxis_title="Price ($)",
                            margin=dict(l=0, r=0, t=10, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                # ── Tab 3: Trades ────────────────────────────────────────────────
                with tab_trades:
                    for r in ticker_results:
                        trades = r["result"]["trades"]
                        dates = r["prices_df"]["date"].to_list()
                        if len(ticker_results) > 1:
                            st.subheader(r["ticker"])
                        if trades.is_empty():
                            st.info("No completed trades in the selected period.")
                        else:
                            display_trades = trades.with_columns([
                                pl.col("entry_price").round(2),
                                pl.col("exit_price").round(2),
                                pl.col("pnl").round(2),
                            ])
                            entry_dates = [
                                dates[i] if 0 <= i < len(dates) else None
                                for i in display_trades["entry_date"].to_list()
                            ]
                            exit_dates = [
                                dates[i] if 0 <= i < len(dates) else None
                                for i in display_trades["exit_date"].to_list()
                            ]
                            display_df = display_trades.with_columns([
                                pl.Series("entry_date", entry_dates),
                                pl.Series("exit_date", exit_dates),
                            ])
                            st.dataframe(display_df, use_container_width=True)
                            st.caption(
                                f"Total trades: {trades.height} | "
                                f"Total PnL: ${trades['pnl'].sum():,.2f} | "
                                f"Win rate: {r['metrics']['win_rate']:.1%}"
                            )
                        if len(ticker_results) > 1:
                            st.divider()

                # ── Tab 4: Advanced (single ticker only) ────────────────────────
                with tab_advanced:
                    if len(ticker_results) > 1:
                        st.info("Advanced analytics are available for single-ticker backtests only.")
                    else:
                        r0 = ticker_results[0]
                        equity_s = r0["result"]["equity"]
                        trades_df = r0["result"]["trades"]
                        prices_dates = pl.Series(
                            "date",
                            r0["prices_df"]["date"].to_list(),
                            dtype=pl.Date,
                        )

                        st.subheader("Monthly Return Table")
                        try:
                            monthly_df = monthly_returns(equity_s, prices_dates)
                            display_monthly = monthly_df.with_columns(
                                (pl.col("return_pct") * 100).round(2).alias("return_%")
                            ).select(["year", "month", "return_%"])
                            st.dataframe(display_monthly, use_container_width=True)
                        except Exception as exc:
                            st.info(f"Monthly returns unavailable: {exc}")

                        st.divider()

                        st.subheader("Rolling 63-Day Metrics")
                        try:
                            daily_ret = equity_s.pct_change().drop_nulls()
                            r_sharpe = rolling_sharpe(daily_ret, window=63)
                            r_vol = rolling_volatility(daily_ret, window=63)
                            chart_dates = prices_dates.slice(1).to_list()

                            fig_roll = go.Figure()
                            fig_roll.add_trace(go.Scatter(
                                x=chart_dates, y=r_sharpe.to_list(),
                                name="Rolling Sharpe", line=dict(color="#2196F3"),
                            ))
                            fig_roll.add_trace(go.Scatter(
                                x=chart_dates, y=r_vol.to_list(),
                                name="Rolling Volatility", line=dict(color="#FF5722"),
                                yaxis="y2",
                            ))
                            fig_roll.update_layout(
                                template="plotly_dark",
                                height=350,
                                yaxis=dict(title="Sharpe Ratio"),
                                yaxis2=dict(title="Volatility", overlaying="y", side="right"),
                                legend=dict(orientation="h", y=1.1),
                                margin=dict(l=0, r=0, t=30, b=0),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_roll, use_container_width=True)
                        except Exception as exc:
                            st.info(f"Rolling metrics unavailable: {exc}")

                        st.divider()

                        st.subheader("Holding Period Statistics (days)")
                        if trades_df.is_empty():
                            st.info("No trades to compute holding periods.")
                        else:
                            hp = holding_period_stats(trades_df)
                            hc1, hc2, hc3, hc4 = st.columns(4)
                            hc1.metric("Avg Days", f"{hp['mean']:.1f}" if hp["mean"] is not None else "—", help="Mean number of calendar days a position was held.")
                            hc2.metric("Median Days", f"{hp['median']:.1f}" if hp["median"] is not None else "—", help="Median holding duration — less sensitive to outlier trades.")
                            hc3.metric("Min Days", str(hp["min"]) if hp["min"] is not None else "—", help="Shortest holding period among all completed trades.")
                            hc4.metric("Max Days", str(hp["max"]) if hp["max"] is not None else "—", help="Longest holding period among all completed trades.")

# ── Factor Analysis ───────────────────────────────────────────────────────────
else:
    if not run_factor_btn:
        st.info("Configure parameters in the sidebar and click **Run Factor Analysis** to start.")
    else:
        factor_tickers = parse_tickers(factor_raw_tickers)
        top_n    = int(factor_top_n)
        bottom_n = int(factor_bottom_n)

        if not factor_tickers:
            st.error("Please enter at least one ticker symbol.")
        elif top_n + bottom_n > len(factor_tickers):
            st.error(
                f"Long {top_n} + Short {bottom_n} = {top_n + bottom_n} "
                f"exceeds the universe size ({len(factor_tickers)}). "
                "Reduce top_n / bottom_n or add more tickers."
            )
        else:
            prices_dict: dict[str, pl.DataFrame] = {}
            failed_f: list[str] = []
            with st.spinner(f"Downloading data for {len(factor_tickers)} tickers…"):
                for tkr in factor_tickers:
                    try:
                        prices_dict[tkr] = cached_fetch_prices(tkr, str(factor_start), str(factor_end))
                    except Exception as exc:
                        logger.error("Factor: failed to fetch %s: %s", tkr, exc)
                        failed_f.append(tkr)

            for bad in failed_f:
                st.warning(f"⚠️ Could not fetch data for **{bad}** — skipped.")

            if len(prices_dict) < top_n + bottom_n:
                st.error("Not enough valid tickers to run the long-short backtest.")
            else:
                ls_result = None
                with st.spinner("Running long-short backtest…"):
                    try:
                        ls_result = run_long_short_backtest(
                            prices_dict=prices_dict,
                            top_n=top_n,
                            bottom_n=bottom_n,
                            lookback=factor_lookback,
                            initial_capital=10_000.0,
                            commission_rate=factor_commission / 100.0,
                            spread_bps=factor_spread,
                        )
                    except ValueError as exc:
                        st.error(f"Long-short backtest failed: {exc}")

                if ls_result is not None:
                    spy_factor_df = None
                    spy_returns_raw = None
                    with st.spinner("Fetching SPY as CAPM benchmark…"):
                        try:
                            spy_factor_df = cached_fetch_prices("SPY", str(factor_start), str(factor_end))
                            spy_returns_raw = spy_factor_df["close"].pct_change().drop_nulls()
                        except Exception:
                            pass

                    # ── Portfolio performance metrics ─────────────────────────
                    st.subheader("📊 Portfolio Performance")
                    ls_equity_s = ls_result["equity"]
                    ls_daily_ret = ls_result["daily_returns"]
                    fa1, fa2, fa3, fa4, fa5 = st.columns(5)
                    try:
                        fa1.metric("CAGR", f"{calculate_cagr(ls_equity_s):.1%}", help="Compound Annual Growth Rate: the smoothed annual return over the full period.")
                    except Exception:
                        fa1.metric("CAGR", "—", help="Compound Annual Growth Rate: the smoothed annual return over the full period.")
                    try:
                        fa2.metric("Ann. Volatility", f"{calculate_annualized_volatility(ls_daily_ret):.1%}" if len(ls_daily_ret) > 0 else "—", help="Annualized standard deviation of daily returns — a measure of risk.")
                    except Exception:
                        fa2.metric("Ann. Volatility", "—", help="Annualized standard deviation of daily returns — a measure of risk.")
                    try:
                        fa3.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(ls_daily_ret):.2f}" if len(ls_daily_ret) > 0 else "—", help="Risk-adjusted return: (portfolio return − risk-free rate) ÷ volatility. Higher is better.")
                    except Exception:
                        fa3.metric("Sharpe Ratio", "—", help="Risk-adjusted return: (portfolio return − risk-free rate) ÷ volatility. Higher is better.")
                    try:
                        fa4.metric("Max Drawdown", f"{calculate_max_drawdown(ls_equity_s):.1%}", help="Largest peak-to-trough decline in portfolio value. Measures worst-case loss.")
                    except Exception:
                        fa4.metric("Max Drawdown", "—", help="Largest peak-to-trough decline in portfolio value. Measures worst-case loss.")
                    fa5.metric("Rebalances", str(ls_result["monthly_holdings"].height), help="Number of monthly portfolio rebalancing events during the backtest.")

                    st.divider()

                    # ── CAPM metrics ──────────────────────────────────────────
                    st.subheader("📐 CAPM Regression")
                    port_returns = ls_result["daily_returns"]
                    if spy_returns_raw is not None and len(port_returns) >= 3:
                        min_len = min(len(port_returns), len(spy_returns_raw))
                        try:
                            capm = run_capm_regression(
                                port_returns[-min_len:],
                                spy_returns_raw[-min_len:],
                            )
                            ca, cb, cc, cd, ce = st.columns(5)
                            ca.metric("Ann. Alpha", f"{capm['alpha']:.2%}", help="Annualized excess return above what CAPM predicts. Positive = outperformed the market.")
                            cb.metric("Beta", f"{capm['beta']:.3f}", help="Sensitivity to market moves. β > 1: more volatile than market; β < 1: less volatile.")
                            cc.metric("t(Alpha)", f"{capm['t_alpha']:.2f}", help="t-statistic for alpha. Values beyond ±2 suggest statistical significance.")
                            cd.metric("t(Beta)", f"{capm['t_beta']:.2f}", help="t-statistic for beta. Values beyond ±2 suggest statistical significance.")
                            ce.metric("R²", f"{capm['r_squared']:.3f}", help="Proportion of portfolio return variance explained by market returns (0–1).")
                        except ValueError as exc:
                            st.warning(f"CAPM regression failed: {exc}")
                    else:
                        st.warning("Insufficient data or SPY download failed — CAPM metrics unavailable.")

                    # ── Equity curve ──────────────────────────────────────────
                    st.subheader("📉 Long-Short Portfolio Equity Curve")
                    ls_equity = ls_result["equity"].to_list()
                    ls_dates  = ls_result["dates"].to_list()
                    fig_ls = go.Figure()

                    if ls_equity:
                        first_val = ls_equity[0] if ls_equity[0] != 0 else 1.0
                        ls_norm = [v / first_val * 100 for v in ls_equity]
                        fig_ls.add_trace(go.Scatter(
                            x=ls_dates, y=ls_norm, mode="lines",
                            name="Long-Short Portfolio",
                            line=dict(color="#1f77b4", width=2),
                        ))

                    if spy_factor_df is not None:
                        spy_closes = spy_factor_df["close"].to_list()
                        spy_norm_f = [c / spy_closes[0] * 100 for c in spy_closes]
                        fig_ls.add_trace(go.Scatter(
                            x=spy_factor_df["date"].to_list(), y=spy_norm_f,
                            mode="lines", name="SPY (Buy & Hold)",
                            line=dict(color="#ff7f0e", width=2, dash="dash"),
                        ))

                    fig_ls.update_layout(
                        template="plotly_dark",
                        xaxis_title="Date", yaxis_title="Normalized Value (base 100)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_ls, use_container_width=True)

                    # ── Monthly holdings ──────────────────────────────────────
                    st.subheader("📅 Monthly Holdings")
                    holdings = ls_result["monthly_holdings"]
                    if holdings.is_empty():
                        st.info("No rebalance records found (period may be shorter than one month).")
                    else:
                        st.dataframe(holdings, use_container_width=True)
