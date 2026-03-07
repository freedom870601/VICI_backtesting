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
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    drawdown_series,
    holding_period_stats,
    monthly_returns,
)
from backtest.strategy import generate_sma_signals

logging.basicConfig(level=logging.INFO)

UNIVERSES: dict[str, list[str]] = {
    "S&P 100": [
        "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","LLY","AVGO","TSLA",
        "WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD","PG","JNJ","ABBV",
        "BAC","NFLX","KO","CRM","CVX","MRK","AMD","PEP","TMO","ACN","LIN","MCD",
        "CSCO","IBM","GE","PM","NOW","CAT","TXN","ISRG","INTU","AMGN","GS","SPGI",
        "HON","BLK","AXP","VZ","MS","DE","BKNG","GILD","SYK","T","PLD","RTX","ADP",
        "VRTX","ADI","MDLZ","SCHW","TJX","MMC","CB","REGN","ETN","MO","CI","PGR",
        "CME","BMY","LRCX","SO","EOG","BDX","ITW","ZTS","AON","SLB","DUK","WM",
        "USB","MCO","CL","PH","APH","TT","EMR","HUM","NOC","GD","EQIX","NSC",
        "APD","OKE",
    ],
    "NASDAQ-100": [
        "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST",
        "NFLX","AMD","TMUS","ASML","CSCO","AZN","INTU","PEP","ADBE","QCOM","AMAT",
        "TXN","AMGN","ISRG","MU","HON","BKNG","VRTX","REGN","LRCX","PANW","ADI",
        "SBUX","MDLZ","GILD","ADP","MELI","CTAS","SNPS","CDNS","KLAC","MAR","FTNT",
        "ABNB","ORLY","CHTR","DASH","WDAY","MRVL","PCAR","MNST","PAYX","CRWD","KDP",
        "ROST","NXPI","CSX","AEP","EA","ODFL","FAST","EXC","GEHC","XEL","IDXX","KHC",
        "FANG","ON","CTSH","VRSK","BIIB","DDOG","ANSS","CSGP","TEAM","WBD","DLTR",
        "GFS","ILMN","TTD","ZS","ALGN","DXCM","CPRT","MRNA","OKTA","EBAY","CEG",
        "ENPH","MCHP","LULU","CDW","TTWO","ADSK","ROP","ARM","PLTR",
    ],
    "S&P 500": [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","AKAM",
        "ALK","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
        "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
        "ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET",
        "AJG","AIZ","T","ATO","ADSK","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
        "BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA",
        "BMY","AVGO","BR","BRO","BLDR","BG","CDNS","CZR","CPT","CPB",
        "COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC",
        "CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF",
        "CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA",
        "CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTLT","CRM",
        "CTRA","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DECK","DE","DAL",
        "DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DTE",
        "DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH",
        "ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
        "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
        "FIS","FITB","FSLR","FE","FLT","FMC","F","FTNT","FTV","FOXA","FOX",
        "BEN","FCX","GRMN","IT","GE","GEHC","GEN","GIS","GM","GPC","GILD","GPN",
        "GL","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
        "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM",
        "IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
        "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM",
        "K","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR",
        "LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT",
        "L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS",
        "MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP",
        "MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS",
        "MOS","MSI","MSCI","NDAQ","NTAP","NWSA","NWS","NEE","NKE","NEM","NFLX",
        "NWL","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE",
        "ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE",
        "PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
        "PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG",
        "REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI",
        "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SO",
        "LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS",
        "TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT",
        "TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER",
        "UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN",
        "VRSK","VZ","VRTX","VTRS","VICI","V","VST","VNO","VMC","WAB","WMT",
        "WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","WRB",
        "GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
    ],
}
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

    equity_list = equity.to_list()
    return {
        "ticker": ticker,
        "prices_df": prices_df,
        "signals": signals,
        "result": result,
        "metrics": {
            "cagr": calculate_cagr(equity),
            "vol": calculate_annualized_volatility(daily_returns) if len(daily_returns) > 0 else 0.0,
            "sharpe": calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0,
            "sortino": calculate_sortino_ratio(daily_returns) if len(daily_returns) > 0 else 0.0,
            "calmar": calculate_calmar_ratio(equity_list) if len(equity_list) >= 2 else 0.0,
            "mdd": calculate_max_drawdown(equity),
            "win_rate": calculate_win_rate(trades),
            "profit_factor": calculate_profit_factor(trades),
            "n_trades": trades.height,
        },
    }


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="VICI Backtesting", page_icon="📈", layout="wide")

# ── Global CSS: dark financial terminal theme ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Root palette ──────────────────────────────────────────────────────── */
:root {
    --bg-base:      #080c12;
    --bg-surface:   #0d1420;
    --bg-elevated:  #121a28;
    --bg-border:    #1e2d42;
    --accent-green: #00e87a;
    --accent-blue:  #3b82f6;
    --accent-red:   #f43f5e;
    --accent-gold:  #f0c040;
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --text-faint:   #334155;
}

/* ── App-wide background & font ────────────────────────────────────────── */
.stApp, .stApp > * {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* hide default streamlit header bar */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
footer { display: none !important; }


/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--bg-border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] li {
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar: slider label rows ────────────────────────────────────────── */
[data-testid="stSidebar"] .sidebar-param-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 10px 0 2px 0;
}

/* ── Inputs & Selects ──────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="select"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0,232,122,0.15) !important;
    outline: none !important;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent-green) !important;
    border-color: var(--accent-green) !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackActive"] {
    background-color: var(--accent-green) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00e87a 0%, #00c563 100%) !important;
    color: #080c12 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0,232,122,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(0,232,122,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--bg-border) !important;
    color: var(--text-muted) !important;
    box-shadow: none !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-green);
    border-radius: 10px 0 0 10px;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: var(--bg-surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--bg-border) !important;
}
[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 6px 14px !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background-color: var(--bg-elevated) !important;
    color: var(--accent-green) !important;
    font-weight: 600 !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── DataFrames / Tables ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Info / Warning / Error boxes ──────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
    font-size: 0.84rem !important;
}

/* ── Radio (mode switcher) ──────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
    border-color: var(--bg-border) !important;
    margin: 12px 0 !important;
}

/* ── Subheaders / section headings in main content ─────────────────────── */
.stMarkdown h3, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.02em !important;
    margin: 20px 0 10px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    padding: 28px 32px 20px 32px;
    background: linear-gradient(135deg, #0d1420 0%, #080c12 60%, #0a1a0e 100%);
    border: 1px solid #1e2d42;
    border-radius: 14px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
">
  <div style="
      position: absolute; top: -40px; right: -40px;
      width: 200px; height: 200px;
      background: radial-gradient(circle, rgba(0,232,122,0.06) 0%, transparent 70%);
      pointer-events: none;
  "></div>
  <div style="display: flex; align-items: baseline; gap: 14px; margin-bottom: 6px;">
    <span style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: #00e87a;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        background: rgba(0,232,122,0.08);
        border: 1px solid rgba(0,232,122,0.2);
        padding: 3px 8px;
        border-radius: 4px;
    ">VICI</span>
    <h1 style="
        margin: 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 1.65rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
    ">US Stock Backtesting System</h1>
  </div>
  <p style="
      margin: 0;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.82rem;
      color: #64748b;
      letter-spacing: 0.02em;
  "></p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: mode switcher + context-sensitive controls ──────────────────────
# ── Session state defaults ────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "fast_window": 20,      "_fast_s": 20,      "_fast_n": 20,
    "slow_window": 50,      "_slow_s": 50,      "_slow_n": 50,
    "commission_pct": 0.00, "_comm_s": 0.00,    "_comm_n": 0.00,
    "slippage_pct": 0.00,   "_slip_s": 0.00,    "_slip_n": 0.00,
    "spread_bps_val": 0.0,  "_spread_s": 0.0,   "_spread_n": 0.0,
    "factor_lookback": 63,  "_flookback_s": 63, "_flookback_n": 63,
    "factor_commission": 0.10, "_fcomm_s": 0.10, "_fcomm_n": 0.10,
    "factor_spread": 5.0,   "_fspread_s": 5.0,  "_fspread_n": 5.0,
    "factor_rebal_n_weeks": 4, "_frebal_n": 4, "_frebal_preset": "Monthly (4w)",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Sync callbacks (slider → state, number_input → state)
def _sync_fast_s():
    v = st.session_state["_fast_s"];        st.session_state["fast_window"] = v;       st.session_state["_fast_n"] = v
def _sync_fast_n():
    v = st.session_state["_fast_n"];        st.session_state["fast_window"] = v;       st.session_state["_fast_s"] = v
def _sync_slow_s():
    v = st.session_state["_slow_s"];        st.session_state["slow_window"] = v;       st.session_state["_slow_n"] = v
def _sync_slow_n():
    v = st.session_state["_slow_n"];        st.session_state["slow_window"] = v;       st.session_state["_slow_s"] = v
def _sync_comm_s():
    v = st.session_state["_comm_s"];        st.session_state["commission_pct"] = v;    st.session_state["_comm_n"] = v
def _sync_comm_n():
    v = st.session_state["_comm_n"];        st.session_state["commission_pct"] = v;    st.session_state["_comm_s"] = v
def _sync_slip_s():
    v = st.session_state["_slip_s"];        st.session_state["slippage_pct"] = v;      st.session_state["_slip_n"] = v
def _sync_slip_n():
    v = st.session_state["_slip_n"];        st.session_state["slippage_pct"] = v;      st.session_state["_slip_s"] = v
def _sync_spread_s():
    v = st.session_state["_spread_s"];      st.session_state["spread_bps_val"] = v;    st.session_state["_spread_n"] = v
def _sync_spread_n():
    v = st.session_state["_spread_n"];      st.session_state["spread_bps_val"] = v;    st.session_state["_spread_s"] = v
def _sync_flookback_s():
    v = st.session_state["_flookback_s"];   st.session_state["factor_lookback"] = v;   st.session_state["_flookback_n"] = v
def _sync_flookback_n():
    v = st.session_state["_flookback_n"];   st.session_state["factor_lookback"] = v;   st.session_state["_flookback_s"] = v
def _sync_fcomm_s():
    v = st.session_state["_fcomm_s"];       st.session_state["factor_commission"] = v; st.session_state["_fcomm_n"] = v
def _sync_fcomm_n():
    v = st.session_state["_fcomm_n"];       st.session_state["factor_commission"] = v; st.session_state["_fcomm_s"] = v
def _sync_fspread_s():
    v = st.session_state["_fspread_s"];     st.session_state["factor_spread"] = v;     st.session_state["_fspread_n"] = v
def _sync_fspread_n():
    v = st.session_state["_fspread_n"];     st.session_state["factor_spread"] = v;     st.session_state["_fspread_s"] = v

_REBAL_PRESETS = {"Weekly (1w)": 1, "Monthly (4w)": 4, "Quarterly (13w)": 13}
def _sync_frebal_preset():
    preset = st.session_state["_frebal_preset"]
    if preset in _REBAL_PRESETS:
        n = _REBAL_PRESETS[preset]
        st.session_state["factor_rebal_n_weeks"] = n
        st.session_state["_frebal_n"] = n
def _sync_frebal_n():
    st.session_state["factor_rebal_n_weeks"] = st.session_state["_frebal_n"]
    st.session_state["_frebal_preset"] = "Custom"

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
        st.markdown("**Fast SMA**", help="Fast Simple Moving Average window (days). A shorter window reacts quickly to price changes.")
        _fs_col, _fi_col = st.columns([3, 1])
        with _fs_col:
            st.slider("Fast SMA slider", min_value=5, max_value=100, step=1,
                      key="_fast_s",
                      on_change=_sync_fast_s, label_visibility="collapsed")
        with _fi_col:
            st.number_input("Fast SMA input", min_value=5, max_value=100, step=1,
                            key="_fast_n",
                            on_change=_sync_fast_n, label_visibility="collapsed")
        fast_window = st.session_state["fast_window"]
        st.markdown("**Slow SMA**", help="Slow Simple Moving Average window (days). A longer window reflects the broader trend.")
        _ss_col, _si_col = st.columns([3, 1])
        with _ss_col:
            st.slider("Slow SMA slider", min_value=10, max_value=300, step=1,
                      key="_slow_s",
                      on_change=_sync_slow_s, label_visibility="collapsed")
        with _si_col:
            st.number_input("Slow SMA input", min_value=10, max_value=300, step=1,
                            key="_slow_n",
                            on_change=_sync_slow_n, label_visibility="collapsed")
        slow_window = st.session_state["slow_window"]
        sma_valid = fast_window < slow_window
        if not sma_valid:
            st.error("⚠️ Fast SMA must be less than Slow SMA.")

        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
            value=10_000, step=1_000,
        )

        st.markdown("---")
        st.subheader("Transaction Costs")
        st.markdown("**Commission (%)**", help="Broker fee charged per trade, as a percentage of the trade value.")
        _cc_col, _ci_col = st.columns([3, 1])
        with _cc_col:
            st.slider("Commission slider", min_value=0.00, max_value=2.00, step=0.01,
                      key="_comm_s",
                      on_change=_sync_comm_s, label_visibility="collapsed")
        with _ci_col:
            st.number_input("Commission input", min_value=0.00, max_value=2.00, step=0.01,
                            format="%.2f", key="_comm_n",
                            on_change=_sync_comm_n, label_visibility="collapsed")
        commission_pct = st.session_state["commission_pct"]
        st.markdown("**Slippage (%)**", help="Estimated price impact: the difference between the expected and actual fill price.")
        _sc_col, _si2_col = st.columns([3, 1])
        with _sc_col:
            st.slider("Slippage slider", min_value=0.00, max_value=2.00, step=0.01,
                      key="_slip_s",
                      on_change=_sync_slip_s, label_visibility="collapsed")
        with _si2_col:
            st.number_input("Slippage input", min_value=0.00, max_value=2.00, step=0.01,
                            format="%.2f", key="_slip_n",
                            on_change=_sync_slip_n, label_visibility="collapsed")
        slippage_pct = st.session_state["slippage_pct"]
        st.markdown("**Bid-Ask Spread (bps)**", help="Bid-ask spread in basis points (1 bps = 0.01%). Buying costs more; selling nets less.")
        _sp_col, _spi_col = st.columns([3, 1])
        with _sp_col:
            st.slider("Spread slider", min_value=0.0, max_value=50.0, step=0.5,
                      key="_spread_s",
                      on_change=_sync_spread_s, label_visibility="collapsed")
        with _spi_col:
            st.number_input("Spread input", min_value=0.0, max_value=50.0, step=0.5,
                            format="%.1f", key="_spread_n",
                            on_change=_sync_spread_n, label_visibility="collapsed")
        spread_bps_val = st.session_state["spread_bps_val"]
        entry_price_type = st.selectbox("Entry Price", ["close", "open"], index=0, help="'close' fills at today's closing price; 'open' fills at the next bar's opening price.")

        run_btn = st.button(
            "▶ Run Backtest", type="primary", use_container_width=True,
            disabled=not sma_valid,
        )
        run_factor_btn = False

    # ── Factor Analysis sidebar ──────────────────────────────────────────────
    else:
        st.header("⚙️ Factor Analysis Parameters")

        universe_choice = st.selectbox(
            "Stock Universe",
            options=["S&P 100", "S&P 500", "NASDAQ-100", "Custom"],
            index=0,
        )
        if universe_choice == "Custom":
            factor_raw_tickers = st.text_input(
                "Custom Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM",
            )
        else:
            factor_raw_tickers = ",".join(UNIVERSES[universe_choice])
            st.caption(f"{len(UNIVERSES[universe_choice])} tickers loaded")

        fc1, fc2 = st.columns(2)
        with fc1:
            factor_start = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
        with fc2:
            factor_end   = st.date_input("End Date",   value=datetime.date(2024, 1, 1))

        st.markdown("---")
        st.subheader("Long-Short Settings")
        factor_top_n    = st.number_input("Long Top N", min_value=1, max_value=20, value=3, help="Number of highest-momentum stocks to go long (buy) each rebalance period.")
        factor_bottom_n = st.number_input("Short Bottom N", min_value=1, max_value=20, value=3, help="Number of lowest-momentum stocks to go short (sell) each rebalance period.")
        st.markdown("**Rebalance Frequency**", help="Rebalance the portfolio every N calendar weeks. Use a preset or set a custom interval.")
        st.selectbox(
            "Rebalance preset",
            options=list(_REBAL_PRESETS.keys()) + ["Custom"],
            key="_frebal_preset",
            on_change=_sync_frebal_preset,
            label_visibility="collapsed",
        )
        st.number_input(
            "Rebalance every N weeks",
            min_value=1, max_value=52, step=1,
            key="_frebal_n",
            on_change=_sync_frebal_n,
            label_visibility="collapsed",
        )
        factor_rebal_n_weeks = st.session_state["factor_rebal_n_weeks"]

        st.markdown("---")
        st.subheader("Strategy")
        st.markdown("**Momentum Lookback (days)**", help="Look-back window used to rank stocks by past return. Longer = slower-changing rankings.")
        _fl_col, _fli_col = st.columns([3, 1])
        with _fl_col:
            st.slider("Lookback slider", min_value=20, max_value=252, step=1,
                      key="_flookback_s",
                      on_change=_sync_flookback_s, label_visibility="collapsed")
        with _fli_col:
            st.number_input("Lookback input", min_value=20, max_value=252, step=1,
                            key="_flookback_n",
                            on_change=_sync_flookback_n, label_visibility="collapsed")
        factor_lookback = st.session_state["factor_lookback"]

        st.markdown("---")
        st.subheader("Transaction Costs")
        st.markdown("**Commission (%)**", help="Broker fee charged per trade, as a percentage of the trade value.")
        _fc_col, _fci_col = st.columns([3, 1])
        with _fc_col:
            st.slider("Factor commission slider", min_value=0.00, max_value=2.00, step=0.01,
                      key="_fcomm_s",
                      on_change=_sync_fcomm_s, label_visibility="collapsed")
        with _fci_col:
            st.number_input("Factor commission input", min_value=0.00, max_value=2.00, step=0.01,
                            format="%.2f", key="_fcomm_n",
                            on_change=_sync_fcomm_n, label_visibility="collapsed")
        factor_commission = st.session_state["factor_commission"]
        st.markdown("**Bid-Ask Spread (bps)**", help="Bid-ask spread in basis points (1 bps = 0.01%). Buying costs more; selling nets less.")
        _fsp_col, _fspi_col = st.columns([3, 1])
        with _fsp_col:
            st.slider("Factor spread slider", min_value=0.0, max_value=50.0, step=0.5,
                      key="_fspread_s",
                      on_change=_sync_fspread_s, label_visibility="collapsed")
        with _fspi_col:
            st.number_input("Factor spread input", min_value=0.0, max_value=50.0, step=0.5,
                            format="%.1f", key="_fspread_n",
                            on_change=_sync_fspread_n, label_visibility="collapsed")
        factor_spread = st.session_state["factor_spread"]

        run_factor_btn = st.button("▶ Run Factor Analysis", type="primary", use_container_width=True)
        run_btn = False


# ============================================================
# Main content: renders based on selected mode
# ============================================================
_subtitle = (
    "Strategy: SMA Crossover &nbsp;·&nbsp; Benchmark: Buy &amp; Hold SPY &nbsp;·&nbsp; Data: yfinance"
    if mode == "📈 Single Stock"
    else "Strategy: Long-Short Momentum &nbsp;·&nbsp; Benchmark: SPY (CAPM) &nbsp;·&nbsp; Data: yfinance"
)
st.markdown(f"""<p style="
    margin: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #64748b;
    letter-spacing: 0.02em;
">{_subtitle}</p>""", unsafe_allow_html=True)

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
                        st.markdown('<p style="font-size:0.72rem;font-weight:500;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin:0 0 8px 0;">Return Metrics</p>', unsafe_allow_html=True)
                        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                        r1c1.metric("CAGR", f"{m['cagr']:.1%}", help="Compound Annual Growth Rate: the smoothed annual return over the full period.")
                        r1c2.metric("Sharpe Ratio", f"{m['sharpe']:.2f}", help="Risk-adjusted return: (portfolio return − risk-free rate) ÷ volatility. Higher is better.")
                        r1c3.metric("Sortino Ratio", f"{m['sortino']:.2f}", help="Like Sharpe but only penalizes downside volatility. Higher is better.")
                        r1c4.metric("Calmar Ratio", f"{m['calmar']:.2f}", help="CAGR divided by Max Drawdown. Measures return per unit of drawdown risk.")
                        st.markdown('<p style="font-size:0.72rem;font-weight:500;text-transform:uppercase;letter-spacing:0.12em;color:#64748b;margin:16px 0 8px 0;">Risk &amp; Trade Metrics</p>', unsafe_allow_html=True)
                        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
                        r2c1.metric("Max Drawdown", f"{m['mdd']:.1%}", help="Largest peak-to-trough decline in portfolio value. Measures worst-case loss.")
                        r2c2.metric("Ann. Volatility", f"{m['vol']:.1%}", help="Annualized standard deviation of daily returns — a measure of risk.")
                        r2c3.metric("Profit Factor", f"{m['profit_factor']:.2f}", help="Gross profit divided by gross loss. Values > 1 indicate a net-profitable strategy.")
                        r2c4.metric("Win Rate", f"{m['win_rate']:.1%}", help="Proportion of completed trades that were profitable.")
                        r2c5.metric("# Trades", str(m['n_trades']), help="Total number of completed round-trip trades.")

                        # ── Strategy vs Benchmark comparison table ───────────────
                        if spy_df is not None and len(spy_df) >= 2:
                            st.subheader("Strategy vs Benchmark")
                            spy_close = spy_df["close"].to_list()
                            spy_returns = pl.Series([spy_close[i] / spy_close[i - 1] - 1 for i in range(1, len(spy_close))])
                            spy_equity_list = [float(initial_capital) * c / spy_close[0] for c in spy_close]
                            spy_equity_s = pl.Series("equity", spy_equity_list)
                            strat_metrics = {
                                "CAGR": f"{m['cagr']:.1%}",
                                "Sharpe": f"{m['sharpe']:.2f}",
                                "Sortino": f"{m['sortino']:.2f}",
                                "Calmar": f"{m['calmar']:.2f}",
                                "Max Drawdown": f"{m['mdd']:.1%}",
                                "Ann. Volatility": f"{m['vol']:.1%}",
                                "Profit Factor": f"{m['profit_factor']:.2f}",
                                "Win Rate": f"{m['win_rate']:.1%}",
                            }
                            spy_cagr = calculate_cagr(spy_equity_s)
                            spy_vol = calculate_annualized_volatility(spy_returns) if len(spy_returns) > 0 else 0.0
                            spy_sharpe = calculate_sharpe_ratio(spy_returns) if len(spy_returns) > 0 else 0.0
                            spy_sortino = calculate_sortino_ratio(spy_returns) if len(spy_returns) > 0 else 0.0
                            spy_calmar = calculate_calmar_ratio(spy_equity_list)
                            spy_mdd = calculate_max_drawdown(spy_equity_s)
                            spy_metrics = {
                                "CAGR": f"{spy_cagr:.1%}",
                                "Sharpe": f"{spy_sharpe:.2f}",
                                "Sortino": f"{spy_sortino:.2f}",
                                "Calmar": f"{spy_calmar:.2f}",
                                "Max Drawdown": f"{spy_mdd:.1%}",
                                "Ann. Volatility": f"{spy_vol:.1%}",
                                "Profit Factor": "—",
                                "Win Rate": "—",
                            }
                            comparison_df = pl.DataFrame({
                                "Metric": list(strat_metrics.keys()),
                                "Strategy": list(strat_metrics.values()),
                                "SPY": list(spy_metrics.values()),
                            })
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
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

                    # ── Underwater (Drawdown) Chart ───────────────────────────
                    if len(ticker_results) == 1:
                        r0 = ticker_results[0]
                        eq0 = r0["result"]["equity"].to_list()
                        dates0 = r0["prices_df"]["date"].to_list()
                        dd0 = drawdown_series(eq0)
                        fig_uw = go.Figure()
                        fig_uw.add_trace(go.Scatter(
                            x=dates0, y=dd0, fill="tozeroy",
                            name="Strategy", line=dict(color="red"),
                        ))
                        if spy_df is not None:
                            spy_close_uw = spy_df["close"].to_list()
                            spy_eq_uw = [eq0[0] * c / spy_close_uw[0] for c in spy_close_uw]
                            spy_dd_uw = drawdown_series(spy_eq_uw)
                            fig_uw.add_trace(go.Scatter(
                                x=spy_df["date"].to_list(), y=spy_dd_uw, fill="tozeroy",
                                name="SPY", line=dict(color="orange", dash="dash"),
                            ))
                        fig_uw.update_layout(
                            title="Underwater (Drawdown) Chart",
                            yaxis_title="Drawdown %", template="plotly_dark", height=300,
                            margin=dict(l=0, r=0, t=40, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_uw, use_container_width=True)

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
                            rebal_every_n_weeks=int(factor_rebal_n_weeks),
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
