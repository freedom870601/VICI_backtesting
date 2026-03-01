# CLAUDE.md — US Stock Backtesting System

This file defines the project conventions, workflow rules, and AI collaboration guidelines for this backtesting system. Claude Code should always read this file before starting any task.

---

## 🎯 Project Overview

A minimal but verifiable US stock backtesting system built with an AI-only workflow (Claude Code). Deployed publicly on Zeabur.

**Key deliverables:**
- Backtesting engine with SMA crossover strategy
- Performance metrics: CAGR, Annualized Volatility, Sharpe Ratio, Max Drawdown, Win Rate
- Interactive Streamlit dashboard (equity curve, trade log, performance summary)
- Public URL deployment via Zeabur: **https://vici-backtesting.zeabur.app**

---

## 🌐 Deployment Info

- **Platform**: Zeabur
- **Public URL**: https://vici-backtesting.zeabur.app
- **GitHub**: https://github.com/freedom870601/VICI_backtesting
- **Build**: Docker (`python:3.11-slim` + uv)
- **Key fix**: `--server.enableCORS false --server.enableXsrfProtection false` required for Zeabur reverse proxy
- **uv**: Installed at `~/.local/bin/uv`; run with `~/.local/bin/uv run pytest` if not in PATH
- **Python**: Pinned to 3.11 via `.python-version` (resolves conda version conflict)

---

## 🏗️ Tech Stack

- **Data**: `yfinance` (public, no API key required)
- **Data processing**: `polars` (prefer over pandas for performance)
- **Backtesting**: custom engine (no black-box frameworks)
- **Visualization**: `plotly` + `streamlit`
- **Testing**: `pytest` + `pytest-cov`
- **Package management**: `uv` (prefer over pip)
- **Deployment**: Zeabur (Docker or Python buildpack)

---

## 📁 Project Structure

```
.
├── CLAUDE.md                  # This file
├── README.md                  # Setup, assumptions, AI workflow doc
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Dev/test dependencies
├── .gitignore
├── Dockerfile                 # For Zeabur deployment
├── app.py                     # Streamlit entrypoint
├── backtest/
│   ├── __init__.py
│   ├── data.py                # yfinance data fetching
│   ├── engine.py              # Core backtesting loop
│   ├── strategy.py            # SMA crossover strategy
│   ├── metrics.py             # CAGR, Sharpe, MDD, etc.
│   └── factor.py              # Momentum scores, CAPM regression, long-short backtest
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures
│   ├── test_data.py
│   ├── test_engine.py
│   ├── test_metrics.py
│   ├── test_strategy.py
│   └── test_factor.py
└── .claude/
    └── skills/                # Project-level Claude Code skills
```

---

## 🧪 Testing Rules

- **Use pytest** for all tests
- **TDD first**: write a failing test before implementing any function
- Follow the **Red → Green → Refactor** cycle strictly
- Every metric calculation function MUST have unit tests with known expected values
- Test coverage target: **≥ 80%**
- Run tests before every commit:

```bash
uv run pytest -v
uv run pytest --cov=backtest --cov-report=term-missing
```

**Critical test cases to cover:**
- `test_cagr`: known return series → expected CAGR value
- `test_sharpe`: known returns + risk-free rate → expected Sharpe
- `test_max_drawdown`: known equity curve → expected MDD %
- `test_strategy_signals`: SMA crossover produces correct buy/sell signals
- `test_empty_data`: graceful handling of empty or single-row data

---

## 📝 Git Workflow

Maintain a **meaningful commit history** that documents the development process:

**Commit Message Format (Conventional Commits):**
```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat`: New feature (e.g., `feat(metrics): add Sharpe ratio calculation`)
- `fix`: Bug fix (e.g., `fix(engine): handle empty price data`)
- `test`: Adding or updating tests (e.g., `test(metrics): add CAGR edge cases`)
- `docs`: Documentation changes (e.g., `docs: update README with setup instructions`)
- `refactor`: Code refactoring without functional change
- `chore`: Build, config, or maintenance tasks

**Commit Frequency:**
- Make **small, atomic commits** — each commit should represent one logical change
- Commit after each passing test in TDD cycle
- Never commit broken code to main branch

**Pre-commit Checklist:**
```bash
uv run pytest -v              # All tests pass
uv run pytest --cov=backtest  # Coverage meets target
```

---

## 🐍 Python Coding Standards

- **Type hints** on all function signatures
- **Docstrings** on all public functions (one-liner minimum)
- Use `polars` over `pandas` where possible
- Use `logging` module, not `print()`
- Handle edge cases explicitly: empty data, single trade, zero volatility
- Preferred naming: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants

```python
# Good example
def calculate_sharpe_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio from daily return series."""
    ...
```

---

## 🚀 Deployment

- Target platform: **Zeabur**
- App entrypoint: `streamlit run app.py --server.port 8080 --server.address 0.0.0.0`
- All config via environment variables (no hardcoded values)
- `Dockerfile` must be self-contained and reproducible

---

## 🤖 AI Workflow Notes (for README documentation)

This project is built using an **AI-only workflow** with Claude Code. The README must document:

**Required README Sections:**
1. **How to Run Locally**
   - Prerequisites (Python version, uv installation)
   - Install dependencies: `uv sync` or `uv pip install -r requirements.txt`
   - Run tests: `uv run pytest -v`
   - Start app: `uv run streamlit run app.py`

2. **Key Assumptions**
   - Data source limitations
   - Strategy constraints
   - Calculation methodology

3. **AI Workflow Documentation**
   - Example prompts used with Claude Code
   - Skills invoked (e.g., `test-driven-development`, `python-testing-patterns`)
   - Manual corrections or iterations needed

**Documenting AI Interactions:**
When Claude Code is used, log:
- The prompt or instruction given
- Which skill was active (if any)
- Output generated and any manual adjustments
- Commit message referencing the AI-assisted change

---

## ⚠️ Key Assumptions

- Data source: `yfinance` only (public, reproducible)
- No transaction costs or slippage by default (configurable as bonus feature)
- Strategy: SMA crossover (fast/slow window configurable via UI)
- Benchmark: Buy & Hold SPY for comparison
- Time zone: all prices in US market local time (NYSE)
- Minimum backtest period: 1 year of daily data

---

## ✨ Bonus Features (Optional Enhancements)

The core deliverable is intentionally **minimal**. These are optional enhancements:

**Transaction Costs:**
- Configurable commission per trade (flat fee or percentage)
- Slippage simulation (fixed or volatility-based)

**Strategy Extensions:**
- Multiple strategy support (RSI, MACD, Bollinger Bands)
- Strategy parameter optimization (grid search)
- Walk-forward analysis

**Position Sizing:**
- Fixed fractional sizing
- Volatility-adjusted position sizing (ATR-based)
- Kelly criterion calculator

**Additional Benchmarks:**
- Compare against multiple indices (QQQ, IWM)
- Risk-adjusted benchmark comparison

**Enhanced Reporting:**
- Monthly/yearly return breakdown
- Rolling metrics visualization
- Trade analytics (avg win/loss, holding period)

---

## 📋 Scope Note

This specification represents the **full feature set**. For MVP delivery, prioritize:

1. ✅ Core backtesting engine with SMA crossover
2. ✅ Five key metrics (CAGR, Volatility, Sharpe, Max Drawdown, Win Rate)
3. ✅ Streamlit dashboard with equity curve
4. ✅ Test coverage ≥ 80%
5. ✅ Public deployment URL

Bonus features can be added incrementally after core functionality is verified.
