"""tests/test_phase2.py — Unit tests for Phase 2 features.

Tests are network-free: all price data is synthesised in-memory.
Run with:
    PYTHONPATH=src python -m pytest tests/test_phase2.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure src/ is on the path when running directly or via pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oilenergy.pipeline import (
    PriceRow,
    build_samples,
)
from oilenergy.correlation_analysis import (
    pearson_correlation as corr_pearson,
    _log_returns,
)
from oilenergy.backtest import run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(prices: list[float], start_year: int = 2000) -> list[PriceRow]:
    """Generate PriceRow list from a prices list with synthetic YYYY-MM-DD dates."""
    rows: list[PriceRow] = []
    day = 1
    month = 1
    year = start_year
    for p in prices:
        rows.append(PriceRow(date=f"{year:04d}-{month:02d}-{day:02d}", price=p))
        day += 1
        if day > 28:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return rows


def _linear_prices(n: int, start: float = 50.0, slope: float = 0.1) -> list[float]:
    return [start + slope * i for i in range(n)]


def _sine_prices(n: int, amplitude: float = 10.0, base: float = 50.0) -> list[float]:
    return [base + amplitude * math.sin(2 * math.pi * i / 252) for i in range(n)]


# ---------------------------------------------------------------------------
# Phase 1 regression guard: build_samples with horizon=1 (default)
# ---------------------------------------------------------------------------

def test_build_samples_default_horizon_produces_correct_targets():
    """Horizon=1 must yield target = next row's price (Phase 1 backward compatibility)."""
    prices = list(range(1, 60))  # 1..59
    rows = _make_rows(prices)
    samples, _ = build_samples(rows, horizon=1)
    # For each sample at index i, target should be rows[i+1].price
    for s in samples:
        date_idx = next(j for j, r in enumerate(rows) if r.date == s.date)
        assert s.target_price == rows[date_idx + 1].price


def test_build_samples_horizon_5_produces_correct_targets():
    """Horizon=5 must yield target = price 5 rows ahead."""
    prices = list(range(1, 80))
    rows = _make_rows(prices)
    samples, _ = build_samples(rows, horizon=5)
    for s in samples:
        date_idx = next(j for j, r in enumerate(rows) if r.date == s.date)
        assert s.target_price == rows[date_idx + 5].price


def test_build_samples_horizon_validation():
    """horizon < 1 must raise ValueError."""
    rows = _make_rows(list(range(1, 50)))
    try:
        build_samples(rows, horizon=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Correlation bug fix: log-return based Pearson
# ---------------------------------------------------------------------------

def test_log_returns_basic():
    """Log returns of a doubling price series should all equal log(2)."""
    prices = [1.0, 2.0, 4.0, 8.0]
    returns = _log_returns(prices)
    assert len(returns) == 3
    for r in returns:
        assert abs(r - math.log(2)) < 1e-9


def test_log_returns_flat_series():
    """Flat price series → all log returns = 0."""
    prices = [10.0] * 10
    returns = _log_returns(prices)
    assert all(abs(r) < 1e-12 for r in returns)


def test_pearson_identical_series_is_one():
    """Two identical series must yield r=1.0."""
    prices = _linear_prices(100)
    r = corr_pearson(prices, prices)
    assert abs(r - 1.0) < 1e-9


def test_pearson_two_trending_series_not_spuriously_one():
    """Brent-like (oil, $20→$120) and gas-like ($1→$10) series must NOT give r=1.

    Phase 1 bug: raw-level Pearson gave 1.0 for any two upward-trending series.
    Log-return based Pearson correctly reflects the actual co-movement.
    """
    n = 500
    oil = [20.0 + 0.2 * i + (i % 30) * 0.5 for i in range(n)]   # trending oil
    gas = [1.0 + 0.015 * i - (i % 15) * 0.2 for i in range(n)]  # different gas trend
    r = corr_pearson(oil, gas)
    # Should NOT be 1.0 (was the Phase 1 bug)
    assert r < 0.99, f"Expected r < 0.99 for dissimilar series, got {r}"


def test_pearson_short_series_returns_zero():
    """Series shorter than 3 pairs must return 0.0 gracefully."""
    assert corr_pearson([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_pearson_negatively_correlated():
    """A series that rises when the other falls should give r < 0.

    We use two sine waves exactly 180° out of phase so log returns are
    consistently in opposite directions.
    """
    n = 400
    base = 50.0
    amp = 20.0
    x = [base + amp * math.sin(2 * math.pi * i / 100) for i in range(n)]
    y = [base - amp * math.sin(2 * math.pi * i / 100) for i in range(n)]
    r = corr_pearson(x, y)
    assert r < -0.5, f"Expected negative correlation for anti-phase sine waves, got {r}"


# ---------------------------------------------------------------------------
# Phase 2: multi-horizon forecasting
# ---------------------------------------------------------------------------

def test_build_samples_fewer_samples_with_larger_horizon():
    """Larger horizon → fewer valid samples (rows at end can't be labelled)."""
    prices = list(range(1, 150))
    rows = _make_rows(prices)
    s1, _ = build_samples(rows, horizon=1)
    s21, _ = build_samples(rows, horizon=21)
    assert len(s1) > len(s21)
    # Difference should be exactly horizon - 1
    assert len(s1) - len(s21) == 20


# ---------------------------------------------------------------------------
# Phase 2: walk-forward backtest
# ---------------------------------------------------------------------------

def _make_sufficient_rows(n: int = 700) -> list[PriceRow]:
    """Create a synthetic price series long enough for backtesting."""
    prices = _sine_prices(n, amplitude=10.0, base=80.0)
    return _make_rows(prices)


def test_run_backtest_returns_expected_keys():
    rows = _make_sufficient_rows()
    result = run_backtest(rows, min_train=200, step=50, horizon=1)
    assert "steps" in result
    assert "metrics" in result
    assert "parameters" in result
    assert "feature_names" in result


def test_run_backtest_metrics_are_finite():
    rows = _make_sufficient_rows()
    result = run_backtest(rows, min_train=200, step=50, horizon=1)
    m = result["metrics"]
    assert math.isfinite(m["mae"])
    assert math.isfinite(m["rmse"])
    assert 0.0 <= m["directional_accuracy"] <= 1.0
    assert m["n_steps"] > 0


def test_run_backtest_steps_are_chronological():
    rows = _make_sufficient_rows()
    result = run_backtest(rows, min_train=200, step=50, horizon=1)
    dates = [s["test_date"] for s in result["steps"]]
    assert dates == sorted(dates), "Backtest steps must be in chronological order"


def test_run_backtest_horizon_5():
    rows = _make_sufficient_rows(800)
    result = run_backtest(rows, min_train=200, step=50, horizon=5)
    assert result["parameters"]["horizon"] == 5
    assert result["metrics"]["n_steps"] > 0


def test_run_backtest_too_short_raises():
    rows = _make_rows(list(range(1, 30)))
    try:
        run_backtest(rows, min_train=500)
        assert False, "Expected ValueError for too-short series"
    except ValueError:
        pass


def test_run_backtest_with_seasonality():
    rows = _make_sufficient_rows()
    result = run_backtest(rows, min_train=200, step=50, horizon=1, use_seasonality=True)
    assert "month" in result["feature_names"]
    assert result["metrics"]["n_steps"] > 0


# ---------------------------------------------------------------------------
# Phase 2: commodities registry includes dubai_crude
# ---------------------------------------------------------------------------

def test_dubai_crude_in_commodities():
    from oilenergy.commodities import COMMODITIES
    assert "dubai_crude" in COMMODITIES
    cfg = COMMODITIES["dubai_crude"]
    assert cfg["type"] == "oil"
    assert cfg["region"] == "middle_east"
    assert cfg["fred_series"] == "POILDUBUSDM"
