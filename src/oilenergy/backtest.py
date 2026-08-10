"""backtest.py — Walk-forward (rolling-window) backtesting for the OilEnergy pipeline.

Phase 2 extension. Provides a rigorous out-of-sample evaluation by training the
model repeatedly on a sliding training window and recording one-step-ahead
predictions for each test step.  Unlike a single 80/20 split this method:

  * Prevents look-ahead bias by never training on future data.
  * Produces a time-series of prediction errors that reveals how model quality
    evolves over different market regimes.
  * Is directly comparable to professional commodity desk backtesting practice.

Usage (CLI):
    python scripts/train_model.py --commodity brent --backtest

Usage (API):
    from oilenergy.backtest import run_backtest
    result = run_backtest(rows, min_train=500, step=20, horizon=1)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any

from .pipeline import (
    PriceRow,
    build_features,
    build_seasonality_features,
    fit_ridge_regression,
    predict,
    round_float,
    Sample,
    BASE_FEATURE_NAMES,
    SEASONALITY_FEATURE_NAMES,
)


@dataclass
class BacktestStep:
    """Single walk-forward step: one train window → one out-of-sample prediction."""

    step: int
    train_start_date: str
    train_end_date: str
    test_date: str
    current_price: float
    actual_next_price: float
    predicted_next_price: float
    absolute_error: float
    predicted_direction_up: bool
    actual_direction_up: bool


def _build_samples_slice(
    rows: list[PriceRow],
    start: int,
    end: int,
    use_seasonality: bool,
    horizon: int,
) -> list[Sample]:
    """Build samples from rows[start:end+horizon] with the given horizon."""
    samples: list[Sample] = []
    for index in range(start, end):
        if index < 10 or index + horizon >= len(rows):
            continue
        current_price = rows[index].price
        base_feats = build_features(rows, index)
        combined = list(base_feats)
        if use_seasonality:
            combined = combined + build_seasonality_features(rows[index].date)
        samples.append(
            Sample(
                date=rows[index].date,
                current_price=current_price,
                target_price=rows[index + horizon].price,
                features=combined,
            )
        )
    return samples


def run_backtest(
    rows: list[PriceRow],
    min_train: int = 500,
    step: int = 20,
    horizon: int = 1,
    use_seasonality: bool = False,
    alpha: float = 1.0,
    max_train_window: int = 2000,
) -> dict[str, Any]:
    """Run walk-forward backtesting on a price series.

    Each iteration:
      1. Trains a ridge regression model on the most recent ``max_train_window``
         observations before ``train_end``.
      2. Predicts the price ``horizon`` steps ahead for rows[train_end].
      3. Advances ``train_end`` by ``step`` and repeats until the series ends.

    Args:
        rows:              Ordered list of price observations.
        min_train:         Minimum number of rows required before the first test step.
        step:              Number of observations to advance between test steps.
        horizon:           Forecast horizon in trading sessions.
        use_seasonality:   Include calendar features in the model.
        alpha:             Ridge regularisation strength.
        max_train_window:  Maximum number of most-recent rows to include in each
                           training window (default 2000 ≈ 8 years of daily data).
                           Capping the window keeps each training step O(1) in
                           elapsed time as the series grows.

    Returns:
        dict with keys:
          ``steps``         — list of BacktestStep dicts
          ``metrics``       — aggregate MAE, RMSE, directional accuracy
          ``parameters``    — run parameters for audit
          ``feature_names`` — feature names used
    """
    if len(rows) < min_train + horizon + 10:
        raise ValueError(
            f"Not enough rows ({len(rows)}) for walk-forward backtest "
            f"(need at least {min_train + horizon + 10})."
        )

    feature_names = list(BASE_FEATURE_NAMES)
    if use_seasonality:
        feature_names = feature_names + list(SEASONALITY_FEATURE_NAMES)

    backtest_steps: list[BacktestStep] = []
    train_end = min_train + 10  # leave room for lag features (index >= 10)

    step_number = 0
    while train_end + horizon < len(rows):
        # Cap the training window to keep each step O(1) in time
        window_start = max(10, train_end - max_train_window)
        train_samples = _build_samples_slice(rows, window_start, train_end, use_seasonality, horizon)
        if len(train_samples) < 20:
            train_end += step
            continue

        try:
            weights = fit_ridge_regression(train_samples, alpha=alpha)
        except ValueError:
            train_end += step
            continue

        test_index = train_end
        if test_index >= len(rows) or test_index + horizon >= len(rows):
            break

        current_price = rows[test_index].price
        base_feats = build_features(rows, test_index)
        combined = list(base_feats)
        if use_seasonality:
            combined = combined + build_seasonality_features(rows[test_index].date)

        predicted_price = predict(weights, combined)
        actual_price = rows[test_index + horizon].price
        abs_error = abs(predicted_price - actual_price)
        pred_up = bool(predicted_price >= current_price)
        actual_up = bool(actual_price >= current_price)

        backtest_steps.append(
            BacktestStep(
                step=step_number,
                train_start_date=rows[window_start].date,
                train_end_date=rows[train_end - 1].date,
                test_date=rows[test_index].date,
                current_price=round_float(current_price, 4),
                actual_next_price=round_float(actual_price, 4),
                predicted_next_price=round_float(predicted_price, 4),
                absolute_error=round_float(abs_error, 4),
                predicted_direction_up=pred_up,
                actual_direction_up=actual_up,
            )
        )
        step_number += 1
        train_end += step

    if not backtest_steps:
        raise ValueError("Walk-forward backtest produced no steps. Check min_train and series length.")

    abs_errors = [s.absolute_error for s in backtest_steps]
    direction_hits = sum(1 for s in backtest_steps if s.predicted_direction_up == s.actual_direction_up)
    mae = round_float(sum(abs_errors) / len(abs_errors), 6)
    rmse = round_float(math.sqrt(sum(e**2 for e in abs_errors) / len(abs_errors)), 6)
    dir_acc = round_float(direction_hits / len(backtest_steps), 6)

    return {
        "steps": [asdict(s) for s in backtest_steps],
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": dir_acc,
            "n_steps": len(backtest_steps),
        },
        "parameters": {
            "min_train": min_train,
            "step": step,
            "horizon": horizon,
            "use_seasonality": use_seasonality,
            "alpha": alpha,
            "max_train_window": max_train_window,
        },
        "feature_names": feature_names,
    }
