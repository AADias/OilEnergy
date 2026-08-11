from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DATASET_URL = "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv"

# Base time-series features (price-only model)
BASE_FEATURE_NAMES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "mean_5",
    "mean_10",
    "momentum_5",
    "volatility_5",
]

# Seasonality feature names appended when --features includes "seasonality"
SEASONALITY_FEATURE_NAMES = [
    "month",
    "quarter",
    "day_of_week",
    "is_heating_season",  # Nov–Mar (Northern Hemisphere winter demand)
    "is_cooling_season",  # Jun–Sep (summer cooling demand peak)
]

AVAILABLE_MODELS = {
    "ridge": "Ridge regression baseline",
    "naive": "Naive persistence baseline (next price = current price)",
    "exponential_smoothing": "Exponential smoothing (optimal α fitted on training data, pure Python)",
    "xgboost": "XGBoost gradient boosting (requires: pip install xgboost)",
}

# Legacy alias — keeps existing code that imports FEATURE_NAMES working
FEATURE_NAMES = BASE_FEATURE_NAMES


@dataclass
class PriceRow:
    date: str
    price: float


@dataclass
class Sample:
    date: str
    current_price: float
    target_price: float
    features: list[float]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_for_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download_dataset(url: str, destination: Path) -> dict[str, Any]:
    ensure_directory(destination.parent)
    with urlopen(url, timeout=30) as response:
        content = response.read()
    destination.write_bytes(content)
    return {
        "downloaded_at": utc_now(),
        "sha256": sha256_for_bytes(content),
        "byte_count": len(content),
    }


def load_rows(csv_path: Path) -> list[PriceRow]:
    rows: list[PriceRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(PriceRow(date=row["Date"], price=float(row["Price"])))
    return rows


def rolling_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("rolling_mean requires at least one value.")
    return sum(values) / len(values)


def rolling_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    # Population volatility is intentional here because the feature window is the full observation window used for prediction.
    return statistics.pstdev(values)


def build_features(rows: list[PriceRow], index: int) -> list[float]:
    # These features intentionally include the current observation because the model predicts the next trading day's price.
    window_5 = [rows[index - offset].price for offset in range(0, 5)]
    window_10 = [rows[index - offset].price for offset in range(0, 10)]
    current_price = rows[index].price
    return [
        rows[index - 1].price,
        rows[index - 2].price,
        rows[index - 3].price,
        rows[index - 5].price,
        rows[index - 10].price,
        rolling_mean(window_5),
        rolling_mean(window_10),
        current_price - rows[index - 5].price,
        rolling_std(window_5),
    ]


def build_seasonality_features(date_str: str) -> list[float]:
    """Build calendar/seasonal features from a date string (YYYY-MM-DD).

    Features:
        month           — 1–12 (captures annual cycle)
        quarter         — 1–4 (broad seasonal grouping)
        day_of_week     — 0=Monday … 6=Sunday
        is_heating_season — 1 if Nov–Mar (Northern Hemisphere winter energy demand)
        is_cooling_season — 1 if Jun–Sep (summer cooling demand peak for gas/electricity)
    """
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        return [0.0] * len(SEASONALITY_FEATURE_NAMES)
    month = dt.month
    quarter = (month - 1) // 3 + 1
    day_of_week = dt.weekday()
    is_heating = 1.0 if month in {11, 12, 1, 2, 3} else 0.0
    is_cooling = 1.0 if month in {6, 7, 8, 9} else 0.0
    return [float(month), float(quarter), float(day_of_week), is_heating, is_cooling]


def _next_calendar_date(date_str: str) -> str:
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _context_feature_names(context_feature_series: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for category in context_feature_series:
        names.extend([f"{category}_lag_1", f"{category}_mean_3"])
    return names


def build_samples(
    rows: list[PriceRow],
    use_seasonality: bool = False,
    external_feature_sets: list[Any] | None = None,
    context_feature_series: dict[str, Any] | None = None,
) -> tuple[list[Sample], list[str]]:
    """Build feature samples from a price series.

    Args:
        rows: Ordered list of price observations.
        use_seasonality: If True, append calendar/seasonal features.
        external_feature_sets: Optional list of ExternalFeatureSet objects.

    Returns:
        (samples, feature_names)
    """
    from .context_features import build_context_features
    from .external_features import enrich_features

    base_names = list(BASE_FEATURE_NAMES)
    seasonal_names = list(SEASONALITY_FEATURE_NAMES) if use_seasonality else []
    # Number of features before any external series are appended
    n_pre_external = len(base_names) + len(seasonal_names)
    pre_external_names = base_names + seasonal_names

    feature_names = pre_external_names[:]
    context_series = context_feature_series or {}
    for category, series in context_series.items():
        feature_names += [f"{category}_lag_1", f"{category}_mean_3"]
    ext_sets = external_feature_sets or []
    for fs in ext_sets:
        feature_names = feature_names + list(fs.feature_names)

    samples: list[Sample] = []
    for index in range(10, len(rows) - 1):
        current_price = rows[index].price
        base_feats = build_features(rows, index)
        combined = list(base_feats)
        if use_seasonality:
            combined = combined + build_seasonality_features(rows[index].date)
        for category, series in context_series.items():
            combined += build_context_features(rows[index].date, series)
        if ext_sets:
            combined, _ = enrich_features(rows[index].date, combined, pre_external_names, ext_sets)
        samples.append(
            Sample(
                date=rows[index].date,
                current_price=current_price,
                target_price=rows[index + 1].price,
                features=combined,
            )
        )
    return samples, feature_names


def split_samples(samples: list[Sample], train_ratio: float = 0.8) -> tuple[list[Sample], list[Sample]]:
    split_index = int(len(samples) * train_ratio)
    return samples[:split_index], samples[split_index:]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = transpose(right)
    return [[sum(a * b for a, b in zip(row, column)) for column in right_t] for row in left]


def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row_index: abs(augmented[row_index][column]))
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        if abs(pivot) < 1e-12:
            raise ValueError("Unable to solve linear system because the matrix is singular.")

        for target_row in range(column + 1, size):
            factor = augmented[target_row][column] / pivot
            for target_column in range(column, size + 1):
                augmented[target_row][target_column] -= factor * augmented[column][target_column]

    solution = [0.0] * size
    for row_index in range(size - 1, -1, -1):
        total = augmented[row_index][size]
        for column in range(row_index + 1, size):
            total -= augmented[row_index][column] * solution[column]
        solution[row_index] = total / augmented[row_index][row_index]
    return solution


def fit_ridge_regression(train_samples: list[Sample], alpha: float = 1.0) -> list[float]:
    design_matrix = [[1.0] + sample.features for sample in train_samples]
    targets = [sample.target_price for sample in train_samples]
    design_matrix_t = transpose(design_matrix)
    gram_matrix = matrix_multiply(design_matrix_t, design_matrix)
    target_vector = matrix_vector_multiply(design_matrix_t, targets)

    for row_index in range(1, len(gram_matrix)):
        # Keep the intercept unregularized while applying ridge shrinkage to feature weights.
        gram_matrix[row_index][row_index] += alpha

    return solve_linear_system(gram_matrix, target_vector)


def predict_ridge(weights: list[float], features: list[float]) -> float:
    return weights[0] + sum(weight * value for weight, value in zip(weights[1:], features))


# ---------------------------------------------------------------------------
# Exponential-smoothing model (pure Python, no external dependencies)
# ---------------------------------------------------------------------------

def _ewa_predict(alpha: float, features: list[float]) -> float:
    """Exponentially weighted average using lag_1, lag_2, lag_3 features."""
    lag_1, lag_2, lag_3 = features[0], features[1], features[2]
    w1 = alpha
    w2 = alpha * (1.0 - alpha)
    w3 = alpha * (1.0 - alpha) ** 2
    total_w = w1 + w2 + w3
    return (w1 * lag_1 + w2 * lag_2 + w3 * lag_3) / total_w if total_w > 0 else lag_1


def _ewa_mae(samples: list[Sample], alpha: float) -> float:
    if not samples:
        return float("inf")
    return sum(abs(_ewa_predict(alpha, s.features) - s.target_price) for s in samples) / len(samples)


def fit_exponential_smoothing(train_samples: list[Sample]) -> dict[str, Any]:
    """Grid-search the optimal smoothing parameter α on training data."""
    best_alpha = 0.3
    best_mae = float("inf")
    for alpha_int in range(5, 96, 5):  # α ∈ {0.05, 0.10, …, 0.95}
        alpha = alpha_int / 100.0
        mae = _ewa_mae(train_samples, alpha)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return {
        "model_name": "exponential_smoothing",
        "alpha": best_alpha,
        "train_mae_at_best_alpha": round(best_mae, 6),
    }


# ---------------------------------------------------------------------------
# XGBoost model (optional dependency)
# ---------------------------------------------------------------------------

def _check_xgboost() -> Any:
    try:
        import xgboost as xgb  # type: ignore[import]
        return xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the 'xgboost' model. Install it: pip install xgboost"
        ) from exc


def fit_xgboost(train_samples: list[Sample]) -> dict[str, Any]:
    """Train an XGBoost regressor.  Requires: pip install xgboost."""
    xgb = _check_xgboost()
    X = [s.features for s in train_samples]
    y = [s.target_price for s in train_samples]
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return {"model_name": "xgboost", "_model": model}


def predict_xgboost(model_state: dict[str, Any], features: list[float]) -> float:
    import numpy as np  # type: ignore[import]
    X = np.array(features, dtype=float).reshape(1, -1)
    return float(model_state["_model"].predict(X)[0])


def fit_model(train_samples: list[Sample], model_name: str) -> dict[str, Any]:
    if model_name == "ridge":
        return {"model_name": model_name, "weights": fit_ridge_regression(train_samples, alpha=1.0)}
    if model_name == "naive":
        return {"model_name": model_name}
    if model_name == "exponential_smoothing":
        return fit_exponential_smoothing(train_samples)
    if model_name == "xgboost":
        return fit_xgboost(train_samples)
    raise ValueError(f"Unsupported model '{model_name}'. Available models: {', '.join(sorted(AVAILABLE_MODELS))}")


def predict_model(model_state: dict[str, Any], current_price: float, features: list[float]) -> float:
    model_name = model_state["model_name"]
    if model_name == "ridge":
        return predict_ridge(model_state["weights"], features)
    if model_name == "naive":
        return current_price
    if model_name == "exponential_smoothing":
        return _ewa_predict(model_state["alpha"], features)
    if model_name == "xgboost":
        return predict_xgboost(model_state, features)
    raise ValueError(f"Unsupported model '{model_name}'.")


def round_float(value: float, digits: int = 6) -> float:
    return round(value, digits)


def evaluate(model_state: dict[str, Any], samples: list[Sample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    direction_hits = 0

    for sample in samples:
        predicted_price = predict_model(model_state, sample.current_price, sample.features)
        absolute_error = abs(predicted_price - sample.target_price)
        squared_error = (predicted_price - sample.target_price) ** 2
        predicted_direction = 1 if predicted_price >= sample.current_price else 0
        actual_direction = 1 if sample.target_price >= sample.current_price else 0
        direction_hits += int(predicted_direction == actual_direction)
        absolute_errors.append(absolute_error)
        squared_errors.append(squared_error)
        predictions.append(
            {
                "date": sample.date,
                "current_price": round_float(sample.current_price, 4),
                "actual_next_price": round_float(sample.target_price, 4),
                "predicted_next_price": round_float(predicted_price, 4),
                "absolute_error": round_float(absolute_error, 4),
                "predicted_direction_up": bool(predicted_direction),
                "actual_direction_up": bool(actual_direction),
            }
        )

    metrics = {
        "mae": round_float(sum(absolute_errors) / len(absolute_errors)),
        "rmse": round_float(math.sqrt(sum(squared_errors) / len(squared_errors))),
        "directional_accuracy": round_float(direction_hits / len(samples)),
    }
    return metrics, predictions


def latest_prediction(model_state: dict[str, Any], rows: list[PriceRow]) -> dict[str, Any]:
    current_index = len(rows) - 1
    current_price = rows[current_index].price
    features = build_features(rows, current_index)
    predicted_price = predict_model(model_state, current_price, features)
    return {
        "latest_observation_date": rows[current_index].date,
        "latest_observation_price": round_float(current_price, 4),
        "predicted_next_price": round_float(predicted_price, 4),
        "predicted_direction_up": bool(predicted_price >= current_price),
    }


def data_audit(
    rows: list[PriceRow],
    dataset_path: Path,
    project_root: Path,
    source_url: str,
    download_metadata: dict[str, Any],
) -> dict[str, Any]:
    prices = [row.price for row in rows]
    return {
        "source_url": source_url,
        "local_dataset_path": str(dataset_path.relative_to(project_root)),
        **download_metadata,
        "row_count": len(rows),
        "date_range": {"start": rows[0].date, "end": rows[-1].date},
        "price_summary": {
            "minimum": round_float(min(prices), 4),
            "maximum": round_float(max(prices), 4),
            "average": round_float(sum(prices) / len(prices), 4),
        },
        "head": [asdict(row) for row in rows[:3]],
        "tail": [asdict(row) for row in rows[-3:]],
    }


def write_json(path: Path, content: dict[str, Any] | list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    if not predictions:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)


def write_manual_audit(
    path: Path,
    dataset_audit_content: dict[str, Any],
    model_audit_content: dict[str, Any],
    prediction_summary: dict[str, Any],
    interpretation: dict[str, Any] | None = None,
) -> None:
    ensure_directory(path.parent)
    commodity_name = model_audit_content.get("commodity_name", model_audit_content.get("commodity", "Unknown"))
    lines = [
        "# Manual Verification Audit",
        "",
        f"## Commodity: {commodity_name}",
        "",
        "## Dataset checks",
        f"- Source URL: {dataset_audit_content.get('source_url', 'N/A')}",
        f"- Local file: {dataset_audit_content.get('local_dataset_path', dataset_audit_content.get('source_url', 'N/A'))}",
        f"- SHA256: {dataset_audit_content.get('sha256', 'N/A')}",
        f"- Rows: {dataset_audit_content['row_count']}",
        f"- Date range: {dataset_audit_content['date_range']['start']} to {dataset_audit_content['date_range']['end']}",
        f"- Data lineage: {dataset_audit_content.get('data_lineage', 'N/A')}",
        "",
        "## Model checks",
        f"- Trained at: {model_audit_content['trained_at']}",
        f"- Feature flags: {model_audit_content.get('feature_flags', 'base')}",
        f"- Training samples: {model_audit_content['train_sample_count']}",
        f"- Test samples: {model_audit_content['test_sample_count']}",
        f"- Test MAE: {model_audit_content['test_metrics']['mae']}",
        f"- Test RMSE: {model_audit_content['test_metrics']['rmse']}",
        f"- Directional accuracy: {model_audit_content['test_metrics']['directional_accuracy']}",
        "",
        "## Latest model output",
        f"- Latest observation date: {prediction_summary['latest_observation_date']}",
        f"- Latest observation price: {prediction_summary['latest_observation_price']}",
        f"- Predicted next price: {prediction_summary['predicted_next_price']}",
        f"- Predicted direction up: {prediction_summary['predicted_direction_up']}",
        "",
    ]
    if interpretation:
        lines += [
            "## AI Interpretation",
            f"- Source: {interpretation.get('source', 'template')}",
            f"- Model: {interpretation.get('model', 'N/A')}",
            "",
            interpretation.get("summary", ""),
            "",
        ]
    lines += [
        "## Manual verification steps",
        "- Open the raw CSV and confirm the first and last rows match the data audit JSON.",
        "- Recompute the SHA256 of the raw CSV and confirm it matches the recorded digest.",
        "- Open the predictions CSV and confirm dates are in chronological order.",
        "- Spot-check that predicted_direction_up matches whether predicted_next_price is greater than or equal to current_price.",
        "- Re-run the pipeline and confirm the audit files refresh successfully.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_multi_day_forecast(
    model_state: dict[str, Any],
    rows: list[PriceRow],
    horizon_days: int,
    use_seasonality: bool,
    external_feature_sets: list[Any],
    context_feature_series: dict[str, Any],
) -> list[dict[str, Any]]:
    from .context_features import build_context_features
    from .external_features import enrich_features

    working_rows = [PriceRow(date=r.date, price=r.price) for r in rows]
    forecasts: list[dict[str, Any]] = []
    for step in range(1, horizon_days + 1):
        current_idx = len(working_rows) - 1
        current_row = working_rows[current_idx]
        feature_date = current_row.date
        forecast_date = _next_calendar_date(current_row.date)
        base_feats = build_features(working_rows, current_idx)
        combined_feats = list(base_feats)
        if use_seasonality:
            combined_feats += build_seasonality_features(feature_date)
        for category, series in context_feature_series.items():
            combined_feats += build_context_features(feature_date, series)
        if external_feature_sets:
            pre_external_names = (
                list(BASE_FEATURE_NAMES)
                + (list(SEASONALITY_FEATURE_NAMES) if use_seasonality else [])
                + _context_feature_names(context_feature_series)
            )
            combined_feats, _ = enrich_features(
                feature_date, combined_feats, pre_external_names, external_feature_sets
            )
        predicted_price = predict_model(model_state, current_row.price, combined_feats)
        forecasts.append(
            {
                "step_day": step,
                "forecast_date": forecast_date,
                "predicted_price": round_float(predicted_price, 4),
                "predicted_direction_up_vs_previous_day": bool(predicted_price >= current_row.price),
            }
        )
        working_rows.append(PriceRow(date=forecast_date, price=float(predicted_price)))
    return forecasts


def run_pipeline(
    project_root: Path,
    commodity: str = "brent",
    features: str = "base",
    model_name: str = "ridge",
    horizon_days: int = 1,
    allow_demo_fallback: bool = False,
    category: str | None = None,
    context_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full OilEnergy forecasting pipeline.

    Args:
        project_root: Root directory of the project.
        commodity: Commodity key (e.g., "brent", "qatar_lng", "wti").
                   See config/commodities.yaml for all options.
        features: Comma-separated feature flags:
                  "base"        — price lags and rolling statistics only (default)
                  "seasonality" — add calendar/seasonal features
                  "weather"     — add weather context features from local CSV input
                  "demand"      — add local/regional demand context features from local CSV input
                  "external"    — add correlated commodity cross-features
                  "all"         — enable all feature groups

    Returns:
        dict containing dataset_audit, model_audit, correlation_audit (if external),
        and interpretation (LLM summary).
    """
    from .commodities import (
        COMMODITIES,
        commodity_data_audit,
        list_commodities_by_type,
        load_commodity,
    )
    from .context_features import build_context_features, load_context_series
    from .external_features import enrich_features
    from .llm_interpreter import interpret_results

    feature_flags = {f.strip().lower() for f in features.split(",")}
    if "all" in feature_flags:
        feature_flags |= {"base", "seasonality", "external", "weather", "demand"}
    use_seasonality = "seasonality" in feature_flags
    use_external = "external" in feature_flags
    use_weather = "weather" in feature_flags
    use_demand = "demand" in feature_flags

    if horizon_days < 1 or horizon_days > 30:
        raise ValueError("horizon_days must be between 1 and 30.")
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {', '.join(sorted(AVAILABLE_MODELS))}"
        )
    commodity = commodity.strip().lower()
    if category:
        category_norm = category.strip().lower()
        if category_norm not in {"oil", "gas"}:
            raise ValueError("category must be either 'oil' or 'gas'.")
        if commodity not in COMMODITIES:
            category_options = list_commodities_by_type(category_norm)
            if not category_options:
                raise ValueError(f"No commodities configured for category '{category_norm}'.")
            commodity = category_options[0]["key"]
        elif COMMODITIES[commodity]["type"] != category_norm:
            raise ValueError(
                f"Commodity '{commodity}' is type '{COMMODITIES[commodity]['type']}', not '{category_norm}'."
            )

    artifacts_dir = project_root / "artifacts"
    audits_dir = project_root / "audits"
    cache_dir = project_root / "data" / "raw"

    # Load primary commodity
    commodity_data = load_commodity(
        commodity, cache_dir=cache_dir, allow_demo_fallback=allow_demo_fallback
    )
    rows = commodity_data.rows

    contextual_feature_availability: dict[str, Any] = {}
    context_feature_series: dict[str, Any] = {}
    if use_weather:
        weather_series, weather_availability = load_context_series(
            "weather", project_root, context_data_dir=context_data_dir
        )
        contextual_feature_availability["weather"] = weather_availability
        if weather_series:
            context_feature_series["weather"] = weather_series
    if use_demand:
        demand_series, demand_availability = load_context_series(
            "demand", project_root, context_data_dir=context_data_dir
        )
        contextual_feature_availability["demand"] = demand_availability
        if demand_series:
            context_feature_series["demand"] = demand_series

    # Build correlation matrix and external feature sets if requested
    external_feature_sets: list[Any] = []
    correlation_audit: dict[str, Any] = {}

    if use_external:
        from .correlation_analysis import compute_correlation_matrix, correlation_matrix_to_dict, significant_partners
        from .external_features import load_external_feature_sets

        all_keys = [k for k in COMMODITIES if k != commodity]
        correlation_threshold = 0.5
        matrix, corr_errors = compute_correlation_matrix(
            [commodity] + all_keys,
            cache_dir=cache_dir,
            threshold=correlation_threshold,
            allow_demo_fallback=allow_demo_fallback,
        )
        partners = significant_partners(commodity, matrix)
        correlation_audit = correlation_matrix_to_dict(matrix, [commodity] + all_keys, threshold=correlation_threshold)
        correlation_audit["significant_partners_for_target"] = partners
        correlation_audit["correlation_errors"] = corr_errors

        if partners:
            external_feature_sets, ext_errors = load_external_feature_sets(
                partners, cache_dir=cache_dir, allow_demo_fallback=allow_demo_fallback
            )
            correlation_audit["external_feature_load_errors"] = ext_errors

    samples, feature_names = build_samples(
        rows,
        use_seasonality=use_seasonality,
        external_feature_sets=external_feature_sets if use_external else None,
        context_feature_series=context_feature_series,
    )
    train_samples, test_samples = split_samples(samples)

    model_state = fit_model(train_samples, model_name=model_name)
    test_metrics, predictions = evaluate(model_state, test_samples)

    # Latest prediction
    current_index = len(rows) - 1
    current_price = rows[current_index].price
    base_feats = build_features(rows, current_index)
    combined_feats = list(base_feats)
    feature_date = rows[current_index].date
    if use_seasonality:
        combined_feats = combined_feats + build_seasonality_features(feature_date)
    for context_category, context_series in context_feature_series.items():
        combined_feats += build_context_features(feature_date, context_series)
    if use_external and external_feature_sets:
        pre_external_names = (
            list(BASE_FEATURE_NAMES)
            + (list(SEASONALITY_FEATURE_NAMES) if use_seasonality else [])
            + _context_feature_names(context_feature_series)
        )
        combined_feats, _ = enrich_features(
            feature_date, combined_feats, pre_external_names, external_feature_sets
        )
    predicted_price = predict_model(model_state, current_price, combined_feats)
    prediction_summary = {
        "latest_observation_date": rows[current_index].date,
        "latest_observation_price": round_float(current_price, 4),
        "predicted_next_price": round_float(predicted_price, 4),
        "predicted_direction_up": bool(predicted_price >= current_price),
    }
    multi_day_forecast = generate_multi_day_forecast(
        model_state=model_state,
        rows=rows,
        horizon_days=horizon_days,
        use_seasonality=use_seasonality,
        external_feature_sets=external_feature_sets if use_external else [],
        context_feature_series=context_feature_series,
    )

    dataset_audit_content = commodity_data_audit(commodity_data, project_root)
    commodity_name = COMMODITIES.get(commodity, {}).get("name", commodity)

    coefficients: dict[str, Any] = {}
    if model_name == "ridge":
        weights = model_state["weights"]
        coefficients = {
            "intercept": round_float(weights[0]),
            **{name: round_float(weight) for name, weight in zip(feature_names, weights[1:])},
        }
    elif model_name == "exponential_smoothing":
        coefficients = {
            "alpha": round_float(model_state["alpha"]),
            "train_mae_at_best_alpha": round_float(model_state.get("train_mae_at_best_alpha", 0.0)),
        }
    elif model_name == "xgboost":
        try:
            importances = model_state["_model"].feature_importances_
            coefficients = {name: round_float(float(imp)) for name, imp in zip(feature_names, importances)}
        except Exception:
            coefficients = {}

    model_audit_content = {
        "commodity": commodity,
        "commodity_name": commodity_name,
        "commodity_type": COMMODITIES.get(commodity, {}).get("type"),
        "is_proxy_commodity": bool(COMMODITIES.get(commodity, {}).get("is_proxy", False)),
        "proxy_for": COMMODITIES.get(commodity, {}).get("proxy_for"),
        "trained_at": utc_now(),
        "model_name": model_name,
        "horizon_days": horizon_days,
        "forecast_method": "recursive multi-step forecast",
        "forecast_limitations": (
            "Recursive forecasting compounds model error with each step. "
            "Future exogenous features (weather/demand/external prices) use latest known values when future dates are unavailable."
        ),
        "feature_flags": features,
        "feature_names": feature_names,
        "coefficients": coefficients,
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "test_metrics": test_metrics,
        "latest_prediction": prediction_summary,
        "multi_day_forecast": multi_day_forecast,
        "contextual_feature_availability": contextual_feature_availability,
        "external_feature_partners": [fs.commodity_key for fs in external_feature_sets],
        "allow_demo_fallback": allow_demo_fallback,
    }

    # LLM interpretation
    interpretation = interpret_results(model_audit_content, commodity_name=commodity_name)

    # Write artifacts
    write_json(artifacts_dir / "model.json", model_audit_content)
    write_predictions_csv(artifacts_dir / "test_predictions.csv", predictions)
    write_json(audits_dir / "data_audit.json", dataset_audit_content)
    write_json(audits_dir / "model_audit.json", model_audit_content)
    if correlation_audit:
        write_json(audits_dir / "correlation_audit.json", correlation_audit)
    write_manual_audit(
        audits_dir / "manual_verification.md",
        dataset_audit_content,
        model_audit_content,
        prediction_summary,
        interpretation=interpretation,
    )

    return {
        "dataset_audit": dataset_audit_content,
        "model_audit": model_audit_content,
        "correlation_audit": correlation_audit,
        "interpretation": interpretation,
    }
