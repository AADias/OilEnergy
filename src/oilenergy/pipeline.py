from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


def build_samples(
    rows: list[PriceRow],
    use_seasonality: bool = False,
    external_feature_sets: list[Any] | None = None,
    context_series_list: list[Any] | None = None,
    context_start: str | None = None,
    context_end: str | None = None,
) -> tuple[list[Sample], list[str]]:
    """Build feature samples from a price series.

    Args:
        rows: Ordered list of price observations.
        use_seasonality: If True, append calendar/seasonal features.
        external_feature_sets: Optional list of ExternalFeatureSet objects.
        context_series_list: Optional list of ContextSeries objects (weather/demand).
        context_start: If set, skip samples dated before this (ISO date string).
                       Used to avoid backfilling context data pre-history.
        context_end: If set, skip samples dated after this (ISO date string).

    Returns:
        (samples, feature_names)
    """
    from .external_features import enrich_features

    base_names = list(BASE_FEATURE_NAMES)
    seasonal_names = list(SEASONALITY_FEATURE_NAMES) if use_seasonality else []
    pre_external_names = base_names + seasonal_names

    feature_names = pre_external_names[:]
    ext_sets = external_feature_sets or []
    ctx_sets = context_series_list or []

    for fs in ext_sets:
        feature_names = feature_names + list(fs.feature_names)
    for cs in ctx_sets:
        feature_names = feature_names + list(cs.feature_names)

    samples: list[Sample] = []
    for index in range(10, len(rows) - 1):
        date_str = rows[index].date

        # Restrict to context coverage window to prevent backfill leakage
        if context_start and date_str < context_start:
            continue
        if context_end and date_str > context_end:
            continue

        current_price = rows[index].price
        base_feats = build_features(rows, index)
        combined = list(base_feats)
        if use_seasonality:
            combined = combined + build_seasonality_features(date_str)
        if ext_sets:
            combined, _ = enrich_features(date_str, combined, pre_external_names, ext_sets)
        if ctx_sets:
            from .context_features import build_context_features
            for cs in ctx_sets:
                ctx_feats = build_context_features(date_str, cs)
                if ctx_feats is not None:
                    combined.extend(ctx_feats)
                else:
                    # Outside coverage — skip this sample entirely
                    combined = None  # type: ignore[assignment]
                    break
            if combined is None:
                continue

        samples.append(
            Sample(
                date=date_str,
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


def predict(weights: list[float], features: list[float]) -> float:
    return weights[0] + sum(weight * value for weight, value in zip(weights[1:], features))


def round_float(value: float, digits: int = 6) -> float:
    return round(value, digits)


def evaluate(weights: list[float], samples: list[Sample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    direction_hits = 0

    for sample in samples:
        predicted_price = predict(weights, sample.features)
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


def naive_predict(rows: list[PriceRow]) -> float:
    """Naive (persistence) baseline: predict last known price as the next price."""
    return rows[-1].price


def recursive_forecast(
    weights: list[float],
    rows: list[PriceRow],
    horizon_days: int,
    use_seasonality: bool = False,
    external_feature_sets: list[Any] | None = None,
    model: str = "ridge",
) -> list[dict[str, Any]]:
    """Produce a multi-day recursive forecast without using future observed values.

    Disclosure: recursive forecasting feeds each predicted price back as the
    latest observation for the next step.  External commodity series and context
    data are carried forward at their LAST KNOWN value — no future observations
    are used.  Forecast dates are calendar days; some may not be trading days.

    Args:
        weights:    Fitted ridge regression weights (ignored when model='naive').
        rows:       Historical price rows (read-only, not extended).
        horizon_days: Number of calendar days to forecast ahead.
        use_seasonality: Include calendar features.
        external_feature_sets: Optional external price feature sets.
        model:      "ridge" or "naive".

    Returns:
        List of dicts with keys: step, date (calendar), predicted_price,
        predicted_direction_up, method, carry_forward_disclosure.
    """
    from datetime import date as date_cls, timedelta
    import datetime as dt_module

    ext_sets = external_feature_sets or []
    # Work on a mutable copy so we can append synthetic rows without mutating caller state
    working_rows = list(rows)
    last_date = working_rows[-1].date

    results: list[dict[str, Any]] = []
    try:
        last_dt = date_cls.fromisoformat(last_date[:10])
    except ValueError:
        last_dt = date_cls.today()

    for step in range(1, horizon_days + 1):
        forecast_dt = last_dt + timedelta(days=step)
        forecast_date = forecast_dt.isoformat()

        if model == "naive":
            predicted_price = working_rows[-1].price
        else:
            idx = len(working_rows) - 1
            base_feats = build_features(working_rows, idx)
            combined = list(base_feats)
            if use_seasonality:
                combined += build_seasonality_features(forecast_date)
            if ext_sets:
                from .external_features import enrich_features
                pre_names = list(BASE_FEATURE_NAMES) + (list(SEASONALITY_FEATURE_NAMES) if use_seasonality else [])
                combined, _ = enrich_features(
                    # Use the last available date in each external series (carry-forward)
                    working_rows[-1].date,
                    combined,
                    pre_names,
                    ext_sets,
                )
            predicted_price = predict(weights, combined)

        results.append(
            {
                "step": step,
                "date": forecast_date,
                "date_type": "calendar_day",
                "predicted_price": round_float(predicted_price, 4),
                "predicted_direction_up": bool(predicted_price >= working_rows[-1].price),
                "method": model,
                "carry_forward_disclosure": (
                    "Recursive: each predicted price is the input for the next step. "
                    "External/context series held at last known value. "
                    "Forecast dates are calendar days and may include weekends/holidays."
                ) if step > 1 or ext_sets else (
                    "Single-step forecast. "
                    "Forecast date is a calendar day and may not be a trading day."
                ),
            }
        )
        # Append predicted price as a synthetic observation for the next recursive step
        working_rows.append(PriceRow(date=forecast_date, price=predicted_price))

    return results


def evaluate_naive(samples: list[Sample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a naive (persistence) model: predicted == current_price."""
    predictions: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    direction_hits = 0

    for sample in samples:
        predicted_price = sample.current_price  # naive: no change
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


def run_pipeline(
    project_root: Path,
    commodity: str = "brent",
    features: str = "base",
    model: str = "ridge",
    horizon_days: int = 1,
    category: str | None = None,
) -> dict[str, Any]:
    """Run the full OilEnergy forecasting pipeline.

    Args:
        project_root: Root directory of the project.
        commodity: Commodity key (e.g., "brent", "qatar_lng", "wti").
                   See config/commodities.yaml for all options.
        features: Comma-separated feature flags:
                  "base"        — price lags and rolling statistics only (default)
                  "seasonality" — add calendar/seasonal features
                  "external"    — add correlated commodity cross-features
                  "context"     — add local weather/demand context features (requires
                                  data/context/weather.csv and data/context/demand.csv;
                                  features are applied ONLY for dates within the
                                  explicit coverage of the context files)
                  "all"         — enable all of the above
        model: "ridge" (default) or "naive" (persistence baseline).
        horizon_days: Number of calendar days to forecast ahead (default: 1).
                      Multi-step forecasts use recursive carry-forward; see
                      the "carry_forward_disclosure" field in the output.
        category: Optional category filter label ("oil" or "gas") for auditing.

    Returns:
        dict with dataset_audit, model_audit, correlation_audit, context_audit,
        forecast (multi-step), and interpretation keys.
    """
    from .commodities import load_commodity, commodity_data_audit, COMMODITIES
    from .llm_interpreter import interpret_results

    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if model not in ("ridge", "naive"):
        raise ValueError(f"Unknown model '{model}'. Choose 'ridge' or 'naive'.")

    feature_flags = {f.strip().lower() for f in features.split(",")}
    use_seasonality = "seasonality" in feature_flags or "all" in feature_flags
    use_external = "external" in feature_flags or "all" in feature_flags
    use_context = "context" in feature_flags or "all" in feature_flags

    artifacts_dir = project_root / "artifacts"
    audits_dir = project_root / "audits"
    cache_dir = project_root / "data" / "raw"

    # Load primary commodity
    commodity_data = load_commodity(commodity, cache_dir=cache_dir)
    rows = commodity_data.rows

    # --- Load context features (weather, demand) with date-range restriction ---
    context_audit: dict[str, Any] = {"requested": use_context, "series": {}}
    context_series_list: list[Any] = []  # list of context_features.ContextSeries

    if use_context:
        from .context_features import load_context_series, build_context_features

        for cat in ("weather", "demand"):
            series, avail = load_context_series(cat, project_root)
            context_audit["series"][cat] = avail
            if series is not None:
                context_series_list.append(series)

        if not context_series_list:
            context_audit["status"] = (
                "unavailable — no context CSV files found in data/context/. "
                "Run scripts/fetch_weather.py to fetch real data, or provide "
                "your own CSV with columns: date, value."
            )
        else:
            context_audit["status"] = "loaded"
            loaded_cats = [s.category for s in context_series_list]
            context_audit["loaded_categories"] = loaded_cats

    # --- Build correlation matrix and external feature sets if requested ---
    external_feature_sets: list[Any] = []
    correlation_audit: dict[str, Any] = {}

    if use_external:
        from .correlation_analysis import compute_correlation_matrix, correlation_matrix_to_dict, significant_partners
        from .external_features import load_external_feature_sets

        all_keys = [k for k in COMMODITIES if k != commodity]
        correlation_threshold = 0.5
        matrix, corr_errors = compute_correlation_matrix(
            [commodity] + all_keys, cache_dir=cache_dir, threshold=correlation_threshold
        )
        partners = significant_partners(commodity, matrix)
        correlation_audit = correlation_matrix_to_dict(matrix, [commodity] + all_keys, threshold=correlation_threshold)
        correlation_audit["significant_partners_for_target"] = partners
        correlation_audit["correlation_errors"] = corr_errors
        correlation_audit["duplicate_source_exclusion_note"] = (
            "Commodities sharing the same underlying data source (e.g. qatar_lng/henry_hub "
            "or brent/opec_basket) are excluded from significant_partners to prevent "
            "near-1.0 spurious correlations from contaminating external features."
        )

        if partners:
            external_feature_sets, ext_errors = load_external_feature_sets(
                partners, cache_dir=cache_dir
            )
            correlation_audit["external_feature_load_errors"] = ext_errors

    # --- Determine effective date range for sample building ---
    # When context features are active, restrict samples to dates within the
    # context coverage window to avoid any backfill or leakage.
    context_coverage_start: str | None = None
    context_coverage_end: str | None = None
    if use_context and context_series_list:
        starts = [s.coverage_start for s in context_series_list if s.coverage_start]
        ends = [s.coverage_end for s in context_series_list if s.coverage_end]
        if starts and ends:
            context_coverage_start = max(starts)
            context_coverage_end = min(ends)
            context_audit["effective_coverage"] = {
                "start": context_coverage_start,
                "end": context_coverage_end,
                "note": (
                    "Training samples are restricted to this window to prevent "
                    "backfilling pre-history context values."
                ),
            }

    samples, feature_names = build_samples(
        rows,
        use_seasonality=use_seasonality,
        external_feature_sets=external_feature_sets if use_external else None,
        context_series_list=context_series_list if use_context else None,
        context_start=context_coverage_start,
        context_end=context_coverage_end,
    )
    train_samples, test_samples = split_samples(samples)

    if model == "naive":
        weights: list[float] = []
    else:
        weights = fit_ridge_regression(train_samples, alpha=1.0)

    if model == "naive":
        test_metrics, predictions = evaluate_naive(test_samples)
    else:
        test_metrics, predictions = evaluate(weights, test_samples)

    # --- Single-step latest prediction ---
    current_index = len(rows) - 1
    current_price = rows[current_index].price

    if model == "naive":
        predicted_price = naive_predict(rows)
    else:
        base_feats = build_features(rows, current_index)
        combined_feats = list(base_feats)
        if use_seasonality:
            combined_feats = combined_feats + build_seasonality_features(rows[current_index].date)
        if use_external and external_feature_sets:
            from .external_features import enrich_features
            pre_external_names = list(BASE_FEATURE_NAMES) + (list(SEASONALITY_FEATURE_NAMES) if use_seasonality else [])
            combined_feats, _ = enrich_features(
                rows[current_index].date, combined_feats, pre_external_names, external_feature_sets
            )
        if use_context and context_series_list:
            from .context_features import build_context_features
            for cs in context_series_list:
                ctx_feats = build_context_features(rows[current_index].date, cs)
                if ctx_feats is not None:
                    combined_feats.extend(ctx_feats)
        predicted_price = predict(weights, combined_feats)

    prediction_summary = {
        "latest_observation_date": rows[current_index].date,
        "latest_observation_price": round_float(current_price, 4),
        "predicted_next_price": round_float(predicted_price, 4),
        "predicted_direction_up": bool(predicted_price >= current_price),
    }

    # --- Multi-day recursive forecast ---
    forecast: list[dict[str, Any]] = []
    if horizon_days > 1:
        forecast = recursive_forecast(
            weights,
            rows,
            horizon_days=horizon_days,
            use_seasonality=use_seasonality,
            external_feature_sets=external_feature_sets if use_external else None,
            model=model,
        )
    else:
        from datetime import date as _date, timedelta as _timedelta
        try:
            _next_dt = _date.fromisoformat(rows[-1].date[:10]) + _timedelta(days=1)
            _next_date = _next_dt.isoformat()
        except ValueError:
            _next_date = prediction_summary["latest_observation_date"]
        forecast = [
            {
                "step": 1,
                "date": _next_date,
                "date_type": "calendar_day",
                "predicted_price": prediction_summary["predicted_next_price"],
                "predicted_direction_up": prediction_summary["predicted_direction_up"],
                "method": model,
                "carry_forward_disclosure": "Single-step forecast. Forecast date is a calendar day and may not be a trading day.",
            }
        ]

    dataset_audit_content = commodity_data_audit(commodity_data, project_root)
    commodity_name = COMMODITIES.get(commodity, {}).get("name", commodity)

    model_audit_content = {
        "commodity": commodity,
        "commodity_name": commodity_name,
        "category": category or COMMODITIES.get(commodity, {}).get("type", "unknown"),
        "model": model,
        "horizon_days": horizon_days,
        "trained_at": utc_now(),
        "feature_flags": features,
        "feature_names": feature_names,
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "test_metrics": test_metrics,
        "latest_prediction": prediction_summary,
        "forecast": forecast,
    }
    if model == "ridge":
        model_audit_content["coefficients"] = {
            "intercept": round_float(weights[0]),
            **{name: round_float(weight) for name, weight in zip(feature_names, weights[1:])},
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
    if context_audit and use_context:
        write_json(audits_dir / "context_audit.json", context_audit)
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
        "context_audit": context_audit,
        "forecast": forecast,
        "interpretation": interpretation,
    }
