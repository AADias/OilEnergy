from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .commodities import PriceRow, download_dataset, get_commodity_definition, load_config, load_rows, utc_now
from .external_features import build_external_feature_lookup

FEATURE_NAMES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "mean_5",
    "mean_10",
    "momentum_5",
    "volatility_5",
    "month_of_year",
    "quarter",
    "day_of_week",
    "is_weekend_or_holiday",
]


@dataclass
class Sample:
    date: str
    current_price: float
    target_price: float
    features: list[float]


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    current_day = date(year, month, 1)
    while current_day.weekday() != weekday:
        current_day += timedelta(days=1)
    current_day += timedelta(days=(occurrence - 1) * 7)
    return current_day


def observed_fixed_holiday(year: int, month: int, day_of_month: int) -> date:
    holiday = date(year, month, day_of_month)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def is_weekend_or_holiday(observation_date: date) -> bool:
    # Keep this deliberately small and transparent: weekends plus a few widely-recognized US holidays.
    # Users who need market-specific holiday calendars can layer them in as custom event features later.
    holidays = {
        observed_fixed_holiday(observation_date.year - 1, 1, 1),
        observed_fixed_holiday(observation_date.year, 1, 1),
        observed_fixed_holiday(observation_date.year + 1, 1, 1),
        observed_fixed_holiday(observation_date.year, 7, 4),
        observed_fixed_holiday(observation_date.year, 12, 25),
        nth_weekday_of_month(observation_date.year, 11, weekday=3, occurrence=4),
    }
    return observation_date.weekday() >= 5 or observation_date in holidays


def seasonal_features(observation_date: str) -> list[float]:
    parsed_date = parse_iso_date(observation_date)
    return [
        float(parsed_date.month),
        float(((parsed_date.month - 1) // 3) + 1),
        float(parsed_date.weekday()),
        float(is_weekend_or_holiday(parsed_date)),
    ]


def rolling_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("rolling_mean requires at least one value.")
    return sum(values) / len(values)


def rolling_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def build_features(rows: list[PriceRow], index: int, external_feature_values: list[float] | None = None) -> list[float]:
    window_5 = [rows[index - offset].price for offset in range(0, 5)]
    window_10 = [rows[index - offset].price for offset in range(0, 10)]
    current_price = rows[index].price
    feature_values = [
        rows[index - 1].price,
        rows[index - 2].price,
        rows[index - 3].price,
        rows[index - 5].price,
        rows[index - 10].price,
        rolling_mean(window_5),
        rolling_mean(window_10),
        current_price - rows[index - 5].price,
        rolling_std(window_5),
        *seasonal_features(rows[index].date),
    ]
    if external_feature_values is not None:
        feature_values.extend(external_feature_values)
    return feature_values


def build_samples(rows: list[PriceRow], external_feature_lookup: dict[str, list[float]] | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for index in range(10, len(rows) - 1):
        current_price = rows[index].price
        external_feature_values = None
        if external_feature_lookup is not None:
            external_feature_values = external_feature_lookup.get(rows[index].date)
            if external_feature_values is None:
                raise KeyError(f"Missing external features for {rows[index].date}.")
        samples.append(
            Sample(
                date=rows[index].date,
                current_price=current_price,
                target_price=rows[index + 1].price,
                features=build_features(rows, index, external_feature_values),
            )
        )
    return samples


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


def latest_prediction(
    weights: list[float],
    rows: list[PriceRow],
    external_feature_lookup: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    current_index = len(rows) - 1
    current_price = rows[current_index].price
    external_feature_values = None
    if external_feature_lookup is not None:
        external_feature_values = external_feature_lookup.get(rows[current_index].date)
        if external_feature_values is None:
            raise KeyError(f"Missing external features for {rows[current_index].date}.")
    features = build_features(rows, current_index, external_feature_values)
    predicted_price = predict(weights, features)
    return {
        "latest_observation_date": rows[current_index].date,
        "latest_observation_price": round_float(current_price, 4),
        "predicted_next_price": round_float(predicted_price, 4),
        "predicted_direction_up": bool(predicted_price >= current_price),
    }


def significant_features(feature_names: list[str], weights: list[float], top_n: int = 5) -> list[dict[str, float | str]]:
    significant = [
        {"feature": feature_name, "coefficient": round_float(weight), "absolute_weight": round_float(abs(weight))}
        for feature_name, weight in zip(feature_names, weights[1:])
    ]
    significant.sort(key=lambda item: item["absolute_weight"], reverse=True)
    return significant[:top_n]


def data_audit(
    rows: list[PriceRow],
    dataset_path: Path,
    project_root: Path,
    commodity_name: str,
    commodity_definition: Any,
    download_metadata: dict[str, Any],
    external_dataset_audits: list[dict[str, Any]],
    external_alignment_start: str,
) -> dict[str, Any]:
    prices = [row.price for row in rows]
    return {
        "commodity": commodity_name,
        "commodity_display_name": commodity_definition.display_name,
        "source_url": commodity_definition.source_url,
        "proxy_note": commodity_definition.proxy_note,
        "local_dataset_path": str(dataset_path.relative_to(project_root)),
        **download_metadata,
        "row_count": len(rows),
        "date_range": {"start": rows[0].date, "end": rows[-1].date},
        "price_summary": {
            "minimum": round_float(min(prices), 4),
            "maximum": round_float(max(prices), 4),
            "average": round_float(sum(prices) / len(prices), 4),
        },
        "external_sources": external_dataset_audits,
        "external_alignment_start": external_alignment_start,
        "head": [asdict(row) for row in rows[:3]],
        "tail": [asdict(row) for row in rows[-3:]],
    }


def write_json(path: Path, content: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
) -> None:
    external_feature_list = model_audit_content["external_features"]
    lines = [
        "# Manual Verification Audit",
        "",
        "## Dataset checks",
        f"- Commodity: {dataset_audit_content['commodity_display_name']} ({dataset_audit_content['commodity']})",
        f"- Source URL: {dataset_audit_content['source_url']}",
        f"- Local file: {dataset_audit_content['local_dataset_path']}",
        f"- SHA256: {dataset_audit_content['sha256']}",
        f"- Rows: {dataset_audit_content['row_count']}",
        f"- Date range: {dataset_audit_content['date_range']['start']} to {dataset_audit_content['date_range']['end']}",
        "",
        "## Model checks",
        f"- Trained at: {model_audit_content['trained_at']}",
        f"- External feature commodities: {', '.join(external_feature_list) if external_feature_list else 'None'}",
        f"- Training samples: {model_audit_content['train_sample_count']}",
        f"- Test samples: {model_audit_content['test_sample_count']}",
        f"- Test MAE: {model_audit_content['test_metrics']['mae']}",
        f"- Test RMSE: {model_audit_content['test_metrics']['rmse']}",
        f"- Directional accuracy: {model_audit_content['test_metrics']['directional_accuracy']}",
        "",
        "## Feature significance",
        *[
            f"- {feature['feature']}: coefficient={feature['coefficient']} abs_weight={feature['absolute_weight']}"
            for feature in model_audit_content["significant_features"]
        ],
        "",
        "## Latest model output",
        f"- Latest observation date: {prediction_summary['latest_observation_date']}",
        f"- Latest observation price: {prediction_summary['latest_observation_price']}",
        f"- Predicted next price: {prediction_summary['predicted_next_price']}",
        f"- Predicted direction up: {prediction_summary['predicted_direction_up']}",
        "",
        "## Manual verification steps",
        "- Open the raw CSV and confirm the first and last rows match the data audit JSON.",
        "- Recompute the SHA256 of the raw CSV and confirm it matches the recorded digest.",
        "- If external features were enabled, spot-check one aligned commodity series in the raw CSV and confirm the fill method used in the audit.",
        "- Open the predictions CSV and confirm dates are in chronological order.",
        "- Spot-check that predicted_direction_up matches whether predicted_next_price is greater than or equal to current_price.",
        "- Re-run `python -m scripts.train_model` and confirm the audit files refresh successfully.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    project_root: Path,
    commodity_name: str = "brent",
    external_feature_commodities: list[str] | None = None,
    fill_method: str | None = None,
) -> dict[str, Any]:
    config = load_config(project_root)
    fill_method = fill_method or config["defaults"]["external_fill_method"]
    commodity_definition = get_commodity_definition(project_root, commodity_name)
    dataset_path = project_root / "data" / "raw" / commodity_definition.local_filename
    artifacts_dir = project_root / "artifacts"
    audits_dir = project_root / "audits"

    download_metadata = download_dataset(commodity_definition.source_url, dataset_path)
    rows = load_rows(dataset_path, commodity_definition)

    external_feature_commodities = [
        external_name for external_name in (external_feature_commodities or []) if external_name != commodity_name
    ]
    external_feature_lookup: dict[str, list[float]] | None = None
    external_feature_names: list[str] = []
    external_dataset_audits: list[dict[str, Any]] = []
    external_alignment_start = ""
    if external_feature_commodities:
        external_feature_result = build_external_feature_lookup(
            project_root,
            [row.date for row in rows],
            external_feature_commodities,
            fill_method=fill_method,
        )
        external_alignment_start = external_feature_result["available_from"]
        external_feature_lookup = external_feature_result["aligned_by_date"]
        external_feature_names = external_feature_result["feature_names"]
        external_dataset_audits = external_feature_result["dataset_audits"]
        if external_alignment_start:
            rows = [row for row in rows if row.date >= external_alignment_start]
            external_feature_lookup = {
                target_date: feature_values
                for target_date, feature_values in external_feature_lookup.items()
                if target_date >= external_alignment_start
            }
        for target_date, feature_values in external_feature_lookup.items():
            if any(math.isnan(value) for value in feature_values):
                raise ValueError(f"External feature alignment produced NaN values for {target_date}.")

    samples = build_samples(rows, external_feature_lookup)
    train_samples, test_samples = split_samples(samples)

    alpha = commodity_definition.model_hyperparameters.get("alpha", 1.0)
    weights = fit_ridge_regression(train_samples, alpha=alpha)
    test_metrics, predictions = evaluate(weights, test_samples)
    prediction_summary = latest_prediction(weights, rows, external_feature_lookup)

    feature_names = FEATURE_NAMES + external_feature_names
    dataset_audit_content = data_audit(
        rows,
        dataset_path,
        project_root,
        commodity_name,
        commodity_definition,
        download_metadata,
        external_dataset_audits,
        external_alignment_start,
    )
    model_audit_content = {
        "trained_at": utc_now(),
        "commodity": commodity_name,
        "commodity_display_name": commodity_definition.display_name,
        "feature_names": feature_names,
        "external_features": external_feature_commodities,
        "fill_method": fill_method,
        "model_hyperparameters": {"alpha": alpha},
        "coefficients": {
            "intercept": round_float(weights[0]),
            **{name: round_float(weight) for name, weight in zip(feature_names, weights[1:])},
        },
        "significant_features": significant_features(feature_names, weights),
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "test_metrics": test_metrics,
        "latest_prediction": prediction_summary,
    }

    write_json(artifacts_dir / "model.json", model_audit_content)
    write_predictions_csv(artifacts_dir / "test_predictions.csv", predictions)
    write_json(audits_dir / "data_audit.json", dataset_audit_content)
    write_json(audits_dir / "model_audit.json", model_audit_content)
    write_manual_audit(audits_dir / "manual_verification.md", dataset_audit_content, model_audit_content, prediction_summary)

    return {
        "dataset_audit": dataset_audit_content,
        "model_audit": model_audit_content,
    }
