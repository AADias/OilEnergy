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
]


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


def build_samples(rows: list[PriceRow]) -> list[Sample]:
    samples: list[Sample] = []
    for index in range(10, len(rows) - 1):
        current_price = rows[index].price
        samples.append(
            Sample(
                date=rows[index].date,
                current_price=current_price,
                target_price=rows[index + 1].price,
                features=build_features(rows, index),
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


def latest_prediction(weights: list[float], rows: list[PriceRow]) -> dict[str, Any]:
    current_index = len(rows) - 1
    current_price = rows[current_index].price
    features = build_features(rows, current_index)
    predicted_price = predict(weights, features)
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
) -> None:
    ensure_directory(path.parent)
    lines = [
        "# Manual Verification Audit",
        "",
        "## Dataset checks",
        f"- Source URL: {dataset_audit_content['source_url']}",
        f"- Local file: {dataset_audit_content['local_dataset_path']}",
        f"- SHA256: {dataset_audit_content['sha256']}",
        f"- Rows: {dataset_audit_content['row_count']}",
        f"- Date range: {dataset_audit_content['date_range']['start']} to {dataset_audit_content['date_range']['end']}",
        "",
        "## Model checks",
        f"- Trained at: {model_audit_content['trained_at']}",
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
        "## Manual verification steps",
        "- Open the raw CSV and confirm the first and last rows match the data audit JSON.",
        "- Recompute the SHA256 of the raw CSV and confirm it matches the recorded digest.",
        "- Open the predictions CSV and confirm dates are in chronological order.",
        "- Spot-check that predicted_direction_up matches whether predicted_next_price is greater than or equal to current_price.",
        "- Re-run `PYTHONPATH=src python3 scripts/train_model.py` and confirm the audit files refresh successfully.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(project_root: Path) -> dict[str, Any]:
    dataset_path = project_root / "data" / "raw" / "brent-daily.csv"
    artifacts_dir = project_root / "artifacts"
    audits_dir = project_root / "audits"

    download_metadata = download_dataset(DATASET_URL, dataset_path)
    rows = load_rows(dataset_path)
    samples = build_samples(rows)
    train_samples, test_samples = split_samples(samples)

    weights = fit_ridge_regression(train_samples, alpha=1.0)
    test_metrics, predictions = evaluate(weights, test_samples)
    prediction_summary = latest_prediction(weights, rows)

    dataset_audit_content = data_audit(rows, dataset_path, project_root, DATASET_URL, download_metadata)
    model_audit_content = {
        "trained_at": utc_now(),
        "feature_names": FEATURE_NAMES,
        "coefficients": {
            "intercept": round_float(weights[0]),
            **{name: round_float(weight) for name, weight in zip(FEATURE_NAMES, weights[1:])},
        },
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
