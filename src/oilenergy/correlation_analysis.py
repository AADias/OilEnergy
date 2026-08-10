from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .commodities import download_dataset, get_commodity_definition, load_config, load_rows
from .external_features import align_price_series


def round_float(value: float, digits: int = 6) -> float:
    return round(value, digits)


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0:
        return 0.0
    return covariance / denominator


def rank_values(values: list[float]) -> list[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed_values):
        tie_end = index
        while tie_end + 1 < len(indexed_values) and indexed_values[tie_end + 1][1] == indexed_values[index][1]:
            tie_end += 1
        average_rank = (index + tie_end + 2) / 2
        for rank_index in range(index, tie_end + 1):
            original_index = indexed_values[rank_index][0]
            ranks[original_index] = average_rank
        index = tie_end + 1
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float:
    return pearson_correlation(rank_values(left), rank_values(right))


def compute_correlation_matrix(series_by_name: dict[str, list[float]], method: str) -> dict[str, dict[str, float]]:
    correlation_function = pearson_correlation if method == "pearson" else spearman_correlation
    matrix: dict[str, dict[str, float]] = {}
    commodity_names = list(series_by_name)
    for left_name in commodity_names:
        matrix[left_name] = {}
        for right_name in commodity_names:
            if left_name == right_name:
                matrix[left_name][right_name] = 1.0
                continue
            matrix[left_name][right_name] = round_float(
                correlation_function(series_by_name[left_name], series_by_name[right_name])
            )
    return matrix


def identify_high_correlation_pairs(
    pearson_matrix: dict[str, dict[str, float]],
    spearman_matrix: dict[str, dict[str, float]],
    threshold: float,
) -> list[dict[str, Any]]:
    commodity_names = list(pearson_matrix)
    pairs: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(commodity_names):
        for right_name in commodity_names[left_index + 1 :]:
            pearson_value = pearson_matrix[left_name][right_name]
            spearman_value = spearman_matrix[left_name][right_name]
            max_score = max(abs(pearson_value), abs(spearman_value))
            if max_score < threshold:
                continue
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                    "strength": round_float(max_score),
                }
            )
    return sorted(pairs, key=lambda pair: pair["strength"], reverse=True)


def feature_recommendations(
    target_commodity: str,
    pearson_matrix: dict[str, dict[str, float]],
    spearman_matrix: dict[str, dict[str, float]],
    threshold: float,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for commodity_name in pearson_matrix:
        if commodity_name == target_commodity:
            continue
        pearson_value = pearson_matrix[target_commodity][commodity_name]
        spearman_value = spearman_matrix[target_commodity][commodity_name]
        strongest_score = max(abs(pearson_value), abs(spearman_value))
        recommendations.append(
            {
                "commodity": commodity_name,
                "include_as_feature": strongest_score >= threshold,
                "pearson": pearson_value,
                "spearman": spearman_value,
                "reason": (
                    "Strong monotonic or linear relationship to the target commodity."
                    if strongest_score >= threshold
                    else "Useful for experimentation, but below the default high-correlation threshold."
                ),
            }
        )
    return sorted(
        recommendations,
        key=lambda recommendation: max(abs(recommendation["pearson"]), abs(recommendation["spearman"])),
        reverse=True,
    )


def analyze_correlations(
    project_root: Path,
    target_commodity: str = "brent",
    comparison_commodities: list[str] | None = None,
    fill_method: str | None = None,
) -> dict[str, Any]:
    config = load_config(project_root)
    threshold = config["defaults"]["correlation_threshold"]
    comparison_commodities = comparison_commodities or ["wti", "henry_hub", "qatar_lng"]
    fill_method = fill_method or config["defaults"]["external_fill_method"]

    target_definition = get_commodity_definition(project_root, target_commodity)
    target_dataset_path = project_root / "data" / "raw" / target_definition.local_filename
    target_download_metadata = download_dataset(target_definition.source_url, target_dataset_path)
    target_rows = load_rows(target_dataset_path, target_definition)
    target_dates = [row.date for row in target_rows]

    series_by_name: dict[str, list[float]] = {target_commodity: [row.price for row in target_rows]}
    dataset_audits = [
        {
            "commodity": target_commodity,
            "display_name": target_definition.display_name,
            "source_url": target_definition.source_url,
            "local_dataset_path": str(target_dataset_path.relative_to(project_root)),
            "row_count": len(target_rows),
            "date_range": {"start": target_rows[0].date, "end": target_rows[-1].date},
            **target_download_metadata,
        }
    ]

    for comparison_commodity in comparison_commodities:
        if comparison_commodity == target_commodity:
            continue
        definition = get_commodity_definition(project_root, comparison_commodity)
        dataset_path = project_root / "data" / "raw" / definition.local_filename
        download_metadata = download_dataset(definition.source_url, dataset_path)
        rows = load_rows(dataset_path, definition)
        aligned_prices = align_price_series(
            [row.date for row in rows],
            [row.price for row in rows],
            target_dates,
            fill_method,
        )
        series_by_name[comparison_commodity] = aligned_prices
        dataset_audits.append(
            {
                "commodity": comparison_commodity,
                "display_name": definition.display_name,
                "source_url": definition.source_url,
                "local_dataset_path": str(dataset_path.relative_to(project_root)),
                "row_count": len(rows),
                "date_range": {"start": rows[0].date, "end": rows[-1].date},
                "fill_method": fill_method,
                "proxy_note": definition.proxy_note,
                **download_metadata,
            }
        )

    pearson_matrix = compute_correlation_matrix(series_by_name, method="pearson")
    spearman_matrix = compute_correlation_matrix(series_by_name, method="spearman")
    recommendations = feature_recommendations(target_commodity, pearson_matrix, spearman_matrix, threshold)

    return {
        "target_commodity": target_commodity,
        "comparison_commodities": [name for name in series_by_name if name != target_commodity],
        "fill_method": fill_method,
        "correlation_threshold": threshold,
        "dataset_audits": dataset_audits,
        "pearson_correlation": pearson_matrix,
        "spearman_correlation": spearman_matrix,
        "high_correlation_pairs": identify_high_correlation_pairs(pearson_matrix, spearman_matrix, threshold),
        "feature_recommendations": recommendations,
        "visualization_recommendations": [
            "Use a heatmap for the Pearson matrix to spot tightly-linked commodities quickly.",
            "Plot Brent versus WTI as a scatter chart because those markets are typically the strongest pair.",
            "Overlay Qatar LNG proxy and Henry Hub on a normalized line chart to compare gas market divergence.",
        ],
    }


def write_correlation_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
