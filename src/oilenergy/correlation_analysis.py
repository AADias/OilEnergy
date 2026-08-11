"""correlation_analysis.py — Analyze cross-commodity correlations for feature engineering.

Computes pairwise Pearson correlation coefficients between commodity price series,
aligns them to a common date index, and returns a matrix that downstream modules
use to decide which commodities to include as external features.

Duplicate-source safeguard
--------------------------
Commodities that share the same underlying data source (e.g. ``qatar_lng`` and
``henry_hub`` both use FRED/DHHNGSP; ``brent`` and ``opec_basket`` both use
FRED/DCOILBRENTEU) would produce artifically near-1.0 correlations because
they are fetched from the same series.  ``significant_partners`` excludes these
pairs automatically so they never contaminate external feature sets.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .commodities import CommodityData, PriceRow, load_commodity, COMMODITIES
from pathlib import Path


@dataclass
class CorrelationResult:
    commodity_a: str
    commodity_b: str
    pearson_r: float
    n_observations: int
    is_significant: bool  # |r| >= threshold

def _underlying_source_id(key: str) -> str:
    """Return a canonical identifier for the underlying data source of *key*.

    Two commodities with the same source ID share data; their correlation is
    not independently meaningful and must not be used as a cross-feature.
    """
    cfg = COMMODITIES.get(key, {})
    fred = cfg.get("fred_series")
    csv_url = cfg.get("csv_url")
    # Prefer FRED series ID as the canonical identifier
    return fred or csv_url or key


def shared_underlying_source(key_a: str, key_b: str) -> bool:
    """Return True when key_a and key_b share the same underlying data source."""
    return _underlying_source_id(key_a) == _underlying_source_id(key_b)



def _align_series(
    series_a: list[PriceRow], series_b: list[PriceRow]
) -> tuple[list[float], list[float]]:
    """Align two price series by date, returning only dates present in both."""
    map_b = {row.date: row.price for row in series_b}
    prices_a: list[float] = []
    prices_b: list[float] = []
    for row in series_a:
        if row.date in map_b:
            prices_a.append(row.price)
            prices_b.append(map_b[row.date])
    return prices_a, prices_b


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n < 3:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = statistics.pstdev(x)
    std_y = statistics.pstdev(y)
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)


def compute_correlation_matrix(
    commodity_keys: list[str],
    cache_dir: Path | None = None,
    threshold: float = 0.5,
) -> tuple[dict[tuple[str, str], CorrelationResult], list[str]]:
    """Compute pairwise correlations for a list of commodity keys.

    Returns:
        matrix: dict keyed by (key_a, key_b) -> CorrelationResult
        errors: list of commodities that failed to load (skipped)
    """
    loaded: dict[str, CommodityData] = {}
    errors: list[str] = []

    for key in commodity_keys:
        try:
            loaded[key] = load_commodity(key, cache_dir=cache_dir)
        except Exception as exc:
            errors.append(f"{key}: {exc}")

    matrix: dict[tuple[str, str], CorrelationResult] = {}
    keys = list(loaded.keys())

    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            prices_a, prices_b = _align_series(loaded[key_a].rows, loaded[key_b].rows)
            r = pearson_correlation(prices_a, prices_b)
            result = CorrelationResult(
                commodity_a=key_a,
                commodity_b=key_b,
                pearson_r=round(r, 6),
                n_observations=len(prices_a),
                is_significant=abs(r) >= threshold,
            )
            matrix[(key_a, key_b)] = result
            matrix[(key_b, key_a)] = CorrelationResult(
                commodity_a=key_b,
                commodity_b=key_a,
                pearson_r=result.pearson_r,
                n_observations=result.n_observations,
                is_significant=result.is_significant,
            )

    return matrix, errors


def correlation_matrix_to_dict(
    matrix: dict[tuple[str, str], CorrelationResult],
    commodity_keys: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Serialise correlation matrix to a JSON-friendly dict for auditing."""
    loaded_keys = list({k for pair in matrix.keys() for k in pair})
    rows: list[dict[str, Any]] = []
    for key_a in loaded_keys:
        for key_b in loaded_keys:
            if key_a == key_b:
                rows.append({"from": key_a, "to": key_b, "pearson_r": 1.0, "n_observations": None, "is_significant": True})
            elif (key_a, key_b) in matrix:
                r = matrix[(key_a, key_b)]
                rows.append({
                    "from": r.commodity_a,
                    "to": r.commodity_b,
                    "pearson_r": r.pearson_r,
                    "n_observations": r.n_observations,
                    "is_significant": r.is_significant,
                })
    return {
        "correlation_matrix": rows,
        "threshold_used": threshold,
        "notes": (
            "Pearson correlation on price levels. High correlation means the two "
            "series move together and may improve prediction when used as cross-features. "
            "All correlations should be verified against economic reasoning."
        ),
    }


def significant_partners(
    target_key: str,
    matrix: dict[tuple[str, str], CorrelationResult],
) -> list[str]:
    """Return keys of commodities significantly correlated with target_key.

    Excludes commodities that share the same underlying data source as
    *target_key* (e.g. qatar_lng vs henry_hub) to prevent duplicate-series
    contamination of external feature sets.
    """
    return [
        key_b
        for (key_a, key_b), result in matrix.items()
        if key_a == target_key
        and result.is_significant
        and key_b != target_key
        and not shared_underlying_source(target_key, key_b)
    ]
