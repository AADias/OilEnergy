"""external_features.py — Enrich the pipeline with correlated commodity prices.

Aligns external commodity price series to the dates of the target commodity,
then generates lag and rolling-mean features from each external series.
These features supplement the base time-series features in pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .commodities import PriceRow, load_commodity, CommodityData


@dataclass
class ExternalFeatureSet:
    """Holds aligned external price data and the feature names generated from it."""
    commodity_key: str
    commodity_name: str
    # date -> price for fast lookup
    price_map: dict[str, float]
    feature_names: list[str]
    # Pre-sorted date list to avoid re-sorting on every sample
    sorted_dates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sorted_dates:
            self.sorted_dates = sorted(self.price_map.keys())


def _build_external_feature_names(key: str) -> list[str]:
    """Feature names for an external commodity: lag_1 and mean_3."""
    return [f"{key}_lag_1", f"{key}_mean_3"]


def build_external_features(
    date: str,
    price_map: dict[str, float],
    all_dates: list[str],
) -> list[float]:
    """Build lag_1 and mean_3 features for an external series at a given date.

    If the date is not present in the external series, uses the nearest earlier value.
    Returns zeros if no data is available.
    """
    # Build sorted date list once per call (or caller caches)
    date_index = {d: i for i, d in enumerate(all_dates)}
    if date not in date_index:
        return [0.0, 0.0]

    idx = date_index[date]

    def _price_at(offset: int) -> float | None:
        i = idx - offset
        if i < 0:
            return None
        d = all_dates[i]
        return price_map.get(d)

    p1 = _price_at(1)
    p2 = _price_at(2)
    p3 = _price_at(3)

    lag_1 = p1 if p1 is not None else 0.0
    values = [v for v in [p1, p2, p3] if v is not None]
    mean_3 = sum(values) / len(values) if values else 0.0

    return [lag_1, mean_3]


def load_external_feature_sets(
    commodity_keys: list[str],
    cache_dir: Path | None = None,
) -> tuple[list[ExternalFeatureSet], list[str]]:
    """Load price data for each commodity key and build ExternalFeatureSet objects.

    Returns:
        feature_sets: list of ExternalFeatureSet (one per successfully loaded commodity)
        errors: list of error messages for commodities that failed to load
    """
    feature_sets: list[ExternalFeatureSet] = []
    errors: list[str] = []

    for key in commodity_keys:
        try:
            data: CommodityData = load_commodity(key, cache_dir=cache_dir)
            price_map = {row.date: row.price for row in data.rows}
            feature_sets.append(
                ExternalFeatureSet(
                    commodity_key=key,
                    commodity_name=data.commodity_name,
                    price_map=price_map,
                    feature_names=_build_external_feature_names(key),
                )
            )
        except Exception as exc:
            errors.append(f"{key}: {exc}")

    return feature_sets, errors


def enrich_features(
    date: str,
    base_features: list[float],
    base_feature_names: list[str],
    feature_sets: list[ExternalFeatureSet],
) -> tuple[list[float], list[str]]:
    """Append external commodity features to the base feature vector.

    Args:
        date: The date for which to build features (YYYY-MM-DD).
        base_features: The existing feature values.
        base_feature_names: Names corresponding to base_features.
        feature_sets: External commodity feature sets.

    Returns:
        enriched_features: base_features + external features
        enriched_names: base_feature_names + external feature names
    """
    enriched = list(base_features)
    enriched_names = list(base_feature_names)

    for fs in feature_sets:
        ext_values = build_external_features(date, fs.price_map, fs.sorted_dates)
        enriched.extend(ext_values)
        enriched_names.extend(fs.feature_names)

    return enriched, enriched_names
