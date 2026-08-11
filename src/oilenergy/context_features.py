from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextSeries:
    category: str
    source_path: Path
    values_by_date: dict[str, float]
    feature_names: list[str]
    lineage: str
    sorted_dates: list[str] = field(default_factory=list)
    date_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sorted_dates:
            self.sorted_dates = sorted(self.values_by_date.keys())
        if not self.date_index:
            self.date_index = {d: i for i, d in enumerate(self.sorted_dates)}


def _feature_names(category: str) -> list[str]:
    return [f"{category}_lag_1", f"{category}_mean_3"]


def load_context_series(
    category: str,
    project_root: Path,
    context_data_dir: Path | None = None,
) -> tuple[ContextSeries | None, dict[str, Any]]:
    base_dir = context_data_dir or (project_root / "data" / "context")
    source_path = base_dir / f"{category}.csv"
    try:
        source_path_display = str(source_path.relative_to(project_root))
    except ValueError:
        source_path_display = str(source_path)
    availability: dict[str, Any] = {
        "category": category,
        "available": False,
        "source_path": source_path_display,
        "lineage": "User-provided local CSV input",
        "status": "missing_file",
    }
    if not source_path.exists():
        return None, availability

    values_by_date: dict[str, float] = {}
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date_value = (row.get("date") or "").strip()
            raw_value = (row.get("value") or "").strip()
            if not date_value or not raw_value:
                continue
            try:
                values_by_date[date_value] = float(raw_value)
            except ValueError:
                continue

    if not values_by_date:
        availability["status"] = "invalid_or_empty_csv"
        return None, availability

    availability["available"] = True
    availability["status"] = "loaded"
    try:
        lineage_path = str(source_path.relative_to(project_root))
    except ValueError:
        lineage_path = str(source_path)
    series = ContextSeries(
        category=category,
        source_path=source_path,
        values_by_date=values_by_date,
        feature_names=_feature_names(category),
        lineage=f"Local CSV ({lineage_path})",
    )
    return series, availability


def build_context_features(
    date: str,
    context_series: ContextSeries,
) -> list[float]:
    all_dates = context_series.sorted_dates
    if date in context_series.date_index:
        idx = context_series.date_index[date]
    else:
        insert_at = bisect_right(all_dates, date)
        idx = insert_at - 1
        if idx < 0:
            return [0.0, 0.0]

    def _value_at(offset: int) -> float | None:
        i = idx - offset
        if i < 0:
            return None
        d = all_dates[i]
        return context_series.values_by_date.get(d)

    v1 = _value_at(1)
    v2 = _value_at(2)
    v3 = _value_at(3)
    lag_1 = v1 if v1 is not None else 0.0
    values = [v for v in [v1, v2, v3] if v is not None]
    mean_3 = sum(values) / len(values) if values else 0.0
    return [lag_1, mean_3]
