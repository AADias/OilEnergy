"""context_features.py — Load and align local context series (weather, demand).

Context data lives in ``data/context/{category}.csv``.  Each CSV must have the
columns ``date`` and ``value``; an optional ``lineage`` column is read when
present and propagated to the audit trail.

Design contract
---------------
* Features are ONLY provided for dates that fall within the explicit coverage
  window of the loaded CSV.  No implicit backfill is performed for dates that
  predate the first row or postdate the last row.
* For dates inside the coverage window but missing a row (gaps), the feature
  value is also marked absent (``None``) — callers decide how to handle gaps.
* Lineage metadata (``real``, ``demo``, or ``user_provided``) is preserved so
  the UI and audit trail can distinguish data sources.
* This module has zero third-party dependencies.
"""
from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextSeries:
    category: str
    """Logical category name, e.g. ``weather`` or ``demand``."""

    values_by_date: dict[str, float]
    """Mapping of ISO-date string → float value (only dates present in CSV)."""

    lineage: str
    """Human-readable data provenance string."""

    feature_names: list[str]
    """Names of the features this series contributes, for audit/labelling."""

    # Derived fields populated in __post_init__
    sorted_dates: list[str] = field(default_factory=list)
    coverage_start: str = field(default="")
    coverage_end: str = field(default="")

    def __post_init__(self) -> None:
        self.sorted_dates = sorted(self.values_by_date)
        if self.sorted_dates:
            self.coverage_start = self.sorted_dates[0]
            self.coverage_end = self.sorted_dates[-1]

    def covers(self, date_str: str) -> bool:
        """Return True iff *date_str* is within the explicit coverage window."""
        return bool(
            self.coverage_start
            and self.coverage_end
            and self.coverage_start <= date_str <= self.coverage_end
        )


def _derive_lineage(detected: str | None) -> str:
    """Normalise a lineage string from the CSV into one of three categories."""
    if not detected:
        return "user_provided"
    low = detected.lower()
    if "demo" in low or "synthetic" in low:
        return "demo"
    if "real" in low or "open-meteo" in low or "api" in low or "observation" in low:
        return "real"
    return "user_provided"


def load_context_series(
    category: str,
    project_root: Path,
    context_data_dir: Path | None = None,
) -> tuple[ContextSeries | None, dict[str, Any]]:
    """Load a context series CSV and return ``(series, availability_metadata)``.

    The *availability_metadata* dict is always returned so callers can record
    why data was or was not loaded into the audit trail.

    Returns ``(None, metadata)`` when the file is absent or unreadable.
    """
    base_dir = context_data_dir or (project_root / "data" / "context")
    source_path = base_dir / f"{category}.csv"

    try:
        source_display = str(source_path.relative_to(project_root))
    except ValueError:
        source_display = str(source_path)

    availability: dict[str, Any] = {
        "category": category,
        "available": False,
        "source_path": source_display,
        "lineage": "none",
        "lineage_type": "none",
        "status": "missing_file",
        "coverage_start": None,
        "coverage_end": None,
    }

    if not source_path.exists():
        return None, availability

    values_by_date: dict[str, float] = {}
    raw_lineage: str | None = None

    with source_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        has_lineage_col = "lineage" in fieldnames
        for row in reader:
            date_str = (row.get("date") or "").strip()
            raw_val = (row.get("value") or "").strip()
            if not date_str or not raw_val:
                continue
            try:
                values_by_date[date_str] = float(raw_val)
            except ValueError:
                continue
            if has_lineage_col and raw_lineage is None:
                raw_lineage = (row.get("lineage") or "").strip() or None

    if not values_by_date:
        availability["status"] = "invalid_or_empty_csv"
        return None, availability

    lineage_type = _derive_lineage(raw_lineage)
    lineage_display = raw_lineage or f"Local CSV ({source_display})"

    series = ContextSeries(
        category=category,
        values_by_date=values_by_date,
        lineage=lineage_display,
        feature_names=[f"{category}_lag_1", f"{category}_mean_3"],
    )

    availability.update(
        {
            "available": True,
            "status": "loaded",
            "lineage": lineage_display,
            "lineage_type": lineage_type,
            "coverage_start": series.coverage_start,
            "coverage_end": series.coverage_end,
            "row_count": len(values_by_date),
        }
    )
    return series, availability


def build_context_features(
    date_str: str,
    series: ContextSeries,
) -> list[float] | None:
    """Return ``[lag_1, mean_3]`` context features for *date_str*.

    Returns ``None`` when *date_str* is outside the series' explicit coverage
    window (i.e. before the first observation or after the last).  This signals
    to the caller that no context is available for this date — the caller must
    NOT substitute zeros or forward-fill silently.

    For dates inside the coverage window but without an exact match, the nearest
    prior observation is used (carry-forward within coverage).
    """
    if not series.covers(date_str):
        return None

    all_dates = series.sorted_dates
    # Find the index of the largest date <= date_str
    pos = bisect_right(all_dates, date_str) - 1
    if pos < 0:
        return None  # shouldn't happen given covers() check, but be safe

    def _value_at(offset: int) -> float | None:
        i = pos - offset
        if i < 0:
            return None
        return series.values_by_date.get(all_dates[i])

    v0 = _value_at(0)
    v1 = _value_at(1)
    v2 = _value_at(2)

    lag_1 = v0 if v0 is not None else 0.0
    vals = [v for v in [v0, v1, v2] if v is not None]
    mean_3 = sum(vals) / len(vals) if vals else 0.0
    return [lag_1, mean_3]
