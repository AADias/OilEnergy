from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from .commodities import download_dataset, get_commodity_definition, load_rows


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def interpolate_value(
    target_day: date,
    previous_day: date,
    previous_value: float,
    next_day: date,
    next_value: float,
) -> float:
    total_days = (next_day - previous_day).days
    if total_days <= 0:
        return previous_value
    elapsed_days = (target_day - previous_day).days
    ratio = elapsed_days / total_days
    return previous_value + ((next_value - previous_value) * ratio)


def align_price_series(source_dates: list[str], source_prices: list[float], target_dates: list[str], fill_method: str) -> list[float]:
    if not source_dates or not source_prices:
        raise ValueError("align_price_series requires at least one source observation.")

    observations = list(zip((parse_iso_date(value) for value in source_dates), source_prices))
    aligned_prices: list[float] = []
    observation_index = 0
    last_known_value = observations[0][1]

    for target_date in (parse_iso_date(value) for value in target_dates):
        while observation_index + 1 < len(observations) and observations[observation_index + 1][0] <= target_date:
            observation_index += 1
            last_known_value = observations[observation_index][1]

        current_day, current_value = observations[observation_index]
        next_observation = observations[observation_index + 1] if observation_index + 1 < len(observations) else None

        if fill_method == "interpolate" and next_observation and current_day < target_date < next_observation[0]:
            aligned_prices.append(
                interpolate_value(target_date, current_day, current_value, next_observation[0], next_observation[1])
            )
            continue

        if target_date < current_day:
            aligned_prices.append(current_value)
            continue

        aligned_prices.append(last_known_value)

    return aligned_prices


def compute_returns(prices: list[float]) -> list[float]:
    if not prices:
        return []
    returns = [0.0]
    for previous_price, current_price in zip(prices, prices[1:]):
        if previous_price == 0:
            returns.append(0.0)
            continue
        returns.append((current_price - previous_price) / previous_price)
    return returns


def feature_names_for_commodities(commodity_names: list[str]) -> list[str]:
    feature_names: list[str] = []
    for commodity_name in commodity_names:
        feature_names.extend([f"{commodity_name}_price", f"{commodity_name}_return"])
    return feature_names


def build_external_feature_lookup(
    project_root: Path,
    target_dates: list[str],
    commodity_names: list[str],
    fill_method: str = "forward_fill",
) -> dict[str, Any]:
    aligned_by_date = {target_date: [] for target_date in target_dates}
    dataset_audits: list[dict[str, Any]] = []

    for commodity_name in commodity_names:
        definition = get_commodity_definition(project_root, commodity_name)
        dataset_path = project_root / "data" / "raw" / definition.local_filename
        download_metadata = download_dataset(definition.source_url, dataset_path)
        rows = load_rows(dataset_path, definition)
        aligned_prices = align_price_series(
            [row.date for row in rows],
            [row.price for row in rows],
            target_dates,
            fill_method,
        )
        aligned_returns = compute_returns(aligned_prices)
        for target_date, aligned_price, aligned_return in zip(target_dates, aligned_prices, aligned_returns):
            aligned_by_date[target_date].extend([aligned_price, aligned_return])
        dataset_audits.append(
            {
                "commodity": commodity_name,
                "display_name": definition.display_name,
                "source_url": definition.source_url,
                "local_dataset_path": str(dataset_path.relative_to(project_root)),
                "row_count": len(rows),
                "date_range": {"start": rows[0].date, "end": rows[-1].date},
                "fill_method": fill_method,
                **download_metadata,
            }
        )

    return {
        "aligned_by_date": aligned_by_date,
        "feature_names": feature_names_for_commodities(commodity_names),
        "dataset_audits": dataset_audits,
    }
