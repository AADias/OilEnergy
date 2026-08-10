from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


@dataclass(frozen=True)
class PriceRow:
    date: str
    price: float


@dataclass(frozen=True)
class CommodityDefinition:
    key: str
    display_name: str
    source_type: str
    source_url: str
    local_filename: str
    date_column: str
    price_column: str
    date_formats: list[str]
    missing_values: list[str] = field(default_factory=list)
    feature_set: list[str] = field(default_factory=list)
    model_hyperparameters: dict[str, float] = field(default_factory=dict)
    proxy_note: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_for_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def config_path(project_root: Path) -> Path:
    return project_root / "config.yaml"


def load_config(project_root: Path) -> dict[str, Any]:
    return json.loads(config_path(project_root).read_text(encoding="utf-8"))


def commodity_definitions(project_root: Path) -> dict[str, CommodityDefinition]:
    definitions: dict[str, CommodityDefinition] = {}
    for key, definition in load_config(project_root)["commodities"].items():
        definitions[key] = CommodityDefinition(key=key, **definition)
    return definitions


def get_commodity_definition(project_root: Path, commodity_name: str) -> CommodityDefinition:
    definitions = commodity_definitions(project_root)
    if commodity_name not in definitions:
        available = ", ".join(sorted(definitions))
        raise KeyError(f"Unknown commodity '{commodity_name}'. Available commodities: {available}")
    return definitions[commodity_name]


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


def normalize_date(value: str, date_formats: list[str]) -> str:
    cleaned = value.strip()
    for date_format in date_formats:
        try:
            return datetime.strptime(cleaned, date_format).date().isoformat()
        except ValueError:
            continue
    return datetime.fromisoformat(cleaned).date().isoformat()


def load_rows(csv_path: Path, definition: CommodityDefinition) -> list[PriceRow]:
    rows: list[PriceRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_price = (row.get(definition.price_column) or "").strip()
            if raw_price in definition.missing_values:
                continue
            rows.append(
                PriceRow(
                    date=normalize_date(row[definition.date_column], definition.date_formats),
                    price=float(raw_price),
                )
            )
    rows.sort(key=lambda row: row.date)
    return rows
