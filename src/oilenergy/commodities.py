from __future__ import annotations

import csv
import hashlib
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


def strip_inline_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, character in enumerate(line):
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def parse_yaml_scalar(value: str) -> Any:
    cleaned = value.strip()
    if cleaned == "":
        return ""
    if cleaned in {"null", "~"}:
        return None
    if cleaned in {"true", "True"}:
        return True
    if cleaned in {"false", "False"}:
        return False
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        return cleaned[1:-1]
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def parse_simple_yaml(text: str) -> dict[str, Any]:
    # This parser intentionally supports only the subset used by config.yaml:
    # nested mappings, scalar lists, quoted or plain scalars, and full-line/inline comments.
    # If future configuration needs richer YAML features, replace this helper with a dedicated parser.
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        uncommented = strip_inline_comment(raw_line).rstrip()
        if not uncommented.strip():
            continue
        indent = len(uncommented) - len(uncommented.lstrip(" "))
        lines.append((indent, uncommented.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if lines[index][1].startswith("- "):
            list_values: list[Any] = []
            while index < len(lines):
                current_indent, current_line = lines[index]
                if current_indent < indent or not current_line.startswith("- "):
                    break
                if current_indent != indent:
                    raise ValueError("Unsupported YAML indentation in list item.")
                item = current_line[2:].strip()
                index += 1
                if item:
                    list_values.append(parse_yaml_scalar(item))
                    continue
                if index >= len(lines) or lines[index][0] <= current_indent:
                    list_values.append({})
                    continue
                nested_value, index = parse_block(index, lines[index][0])
                list_values.append(nested_value)
            return list_values, index

        mapping_values: dict[str, Any] = {}
        while index < len(lines):
            current_indent, current_line = lines[index]
            if current_indent < indent or current_line.startswith("- "):
                break
            if current_indent != indent:
                raise ValueError("Unsupported YAML indentation in mapping entry.")
            key, separator, remainder = current_line.partition(":")
            if not separator:
                raise ValueError(f"Invalid YAML line: {current_line}")
            index += 1
            if remainder.strip():
                mapping_values[key.strip()] = parse_yaml_scalar(remainder.strip())
                continue
            if index < len(lines) and lines[index][0] > current_indent:
                nested_value, index = parse_block(index, lines[index][0])
                mapping_values[key.strip()] = nested_value
            else:
                mapping_values[key.strip()] = {}
        return mapping_values, index

    parsed, _ = parse_block(0, lines[0][0])
    return parsed


def load_config(project_root: Path) -> dict[str, Any]:
    return parse_simple_yaml(config_path(project_root).read_text(encoding="utf-8"))


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
