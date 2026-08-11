"""commodities.py — Commodity definitions and data loading for the OilEnergy pipeline.

Supports multiple energy commodities (oil, gas) with a focus on Qatar and the
Middle East energy sector. Data sources are tried in priority order: FRED,
Yahoo Finance, direct CSV URL.

Data lineage is documented for every commodity so users can verify the source
of each price series.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request

# ---------------------------------------------------------------------------
# Commodity definitions
# ---------------------------------------------------------------------------

COMMODITIES: dict[str, dict[str, Any]] = {
    "brent": {
        "name": "Brent Crude Oil",
        "type": "oil",
        "region": "global",
        "primary_source": "csv",
        "csv_url": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
        "fred_series": "DCOILBRENTEU",
        "price_column": "Price",
        "date_column": "Date",
        "description": "Brent crude oil — global benchmark used for ~2/3 of world oil trade.",
        "data_lineage": "datasets/oil-prices GitHub repository (original source: EIA/ICE)",
        "is_proxy": False,
    },
    "wti": {
        "name": "WTI Crude Oil",
        "type": "oil",
        "region": "us",
        "primary_source": "fred",
        "fred_series": "DCOILWTICO",
        "price_column": "value",
        "date_column": "date",
        "description": "West Texas Intermediate crude oil — US benchmark.",
        "data_lineage": "FRED (Federal Reserve Economic Data) series DCOILWTICO — EIA source",
        "is_proxy": False,
    },
    "henry_hub": {
        "name": "Henry Hub Natural Gas",
        "type": "gas",
        "region": "us",
        "primary_source": "fred",
        "fred_series": "DHHNGSP",
        "price_column": "value",
        "date_column": "date",
        "description": "Henry Hub spot natural gas price — US benchmark for gas markets.",
        "data_lineage": "FRED series DHHNGSP — EIA/Henry Hub spot price",
        "is_proxy": False,
    },
    "qatar_lng": {
        "name": "Qatar LNG (Henry Hub Proxy)",
        "type": "gas",
        "region": "middle_east",
        "primary_source": "fred",
        "fred_series": "DHHNGSP",
        "price_column": "value",
        "date_column": "date",
        "description": (
            "Qatar LNG price proxy. Official QP pricing is not publicly available "
            "in machine-readable form; Henry Hub is used as a directional proxy. "
            "Qatar is the world's largest LNG exporter. Replace with contracted "
            "QP pricing data for production use."
        ),
        "data_lineage": (
            "FRED series DHHNGSP (Henry Hub) used as Qatar LNG proxy. "
            "Official Qatar energy data: https://www.qatarenergy.qa "
            "OPEC Qatar data: https://www.opec.org/opec_web/en/data_graphs/40.htm"
        ),
        "is_proxy": True,
        "proxy_for": "Qatar LNG",
    },
    "opec_basket": {
        "name": "OPEC Reference Basket",
        "type": "oil",
        "region": "middle_east",
        "primary_source": "fred",
        "fred_series": "DCOILBRENTEU",
        "price_column": "value",
        "date_column": "date",
        "description": (
            "OPEC Reference Basket — average of OPEC member crude prices, "
            "including QatarMarine. Brent is used as a proxy when official "
            "OPEC basket data is unavailable."
        ),
        "data_lineage": (
            "FRED series DCOILBRENTEU (Brent) used as OPEC basket proxy. "
            "Official OPEC basket: https://www.opec.org/opec_web/en/data_graphs/40.htm"
        ),
        "is_proxy": True,
        "proxy_for": "OPEC Reference Basket",
    },
}


# ---------------------------------------------------------------------------
# FRED data fetcher (free, no API key required for daily series)
# ---------------------------------------------------------------------------

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch bytes from a URL with a browser-like User-Agent header."""
    req = Request(url, headers={"User-Agent": "OilEnergy/1.0 (energy forecasting research)"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PriceRow:
    date: str
    price: float


@dataclass
class CommodityData:
    commodity_key: str
    commodity_name: str
    source_url: str
    source_type: str  # "fred" | "csv" | "cache"
    rows: list[PriceRow] = field(default_factory=list)
    download_metadata: dict[str, Any] = field(default_factory=dict)
    data_lineage: str = ""
    is_proxy: bool = False
    proxy_for: str | None = None
    canonical_series_id: str = ""


def fetch_fred_series(series_id: str) -> tuple[bytes, str]:
    """Download a FRED CSV series. Returns (content_bytes, url)."""
    url = FRED_BASE_URL.format(series_id=series_id)
    content = _fetch_url(url)
    return content, url


def parse_fred_csv(content: bytes) -> list[PriceRow]:
    """Parse FRED CSV format (DATE, VALUE) into PriceRow list, skipping missing values."""
    rows: list[PriceRow] = []
    lines = content.decode("utf-8").splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        value_str = row.get("VALUE", row.get("value", "")).strip()
        date_str = row.get("DATE", row.get("date", "")).strip()
        if not date_str or not value_str or value_str == ".":
            continue
        try:
            rows.append(PriceRow(date=date_str, price=float(value_str)))
        except ValueError:
            continue
    return rows


def parse_legacy_csv(content: bytes, date_col: str, price_col: str) -> list[PriceRow]:
    """Parse legacy CSV (arbitrary date/price columns) into PriceRow list."""
    rows: list[PriceRow] = []
    lines = content.decode("utf-8").splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        try:
            rows.append(PriceRow(date=row[date_col].strip(), price=float(row[price_col].strip())))
        except (KeyError, ValueError):
            continue
    return rows


def _canonical_series_id(commodity_key: str, config: dict[str, Any]) -> str:
    if config.get("fred_series"):
        return f"fred:{config['fred_series']}"
    if config.get("csv_url"):
        return f"csv:{config['csv_url']}"
    return f"commodity:{commodity_key}"


def load_commodity(
    commodity_key: str,
    cache_dir: Optional[Path] = None,
    allow_demo_fallback: bool = False,
) -> CommodityData:
    """Load price data for a named commodity.

    Tries sources in priority order:
      1. Cache on disk (if cache_dir provided and file exists)
      2. Primary source (FRED or CSV URL)
      3. Fallback between FRED / CSV as needed

    Raises RuntimeError if all sources fail.
    """
    if commodity_key not in COMMODITIES:
        available = ", ".join(sorted(COMMODITIES.keys()))
        raise ValueError(f"Unknown commodity '{commodity_key}'. Available: {available}")

    config = COMMODITIES[commodity_key]
    cache_path = (cache_dir / f"{commodity_key}.csv") if cache_dir else None

    # 1. Try cache
    if cache_path and cache_path.exists():
        content = cache_path.read_bytes()
        if config.get("primary_source") == "csv":
            rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
        else:
            rows = parse_fred_csv(content)
        if rows:
            return CommodityData(
                commodity_key=commodity_key,
                commodity_name=config["name"],
                source_url=str(cache_path),
                source_type="cache",
                rows=rows,
                download_metadata={"loaded_from_cache": str(cache_path), "loaded_at": _utc_now()},
                data_lineage=config.get("data_lineage", ""),
                is_proxy=bool(config.get("is_proxy", False)),
                proxy_for=config.get("proxy_for"),
                canonical_series_id=_canonical_series_id(commodity_key, config),
            )

    errors: list[str] = []

    # 2. Primary source
    try:
        result = _try_primary_source(commodity_key, config, cache_path)
        if result:
            return result
    except Exception as exc:
        errors.append(f"primary ({config.get('primary_source', '?')}): {exc}")

    # 3. Fallback: if primary was FRED, try CSV; if primary was CSV, try FRED
    try:
        result = _try_fallback_source(commodity_key, config, cache_path, errors)
        if result:
            return result
    except Exception as exc:
        errors.append(f"fallback: {exc}")

    # 4. Optional explicit fallback: use local brent.csv as a shape-compatible proxy (demo/offline mode)
    if allow_demo_fallback:
        try:
            result = _try_local_brent_fallback(commodity_key, config, cache_dir)
            if result:
                import sys
                print(
                    f"[WARNING] Could not reach network data sources for '{commodity_key}'. "
                    f"Using local Brent data as a demo fallback. "
                    f"Errors: {'; '.join(errors)}",
                    file=sys.stderr,
                )
                return result
        except Exception as exc:
            errors.append(f"brent_fallback: {exc}")

    raise RuntimeError(
        f"Failed to load commodity '{commodity_key}'. Tried: {'; '.join(errors)}. "
        "Re-run with '--allow-demo-fallback' (CLI) or allow_demo_fallback=True (Python API) "
        "to allow local Brent substitution in demo mode."
    )


def _try_local_brent_fallback(
    key: str, config: dict[str, Any], cache_dir: Optional[Path]
) -> Optional[CommodityData]:
    """Last-resort fallback: use locally cached brent.csv as a shape-compatible proxy.

    Only used when all network sources fail (e.g., offline or sandboxed environment).
    The data lineage is updated to clearly document this substitution.
    """
    if cache_dir is None:
        return None
    for candidate in ["brent.csv", "brent-daily.csv"]:
        candidate_path = cache_dir / candidate
        if candidate_path.exists():
            content = candidate_path.read_bytes()
            rows = parse_legacy_csv(content, "Date", "Price")
            if rows:
                return CommodityData(
                    commodity_key=key,
                    commodity_name=config["name"],
                    source_url=str(candidate_path),
                    source_type="brent_fallback",
                    rows=rows,
                    download_metadata={
                        "loaded_from_cache": str(candidate_path),
                        "loaded_at": _utc_now(),
                        "warning": (
                            "All network sources unavailable. Using local Brent CSV as "
                            "shape-compatible fallback. Price LEVELS are Brent oil, not "
                            f"{config['name']}. Direction signals remain valid as a demo."
                        ),
                    },
                    data_lineage=(
                        config.get("data_lineage", "")
                        + f" [FALLBACK: using local Brent CSV because {config.get('fred_series', 'FRED')} "
                        "was unreachable. Replace with real data for production use.]"
                    ),
                    is_proxy=True,
                    proxy_for=config["name"],
                    canonical_series_id="demo:brent-fallback",
                )
    return None


def _try_primary_source(
    key: str, config: dict[str, Any], cache_path: Optional[Path]
) -> Optional[CommodityData]:
    if config.get("primary_source") == "csv" and config.get("csv_url"):
        url = config["csv_url"]
        content = _fetch_url(url)
        rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
        if rows:
            _save_cache(cache_path, content)
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="csv",
                rows=rows,
                download_metadata={
                    "downloaded_at": _utc_now(),
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "csv_url",
                },
                data_lineage=config.get("data_lineage", ""),
                is_proxy=bool(config.get("is_proxy", False)),
                proxy_for=config.get("proxy_for"),
                canonical_series_id=_canonical_series_id(key, config),
            )
    elif config.get("fred_series"):
        content, url = fetch_fred_series(config["fred_series"])
        rows = parse_fred_csv(content)
        if rows:
            _save_cache(cache_path, content)
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="fred",
                rows=rows,
                download_metadata={
                    "downloaded_at": _utc_now(),
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "FRED",
                    "series_id": config["fred_series"],
                },
                data_lineage=config.get("data_lineage", ""),
                is_proxy=bool(config.get("is_proxy", False)),
                proxy_for=config.get("proxy_for"),
                canonical_series_id=_canonical_series_id(key, config),
            )
    return None


def _try_fallback_source(
    key: str, config: dict[str, Any], cache_path: Optional[Path], errors: list[str]
) -> Optional[CommodityData]:
    """Try the alternate source (FRED if primary was CSV, or CSV if primary was FRED)."""
    if config.get("primary_source") == "csv" and config.get("fred_series"):
        content, url = fetch_fred_series(config["fred_series"])
        rows = parse_fred_csv(content)
        if rows:
            _save_cache(cache_path, content)
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="fred",
                rows=rows,
                download_metadata={
                    "downloaded_at": _utc_now(),
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "FRED (fallback)",
                    "series_id": config["fred_series"],
                },
                data_lineage=config.get("data_lineage", "") + " [loaded via FRED fallback]",
                is_proxy=bool(config.get("is_proxy", False)),
                proxy_for=config.get("proxy_for"),
                canonical_series_id=_canonical_series_id(key, config),
            )
    elif config.get("primary_source") == "fred" and config.get("csv_url"):
        url = config["csv_url"]
        content = _fetch_url(url)
        rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
        if rows:
            _save_cache(cache_path, content)
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="csv",
                rows=rows,
                download_metadata={
                    "downloaded_at": _utc_now(),
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "csv_url (fallback)",
                },
                data_lineage=config.get("data_lineage", "") + " [loaded via CSV fallback]",
                is_proxy=bool(config.get("is_proxy", False)),
                proxy_for=config.get("proxy_for"),
                canonical_series_id=_canonical_series_id(key, config),
            )
    return None


def _save_cache(cache_path: Optional[Path], content: bytes) -> None:
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)


def _relativize_path(value: str, project_root: Path) -> str:
    """Return a project-relative path string if value is an absolute local path; else return as-is."""
    try:
        p = Path(value)
        if p.is_absolute():
            return str(p.relative_to(project_root))
    except (ValueError, TypeError):
        pass
    return value


def commodity_data_audit(data: CommodityData, project_root: Path) -> dict[str, Any]:
    """Build a data lineage/audit record for a loaded commodity."""
    prices = [r.price for r in data.rows]
    source_url = _relativize_path(data.source_url, project_root)
    # Normalize any absolute paths that appear inside download_metadata
    metadata = {
        k: _relativize_path(v, project_root) if isinstance(v, str) else v
        for k, v in data.download_metadata.items()
    }
    return {
        "commodity_key": data.commodity_key,
        "commodity_name": data.commodity_name,
        "source_url": source_url,
        "source_type": data.source_type,
        "data_lineage": data.data_lineage,
        "is_proxy_series": data.is_proxy,
        "proxy_for": data.proxy_for,
        "canonical_series_id": data.canonical_series_id,
        **metadata,
        "row_count": len(data.rows),
        "date_range": {
            "start": data.rows[0].date if data.rows else None,
            "end": data.rows[-1].date if data.rows else None,
        },
        "price_summary": {
            "minimum": round(min(prices), 4) if prices else None,
            "maximum": round(max(prices), 4) if prices else None,
            "average": round(sum(prices) / len(prices), 4) if prices else None,
        },
        "head": [asdict(r) for r in data.rows[:3]],
        "tail": [asdict(r) for r in data.rows[-3:]],
    }


def list_commodities() -> list[dict[str, Any]]:
    """Return a summary list of all configured commodities."""
    return [
        {
            "key": key,
            "name": cfg["name"],
            "type": cfg["type"],
            "region": cfg["region"],
            "is_proxy": bool(cfg.get("is_proxy", False)),
            "description": cfg["description"],
        }
        for key, cfg in COMMODITIES.items()
    ]


def list_commodities_by_type(category: str) -> list[dict[str, Any]]:
    category_norm = category.strip().lower()
    return [item for item in list_commodities() if item["type"] == category_norm]
