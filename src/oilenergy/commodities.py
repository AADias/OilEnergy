"""commodities.py — Commodity definitions and data loading for the OilEnergy pipeline.

Supports multiple energy commodities (oil, gas) with a focus on Qatar and the
Middle East energy sector.

Data freshness policy
---------------------
By default the loader attempts a **live refresh** from the declared source before
falling back to a validated cache.  Pass ``offline=True`` to skip network access
and use the cache only.

Cache integrity
---------------
Every cached CSV is accompanied by a JSON sidecar (``<key>.cache_meta.json``)
that records the commodity key, canonical series ID, source type, retrieval
timestamp, and proxy/demo flags.  On read the sidecar is validated; any legacy
cache without a matching sidecar, or with a mismatched key/series, is rejected
and will **not** silently serve a different commodity.

Data lineage is documented for every commodity so users can verify the source
of each price series.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import date as _date, datetime, timezone
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
    source_type: str  # "fred" | "csv" | "cache" | "demo" | "proxy"
    rows: list[PriceRow] = field(default_factory=list)
    download_metadata: dict[str, Any] = field(default_factory=dict)
    data_lineage: str = ""
    # Freshness fields
    source_mode: str = "unknown"      # "live" | "cache" | "demo" | "proxy" | "unavailable"
    retrieved_at: str = ""            # ISO-8601 UTC timestamp of the data retrieval
    latest_obs_date: str = ""         # Date of the last observation in the series (YYYY-MM-DD)
    staleness_days: int = -1          # Calendar days between latest_obs_date and today
    refresh_warning: str = ""         # Non-empty when refresh failed or data is stale


def _compute_staleness(latest_date_str: str) -> int:
    """Return calendar days between *latest_date_str* (YYYY-MM-DD) and today (UTC).

    Returns -1 if the date cannot be parsed.
    """
    try:
        latest = _date.fromisoformat(latest_date_str[:10])
        today = datetime.now(timezone.utc).date()
        return (today - latest).days
    except (ValueError, TypeError):
        return -1


def _load_cache_meta(meta_path: Path) -> Optional[dict[str, Any]]:
    """Load a JSON sidecar file; return None if missing or unreadable."""
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _validate_cache_meta(meta: Optional[dict[str, Any]], commodity_key: str, config: dict[str, Any]) -> Optional[str]:
    """Return an error string if the metadata does not match the expected commodity, else None."""
    if meta is None:
        return "no sidecar metadata (legacy cache)"
    stored_key = meta.get("commodity_key")
    stored_series = meta.get("canonical_series_id")
    expected_series = config.get("fred_series") or config.get("csv_url", "")
    if stored_key != commodity_key:
        return f"key mismatch: cache has '{stored_key}', expected '{commodity_key}'"
    if stored_series and expected_series and stored_series != expected_series:
        return f"series mismatch: cache has '{stored_series}', expected '{expected_series}'"
    return None


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


def load_commodity(
    commodity_key: str,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
) -> CommodityData:
    """Load price data for a named commodity.

    Default behaviour (``offline=False``):
      1. Attempt a live refresh from the declared remote source.
      2. If the refresh fails, fall back to a **validated** cache (requires a
         matching ``.cache_meta.json`` sidecar; legacy bare CSV files without
         metadata are rejected for non-Brent commodities to prevent cross-
         commodity contamination).
      3. If no valid cache exists, raise RuntimeError with a clear message.

    Offline mode (``offline=True``):
      Skip network access entirely and use only the local cache.  If no valid
      cache exists the call fails clearly.

    Cache integrity
    ---------------
    Cached CSVs are accompanied by a ``<key>.cache_meta.json`` sidecar that
    records ``commodity_key``, ``canonical_series_id``, ``source_type``,
    ``retrieved_at``, and ``is_proxy``.  Any cache whose sidecar is absent or
    whose key/series fields do not match the current commodity definition is
    **rejected** — it will not be used, even as a last resort, for a different
    commodity (specifically: Brent CSV must never be served for qatar_lng).

    Raises RuntimeError if all sources fail.
    """
    if commodity_key not in COMMODITIES:
        available = ", ".join(sorted(COMMODITIES.keys()))
        raise ValueError(f"Unknown commodity '{commodity_key}'. Available: {available}")

    config = COMMODITIES[commodity_key]
    cache_path = (cache_dir / f"{commodity_key}.csv") if cache_dir else None
    meta_path = (cache_dir / f"{commodity_key}.cache_meta.json") if cache_dir else None

    errors: list[str] = []
    refresh_warning: str = ""

    # ------------------------------------------------------------------
    # LIVE REFRESH (unless offline mode)
    # ------------------------------------------------------------------
    if not offline:
        try:
            result = _try_primary_source(commodity_key, config, cache_path, meta_path)
            if result:
                return result
        except Exception as exc:
            errors.append(f"primary ({config.get('primary_source', '?')}): {exc}")

        # Fallback source (FRED↔CSV swap)
        try:
            result = _try_fallback_source(commodity_key, config, cache_path, meta_path, errors)
            if result:
                if errors:
                    refresh_warning = (
                        f"Primary source failed ({errors[0]}); loaded from fallback source."
                    )
                    result.refresh_warning = refresh_warning
                return result
        except Exception as exc:
            errors.append(f"fallback: {exc}")

    # ------------------------------------------------------------------
    # VALIDATED CACHE FALLBACK
    # ------------------------------------------------------------------
    if cache_path and cache_path.exists():
        meta = _load_cache_meta(meta_path) if meta_path else None
        meta_error = _validate_cache_meta(meta, commodity_key, config)
        if meta_error:
            errors.append(f"cache rejected ({meta_error})")
            print(
                f"[WARNING] Cache for '{commodity_key}' rejected: {meta_error}. "
                "Refusing to use a potentially mismatched cache.",
                file=sys.stderr,
            )
        else:
            content = cache_path.read_bytes()
            if config.get("primary_source") == "csv":
                rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
            else:
                rows = parse_fred_csv(content)
            if rows:
                latest = rows[-1].date
                staleness = _compute_staleness(latest)
                retrieved_at = (meta or {}).get("retrieved_at", _utc_now())
                warn = (
                    f"[STALE DATA] Using cached data (last observation: {latest}, "
                    f"{staleness} calendar days old). "
                    f"Refresh errors: {'; '.join(errors) or 'offline mode'}."
                ) if errors or offline else ""
                if warn:
                    print(warn, file=sys.stderr)
                return CommodityData(
                    commodity_key=commodity_key,
                    commodity_name=config["name"],
                    source_url=str(cache_path),
                    source_type="cache",
                    rows=rows,
                    download_metadata={
                        "loaded_from_cache": str(cache_path),
                        "loaded_at": _utc_now(),
                        **(meta or {}),
                    },
                    data_lineage=config.get("data_lineage", ""),
                    source_mode="cache",
                    retrieved_at=retrieved_at,
                    latest_obs_date=latest,
                    staleness_days=staleness,
                    refresh_warning=warn,
                )

    raise RuntimeError(
        f"Failed to load commodity '{commodity_key}'. "
        f"No valid cache and all remote sources failed. Tried: {'; '.join(errors) or 'offline mode, no cache'}. "
        "Run without --offline to attempt a live refresh, or supply a valid cache."
    )


def _try_primary_source(
    key: str, config: dict[str, Any], cache_path: Optional[Path], meta_path: Optional[Path]
) -> Optional[CommodityData]:
    now = _utc_now()
    if config.get("primary_source") == "csv" and config.get("csv_url"):
        url = config["csv_url"]
        content = _fetch_url(url)
        rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
        if rows:
            canonical = config.get("csv_url", "")
            _save_cache(cache_path, content)
            _save_cache_meta(meta_path, {
                "commodity_key": key,
                "canonical_series_id": canonical,
                "source_type": "csv",
                "source_url": url,
                "retrieved_at": now,
                "is_proxy": config.get("type") in {"proxy"},
            }) if meta_path else None
            latest = rows[-1].date
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="csv",
                rows=rows,
                download_metadata={
                    "downloaded_at": now,
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "csv_url",
                    "canonical_series_id": canonical,
                },
                data_lineage=config.get("data_lineage", ""),
                source_mode="live",
                retrieved_at=now,
                latest_obs_date=latest,
                staleness_days=_compute_staleness(latest),
            )
    elif config.get("fred_series"):
        content, url = fetch_fred_series(config["fred_series"])
        rows = parse_fred_csv(content)
        if rows:
            series_id = config["fred_series"]
            _save_cache(cache_path, content)
            if meta_path:
                _save_cache_meta(meta_path, {
                    "commodity_key": key,
                    "canonical_series_id": series_id,
                    "source_type": "fred",
                    "source_url": url,
                    "retrieved_at": now,
                    "is_proxy": key in {"qatar_lng", "opec_basket"},
                })
            latest = rows[-1].date
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="fred",
                rows=rows,
                download_metadata={
                    "downloaded_at": now,
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "FRED",
                    "series_id": series_id,
                    "canonical_series_id": series_id,
                },
                data_lineage=config.get("data_lineage", ""),
                source_mode="live",
                retrieved_at=now,
                latest_obs_date=latest,
                staleness_days=_compute_staleness(latest),
            )
    return None


def _try_fallback_source(
    key: str, config: dict[str, Any], cache_path: Optional[Path], meta_path: Optional[Path], errors: list[str]
) -> Optional[CommodityData]:
    """Try the alternate source (FRED if primary was CSV, or CSV if primary was FRED)."""
    now = _utc_now()
    if config.get("primary_source") == "csv" and config.get("fred_series"):
        series_id = config["fred_series"]
        content, url = fetch_fred_series(series_id)
        rows = parse_fred_csv(content)
        if rows:
            _save_cache(cache_path, content)
            if meta_path:
                _save_cache_meta(meta_path, {
                    "commodity_key": key,
                    "canonical_series_id": series_id,
                    "source_type": "fred",
                    "source_url": url,
                    "retrieved_at": now,
                    "is_proxy": key in {"qatar_lng", "opec_basket"},
                })
            latest = rows[-1].date
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="fred",
                rows=rows,
                download_metadata={
                    "downloaded_at": now,
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "FRED (fallback)",
                    "series_id": series_id,
                    "canonical_series_id": series_id,
                },
                data_lineage=config.get("data_lineage", "") + " [loaded via FRED fallback]",
                source_mode="live",
                retrieved_at=now,
                latest_obs_date=latest,
                staleness_days=_compute_staleness(latest),
            )
    elif config.get("primary_source") == "fred" and config.get("csv_url"):
        url = config["csv_url"]
        content = _fetch_url(url)
        rows = parse_legacy_csv(content, config["date_column"], config["price_column"])
        if rows:
            _save_cache(cache_path, content)
            if meta_path:
                _save_cache_meta(meta_path, {
                    "commodity_key": key,
                    "canonical_series_id": url,
                    "source_type": "csv",
                    "source_url": url,
                    "retrieved_at": now,
                    "is_proxy": key in {"qatar_lng", "opec_basket"},
                })
            latest = rows[-1].date
            return CommodityData(
                commodity_key=key,
                commodity_name=config["name"],
                source_url=url,
                source_type="csv",
                rows=rows,
                download_metadata={
                    "downloaded_at": now,
                    "sha256": _sha256(content),
                    "byte_count": len(content),
                    "source": "csv_url (fallback)",
                    "canonical_series_id": url,
                },
                data_lineage=config.get("data_lineage", "") + " [loaded via CSV fallback]",
                source_mode="live",
                retrieved_at=now,
                latest_obs_date=latest,
                staleness_days=_compute_staleness(latest),
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
        "source_mode": data.source_mode,
        "retrieved_at": data.retrieved_at,
        "latest_obs_date": data.latest_obs_date,
        "staleness_days": data.staleness_days,
        "refresh_warning": data.refresh_warning,
        "data_lineage": data.data_lineage,
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


def list_commodities() -> list[dict[str, str]]:
    """Return a summary list of all configured commodities."""
    return [
        {
            "key": key,
            "name": cfg["name"],
            "type": cfg["type"],
            "region": cfg["region"],
            "description": cfg["description"],
        }
        for key, cfg in COMMODITIES.items()
    ]
