"""Tests for data-freshness, cache-integrity, and Qatar LNG proxy isolation.

All tests use only the standard library and mock/fake network behaviour as
needed — no third-party test framework required.

Run with:
    python -m unittest discover tests
or:
    python -m unittest tests.test_data_freshness
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oilenergy.commodities import (
    COMMODITIES,
    CommodityData,
    PriceRow,
    _compute_staleness,
    _validate_cache_meta,
    load_commodity,
    _utc_now,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fred_csv(series: list[tuple[str, str]]) -> bytes:
    """Build a minimal FRED-format CSV (DATE,VALUE)."""
    lines = ["DATE,VALUE"]
    for date_str, val in series:
        lines.append(f"{date_str},{val}")
    return "\n".join(lines).encode()


def _make_rows(n: int = 20, base_date: str = "2026-01-01", price: float = 50.0) -> list[PriceRow]:
    from datetime import date as _date, timedelta
    d = _date.fromisoformat(base_date)
    rows = []
    for i in range(n):
        rows.append(PriceRow(date=(d + timedelta(days=i)).isoformat(), price=round(price + i * 0.1, 2)))
    return rows


def _write_cache(cache_dir: Path, key: str, rows: list[PriceRow], write_meta: bool = True,
                 meta_override: dict | None = None) -> None:
    """Write a CSV cache and optional sidecar for a commodity."""
    csv_path = cache_dir / f"{key}.csv"
    config = COMMODITIES[key]
    is_fred = config.get("primary_source") == "fred"

    with csv_path.open("w", encoding="utf-8") as fh:
        if is_fred:
            fh.write("DATE,VALUE\n")
            for r in rows:
                fh.write(f"{r.date},{r.price}\n")
        else:
            fh.write("Date,Price\n")
            for r in rows:
                fh.write(f"{r.date},{r.price}\n")

    if write_meta:
        meta = meta_override or {
            "commodity_key": key,
            "canonical_series_id": config.get("fred_series") or config.get("csv_url", ""),
            "source_type": "fred" if is_fred else "csv",
            "source_url": "https://example.com",
            "retrieved_at": _utc_now(),
            "is_proxy": key in {"qatar_lng", "opec_basket"},
        }
        meta_path = cache_dir / f"{key}.cache_meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Live refresh updates a valid commodity cache
# ---------------------------------------------------------------------------

class TestLiveRefreshUpdatesCache(unittest.TestCase):
    """Verify that when offline=False a successful network fetch is returned
    (source_mode == 'live') even if a stale cache exists."""

    def test_live_refresh_preferred_over_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Write a stale cache (old date)
            stale_rows = _make_rows(5, base_date="2020-01-01")
            _write_cache(cache_dir, "brent", stale_rows)

            # Mock the network call to return fresher data
            legacy_csv_content = b"Date,Price\n2026-08-10,90.0\n2026-08-11,91.0\n"

            with patch("oilenergy.commodities._fetch_url", return_value=legacy_csv_content):
                data = load_commodity("brent", cache_dir=cache_dir, offline=False)

            self.assertEqual(data.source_mode, "live")
            self.assertEqual(data.latest_obs_date, "2026-08-11")

    def test_live_refresh_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            fresh_csv = b"Date,Price\n2026-08-11,92.0\n2026-08-12,93.0\n"

            with patch("oilenergy.commodities._fetch_url", return_value=fresh_csv):
                load_commodity("brent", cache_dir=cache_dir, offline=False)

            meta_path = cache_dir / "brent.cache_meta.json"
            self.assertTrue(meta_path.exists(), "Sidecar .cache_meta.json should be written after refresh")
            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["commodity_key"], "brent")
            self.assertIn("canonical_series_id", meta)
            self.assertIn("retrieved_at", meta)


# ---------------------------------------------------------------------------
# 2. --offline does not fetch
# ---------------------------------------------------------------------------

class TestOfflineMode(unittest.TestCase):
    """Verify that offline=True never calls the network."""

    def test_offline_uses_only_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            rows = _make_rows(5, base_date="2026-07-01")
            _write_cache(cache_dir, "brent", rows)

            with patch("oilenergy.commodities._fetch_url") as mock_fetch:
                data = load_commodity("brent", cache_dir=cache_dir, offline=True)
                mock_fetch.assert_not_called()

            self.assertEqual(data.source_mode, "cache")
            self.assertEqual(data.source_type, "cache")

    def test_offline_without_cache_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError) as ctx:
                load_commodity("brent", cache_dir=Path(tmpdir), offline=True)
            self.assertIn("offline", str(ctx.exception).lower())

    def test_offline_reports_stale_warning(self) -> None:
        """When offline=True with a stale cache, refresh_warning should mention staleness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_rows = _make_rows(5, base_date="2020-01-01")
            _write_cache(cache_dir, "brent", old_rows)

            data = load_commodity("brent", cache_dir=cache_dir, offline=True)

            # Should fall back to cache and flag stale
            self.assertEqual(data.source_mode, "cache")
            self.assertGreater(data.staleness_days, 100)


# ---------------------------------------------------------------------------
# 3. Stale cache warning and max-staleness enforcement
# ---------------------------------------------------------------------------

class TestStalenessCalculation(unittest.TestCase):

    def test_stale_days_computed_correctly(self) -> None:
        today = datetime.now(timezone.utc).date()
        old_date = (today - timedelta(days=9)).isoformat()
        self.assertEqual(_compute_staleness(old_date), 9)

    def test_recent_data_returns_small_staleness(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        self.assertEqual(_compute_staleness(today), 0)

    def test_invalid_date_returns_minus_one(self) -> None:
        self.assertEqual(_compute_staleness("not-a-date"), -1)

    def test_cache_staleness_reflected_in_commodity_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_rows = _make_rows(5, base_date="2020-01-01")
            _write_cache(cache_dir, "brent", old_rows)

            data = load_commodity("brent", cache_dir=cache_dir, offline=True)
            self.assertGreater(data.staleness_days, 365 * 5)

    def test_fresh_live_data_has_small_staleness(self) -> None:
        today = datetime.now(timezone.utc).date()
        yesterday = (today - timedelta(days=1)).isoformat()
        two_days_ago = (today - timedelta(days=2)).isoformat()

        fresh_csv = f"Date,Price\n{two_days_ago},88.0\n{yesterday},89.0\n".encode()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("oilenergy.commodities._fetch_url", return_value=fresh_csv):
                data = load_commodity("brent", cache_dir=Path(tmpdir), offline=False)

        self.assertLessEqual(data.staleness_days, 3)


# ---------------------------------------------------------------------------
# 4. Cache metadata mismatch rejection
# ---------------------------------------------------------------------------

class TestCacheMetaValidation(unittest.TestCase):

    def test_valid_meta_accepted(self) -> None:
        meta = {
            "commodity_key": "brent",
            "canonical_series_id": "DCOILBRENTEU",
        }
        config = COMMODITIES["brent"]
        error = _validate_cache_meta(meta, "brent", config)
        self.assertIsNone(error, f"Expected no error but got: {error}")

    def test_missing_meta_rejected(self) -> None:
        config = COMMODITIES["brent"]
        error = _validate_cache_meta(None, "brent", config)
        self.assertIsNotNone(error)
        self.assertIn("legacy", error)

    def test_key_mismatch_rejected(self) -> None:
        meta = {
            "commodity_key": "qatar_lng",   # <-- wrong key
            "canonical_series_id": "DHHNGSP",
        }
        config = COMMODITIES["brent"]
        error = _validate_cache_meta(meta, "brent", config)
        self.assertIsNotNone(error)
        self.assertIn("mismatch", error)

    def test_series_mismatch_rejected(self) -> None:
        meta = {
            "commodity_key": "brent",
            "canonical_series_id": "DHHNGSP",   # <-- Henry Hub, not Brent
        }
        config = COMMODITIES["brent"]
        error = _validate_cache_meta(meta, "brent", config)
        self.assertIsNotNone(error)
        self.assertIn("mismatch", error)

    def test_legacy_cache_without_meta_rejected_at_load(self) -> None:
        """A bare CSV with no sidecar must not be silently served."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Write a brent CSV but label it as qatar_lng (simulates legacy contamination)
            brent_csv = "Date,Price\n2026-01-01,88.0\n2026-01-02,89.0\n"
            (cache_dir / "qatar_lng.csv").write_text(brent_csv, encoding="utf-8")
            # No sidecar → must be rejected

            # Network also unavailable → should raise RuntimeError
            with patch("oilenergy.commodities._fetch_url", side_effect=OSError("network error")):
                with self.assertRaises(RuntimeError):
                    load_commodity("qatar_lng", cache_dir=cache_dir, offline=False)


# ---------------------------------------------------------------------------
# 5. Qatar LNG cache cannot reuse Brent data
# ---------------------------------------------------------------------------

class TestQatarLNGCacheIsolation(unittest.TestCase):
    """Qatar LNG must never silently serve Brent data."""

    def test_qatar_lng_rejects_brent_sidecar(self) -> None:
        """If the sidecar says commodity_key=brent, the cache must be rejected for qatar_lng."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Write what appears to be a Qatar LNG cache, but the sidecar says it's Brent
            brent_rows = _make_rows(5)
            _write_cache(
                cache_dir,
                "qatar_lng",
                brent_rows,
                write_meta=True,
                meta_override={
                    "commodity_key": "brent",            # <-- deliberate mismatch
                    "canonical_series_id": "DCOILBRENTEU",
                    "source_type": "csv",
                    "source_url": "https://example.com/brent",
                    "retrieved_at": _utc_now(),
                    "is_proxy": False,
                },
            )

            # Network also unavailable
            with patch("oilenergy.commodities._fetch_url", side_effect=OSError("network error")):
                with self.assertRaises(RuntimeError) as ctx:
                    load_commodity("qatar_lng", cache_dir=cache_dir, offline=False)

            self.assertIn("mismatch", str(ctx.exception).lower())

    def test_qatar_lng_uses_henry_hub_series(self) -> None:
        """Qatar LNG live data must come from DHHNGSP, not DCOILBRENTEU."""
        henry_hub_csv = _make_fred_csv([
            ("2026-08-10", "3.10"),
            ("2026-08-11", "3.15"),
            ("2026-08-12", "3.20"),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("oilenergy.commodities._fetch_url", return_value=henry_hub_csv) as mock_fetch:
                data = load_commodity("qatar_lng", cache_dir=Path(tmpdir), offline=False)

            # Verify the FRED URL contained DHHNGSP
            called_url = mock_fetch.call_args[0][0]
            self.assertIn("DHHNGSP", called_url, "Qatar LNG must fetch DHHNGSP, not Brent")
            self.assertEqual(data.commodity_key, "qatar_lng")
            self.assertIn("DHHNGSP", data.download_metadata.get("canonical_series_id", ""))

    def test_qatar_lng_and_brent_are_never_identical_from_cache(self) -> None:
        """With separate valid caches, Qatar LNG and Brent must return different data."""
        today = datetime.now(timezone.utc).date()
        d1 = (today - timedelta(days=2)).isoformat()
        d2 = (today - timedelta(days=1)).isoformat()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Brent cache: Brent prices (~88 USD/bbl)
            brent_rows = [PriceRow(date=d1, price=88.0), PriceRow(date=d2, price=89.0)]
            _write_cache(cache_dir, "brent", brent_rows, write_meta=True, meta_override={
                "commodity_key": "brent",
                "canonical_series_id": "DCOILBRENTEU",
                "source_type": "csv",
                "source_url": "https://example.com",
                "retrieved_at": _utc_now(),
                "is_proxy": False,
            })

            # Qatar LNG cache: Henry Hub prices (~3 USD/MMBtu)
            hhub_rows = [PriceRow(date=d1, price=3.10), PriceRow(date=d2, price=3.15)]
            _write_cache(cache_dir, "qatar_lng", hhub_rows, write_meta=True, meta_override={
                "commodity_key": "qatar_lng",
                "canonical_series_id": "DHHNGSP",
                "source_type": "fred",
                "source_url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP",
                "retrieved_at": _utc_now(),
                "is_proxy": True,
            })

            brent_data = load_commodity("brent", cache_dir=cache_dir, offline=True)
            qatar_data = load_commodity("qatar_lng", cache_dir=cache_dir, offline=True)

        brent_prices = [r.price for r in brent_data.rows]
        qatar_prices = [r.price for r in qatar_data.rows]
        self.assertNotEqual(brent_prices, qatar_prices,
                            "Qatar LNG prices must not equal Brent prices")
        self.assertAlmostEqual(qatar_prices[-1], 3.15, places=2)
        self.assertAlmostEqual(brent_prices[-1], 89.0, places=2)


# ---------------------------------------------------------------------------
# 6. Refresh failure with valid cache gives truthful degraded/stale status
# ---------------------------------------------------------------------------

class TestRefreshFailureWithValidCache(unittest.TestCase):

    def test_stale_cache_used_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_date = (datetime.now(timezone.utc).date() - timedelta(days=10)).isoformat()
            rows = _make_rows(5, base_date=old_date)
            _write_cache(cache_dir, "brent", rows)

            with patch("oilenergy.commodities._fetch_url", side_effect=OSError("timeout")):
                data = load_commodity("brent", cache_dir=cache_dir, offline=False)

            self.assertEqual(data.source_mode, "cache")
            self.assertGreater(data.staleness_days, 5)
            self.assertIn("STALE", data.refresh_warning)

    def test_refresh_failure_shows_error_in_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_date = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
            rows = _make_rows(5, base_date=old_date)
            _write_cache(cache_dir, "brent", rows)

            with patch("oilenergy.commodities._fetch_url", side_effect=OSError("connection refused")):
                data = load_commodity("brent", cache_dir=cache_dir, offline=False)

            self.assertIn("connection refused", data.refresh_warning.lower())

    def test_no_valid_cache_after_refresh_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("oilenergy.commodities._fetch_url", side_effect=OSError("no network")):
                with self.assertRaises(RuntimeError) as ctx:
                    load_commodity("brent", cache_dir=Path(tmpdir), offline=False)
            self.assertIn("Failed to load", str(ctx.exception))


# ---------------------------------------------------------------------------
# 7. Output/audit freshness fields
# ---------------------------------------------------------------------------

class TestFreshnessFieldsInAudit(unittest.TestCase):

    def test_commodity_data_audit_includes_freshness_fields(self) -> None:
        from oilenergy.commodities import commodity_data_audit

        today = datetime.now(timezone.utc).date().isoformat()
        data = CommodityData(
            commodity_key="brent",
            commodity_name="Brent Crude Oil",
            source_url="https://example.com",
            source_type="fred",
            rows=[PriceRow(date=today, price=90.0)],
            source_mode="live",
            retrieved_at=_utc_now(),
            latest_obs_date=today,
            staleness_days=0,
            refresh_warning="",
        )
        audit = commodity_data_audit(data, Path("/some/root"))
        self.assertIn("source_mode", audit)
        self.assertIn("retrieved_at", audit)
        self.assertIn("latest_obs_date", audit)
        self.assertIn("staleness_days", audit)
        self.assertIn("refresh_warning", audit)
        self.assertEqual(audit["source_mode"], "live")
        self.assertEqual(audit["staleness_days"], 0)

    def test_live_commodity_data_has_source_mode_live(self) -> None:
        fresh_csv = b"Date,Price\n2026-08-11,88.0\n2026-08-12,89.0\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("oilenergy.commodities._fetch_url", return_value=fresh_csv):
                data = load_commodity("brent", cache_dir=Path(tmpdir), offline=False)
        self.assertEqual(data.source_mode, "live")
        self.assertNotEqual(data.retrieved_at, "")
        self.assertIsInstance(data.staleness_days, int)

    def test_cache_commodity_data_has_source_mode_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            rows = _make_rows(5)
            _write_cache(cache_dir, "brent", rows)
            data = load_commodity("brent", cache_dir=cache_dir, offline=True)
        self.assertEqual(data.source_mode, "cache")


if __name__ == "__main__":
    unittest.main()
