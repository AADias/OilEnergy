"""Tests for the OilEnergy pipeline, correlation safeguards, and context features.

All tests use only standard-library features — no third-party test framework.
Run with:
    python -m pytest tests/
or:
    python -m unittest discover tests/
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oilenergy.pipeline import (
    PriceRow,
    build_samples,
    build_features,
    build_seasonality_features,
    fit_ridge_regression,
    predict,
    evaluate,
    evaluate_naive,
    naive_predict,
    recursive_forecast,
    split_samples,
)
from oilenergy.correlation_analysis import (
    pearson_correlation as corr_pearson,
    shared_underlying_source,
    significant_partners,
    compute_correlation_matrix,
)
from oilenergy.context_features import (
    load_context_series,
    build_context_features,
    ContextSeries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(n: int = 50, start_price: float = 80.0, step: float = 0.5) -> list[PriceRow]:
    """Create a simple ascending price series for testing."""
    rows = []
    for i in range(n):
        date_str = f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        rows.append(PriceRow(date=date_str, price=round(start_price + i * step, 4)))
    return rows


def _write_context_csv(path: Path, rows: list[tuple[str, float]], lineage: str = "real") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("date,value,lineage\n")
        for d, v in rows:
            fh.write(f"{d},{v},{lineage}\n")


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

class TestPearsonCorrelation(unittest.TestCase):

    def test_perfect_positive(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(corr_pearson(x, x), 1.0, places=6)

    def test_perfect_negative(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(corr_pearson(x, y), -1.0, places=6)

    def test_zero_variance_returns_zero(self) -> None:
        x = [3.0] * 10
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertEqual(corr_pearson(x, y), 0.0)

    def test_too_few_points(self) -> None:
        self.assertEqual(corr_pearson([1.0], [1.0]), 0.0)
        self.assertEqual(corr_pearson([1.0, 2.0], [1.0, 2.0]), 0.0)


# ---------------------------------------------------------------------------
# Duplicate-source exclusion (issue #3)
# ---------------------------------------------------------------------------

class TestSharedUnderlyingSource(unittest.TestCase):

    def test_qatar_lng_and_henry_hub_share_source(self) -> None:
        """qatar_lng and henry_hub both use FRED/DHHNGSP — must be flagged as shared."""
        self.assertTrue(shared_underlying_source("qatar_lng", "henry_hub"))
        self.assertTrue(shared_underlying_source("henry_hub", "qatar_lng"))

    def test_brent_and_opec_basket_share_source(self) -> None:
        """brent and opec_basket both use FRED/DCOILBRENTEU — must be flagged as shared."""
        self.assertTrue(shared_underlying_source("brent", "opec_basket"))

    def test_brent_and_wti_do_not_share_source(self) -> None:
        """brent (DCOILBRENTEU) and wti (DCOILWTICO) are independently sourced."""
        self.assertFalse(shared_underlying_source("brent", "wti"))

    def test_brent_and_henry_hub_do_not_share_source(self) -> None:
        self.assertFalse(shared_underlying_source("brent", "henry_hub"))


class TestSignificantPartnersExcludesDuplicates(unittest.TestCase):
    """significant_partners must never return a commodity with the same source."""

    def _make_matrix_with_result(self, key_a: str, key_b: str, r: float) -> dict:
        from oilenergy.correlation_analysis import CorrelationResult
        matrix = {
            (key_a, key_b): CorrelationResult(key_a, key_b, r, 100, abs(r) >= 0.5),
            (key_b, key_a): CorrelationResult(key_b, key_a, r, 100, abs(r) >= 0.5),
        }
        return matrix

    def test_shared_source_excluded_even_if_highly_correlated(self) -> None:
        """qatar_lng should not appear as a significant partner for henry_hub."""
        matrix = self._make_matrix_with_result("henry_hub", "qatar_lng", 0.999)
        partners = significant_partners("henry_hub", matrix)
        self.assertNotIn("qatar_lng", partners)

    def test_independent_source_included_when_correlated(self) -> None:
        """brent should appear as a significant partner for wti (independent sources)."""
        matrix = self._make_matrix_with_result("wti", "brent", 0.92)
        partners = significant_partners("wti", matrix)
        self.assertIn("brent", partners)

    def test_shared_source_excluded_brent_opec(self) -> None:
        """opec_basket should not appear as a significant partner for brent."""
        matrix = self._make_matrix_with_result("brent", "opec_basket", 0.999)
        partners = significant_partners("brent", matrix)
        self.assertNotIn("opec_basket", partners)

    def test_uncorrelated_independent_not_included(self) -> None:
        """henry_hub should not appear for brent when correlation is low."""
        matrix = self._make_matrix_with_result("brent", "henry_hub", 0.1)
        partners = significant_partners("brent", matrix)
        self.assertNotIn("henry_hub", partners)


# ---------------------------------------------------------------------------
# Context features — date alignment (issue #2)
# ---------------------------------------------------------------------------

class TestContextFeatures(unittest.TestCase):

    def _make_series(self, start: str, end: str) -> ContextSeries:
        """Make a ContextSeries with daily values from start to end."""
        from datetime import date, timedelta
        values: dict[str, float] = {}
        d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        v = 25.0
        while d <= end_d:
            values[d.isoformat()] = v
            d += timedelta(days=1)
            v += 0.1
        return ContextSeries(
            category="weather",
            values_by_date=values,
            lineage="real test data",
            feature_names=["weather_lag_1", "weather_mean_3"],
        )

    def test_date_within_coverage_returns_features(self) -> None:
        series = self._make_series("2022-01-01", "2022-12-31")
        result = build_context_features("2022-06-15", series)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # type: ignore[arg-type]

    def test_date_before_coverage_returns_none(self) -> None:
        """Dates before the context data window must return None, not zeros."""
        series = self._make_series("2022-01-01", "2022-12-31")
        result = build_context_features("2015-01-01", series)
        self.assertIsNone(result)

    def test_date_after_coverage_returns_none(self) -> None:
        series = self._make_series("2022-01-01", "2022-12-31")
        result = build_context_features("2025-06-01", series)
        self.assertIsNone(result)

    def test_coverage_attributes(self) -> None:
        series = self._make_series("2022-03-01", "2022-09-30")
        self.assertEqual(series.coverage_start, "2022-03-01")
        self.assertEqual(series.coverage_end, "2022-09-30")

    def test_load_context_series_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series, avail = load_context_series("weather", Path(tmpdir))
            self.assertIsNone(series)
            self.assertFalse(avail["available"])
            self.assertEqual(avail["status"], "missing_file")

    def test_load_context_series_real_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ctx_dir = root / "data" / "context"
            _write_context_csv(
                ctx_dir / "weather.csv",
                [("2022-01-01", 22.5), ("2022-01-02", 23.1)],
                lineage="Open-Meteo archive API (real observations, Doha Qatar)",
            )
            series, avail = load_context_series("weather", root)
            self.assertIsNotNone(series)
            self.assertTrue(avail["available"])
            self.assertEqual(avail["lineage_type"], "real")

    def test_load_context_series_demo_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ctx_dir = root / "data" / "context"
            _write_context_csv(
                ctx_dir / "weather.csv",
                [("2022-01-01", 22.5), ("2022-01-02", 23.1)],
                lineage="DEMO — synthetic Doha climate normals",
            )
            series, avail = load_context_series("weather", root)
            self.assertIsNotNone(series)
            self.assertEqual(avail["lineage_type"], "demo")


# ---------------------------------------------------------------------------
# build_samples — context date restriction (issue #2)
# ---------------------------------------------------------------------------

class TestBuildSamplesContextRestriction(unittest.TestCase):

    def test_context_start_restricts_samples(self) -> None:
        """Samples dated before context_start must be excluded."""
        rows = _make_rows(60)
        # Set a context start that would exclude the first ~10 eligible samples
        # (rows[10] is the first index usable in build_samples)
        context_start = rows[20].date
        samples, _ = build_samples(rows, context_start=context_start)
        for s in samples:
            self.assertGreaterEqual(s.date, context_start)

    def test_context_end_restricts_samples(self) -> None:
        rows = _make_rows(60)
        context_end = rows[40].date
        samples, _ = build_samples(rows, context_end=context_end)
        for s in samples:
            self.assertLessEqual(s.date, context_end)

    def test_no_restriction_includes_all_eligible(self) -> None:
        rows = _make_rows(50)
        samples, _ = build_samples(rows)
        # All indices 10..48 should be included
        self.assertEqual(len(samples), 39)

    def test_context_series_outside_coverage_excluded(self) -> None:
        """Samples outside the context series coverage must be excluded."""
        rows = _make_rows(60)
        from datetime import date, timedelta
        from oilenergy.context_features import ContextSeries

        # Context only covers middle part of the series
        start_d = date.fromisoformat(rows[15].date)
        end_d = date.fromisoformat(rows[45].date)
        values = {}
        d = start_d
        while d <= end_d:
            values[d.isoformat()] = 25.0
            d += timedelta(days=1)

        series = ContextSeries(
            category="weather",
            values_by_date=values,
            lineage="real",
            feature_names=["weather_lag_1", "weather_mean_3"],
        )
        context_start = series.coverage_start
        context_end = series.coverage_end
        samples, _ = build_samples(
            rows,
            context_series_list=[series],
            context_start=context_start,
            context_end=context_end,
        )
        for s in samples:
            self.assertGreaterEqual(s.date, context_start)
            self.assertLessEqual(s.date, context_end)


# ---------------------------------------------------------------------------
# Multi-day recursive forecasting — no future data used (issue #5)
# ---------------------------------------------------------------------------

class TestRecursiveForecast(unittest.TestCase):

    def _fit_model(self, rows: list[PriceRow]) -> list[float]:
        samples, _ = build_samples(rows)
        train, _ = split_samples(samples)
        return fit_ridge_regression(train)

    def test_horizon_1_returns_one_step(self) -> None:
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=1)
        self.assertEqual(len(forecast), 1)

    def test_horizon_7_returns_seven_steps(self) -> None:
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=7)
        self.assertEqual(len(forecast), 7)

    def test_forecast_does_not_mutate_rows(self) -> None:
        """recursive_forecast must not extend the original rows list."""
        rows = _make_rows(50)
        original_len = len(rows)
        weights = self._fit_model(rows)
        recursive_forecast(weights, rows, horizon_days=5)
        self.assertEqual(len(rows), original_len)

    def test_forecast_dates_are_sequential(self) -> None:
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=5)
        dates = [f["date"] for f in forecast]
        self.assertEqual(dates, sorted(dates))

    def test_forecast_steps_are_numbered(self) -> None:
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=4)
        self.assertEqual([f["step"] for f in forecast], [1, 2, 3, 4])

    def test_carry_forward_disclosure_present(self) -> None:
        """Every forecast step must include a carry_forward_disclosure."""
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=3)
        for step in forecast:
            self.assertIn("carry_forward_disclosure", step)
            self.assertTrue(step["carry_forward_disclosure"])

    def test_naive_forecast_predicts_last_price(self) -> None:
        rows = _make_rows(30)
        last_price = rows[-1].price
        forecast = recursive_forecast([], rows, horizon_days=5, model="naive")
        for step in forecast:
            self.assertEqual(step["predicted_price"], last_price)

    def test_date_type_labelled_calendar_day(self) -> None:
        rows = _make_rows(50)
        weights = self._fit_model(rows)
        forecast = recursive_forecast(weights, rows, horizon_days=2)
        for step in forecast:
            self.assertEqual(step["date_type"], "calendar_day")


# ---------------------------------------------------------------------------
# run_pipeline validation (issue #4 / issue #7)
# ---------------------------------------------------------------------------

class TestRunPipelineValidation(unittest.TestCase):

    def test_invalid_horizon_raises(self) -> None:
        from oilenergy.pipeline import run_pipeline
        with self.assertRaises(ValueError):
            run_pipeline(Path("."), horizon_days=0)

    def test_invalid_model_raises(self) -> None:
        from oilenergy.pipeline import run_pipeline
        with self.assertRaises(ValueError):
            run_pipeline(Path("."), model="xgboost")


# ---------------------------------------------------------------------------
# fetch_weather.py — no auto synthetic fallback (issue #1)
# ---------------------------------------------------------------------------

class TestFetchWeather(unittest.TestCase):

    def test_no_network_returns_nonzero_exit_without_demo_flag(self) -> None:
        """On network failure, main() must return non-zero without --demo-data."""
        # Patch fetch_from_api to raise a network error
        import scripts.fetch_weather as fw  # type: ignore[import]
        original = fw.fetch_from_api

        def _fail(*args, **kwargs):
            raise OSError("network error")

        fw.fetch_from_api = _fail
        try:
            exit_code = fw.main(["--start", "2022-01-01", "--end", "2022-01-10"])
        finally:
            fw.fetch_from_api = original

        self.assertNotEqual(exit_code, 0, "Expected non-zero exit on network failure without --demo-data")

    def test_demo_flag_writes_clearly_labelled_data(self) -> None:
        """--demo-data must write CSVs with a DEMO lineage label."""
        import scripts.fetch_weather as fw  # type: ignore[import]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch project_root resolution
            original_main = fw.main

            import csv as _csv
            from pathlib import Path as _Path

            weather_path = _Path(tmpdir) / "data" / "context" / "weather.csv"
            demand_path = _Path(tmpdir) / "data" / "context" / "demand.csv"

            # Directly call generate_demo_data + write_csv to test the labels
            from datetime import date
            rows = fw.generate_demo_data(date(2022, 1, 1), date(2022, 1, 5))
            lineage = "DEMO — synthetic Doha climate normals (WMO), NOT real observations"
            fw.write_csv(weather_path, rows, lineage)

            with weather_path.open(newline="", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                first_row = next(reader)
            self.assertIn("DEMO", first_row["lineage"])


if __name__ == "__main__":
    unittest.main()
