from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oilenergy.commodities import CommodityDefinition, load_rows
from oilenergy.correlation_analysis import pearson_correlation, spearman_correlation
from oilenergy.external_features import align_price_series, compute_returns
from oilenergy.pipeline import FEATURE_NAMES, seasonal_features


class EnrichmentTests(unittest.TestCase):
    def test_feature_names_include_seasonality(self) -> None:
        self.assertIn("month_of_year", FEATURE_NAMES)
        self.assertIn("quarter", FEATURE_NAMES)
        self.assertIn("day_of_week", FEATURE_NAMES)
        self.assertIn("is_weekend_or_holiday", FEATURE_NAMES)

    def test_seasonal_features_encode_calendar_values(self) -> None:
        features = seasonal_features("2024-12-25")
        self.assertEqual(features, [12.0, 4.0, 2.0, 1.0])

    def test_loader_normalizes_dates_and_skips_missing_values(self) -> None:
        definition = CommodityDefinition(
            key="test",
            display_name="Test",
            source_type="public_csv",
            source_url="https://example.com/test.csv",
            local_filename="test.csv",
            date_column="DATE",
            price_column="VALUE",
            date_formats=["%d/%m/%Y"],
            missing_values=["."],
        )
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("DATE,VALUE\n01/08/2024,10\n02/08/2024,.\n03/08/2024,11\n", encoding="utf-8")
            rows = load_rows(csv_path, definition)
        self.assertEqual([row.date for row in rows], ["2024-08-01", "2024-08-03"])
        self.assertEqual([row.price for row in rows], [10.0, 11.0])

    def test_external_alignment_supports_forward_fill_and_returns(self) -> None:
        prices = align_price_series(
            ["2024-01-01", "2024-01-03"],
            [10.0, 12.0],
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            "forward_fill",
        )
        self.assertEqual(prices, [10.0, 10.0, 12.0])
        self.assertEqual(compute_returns(prices), [0.0, 0.0, 0.2])

    def test_correlation_helpers_match_expected_ordering(self) -> None:
        left = [1.0, 2.0, 3.0, 4.0]
        right = [2.0, 4.0, 6.0, 8.0]
        inverse = [8.0, 6.0, 4.0, 2.0]
        self.assertAlmostEqual(pearson_correlation(left, right), 1.0)
        self.assertAlmostEqual(spearman_correlation(left, right), 1.0)
        self.assertAlmostEqual(pearson_correlation(left, inverse), -1.0)


if __name__ == "__main__":
    unittest.main()
