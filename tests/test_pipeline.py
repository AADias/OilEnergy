from __future__ import annotations

import unittest
from pathlib import Path

from oilenergy.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_horizon_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline(PROJECT_ROOT, commodity="brent", horizon_days=0)

    def test_multi_day_forecast_length(self) -> None:
        result = run_pipeline(
            PROJECT_ROOT,
            commodity="brent",
            model_name="naive",
            horizon_days=3,
            features="base",
        )
        forecast = result["model_audit"]["multi_day_forecast"]
        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast[0]["step_day"], 1)
        self.assertEqual(forecast[-1]["step_day"], 3)

    def test_external_correlation_excludes_demo_fallbacks_and_duplicates(self) -> None:
        result = run_pipeline(
            PROJECT_ROOT,
            commodity="brent",
            features="external",
            allow_demo_fallback=True,
        )
        audit = result["correlation_audit"]
        self.assertEqual(audit.get("significant_partners_for_target"), [])
        errors = " | ".join(audit.get("correlation_errors", []))
        self.assertIn("excluded from correlations", errors)


if __name__ == "__main__":
    unittest.main()
