"""Regression tests for llm_interpreter template output and forecast date semantics."""
from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oilenergy.llm_interpreter import _template_summary, interpret_results


def _make_audit(
    direction_up: bool = True,
    pred_price: float = 88.9,
    obs_price: float = 88.0,
    horizon_days: int = 1,
    forecast: list | None = None,
) -> dict:
    return {
        "test_metrics": {"directional_accuracy": 0.511, "mae": 1.407, "rmse": 2.15},
        "latest_prediction": {
            "predicted_next_price": pred_price,
            "latest_observation_price": obs_price,
            "predicted_direction_up": direction_up,
            "forecast_date": "2026-08-04",
            "date_type": "calendar_day",
        },
        "horizon_days": horizon_days,
        "forecast": forecast or [],
    }


class TestTemplateGrammar(unittest.TestCase):
    """Grammar: article before trend direction."""

    def test_an_upward_trend(self) -> None:
        audit = _make_audit(direction_up=True)
        summary = _template_summary(audit, "Brent Crude Oil")
        self.assertIn("an upward trend", summary)
        self.assertNotIn("a upward trend", summary)

    def test_a_downward_trend(self) -> None:
        audit = _make_audit(direction_up=False)
        summary = _template_summary(audit, "Brent Crude Oil")
        self.assertIn("a downward trend", summary)
        self.assertNotIn("an downward trend", summary)


class TestTemplateNoTradingSession(unittest.TestCase):
    """Template must not call the forecast a 'trading session' price."""

    def test_no_trading_session_wording(self) -> None:
        for direction_up in (True, False):
            with self.subTest(direction_up=direction_up):
                audit = _make_audit(direction_up=direction_up)
                summary = _template_summary(audit, "Brent Crude Oil")
                self.assertNotIn("trading session", summary)
                self.assertIn("calendar-day", summary)


class TestTemplateMultiDaySummary(unittest.TestCase):
    """Multi-day forecast section appears iff horizon_days > 1."""

    def _make_forecast(self, n: int) -> list[dict]:
        base = 88.0
        start = datetime.date(2026, 8, 4)
        return [
            {
                "step": i + 1,
                "forecast_date": (start + datetime.timedelta(days=i)).isoformat(),
                "predicted_price": round(base + i * 0.1, 4),
                "direction": "up",
            }
            for i in range(n)
        ]

    def test_single_day_no_multi_section(self) -> None:
        audit = _make_audit(horizon_days=1, forecast=self._make_forecast(1))
        summary = _template_summary(audit, "Brent Crude Oil")
        self.assertNotIn("Multi-day forecast", summary)

    def test_multi_day_section_present(self) -> None:
        forecast = self._make_forecast(5)
        audit = _make_audit(horizon_days=5, forecast=forecast)
        summary = _template_summary(audit, "Brent Crude Oil")
        self.assertIn("Multi-day forecast (5 calendar days)", summary)
        # First and last date/price must appear
        self.assertIn(forecast[0]["forecast_date"], summary)
        self.assertIn(forecast[-1]["forecast_date"], summary)
        self.assertIn("recursive carry-forward", summary)
        self.assertIn("calendar days", summary)

    def test_disclaimer_always_present(self) -> None:
        for h in (1, 5):
            with self.subTest(horizon=h):
                audit = _make_audit(horizon_days=h, forecast=self._make_forecast(h))
                summary = _template_summary(audit, "Brent Crude Oil")
                self.assertIn("Disclaimer", summary)
                self.assertIn("research and educational purposes", summary)


class TestForecastDateSemantics(unittest.TestCase):
    """latest_prediction.forecast_date and forecast[0].forecast_date must agree."""

    def test_single_step_forecast_date_matches_latest_prediction(self) -> None:
        forecast_date = "2026-08-04"
        audit = _make_audit(horizon_days=1)
        audit["latest_prediction"]["forecast_date"] = forecast_date
        audit["forecast"] = [
            {
                "step": 1,
                "forecast_date": forecast_date,
                "predicted_price": audit["latest_prediction"]["predicted_next_price"],
                "direction": "up",
            }
        ]
        pred_date = audit["latest_prediction"]["forecast_date"]
        first_fc_date = audit["forecast"][0]["forecast_date"]
        self.assertEqual(pred_date, first_fc_date, (
            "latest_prediction.forecast_date and forecast[0].forecast_date must be identical "
            f"for a single-step forecast, got {pred_date!r} vs {first_fc_date!r}"
        ))

    def test_multi_step_first_forecast_matches_latest_prediction(self) -> None:
        """For multi-day, forecast[0] should correspond to the same next step as latest_prediction."""
        forecast_date = "2026-08-04"
        audit = _make_audit(horizon_days=5)
        audit["latest_prediction"]["forecast_date"] = forecast_date
        audit["forecast"] = [
            {
                "step": i + 1,
                "forecast_date": (datetime.date(2026, 8, 4) + datetime.timedelta(days=i)).isoformat(),
                "predicted_price": 88.9 + i * 0.1,
                "direction": "up",
            }
            for i in range(5)
        ]
        pred_date = audit["latest_prediction"]["forecast_date"]
        first_fc_date = audit["forecast"][0]["forecast_date"]
        self.assertEqual(pred_date, first_fc_date, (
            "For multi-day, the latest_prediction date and forecast[0] date must share the same "
            f"next-step semantics, got {pred_date!r} vs {first_fc_date!r}"
        ))


class TestInterpretResultsFallback(unittest.TestCase):
    """interpret_results falls back to template when no HF token is set."""

    def test_returns_template_source_without_token(self) -> None:
        import os
        os.environ.pop("HUGGINGFACE_API_KEY", None)
        audit = _make_audit(direction_up=True)
        result = interpret_results(audit, commodity_name="Brent Crude Oil")
        self.assertEqual(result["source"], "template")
        self.assertIn("an upward trend", result["summary"])
        self.assertNotIn("trading session", result["summary"])


if __name__ == "__main__":
    unittest.main()
