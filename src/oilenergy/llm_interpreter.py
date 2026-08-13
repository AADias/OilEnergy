"""llm_interpreter.py — HuggingFace-powered result interpretation for OilEnergy.

Uses the HuggingFace Inference API (facebook/bart-large-cnn) to generate
intelligent, context-aware summaries of model predictions. This helps
Middle East energy professionals interpret forecasting results without
needing deep data-science expertise.

API token is read from the HUGGINGFACE_API_KEY environment variable (or .env file).
If no token is available, a professional template-based summary is returned instead.

Usage:
    from oilenergy.llm_interpreter import interpret_results
    summary = interpret_results(model_audit, commodity_name="Qatar LNG")
    print(summary)
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# HuggingFace model — bart-large-cnn is fast, free, and reliable for summarization
HF_MODEL = "facebook/bart-large-cnn"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Maximum characters for the prompt sent to the model
MAX_PROMPT_CHARS = 1024


def _get_hf_token() -> str | None:
    """Read HuggingFace API token from environment or .env file."""
    token = os.environ.get("HUGGINGFACE_API_KEY", "").strip()
    if token:
        return token
    # Try loading from a local .env file
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    env_file = os.path.normpath(env_file)
    if os.path.isfile(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HUGGINGFACE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _build_prompt(audit: dict[str, Any], commodity_name: str) -> str:
    """Build a factual summary prompt from the model audit dict."""
    metrics = audit.get("test_metrics", {})
    mae = metrics.get("mae", "N/A")
    rmse = metrics.get("rmse", "N/A")
    directional = metrics.get("directional_accuracy", "N/A")
    latest = audit.get("latest_prediction", {})
    pred_price = latest.get("predicted_next_price", "N/A")
    obs_price = latest.get("latest_observation_price", "N/A")
    obs_date = latest.get("latest_observation_date", "N/A")
    direction = "upward" if latest.get("predicted_direction_up", False) else "downward"
    article = "an" if direction == "upward" else "a"
    commodity = commodity_name or "the commodity"

    prompt = (
        f"Energy market forecast report for {commodity}. "
        f"As of {obs_date}, the current observed price is {obs_price}. "
        f"The model predicts the next calendar-day price will be {pred_price}, "
        f"indicating {article} {direction} trend. "
        f"Model accuracy on historical test data: Mean Absolute Error {mae}, "
        f"Root Mean Squared Error {rmse}, "
        f"directional accuracy {directional} (proportion of correct up/down calls). "
        f"Please summarize these findings for an energy market professional in 2-3 sentences."
    )
    return prompt[:MAX_PROMPT_CHARS]


def _call_hf_api(prompt: str, token: str) -> str | None:
    """Call the HuggingFace Inference API and return the generated text."""
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {"max_length": 200, "min_length": 40, "do_sample": False},
    }).encode("utf-8")

    req = Request(
        HF_API_URL,
        data=payload,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        # Response is a list with one dict containing "summary_text"
        if isinstance(result, list) and result:
            return result[0].get("summary_text") or result[0].get("generated_text")
        if isinstance(result, dict):
            return result.get("summary_text") or result.get("generated_text")
        return None
    except (HTTPError, URLError, OSError, json.JSONDecodeError):
        return None


def _classify_trend(
    start_price: float,
    end_price: float,
    sideways_threshold_pct: float = 0.15,
) -> tuple[str, str, str]:
    """Classify a price movement as bullish, bearish, or sideways.

    Returns a tuple of (label, article, modifier) where:
      label    — "bullish/upward", "bearish/downward", or "sideways/neutral"
      article  — "a" or "an"
      modifier — "slight", "moderate", or "" for sideways
    """
    if start_price == 0:
        return "sideways/neutral", "a", ""
    pct_change = (end_price - start_price) / abs(start_price) * 100
    abs_pct = abs(pct_change)
    if abs_pct < sideways_threshold_pct:
        return "sideways/neutral", "a", ""
    if pct_change > 0:
        modifier = "slight" if abs_pct < 1.0 else "moderate"
        return "bullish/upward", "a", modifier
    modifier = "slight" if abs_pct < 1.0 else "moderate"
    return "bearish/downward", "a", modifier


def _template_summary(audit: dict[str, Any], commodity_name: str) -> str:
    """Return a professional template-based summary (no API required)."""
    metrics = audit.get("test_metrics", {})
    directional = metrics.get("directional_accuracy", 0)
    mae = metrics.get("mae", "N/A")
    latest = audit.get("latest_prediction", {})
    pred_price = latest.get("predicted_next_price", "N/A")
    obs_price = latest.get("latest_observation_price", "N/A")
    commodity = commodity_name or "the commodity"
    horizon_days: int = audit.get("horizon_days", 1) or 1
    forecast: list[dict[str, Any]] = audit.get("forecast", []) or []

    directional_pct = f"{float(directional) * 100:.1f}%" if isinstance(directional, (int, float)) else directional

    # Determine trend from forecast sequence (or single next-price vs observation)
    try:
        _obs = float(obs_price)
        _pred = float(pred_price)
    except (TypeError, ValueError):
        _obs = 0.0
        _pred = 0.0

    if horizon_days > 1 and len(forecast) >= 2:
        try:
            _first = float(forecast[0].get("predicted_price", _pred))
            _last = float(forecast[-1].get("predicted_price", _pred))
        except (TypeError, ValueError):
            _first, _last = _pred, _pred
        trend_label, article, modifier = _classify_trend(_first, _last)
    else:
        trend_label, article, modifier = _classify_trend(_obs, _pred)

    modifier_phrase = f" {modifier}" if modifier else ""

    lines = [
        f"Forecast Summary — {commodity}",
        "",
        f"Based on current market data, {commodity} is showing{modifier_phrase} "
        f"{article} {trend_label} trend. "
        f"The model forecasts the next calendar-day price at {pred_price} "
        f"(current observation: {obs_price}).",
    ]

    # Multi-day summary when a horizon was requested
    if horizon_days > 1 and len(forecast) >= 2:
        first = forecast[0]
        last = forecast[-1]
        first_date = first.get("forecast_date") or first.get("date") or "N/A"
        last_date = last.get("forecast_date") or last.get("date") or "N/A"
        first_price = first.get("predicted_price", "N/A")
        last_price = last.get("predicted_price", "N/A")
        lines += [
            "",
            f"Multi-day forecast ({horizon_days} calendar days): "
            f"{first_date} → {first_price}; {last_date} → {last_price}. "
            "Forecast dates are calendar days and may include weekends and holidays. "
            "Each step uses the previously predicted price as input (recursive carry-forward); "
            "uncertainty accumulates with each additional step.",
        ]

    lines += [
        "",
        f"The model achieves {directional_pct} directional accuracy on historical test data, "
        f"with a mean absolute error of {mae}. "
        "This provides a quantitative baseline for price direction; "
        "regional energy professionals should combine these signals with "
        "geopolitical context, OPEC supply decisions, and seasonal demand factors.",
        "",
        "Disclaimer: This forecast is generated by a statistical model for "
        "research and educational purposes only. It does not constitute financial "
        "or trading advice. Past model accuracy does not guarantee future performance.",
    ]
    return "\n".join(lines)


def interpret_results(
    model_audit: dict[str, Any],
    commodity_name: str = "",
) -> dict[str, Any]:
    """Generate an intelligent interpretation of model results.

    Tries the HuggingFace Inference API first; falls back to a template summary.

    Args:
        model_audit: The model audit dictionary from run_pipeline.
        commodity_name: Human-readable commodity name for context.

    Returns:
        dict with keys:
            "summary": str — the generated or template summary
            "source": "huggingface" | "template"
            "model": HF model name (if used)
    """
    token = _get_hf_token()
    if token:
        prompt = _build_prompt(model_audit, commodity_name)
        generated = _call_hf_api(prompt, token)
        if generated:
            return {
                "summary": generated.strip(),
                "source": "huggingface",
                "model": HF_MODEL,
            }

    # Fallback to template
    return {
        "summary": _template_summary(model_audit, commodity_name),
        "source": "template",
        "model": None,
    }
