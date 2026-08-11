import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oilenergy import run_pipeline
from oilenergy.commodities import list_commodities
from oilenergy.pipeline import AVAILABLE_MODELS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OilEnergy — Middle East Energy Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Brent crude oil, base features (backwards-compatible default)
  python scripts/train_model.py

  # Qatar LNG with seasonality features
  python scripts/train_model.py --commodity qatar_lng --features seasonality

  # WTI oil with all features (seasonality + cross-commodity correlations)
  python scripts/train_model.py --commodity wti --features all

  # List all available commodities
  python scripts/train_model.py --list-commodities
""",
    )
    parser.add_argument(
        "--commodity",
        default="brent",
        help="Commodity to forecast (default: brent). Use --list-commodities to see options.",
    )
    parser.add_argument(
        "--category",
        choices=["oil", "gas"],
        help="Optional commodity category filter/validation (oil or gas).",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        choices=sorted(AVAILABLE_MODELS.keys()),
        help="Forecasting model to use.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="Number of future days to forecast recursively (1-30).",
    )
    parser.add_argument(
        "--features",
        default="base",
        help=(
            "Comma-separated feature flags: "
            "base (default), seasonality, weather, demand, external, all. "
            "Example: --features seasonality,weather,demand,external"
        ),
    )
    parser.add_argument(
        "--allow-demo-fallback",
        action="store_true",
        help=(
            "Explicitly allow local Brent CSV fallback when remote sources are unavailable. "
            "Disabled by default to avoid hidden proxy substitution."
        ),
    )
    parser.add_argument(
        "--list-commodities",
        action="store_true",
        help="Print all available commodities and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list_commodities:
        print("\nAvailable commodities:\n")
        for c in list_commodities():
            print(f"  {c['key']:20s}  [{c['type']:3s}] {c['name']}")
            print(f"  {' ':20s}        Region: {c['region']}")
            print(f"  {' ':20s}        Proxy: {str(c.get('is_proxy', False)).lower()}")
            print(f"  {' ':20s}        {c['description'][:80]}")
            print()
        raise SystemExit(0)

    project_root = PROJECT_ROOT
    result = run_pipeline(
        project_root,
        commodity=args.commodity,
        category=args.category,
        model_name=args.model,
        horizon_days=args.horizon_days,
        features=args.features,
        allow_demo_fallback=args.allow_demo_fallback,
    )

    print("Commodity:", result["model_audit"].get("commodity_name", args.commodity))
    print("Category:", result["model_audit"].get("commodity_type", "unknown"))
    print("Model:", result["model_audit"].get("model_name", args.model))
    print("Forecast horizon (days):", result["model_audit"].get("horizon_days", args.horizon_days))
    print("Proxy commodity:", result["model_audit"].get("is_proxy_commodity", False))
    if result["model_audit"].get("proxy_for"):
        print("Proxy for:", result["model_audit"]["proxy_for"])
    print("Feature flags:", result["model_audit"].get("feature_flags", "base"))
    print("Dataset rows:", result["dataset_audit"]["row_count"])
    print(
        "Dataset date range:",
        result["dataset_audit"]["date_range"]["start"],
        "to",
        result["dataset_audit"]["date_range"]["end"],
    )
    print("Test MAE:", result["model_audit"]["test_metrics"]["mae"])
    print("Test RMSE:", result["model_audit"]["test_metrics"]["rmse"])
    print("Directional accuracy:", result["model_audit"]["test_metrics"]["directional_accuracy"])
    print("Latest next-price prediction:", result["model_audit"]["latest_prediction"]["predicted_next_price"])
    forecast = result["model_audit"].get("multi_day_forecast", [])
    if forecast:
        print("\nMulti-day forecast:")
        for row in forecast:
            direction = "up" if row["predicted_direction_up_vs_previous_day"] else "down"
            print(f"  Day {row['step_day']:2d} ({row['forecast_date']}): {row['predicted_price']} ({direction})")

    feature_availability = result["model_audit"].get("contextual_feature_availability", {})
    if feature_availability:
        print("\nContextual feature availability:")
        for key, value in feature_availability.items():
            print(
                f"  - {key}: {'available' if value.get('available') else 'unavailable'} "
                f"({value.get('source_path')})"
            )

    interp = result.get("interpretation", {})
    if interp:
        print(f"\n--- AI Interpretation ({interp.get('source', 'template')}) ---")
        print(interp.get("summary", ""))

    if result.get("correlation_audit", {}).get("significant_partners_for_target"):
        partners = result["correlation_audit"]["significant_partners_for_target"]
        print(f"\nSignificant correlated commodities: {', '.join(partners)}")
