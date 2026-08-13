import argparse
import sys
from pathlib import Path

# Allow running directly from repo root: `python scripts\train_model.py`
# without needing to set PYTHONPATH=src first.
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from oilenergy import run_pipeline
from oilenergy.commodities import list_commodities


def _horizon_days_arg(value: str) -> int:
    days = int(value)
    if days < 1 or days > 30:
        raise argparse.ArgumentTypeError("--horizon-days must be between 1 and 30")
    return days


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OilEnergy — Middle East Energy Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Brent crude oil, base features (backwards-compatible default)
  python scripts\\train_model.py

  # Qatar LNG with seasonality features
  python scripts\\train_model.py --commodity qatar_lng --features seasonality

  # WTI oil with all features (seasonality + cross-commodity correlations)
  python scripts\\train_model.py --commodity wti --features all

  # Naive baseline, 7-day recursive calendar forecast
  python scripts\\train_model.py --model naive --horizon-days 7

  # Ridge model, 5-day forecast, with local weather/demand context (requires data/context/*.csv)
  python scripts\\train_model.py --features seasonality,context --horizon-days 5

  # List all available commodities
  python scripts\\train_model.py --list-commodities
""",
    )
    parser.add_argument(
        "--commodity",
        default="brent",
        help="Commodity to forecast (default: brent). Use --list-commodities to see options.",
    )
    parser.add_argument(
        "--features",
        default="base",
        help=(
            "Comma-separated feature flags: "
            "base (default), seasonality, external, context, all. "
            "context requires data/context/weather.csv and data/context/demand.csv — "
            "run scripts/fetch_weather.py to generate them. "
            "Example: --features seasonality,external"
        ),
    )
    parser.add_argument(
        "--model",
        default="ridge",
        choices=["ridge", "naive"],
        help=(
            "Forecasting model (default: ridge). "
            "ridge — regularised linear regression on price lags. "
            "naive — persistence baseline (predict last known price)."
        ),
    )
    parser.add_argument(
        "--horizon-days",
        type=_horizon_days_arg,
        default=1,
        metavar="N",
        help=(
            "Number of calendar days to forecast ahead (1-30, default: 1). "
            "Multi-day forecasts use recursive carry-forward and include "
            "a disclosure note. Dates are calendar days, not guaranteed trading days."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=["oil", "gas"],
        help="Optional category filter label (oil or gas) — for auditing/display only.",
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
            print(f"  {' ':20s}        {c['description'][:80]}")
            print()
        raise SystemExit(0)

    project_root = Path(__file__).resolve().parents[1]
    result = run_pipeline(
        project_root,
        commodity=args.commodity,
        features=args.features,
        model=args.model,
        horizon_days=args.horizon_days,
        category=args.category,
    )

    ma = result["model_audit"]
    print("Commodity:", ma.get("commodity_name", args.commodity))
    print("Category:", ma.get("category", "unknown"))
    print("Model:", ma.get("model", "ridge"))
    print("Feature flags:", ma.get("feature_flags", "base"))
    print("Dataset rows:", result["dataset_audit"]["row_count"])
    print(
        "Dataset date range:",
        result["dataset_audit"]["date_range"]["start"],
        "to",
        result["dataset_audit"]["date_range"]["end"],
    )
    print("Test MAE:", ma["test_metrics"]["mae"])
    print("Test RMSE:", ma["test_metrics"]["rmse"])
    print("Directional accuracy:", ma["test_metrics"]["directional_accuracy"])
    print("Latest next-price prediction:", ma["latest_prediction"]["predicted_next_price"])

    # Multi-day forecast output
    forecast = result.get("forecast", [])
    if args.horizon_days > 1 and forecast:
        print(f"\n--- {args.horizon_days}-day calendar forecast ---")
        for step in forecast:
            direction = "↑" if step["predicted_direction_up"] else "↓"
            forecast_date = step.get("forecast_date", step.get("date", "N/A"))
            print(f"  Day {step['step']:2d} ({forecast_date})  {step['predicted_price']:8.4f}  {direction}")
        print(f"  [{forecast[-1]['carry_forward_disclosure']}]")

    # Context data availability
    ctx = result.get("context_audit", {})
    if ctx.get("requested"):
        status = ctx.get("status", "unknown")
        print(f"\nContext data: {status}")
        for cat, avail in ctx.get("series", {}).items():
            lineage_type = avail.get("lineage_type", "unknown")
            if avail.get("available"):
                coverage = f"{avail.get('coverage_start')} → {avail.get('coverage_end')}"
                print(f"  {cat}: {lineage_type} | coverage: {coverage}")
            else:
                print(f"  {cat}: unavailable ({avail.get('status')})")

    interp = result.get("interpretation", {})
    if interp:
        print(f"\n--- AI Interpretation ({interp.get('source', 'template')}) ---")
        print(interp.get("summary", ""))

    if result.get("correlation_audit", {}).get("significant_partners_for_target"):
        partners = result["correlation_audit"]["significant_partners_for_target"]
        print(f"\nSignificant correlated commodities: {', '.join(partners)}")
