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

# Conservative stale-data default: warn at 7 days, allow user to enforce failure.
DEFAULT_MAX_STALENESS_DAYS = 7


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
  # Brent crude oil, base features — live refresh (default)
  python scripts\\train_model.py

  # Offline mode: use validated local cache only, no network access
  python scripts\\train_model.py --offline

  # Fail if data is more than 3 days stale
  python scripts\\train_model.py --max-staleness-days 3

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
        "--offline",
        action="store_true",
        help=(
            "Offline mode: skip network access and use only the local validated cache. "
            "The cache must exist and carry a matching .cache_meta.json sidecar. "
            "Use this when network access is unavailable or unwanted. "
            "Data may be stale; staleness is reported explicitly."
        ),
    )
    parser.add_argument(
        "--max-staleness-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Fail with a non-zero exit code if the latest observation in the loaded "
            f"dataset is more than N calendar days old relative to today. "
            f"Default: warn at {DEFAULT_MAX_STALENESS_DAYS} days but do not fail. "
            "Set to 0 to require same-day data. "
            "Note: upstream publication schedules mean same-day data is not always available."
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
        offline=args.offline,
    )

    ma = result["model_audit"]
    da = result["dataset_audit"]
    print("Commodity:", ma.get("commodity_name", args.commodity))
    print("Category:", ma.get("category", "unknown"))
    print("Model:", ma.get("model", "ridge"))
    print("Feature flags:", ma.get("feature_flags", "base"))

    # --- Data freshness block ---
    source_mode = da.get("source_mode", "unknown")
    retrieved_at = da.get("retrieved_at", "")
    latest_obs = da.get("latest_obs_date", da.get("date_range", {}).get("end", ""))
    staleness = da.get("staleness_days", -1)
    refresh_warning = da.get("refresh_warning", "")

    print(f"Data source: {da.get('source_type', 'unknown')} ({source_mode})")
    if retrieved_at:
        print(f"Retrieved at: {retrieved_at}")
    print(f"Latest observation: {latest_obs}")
    if staleness >= 0:
        stale_label = f"{staleness} calendar day(s) old"
        warn_threshold = DEFAULT_MAX_STALENESS_DAYS
        if staleness >= warn_threshold:
            stale_label += f"  *** DATA IS STALE (>{warn_threshold} days) ***"
        print(f"Staleness: {stale_label}")
    if refresh_warning:
        print(f"[WARNING] {refresh_warning}", file=sys.stderr)

    # Enforce --max-staleness-days failure
    if args.max_staleness_days is not None and staleness >= 0 and staleness > args.max_staleness_days:
        print(
            f"[ERROR] Data is {staleness} calendar day(s) old, which exceeds "
            f"--max-staleness-days {args.max_staleness_days}. Aborting.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print("Dataset rows:", da.get("row_count", result.get("row_count", "?")))
    print(
        "Dataset date range:",
        da.get("date_range", {}).get("start", ""),
        "to",
        da.get("date_range", {}).get("end", ""),
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
