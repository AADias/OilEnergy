import argparse
from pathlib import Path

from oilenergy import run_pipeline
from oilenergy.commodities import list_commodities


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
        "--features",
        default="base",
        help=(
            "Comma-separated feature flags: "
            "base (default), seasonality, external, all. "
            "Example: --features seasonality,external"
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
    result = run_pipeline(project_root, commodity=args.commodity, features=args.features)

    print("Commodity:", result["model_audit"].get("commodity_name", args.commodity))
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

    interp = result.get("interpretation", {})
    if interp:
        print(f"\n--- AI Interpretation ({interp.get('source', 'template')}) ---")
        print(interp.get("summary", ""))

    if result.get("correlation_audit", {}).get("significant_partners_for_target"):
        partners = result["correlation_audit"]["significant_partners_for_target"]
        print(f"\nSignificant correlated commodities: {', '.join(partners)}")

