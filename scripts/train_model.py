import argparse
from pathlib import Path

from oilenergy import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OilEnergy forecasting pipeline.")
    parser.add_argument("--commodity", default="brent", help="Commodity key from config.yaml")
    parser.add_argument(
        "--external-feature",
        action="append",
        default=[],
        help="Commodity key to add as an external feature. Repeat to add multiple series.",
    )
    parser.add_argument(
        "--fill-method",
        choices=["forward_fill", "interpolate"],
        default=None,
        help="How to align sparse external series such as LNG proxies to the main commodity dates.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    result = run_pipeline(
        project_root,
        commodity_name=args.commodity,
        external_feature_commodities=args.external_feature,
        fill_method=args.fill_method,
    )
    print("Commodity:", result["dataset_audit"]["commodity_display_name"])
    print("Dataset rows:", result["dataset_audit"]["row_count"])
    print("Dataset date range:", result["dataset_audit"]["date_range"]["start"], "to", result["dataset_audit"]["date_range"]["end"])
    print("Test MAE:", result["model_audit"]["test_metrics"]["mae"])
    print("Test RMSE:", result["model_audit"]["test_metrics"]["rmse"])
    print("Directional accuracy:", result["model_audit"]["test_metrics"]["directional_accuracy"])
    if result["model_audit"]["external_features"]:
        print("External features:", ", ".join(result["model_audit"]["external_features"]))
    print("Latest next-price prediction:", result["model_audit"]["latest_prediction"]["predicted_next_price"])
