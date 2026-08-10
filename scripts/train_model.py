import argparse
import json
from pathlib import Path

from oilenergy import run_pipeline, run_backtest
from oilenergy.commodities import list_commodities, load_commodity


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

  # Brent — 5-day ahead forecast (Phase 2 multi-horizon)
  python scripts/train_model.py --commodity brent --horizon 5

  # Brent — walk-forward backtest (Phase 2)
  python scripts/train_model.py --commodity brent --backtest

  # Walk-forward backtest with seasonality features and 5-day horizon
  python scripts/train_model.py --commodity brent --backtest --features seasonality --horizon 5

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
        "--horizon",
        type=int,
        default=1,
        metavar="N",
        help=(
            "[Phase 2] Number of trading sessions ahead to forecast (default: 1). "
            "Common values: 1 (next session), 5 (one week), 10 (two weeks), 21 (one month)."
        ),
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help=(
            "[Phase 2] Run walk-forward (rolling-window) backtest instead of the standard "
            "single 80/20 train-test split. Writes audits/backtest_audit.json."
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

    if args.horizon < 1:
        print("Error: --horizon must be a positive integer.")
        raise SystemExit(1)

    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------------------------------------------------
    # Phase 2: Walk-forward backtest mode
    # -----------------------------------------------------------------------
    if args.backtest:
        feature_flags = {f.strip().lower() for f in args.features.split(",")}
        use_seasonality = "seasonality" in feature_flags or "all" in feature_flags
        cache_dir = project_root / "data" / "raw"
        audits_dir = project_root / "audits"

        print(f"Running walk-forward backtest for {args.commodity} (horizon={args.horizon} days) …")
        commodity_data = load_commodity(args.commodity, cache_dir=cache_dir)
        backtest_result = run_backtest(
            commodity_data.rows,
            min_train=500,
            step=100,
            horizon=args.horizon,
            use_seasonality=use_seasonality,
            max_train_window=2000,
        )

        bt_metrics = backtest_result["metrics"]
        print(f"\nWalk-forward backtest results ({bt_metrics['n_steps']} steps):")
        print(f"  MAE:                  {bt_metrics['mae']}")
        print(f"  RMSE:                 {bt_metrics['rmse']}")
        print(f"  Directional accuracy: {bt_metrics['directional_accuracy']}")

        # Write audit
        audits_dir.mkdir(parents=True, exist_ok=True)
        backtest_audit = {
            "commodity": args.commodity,
            "feature_flags": args.features,
            **backtest_result,
        }
        audit_path = audits_dir / "backtest_audit.json"
        audit_path.write_text(json.dumps(backtest_audit, indent=2), encoding="utf-8")
        print(f"\nBacktest audit written to {audit_path.relative_to(project_root)}")
        raise SystemExit(0)

    # -----------------------------------------------------------------------
    # Standard pipeline (Phase 1 + Phase 2 horizon extension)
    # -----------------------------------------------------------------------
    result = run_pipeline(
        project_root,
        commodity=args.commodity,
        features=args.features,
        horizon=args.horizon,
    )

    print("Commodity:", result["model_audit"].get("commodity_name", args.commodity))
    print("Feature flags:", result["model_audit"].get("feature_flags", "base"))
    print("Forecast horizon:", result["model_audit"].get("forecast_horizon_days", 1), "day(s)")
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


