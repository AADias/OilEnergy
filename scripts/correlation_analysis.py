import argparse
from pathlib import Path

from oilenergy.correlation_analysis import analyze_correlations, write_correlation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cross-commodity correlations for OilEnergy.")
    parser.add_argument("--target", default="brent", help="Target commodity key from config.yaml")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Commodity key to compare against the target. Repeat to add more commodities.",
    )
    parser.add_argument(
        "--fill-method",
        choices=["forward_fill", "interpolate"],
        default=None,
        help="How to align non-daily external commodity data to the target commodity dates.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    report = analyze_correlations(
        project_root,
        target_commodity=args.target,
        comparison_commodities=args.compare or None,
        fill_method=args.fill_method,
    )
    report_path = project_root / "artifacts" / "correlation_analysis.json"
    write_correlation_report(report_path, report)
    print("Target commodity:", report["target_commodity"])
    print("Compared commodities:", ", ".join(report["comparison_commodities"]))
    print("Saved correlation report to:", report_path.relative_to(project_root))
    print("High-correlation pairs:", len(report["high_correlation_pairs"]))
