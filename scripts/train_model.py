from pathlib import Path

from oilenergy import run_pipeline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = run_pipeline(project_root)
    print("Dataset rows:", result["dataset_audit"]["row_count"])
    print("Dataset date range:", result["dataset_audit"]["date_range"]["start"], "to", result["dataset_audit"]["date_range"]["end"])
    print("Test MAE:", result["model_audit"]["test_metrics"]["mae"])
    print("Test RMSE:", result["model_audit"]["test_metrics"]["rmse"])
    print("Directional accuracy:", result["model_audit"]["test_metrics"]["directional_accuracy"])
    print("Latest next-price prediction:", result["model_audit"]["latest_prediction"]["predicted_next_price"])

