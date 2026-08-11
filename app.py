from __future__ import annotations

import tkinter as tk
import sys
from pathlib import Path
from tkinter import ttk

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oilenergy import run_pipeline
from oilenergy.commodities import list_commodities
from oilenergy.pipeline import AVAILABLE_MODELS


class OilEnergyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OilEnergy Forecast App")
        self.project_root = PROJECT_ROOT

        self.commodities = list_commodities()
        self.commodities_by_category = {
            "oil": [c for c in self.commodities if c["type"] == "oil"],
            "gas": [c for c in self.commodities if c["type"] == "gas"],
        }

        self.category = tk.StringVar(value="oil")
        self.commodity = tk.StringVar(value="brent")
        self.model_name = tk.StringVar(value="ridge")
        self.horizon_days = tk.IntVar(value=3)
        self.use_seasonality = tk.BooleanVar(value=True)
        self.use_weather = tk.BooleanVar(value=False)
        self.use_demand = tk.BooleanVar(value=False)
        self.use_external = tk.BooleanVar(value=False)
        self.allow_demo_fallback = tk.BooleanVar(value=False)
        self.result_text = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(column=0, row=0, sticky="nsew")

        ttk.Label(frame, text="Category").grid(column=0, row=0, sticky="w")
        category_combo = ttk.Combobox(
            frame, textvariable=self.category, values=["oil", "gas"], state="readonly"
        )
        category_combo.grid(column=1, row=0, sticky="ew")
        category_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        ttk.Label(frame, text="Commodity").grid(column=0, row=1, sticky="w")
        self.commodity_combo = ttk.Combobox(frame, textvariable=self.commodity, state="readonly")
        self.commodity_combo.grid(column=1, row=1, sticky="ew")

        ttk.Label(frame, text="Model").grid(column=0, row=2, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.model_name,
            values=sorted(AVAILABLE_MODELS.keys()),
            state="readonly",
        ).grid(column=1, row=2, sticky="ew")

        ttk.Label(frame, text="Horizon Days (1-30)").grid(column=0, row=3, sticky="w")
        ttk.Spinbox(frame, from_=1, to=30, textvariable=self.horizon_days).grid(
            column=1, row=3, sticky="ew"
        )

        ttk.Label(frame, text="Features").grid(column=0, row=4, sticky="w")
        feature_frame = ttk.Frame(frame)
        feature_frame.grid(column=1, row=4, sticky="w")
        ttk.Checkbutton(feature_frame, text="seasonality", variable=self.use_seasonality).grid(
            column=0, row=0, sticky="w"
        )
        ttk.Checkbutton(feature_frame, text="weather", variable=self.use_weather).grid(
            column=1, row=0, sticky="w"
        )
        ttk.Checkbutton(feature_frame, text="demand", variable=self.use_demand).grid(
            column=2, row=0, sticky="w"
        )
        ttk.Checkbutton(feature_frame, text="external correlation", variable=self.use_external).grid(
            column=3, row=0, sticky="w"
        )

        ttk.Checkbutton(
            frame,
            text="Allow demo fallback (explicit proxy substitution)",
            variable=self.allow_demo_fallback,
        ).grid(column=1, row=5, sticky="w")

        ttk.Button(frame, text="Run Forecast", command=self._run_forecast).grid(
            column=1, row=6, sticky="e"
        )

        output = tk.Text(frame, width=100, height=18)
        output.grid(column=0, row=7, columnspan=2, sticky="nsew", pady=(8, 0))
        self.output_widget = output

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)
        self._on_category_change()

    def _on_category_change(self, *_: object) -> None:
        category = self.category.get().strip().lower()
        options = self.commodities_by_category.get(category, [])
        keys = [o["key"] for o in options]
        self.commodity_combo["values"] = keys
        if keys and self.commodity.get() not in keys:
            self.commodity.set(keys[0])

    def _run_forecast(self) -> None:
        try:
            selected_features = ["base"]
            if self.use_seasonality.get():
                selected_features.append("seasonality")
            if self.use_weather.get():
                selected_features.append("weather")
            if self.use_demand.get():
                selected_features.append("demand")
            if self.use_external.get():
                selected_features.append("external")

            result = run_pipeline(
                self.project_root,
                commodity=self.commodity.get(),
                category=self.category.get(),
                model_name=self.model_name.get(),
                horizon_days=int(self.horizon_days.get()),
                features=",".join(selected_features),
                allow_demo_fallback=self.allow_demo_fallback.get(),
            )
        except Exception as exc:
            self._set_output(f"Error: {exc}")
            return

        model_audit = result["model_audit"]
        lines = [
            f"Commodity: {model_audit.get('commodity_name')} ({model_audit.get('commodity_type')})",
            f"Model: {model_audit.get('model_name')}",
            f"Feature flags: {model_audit.get('feature_flags')}",
            f"Proxy commodity: {model_audit.get('is_proxy_commodity')}  Proxy for: {model_audit.get('proxy_for')}",
            f"MAE: {model_audit['test_metrics']['mae']}",
            f"RMSE: {model_audit['test_metrics']['rmse']}",
            f"Directional accuracy: {model_audit['test_metrics']['directional_accuracy']}",
            "",
            "Multi-day forecast:",
        ]
        for row in model_audit.get("multi_day_forecast", []):
            direction = "up" if row["predicted_direction_up_vs_previous_day"] else "down"
            lines.append(
                f"  Day {row['step_day']:2d} {row['forecast_date']}: {row['predicted_price']} ({direction})"
            )
        lines.append("")
        lines.append("Contextual feature availability:")
        for name, status in model_audit.get("contextual_feature_availability", {}).items():
            lines.append(
                f"  - {name}: {'available' if status.get('available') else 'unavailable'} "
                f"({status.get('source_path')})"
            )
        self._set_output("\n".join(lines))

    def _set_output(self, content: str) -> None:
        self.output_widget.delete("1.0", tk.END)
        self.output_widget.insert(tk.END, content)


if __name__ == "__main__":
    root = tk.Tk()
    app = OilEnergyApp(root)
    root.geometry("1000x620")
    root.mainloop()
