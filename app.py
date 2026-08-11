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

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


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
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

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

        # Notebook: Results tab + Chart tab
        notebook = ttk.Notebook(frame)
        notebook.grid(column=0, row=7, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)

        # Results text tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        output = tk.Text(results_frame, width=100, height=18, wrap="none")
        output.pack(fill="both", expand=True)
        self.output_widget = output

        # Chart tab
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="Price Chart")
        if _MATPLOTLIB_AVAILABLE:
            self._fig = Figure(figsize=(10, 4), tight_layout=True)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self._ax.set_title("Run a forecast to see the price chart")
            self._ax.set_xlabel("Date")
            self._ax.set_ylabel("Price (USD)")
            self._canvas.draw()
        else:
            ttk.Label(chart_frame, text="matplotlib not installed — pip install matplotlib").pack()

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
        self._update_chart(result)

    def _update_chart(self, result: dict) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            return
        model_audit = result["model_audit"]
        dataset_audit = result.get("dataset_audit", {})
        commodity_name = model_audit.get("commodity_name", "")

        # Collect historical price data from test predictions (last 90 rows)
        try:
            import csv as csv_mod
            pred_path = self.project_root / "artifacts" / "test_predictions.csv"
            hist_dates: list[str] = []
            hist_prices: list[float] = []
            with pred_path.open(newline="", encoding="utf-8") as fh:
                reader = csv_mod.DictReader(fh)
                rows_all = list(reader)
            rows_tail = rows_all[-90:]
            for r in rows_tail:
                hist_dates.append(r["date"])
                hist_prices.append(float(r["current_price"]))
            # Append the last actual price as anchor for the forecast
            if rows_all:
                last = rows_all[-1]
                hist_dates.append(last["date"])
                hist_prices.append(float(last["actual_next_price"]))
        except Exception:
            hist_dates = []
            hist_prices = []

        forecast = model_audit.get("multi_day_forecast", [])

        self._ax.clear()

        # Plot historical prices
        if hist_dates:
            step = max(1, len(hist_dates) // 8)
            tick_positions = list(range(0, len(hist_dates), step))
            tick_labels = [hist_dates[i] for i in tick_positions]
            self._ax.plot(range(len(hist_dates)), hist_prices, color="#1f77b4", linewidth=1.2, label="Historical")
            self._ax.set_xticks(tick_positions)
            self._ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)

        # Plot forecast as continuation
        if forecast and hist_prices:
            n_hist = len(hist_prices)
            fc_x = list(range(n_hist - 1, n_hist - 1 + len(forecast)))
            fc_y = [hist_prices[-1]] + [f["predicted_price"] for f in forecast]
            fc_x2 = list(range(n_hist - 1, n_hist + len(forecast)))
            self._ax.plot(fc_x2, fc_y, color="#d62728", linewidth=1.5, linestyle="--", marker="o",
                          markersize=4, label=f"Forecast ({len(forecast)}d)")
            # Add forecast date labels
            for i, f in enumerate(forecast):
                self._ax.annotate(
                    f["forecast_date"],
                    xy=(n_hist - 1 + i + 1, f["predicted_price"]),
                    fontsize=6,
                    rotation=30,
                    ha="left",
                    color="#d62728",
                )

        self._ax.set_title(f"{commodity_name} — Price History & Forecast")
        self._ax.set_ylabel("Price (USD)")
        self._ax.legend(fontsize=8)
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()

    def _set_output(self, content: str) -> None:
        self.output_widget.delete("1.0", tk.END)
        self.output_widget.insert(tk.END, content)


if __name__ == "__main__":
    root = tk.Tk()
    app = OilEnergyApp(root)
    root.geometry("1100x780")
    root.mainloop()
