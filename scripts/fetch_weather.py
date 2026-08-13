#!/usr/bin/env python3
"""fetch_weather.py — Download real weather data and compute demand proxies.

Fetches daily mean temperature for Doha, Qatar from the Open-Meteo API
(free, no API key required) and writes two context CSV files:

  data/context/weather.csv   — daily mean temperature in °C
  data/context/demand.csv    — daily cooling/heating degree-day proxy
                               (positive = cooling demand, negative = heating)

Each output file includes a header row and a ``lineage`` metadata comment at
the top indicating whether the data is real or demo.

On network failure the script exits with a non-zero status and clearly reports
the error.  It does NOT silently substitute synthetic values.

Usage:
  # Fetch real data (default: 2020-01-01 to today)
  python scripts/fetch_weather.py

  # Custom date range
  python scripts/fetch_weather.py --start 2022-01-01 --end 2024-12-31

  # Explicit demo/test mode — clearly labelled, NEVER used in normal operation
  python scripts/fetch_weather.py --demo-data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Doha, Qatar — latitude/longitude for Open-Meteo
# ---------------------------------------------------------------------------
DOHA_LAT = 25.2854
DOHA_LON = 51.5310

# Doha monthly mean temperature (°C) based on WMO climate normals.
# Used ONLY in --demo-data mode.
DOHA_MONTHLY_MEAN_TEMP = {
    1: 18.8, 2: 20.2, 3: 24.0, 4: 28.9, 5: 34.2, 6: 36.6,
    7: 38.4, 8: 38.7, 9: 36.7, 10: 31.9, 11: 25.5, 12: 20.4,
}
DOHA_MONTHLY_STD = {
    1: 3.2, 2: 3.4, 3: 3.8, 4: 3.9, 5: 3.7, 6: 3.1,
    7: 2.9, 8: 2.8, 9: 3.0, 10: 3.5, 11: 3.6, 12: 3.3,
}

BASE_TEMP = 18.0  # Comfort base temperature (°C)


def _open_meteo_url(start: str, end: str) -> str:
    return (
        f"https://api.open-meteo.com/v1/archive"
        f"?latitude={DOHA_LAT}&longitude={DOHA_LON}"
        f"&daily=temperature_2m_mean"
        f"&start_date={start}&end_date={end}"
        f"&timezone=UTC"
    )


def fetch_from_api(start: str, end: str, timeout: int = 30) -> list[tuple[str, float]]:
    """Fetch daily mean temperatures from Open-Meteo archive API.

    Returns a list of (date_str, temperature_celsius) tuples.
    Raises RuntimeError on network or parsing failure.
    """
    url = _open_meteo_url(start, end)
    req = Request(url, headers={"User-Agent": "OilEnergy/1.0 (energy research)"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    dates = payload["daily"]["time"]
    temps = payload["daily"]["temperature_2m_mean"]
    if len(dates) != len(temps):
        raise RuntimeError("Mismatched date/temperature lengths in Open-Meteo response.")
    return [(d, float(t)) for d, t in zip(dates, temps) if t is not None]


def generate_demo_data(start: date, end: date, seed: int = 42) -> list[tuple[str, float]]:
    """Generate labelled demo-only temperature data.

    Uses Doha climate normals with Gaussian noise.  Values are NOT historically
    accurate and must ONLY be used in explicitly named demo/test workflows.
    """
    rng = random.Random(seed)
    result: list[tuple[str, float]] = []
    current = start
    while current <= end:
        mean = DOHA_MONTHLY_MEAN_TEMP[current.month]
        std = DOHA_MONTHLY_STD[current.month]
        result.append((current.isoformat(), round(mean + rng.gauss(0.0, std), 2)))
        current += timedelta(days=1)
    return result


def compute_demand_proxy(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Cooling/heating degree-day proxy: positive = cooling, negative = heating."""
    return [(d, round(t - BASE_TEMP, 4)) for d, t in rows]


def write_csv(path: Path, rows: list[tuple[str, float]], lineage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "value", "lineage"])
        for d, v in rows:
            writer.writerow([d, v, lineage])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help=(
            "Write clearly-labelled demo/test data instead of fetching real data. "
            "Values are NOT historically accurate. Never use for production forecasts."
        ),
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    weather_path = project_root / "data" / "context" / "weather.csv"
    demand_path = project_root / "data" / "context" / "demand.csv"

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if args.demo_data:
        print("WARNING: --demo-data mode.  Output is NOT real historical data.")
        print("         Use only for testing/demonstration.  Never for production forecasts.")
        weather_rows = generate_demo_data(start_date, end_date)
        lineage = "DEMO — synthetic Doha climate normals (WMO), NOT real observations"
        source_label = "demo"
    else:
        print(f"Fetching real weather data from Open-Meteo archive for Doha, Qatar ({args.start} → {args.end}) …")
        try:
            weather_rows = fetch_from_api(args.start, args.end)
            lineage = "Open-Meteo archive API (real observations, Doha Qatar)"
            source_label = "real"
            print(f"  ✓ Fetched {len(weather_rows)} days of real temperature data.")
        except Exception as exc:
            print(f"\nERROR: Failed to fetch weather data: {exc}", file=sys.stderr)
            print(
                "Context data is unavailable.  The pipeline will run without weather/demand features.\n"
                "To use demo data for testing, pass --demo-data explicitly.",
                file=sys.stderr,
            )
            return 1

    demand_rows = compute_demand_proxy(weather_rows)

    write_csv(weather_path, weather_rows, lineage)
    write_csv(demand_path, demand_rows, lineage)

    print(f"\nWritten ({source_label}):")
    print(f"  {weather_path.relative_to(project_root)}  ({len(weather_rows)} rows)")
    print(f"  {demand_path.relative_to(project_root)}  ({len(demand_rows)} rows — cooling/heating degree-day proxy)")
    print(f"\nCoverage: {weather_rows[0][0]} → {weather_rows[-1][0]}")
    print(f"Lineage:  {lineage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
