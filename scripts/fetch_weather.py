#!/usr/bin/env python3
"""fetch_weather.py — Download real weather data and compute demand proxies.

Fetches daily mean temperature for Doha, Qatar from the Open-Meteo API
(free, no API key required) and writes two context CSV files:

  data/context/weather.csv   — daily mean temperature in °C
  data/context/demand.csv    — daily cooling/heating degree-day proxy
                               (positive value = cooling demand, negative = heating)

If the network is unavailable, the script falls back to generating
statistically representative synthetic data based on Doha's historical
climate averages.

Usage:
  python scripts/fetch_weather.py                        # default: 2020-01-01 to today
  python scripts/fetch_weather.py --start 2022-01-01    # custom start date
  python scripts/fetch_weather.py --synthetic           # force synthetic data generation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

# Doha monthly mean temperature (°C) based on historical climate data
# Source: worldweatheronline.com / WMO climate normals
DOHA_MONTHLY_MEAN_TEMP = {
    1: 18.8,
    2: 20.2,
    3: 24.0,
    4: 28.9,
    5: 34.2,
    6: 36.6,
    7: 38.4,
    8: 38.7,
    9: 36.7,
    10: 31.9,
    11: 25.5,
    12: 20.4,
}

# Monthly standard deviation (°C) — how much daily temperatures vary
DOHA_MONTHLY_STD = {
    1: 3.2,
    2: 3.4,
    3: 3.8,
    4: 3.9,
    5: 3.7,
    6: 3.1,
    7: 2.9,
    8: 2.8,
    9: 3.0,
    10: 3.5,
    11: 3.6,
    12: 3.3,
}

# Comfort base temperature (°C): days above this create cooling demand,
# days below this create heating demand — used for the demand proxy.
BASE_TEMP = 18.0


def _open_meteo_url(start: str, end: str) -> str:
    return (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={DOHA_LAT}&longitude={DOHA_LON}"
        f"&daily=temperature_2m_mean"
        f"&start_date={start}&end_date={end}"
        f"&timezone=UTC"
    )


def fetch_from_api(start: str, end: str, timeout: int = 20) -> list[tuple[str, float]]:
    """Fetch daily mean temperatures from Open-Meteo API.

    Returns a list of (date_str, temperature_celsius) tuples.
    Raises RuntimeError on network or parsing failure.
    """
    url = _open_meteo_url(start, end)
    req = Request(url, headers={"User-Agent": "OilEnergy/1.0 (weather fetch for energy research)"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    dates = payload["daily"]["time"]
    temps = payload["daily"]["temperature_2m_mean"]
    if len(dates) != len(temps):
        raise RuntimeError("Mismatched date/temperature array lengths in Open-Meteo response.")
    return [(d, float(t)) for d, t in zip(dates, temps) if t is not None]


def generate_synthetic(start: date, end: date, seed: int = 42) -> list[tuple[str, float]]:
    """Generate statistically representative synthetic daily temperatures.

    Uses Doha climate normals with a random Gaussian perturbation per day.
    Temperature values are realistic but not historically accurate.
    """
    rng = random.Random(seed)
    result: list[tuple[str, float]] = []
    current = start
    while current <= end:
        mean = DOHA_MONTHLY_MEAN_TEMP[current.month]
        std = DOHA_MONTHLY_STD[current.month]
        temp = mean + rng.gauss(0.0, std)
        result.append((current.isoformat(), round(temp, 2)))
        current += timedelta(days=1)
    return result


def compute_demand_proxy(weather_rows: list[tuple[str, float]], base_temp: float = BASE_TEMP) -> list[tuple[str, float]]:
    """Convert temperature to a cooling/heating degree-day demand proxy.

    Positive = cooling demand (temperature above base).
    Negative = heating demand (temperature below base).
    The magnitude represents the energy demand intensity.
    """
    return [(d, round(t - base_temp, 4)) for d, t in weather_rows]


def write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "value"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data (skip network fetch)")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    weather_path = project_root / "data" / "context" / "weather.csv"
    demand_path = project_root / "data" / "context" / "demand.csv"

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    weather_rows: list[tuple[str, float]] = []
    source = "synthetic"

    if not args.synthetic:
        print(f"Fetching real weather data from Open-Meteo API for Doha, Qatar ({args.start} → {args.end}) ...")
        try:
            weather_rows = fetch_from_api(args.start, args.end)
            source = "Open-Meteo API (real data, Doha Qatar)"
            print(f"  ✓ Fetched {len(weather_rows)} days of real temperature data.")
        except Exception as exc:
            print(f"  ✗ Network fetch failed: {exc}")
            print("  → Falling back to synthetic Doha climate data.")

    if not weather_rows:
        weather_rows = generate_synthetic(start_date, end_date)
        print(f"  Generated {len(weather_rows)} days of synthetic Doha climate data.")

    demand_rows = compute_demand_proxy(weather_rows)

    write_csv(weather_path, weather_rows)
    write_csv(demand_path, demand_rows)

    print(f"\nWritten:")
    print(f"  {weather_path.relative_to(project_root)}  ({len(weather_rows)} rows, source: {source})")
    print(f"  {demand_path.relative_to(project_root)}  ({len(demand_rows)} rows — cooling/heating degree-day proxy)")
    print(f"\nWeather coverage: {weather_rows[0][0]} → {weather_rows[-1][0]}")
    min_t = min(t for _, t in weather_rows)
    max_t = max(t for _, t in weather_rows)
    avg_t = sum(t for _, t in weather_rows) / len(weather_rows)
    print(f"Temperature range: {min_t:.1f}°C – {max_t:.1f}°C, mean: {avg_t:.1f}°C")
    if source != "synthetic":
        print("\nData lineage: Open-Meteo API (https://open-meteo.com). Free, no API key required.")
    else:
        print("\nData lineage: Synthetic — based on WMO climate normals for Doha, Qatar.")
        print("  Run this script with network access to replace with real observed data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
