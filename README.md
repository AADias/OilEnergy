# OilEnergy — Middle East Energy Forecasting

A quantitative energy price forecasting pipeline focused on **Qatar and the
Middle East energy sector**. Built for regional energy professionals who want
to apply machine learning to commodity price prediction with no data-science
team required.

Supports Brent crude, WTI, Henry Hub, Qatar LNG (proxy), and OPEC Basket —
using only publicly available, free data sources with zero mandatory
third-party dependencies for the core CLI.

---

## Quick Start (Windows)

```bat
git clone https://github.com/AADias/OilEnergy.git
cd OilEnergy

:: Brent crude oil — default run (no PYTHONPATH needed from repo root)
python scripts\train_model.py

:: Qatar LNG with seasonality features
python scripts\train_model.py --commodity qatar_lng --features seasonality

:: 7-day recursive calendar forecast, naive persistence baseline
python scripts\train_model.py --model naive --horizon-days 7

:: Brent with cross-commodity correlations + seasonality
python scripts\train_model.py --commodity brent --features all

:: List all available commodities
python scripts\train_model.py --list-commodities
```

> **Note:** `python scripts\train_model.py` works directly from the repository
> root on Windows — no environment variables required.

### Mac / Linux

```bash
PYTHONPATH=src python3 scripts/train_model.py
PYTHONPATH=src python3 scripts/train_model.py --commodity qatar_lng --features seasonality
```

---

## Commodities Supported

| Key           | Name                          | Type | Region      | Proxy? |
|---------------|-------------------------------|------|-------------|--------|
| `brent`       | Brent Crude Oil               | oil  | Global      | No     |
| `wti`         | WTI Crude Oil                 | oil  | US          | No     |
| `henry_hub`   | Henry Hub Natural Gas         | gas  | US          | No     |
| `qatar_lng`   | **Qatar LNG (Henry Hub proxy)** | gas | Middle East | **Yes** |
| `opec_basket` | OPEC Reference Basket (Brent proxy) | oil | Middle East | **Yes** |

> **Qatar LNG proxy disclosure:** Official QatarEnergy pricing is not publicly
> available in machine-readable form. Henry Hub (FRED/DHHNGSP) is used as a
> directional proxy. For production use, replace with contracted QP pricing.
> See [`docs/MIDDLE_EAST_FOCUS.md`](docs/MIDDLE_EAST_FOCUS.md) for details.

See [`config/commodities.yaml`](config/commodities.yaml) to add your own sources.

---

## CLI Flags

```
python scripts\train_model.py [options]

  --commodity KEY         Commodity to forecast (default: brent)
  --features FLAGS        Comma-separated feature groups (default: base)
  --model MODEL           ridge (default) or naive (persistence baseline)
  --horizon-days N        Calendar days to forecast ahead (default: 1)
  --category {oil,gas}    Category label for auditing/display (optional)
  --list-commodities      Print available commodities and exit
```

### Feature Flags

| Flag           | What it adds |
|----------------|-------------|
| `base`         | Price lags, rolling mean, momentum, volatility (default) |
| `seasonality`  | Month, quarter, day-of-week, heating/cooling season indicators |
| `external`     | Cross-commodity prices that are independently correlated with the target |
| `context`      | Local weather/demand context (requires `data/context/*.csv` — see below) |
| `all`          | All of the above |

> **External feature safeguard:** Commodities that share the same underlying
> data source (e.g. `qatar_lng` and `henry_hub` both use FRED/DHHNGSP) are
> automatically excluded from external feature sets to prevent spurious
> near-1.0 correlations from contaminating the model.

---

## Context Data Contract (Weather & Demand)

Context features (`--features context`) require real, user-provided CSV files
in `data/context/`:

```
data/context/weather.csv   — daily mean temperature (°C), columns: date, value[, lineage]
data/context/demand.csv    — daily cooling/heating degree-day proxy, same columns
```

**To fetch real Doha weather data:**
```bat
python scripts\fetch_weather.py
```

**Important rules:**
- Context features are applied **only for dates within the explicit coverage
  window of the CSV**. Training samples outside that window are excluded to
  prevent backfill and look-ahead leakage.
- The pipeline does NOT substitute zeros or synthetic values for missing dates.
- The `lineage` column in the CSV distinguishes `real` (Open-Meteo API),
  `demo` (climate normals), and `user_provided` data. The audit trail records
  which type was used.
- `fetch_weather.py` exits with an error on network failure — it does **not**
  fall back to synthetic data automatically. To generate clearly-labelled demo
  data for testing, use `--demo-data` explicitly.

---

## Multi-Day Forecasting

```bat
:: 7-day recursive forecast
python scripts\train_model.py --horizon-days 7

:: 5-day forecast with seasonality
python scripts\train_model.py --features seasonality --horizon-days 5 --model naive
```

> **Disclosure:** Multi-day recursive forecasts feed each predicted price back
> as input for the next step. External/context series are held at their last
> known value — no future observations are used. Forecast dates are **calendar
> days** and may include weekends and holidays.

---

## Models

| Model   | Description |
|---------|-------------|
| `ridge` | Regularised linear regression on price lags and feature groups (default) |
| `naive` | Persistence baseline — predicts last known price unchanged |

---

## AI-Powered Interpretation (Optional)

The pipeline generates a professional summary using the HuggingFace Inference
API (`facebook/bart-large-cnn`). If no token is provided, a template summary
is used instead.

```bat
copy .env.example .env
:: Edit .env and set HUGGINGFACE_API_KEY=hf_your_token
```

---

## Running Tests

```bat
python -m unittest discover tests\
```

Tests use only the Python standard library — no `pytest` or other test
frameworks required.

---

## Project Structure

```
OilEnergy/
├── src/oilenergy/
│   ├── pipeline.py           — Core ML pipeline (ridge regression, recursive forecast)
│   ├── commodities.py        — Commodity definitions and multi-source data loading
│   ├── correlation_analysis.py — Cross-commodity Pearson correlation + proxy exclusion
│   ├── context_features.py   — Weather/demand context loading with lineage tracking
│   ├── external_features.py  — Enrich samples with correlated commodity prices
│   └── llm_interpreter.py    — HuggingFace AI interpretation of results
├── scripts/
│   ├── train_model.py        — CLI entry point
│   └── fetch_weather.py      — Fetch real Doha weather from Open-Meteo (optional)
├── tests/
│   └── test_pipeline.py      — Standard-library unit tests
├── config/
│   └── commodities.yaml      — User-configurable commodity sources
├── docs/
│   └── MIDDLE_EAST_FOCUS.md  — Middle East energy sector positioning & case studies
├── data/
│   ├── raw/                  — Downloaded price datasets (auto-created)
│   └── context/              — Optional user-provided weather/demand CSVs
└── Dockerfile                — Container (core CLI only; mount context data externally)
```

> `artifacts/` and `audits/` are in `.gitignore` — generated on each run, not committed.

---

## Dataset Sources

| Commodity       | Source                              | Independent of other listed commodities? |
|-----------------|-------------------------------------|------------------------------------------|
| Brent Crude     | datasets/oil-prices (GitHub/EIA)   | Yes |
| WTI Crude       | FRED `DCOILWTICO`                  | Yes |
| Henry Hub Gas   | FRED `DHHNGSP`                     | Yes |
| Qatar LNG       | FRED `DHHNGSP` (**proxy**)         | **No** — same series as Henry Hub |
| OPEC Basket     | FRED `DCOILBRENTEU` (**proxy**)    | **No** — same series as Brent |

---

## Disclaimer

This tool is for research and educational purposes. Forecasts do not constitute
financial or trading advice. Past model accuracy does not guarantee future
performance.
