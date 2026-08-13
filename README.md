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

:: Brent crude oil — live refresh (default; fetches latest data before forecasting)
python scripts\train_model.py

:: Intuitive shortcuts using --asset
python scripts\train_model.py --asset oil
python scripts\train_model.py --asset gas --horizon-days 7
python scripts\train_model.py --asset lng

:: Offline mode — use validated local cache only, no network access
python scripts\train_model.py --offline

:: Fail if data is more than 3 calendar days old
python scripts\train_model.py --max-staleness-days 3

:: Qatar LNG with seasonality features (live refresh, proxy labelled)
python scripts\train_model.py --commodity qatar_lng --features seasonality

:: 7-day recursive calendar forecast, naive persistence baseline
python scripts\train_model.py --model naive --horizon-days 7

:: Brent with cross-commodity correlations + seasonality
python scripts\train_model.py --commodity brent --features all

:: List all available commodities and their aliases
python scripts\train_model.py --list-commodities
```

> **Note:** `python scripts\train_model.py` works directly from the repository
> root on Windows — no environment variables required.

### Mac / Linux

```bash
PYTHONPATH=src python3 scripts/train_model.py
PYTHONPATH=src python3 scripts/train_model.py --offline
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

  --asset ALIAS              Intuitive shortcut: oil, brent, crude, gas, natgas, lng, qatar …
                             Equivalent to --commodity but easier to type.
  --commodity KEY            Commodity key to forecast (default: brent).
                             Canonical keys: brent, wti, henry_hub, qatar_lng, opec_basket …
  --features FLAGS           Comma-separated feature groups (default: base)
  --model MODEL              ridge (default) or naive (persistence baseline)
  --horizon-days N           Calendar days to forecast ahead, 1-30 (default: 1)
  --category {oil,gas}       Category label for auditing/display (optional)
  --offline                  Skip network access; use only validated local cache.
                             Fails clearly if no valid cache exists.
  --max-staleness-days N     Fail with exit code 2 if the latest observation is
                             more than N calendar days old. Default: warn at 7 days
                             but do not fail. Use 0 to require same-day data.
                             Note: upstream publication schedules mean same-day
                             data is not always available from FRED or GitHub.
  --list-commodities         Print available commodities with aliases and examples, then exit
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

## Data Freshness and Staleness Policy

Every run reports:

| Output field         | Meaning |
|----------------------|---------|
| `Data source`        | Provider and mode (`live`, `cache`, `demo`, `proxy`, `unavailable`) |
| `Retrieved at`       | UTC timestamp of the data retrieval |
| `Latest observation` | Date of the last observation in the loaded dataset |
| `Staleness`          | Calendar days between the latest observation and today |

**Default behaviour (live refresh):**
- The CLI attempts a live refresh from the declared source (FRED or CSV URL)
  before forecasting. If the refresh succeeds, `source_mode=live` is reported.
- If the refresh fails, the CLI falls back to the most recent validated local
  cache and reports `source_mode=cache` with a prominent `[STALE DATA]` warning
  including the refresh error and the data age.
- If no valid cache exists, the run fails with a clear error message.

**Conservative stale-data policy:**
- A warning is printed for data ≥ **7 calendar days** old regardless of whether
  the weekend or a public holiday explains the gap.
- Use `--max-staleness-days N` to enforce a hard failure for data older than N
  days (exit code 2). Setting `--max-staleness-days 3` is recommended for
  production use.
- The CLI never claims data is current when it is not. Availability follows the
  upstream publication schedule; FRED daily series are typically published 1–2
  business days behind.

**Offline mode (`--offline`):**
- Skips all network access. Uses only the validated local cache (`.cache_meta.json`
  sidecar required). Fails clearly if no valid cache exists.

**Cache integrity:**
- Every cached CSV is accompanied by a `.cache_meta.json` sidecar that records
  the commodity key, canonical series ID, source type, and retrieval timestamp.
- On read the sidecar is validated. A cache with a missing or mismatched sidecar
  is rejected — it will never silently serve a different commodity.
- This specifically prevents Brent (`DCOILBRENTEU`) data from being served for
  Qatar LNG (`DHHNGSP`), which was observed as a contamination issue on
  2026-08-12.

---

## Qatar LNG Proxy Provenance

Qatar LNG (`qatar_lng`) uses **FRED series DHHNGSP** (Henry Hub) as a
directional proxy because official QatarEnergy pricing is not publicly
available in machine-readable form.

**Identity guarantee:** The `qatar_lng` loader always fetches and caches data
under the `DHHNGSP` series identifier. It will never silently fall back to
Brent (`DCOILBRENTEU`) or any other series. If `DHHNGSP` is unreachable and no
valid `qatar_lng` cache exists, the run fails with a clear error rather than
substituting unrelated data.

**Correlation exclusion:** `qatar_lng` and `henry_hub` share the same
underlying series and are automatically excluded from each other's external
feature sets to prevent spurious near-1.0 correlations.

For production use, replace `DHHNGSP` with contracted QatarEnergy pricing data.
See [`docs/MIDDLE_EAST_FOCUS.md`](docs/MIDDLE_EAST_FOCUS.md) for more.

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
