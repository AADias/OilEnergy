# OilEnergy — Middle East Energy Forecasting

A quantitative energy price forecasting pipeline with a focus on **Qatar and
the Middle East energy sector**. Built for regional energy professionals who
want to apply machine learning to commodity price prediction without needing a
dedicated data-science team.

Supports Brent crude, WTI, Henry Hub, Qatar LNG (proxy), and OPEC Basket — all
using only publicly available, free data sources.

---

## Quick Start (Windows / Mac / Linux)

```bash
# Clone the repository
git clone https://github.com/AADias/OilEnergy.git
cd OilEnergy

# Brent crude oil — default (backwards-compatible)
set PYTHONPATH=src && python scripts/train_model.py           # Windows
PYTHONPATH=src python3 scripts/train_model.py                 # Mac/Linux

# Qatar LNG with seasonality features
set PYTHONPATH=src && python scripts/train_model.py --commodity qatar_lng --features seasonality

# Brent with cross-commodity correlations + seasonality
set PYTHONPATH=src && python scripts/train_model.py --commodity brent --features all

# List all available commodities
set PYTHONPATH=src && python scripts/train_model.py --list-commodities
```

---

## Commodities Supported

| Key           | Name                     | Type | Region        |
|---------------|--------------------------|------|---------------|
| `brent`       | Brent Crude Oil          | oil  | Global        |
| `wti`         | WTI Crude Oil            | oil  | US            |
| `henry_hub`   | Henry Hub Natural Gas    | gas  | US            |
| `qatar_lng`   | Qatar LNG (proxy)        | gas  | Middle East   |
| `opec_basket` | OPEC Reference Basket    | oil  | Middle East   |

See [`config/commodities.yaml`](config/commodities.yaml) to add your own.

---

## Feature Flags

Pass `--features` with one or more of:

| Flag           | What it adds |
|----------------|-------------|
| `base`         | Price lags, rolling mean, momentum, volatility (default) |
| `seasonality`  | Month, quarter, day-of-week, heating/cooling season indicators |
| `external`     | Cross-commodity prices that correlate with the target |
| `all`          | All of the above |

Example:
```bash
python scripts/train_model.py --commodity qatar_lng --features seasonality,external
```

---

## AI-Powered Result Interpretation (HuggingFace)

The pipeline generates an intelligent, professional summary of each forecast
using the **HuggingFace Inference API** (`facebook/bart-large-cnn`). If no
API token is provided, a template-based summary is used instead.

### Setup

1. Get a free token at <https://huggingface.co/settings/tokens>
2. Copy `.env.example` to `.env` and add your token:

```bash
cp .env.example .env
# Edit .env and set HUGGINGFACE_API_KEY=hf_your_token_here
```

The `.env` file is listed in `.gitignore` and will not be committed.

---

## Project Structure

```
OilEnergy/
├── src/oilenergy/
│   ├── pipeline.py           — Core ML pipeline (feature engineering, ridge regression)
│   ├── commodities.py        — Commodity definitions and multi-source data loading
│   ├── correlation_analysis.py — Cross-commodity Pearson correlation
│   ├── external_features.py  — Enrich samples with correlated commodity prices
│   └── llm_interpreter.py    — HuggingFace AI interpretation of results
├── scripts/
│   └── train_model.py        — CLI entry point (--commodity, --features flags)
├── config/
│   └── commodities.yaml      — User-configurable commodity sources
├── docs/
│   └── MIDDLE_EAST_FOCUS.md  — Middle East energy sector positioning & case studies
├── data/raw/                 — Downloaded datasets (auto-created on first run)
├── artifacts/                — model.json, test_predictions.csv
└── audits/                   — data_audit.json, model_audit.json, correlation_audit.json,
                                manual_verification.md
```

---

## Dataset Sources

| Commodity       | Source                              | Data Lineage |
|-----------------|-------------------------------------|--------------|
| Brent Crude     | datasets/oil-prices (GitHub/EIA)   | Free, public CSV |
| WTI Crude       | FRED `DCOILWTICO`                  | EIA via Federal Reserve |
| Henry Hub Gas   | FRED `DHHNGSP`                     | EIA via Federal Reserve |
| Qatar LNG       | FRED `DHHNGSP` (proxy)             | Henry Hub — see [docs](docs/MIDDLE_EAST_FOCUS.md) for why |
| OPEC Basket     | FRED `DCOILBRENTEU` (proxy)        | OPEC basket ≈ Brent in practice |

All sources are **free and publicly available**. No API keys are required for
data fetching (only HuggingFace token for AI summaries).

---

## Manual Verification / Audits

Every run writes audit files to `audits/`:

| File | Contents |
|------|---------|
| `data_audit.json` | Source URL, SHA256 hash, row count, date range, data lineage |
| `model_audit.json` | Feature names, coefficients, train/test counts, MAE, RMSE, directional accuracy |
| `correlation_audit.json` | Pairwise Pearson correlations, significant partners (when `--features external`) |
| `manual_verification.md` | Human-readable checklist + AI interpretation summary |

---

## For Middle East Energy Professionals

See [`docs/MIDDLE_EAST_FOCUS.md`](docs/MIDDLE_EAST_FOCUS.md) for:
- Why this tool is designed for the Middle East energy sector
- Qatar LNG forecasting case study
- How to adapt for Saudi Arabia, UAE, and other OPEC members
- Data source guidance and limitations

---

## Disclaimer

This tool is built for research and educational purposes. Forecasts do not
constitute financial or trading advice. Past model accuracy does not guarantee
future performance.
