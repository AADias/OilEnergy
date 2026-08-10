# OilEnergy

Real-data commodity modeling scaffold built around public Brent, WTI, Henry Hub, and LNG proxy datasets.

## What changed

- Multi-commodity configuration is now centralized in `/home/runner/work/OilEnergy/OilEnergy/config.yaml`
- The same ridge-regression pipeline can run for Brent, WTI, Henry Hub gas, or a Qatar LNG proxy
- Seasonality features are included by default: month, quarter, day of week, and weekend/holiday flag
- Optional external commodity enrichment can add aligned prices and returns from other commodities
- Correlation analysis can recommend which extra commodities are worth adding as features

## Dataset configuration

Commodity definitions live in `config.yaml`. Each entry defines:

- display name
- public CSV source URL
- local raw file name
- date/value columns and date formats
- default model hyperparameters

`qatar_lng` uses a public Asia LNG benchmark proxy because free Qatar spot LNG series are not consistently published.

## Project structure

- `src/oilenergy/pipeline.py` downloads a configured commodity, engineers features, trains a ridge model, and writes audits
- `src/oilenergy/commodities.py` stores commodity definitions and the generalized CSV loader
- `src/oilenergy/external_features.py` aligns external commodity series and adds price/return features
- `src/oilenergy/correlation_analysis.py` computes Pearson/Spearman correlations and feature recommendations
- `scripts/train_model.py` runs the forecasting pipeline end to end
- `scripts/correlation_analysis.py` writes an example correlation report to `artifacts/correlation_analysis.json`
- `docs/feature_limitations.md` documents manual enrichment ideas and limitations

## Run

### Brent baseline

```bash
python -m scripts.train_model
```

### Qatar LNG proxy

```bash
python -m scripts.train_model --commodity qatar_lng
```

### Brent with external commodity features

```bash
python -m scripts.train_model --external-feature wti --external-feature henry_hub --external-feature qatar_lng
```

### Correlation analysis

```bash
python -m scripts.correlation_analysis --target brent --compare wti --compare henry_hub --compare qatar_lng
```

## Manual verification

After running the script, review:

- `audits/data_audit.json`
- `audits/model_audit.json`
- `audits/manual_verification.md`
- `artifacts/correlation_analysis.json` when you run the correlation script

These files capture the dataset source, file digest, row counts, date range, model metrics, feature significance, and a checklist for manual review.

## Notes on feature impact

The pipeline now records `significant_features` in `audits/model_audit.json`, ranked by absolute coefficient magnitude. This gives a simple audit trail for which lag, seasonality, or external commodity inputs are influencing the fitted ridge model most strongly.
