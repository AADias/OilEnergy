# OilEnergy

Real-data oil price modeling scaffold built around the public Brent crude daily spot price dataset.

## Dataset

- Source: U.S. Energy Information Administration data packaged by `datasets/oil-prices`
- URL: `https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv`
- Coverage verified in this project run: 1987-05-20 through 2026-08-03

## Project structure

- `/home/runner/work/OilEnergy/OilEnergy/src/oilenergy/pipeline.py` downloads the real dataset, engineers features, trains a model, and writes audits
- `/home/runner/work/OilEnergy/OilEnergy/scripts/train_model.py` runs the pipeline end to end
- `/home/runner/work/OilEnergy/OilEnergy/data/raw/brent-daily.csv` is the downloaded source dataset after a run
- `/home/runner/work/OilEnergy/OilEnergy/artifacts/model.json` stores the trained model coefficients and metrics
- `/home/runner/work/OilEnergy/OilEnergy/artifacts/test_predictions.csv` stores test-set predictions
- `/home/runner/work/OilEnergy/OilEnergy/audits/` contains manual verification artifacts

## Run

```bash
cd /home/runner/work/OilEnergy/OilEnergy
PYTHONPATH=src python3 scripts/train_model.py
```

## Manual verification

After running the script, review:

- `/home/runner/work/OilEnergy/OilEnergy/audits/data_audit.json`
- `/home/runner/work/OilEnergy/OilEnergy/audits/model_audit.json`
- `/home/runner/work/OilEnergy/OilEnergy/audits/manual_verification.md`

These files capture the dataset source, file digest, row counts, date range, model metrics, and a checklist for manual review.
