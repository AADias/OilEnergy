# Manual Verification Audit

## Dataset checks
- Source URL: https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv
- Local file: data/raw/brent-daily.csv
- SHA256: b318845619753e4985c9da7004774d71c52575d421df2be414d3e565cc84140d
- Rows: 9947
- Date range: 1987-05-20 to 2026-08-03

## Model checks
- Trained at: 2026-08-10T08:07:18+00:00
- Training samples: 7948
- Test samples: 1988
- Test MAE: 1.40788
- Test RMSE: 2.152758
- Directional accuracy: 0.511066

## Latest model output
- Latest observation date: 2026-08-03
- Latest observation price: 88.9
- Predicted next price: 88.9064
- Predicted direction up: True

## Manual verification steps
- Open the raw CSV and confirm the first and last rows match the data audit JSON.
- Recompute the SHA256 of the raw CSV and confirm it matches the recorded digest.
- Open the predictions CSV and confirm dates are in chronological order.
- Spot-check that predicted_direction_up matches whether predicted_next_price is greater than or equal to current_price.
- Re-run `PYTHONPATH=src python3 scripts/train_model.py` and confirm the audit files refresh successfully.
