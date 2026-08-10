# Feature Enrichment Limitations

## Geopolitical events

Binary geopolitical shocks are difficult to automate because the market impact depends on timing, severity, duration, and whether the event is already priced in. Public price series usually react faster than any manually curated event feed, so a naive `0/1` indicator can easily overfit or miss the real effect.

### Manual event annotation approach

If you want to enrich the pipeline manually, create a CSV with one row per event window:

```csv
date,event_name,event_group,event_flag,event_weight,notes
2022-02-24,russia_ukraine_invasion,geopolitics,1,1.0,Initial invasion shock
2023-10-07,middle_east_conflict,geopolitics,1,0.7,Regional escalation affecting shipping risk
```

Suggested usage:

- `event_flag`: binary on/off feature
- `event_weight`: optional severity score
- `event_group`: lets you separate supply shocks from shipping, sanctions, or OPEC policy changes

To integrate this later, load the CSV alongside the commodity rows and align by date in the same way `external_features.py` aligns external commodity series.

## Demand proxies

Demand is often easier to proxy than to observe directly. Useful public candidates include:

- Purchasing Managers' Index (PMI)
- Industrial production indices
- Electricity demand
- Shipping or freight indices
- Heating/cooling degree days and temperature anomalies

Common sources:

- FRED
- World Bank commodity and macro datasets
- National statistical agencies
- Public weather datasets from NOAA, ECMWF, or similar providers

These are not wired in by default because they often use monthly or regional frequencies and need careful alignment to the target commodity horizon.

## Seasonality and weather

The pipeline now includes lightweight seasonality features:

- month of year
- quarter
- day of week
- weekend/holiday flag

These are safe baseline demand-shape proxies, but they are not substitutes for real weather or consumption data.

## Recommended extension path

1. Run `scripts/correlation_analysis.py` to find which existing commodities are correlated enough to test as external features.
2. Add one new feature family at a time.
3. Re-run the audits and compare `significant_features` plus test metrics.
4. Keep manual event or demand-proxy files separate and versioned so the audit trail stays clear.
