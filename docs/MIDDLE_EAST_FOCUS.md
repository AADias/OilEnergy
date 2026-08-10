# Middle East Energy Sector Focus

## Why This Tool Exists

The global energy market is increasingly shaped by Middle East producers.
Qatar is the world's largest LNG exporter, Saudi Arabia anchors OPEC supply
decisions, and the UAE is rapidly diversifying its energy exports. Yet most
open-source energy forecasting tools are built around US and European benchmarks
(Henry Hub, Brent, WTI) with little attention to regional dynamics.

**OilEnergy** is designed as a starting point for **Middle East energy
professionals** who want to apply quantitative forecasting to their own markets
without needing a PhD in machine learning. It runs on a laptop, uses only
publicly available data, and produces auditable, reproducible results.

---

## Commodities Supported

| Key           | Name                     | Type | Data Source               | Notes |
|---------------|--------------------------|------|---------------------------|-------|
| `brent`       | Brent Crude Oil          | oil  | GitHub/EIA CSV            | Global benchmark |
| `wti`         | WTI Crude Oil            | oil  | FRED (DCOILWTICO)         | US benchmark |
| `henry_hub`   | Henry Hub Natural Gas    | gas  | FRED (DHHNGSP)            | US gas benchmark |
| `qatar_lng`   | Qatar LNG (proxy)        | gas  | FRED (DHHNGSP)            | Henry Hub proxy — see note below |
| `opec_basket` | OPEC Reference Basket    | oil  | FRED (DCOILBRENTEU)       | Brent proxy |

### Qatar LNG Data Note

Official QatarEnergy (formerly Qatar Petroleum) pricing data is **not publicly
available** in machine-readable form. QP sells LNG under long-term contracts
that are confidential. The public data that does exist:

- **[OPEC Annual Statistical Bulletin](https://www.opec.org/opec_web/en/data_graphs/40.htm)**
  — includes Qatar production volumes but not spot prices
- **[IEA World Energy Statistics](https://www.iea.org/data-and-statistics)**
  — macroeconomic energy data, not daily spot prices
- **[World Bank Commodity Markets](https://www.worldbank.org/en/research/commodity-markets)**
  — LNG price indices, monthly resolution

For this tool, **Henry Hub (FRED: DHHNGSP)** is used as a directional proxy.
Henry Hub and global LNG prices are correlated on a directional basis (both
respond to demand shocks, supply disruptions, and seasonal patterns) though
the absolute price levels differ significantly.

**For production use:** Replace the `qatar_lng` source in
`config/commodities.yaml` with your actual QP contract price data.

---

## Use Cases for Middle East Professionals

### 1. Qatar LNG Price Direction Forecasting

```bash
python scripts/train_model.py --commodity qatar_lng --features seasonality
```

Output includes:
- Next-session price direction (up/down)
- Directional accuracy on historical data
- Seasonal patterns (heating/cooling season demand)
- AI-generated summary for non-technical stakeholders

### 2. Brent Oil with Cross-Commodity Features

```bash
python scripts/train_model.py --commodity brent --features seasonality,external
```

This fetches WTI, Henry Hub, and other available commodity prices,
computes their correlation with Brent, and automatically includes
highly correlated series as features. The correlation audit is saved
to `audits/correlation_audit.json`.

### 3. OPEC Basket Analysis

```bash
python scripts/train_model.py --commodity opec_basket --features all
```

### 4. Adding a Custom Commodity

Edit `config/commodities.yaml` and add your commodity definition.
Then update `src/oilenergy/commodities.py` `COMMODITIES` dict with the
same entry (the YAML is provided for documentation and future config-driven
loading; the Python dict is the active configuration).

---

## Case Study: Qatar LNG Forecasting

Qatar produces approximately 77–80 million tonnes of LNG per year, making it
the world's largest LNG exporter. The state-owned QatarEnergy supplies
customers in Asia, Europe, and the Americas under long-term contracts.

### Research Question

*Can we predict whether Qatar LNG prices will be higher or lower tomorrow
than today, using only publicly available gas price data?*

### Methodology

1. **Data:** Henry Hub spot price (FRED DHHNGSP) as Qatar LNG proxy
2. **Features:** 9 time-series lags + 5 seasonality features (month, quarter,
   day-of-week, heating season indicator, cooling season indicator)
3. **Model:** Ridge regression (alpha=1.0)
4. **Evaluation:** 80/20 time-ordered train/test split

### Running the Case Study

```bash
# Set up (first time only)
git clone https://github.com/AADias/OilEnergy.git
cd OilEnergy

# Run Qatar LNG forecast
set PYTHONPATH=src && python scripts/train_model.py --commodity qatar_lng --features seasonality

# With AI interpretation (add your HuggingFace token to .env)
echo HUGGINGFACE_API_KEY=your_token_here > .env
set PYTHONPATH=src && python scripts/train_model.py --commodity qatar_lng --features seasonality
```

### Interpreting Results

| Metric | What it means for LNG trading |
|--------|-------------------------------|
| **Directional accuracy ~51–55%** | The model captures market direction slightly better than random chance. Use as one signal among many. |
| **MAE** | Average USD/MMBtu error on the test set. Compare to your contract margin. |
| **Heating season indicator** | Gas demand spikes in winter (Nov–Mar). The model automatically assigns higher weight to this period. |
| **Correlation with Brent** | High oil-gas correlation suggests supply-side linkage. |

---

## How Saudi, UAE, and Other OPEC Members Can Use This

### Saudi Arabia (Arab Light crude)

The closest publicly available proxy is **Brent** or **OPEC Basket**.
Replace `fred_series: "DCOILBRENTEU"` in `config/commodities.yaml`
with a FRED series closer to Arab Light when one becomes available.

### UAE (Dubai/Oman crude)

Dubai crude prices are published by the Dubai Mercantile Exchange (DME).
FRED does not carry DME prices; a web scraper or DME data subscription would
be needed. The tool's architecture supports this — add a `csv_url` pointing
to a downloaded DME price CSV.

### General OPEC Members

OPEC publishes a monthly [Statistical Bulletin](https://www.opec.org/opec_web/static_files_project/media/downloads/publications/ASB2024.pdf)
with production data for all member states. This can be used for longer-horizon
(monthly) supply-side models.

---

## Limitations and Disclaimers

1. **Not financial advice.** This tool is built for research and education.
   Do not use model outputs to make trading or investment decisions.

2. **Proxy data.** Qatar LNG uses Henry Hub as a proxy. Absolute price levels
   will differ from actual QP contract prices.

3. **Ridge regression is a baseline.** The model captures linear patterns in
   historical prices. It does not model geopolitical events, OPEC decisions,
   demand shocks, or structural breaks. For professional-grade forecasting,
   augment with domain knowledge and more sophisticated models.

4. **Past accuracy ≠ future accuracy.** Energy markets are non-stationary.
   A model trained on 2000–2020 data may perform differently in 2025+.

5. **Data latency.** FRED data has a ~1-day lag. Results are for the next
   trading session based on the most recent available observation.

---

## Adding Your Own Data Sources

The tool is designed to be extended. To add a new data source:

1. **Edit `config/commodities.yaml`** — add your commodity entry
2. **Edit `src/oilenergy/commodities.py`** — add the same entry to `COMMODITIES`
3. **Run:**
   ```bash
   python scripts/train_model.py --commodity your_new_commodity --features base
   ```

If your data is in a CSV with date and price columns, use `primary_source: csv`
and set `csv_url` to the local file path or HTTP URL.

---

## Contact & Contributing

This project is open-source and welcomes contributions from the Middle East
energy sector community. If you have access to better Qatar/GCC data sources,
please open a pull request or issue.

Repository: https://github.com/AADias/OilEnergy
