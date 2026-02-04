# Part 1: Data (Macro + Micro)

This part builds the datasets that every later notebook relies on. If you get the data timing wrong, the model results can look impressive but be meaningless.

## What You Will Learn
- How to call real APIs (FRED + Census/ACS) and cache results
- How to align mixed-frequency time series (daily/monthly/quarterly)
- How to compute GDP growth rates and construct a technical recession label
- How to build clean, typed, analysis-ready tables

## Prerequisites
- Foundations (Part 0)
- Comfort reading pandas code (`read_csv`, `resample`, `merge`)

## How To Study This Part
- Treat this as data engineering practice.
- For every transformation, ask: "Could this accidentally use future information?"
- Keep a small "data dictionary" in your notes: what each column means and its units.

## Chapters
- [00_fred_api_and_caching](00_fred_api_and_caching.md) — Notebook: [00_fred_api_and_caching.ipynb](../../../notebooks/01_data/00_fred_api_and_caching.ipynb)
- [01_build_macro_monthly_panel](01_build_macro_monthly_panel.md) — Notebook: [01_build_macro_monthly_panel.ipynb](../../../notebooks/01_data/01_build_macro_monthly_panel.ipynb)
- [02_gdp_growth_and_recession_label](02_gdp_growth_and_recession_label.md) — Notebook: [02_gdp_growth_and_recession_label.ipynb](../../../notebooks/01_data/02_gdp_growth_and_recession_label.ipynb)
- [03_build_macro_quarterly_features](03_build_macro_quarterly_features.md) — Notebook: [03_build_macro_quarterly_features.ipynb](../../../notebooks/01_data/03_build_macro_quarterly_features.ipynb)
- [04_census_api_microdata_fetch](04_census_api_microdata_fetch.md) — Notebook: [04_census_api_microdata_fetch.ipynb](../../../notebooks/01_data/04_census_api_microdata_fetch.ipynb)
