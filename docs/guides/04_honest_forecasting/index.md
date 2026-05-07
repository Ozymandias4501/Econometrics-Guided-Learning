# Part 4: Honest Forecasting

By §03 you have models that look good on a single train/test split. This part is about whether those numbers survive contact with reality. The answer is usually 'partly, and not in the windows you care about most.'

## What You Will Learn
- Why a single train/test split is not an honest answer for a time-series model.
- How to implement an expanding-window walk-forward backtest from scratch and apply it to OLS, RF, and XGBoost on the same target.
- How to read a rolling-error plot for evidence of regime change.
- How a simple mean ensemble can beat any single model.
- What a structural break is, why the Chow test is the standard diagnostic, and three reasonable strategies once you find one (crisis dummy, drop the period, refit post-break).
- The COVID problem, concretely: what it does to a pre-2020 model and how to keep modeling sensibly.

## Prerequisites
- §01 (data), §02 (regression), §02b (ML regression), §03 (classification).
- §01b (stationarity) is helpful — non-stationarity can masquerade as a break.

## How To Study This Part
- Pick the walk-forward protocol you commit to *before* looking at the results. Otherwise you will tune the protocol to the answer you wanted.
- Always look at the rolling-error plot, not just the aggregate RMSE.
- Treat the Chow p-value as one piece of evidence, not the whole answer. Visualize before testing.

## Chapters
- [00_walk_forward_backtest](00_walk_forward_backtest.md) — Notebook: [00_walk_forward_backtest.ipynb](../../../notebooks/04_honest_forecasting/00_walk_forward_backtest.ipynb)
- [01_structural_breaks_and_covid](01_structural_breaks_and_covid.md) — Notebook: [01_structural_breaks_and_covid.ipynb](../../../notebooks/04_honest_forecasting/01_structural_breaks_and_covid.ipynb)
