# Part 1b: Time Series Diagnostics

The boring-but-load-bearing checks you should run on a time-series panel **before** you regress anything on anything. Skip this and the t-statistics in §02 may be lying to you in ways HAC standard errors cannot fix.

## What You Will Learn
- What stationarity means and why OLS asymptotics depend on it.
- The spurious-regression problem and why it is a different failure mode from autocorrelation.
- ADF and KPSS unit-root tests, why their nulls are deliberately opposite, and how to read them together.
- The differencing decision: when to log-difference, when to first-difference, when to leave alone.
- A one-paragraph introduction to cointegration so you recognize it when it shows up in macro literature.

## Prerequisites
- Sections 00a (statistics primer) and 00b (foundations).
- Section 01 (data) — you have a quarterly macro panel ready.

## How To Study This Part
- Run the spurious-regression Monte Carlo first; the result is more persuasive than any reading.
- Always run ADF and KPSS together. A single test result is not a verdict.
- After you finish, write down a one-line transformation policy for each column you intend to use as a feature or target downstream.

## Chapters
- [00_stationarity_and_unit_roots](00_stationarity_and_unit_roots.md) — Notebook: [00_stationarity_and_unit_roots.ipynb](../../../notebooks/01b_time_series_diagnostics/00_stationarity_and_unit_roots.ipynb)
