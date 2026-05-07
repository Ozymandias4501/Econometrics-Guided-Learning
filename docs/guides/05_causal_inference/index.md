# Part 5: Causal Inference

The earlier sections answered predictive questions: how does $x$ correlate with $y$? This part answers a different question: if we changed $x$, what would happen to $y$? An OLS coefficient is not automatically the answer to either, and not all controls are good controls.

## What You Will Learn
- The difference between predictive and causal estimands; the potential-outcomes framing in plain language.
- DAGs (directed acyclic graphs) as a back-of-envelope tool for thinking about identification.
- Omitted variable bias by simulation, and the bias formula that lets you predict its magnitude.
- The collider trap: why adding a 'control' can *create* bias instead of removing it.
- A working taxonomy of good, bad, and neutral controls.
- Difference-in-differences: the workhorse research design when randomization is impossible.
- The parallel-trends assumption, how to check it visually, and what threatens it.
- Cluster-robust standard errors and the small-cluster pitfall.

## Prerequisites
- §02 (regression).
- §02b (ML regression) and §03 (classification) are helpful for context but not strictly required.

## How To Study This Part
- The first notebook is the conceptual setup. Spend time on the DAG sketches; that is where the discipline lives.
- The second notebook is hands-on. Run the simulation, recover the known effect, then move to the synthesized policy on real-ish panel data.
- Treat parallel trends like an axiom: write down the most plausible *threat* to it for every DiD you fit.

## Chapters
- [00_predictive_vs_causal_dags](00_predictive_vs_causal_dags.md) — Notebook: [00_predictive_vs_causal_dags.ipynb](../../../notebooks/05_causal_inference/00_predictive_vs_causal_dags.ipynb)
- [01_difference_in_differences](01_difference_in_differences.md) — Notebook: [01_difference_in_differences.ipynb](../../../notebooks/05_causal_inference/01_difference_in_differences.ipynb)
