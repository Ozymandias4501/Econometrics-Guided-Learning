### Deep Dive: Walk-forward validation — repeated “train on past, test on next”

Walk-forward (rolling-origin) validation is the most common evaluation scheme for forecasting.

#### 1) Intuition (plain English)

A single time split answers: “How did I do in one future period?”

Walk-forward answers: “How do I do across many future periods as time moves forward?”

This reduces sensitivity to one arbitrary split boundary and better matches how models are used:
you re-train as new data arrives.

#### 2) Notation + setup (define symbols)

Let:
- $t$ be time,
- $h$ be forecast horizon,
- $W$ be a training window length (optional).

A walk-forward scheme produces folds indexed by $m=1,\\dots,M$:
- Train on $\\{1,\\dots,t_m\\}$ (or last $W$ periods),
- Test on $t_m+h$ (or a short future window).

#### 3) Assumptions

Walk-forward assumes:
- you would realistically update the model over time,
- you respect feature/label timing (no leakage),
- the evaluation horizon $h$ is fixed and meaningful.

#### 4) Estimation mechanics: what it estimates

Walk-forward estimates expected future error by averaging over many future evaluation points:

$$
\\widehat{\\mathrm{Err}} = \\frac{1}{M} \\sum_{m=1}^{M} \\ell(\\hat y_{t_m+h}, y_{t_m+h}).
$$

This is closer to the real “live” error than a random split.

#### 5) Inference and uncertainty

Because folds are time-ordered, fold errors can be correlated.
Treat fold-to-fold variability as descriptive, not as independent draws.

#### 6) Diagnostics + robustness (minimum set)

1) **Plot fold errors over time**
- do errors spike in recessions or structural breaks?

2) **Compare expanding vs rolling windows**
- expanding: more data, potentially stale regimes.
- rolling: focuses on recent regime, higher variance.

3) **Check stability across horizons**
- repeat for $h=1$ vs $h=2$ if relevant.

#### 7) Interpretation + reporting

Report:
- how you constructed folds (expanding/rolling),
- window length (if rolling),
- metrics averaged across folds and variation across time.

#### Exercises

- [ ] Implement walk-forward splits and compute fold-by-fold metrics; plot them over time.
- [ ] Compare expanding vs rolling training windows and interpret the trade-off.
- [ ] Identify a period where the model fails and propose a hypothesis (regime change? missing feature?).
