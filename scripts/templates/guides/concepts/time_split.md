### Deep Dive: Time splits — evaluation that matches forecasting reality

Time-aware splitting is not optional in forecasting tasks; it defines what “generalization” means.

#### 1) Intuition (plain English)

If you predict the future, you must train on the past and test on the future.
Random splits answer a different question: “Can I interpolate within a mixed pool of time periods?”

**Story example:** If you train on 2008 and test on 2006 (random split), you are letting crisis-era patterns help predict pre-crisis data—an unrealistic advantage.

#### 2) Notation + setup (define symbols)

Let:
- data be ordered by time $t=1,\\dots,T$,
- training window be $t \\le t_{train}$,
- test window be $t > t_{train}$.

A basic time split is:
- Train: $\\{1,\\dots,t_{train}\\}$
- Test: $\\{t_{train}+1,\\dots,T\\}$

#### 3) Assumptions (what time splits assume)

Time splits assume:
- you can use historical data to learn relationships relevant for the future,
- the feature/label timing is correctly defined (no leakage),
- you accept that regimes can change (so performance can vary).

#### 4) Estimation mechanics: why random splits overestimate performance

Random splits mix early and late periods in both train and test.
That creates two problems:
- **information leakage via time correlation** (nearby periods are similar),
- **regime mixing** (train sees future regimes).

So the test metric can be biased upward relative to true forecasting performance.

#### 5) Inference: splits affect uncertainty

Even if you do inference (p-values), time dependence matters:
- serial correlation inflates effective sample size if ignored,
- time splits help reveal whether relationships are stable across eras.

#### 6) Diagnostics + robustness (minimum set)

1) **Report split dates**
- always print the last train date and first test date.

2) **Try multiple cut points**
- if performance depends heavily on one boundary, results are unstable.

3) **Plot train vs test distributions**
- shifts in feature distributions indicate regime drift.

4) **Compare to walk-forward**
- walk-forward validation often gives a more realistic error estimate.

#### 7) Interpretation + reporting

Report:
- split scheme (single holdout vs multiple folds),
- dates and horizon,
- metrics on the test period (and ideally multiple periods).

**What this does NOT mean**
- One lucky split is not proof of generalization.

#### Exercises

- [ ] Evaluate the same model with a random split and a time split; compare and explain the gap.
- [ ] Move the split boundary forward/backward by a few years and report stability.
- [ ] Plot feature distributions in train vs test; identify at least one shifted feature.
