### Core Causal Inference: Identification Before Estimation

In econometrics, the hard part is rarely the regression code. It is the logic of **identification**.

> **Definition:** **Identification** means: “Under which assumptions can we interpret a statistic as a causal effect?”

> **Definition:** **Estimation** means: “How do we compute that statistic from data?”

Most datasets in this repo are observational (not randomized experiments), so causal claims require extra structure:
- a policy cutoff (RD),
- a shock that changes some units but not others (DiD),
- an instrument that shifts treatment but not outcomes directly (IV),
- or a strong structural model (not covered here).

#### Potential outcomes (one-line model)
For each unit $i$ and time $t$:
- $Y_{it}(1)$: outcome if treated
- $Y_{it}(0)$: outcome if not treated

The causal effect is $Y_{it}(1) - Y_{it}(0)$, but we never observe both at once.

#### Why this repo uses “semi-synthetic” exercises
To practice mechanics safely, some notebooks construct a **known treatment effect** on top of real outcomes.
That lets you verify whether your estimator can recover the truth, without pretending this is a real policy evaluation.

