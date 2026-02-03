### Deep Dive: Correlation vs Causation (What You Can and Cannot Claim)

> **Definition:** **Correlation** means two variables move together.

> **Definition:** **Causation** means changing X would change Y (an intervention claim).

Correlation is a descriptive property of the observed data.
Causation is a claim about a data-generating mechanism.

#### Why this matters in economics
Most economic datasets are observational.
That means you usually have correlations, not controlled experiments.

If you interpret a coefficient as causal without a causal design, you can be confidently wrong.

#### Python demo: confounding creates correlation (commented)
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

n = 1000

# z is a confounder that affects both x and y
z = rng.normal(size=n)
x = 0.8 * z + rng.normal(scale=1.0, size=n)
y = 2.0 * z + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'x': x, 'y': y, 'z': z})

# x and y are correlated, but the relationship is driven by z
print(df.corr())
```

#### Practical interpretation
- "x is correlated with y" is a safe claim.
- "x causes y" requires identification assumptions (experiments, instruments, diff-in-diff, etc.).

#### In this project
We focus on prediction and careful interpretation.
We do not claim causal effects unless explicitly designed.
