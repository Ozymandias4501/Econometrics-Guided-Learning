### Deep Dive: GDP Growth Math + Technical Recession Labels

GDP is a level series. A recession label requires turning levels into growth rates.

#### Key terms (defined)
> **Definition:** A **level** is the raw value of a series (e.g., real GDP in chained dollars).

> **Definition:** A **growth rate** is the percent change over a period.

> **Definition:** **QoQ** (quarter-over-quarter) compares $GDP_t$ to $GDP_{t-1}$.

> **Definition:** **YoY** (year-over-year) compares $GDP_t$ to $GDP_{t-4}$.

> **Definition:** **Annualized QoQ** converts a quarterly growth rate into an annual pace.

#### Growth formulas (math)
QoQ percent growth:

$$
 g_{qoq,t} = 100 \cdot \left(\frac{GDP_t}{GDP_{t-1}} - 1\right)
$$

Annualized QoQ percent growth (quarterly compounding):

$$
 g_{ann,t} = 100 \cdot \left(\left(\frac{GDP_t}{GDP_{t-1}}\right)^4 - 1\right)
$$

YoY percent growth:

$$
 g_{yoy,t} = 100 \cdot \left(\frac{GDP_t}{GDP_{t-4}} - 1\right)
$$

#### Why compute multiple growth measures?
- QoQ is responsive but noisy.
- YoY is smoother but slower to react.
- Annualized QoQ is common in macro reporting.

> **Definition:** A **log growth rate** uses differences of logs: $\Delta \log(GDP_t) = \log(GDP_t) - \log(GDP_{t-1})$.
Log growth is often convenient because it approximates percent growth for small changes and makes compounding math cleaner.

#### Technical recession label used in this project
> **Definition:** A **technical recession** (teaching proxy here) is two consecutive quarters of negative QoQ GDP growth.

Label:

$$
 recession_t = \mathbb{1}[g_{qoq,t} < 0 \;\wedge\; g_{qoq,t-1} < 0]
$$

Next-quarter prediction target:

$$
 target_{t} = recession_{t+1}
$$

#### Edge cases (what to watch)
- Missing GDP values will propagate to growth.
- The first growth observation is undefined (needs a prior quarter).
- YoY growth needs 4 prior quarters.

#### Python demo: compute growth + label (commented)
```python
import pandas as pd

# gdp: Series of GDP levels indexed by quarter-end dates
# gdp = ...

# QoQ growth (percent)
# growth_qoq = 100 * (gdp / gdp.shift(1) - 1)

# Technical recession label
# Two consecutive negative quarters:
# - current quarter growth < 0
# - previous quarter growth < 0
# recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)

# Next-quarter target
# Predict next quarter's label using information as-of this quarter:
# target_next = recession.shift(-1)
```

#### Project touchpoints (where this logic lives in code)
- `src/macro.py` implements these transforms explicitly:
  - `gdp_growth_qoq`, `gdp_growth_qoq_annualized`, `gdp_growth_yoy`
  - `technical_recession_label`
  - `next_period_target`

#### Python demo: using the project helper functions (commented)
```python
from src import macro

# levels: quarterly GDP level series
# levels = gdp['GDPC1']

# Growth variants
# qoq = macro.gdp_growth_qoq(levels)
# yoy = macro.gdp_growth_yoy(levels)

# Label + next-period target
# recession = macro.technical_recession_label(qoq)
# target = macro.next_period_target(recession)
```

#### Important limitation
This is a clean, computable teaching proxy.
It is not an official recession dating rule.

#### Macro caveat: revisions
GDP is revised. If you re-fetch later, historical values can change, which can change your computed label.
This is one reason caching matters.
