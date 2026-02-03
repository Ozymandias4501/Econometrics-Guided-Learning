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
# recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)

# Next-quarter target
# target_next = recession.shift(-1)
```

#### Important limitation
This is a clean, computable teaching proxy.
It is not an official recession dating rule.

#### Macro caveat: revisions
GDP is revised. If you re-fetch later, historical values can change, which can change your computed label.
This is one reason caching matters.
