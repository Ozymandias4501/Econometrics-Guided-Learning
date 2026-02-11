# Guide: 02_gdp_growth_and_recession_label

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/02_gdp_growth_and_recession_label.ipynb`.

This guide focuses on computing GDP growth rates and constructing a binary recession label from continuous GDP data. These are the core target variables for the macro forecasting tasks in this project. The guide covers the formulas, the definition choices, and why those choices matter for downstream modeling and interpretation.

> **Prerequisite:** This guide assumes familiarity with the Core Data primer (schema, timing, reproducibility) and FRED API caching patterns covered in [Guide 00: FRED API and Caching](00_fred_api_and_caching.md). That guide is the canonical reference for data pipeline concepts, notation, and diagnostics that apply across all data guides.

### Key Terms (defined)
- **GDP growth (QoQ)**: quarter-over-quarter change in real GDP, typically expressed as a percentage or annualized percentage rate.
- **GDP growth (YoY)**: year-over-year change in real GDP, comparing the same quarter across consecutive years.
- **Annualized growth rate**: the QoQ growth rate scaled to represent what the annual rate would be if the quarter's pace continued for a full year. This is the convention used by BEA in headline GDP reports.
- **Technical recession**: a rule-of-thumb definition -- two consecutive quarters of negative GDP growth. Not an official definition, but widely used as a mechanical indicator.
- **NBER recession**: the official recession dating by the National Bureau of Economic Research's Business Cycle Dating Committee, which considers a broad set of indicators beyond GDP.
- **Binary label**: a 0/1 variable derived from continuous data by applying a threshold rule. The threshold choice is part of the modeling specification.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Fetch real GDP series from FRED (series ID: `GDP`) or load from cache
- Complete notebook section: Compute quarter-over-quarter GDP growth rate
- Complete notebook section: Compute annualized QoQ growth rate
- Complete notebook section: Define the technical recession label (two consecutive negative-growth quarters)
- Complete notebook section: Define the next-quarter target variable (`target_recession_next_q`)
- Compare your technical recession dates to NBER recession dates (FRED series: `USREC`)
- Verify that the target shift does not introduce leakage (features at $t$ predict label at $t+1$)
- Inspect class balance: count the number of recession vs non-recession quarters

### Alternative Example (Not the Notebook Solution)
```python
# Demonstrate recession label construction (not the notebook's exact code):
import pandas as pd

# --- Simulate quarterly GDP ---
idx = pd.date_range("2006-03-31", periods=20, freq="QE")
gdp = pd.Series(
    [13000, 13100, 13200, 13300, 13350, 13400, 13300, 13150,  # 2006-2007
     13000, 12850, 12900, 13050, 13200, 13350, 13500, 13650,  # 2008-2009
     13800, 13950, 14100, 14250],                               # 2010-2011
    index=idx, name="GDP"
)

# --- QoQ growth (simple percentage) ---
growth_qoq = 100 * (gdp / gdp.shift(1) - 1)

# --- Annualized QoQ growth ---
# BEA convention: annualize by compounding the quarterly rate
growth_annualized = 100 * ((1 + growth_qoq / 100) ** 4 - 1)

# --- Technical recession: two consecutive quarters of negative growth ---
neg_growth = (growth_qoq < 0).astype(int)
technical_recession = ((neg_growth == 1) & (neg_growth.shift(1) == 1)).astype(int)

# --- Next-quarter target (for forecasting) ---
target_next_q = technical_recession.shift(-1)

# --- Inspect ---
result = pd.DataFrame({
    "GDP": gdp,
    "growth_qoq": growth_qoq,
    "growth_annualized": growth_annualized,
    "neg_growth": neg_growth,
    "technical_recession": technical_recession,
    "target_next_q": target_next_q,
})
print(result.to_string())
print(f"\nRecession quarters: {technical_recession.sum()}")
print(f"Non-recession quarters: {(technical_recession == 0).sum()}")
```

### Comparing QoQ vs YoY Growth
```python
# YoY growth compares the same quarter across years (reduces seasonal noise):
growth_yoy = 100 * (gdp / gdp.shift(4) - 1)

print("QoQ growth (Q3 2008):", growth_qoq.loc["2008-09-30"])
print("YoY growth (Q3 2008):", growth_yoy.loc["2008-09-30"])
# QoQ captures the immediate quarter-to-quarter decline
# YoY captures the decline relative to the same quarter last year
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: GDP growth + the "technical recession" label (macro target construction)

This repo's macro tasks rely on a recession label and GDP growth features. Target construction is part of the econometric specification -- the label is not "given by nature" but defined by the researcher.

#### 1) Intuition (plain English)

Labels are not "given by nature." You define them.

**Story example:** A binary recession label is a simplification of a complex economic phenomenon.
The value of the label is not that it is perfect, but that it is:
- clear,
- reproducible,
- aligned to a forecasting horizon.

The choice of label definition affects everything downstream: class balance, what counts as a "false positive" or "false negative," and the economic interpretation of model errors.

**Health economics connection:** Recession labels are directly useful for studying counter-cyclical health spending. During recessions, Medicaid enrollment rises as unemployment increases, hospital uncompensated care grows, and mental health utilization patterns shift. A well-defined recession indicator lets you estimate these effects causally (e.g., difference-in-differences around recession onset) or predictively (e.g., forecasting next-quarter Medicaid enrollment). The label definition matters: if your recession indicator fires two quarters late (because NBER announces with a lag), your "treatment timing" is wrong for causal inference.

#### 2) Notation + setup (define symbols)

Let:
- $GDP_t$ be real GDP level in quarter $t$.

**Quarter-over-quarter (QoQ) growth (simple form):**

$$
g_t^{qoq} = \frac{GDP_t - GDP_{t-1}}{GDP_{t-1}} = \frac{GDP_t}{GDP_{t-1}} - 1.
$$

This gives the percentage change from the previous quarter. It is the most direct measure of short-run momentum.

**Annualized QoQ growth (BEA convention):**

$$
g_t^{ann} = \left(1 + g_t^{qoq}\right)^4 - 1.
$$

This is the rate that would prevail if the quarter's growth pace continued for a full year. BEA reports this as the headline GDP growth number. For small growth rates, $g_t^{ann} \approx 4 \times g_t^{qoq}$, but the exact formula accounts for compounding.

**Year-over-year (YoY) growth:**

$$
g_t^{yoy} = \frac{GDP_t - GDP_{t-4}}{GDP_{t-4}}.
$$

This compares the same quarter across consecutive years. YoY growth is smoother than QoQ because it averages over four quarters of change, reducing seasonal noise and short-run volatility.

**When to use each:**

| Measure         | Formula                                      | Captures                        | Best for                                         |
|-----------------|----------------------------------------------|---------------------------------|--------------------------------------------------|
| QoQ (simple)    | $\frac{GDP_t}{GDP_{t-1}} - 1$               | Immediate quarterly momentum    | Recession detection, turning-point analysis       |
| QoQ (annualized)| $(1 + g_t^{qoq})^4 - 1$                     | Annualized pace of change       | Headline reporting, comparing to annual targets   |
| YoY             | $\frac{GDP_t}{GDP_{t-4}} - 1$               | Year-over-year trend            | Smoothed trend analysis, reducing seasonal noise  |

**Technical recession label (common rule of thumb):**
- recession at $t$ if GDP growth is negative for two consecutive quarters:

$$
R_t = \mathbf{1}\left[g_t^{qoq} < 0 \;\text{and}\; g_{t-1}^{qoq} < 0\right].
$$

**What each term means**
- QoQ captures short-run movements; YoY smooths seasonal/short-run noise.
- The "two quarters" rule is a convention, not a structural definition.
- $\mathbf{1}[\cdot]$ is the indicator function: returns 1 if the condition is true, 0 otherwise.

#### 3) Assumptions (and limitations)

**Technical recession assumptions:**
- GDP growth is an adequate proxy for recession timing.
- Two consecutive quarters of negative growth is a meaningful threshold.
- The GDP data used is the "final" revised version (not the real-time estimate available to policymakers).

**NBER vs technical recession -- why the definition matters:**

The NBER Business Cycle Dating Committee defines a recession as "a significant decline in economic activity that is spread across the economy and lasts more than a few months." Key differences from the technical rule:

| Aspect                  | Technical Recession                     | NBER Recession                                  |
|-------------------------|-----------------------------------------|-------------------------------------------------|
| Definition              | Two consecutive quarters of negative GDP growth | Broad decline in economic activity               |
| Indicators used         | GDP only                                | GDP, employment, income, sales, production       |
| Timing of announcement  | Immediate (mechanical)                  | 6-18 months lag (deliberative)                   |
| Edge cases              | Misses mild broad-based recessions     | Catches recessions that GDP alone misses         |
| Reproducibility         | Fully mechanical                        | Depends on committee judgment                    |

**Why this matters for your project:**
- For forecasting, the technical rule is preferred (reproducible, timely). For causal inference (e.g., "what happens to health spending during recessions?"), NBER dates are preferred (consensus timing).
- The technical rule and NBER will disagree in some periods -- understand when and why.

**GDP revisions:** GDP is revised multiple times (advance, second, third, annual, benchmark). Growth rates can change sign between vintages. Training on final revised data but deploying on advance estimates is a form of look-ahead bias.

#### 4) Mechanics: constructing the label step by step

1. **Compute growth**: `growth_qoq = gdp.pct_change()` (simple QoQ); `growth_ann = (1 + growth_qoq)**4 - 1` (annualized); `growth_yoy = gdp.pct_change(4)` (YoY).
2. **Apply recession rule**: `neg = (growth_qoq < 0).astype(int)` then `recession = ((neg == 1) & (neg.shift(1) == 1)).astype(int)`.
3. **Shift for forecasting**: `target_next_q = recession.shift(-1)`. The last row's target is NaN (no future to predict). Verify features at $t$ do not include information from $t+1$.
4. **Inspect class balance**: typically ~8-12% of quarters are recession (highly imbalanced). Always report `recession.mean()` alongside model performance.

#### 5) Inference: label uncertainty and evaluation

Classification metrics depend on label prevalence and definition. If you change the label rule, you change:
- class imbalance (tighter rules = fewer positives),
- what counts as "false positive/negative,"
- and the economic meaning of a miss.

**Binary labels from continuous data -- the threshold choice:**
The technical recession rule applies a specific threshold (growth < 0) and a duration requirement (two quarters). Both are arbitrary:
- Why zero? A growth rate of -0.01% is classified as negative, but economically it is indistinguishable from +0.01%.
- Why two quarters? One quarter of negative growth could be measurement error or a one-off shock.

These choices are part of your specification. You should test sensitivity:
- What if the threshold is -0.5% instead of 0%?
- What if you require only one quarter of negative growth?
- How does the recession count change, and does your model's performance change materially?

**Asymmetric costs of errors:**
In health economics and policy applications, the costs of false positives and false negatives are asymmetric:
- **Missing a recession** (false negative): policymakers fail to prepare for rising Medicaid enrollment, hospital systems are caught off-guard by increasing uncompensated care.
- **False alarm** (false positive): resources are pre-positioned unnecessarily, but the cost is lower than being unprepared.

This asymmetry should inform your choice of evaluation metric (precision-recall trade-off, not just accuracy).

#### 6) Diagnostics + robustness (minimum set)

1) **Plot GDP growth with recession shading** -- confirm the label activates where you expect (2008-2009, 2020).
2) **Compare to NBER dates** -- fetch `USREC` from FRED, resample to quarterly with `.max()`, and compute agreement rate.
3) **Sensitivity to growth definition** -- compare QoQ vs YoY-based heuristics and simple vs annualized thresholds.
4) **Check for GDP revision effects** -- if possible, compare labels from vintage data vs final revised data.

#### 7) Interpretation + reporting

When you report a model that uses a recession label:
- specify the label definition (technical rule: two consecutive quarters of negative QoQ GDP growth),
- specify the GDP series used (`GDP` from FRED, seasonally adjusted, real),
- specify the forecast horizon (next-quarter: $t$ predicts $R_{t+1}$),
- report class balance (e.g., "12% of quarters are labeled as recession"),
- interpret errors in economic terms (missed recession vs false alarm).

**Example reporting statement:** "We define recession as two consecutive quarters of negative quarter-over-quarter real GDP growth (FRED series GDP). Using this definition, 23 of 200 sample quarters (11.5%) are classified as recession. Our model predicts next-quarter recession status using features available at time $t$."

#### Exercises

- [ ] Compute QoQ and YoY GDP growth; plot both and describe when they diverge.
- [ ] Verify that $g^{ann} \approx 4 \times g^{qoq}$ for small growth rates.
- [ ] Construct the technical recession label and compare to known recession dates (2008-09, 2020).
- [ ] Shift the label to create a one-quarter-ahead target; confirm no leakage.
- [ ] Compare your technical recession dates to NBER dates (`USREC`). Where do they disagree?

### Project Code Map
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

### Common Mistakes
- Confusing simple QoQ with annualized QoQ growth (annualized is roughly 4x larger).
- Using `pct_change(4)` (YoY) when you mean `pct_change(1)` (QoQ) for the recession rule.
- Forgetting that `shift(-1)` means the last observation's target is NaN.
- Not checking class balance: ~90% accuracy is trivially achieved by always predicting "no recession."
- Using final revised GDP in a model deployed on advance estimates (look-ahead bias).

<a id="summary"></a>
## Summary + Suggested Readings

This guide covered GDP growth computation (QoQ, annualized, YoY) and the technical recession label. The key insight: the label is a modeling choice, not an objective fact. The threshold, duration requirement, and GDP measure all affect downstream results.

Key takeaways:
- QoQ captures momentum; YoY captures trend; annualized QoQ is the headline convention.
- The technical recession rule is mechanical and reproducible but crude; NBER dates are more comprehensive but lagged.
- Binary labels always involve a threshold choice that should be tested for sensitivity.
- In health economics, recession labels enable studying counter-cyclical effects on Medicaid and population health.

For the Core Data primer (schema, timing, reproducibility), see [Guide 00](00_fred_api_and_caching.md).

Suggested readings:
- NBER Business Cycle Dating Committee: [methodology](https://www.nber.org/research/business-cycle-dating)
- Hamilton, J.D. (2011), "Calling Recessions in Real Time"
- FRED series documentation: `GDP`, `USREC`, `USRECM`
