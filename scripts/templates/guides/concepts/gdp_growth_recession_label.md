### Deep Dive: GDP growth + the “technical recession” label (macro target construction)

This repo’s macro tasks rely on a recession label and GDP growth features. Target construction is part of the econometric specification.

#### 1) Intuition (plain English)

Labels are not “given by nature.” You define them.

**Story example:** A binary recession label is a simplification of a complex economic phenomenon.
The value of the label is not that it is perfect, but that it is:
- clear,
- reproducible,
- aligned to a forecasting horizon.

#### 2) Notation + setup (define symbols)

Let:
- $GDP_t$ be real GDP level in quarter $t$.

Quarter-over-quarter (QoQ) growth (simple form):

$$
g_t^{qoq} = \\frac{GDP_t - GDP_{t-1}}{GDP_{t-1}}.
$$

Year-over-year (YoY) growth:

$$
g_t^{yoy} = \\frac{GDP_t - GDP_{t-4}}{GDP_{t-4}}.
$$

Technical recession label (common rule of thumb):
- recession at $t$ if GDP growth is negative for two consecutive quarters:

$$
R_t = 1[g_t^{qoq} < 0 \\;\\text{and}\\; g_{t-1}^{qoq} < 0].
$$

**What each term means**
- QoQ captures short-run movements; YoY smooths seasonal/short-run noise.
- The “two quarters” rule is a convention, not a structural definition.

#### 3) Assumptions (and limitations)

This label assumes:
- GDP growth is an adequate proxy for recession timing.

Limitations:
- official recession dating (NBER) uses broader information and can differ,
- GDP revisions can change growth signs ex post,
- the label is coarse and may miss mild downturns.

#### 4) Mechanics: aligning labels to forecasting horizons

If you predict “next quarter recession,” you must shift labels:
- features at time $t$ predict $R_{t+1}$ (or $R_{t+h}$).

That makes timing explicit and reduces leakage risk.

#### 5) Inference: label uncertainty and evaluation

Classification metrics depend on label prevalence and definition.
If you change the label rule, you change:
- class imbalance,
- what counts as “false positive/negative,”
- and the economic meaning of a miss.

#### 6) Diagnostics + robustness (minimum set)

1) **Plot GDP growth with recession shading**
- confirm the label activates where you expect.

2) **Compare to alternative labels**
- NBER recession indicator (if available), unemployment-based rules, etc.

3) **Sensitivity to growth definition**
- compare QoQ vs YoY-based recession heuristics.

#### 7) Interpretation + reporting

When you report a model:
- specify the label definition,
- specify the forecast horizon,
- interpret errors in economic terms (missed recession vs false alarm).

#### Exercises

- [ ] Compute QoQ and YoY GDP growth and plot them.
- [ ] Construct the technical recession label and verify the count of recession quarters.
- [ ] Shift the label to create a one-quarter-ahead target; confirm no leakage.
- [ ] Compare your label to an alternative (if available) and note differences.
