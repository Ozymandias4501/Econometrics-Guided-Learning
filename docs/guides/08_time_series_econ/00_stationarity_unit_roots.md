# Guide: 00_stationarity_unit_roots

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/08_time_series_econ/00_stationarity_unit_roots.ipynb`.

Before fitting any time-series model — regression, VAR, or forecasting — you need to answer one question: **is this series stable enough that the usual statistical tools make sense?** Stationarity is the formal version of that question, and unit-root tests are the primary diagnostic tools for answering it. Getting this wrong leads to spurious regression, meaningless t-statistics, and models that look great on paper but predict nothing.

### Key Terms (defined)
- **Stationarity (weak / covariance)**: a time series whose mean, variance, and autocovariance structure do not change over time. Intuitively, the series "looks the same" no matter where you cut a window.
- **Strict stationarity**: a stronger condition where the *entire* joint distribution of any collection of time points is shift-invariant. Rarely tested in practice; weak stationarity is the working definition.
- **Unit root**: a characteristic root of the autoregressive polynomial equal to 1, meaning shocks to the series never decay. The classic example is a random walk.
- **I(d) — Integrated of order d**: a series that requires $d$ rounds of differencing to become stationary. I(0) = stationary in levels; I(1) = nonstationary in levels but stationary in first differences.
- **Trend stationarity vs difference stationarity**: a trend-stationary series fluctuates around a deterministic trend and can be detrended; a difference-stationary series has a stochastic trend that must be differenced away. The distinction matters for which transformation you apply.
- **Spurious regression**: a regression between unrelated I(1) series that produces misleadingly significant results — high $R^2$, large t-stats — purely because both series trend.
- **ADF (Augmented Dickey–Fuller)**: a unit-root test whose null hypothesis is "unit root present" (nonstationarity).
- **KPSS**: a complementary test whose null is "stationarity."

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Load at least 3 macro series from the panel (e.g., GDP level, unemployment rate, CPI).
- Plot each series in levels — visually assess trend, volatility changes, and possible breaks.
- Apply transformations (first difference, log-difference, growth rate) and plot the result.
- For each series, run both ADF and KPSS tests on (a) levels and (b) the transformed version.
- Build a summary table: series name, ADF p-value (levels), ADF p-value (diff), KPSS p-value (levels), KPSS p-value (diff), your classification (I(0) or I(1)).
- Demonstrate spurious regression: regress one random walk on another and report $R^2$ and t-stats.
- Write a 4–6 sentence interpretation for each series justifying your transformation choice.

### Alternative Example (Not the Notebook Solution)
```python
# Trend-stationary vs difference-stationary processes
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

rng = np.random.default_rng(42)
T = 300

# --- Trend-stationary: y_t = 0.5t + stationary noise ---
trend_stat = 0.5 * np.arange(T) + 5 * rng.normal(size=T)

# --- Difference-stationary (random walk with drift): y_t = 0.5 + y_{t-1} + e_t ---
diff_stat = np.zeros(T)
for t in range(1, T):
    diff_stat[t] = 0.5 + diff_stat[t - 1] + rng.normal()

# ADF on levels: both may fail to reject (both trend upward)
print("Trend-stationary (levels):", adfuller(trend_stat, regression="ct")[1])
print("Difference-stationary (levels):", adfuller(diff_stat, regression="ct")[1])

# After detrending the first:
from numpy.polynomial.polynomial import polyfit, polyval
coeffs = polyfit(np.arange(T), trend_stat, 1)
detrended = trend_stat - polyval(np.arange(T), coeffs)
print("Trend-stationary (detrended):", adfuller(detrended)[1])

# After differencing the second:
diff_of_rw = np.diff(diff_stat)
print("Difference-stationary (first diff):", adfuller(diff_of_rw)[1])
```

This example illustrates a key point: two series can look similar in levels (both trend upward) but require *different* treatments. The trend-stationary series should be detrended; the random walk should be differenced. Applying the wrong transformation can mask the true dynamics.


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Stationarity — the foundation of time-series inference

Classical time-series econometrics starts with one question:

> **Is this series stable enough over time that our regression assumptions make sense?**

If the answer is no, nearly every tool in the OLS/inference toolkit can mislead you. Stationarity is not a technicality — it is the reason you must think before you regress.

#### 1) Intuition (plain English): why stationarity matters

Imagine you want to estimate the relationship between inflation and unemployment over 50 years. If both series hover around stable means and their fluctuations look "similar" decade to decade, you can reasonably estimate correlations, run regressions, and compute standard errors.

But if one or both series trend persistently — GDP in levels, for instance, grows from $2 trillion to $25 trillion — then a regression can "find" a relationship that is nothing more than two lines going up. The residuals themselves will be persistent (autocorrelated), standard errors will be wildly wrong, and you will get statistically "significant" results that mean nothing.

**Three real-world consequences of ignoring nonstationarity:**
1. **Spurious regression**: Granger and Newbold (1974) showed that regressing one random walk on another yields a significant t-statistic over 75% of the time, even though the series are independent by construction.
2. **Nonstandard distributions**: When the regressor is I(1), the usual t and F distributions no longer apply. The OLS estimator converges at rate $T$ (not $\sqrt{T}$), and its limit distribution involves Brownian motion, not a normal.
3. **Wrong forecasts**: A model fitted on spurious trends will project those trends forward indefinitely — absurd for any practical forecast.

#### 2) Formal definition: weak (covariance) stationarity

Let $\{y_t\}_{t \in \mathbb{Z}}$ be a stochastic process. It is **weakly stationary** if:

1. **Constant mean**: $\mathbb{E}[y_t] = \mu$ for all $t$.
2. **Constant variance**: $\mathrm{Var}(y_t) = \gamma_0 < \infty$ for all $t$.
3. **Autocovariance depends only on lag**: $\mathrm{Cov}(y_t, y_{t-k}) = \gamma_k$ for all $t$ and integer $k$.

**What each condition means in plain language:**
- Condition (1): the series does not drift up or down over time. Its average is the same whether you look at the first decade or the last.
- Condition (2): the series does not become more or less volatile over time.
- Condition (3): the dependence between $y_t$ and $y_{t-5}$ is the same whether $t$ is 1970 or 2020.

**Strict stationarity** is stronger: it requires that the joint distribution of $(y_{t_1}, \dots, y_{t_k})$ is the same as $(y_{t_1+h}, \dots, y_{t_k+h})$ for any shift $h$. Under finite second moments, strict implies weak. In practice, we test and work with weak stationarity.

#### 3) AR(1) as the building block: stationary vs unit root

The simplest way to build intuition is through the AR(1) model:

$$
y_t = c + \rho y_{t-1} + \varepsilon_t,
\qquad
\varepsilon_t \sim \text{WN}(0, \sigma^2),
$$

where WN denotes white noise (uncorrelated, zero mean, constant variance).

**Case 1: $|\rho| < 1$ (stationary)**

By recursive substitution:

$$
y_t = \frac{c}{1-\rho} + \sum_{s=0}^{\infty} \rho^s \varepsilon_{t-s}.
$$

Because $|\rho| < 1$, the weights $\rho^s$ decay geometrically. This means:
- **Mean**: $\mu = c/(1-\rho)$ — a finite, constant value.
- **Variance**: $\gamma_0 = \sigma^2 / (1 - \rho^2)$ — finite and constant.
- **Autocovariance at lag $k$**: $\gamma_k = \rho^k \gamma_0$ — decays exponentially.
- **Half-life of a shock**: $h^* = -\log(2)/\log(|\rho|)$. For $\rho = 0.9$, a shock's impact halves in about 6.6 periods. For $\rho = 0.5$, it halves in 1 period.

The series has a well-defined long-run mean $\mu$ and shocks die out — this is **mean reversion**.

<details>
<summary>Derivation: variance of stationary AR(1)</summary>

From $y_t = c + \rho y_{t-1} + \varepsilon_t$, take variances on both sides (exploiting the fact that $\varepsilon_t$ is independent of $y_{t-1}$):

$$
\mathrm{Var}(y_t) = \rho^2 \mathrm{Var}(y_{t-1}) + \sigma^2.
$$

Under stationarity, $\mathrm{Var}(y_t) = \mathrm{Var}(y_{t-1}) = \gamma_0$, so:

$$
\gamma_0 = \rho^2 \gamma_0 + \sigma^2
\quad \Rightarrow \quad
\gamma_0 = \frac{\sigma^2}{1 - \rho^2}.
$$

This is only finite when $|\rho| < 1$. As $\rho \to 1$, the variance diverges, foreshadowing the unit-root case.
</details>

**Case 2: $\rho = 1$ (unit root / random walk)**

With $c = 0$ for simplicity:

$$
y_t = y_{t-1} + \varepsilon_t
\quad \Rightarrow \quad
y_t = y_0 + \sum_{s=1}^{t} \varepsilon_s.
$$

Now the process is a cumulative sum of shocks:
- **Mean**: $\mathbb{E}[y_t] = y_0$ (constant, but only trivially — the series wanders).
- **Variance**: $\mathrm{Var}(y_t) = t \sigma^2$ — **grows linearly with time**. The series becomes more and more dispersed.
- **Autocovariance**: $\mathrm{Cov}(y_t, y_{t-k}) = (t-k)\sigma^2$ — depends on $t$, violating stationarity.

Every shock is permanent. There is no mean reversion. If the economy receives a negative shock today, it shifts the *entire future path* of the series downward.

**Case 2b: Random walk with drift ($c \neq 0$, $\rho = 1$)**

$$
y_t = c + y_{t-1} + \varepsilon_t
\quad \Rightarrow \quad
y_t = y_0 + ct + \sum_{s=1}^{t} \varepsilon_s.
$$

Now the series trends with slope $c$ and has a stochastic component that grows in variance. This is what most macro series in levels look like — nominal GDP, the price level, etc.

**Case 3: Near-unit root ($\rho$ close to 1 but below)**

In finite samples, a process with $\rho = 0.97$ is nearly indistinguishable from $\rho = 1$. Unit-root tests have low power in this "near-unit-root" region, which is why you should never treat a single test result as definitive. With $T = 100$ quarterly observations and $\rho = 0.97$, the ADF test will fail to reject the unit root null more than half the time — even though the series is technically stationary.

#### 4) Integrated processes: the I(d) classification

Econometrics classifies series by the number of times you must difference them to achieve stationarity:

- $y_t$ is **I(0)** if it is (weakly) stationary in levels. Examples: growth rates, inflation rates, interest rate spreads.
- $y_t$ is **I(1)** if $\Delta y_t = y_t - y_{t-1}$ is stationary but $y_t$ itself is not. Examples: log GDP, log CPI, the unemployment rate (debatable).
- $y_t$ is **I(2)** if $\Delta^2 y_t = \Delta(\Delta y_t)$ is needed. This is rare in economics; if you encounter it, check for data issues first.

**Why this matters for regression:**
- Regressing I(0) on I(0): standard OLS inference applies (with appropriate SE if errors are autocorrelated).
- Regressing I(1) on I(1): potentially spurious unless the series are cointegrated (next guide).
- Regressing I(1) on I(0) or vice versa: regression is unbalanced and results are generally meaningless.

The practical implication: **always check the integration order of every series before running a regression**.

#### 5) Trend stationarity vs difference stationarity

Two processes can both trend upward but have fundamentally different dynamics:

**Trend-stationary (TS):**
$$
y_t = \alpha + \delta t + u_t,
\qquad u_t \text{ is I(0).}
$$

The trend is deterministic ($\delta t$), and shocks ($u_t$) are temporary — they die out. The series always returns to its trend line. Appropriate treatment: **detrend** (regress on time, take residuals) or include a time trend in your regression.

**Difference-stationary (DS) — random walk with drift:**
$$
y_t = c + y_{t-1} + \varepsilon_t.
$$

The trend is stochastic: there is no fixed trend line the series returns to. Shocks are permanent. Appropriate treatment: **first-difference** the series.

**Why the distinction matters:**
- If you difference a TS process, you introduce unnecessary negative autocorrelation (overdifferencing).
- If you detrend a DS process, the "detrended" series is still nonstationary — you have not removed the stochastic trend.
- Applying the wrong transformation gives you a series with artificial statistical properties.

In practice, most macro series in levels are treated as DS (differenced), because:
1. It is hard to distinguish TS from DS in finite samples.
2. Differencing a TS process is a minor sin (slight efficiency loss); detrending a DS process is a major sin (inference is wrong).
3. Most economic theory is consistent with permanent shocks (technology, institutional change).

#### 6) Spurious regression: the danger of ignoring unit roots

Granger and Newbold (1974) demonstrated the problem, and Phillips (1986) proved it formally.

**Setup**: Generate two independent random walks:
$$
x_t = x_{t-1} + u_t, \qquad y_t = y_{t-1} + v_t,
\qquad u_t \perp v_t.
$$

Regress $y_t = \alpha + \beta x_t + e_t$.

**What happens:**
- $\hat\beta$ does not converge to zero (it converges to a random variable involving ratios of Brownian motions).
- The t-statistic diverges — it gets "more significant" as $T$ grows. In a standard world, more data should make you *less* likely to reject a true null. Here, more data makes you *more* likely to incorrectly reject.
- $R^2$ converges to a nondegenerate random variable (often 0.3–0.7), not to zero.
- The Durbin-Watson statistic converges to zero — a telltale sign that residuals are massively autocorrelated.

**Practical diagnostic:** If you see $R^2 > DW$ (R-squared exceeds the Durbin-Watson statistic) in a levels regression, be suspicious. This is a classic sign of spurious regression.

**Numerical example:** With $T = 200$, independent random walks yield an average $R^2 \approx 0.44$ and reject $H_0: \beta = 0$ at the 5% level about 77% of the time. With larger $T$, the rejection rate approaches 100%.

#### 7) The Augmented Dickey–Fuller (ADF) test: mechanics and interpretation

The ADF test starts from the AR(1) reparameterization. Subtract $y_{t-1}$ from both sides:

$$
y_t - y_{t-1} = (\rho - 1)y_{t-1} + \varepsilon_t
\quad \Rightarrow \quad
\Delta y_t = \gamma y_{t-1} + \varepsilon_t,
$$

where $\gamma = \rho - 1$. Under the null $\rho = 1$, we have $\gamma = 0$.

The "augmented" version adds deterministic terms and lagged differences to absorb serial correlation:

$$
\Delta y_t = a + bt + \gamma y_{t-1} + \sum_{j=1}^{p} \phi_j \Delta y_{t-j} + e_t.
$$

**What each term does:**
- $a$: intercept — allows the stationary alternative to have a nonzero mean.
- $bt$: time trend — allows the stationary alternative to fluctuate around a linear trend.
- $\gamma y_{t-1}$: the key parameter. If $\gamma < 0$, the series mean-reverts; if $\gamma = 0$, it has a unit root.
- $\sum \phi_j \Delta y_{t-j}$: lagged differences that ensure $e_t$ is approximately white noise (otherwise the test statistic is biased).

**Hypotheses:**
- $H_0: \gamma = 0$ (unit root — the series is I(1) or worse).
- $H_1: \gamma < 0$ (stationary — the series mean-reverts).

**Why the test statistic is nonstandard:**

Under $H_0$, the regressor $y_{t-1}$ is itself a random walk, which violates the standard regression assumption that regressors are either fixed or stationary. As a result, the t-statistic for $\hat\gamma$ does not follow a t or normal distribution — it follows a **Dickey–Fuller distribution**, which is left-skewed and has larger critical values (in absolute terms) than the normal. The 5% critical value is roughly $-2.86$ (with constant, no trend) vs. $-1.96$ for a standard test. This means you need *more* evidence to reject the unit-root null than you would for a standard hypothesis test.

**Choosing the number of augmentation lags $p$:**
- Too few lags: serial correlation in $e_t$ biases the test.
- Too many lags: power decreases (you are estimating more parameters with the same data).
- Common approach: start with a generous $p_{\max}$ (e.g., $\lfloor 12(T/100)^{1/4} \rfloor$) and reduce using AIC/BIC or by testing whether the last lag is significant.

**Choosing deterministic terms ($a$, $bt$):**
- No constant, no trend: only appropriate if you believe the series has zero mean under stationarity — rare in practice.
- Constant only (`regression='c'`): appropriate for most economic series that fluctuate around a nonzero level.
- Constant + trend (`regression='ct'`): appropriate if the series could be trend-stationary (fluctuates around a deterministic trend).
- The choice affects critical values and power. Using `'ct'` when there is no trend reduces power.

**Low power: the Achilles heel of unit-root tests.**

The ADF test often fails to reject the null even when the series is stationary but highly persistent ($\rho$ close to 1). With $T = 100$ and $\rho = 0.95$, the test rejects the unit-root null only about 30–40% of the time at the 5% level. This is why "fail to reject" should never be interpreted as "the series is definitely I(1)."

#### 8) The KPSS test: stationarity as the null

The KPSS test (Kwiatkowski, Phillips, Schmidt, Shin, 1992) takes the opposite approach:

- $H_0$: the series is stationary (possibly around a trend).
- $H_1$: the series has a unit root.

**Why this is useful:** The ADF test is biased toward *not rejecting* the unit root when the series is near-unit-root. KPSS provides a complementary perspective. Running both tests together gives a 2×2 decision matrix:

| | ADF rejects (no unit root) | ADF fails to reject |
|---|---|---|
| **KPSS fails to reject (stationary)** | Strong evidence for stationarity | Inconclusive — possibly near-unit-root |
| **KPSS rejects (not stationary)** | Conflicting — investigate further (breaks? sample size?) | Strong evidence for nonstationarity |

**KPSS mechanics (brief):**

KPSS decomposes the series as:

$$
y_t = \xi t + r_t + u_t,
$$

where $r_t = r_{t-1} + \eta_t$ is a random walk and $u_t$ is stationary noise. Under $H_0$, $\mathrm{Var}(\eta_t) = 0$, so $r_t$ is a constant and the series is (trend-)stationary. The test statistic is based on the partial sums of OLS residuals from regressing $y_t$ on a constant (or constant + trend).

**Practical choice:** Use `regression='c'` (level stationarity) or `regression='ct'` (trend stationarity) to match what you tested in ADF. Mismatch between the two tests' specifications is a common source of confusion.

#### 9) Other unit-root tests (brief mentions)

- **Phillips–Perron (PP)**: Like ADF but uses a nonparametric correction for serial correlation instead of adding lagged differences. Can behave poorly in small samples with large MA components.
- **Zivot–Andrews**: Allows for a single structural break under the alternative hypothesis. Useful when you suspect a break might be masking stationarity.
- **Elliott–Rothenberg–Stock (ERS/DF-GLS)**: A power-improved version of ADF that first detrends the series using GLS. Generally recommended over standard ADF for small samples.

In practice, ADF + KPSS is the most common combination. If results are ambiguous, supplement with visual inspection and economic reasoning.

#### 10) Practical workflow: from raw series to classification

Here is the complete workflow you should follow for each series:

**Step 1: Visual inspection.**
Plot the series in levels. Ask:
- Is there an obvious trend? (Suggests I(1) or trend-stationary.)
- Are there structural breaks? (May affect test results.)
- Does volatility change over time? (May need log transformation first.)

**Step 2: Transform if needed.**
- If the series is positive and right-skewed (GDP, CPI), take logs first to stabilize variance.
- Then consider first differences (or log-differences = growth rates).

**Step 3: Test levels.**
- ADF with appropriate deterministic terms (constant, or constant + trend).
- KPSS with matching specification.
- Record p-values.

**Step 4: Test differences.**
- ADF and KPSS on $\Delta y_t$ (or $\Delta \log y_t$).
- If differences are stationary, the levels series is likely I(1).

**Step 5: Build summary table.**

| Series | ADF (levels) | KPSS (levels) | ADF (diff) | KPSS (diff) | Classification |
|---|---|---|---|---|---|
| log(GDP) | 0.87 | 0.02* | 0.001* | 0.46 | I(1) |
| UNRATE | 0.12 | 0.08 | 0.000* | 0.35 | I(1) |
| Spread (10Y−3M) | 0.03* | 0.15 | 0.000* | 0.41 | I(0) |

**Step 6: State your classification and justify it.** Include both statistical evidence and economic reasoning.

#### 11) Mapping to code (statsmodels)

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF test — constant only
result = adfuller(series.dropna(), regression='c', autolag='AIC')
adf_stat, adf_pval, used_lag, nobs, crit_vals, icbest = result
print(f"ADF stat: {adf_stat:.3f}, p-value: {adf_pval:.4f}, lags used: {used_lag}")

# ADF test — constant + trend
result_ct = adfuller(series.dropna(), regression='ct', autolag='AIC')
print(f"ADF (c+t) stat: {result_ct[0]:.3f}, p-value: {result_ct[1]:.4f}")

# KPSS test — level stationarity
kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(series.dropna(), regression='c')
print(f"KPSS stat: {kpss_stat:.3f}, p-value: {kpss_pval:.4f}")
# Note: KPSS p-values are bounded (0.01 to 0.10 in statsmodels).
# A reported p-value of 0.01 means "≤ 0.01".
```

**Common coding pitfalls:**
- Forgetting to drop NaN values before testing (creates errors or silent data issues).
- Using `regression='ct'` in ADF but `regression='c'` in KPSS (mismatch).
- Ignoring the KPSS p-value bounds — statsmodels reports "0.01" when the true p-value may be much smaller.

#### 12) Diagnostics + robustness (minimum set)

1. **Plot the level series**
   - Is there a visible trend? Multiple regimes? Structural breaks?
   - If you see a break (e.g., the Great Recession), consider subsample tests or the Zivot-Andrews test.

2. **Plot the differenced / growth-rate series**
   - Does it look mean-reverting? Is the mean roughly constant?
   - Check for remaining trends — if the differenced series still drifts, you may need a second difference (I(2)) or the series has a broken trend.

3. **ADF + KPSS on both levels and differences**
   - Do results agree? If they conflict, explain why: sample size? trend misspecification? structural breaks?
   - Try both `regression='c'` and `regression='ct'` to see if the conclusion is sensitive to the specification.

4. **ACF/PACF plots**
   - For the differenced series, the ACF should decay quickly. A slowly decaying ACF in differences suggests overdifferencing (the levels may have been stationary).
   - For levels, a slowly decaying ACF is consistent with a unit root or high persistence — this is expected.

5. **Sensitivity to lag length**
   - Re-run ADF with different augmentation lags. If the conclusion flips with small changes in lag selection, treat the result as fragile.

#### 13) Interpretation + reporting

When you report stationarity analysis:
- State which test(s) you ran and with what deterministic specification.
- Report both the test statistic and the p-value (or critical values).
- Show at least one plot (levels + differences) alongside numerical results.
- State your classification and justify it with both statistical and economic reasoning.
- Acknowledge ambiguity when it exists — "the evidence is mixed, but we proceed with differencing as the more conservative choice."

**What this does NOT mean:**
- "ADF rejects" is not proof that the series is truly stationary — it means the data are inconsistent with the unit-root null under the test's assumptions.
- "ADF fails to reject" is not proof of a unit root — the test may simply lack power.
- A small p-value does not tell you about the *economic* importance of the finding.
- Structural breaks can fool both ADF and KPSS — the tests assume a stable data-generating process.

#### Exercises

- [ ] Pick three macro series from the panel. For each: (a) plot levels and differences, (b) run ADF + KPSS on both, (c) classify as I(0) or I(1), (d) justify your classification in 4–6 sentences that reference both statistical evidence and economic reasoning.
- [ ] Demonstrate spurious regression: generate two independent random walks of length 200, regress one on the other, report $R^2$, t-stat, and Durbin-Watson. Repeat 100 times and show the distribution of rejection rates.
- [ ] For one near-unit-root series ($\rho = 0.97$, $T = 100$): (a) generate 500 simulations, (b) run ADF on each, (c) compute the fraction of times you correctly reject the unit-root null. This demonstrates the power problem.
- [ ] Compare the ADF result with `regression='c'` vs `regression='ct'` for a series with an obvious trend. Explain why the results might differ.
- [ ] Compute the half-life of a shock for an AR(1) with $\rho = 0.85$ and $\rho = 0.99$. What does this tell you about mean reversion speed?

### Project Code Map
- `data/sample/panel_monthly_sample.csv`: offline macro panel
- `src/features.py`: safe lag/diff/rolling feature helpers (`add_pct_change_features`, `add_rolling_features`)
- `src/macro.py`: GDP growth + label helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)

### Common Mistakes
- Running levels-on-levels regressions without checking integration order (spurious regression).
- Treating "fail to reject" in ADF as evidence *for* a unit root (it is not — the test may lack power).
- Mismatching deterministic terms between ADF and KPSS (e.g., testing with a trend in one but not the other).
- Overdifferencing an I(0) series — this introduces artificial negative autocorrelation.
- Ignoring structural breaks when interpreting test results (a break can cause a stationary series to look like a unit root).
- Using the ADF p-value like a standard t-test p-value (the critical values are nonstandard and depend on sample size and deterministic terms).
- Differencing *before* taking logs — this gives absolute changes instead of percentage changes, which is usually less meaningful for macro series.

<a id="summary"></a>
## Summary + Suggested Readings

Stationarity is not a technical formality — it determines whether your regression output is meaningful or gibberish. The core workflow is: **plot → test (ADF + KPSS) → transform → verify → proceed.** Always check both levels and differences, always use more than one test, and always combine statistical results with visual inspection and economic reasoning.

Key takeaways:
1. I(1) series require differencing (or cointegration analysis — see the next guide).
2. Spurious regression is not a theoretical curiosity — it is the default outcome when you regress one I(1) series on another.
3. Unit-root tests have low power near the boundary; treat results as evidence, not proof.
4. The choice of deterministic terms (constant, trend) affects both critical values and power.

Suggested readings:
- Hamilton, J. (1994): *Time Series Analysis* — Ch. 15–17 (the definitive treatment of unit roots and ADF)
- Enders, W. (2014): *Applied Econometric Time Series* — Ch. 4 (accessible introduction with many examples)
- Stock & Watson (2019): *Introduction to Econometrics* — Ch. 14 (time-series regression with nonstationary data)
- Hyndman & Athanasopoulos (2021): *Forecasting: Principles and Practice* — Ch. 9 (ARIMA models and differencing)
- Phillips, P.C.B. (1986): "Understanding Spurious Regressions in Econometrics" — the foundational paper on why I(1)-on-I(1) regressions fail
