# Guide: 04_walk_forward_validation

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/04_walk_forward_validation.ipynb`.

This classification module evaluates a **next-quarter recession probability model** using walk-forward (rolling-origin) validation, the correct evaluation strategy for time-series forecasting tasks.

### Key Terms (defined)
- **Walk-forward validation**: repeated "train on past, test on next" evaluation scheme that respects temporal order.
- **Expanding window**: training set grows with each fold (all history up to the cutoff).
- **Rolling window**: training set is a fixed-length window that slides forward.
- **Embargo / gap period**: observations dropped between training and test sets to prevent information leakage at boundaries.
- **Concept drift**: a shift in the data-generating process over time that degrades model performance.
- **Data leakage**: using information from the future (directly or indirectly) when training or evaluating a model.
- **Fold**: one train/test iteration within a cross-validation scheme.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Walk-forward splits
- Complete notebook section: Metric stability
- Complete notebook section: Failure analysis
- Implement expanding-window walk-forward splits and compute fold-by-fold ROC-AUC.
- Plot metrics across folds to identify periods of degraded performance.
- Compare walk-forward AUC to a random-split AUC and quantify the leakage inflation.

### Alternative Example (Not the Notebook Solution)
```python
# Walk-forward evaluation of a quarterly recession model (toy data):
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
n_quarters = 120  # 30 years of quarterly data

# Simulated macro features (3 indicators)
X = rng.normal(size=(n_quarters, 3))
# Add autocorrelation to mimic real macro data
for t in range(1, n_quarters):
    X[t] = 0.6 * X[t - 1] + 0.4 * X[t]

# Recession probability depends on lagged features
logit = -2.0 + 1.5 * X[:, 0] - 0.8 * X[:, 1]
prob = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, prob)

# --- Random CV (WRONG for time series) ---
from sklearn.model_selection import cross_val_score

clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
random_cv_auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()

# --- Walk-forward (CORRECT) ---
min_train = 40  # at least 10 years of training data
step = 4        # advance 1 year (4 quarters) per fold
embargo = 1     # skip 1 quarter between train and test

wf_aucs = []
for cutoff in range(min_train, n_quarters - step - embargo, step):
    train_idx = list(range(cutoff))
    test_idx = list(range(cutoff + embargo, min(cutoff + embargo + step, n_quarters)))

    if len(set(y[test_idx])) < 2:
        continue  # skip folds with only one class

    clf_wf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    clf_wf.fit(X[train_idx], y[train_idx])
    y_prob = clf_wf.predict_proba(X[test_idx])[:, 1]
    wf_aucs.append(roc_auc_score(y[test_idx], y_prob))

print(f"Random CV AUC:    {random_cv_auc:.3f}")
print(f"Walk-forward AUC: {np.mean(wf_aucs):.3f} +/- {np.std(wf_aucs):.3f}")
print(f"Leakage inflation: {random_cv_auc - np.mean(wf_aucs):.3f}")
```

**Interpreting the output:** With autocorrelated macro data, random CV typically reports an AUC around 0.90--0.92 because test folds contain observations temporally adjacent to training observations, effectively leaking information. Walk-forward AUC, which strictly trains on the past and tests on the future, typically reports 0.75--0.80. The gap (0.10--0.15) is the leakage inflation -- it tells you how much of your apparent performance is an artifact of violating temporal ordering.


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

**Prerequisites:** [Classification foundations](00_recession_classifier_baselines.md) -- core classification concepts (probabilities, losses, thresholds, metrics, calibration overview, class imbalance).

### Deep Dive: Walk-Forward Validation

Walk-forward (rolling-origin) validation is the most common evaluation scheme for forecasting. It simulates the real-world deployment pattern: train on all available history, predict the next period, advance time, repeat.

#### 1) WHY random cross-validation fails for time series

Standard $k$-fold CV randomly assigns observations to folds. For time series, this creates two forms of leakage:

**Direct leakage:** An observation from 2020-Q3 might appear in the training set while 2020-Q2 is in the test set. The model has literally "seen the future" that is adjacent to what it is predicting.

**Autocorrelation leakage:** Even without direct overlap, time-series observations near each other are correlated. If 2019-Q4 is in training and 2020-Q1 is in test, the model benefits from the autocorrelation structure -- the training observation is nearly a copy of the test observation. This inflates apparent performance.

**The consequence:** Random CV reports metrics that are unrealistically optimistic. In macro-financial forecasting, the inflation is typically 0.05--0.15 AUC points. A model that appears to have AUC = 0.92 under random CV may actually have AUC = 0.78 under walk-forward evaluation. Decisions based on the inflated number (e.g., deploying the model, publishing the result) would be based on an illusion.

#### 2) Walk-forward scheme: expanding vs rolling window

**Expanding window** -- at each fold, the training set includes all observations from the beginning of the sample up to the cutoff:

```
Fold 1:  [TRAIN TRAIN TRAIN TRAIN ........] [gap] [TEST]  ............
Fold 2:  [TRAIN TRAIN TRAIN TRAIN TRAIN ...] [gap] [TEST]  ..........
Fold 3:  [TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ........
Fold 4:  [TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ..
```

**Rolling window** -- the training set is a fixed-length window that slides forward:

```
Fold 1:  .... [TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ................
Fold 2:  ...... [TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ..............
Fold 3:  ........ [TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ............
Fold 4:  .......... [TRAIN TRAIN TRAIN TRAIN] [gap] [TEST]  ..........
```

**Trade-offs:**
- Expanding window uses all available data, giving more statistical power but potentially including stale regimes (e.g., pre-2008 data may hurt post-crisis predictions).
- Rolling window focuses on recent data, adapting to regime changes but with higher variance due to smaller training sets.
- In practice, try both and compare. If performance is similar, prefer expanding (more data). If the rolling window is substantially better, the data-generating process has likely shifted (concept drift).

#### 3) Choosing walk-forward parameters

**Minimum training size.** The first fold must have enough data for the model to learn. Rules of thumb:
- At least 10 positive events (recessions) in the training set.
- For quarterly data with ~15% recession rate, this means at least 60--80 quarters (~15--20 years).
- Too little training data makes early folds unreliable; too much wastes evaluation folds.

**Test window size.** How many observations to evaluate per fold:
- One observation per fold gives the most folds but very noisy per-fold metrics.
- One year of observations (4 quarters) per fold balances granularity and stability.
- Match the deployment cadence: if you would retrain quarterly, test on one quarter.

**Number of folds.** Determined by the minimum training size, test window, and total sample:
$$
M = \left\lfloor \frac{T - T_{\min} - g}{s} \right\rfloor,
$$
where $T$ is the total sample size, $T_{\min}$ is the minimum training size, $g$ is the gap (embargo), and $s$ is the step size (how far the cutoff advances per fold). More folds give a more stable average metric but require a longer sample.

**Step size.** How far to advance the cutoff between folds:
- Step = test window size gives non-overlapping test sets (recommended).
- Step < test window size gives overlapping test sets (more folds but correlated metrics).

#### 4) Embargo / gap period

An embargo (or "purge") drops observations between training and test sets to prevent leakage at the boundary:

```
[...TRAIN TRAIN TRAIN] [GAP GAP] [TEST TEST...]
```

**Why it matters:**
- Feature engineering often uses lagged or rolling windows (e.g., 3-quarter rolling average). An observation at $t$ may incorporate information from $t+1$ or $t+2$ if the pipeline is not perfectly aligned.
- In health economics, claims data often has a "run-out" period: a claim from Q1 may not be fully adjudicated until Q2. If the label is based on adjudicated claims, training on Q1 data could implicitly use Q2 information.
- A gap of 1--2 periods (matching the maximum lag in feature engineering) is a safe default.

**Conservative rule:** Set the embargo equal to the longest lag used in feature construction. If you compute a 4-quarter rolling average, use a 4-quarter embargo.

#### 5) Aggregating metrics across folds

The primary output of walk-forward validation is a sequence of per-fold metrics: $\{m_1, m_2, \dots, m_M\}$ (e.g., AUC per fold).

**Summary statistics:**
- Report **mean $\pm$ std** across folds.
- The mean estimates expected future performance.
- The standard deviation captures performance variability (which is itself informative).

**But also plot metrics over time.** A single number (mean AUC = 0.80) hides important information:

- **Stable performance:** all folds near 0.80. The model generalizes consistently.
- **Degrading performance:** early folds at 0.90, recent folds at 0.65. This signals **concept drift** -- the data-generating process has changed and the model is becoming stale.
- **Spiky performance:** large swings across folds. The model may be sensitive to specific economic regimes (e.g., performs well in expansions but fails in recessions -- precisely when you need it most).

**Formal test for trend:** Regress fold metrics on fold index. A significantly negative slope suggests performance degradation. This is informal but useful as a diagnostic.

**Caution on standard errors:** Because folds are time-ordered, fold errors are typically positively autocorrelated. Standard confidence intervals ($\bar{m} \pm 1.96 \cdot s/\sqrt{M}$) assume independence and therefore understate uncertainty. Treat the standard deviation descriptively rather than constructing formal confidence intervals.

#### 6) Connection to production deployment

Walk-forward validation is not just an evaluation trick -- it mirrors how forecasting models are actually deployed:

1. **Train** on all available history up to today.
2. **Predict** the next quarter's recession probability.
3. **Observe** the outcome.
4. **Retrain** with the new observation included.
5. Repeat.

If your walk-forward evaluation shows good performance, you have evidence that this deploy-retrain cycle will work. If random CV shows good performance but walk-forward does not, you have evidence that the model cannot actually predict the future from the past -- it only appears to because random CV leaks future information.

In health economics, this maps to real deployment patterns:
- A hospital readmission model is trained on historical data and predicts readmission risk for newly discharged patients.
- A health plan risk adjustment model is trained on last year's claims and predicts this year's costs.
- A disease surveillance model is trained on past outbreak data and predicts current-week outbreak probability.

In all cases, the model must predict forward in time, and walk-forward validation is the evaluation strategy that matches this requirement.

#### 7) Health econ example: walk-forward evaluation of a quarterly recession probability model

Consider evaluating a logistic regression that predicts next-quarter recession using yield curve, unemployment claims, and consumer sentiment features. The sample covers 1970-Q1 to 2023-Q4 (216 quarters, ~35 recession quarters).

**Setup:**
- Minimum training size: 80 quarters (1970--1989).
- Test window: 4 quarters (1 year).
- Step: 4 quarters (non-overlapping test sets).
- Embargo: 1 quarter.
- This gives $M = \lfloor(216 - 80 - 1) / 4\rfloor = 33$ folds.

**Results (illustrative):**

| Evaluation method | AUC | Brier score |
|---|---|---|
| 5-fold random CV | 0.92 | 0.08 |
| Walk-forward (expanding) | 0.78 | 0.14 |
| Walk-forward (rolling, 60Q window) | 0.76 | 0.15 |

The random CV AUC of 0.92 is inflated by ~0.14 points. This is typical for macro-financial models where features are highly autocorrelated.

**Plotting fold-by-fold AUC reveals additional insights:**
- Folds covering 2001 and 2008 recessions: AUC ~0.85 (the model captures these well, as yield curve inversion is a strong signal).
- Folds covering 2020: AUC ~0.55 (near random). The COVID recession was driven by an exogenous shock, not the macro indicators the model uses. Walk-forward exposes this failure; random CV obscures it.
- Post-2020 folds: AUC recovers to ~0.75 as the economy returns to "normal" macro dynamics.

**Key insight for health economists:** Models trained on pre-pandemic data may perform poorly during and after the pandemic. Walk-forward validation, by evaluating across different regimes, reveals whether your model is robust to structural breaks or only works in stable periods.

#### 8) Diagnostics and robustness

1) **Compare expanding vs rolling windows.** If the rolling window significantly outperforms the expanding window, older data is hurting the model (regime change). Consider using the rolling window or weighting recent observations more heavily.

2) **Vary the embargo period.** If adding a 2-quarter embargo drops AUC substantially compared to no embargo, your features or labels likely have temporal leakage. Investigate feature engineering pipelines for look-ahead bias.

3) **Test multiple horizons.** Repeat the walk-forward scheme for $h=1, 2, 4$ quarters. If performance degrades sharply with horizon, the model captures short-term momentum but not structural risk.

4) **Subset analysis.** Report walk-forward metrics separately for recession and expansion periods. A model that only performs well in expansions (correctly predicting "no recession" when things are calm) is not useful.

5) **Retraining frequency sensitivity.** Compare step sizes of 1, 4, and 8 quarters. If more frequent retraining helps substantially, the model benefits from adapting to recent data -- another sign of concept drift.

#### Exercises

- [ ] Implement expanding-window walk-forward splits. Compute fold-by-fold ROC-AUC and Brier score. Plot them over time and identify the worst-performing period.
- [ ] Compare random 5-fold CV AUC to walk-forward AUC on the same data. Quantify the leakage inflation.
- [ ] Implement a rolling-window variant with a 60-quarter window. Compare average AUC to the expanding-window result. Which performs better, and what does that imply about regime stability?
- [ ] Add a 1-quarter embargo to your walk-forward scheme. Does performance change? If so, investigate what feature(s) might be causing boundary leakage.
- [ ] Identify a fold where the model fails badly. Propose a hypothesis: is it a regime change, a missing feature, or an outlier event?

### Key Terms (Walk-Forward Validation)

| Term | Definition |
|---|---|
| **Walk-forward validation** | Repeated train-on-past, test-on-future evaluation that respects temporal order. |
| **Expanding window** | Training set grows with each fold, using all history up to the cutoff. |
| **Rolling window** | Training set is a fixed-length window that slides forward. |
| **Embargo / gap** | Observations dropped between train and test to prevent boundary leakage. |
| **Concept drift** | A shift in the data-generating process that degrades model performance over time. |
| **Leakage inflation** | The gap between random-CV metrics and walk-forward metrics, caused by temporal information leakage. |
| **Fold** | One train/test iteration within the walk-forward scheme. |
| **Step size** | How many periods the cutoff advances between folds. |
| **Minimum training size** | The smallest training set used (in the first fold). |

### Common Mistakes (Walk-Forward Validation)

- **Using random CV for time-series data.** This is the single most common evaluation error in applied forecasting. Always use walk-forward for temporal data.
- **No embargo period.** Even with walk-forward splits, feature engineering with rolling windows can leak information across the train/test boundary. Add a gap at least as long as your longest feature lag.
- **Reporting only mean AUC.** The mean hides performance variability and trend. Always plot fold metrics over time.
- **Too few folds.** With only 3--4 folds, the average is dominated by noise. Aim for at least 10 folds if the sample allows.
- **Too little training data in early folds.** If your first fold has only 20 observations, the model is undertrained and the fold metric is unreliable. Set a reasonable minimum training size.
- **Confusing walk-forward with time-series split.** A single time split is one fold of walk-forward. Walk-forward repeats this across many cutoffs.
- **Ignoring class balance per fold.** Some folds may have zero positive cases (no recession). Skip these folds or flag them rather than computing undefined metrics.

### Project Code Map
- `src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall) and splits (`time_train_test_split_index`, `walk_forward_splits`)
- `scripts/train_recession.py`: training script that writes artifacts
- `scripts/predict_recession.py`: prediction script that loads artifacts
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- Explain why random cross-validation produces inflated metrics for time-series data and quantify the leakage.
- Implement both expanding-window and rolling-window walk-forward validation schemes.
- Choose appropriate parameters: minimum training size, test window, step size, and embargo period.
- Aggregate and visualize fold-by-fold metrics to detect concept drift and regime-dependent failures.
- Connect walk-forward evaluation to real-world deployment patterns in health economics.

Suggested readings:
- Tashman (2000): "Out-of-sample tests of forecasting accuracy: an analysis and review" -- the classic reference on rolling-origin evaluation.
- Bergmeir & Benitez (2012): "On the use of cross-validation for time series predictor evaluation."
- Cerqueira, Torgo & Mozetic (2020): "Evaluating time series forecasting models: an empirical study on performance estimation methods."
- Hyndman & Athanasopoulos: *Forecasting: Principles and Practice* (3rd ed.), Ch. 5 -- freely available online.
- scikit-learn docs: `TimeSeriesSplit` (a simpler variant of walk-forward).
- Roberts et al. (2017): "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" -- relevant for health econ data with geographic and temporal structure.
