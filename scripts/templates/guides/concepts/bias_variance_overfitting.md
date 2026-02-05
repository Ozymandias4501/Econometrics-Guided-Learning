### Deep Dive: Bias–variance tradeoff and overfitting (why train ≠ test)

Understanding overfitting is essential for both ML and econometrics—especially in small-sample macro settings.

#### 1) Intuition (plain English)

Models can fail in two ways:
- **too simple:** cannot capture real patterns (high bias),
- **too flexible:** fits noise that does not repeat (high variance).

Overfitting is when performance looks great on training data but poor on new data.

#### 2) Notation + setup (define symbols)

Let:
- true outcome be $y = f(x) + \\varepsilon$,
- model prediction be $\\hat f(x)$,
- loss be squared error.

For a fixed $x$, the expected prediction error decomposes as:

$$
\\mathbb{E}[(\\hat f(x) - y)^2]
= \\underbrace{(\\mathbb{E}[\\hat f(x)] - f(x))^2}_{\\text{bias}^2}
\\; + \\;
\\underbrace{\\mathbb{E}[(\\hat f(x) - \\mathbb{E}[\\hat f(x)])^2]}_{\\text{variance}}
\\; + \\;
\\underbrace{\\mathrm{Var}(\\varepsilon)}_{\\text{noise}}.
$$

**What each term means**
- bias: systematic error from model misspecification/underfitting,
- variance: sensitivity to sample fluctuations (overfitting risk),
- noise: irreducible uncertainty in outcomes.

#### 3) Assumptions

This decomposition assumes:
- a stable data-generating process for the evaluation period,
- meaningful train/test separation (no leakage),
- loss function matches the task.

In time series, regime changes can dominate this story; walk-forward evaluation helps reveal it.

#### 4) Estimation mechanics: why complexity increases variance

More flexible models can fit training data better (lower bias) but often:
- increase variance,
- require more data to generalize,
- need regularization/constraints.

In regression, adding more correlated predictors can:
- increase coefficient variance,
- create unstable interpretations,
- improve in-sample fit without improving out-of-sample performance.

#### 5) Inference connection

Overfitting is not only an ML problem:
- specification search (trying many models) is a form of overfitting,
- p-values become misleading under heavy model selection.

#### 6) Diagnostics + robustness (minimum set)

1) **Train vs test gap**
- large gap suggests overfitting or leakage.

2) **Learning curves**
- performance as a function of sample size can reveal high variance.

3) **Cross-validation (time-aware)**
- use walk-forward folds for time series.

4) **Regularization sensitivity**
- ridge/lasso strength vs performance; look for a stable region.

#### 7) Interpretation + reporting

Report:
- out-of-sample metrics (not just in-sample),
- evaluation scheme (time split / walk-forward),
- and a simple overfitting check (train/test comparison).

#### Exercises

- [ ] Fit a simple model and a complex model; compare train vs test performance.
- [ ] Increase feature count and watch the train/test gap change.
- [ ] Plot a learning curve (even crude) by training on increasing time windows.
- [ ] Explain in 6 sentences how overfitting relates to specification search in econometrics.
