### Core Classification: Probabilities, Metrics, and Thresholds

In this project, classification is about predicting recession risk as a probability.

#### Logistic regression mechanics
Logistic regression models probabilities via log-odds:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

Then:

$$
 p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots)}}
$$

#### Metrics you should treat as standard
- ROC-AUC: ranking quality across thresholds
- PR-AUC: often more informative when positives are rare
- Brier score (or log loss): probability quality

#### Thresholding is a decision rule
> **Definition:** A **threshold** converts probabilities into labels.

Default 0.5 is rarely optimal for imbalanced, cost-sensitive problems.
Pick thresholds based on:
- decision costs
- desired recall/precision tradeoff
- calibration quality
