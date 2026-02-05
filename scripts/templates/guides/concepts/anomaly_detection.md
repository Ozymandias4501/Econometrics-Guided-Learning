### Deep Dive: Anomaly detection — finding “unusual” macro periods

Anomaly detection flags observations that look unusual relative to typical patterns.

#### 1) Intuition (plain English)

Crises (2008, 2020) look different in multivariate indicator space.
Anomaly detection can highlight these periods without a recession label.

Use it to:
- detect outliers,
- generate hypotheses,
- build monitoring dashboards.

#### 2) Notation + setup (define symbols)

Let $x_t$ be the feature vector at time $t$.
An anomaly detector produces a score:
$$
s_t = s(x_t),
$$
where larger (or smaller, depending on convention) means “more anomalous.”

Simple baseline: z-score on one feature:
$$
z_t = \\frac{x_t - \\bar x}{s}.
$$

Multivariate baselines:
- distance from mean (Mahalanobis distance),
- isolation forest score,
- reconstruction error from PCA.

#### 3) Assumptions

Anomaly detection assumes:
- features are scaled comparably,
- “typical” behavior exists and is represented in the data.

Structural breaks can shift what is “normal,” so anomaly thresholds should be treated as time-varying in real monitoring.

#### 4) Estimation mechanics (high level)

Common approaches:
- **PCA reconstruction:** anomalies have large reconstruction error using a few PCs.
- **Isolation forest:** anomalies are easier to isolate with random splits.
- **Distance-based:** far from center in standardized space.

#### 5) Inference: focus on false positives/negatives

Anomaly detection is not hypothesis testing; it is a scoring/ranking tool.
Validate by:
- checking whether known crises score high,
- inspecting the top anomalies qualitatively.

#### 6) Diagnostics + robustness (minimum set)

1) **Known-event sanity**
- do 2008/2020 periods appear as anomalies?

2) **Feature contribution**
- which features drive the anomaly score? (inspect deviations)

3) **Threshold sensitivity**
- how many anomalies do you flag under different thresholds?

4) **Stability**
- do top anomalies persist if you change feature set or standardization window?

#### 7) Interpretation + reporting

Report:
- anomaly method and preprocessing,
- how threshold was chosen,
- examples of top anomalies and what indicators drove them.

**What this does NOT mean**
- anomalies are not causal explanations; they are flags.

#### Exercises

- [ ] Fit an anomaly detector and list the top 10 anomalous dates; interpret at least 3.
- [ ] Compare two methods (PCA reconstruction vs isolation forest) and discuss differences.
- [ ] Vary the anomaly threshold and report how many periods are flagged.
