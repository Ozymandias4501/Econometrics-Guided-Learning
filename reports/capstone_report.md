# Capstone Report: Recession Prediction (Template)

## 1) Executive Summary
- What you built:
- Key result (metrics + what they mean in practice):
- Biggest limitation:

## 2) Problem Definition
- Prediction target:
  - Technical recession (2 consecutive negative GDP growth quarters)
  - Predict next quarter: `target_recession_next_q`
- Why this target matters:

## 3) Data Sources
- FRED series used:
- Census/ACS data used (if applicable):
- Time range:

## 4) Label Construction
- GDP series: `GDPC1`
- GDP growth definition used (QoQ / YoY / annualized):
- Technical recession definition implemented:
- Discussion: why this is a proxy (vs official dating):

## 5) Feature Engineering
- Frequency alignment choices:
- Quarterly aggregation method:
- Lags used:
- Feature scaling choices:

## 6) Models
- Baselines:
- Final model:
- Why that model was chosen:

## 7) Evaluation
- Split strategy (time split / walk-forward):
- Metrics (ROC-AUC, PR-AUC, Brier, etc.):
- Threshold decision:

## 8) Interpretation and Diagnostics
- Top drivers (coefficients / permutation importance):
- Error analysis (when did it fail? why?):
- Calibration analysis:

## 9) Limitations and Risks
- Data revisions (GDP and macro indicators):
- Structural breaks:
- Leakage risks checked:
- Ethical / decision risks:

## 10) Suggested Next Steps
- Data improvements:
- Modeling improvements:
- Evaluation improvements:

## Appendix
- Config used:
- Run ID:
- Artifact paths under `outputs/`:
