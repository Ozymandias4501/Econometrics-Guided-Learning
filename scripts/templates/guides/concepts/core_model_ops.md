### Core Model Ops: reproducibility, artifacts, and “make it runnable by someone else”

Model ops in this repo is intentionally lightweight: the goal is not production ML infrastructure, but **reproducible experiments**.

#### 1) Intuition (plain English)

If a result cannot be reproduced, it is not trustworthy.
Reproducibility in applied econometrics means:
- you can re-run data building + training and get the same outputs,
- you can explain which config/dataset produced which figure/table,
- you can compare runs without guessing.

#### 2) Notation + setup (define terms)

> **Definition:** An **artifact** is a saved output: model file, metrics JSON, predictions CSV, plots.

> **Definition:** A **run ID** is a unique identifier for one training/evaluation run.

> **Definition:** A **config** is a recorded set of choices (features, model type, hyperparameters, split dates).

#### 3) Assumptions (what reproducibility requires)

To reproduce results you need:
- fixed code + fixed config,
- fixed data version (processed file hash or timestamp),
- controlled randomness (seeds),
- clear train/test boundaries.

#### 4) Mechanics: the minimal pipeline

This repo’s scripts are organized around:
- build datasets → train → evaluate → save artifacts → make a report/dashboard.

Good run outputs typically include:
- model object (pickle/joblib),
- metrics (JSON),
- predictions (CSV with timestamps),
- a small “run summary” (config + git hash if desired).

#### 5) Diagnostics + robustness (minimum set)

1) **Re-run check**
- run the same script twice with the same config and confirm outputs match.

2) **Config traceability**
- every artifact folder should contain the config that produced it.

3) **Data provenance**
- record whether you used sample vs processed data.

4) **Monitoring mindset (even in a notebook project)**
- if this were shipped, what would you track over time? (drift, calibration, performance)

#### 6) Interpretation + reporting

Model ops is not about claiming your model is “deployable.”
It is about making your analysis:
- auditable,
- comparable,
- easy to extend.

#### Exercises

- [ ] Run a training script that writes outputs and confirm artifacts appear under a run ID directory.
- [ ] Add one field to a metrics JSON (e.g., split dates) and verify it is written.
- [ ] Re-run with a different config and compare runs in a short table.
