### Deep Dive: Walk-Forward Validation (Stability Over Time)

> **Definition:** **Walk-forward validation** repeatedly trains on the past and tests on the next time block.

It answers: "Does my model work across multiple eras, or only in one?"

#### Why walk-forward matters in economics
Economic relationships shift:
- policy changes
- technology shifts
- measurement changes
- financial crises

A single split can hide fragility.

#### Procedure (expanding window)
Typical expanding-window walk-forward:
- fold 1: train [0:t1], test [t1:t2]
- fold 2: train [0:t2], test [t2:t3]
- ...

> **Definition:** An **expanding window** keeps all past data in training.

> **Definition:** A **rolling window** uses only the most recent fixed-size window for training.

#### Pseudo-code
```python
# for each fold:
#   train = data[:train_end]
#   test  = data[train_end:train_end+test_size]
#   fit model on train
#   evaluate on test
#   advance train_end
```

#### Project touchpoints (where walk-forward is implemented)
- `src/evaluation.py` implements `walk_forward_splits` for fold generation.
- The walk-forward notebook uses this helper and asks you to plot metrics by era.

```python
from src.evaluation import walk_forward_splits

# Example: quarterly data with ~120 points
n = 120
splits = list(walk_forward_splits(n, initial_train_size=40, test_size=8))
splits[:3]
```

#### What to interpret
- If metrics vary widely across folds, the model is regime-sensitive.
- If performance collapses in certain periods, analyze what changed:
  - indicator behavior
  - label definition
  - missing data

#### Debug checklist
1. Ensure each fold trains strictly on the past.
2. Avoid reusing test periods for tuning.
3. Plot metrics over time, not just averages.
4. Keep the feature engineering fixed when comparing across folds.
