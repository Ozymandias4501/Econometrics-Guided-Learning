### Core Foundations: The Ideas You Will Reuse Everywhere

This project is built around a simple principle:

> **Definition:** A good model result is one that would still hold up if you had to use the model in the real world.

To get there, you need correct evaluation and correct data timing.

#### Time series vs cross-sectional (defined)
> **Definition:** A **time series** is indexed by time; ordering matters.

> **Definition:** **Cross-sectional** data compares many units at one time; ordering is not temporal.

Many ML defaults assume IID (independent and identically distributed) samples. Time series data is rarely IID.

#### What "leakage" really means
Leakage is not just a bug. It is a violation of the prediction setting.
If you are predicting the future, your features must be available in the past.

#### What "generalization" means in time series
In forecasting, generalization means:
- you train on one historical period
- you perform well in a later period

That is much harder than random-split generalization because the data generating process can change.

#### A practical habit
For every dataset row at time t, write a one-line statement:
- "At time t, we know X, and we are trying to predict Y at time t+h."

If you cannot state that clearly, it is very easy to leak information.
