### Core Data: Build Datasets You Can Trust

Modeling is downstream of data.
If you do not trust your data processing, you cannot trust your model output.

#### The three layers of a real data pipeline
> **Definition:** **Raw data** is the closest representation of what the source returned.

> **Definition:** **Processed data** is cleaned and aligned for analysis.

> **Definition:** A **modeling table** is the final table with features and targets aligned and ready for splitting.

#### Timing is part of the schema
When you build features, you are also defining "what was known when".
That is why frequency alignment and target shifting are first-class topics in this project.

#### A practical habit
Keep a simple data dictionary as you go:
- what is each column?
- what are its units?
- what frequency is it observed?
- what transformations did you apply?
