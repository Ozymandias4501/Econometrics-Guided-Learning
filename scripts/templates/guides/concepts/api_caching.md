### Deep Dive: API caching — reproducible data is part of the method

APIs are convenient, but without caching they can make your analysis non-reproducible.

#### 1) Intuition (plain English)

If you fetch data from an API every time:
- the API can change,
- data can be revised,
- outages/rate limits break your workflow,
- and you cannot guarantee someone else gets the same dataset.

Caching turns “a query” into “a saved dataset artifact.”

#### 2) Notation + setup (define terms)

Think of your ingestion as a function:

$$
\\text{data} = F(\\text{endpoint}, \\text{params}).
$$

Caching adds a persistent mapping:

$$
\\text{cache\_key} = H(\\text{endpoint}, \\text{params}),
\\quad
\\text{cache}[\\text{cache\_key}] = \\text{data}.
$$

**What each term means**
- $F$: API fetch function.
- $H$: hash/key function that uniquely identifies a request.
- cache: local file storage (JSON/CSV).

#### 3) Assumptions (and what caching does/does not solve)

Caching assumes:
- you want repeatability for a given request,
- you can store raw responses (or cleaned versions) locally.

Caching does not solve:
- conceptual mistakes (wrong variables, wrong frequency),
- revisions vs real-time availability (you still must decide which you want).

#### 4) Mechanics: what to cache (and why)

Best practice in this repo:
- cache **raw responses** (so parsing is reproducible),
- also write **processed datasets** (so notebooks can run without re-fetching).

Cache naming should encode:
- dataset name,
- parameters,
- and time range (if applicable).

#### 5) Inference: reproducibility affects credibility

Inference is not just math; it is also:
- “Can someone reproduce the exact table/figure?”
- “Can we trace a result to a specific dataset version?”

Caching is therefore part of scientific validity.

#### 6) Diagnostics + robustness (minimum set)

1) **Cache hit/miss logging**
- print whether you loaded from disk or fetched from API.

2) **Schema checks**
- validate columns and dtypes after loading cached data.

3) **Re-run consistency**
- run the build pipeline twice and confirm identical processed outputs.

#### 7) Interpretation + reporting

When presenting results, state:
- whether data came from cached raw responses or fresh API calls,
- and where the cached artifacts live.

#### Exercises

- [ ] Run a dataset build twice; confirm the second run uses cached raw data.
- [ ] Delete one cached file and confirm the pipeline re-fetches and re-caches it.
- [ ] Add one schema assertion (expected columns) after loading cached data.
