# Guide: 04_census_api_microdata_fetch

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/04_census_api_microdata_fetch.ipynb`.

**Prerequisites:** [Data engineering foundations](00_fred_api_and_caching.md) — core data principles (tidy format, caching, validation, reproducibility).

This guide focuses on **fetching, validating, and assembling county-level data
from the U.S. Census Bureau's American Community Survey (ACS)**. The ACS is the
primary source for demographic, economic, and social characteristics at
sub-national geographies — exactly the kind of data you need for health economics
research on disparities, access, and social determinants of health.

### Key Terms (defined)
- **ACS (American Community Survey)**: an ongoing Census Bureau survey that produces annual estimates of demographic, social, economic, and housing characteristics.
- **FIPS code**: the Federal Information Processing Standards code that uniquely identifies every state (2 digits) and county (state 2 + county 3 = 5 digits).
- **Margin of error (MOE)**: a measure of sampling uncertainty around each ACS estimate; the Census Bureau publishes MOEs alongside point estimates.
- **ACS 1-year vs 5-year estimates**: 1-year estimates cover areas with population 65,000+; 5-year estimates pool five survey years to cover all geographies including small counties.
- **Table shell**: the Census Bureau's published layout showing which variables belong to a given ACS table (e.g., B17001 for poverty status).

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn Census-specific concepts, API mechanics, and common pitfalls.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Browse the ACS variable list for your target geography and identify relevant table IDs (e.g., B17001 for poverty, B27001 for insurance status).
- Fetch county-level data from the Census API (or load the sample cache) for at least two ACS tables.
- Inspect the returned schema: confirm column names match expected variable codes, all values are numeric (watch for string `-666666666` missing-data sentinels), and geography columns parse correctly.
- Validate that FIPS codes are 5 characters with a leading zero where needed (e.g., `'01001'` not `1001`).
- Compute derived rates from raw counts (poverty rate, uninsured rate, etc.) and assert they fall in [0, 1].
- Merge multiple ACS tables on FIPS code; confirm the merge is 1:1 with no unexpected nulls.
- Save the cleaned county-level DataFrame to `data/processed/` and document the vintage year.

### Alternative Example (Not the Notebook Solution)
```python
# Toy county-level Census merge (not the real notebook data):
import pandas as pd

# Two ACS tables for the same counties
pop = pd.DataFrame({
    'fips': ['01001', '01003', '01005'],
    'total_pop': [55000, 220000, 27000],
    'pop_below_poverty': [8250, 26400, 5670],
})
insurance = pd.DataFrame({
    'fips': ['01001', '01003', '01005'],
    'total_pop_ins': [55000, 220000, 27000],
    'uninsured': [4400, 15400, 3780],
})

# Merge on FIPS code
county = pop.merge(insurance, on='fips', validate='1:1')

# Derive rates
county['poverty_rate']   = county['pop_below_poverty'] / county['total_pop']
county['uninsured_rate'] = county['uninsured'] / county['total_pop_ins']

# Validate
for col in ['poverty_rate', 'uninsured_rate']:
    assert (county[col] >= 0).all() and (county[col] <= 1).all(), f"{col} out of range"

print(county[['fips', 'poverty_rate', 'uninsured_rate']])
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### ACS 1-year vs 5-year estimates: when to use each

The Census Bureau publishes two main ACS products:

| Product | Sample period | Minimum population | Best for |
|---|---|---|---|
| **1-year** | Single calendar year | 65,000+ | Large counties/metros; most current data |
| **5-year** | Rolling 5-year pool | No minimum | Small counties, census tracts, rare subpopulations |

**When to choose 1-year:** You need the most recent snapshot and your geography is
large enough (population above 65,000). For health economics, this usually means
state-level or large-metro analyses.

**When to choose 5-year:** You need county-level coverage for the entire U.S.,
including rural counties with small populations. The 5-year product covers all
~3,200 counties. The trade-off is that the estimate blends five years of data, so
it responds slowly to rapid changes (e.g., a factory closure that raises poverty
in a single year may be diluted).

**Rule of thumb for this project:** Use 5-year estimates for county-level panels.
Use 1-year estimates only if you are restricting analysis to large geographies and
need year-to-year comparisons.

### Margins of error in ACS data

Every ACS estimate is based on a sample, not a full census count. The Census Bureau
reports a 90% margin of error (MOE) for each estimate. The true value lies within
$\text{estimate} \pm \text{MOE}$ with approximately 90% confidence.

**Why this matters:** For small counties, MOEs can be large relative to the
estimate. A poverty rate of 15% with a MOE of 8 percentage points means the true
rate could plausibly be anywhere from 7% to 23%.

#### Propagating MOEs when combining estimates

When you add two ACS estimates (e.g., summing male + female uninsured counts):

$$
\text{MOE}_{\text{sum}} = \sqrt{\text{MOE}_1^2 + \text{MOE}_2^2}
$$

When you compute a derived proportion $p = X / Y$ (e.g., poverty rate = people
below poverty / total population), the approximate MOE is:

$$
\text{MOE}_p \approx \frac{1}{Y}\sqrt{\text{MOE}_X^2 - p^2 \cdot \text{MOE}_Y^2}
$$

If the term under the square root is negative (which happens when $p$ is close to
1), the Census Bureau recommends the simpler approximation:

$$
\text{MOE}_p \approx \frac{1}{Y}\sqrt{\text{MOE}_X^2 + p^2 \cdot \text{MOE}_Y^2}
$$

**Practical advice:** Always fetch the MOE columns alongside the estimate columns.
Flag any county where the coefficient of variation ($\text{MOE} / (1.645 \times \text{estimate})$)
exceeds 30% — those estimates are unreliable and should be noted or excluded in
sensitivity analyses.

### FIPS codes: structure and common pitfalls

FIPS codes are the standard geographic identifiers for merging Census data with
other sources (health data, economic data, shapefiles).

**Structure:**

| Level | Digits | Example | Meaning |
|---|---|---|---|
| State | 2 | `01` | Alabama |
| County | 5 (state + county) | `01001` | Autauga County, AL |
| Tract | 11 (state + county + tract) | `01001020100` | Tract in Autauga County |

**Common pitfalls:**

1. **Leading zeros.** FIPS codes must be stored as strings, not integers. `int('01001')` becomes `1001`, and merges silently fail. Always cast to string and zero-pad: `str(code).zfill(5)`.

2. **Code changes over time.** Counties occasionally merge, split, or get reassigned new codes. The most frequent example is Virginia's independent cities. The Census Bureau publishes a FIPS change history; check it if you are building multi-year panels.

3. **State-only vs full FIPS.** Some datasets provide separate `state` and `county` columns. Concatenate them carefully: `fips = state_fips.str.zfill(2) + county_fips.str.zfill(3)`.

4. **Shannon County, SD became Oglala Lakota County** (FIPS changed from `46113` to `46102` in 2015). If your panel crosses 2015, you must recode or you will have a "disappearing" county and a "new" county that are actually the same place.

### Variable selection from ACS: navigating the variable list

The ACS publishes thousands of variables organized into numbered tables. Finding
the right variables is half the battle.

**Table types:**

- **B-tables (Detailed Tables):** Most granular. Example: `B17001` = poverty status by age and sex.
- **S-tables (Subject Tables):** Pre-computed summaries (rates, medians). Example: `S2701` = health insurance coverage.
- **DP-tables (Data Profiles):** Broad demographic, social, economic, and housing snapshots.
- **C-tables (Collapsed Tables):** Simplified versions of B-tables with fewer cross-tabulations.

**How to find variables:**

1. Start at the [Census API variable list](https://api.census.gov/data.html) or the data.census.gov search.
2. Identify the table ID (e.g., B27001 for health insurance by age/sex).
3. Look up the table shell to see the full variable hierarchy — the shell shows which row codes correspond to totals, male/female breakdowns, age brackets, etc.
4. In the API call, request specific variable codes (e.g., `B27001_001E` for the total, `B27001_005E` for uninsured males 6–18).

**Naming convention in this project:** Rename raw variable codes to human-readable
names immediately after fetching. Carry both the original code and the readable
name in your data dictionary.

### Building county-level panels: merging across years

To study trends (e.g., how county-level uninsured rates changed from 2015 to 2022),
you fetch the same variables for each year and stack or merge:

```python
# Pseudocode for building a county-year panel
frames = []
for year in range(2015, 2023):
    df = fetch_acs(year=year, variables=[...], geography='county')
    df['year'] = year
    frames.append(df)
panel = pd.concat(frames, ignore_index=True)
```

**Challenges:**

- **Geography changes.** If a county FIPS code changed between years, you need a crosswalk. The Census Bureau publishes these.
- **Variable code changes.** The same conceptual variable may have a different code in different ACS vintages. Always verify by checking the table shell for each year.
- **5-year overlap.** The 2015 5-year ACS covers 2011–2015; the 2016 5-year covers 2012–2016. These overlap by four years, so year-to-year changes in 5-year estimates are heavily smoothed and should not be interpreted as annual changes.

### Census API mechanics

The Census Bureau API follows a simple URL pattern:

```
https://api.census.gov/data/{year}/acs/acs5?get={variables}&for={geography}&key={api_key}
```

**Key parameters:**

| Parameter | Meaning | Example |
|---|---|---|
| `year` | ACS vintage year | `2022` |
| `acs5` or `acs1` | Product (5-year or 1-year) | `acs5` |
| `get` | Comma-separated variable codes | `B17001_001E,B17001_002E` |
| `for` | Target geography | `county:*` (all counties) |
| `in` | Parent geography constraint | `state:01` (only Alabama counties) |
| `key` | Your API key | (from census.gov/developers) |

**Tips:**

- Request an API key at [census.gov/developers](https://api.census.gov/data/key_signup.html). The key is free and raises your rate limit.
- The API returns JSON: a list of lists where the first row is column headers.
- You can fetch up to 50 variables in a single call. For larger requests, batch into multiple calls and merge.
- Append `E` for estimates and `M` for margins of error to any variable code (e.g., `B17001_001E` and `B17001_001M`).

### Health economics applications

County-level ACS data is central to health economics and health services research:

- **Uninsured rates** (table B27001/S2701): key outcome variable for studying Medicaid expansion effects, coverage gaps, and access barriers.
- **Poverty rates** (table B17001): poverty is one of the strongest social determinants of health. County poverty rates are used as controls, treatment indicators (e.g., poverty thresholds for program eligibility), and outcome measures.
- **Demographic composition** (tables B01001, B03002): age structure, racial/ethnic composition, and sex ratios are essential controls in health disparities research. Age-adjusting mortality rates, for example, requires the county age distribution from ACS.
- **Median household income** (table B19013): used as a proxy for economic well-being and as a stratifying variable in analyses of health spending, utilization, and outcomes.
- **Educational attainment** (table B15003): strongly correlated with health literacy, preventive care utilization, and mortality.

When merging ACS data with health outcomes (e.g., CDC mortality files, HCUP discharge data), FIPS codes are the linking key. Always verify that both datasets use the same FIPS vintage.

### Diagnostics and robustness checks

1. **FIPS validation.** After fetching, assert all FIPS codes are 5-character strings. Check that the count of unique counties matches expectations (~3,200 for a full U.S. pull).

2. **Missing-data sentinels.** The Census API uses `-666666666` (or similar) for suppressed or unavailable estimates. Convert these to `NaN` immediately and count how many counties are affected.

3. **Rate bounds.** Every derived rate (poverty rate, uninsured rate, etc.) must be in [0, 1]. Values outside this range indicate a formula error or a mismatch between numerator and denominator universes.

4. **MOE reasonableness.** Flag estimates where $\text{MOE} > 0.5 \times \text{estimate}$. These are too imprecise for reliable inference.

5. **Merge diagnostics.** When merging multiple ACS tables on FIPS, use `validate='1:1'` and check for unexpected nulls. A mismatch usually means one table has a different geography definition (e.g., county equivalents in Louisiana are called "parishes").

6. **Cross-year consistency.** If building a panel, plot a few counties over time. A sudden jump may indicate a FIPS code change, a variable definition change, or a real demographic shift — investigate before proceeding.

### Exercises

- [ ] Fetch two ACS tables (e.g., B17001 and B27001) for all counties in one state and merge them on FIPS code.
- [ ] Compute the poverty rate and uninsured rate; verify both are between 0 and 1.
- [ ] For one small county, compare the estimate to its MOE and discuss whether the estimate is reliable.
- [ ] Build a two-year panel (e.g., 2020 and 2021 5-year ACS) and check for FIPS code changes.
- [ ] Find a county where the MOE exceeds 50% of the estimate and explain why this happens.

### Project Code Map
- `src/census_api.py`: Census/ACS client (`fetch_variables`, `fetch_acs`)
- `scripts/fetch_census.py`: CLI fetch for Census/ACS
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Storing FIPS codes as integers, which drops leading zeros and breaks merges.
- Fetching estimates without the corresponding MOE columns, making it impossible to assess precision.
- Confusing the ACS universe: the denominator for "uninsured rate" may be the civilian non-institutionalized population, not total population. Mismatching numerator and denominator universes gives rates above 1.
- Treating 5-year estimates as point-in-time snapshots when they are actually 5-year averages.
- Merging ACS data with health data that uses a different FIPS vintage without a crosswalk.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a validated county-level dataset drawn from the American Community
Survey, with FIPS-coded geography, derived rates, and margins of error. This
dataset serves as the cross-sectional building block for health economics analyses
that follow — from disparities research to program evaluation.

**Key takeaways:**
- Use 5-year ACS for county-level coverage; 1-year ACS only for large geographies.
- Always store FIPS codes as zero-padded strings, never integers.
- Fetch MOE columns alongside estimates; flag imprecise estimates.
- Validate derived rates (must be in [0, 1]) and merge keys (1:1, no nulls).
- ACS variable codes change across years — verify table shells for each vintage.

Suggested readings:
- U.S. Census Bureau, *Understanding and Using ACS Data* (official handbook, covers MOE propagation)
- U.S. Census Bureau, [API User Guide](https://www.census.gov/data/developers/guidance.html)
- Spielman, Folch, & Nagle, "Patterns and Causes of Uncertainty in the ACS" (*Applied Geography*, 2014)
- Kind & Buckley, "Making Health-Related Variables More Useful with Census Data" (*Population Health Metrics*, 2018)
- pandas documentation: `.merge()` with `validate` parameter, string methods for FIPS handling
