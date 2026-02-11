# Guide: Building a Census County Panel Dataset

## Table of Contents
- [Introduction](#intro)
- [Key Terms](#key-terms)
- [Step-by-Step Checklist](#step-by-step)
- [Alternative Example](#alternative-example)
- [Technical Explanations](#technical)
- [Diagnostics Checklist](#diagnostics)
- [Common Mistakes](#common-mistakes)
- [Summary + Suggested Readings](#summary)

---

<a id="intro"></a>
## Introduction

This guide accompanies the notebook `notebooks/06_causal/00_build_census_county_panel.ipynb`.

### What is panel data?

Panel data (also called longitudinal data) tracks the same entities across multiple time periods. Instead of a single snapshot (cross-section) or a single aggregate time series, panel data gives you a matrix: **entities x time**. A county-year panel, for example, records demographic and economic variables for each U.S. county in each year.

Panel structure is what makes most causal inference methods in applied economics possible. Fixed effects, difference-in-differences, and event studies all rely on observing the same unit before and after some change. Without a well-constructed panel, these methods cannot be applied.

For a full treatment of causal inference concepts (potential outcomes, identification, selection bias, and the designs that use panel data), see `docs/guides/06_causal/01_panel_fixed_effects_clustered_se.md`. This guide focuses on the upstream task: **building the panel dataset itself**.

### Why Census/ACS data matters for health economics research

The American Community Survey (ACS) is the primary source of county-level socioeconomic data in the United States between decennial censuses. For health economics, ACS variables serve as:

- **Outcome proxies**: poverty rates, insurance coverage, and employment status are direct measures of economic well-being that correlate with health outcomes.
- **Control variables**: demographics, income distributions, and education levels are standard confounders in health policy studies.
- **Treatment context**: Medicaid expansion studies, for instance, use county-level uninsurance rates and poverty rates from the ACS to characterize treatment intensity.

Building a clean, validated panel from ACS data is the first step in any county-level causal analysis. If the panel is malformed (duplicate rows, inconsistent identifiers, mixed estimate types), every downstream regression inherits those problems.

---

<a id="key-terms"></a>
## Key Terms

- **Panel data**: A dataset where the same entities (counties, hospitals, patients) are observed at multiple time points. Also called longitudinal data.

- **Cross-section**: A single observation per entity at one point in time. A single year of ACS data is a cross-section.

- **Balanced panel**: Every entity appears in every time period. If you have 100 counties and 9 years, a balanced panel has exactly 900 rows.

- **Unbalanced panel**: Some entities are missing in some time periods. This can happen when counties are created, merged, or when ACS suppresses data for small populations.

- **FIPS code**: Federal Information Processing Standards code. For counties, this is a 5-digit string: 2-digit state code + 3-digit county code (e.g., "06001" = Alameda County, California). FIPS codes are the standard entity identifier for U.S. county panels.

- **American Community Survey (ACS)**: An ongoing survey administered by the U.S. Census Bureau. It replaced the decennial census long form and provides annual estimates of demographic, social, economic, and housing characteristics.

- **Unit of observation**: The entity-time combination that defines a single row. In this project, the unit of observation is a county-year pair.

- **Panel structure (entity x time)**: The organizing principle of panel data. Each row is uniquely identified by a (entity, time) pair. In pandas, this corresponds to a MultiIndex of `(fips, year)`.

---

<a id="step-by-step"></a>
## Step-by-Step Checklist

These are the tasks you will implement in the notebook. Work through them in order.

### 1. Load the panel configuration
- [ ] Read `configs/census_panel.yaml` to get years, ACS variables, dataset type, and geography settings.
- [ ] Inspect the variable list and write down what each ACS code measures (e.g., `B01003_001E` = total population).

### 2. Fetch or load ACS tables for each year
- [ ] For each year, check if a cached raw CSV exists under `data/raw/census/`.
- [ ] If cached files are missing, fall back to `data/sample/census_county_panel_sample.csv`.
- [ ] (Optional) If you have a Census API key, fetch live data using `census_api.fetch_acs()`.

### 3. Build FIPS identifiers
- [ ] Zero-pad the `state` column to 2 digits and the `county` column to 3 digits.
- [ ] Concatenate them to create a 5-digit `fips` column.
- [ ] Verify that all FIPS codes are exactly 5 characters long.

### 4. Compute derived variables
- [ ] Calculate `unemployment_rate` = unemployed / in labor force (with safe division to handle zeros).
- [ ] Calculate `poverty_rate` = poverty count / total population (with safe division).
- [ ] Verify that rates fall in a reasonable range (roughly 0 to 1).

### 5. Validate the panel structure
- [ ] Set a `(fips, year)` MultiIndex and sort.
- [ ] Check for duplicate (fips, year) pairs.
- [ ] Check whether the panel is balanced (every county appears in every year).
- [ ] Inspect missing values across columns.

### 6. Save the processed panel
- [ ] Write the panel to `data/processed/census_county_panel.csv`.
- [ ] Reload and verify the file is non-empty and the index survived the round-trip.

---

<a id="alternative-example"></a>
## Alternative Example (Not the Notebook Solution)

This example demonstrates how you might fetch and merge two county-level ACS variables from different tables into a single panel. It uses fabricated data to illustrate the transformations; it does not call the real Census API.

```python
# Simulated ACS fetch: county-level education + insurance variables
import pandas as pd
import numpy as np

np.random.seed(42)

# Suppose we fetched education data (bachelor's degree attainment)
# from table B15003 for 3 counties over 3 years.
education_records = []
for year in [2018, 2019, 2020]:
    for state, county, name in [("06", "001", "Alameda"),
                                 ("12", "086", "Miami-Dade"),
                                 ("36", "061", "New York")]:
        education_records.append({
            "state": state,
            "county": county,
            "year": year,
            "NAME": name,
            "B15003_022E": np.random.randint(50000, 200000),   # bachelor's degree
            "B15003_001E": np.random.randint(400000, 1200000), # total 25+
        })

edu_df = pd.DataFrame(education_records)

# Suppose we also fetched insurance data from table B27001.
insurance_records = []
for year in [2018, 2019, 2020]:
    for state, county in [("06", "001"), ("12", "086"), ("36", "061")]:
        insurance_records.append({
            "state": state,
            "county": county,
            "year": year,
            "B27001_001E": np.random.randint(400000, 1200000), # total civilian
            "B27001_005E": np.random.randint(10000, 80000),    # uninsured
        })

ins_df = pd.DataFrame(insurance_records)

# Merge on (state, county, year)
merged = edu_df.merge(ins_df, on=["state", "county", "year"], how="outer")

# Build FIPS
merged["fips"] = merged["state"].str.zfill(2) + merged["county"].str.zfill(3)

# Compute rates
merged["pct_bachelors"] = merged["B15003_022E"] / merged["B15003_001E"]
merged["uninsured_rate"] = merged["B27001_005E"] / merged["B27001_001E"]

# Set panel index
panel = merged.set_index(["fips", "year"]).sort_index()
print(panel[["NAME", "pct_bachelors", "uninsured_rate"]])
```

This pattern -- fetch separate tables, merge on geography + time, build identifiers, compute rates, set panel index -- is the same workflow the notebook uses with real ACS data.

---

<a id="technical"></a>
## Technical Explanations

### 1. What makes data "panel-shaped"

Panel data has a specific structure: every row is uniquely identified by an **(entity, time)** pair. The entity dimension provides cross-sectional variation (differences across units), and the time dimension provides longitudinal variation (changes within units).

In health economics, common panel structures include:

| Entity | Time | Example research question |
|--------|------|--------------------------|
| Counties | Years | Does Medicaid expansion reduce county-level uninsurance? |
| Patients | Visits | Does a new drug protocol reduce readmission rates? |
| Hospitals | Quarters | Do pay-for-performance incentives improve quality metrics? |
| States | Months | Does a smoking ban reduce ER visits for respiratory illness? |

The county-year panel in this project has:
- **Entity**: U.S. counties, identified by 5-digit FIPS codes.
- **Time**: Calendar years (2014--2022 in the default configuration).
- **Variables**: Population, income, rent, home values, labor force counts, poverty counts.

In pandas, panel structure is represented as a MultiIndex:

```python
panel = df.set_index(["fips", "year"]).sort_index()
# panel.index is a MultiIndex with levels (fips, year)
# panel.loc["06001", 2018] returns one row: Alameda County in 2018
```

The helper `to_panel_index()` in `src/causal.py` standardizes this: it ensures the entity column is a string, attempts to cast the time column to int, and sets/sorts the MultiIndex.

### 2. The American Community Survey (ACS)

The ACS is administered continuously by the U.S. Census Bureau. Each year, approximately 3.5 million households are surveyed. The results are published as estimates (not raw counts) with accompanying margins of error.

**1-year vs. 5-year estimates**

| Feature | 1-year estimates | 5-year estimates |
|---------|-----------------|-----------------|
| Sample size | Single year of responses | Five years pooled |
| Geographic coverage | Only areas with 65,000+ population | All areas (down to block groups) |
| Precision | Higher variance | Lower variance (larger sample) |
| Currency | Most current | Lagged (centered on midpoint of 5-year window) |
| Use case | Large counties/metros | Small counties, rural areas |

This project uses **5-year estimates** (`acs/acs5`) because it needs data for all U.S. counties, including those with populations below 65,000.

**Key variable tables for health economics**

| ACS Table | What it contains |
|-----------|-----------------|
| B01003 | Total population |
| B19013 | Median household income |
| B17001 | Poverty status (counts below poverty line) |
| B23025 | Employment status (labor force, employed, unemployed) |
| B25064 | Median gross rent |
| B25077 | Median home value |
| B27001 | Health insurance coverage |
| B15003 | Educational attainment |
| B02001 | Race |
| B03003 | Hispanic or Latino origin |

Variable names follow a convention: the table code, an underscore, a sequence number, and a suffix. The suffix `E` means "estimate" and `M` means "margin of error." For example, `B19013_001E` is the estimate of median household income, and `B19013_001M` is its margin of error.

**Geographic levels**

The ACS provides data at many geographic levels: nation, state, county, tract, block group, metropolitan statistical area, congressional district, and more. This project queries at the county level using:
- `for_geo = "county:*"` (all counties)
- `in_geo = "state:*"` (within all states)

The `fetch_acs()` function in `src/census_api.py` constructs the API request. The Census API returns rows with `state` and `county` columns as separate fields, which you then combine into FIPS codes.

### 3. FIPS codes: structure and pitfalls

A county FIPS code is a 5-digit string composed of:
- **State FIPS** (2 digits): e.g., 06 = California, 36 = New York, 48 = Texas.
- **County FIPS** (3 digits): e.g., 001, 013, 059.

The `make_fips()` helper in `src/causal.py` constructs this:

```python
def make_fips(state: str | int, county: str | int) -> str:
    return str(state).zfill(2) + str(county).zfill(3)
```

**Common pitfalls with FIPS codes**

1. **Leading zeros**: States like Alabama (01), Connecticut (09), and others have FIPS codes that start with zero. If you read a CSV without specifying `dtype=str`, pandas will interpret "01001" as the integer 1001. Always either read FIPS as strings or zero-pad after reading:
   ```python
   df["state"] = df["state"].astype(str).str.zfill(2)
   df["county"] = df["county"].astype(str).str.zfill(3)
   ```

2. **Connecticut planning regions**: In 2022, Connecticut replaced its 8 counties with 9 planning regions for statistical purposes. The Census Bureau updated FIPS codes accordingly. If your panel spans pre- and post-2022 data, Connecticut county codes may not match across years. For a simple solution, either drop Connecticut or create a crosswalk mapping old county FIPS to new planning region FIPS.

3. **Alaska borough changes**: Alaska periodically reorganizes its boroughs and census areas. The Valdez-Cordova Census Area (02261) was split into the Chugach Census Area (02063) and the Copper River Census Area (02066) effective 2019. Similar issues arise when counties merge or split.

4. **Shannon County, South Dakota**: Renamed to Oglala Lakota County in 2015, changing its FIPS from 46113 to 46102. Both old and new codes may appear in different vintages of ACS data.

5. **Independent cities**: Virginia has independent cities that are separate FIPS entities from their surrounding counties. This is unique to Virginia and can cause confusion if you expect all FIPS entities to be "counties."

### 4. Panel data validation

After constructing the panel, you must verify its structure before using it in any analysis.

**Check 1: No duplicate (entity, time) pairs**

Each (fips, year) combination should appear exactly once. Duplicates mean a merge went wrong or a year was double-counted.

```python
dupes = panel.index.duplicated(keep=False)
assert not dupes.any(), f"Found {dupes.sum()} duplicate (fips, year) rows"
```

**Check 2: Balanced vs. unbalanced**

In a balanced panel, every entity appears in every time period.

```python
n_entities = panel.index.get_level_values("fips").nunique()
n_years = panel.index.get_level_values("year").nunique()
expected_rows = n_entities * n_years
is_balanced = len(panel) == expected_rows
print(f"Entities: {n_entities}, Years: {n_years}, "
      f"Expected: {expected_rows}, Actual: {len(panel)}, "
      f"Balanced: {is_balanced}")
```

If the panel is unbalanced, investigate why. Common reasons:
- A county was created or dissolved during the panel period.
- ACS suppressed estimates for a county in some years (small population).
- A merge dropped rows due to mismatched identifiers.

**Check 3: Missing values within existing rows**

Even in a balanced panel, individual cells may be missing (NaN). This is different from missing rows. Suppressed ACS cells are often returned as negative values (e.g., -666666666) or null.

```python
print(panel.isnull().sum().sort_values(ascending=False))
```

**Check 4: Consistent entity identifiers across time**

Verify that the set of FIPS codes does not change unexpectedly across years:

```python
fips_by_year = panel.groupby("year")["fips"].apply(set)
common_fips = set.intersection(*fips_by_year)
print(f"Counties in all years: {len(common_fips)}")
print(f"Total unique counties: {panel['fips'].nunique()}")
```

### 5. Variable selection for health economics

The ACS provides a large number of variables. For health economics research at the county level, the most commonly used categories are:

**Economic variables**
- *Median household income* (`B19013_001E`): Proxy for economic well-being. Often used as a control or to define income quartiles.
- *Poverty counts* (`B17001_002E` / `B01003_001E`): The poverty rate is a standard measure of economic deprivation and a key predictor of health outcomes.
- *Employment status* (`B23025_002E` labor force, `B23025_004E` employed, `B23025_005E` unemployed): Unemployment rate captures labor market conditions that affect health insurance access and stress-related health outcomes.

**Demographic variables**
- *Total population* (`B01003_001E`): Needed as a denominator for rates and as a control for scale.
- *Age distribution* (table `B01001`): Age structure matters because health care utilization is heavily age-dependent.
- *Race and ethnicity* (tables `B02001`, `B03003`): Health disparities research requires these breakdowns.

**Housing variables**
- *Median gross rent* (`B25064_001E`): Housing cost burden is linked to health outcomes through stress, displacement, and neighborhood quality.
- *Median home value* (`B25077_001E`): Proxy for local wealth and neighborhood socioeconomic status.

**Health-related variables**
- *Insurance coverage* (table `B27001`): Available at the county level in 5-year estimates. Critical for studies of coverage expansions (ACA, Medicaid).
- *Disability status* (table `B18101`): Prevalence of disability by age and sex.

**Education variables**
- *Educational attainment* (table `B15003`): Education is a strong predictor of health behaviors and outcomes. Often used as a control or stratification variable.

The project's default configuration (`configs/census_panel.yaml`) includes population, income, rent, home value, poverty, and employment variables. You can expand this by adding table codes to the `get` list in the YAML file.

### 6. Worked example: building a small panel from scratch

This walkthrough shows the data transformations step by step, using a tiny dataset. The goal is to make the logic transparent.

**Step 1: Raw ACS response (one year)**

Suppose the Census API returns this for 2018:

| NAME | B01003_001E | B17001_002E | state | county |
|------|------------|------------|-------|--------|
| Alameda County, CA | 1666753 | 178619 | 6 | 1 |
| Miami-Dade County, FL | 2761581 | 430175 | 12 | 86 |
| New York County, NY | 1632480 | 267231 | 36 | 61 |

**Step 2: Zero-pad and build FIPS**

| fips | year | NAME | B01003_001E | B17001_002E |
|------|------|------|------------|------------|
| 06001 | 2018 | Alameda County, CA | 1666753 | 178619 |
| 12086 | 2018 | Miami-Dade County, FL | 2761581 | 430175 |
| 36061 | 2018 | New York County, NY | 1632480 | 267231 |

Notice that state "6" became "06" and county "1" became "001". Without zero-padding, "6" + "1" = "61", which would collide with New York County (state 36, county 061) in string concatenation.

**Step 3: Repeat for 2019 and 2020, then stack**

After fetching and processing all three years, concatenate the DataFrames:

```python
panel = pd.concat([df_2018, df_2019, df_2020], ignore_index=True)
```

The result has 9 rows: 3 counties x 3 years.

**Step 4: Compute derived rates**

```python
panel["poverty_rate"] = panel["B17001_002E"] / panel["B01003_001E"]
```

Use safe division to handle any county where the denominator might be zero or missing:

```python
def safe_div(num, den):
    den = den.replace({0: pd.NA})
    return (num / den).astype(float)
```

**Step 5: Set the panel index**

```python
panel = panel.set_index(["fips", "year"]).sort_index()
```

The resulting MultiIndex ensures that `panel.loc["06001", 2019]` returns exactly one row.

**Step 6: Validate**

```python
assert not panel.index.duplicated().any()        # no duplicate (fips, year)
assert len(panel) == 3 * 3                        # 3 counties x 3 years
assert panel.index.get_level_values("fips").str.len().eq(5).all()  # all FIPS are 5 chars
```

### 7. Common data quality issues

**Suppressed cells**: For counties with very small populations, the Census Bureau may suppress estimates to protect respondent confidentiality. Suppressed values may appear as negative sentinel values (e.g., -666666666) or as null. Always check for these before computing rates:

```python
sentinel_mask = panel["B19013_001E"] < 0
print(f"Suppressed median income cells: {sentinel_mask.sum()}")
panel.loc[sentinel_mask, "B19013_001E"] = pd.NA
```

**Margin of error (MOE)**: Every ACS estimate has an associated MOE. For small counties, the MOE can be large relative to the estimate. If you request `B19013_001M` alongside `B19013_001E`, you can construct 90% confidence intervals: estimate +/- MOE. In health econ research, ignoring MOE for small-county estimates can lead to imprecise or misleading results. Consider weighting by population or dropping counties below a population threshold.

**Changing geographies over time**: As noted in the FIPS section, county boundaries and codes can change. When building a multi-year panel, always check whether the set of FIPS codes is stable across your time window. If not, you need a geographic crosswalk (the Census Bureau publishes these) or you must restrict to counties that exist in all years.

**Mixed vintages of 5-year estimates**: Each 5-year ACS dataset (e.g., 2015-2019) is a pooled estimate, not a single-year value. The 2018 5-year estimate covers 2014-2018, while the 2019 5-year estimate covers 2015-2019. These overlapping windows mean adjacent years share 4 out of 5 years of underlying data, introducing serial correlation. This is a known issue in panel analyses using 5-year ACS data. Some researchers use only non-overlapping windows (e.g., 2010-2014 and 2015-2019) to avoid this problem.

---

<a id="diagnostics"></a>
## Diagnostics Checklist

Run these checks after building your panel and before passing it to any estimation notebook.

### Panel shape verification
- [ ] Confirm the DataFrame has a `(fips, year)` MultiIndex.
- [ ] Verify the number of unique entities and time periods matches expectations.
- [ ] Check total row count = (expected entities) x (expected years) if aiming for a balanced panel.

### Duplicate check
- [ ] `panel.index.duplicated().sum()` should be 0.
- [ ] If duplicates exist, inspect the offending rows and trace back to the merge or concatenation step.

### Missing data patterns
- [ ] Print `panel.isnull().sum()` for all columns.
- [ ] Check for sentinel values (negative numbers in count/income fields).
- [ ] Examine whether missingness is concentrated in specific years (data availability) or specific counties (suppression).

### Cross-sectional vs. time variation
- [ ] For each key variable, compute the between-entity standard deviation (variation across counties in a given year) and the within-entity standard deviation (variation across years for a given county).
- [ ] Variables with zero within-entity variation cannot be identified by fixed effects.

```python
# Within and between variation
within = panel.groupby("fips")["poverty_rate"].std().mean()
between = panel.groupby("year")["poverty_rate"].std().mean()
print(f"Avg within-county std: {within:.4f}")
print(f"Avg between-county std (by year): {between:.4f}")
```

### Summary statistics by entity and time
- [ ] Compute means per year to check for time trends or suspicious jumps.
- [ ] Compute means per entity to check for outlier counties.
- [ ] Plot at least one variable over time for a few counties to visually inspect the panel.

---

<a id="common-mistakes"></a>
## Common Mistakes

1. **Losing leading zeros in FIPS codes.** Reading a CSV without `dtype={"state": str, "county": str}` converts "06" to 6 and "001" to 1. The resulting FIPS "61" is wrong. Always read geographic identifiers as strings or zero-pad immediately after loading.

2. **Ignoring the ACS margin of error.** ACS estimates for small counties can have margins of error that are 50% or more of the estimate itself. Using these point estimates without acknowledging uncertainty overstates the precision of your analysis. At minimum, report population-weighted results or restrict to counties above a population threshold.

3. **Not checking panel balance.** If your panel is unbalanced and you did not intend it to be, something went wrong in the data build. Unbalanced panels are not inherently a problem, but they require intentional handling (e.g., understanding whether missingness is related to the outcome).

4. **Mixing 1-year and 5-year ACS estimates.** The 1-year ACS covers only counties with 65,000+ population. The 5-year ACS covers all counties. Mixing them within a panel creates a sample that changes composition across years. Use one or the other consistently.

5. **Using wrong geographic identifiers.** Some datasets use Census tract FIPS (11 digits), some use county FIPS (5 digits), some use state FIPS (2 digits). Merging on the wrong level silently drops rows or creates duplicates. Always verify the geographic level before merging.

6. **Ignoring overlapping 5-year windows.** Adjacent 5-year ACS estimates share 4 years of underlying survey data. This induces mechanical serial correlation that is separate from any economic process. Clustering standard errors at the county level helps, but some researchers prefer non-overlapping windows.

7. **Not validating derived rates.** After computing `poverty_rate = poverty_count / total_population`, check that the values are between 0 and 1. Values outside this range indicate a data quality issue (mismatched numerator/denominator tables, sentinel values, or a merge error).

8. **Treating repeated cross-sections as a true panel.** The ACS surveys different households each year. A county-year panel is a panel of *places*, not *people*. The composition of residents changes over time due to migration and demographic change. This matters for interpretation: county fixed effects absorb time-invariant county characteristics, but they do not control for changing population composition within a county.

---

<a id="summary"></a>
## Summary

This guide covers the process of building a county-level panel dataset from American Community Survey data. The key steps are: (1) choose years and variables from the ACS, (2) fetch or cache the data, (3) construct stable FIPS identifiers with proper zero-padding, (4) compute derived rates with safe division, (5) validate the panel structure for duplicates, balance, and missing data, and (6) save the processed panel for use in downstream causal notebooks.

The panel you build here is the input for the fixed effects, difference-in-differences, and other causal inference notebooks in this module. Getting the data right at this stage prevents subtle errors that propagate through every subsequent analysis.

### Project Code Map

- `src/causal.py`: panel setup helpers (`to_panel_index`, `make_fips`)
- `scripts/build_datasets.py`: ACS panel builder (`build_census_county_panel`)
- `src/census_api.py`: Census/ACS client (`fetch_acs`, `fetch_variables`)
- `configs/census_panel.yaml`: panel configuration (years, variables, geography)
- `data/sample/census_county_panel_sample.csv`: offline sample panel dataset

### Suggested Readings

- **ACS documentation**: U.S. Census Bureau, *Understanding and Using American Community Survey Data: What All Data Users Need to Know* (census.gov). Covers survey methodology, margins of error, and proper interpretation of estimates.
- **Census API documentation**: developer.census.gov. Reference for variable names, geography codes, and API query syntax.
- **ACS variable search**: data.census.gov. Interactive tool for finding ACS table codes and understanding variable definitions.
- **FIPS code reference**: Census Bureau FIPS Codes page. Complete list of state and county FIPS codes with vintage information.
- **Wooldridge, J.M.**: *Econometric Analysis of Cross Section and Panel Data*, Chapter 10 (Basic Linear Unobserved Effects Panel Data Models). Covers the econometric foundations for the panel methods that will consume this dataset.
- **Angrist, J.D. and Pischke, J.-S.**: *Mostly Harmless Econometrics*, Chapter 5 (Panel Data Methods). Practical treatment of fixed effects and related designs.
- **Baum-Snow, N. and Ferreira, F.** (2015): "Causal Inference in Urban and Regional Economics" in *Handbook of Regional and Urban Economics*, Vol. 5. Discusses county-level panel data construction for applied research.
