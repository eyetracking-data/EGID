# Random Forest Selector Features

This document specifies the **16 observable context features** used by the domain-specific Random Forest selectors for gap-specific imputation method selection.

The shared feature implementation is:

[`src/gap_imputation_benchmark/benchmark/features.py`](../src/gap_imputation_benchmark/benchmark/features.py)

The feature extractor operates on a **masked recording** and predefined context windows surrounding a gap. It does **not** access hidden ground-truth values inside the masked interval, candidate reconstruction errors, oracle labels, or dataset/group identifiers.

This document describes the selector input format.

For the benchmark protocol and grouped evaluation, see

[methods.md](methods.md); for the released real-data pipeline, see

[algorithm_usage.md](algorithm_usage.md). The strict feature lists stored in

benchmark, evaluation, and artifact `metadata.json` files record the input format

used for each committed run.

> **Important:** The three selectors use the same **semantic set of 16 context features**, but the gap-duration feature has a **domain-specific unit and column name**. Internally, gap duration is represented canonically in milliseconds. Eye tracking retains milliseconds, weather converts the duration to hours, and traffic converts it to minutes before the feature vector is passed to the corresponding selector.

---

## 1. Shared Feature Extraction and Domain-Specific Selector Input Formats

Feature extraction is split into two layers:

1. **Shared canonical extraction**  

   `extract_basic_gap_features(...)` computes the common feature representation. At this stage, gap duration is represented as `realized_gap_duration_ms`.

2. **Domain-specific selector input format**

   Before training or inference, each domain exposes the duration feature in the physical unit used by its selector.

Conceptually, the processing path is: domain time series → canonical gap representation → shared 16-feature extraction → domain-specific duration conversion → Random Forest selector.

The domain-specific selector input formats are:

| Domain | Selector duration feature | Unit | Conversion from canonical milliseconds |
|---|---|---:|---|
| Eye tracking | `realized_gap_duration_ms` | ms | $d_i^{\mathrm{ET}} = d_i^{\mathrm{ms}}$ |
| Weather | `realized_gap_duration_hours` | h | $d_i^{\mathrm{W}} = \dfrac{d_i^{\mathrm{ms}}}{3{,}600{,}000}$ |
| Traffic | `realized_gap_duration_minutes` | min | $d_i^{\mathrm{T}} = \dfrac{d_i^{\mathrm{ms}}}{60{,}000}$ |

The remaining **15 features are shared without a domain-specific rename or unit conversion**.

The corresponding implementation references are consolidated in [Section 7](#7-implementation-reference).

This distinction is important when comparing domains: the **feature semantics**

and extraction logic are **shared**, while the physically meaningful

representation of gap duration is adapted to the domain. The duration column

name must therefore match the selector artifact exactly; artifact loading

rejects an incompatible feature order before inference.

---

<details>
<summary><strong>Technical notation</strong> — optional definitions used by the exact formulas below.</summary>

<br>

Let

$g_i=[s_i,e_i)$

denote a contiguous missing interval with start index $s_i$, exclusive end index $e_i$, and length

$n_i=e_i-s_i.$

The requested context length on each side is

$n_{\mathrm{context},i}=\max(n_i,2).$

For the observable context of gap $g_i$, let

- $L_i$ denote the valid and finite observations in the predefined **left context**,

- $R_i$ denote the valid and finite observations in the predefined **right context**,

- $C_i=L_i\cup R_i$ denote the combined observable context,

- $b_i^{L}$ denote the valid observation immediately before the gap,

- $b_i^{R}$ denote the valid observation immediately after the gap,

- $\mu_i^{L}$ and $\mu_i^{R}$ denote the means of the valid left and right context values,

- $\mathrm{IQR}(C_i)$ denote the interquartile range of the combined context,

- $d_i^{\mathrm{ms}}$ denote the canonical realized gap duration in milliseconds.

The feature-normalization scale is

$\sigma_i^{\mathrm{feature}} = \max\left( \mathrm{IQR}(C_i), s_{\mathrm{floor}} \right).$

In the evaluation described in the paper, $s_{\mathrm{floor}}$ is estimated **exclusively from the corresponding training partition** as the first percentile of positive context-IQR values and is then kept fixed for the held-out dataset, station, or district.

No additional z-score standardization is applied before Random Forest fitting.

</details>

---

## 2. The 16 Selector Features at a Glance

The notation $d_i^{(d)}$ below denotes the **domain-specific selector representation** of gap duration defined in Section 1.

| # | Selector feature | Category | Mathematical definition | Interpretation |
|---:|---|---|---|---|
| 1 | Domain-specific duration feature | Gap geometry | $d_i^{(d)}$ | Realized physical gap duration in the selector's domain-specific unit: ms for eye tracking, h for weather, min for traffic. |
| 2 | `left_context_valid_fraction` | Completeness | $\dfrac{n_{i,L}^{\mathrm{valid}}}{n_{i,L}^{\mathrm{requested}}}$ | Fraction of requested left-context samples that are valid and finite. |
| 3 | `right_context_valid_fraction` | Completeness | $\dfrac{n_{i,R}^{\mathrm{valid}}}{n_{i,R}^{\mathrm{requested}}}$ | Fraction of requested right-context samples that are valid and finite. |
| 4 | `normalized_boundary_jump` | Local level | $\dfrac{\left\lvert b_i^{R}-b_i^{L}\right\rvert}{\sigma_i^{\mathrm{feature}}}$ | Absolute jump between the observations directly adjacent to the gap. |
| 5 | `normalized_mean_difference_right_minus_left` | Local level | $\dfrac{\mu_i^{R}-\mu_i^{L}}{\sigma_i^{\mathrm{feature}}}$ | Signed change in local signal level from the left context to the right context. |
| 6 | `local_std_over_scale` | Variability | $\dfrac{\mathrm{std}(C_i)}{\sigma_i^{\mathrm{feature}}}$ | Standard deviation of the combined observable context relative to the feature scale. |
| 7 | `local_range_over_scale` | Variability | $\dfrac{\max(C_i)-\min(C_i)}{\sigma_i^{\mathrm{feature}}}$ | Range of the combined observable context relative to the feature scale. |
| 8 | `normalized_trend_before` | Linear structure | $\dfrac{\beta_i^{L}}{\sigma_i^{\mathrm{feature}}}$ | OLS slope of the left context, normalized by the feature scale. |
| 9 | `normalized_trend_after` | Linear structure | $\dfrac{\beta_i^{R}}{\sigma_i^{\mathrm{feature}}}$ | OLS slope of the right context, normalized by the feature scale. |
| 10 | `normalized_trend_difference` | Linear structure | $\dfrac{\beta_i^{R}-\beta_i^{L}}{\sigma_i^{\mathrm{feature}}}$ | Difference between the right- and left-context slopes. |
| 11 | `trend_before_r2` | Linear structure | $R_{i,L}^{2}$ | Coefficient of determination of the left-context linear fit. |
| 12 | `trend_after_r2` | Linear structure | $R_{i,R}^{2}$ | Coefficient of determination of the right-context linear fit. |
| 13 | `normalized_median_absolute_velocity_left` | Velocity | $\dfrac{\mathrm{median}(V_i^{L})}{\sigma_i^{\mathrm{feature}}}$ | Median absolute velocity in the left context. |
| 14 | `normalized_median_absolute_velocity_right` | Velocity | $\dfrac{\mathrm{median}(V_i^{R})}{\sigma_i^{\mathrm{feature}}}$ | Median absolute velocity in the right context. |
| 15 | `normalized_p90_absolute_velocity_left` | Velocity | $\dfrac{Q_{0.90}(V_i^{L})}{\sigma_i^{\mathrm{feature}}}$ | 90th percentile of absolute velocity in the left context. |
| 16 | `normalized_p90_absolute_velocity_right` | Velocity | $\dfrac{Q_{0.90}(V_i^{R})}{\sigma_i^{\mathrm{feature}}}$ | 90th percentile of absolute velocity in the right context. |

The exact implementation names of feature 1 are:

| Domain | Feature 1 implementation name |
|---|---|
| Eye tracking | `realized_gap_duration_ms` |
| Weather | `realized_gap_duration_hours` |
| Traffic | `realized_gap_duration_minutes` |

The other 15 implementation names are identical across all three domain-specific selector input formats.

---

## 3. Exact Calculation Details

### 3.1 Observable context

Only samples that are both **marked valid** and **finite** are used in the feature calculations.

The shared extractor operates on a canonical internal representation containing:

- `gaze_x` — the numeric signal value used by the shared benchmark feature code,

- `is_valid` — validity indicator,

- `timestamp_ms` — timestamp represented in milliseconds.

These names form an **internal benchmark interface** and should not be interpreted as restricting the shared feature definitions to eye-tracking data. Domain-specific loaders/workflows map weather and traffic data into the representation expected by the shared feature extractor.

The paper's evaluated configurations require at least 80% of the requested context to be valid and finite on each side of the gap.

For temporal features, each context side must contain at least two valid and finite observations. Their timestamps must be finite and unique; after chronological ordering, they must be strictly increasing. Otherwise, feature extraction rejects the gap.

---

### 3.2 Gap boundaries

The two boundary observations are

$b_i^{L}=x_{s_i-1}, \qquad b_i^{R}=x_{e_i},$

because the missing interval is represented as

$g_i=[s_i,e_i).$

Both boundary observations must be valid and finite.

The normalized boundary jump is

$x_{i,\mathrm{boundary}} = \frac{ \left\lvert b_i^{R}-b_i^{L}\right\rvert }{ \sigma_i^{\mathrm{feature}} }.$

This corresponds to `normalized_boundary_jump`.

---

### 3.3 Context completeness

Let $n_{i,L}^{\mathrm{requested}}$ and $n_{i,R}^{\mathrm{requested}}$ denote the requested numbers of samples in the left and right context windows, and let $n_{i,L}^{\mathrm{valid}}$ and $n_{i,R}^{\mathrm{valid}}$ denote the corresponding numbers of valid and finite samples.

The completeness features are

$x_{i,\mathrm{valid},L} = \frac{ n_{i,L}^{\mathrm{valid}} }{ n_{i,L}^{\mathrm{requested}} },$

and

$x_{i,\mathrm{valid},R} = \frac{ n_{i,R}^{\mathrm{valid}} }{ n_{i,R}^{\mathrm{requested}} }.$

They are stored as `left_context_valid_fraction` and `right_context_valid_fraction`.

---

### 3.4 Context means

The valid context means are

$\mu_i^{L} = \frac{1}{\lvert L_i\rvert} \sum_{x\in L_i}x, \qquad \mu_i^{R} = \frac{1}{\lvert R_i\rvert} \sum_{x\in R_i}x.$

Their normalized signed difference is

$x_{i,\mathrm{mean}} = \frac{ \mu_i^{R}-\mu_i^{L} }{ \sigma_i^{\mathrm{feature}} }.$

The sign is retained. Positive values indicate a higher local level after the gap; negative values indicate a lower local level.

This corresponds to `normalized_mean_difference_right_minus_left`.

---

### 3.5 Local variability and normalization scale

The combined observable context is

$C_i=L_i\cup R_i.$

Its interquartile range is

$\mathrm{IQR}(C_i) = Q_{0.75}(C_i)-Q_{0.25}(C_i).$

The scale used by all amplitude-dependent features is

$\sigma_i^{\mathrm{feature}} = \max\left( \mathrm{IQR}(C_i), s_{\mathrm{floor}} \right).$

This feature scale is distinct from the nRMSE normalization scale defined in [methods.md](methods.md#4-error-metric-and-oracle).

The local standard-deviation feature is

$x_{i,\mathrm{std}} = \frac{ \mathrm{std}(C_i) }{ \sigma_i^{\mathrm{feature}} }.$

The implementation uses NumPy's population standard deviation, i.e. `np.std(...)` with the default `ddof=0`.

The local-range feature is

$x_{i,\mathrm{range}} = \frac{ \max(C_i)-\min(C_i) }{ \sigma_i^{\mathrm{feature}} }.$

These correspond to `local_std_over_scale` and `local_range_over_scale`.

---

### 3.6 Local linear structure

Trend features are calculated separately for the valid left and right context observations.

The implementation converts timestamps from milliseconds to seconds:

$t_k^{(\mathrm{s})} = \frac{ t_k^{(\mathrm{ms})} }{ 1000 }.$

Within each side, observations are ordered by timestamp. An ordinary least-squares linear model with intercept is fitted to centered timestamps:

$x_k = \alpha + \beta\left(t_k-\bar{t}\right) + \varepsilon_k.$

This yields the left and right slopes

$\beta_i^{L} \qquad\text{and}\qquad \beta_i^{R}.$

The three normalized trend features are

$x_{i,\mathrm{trend},L} = \frac{ \beta_i^{L} }{ \sigma_i^{\mathrm{feature}} },$

$x_{i,\mathrm{trend},R} = \frac{ \beta_i^{R} }{ \sigma_i^{\mathrm{feature}} },$

and

$x_{i,\mathrm{trend},\Delta} = \frac{ \beta_i^{R}-\beta_i^{L} }{ \sigma_i^{\mathrm{feature}} }.$

These correspond to:

- `normalized_trend_before`

- `normalized_trend_after`

- `normalized_trend_difference`

Because the fit uses time in seconds, the unnormalized slopes have units of signal-units per second.

---

### 3.7 Trend coefficient of determination

For each context side, the coefficient of determination is computed as

$R^2 = 1- \frac{ \sum_k\left(x_k-\hat{x}_k\right)^2 }{ \sum_k\left(x_k-\bar{x}\right)^2 }.$

The two selector features are

$R_{i,L}^{2} \qquad\text{and}\qquad R_{i,R}^{2},$

stored as `trend_before_r2` and `trend_after_r2`.

The implementation explicitly handles constant contexts:

- if the total sum of squares is zero and the residual sum of squares is numerically zero, $R^2=1$;

- if the total sum of squares is zero but the residual sum of squares is non-zero, $R^2=0$.

The $R^2$ features are not amplitude-normalized.

---

### 3.8 Absolute velocity

For consecutive valid observations within one context side, the implementation calculates absolute velocity as

$v_k = \left\lvert \frac{ x_{k+1}-x_k }{ t_{k+1}-t_k } \right\rvert.$

Let

$V_i^{L} = \{v_k : k\text{ indexes consecutive valid observations in }L_i\},$

and analogously define $V_i^{R}$ for the right context.

For each side, the median and 90th percentile are calculated:

$\widetilde{v}_i^{L} = \mathrm{median}(V_i^{L}), \qquad \widetilde{v}_i^{R} = \mathrm{median}(V_i^{R}),$

and

$v_{i,0.90}^{L} = Q_{0.90}(V_i^{L}), \qquad v_{i,0.90}^{R} = Q_{0.90}(V_i^{R}).$

The four normalized selector features are

$x_{i,\mathrm{velmed},L} = \frac{ \widetilde{v}_i^{L} }{ \sigma_i^{\mathrm{feature}} },$

$x_{i,\mathrm{velmed},R} = \frac{ \widetilde{v}_i^{R} }{ \sigma_i^{\mathrm{feature}} },$

$x_{i,\mathrm{vel90},L} = \frac{ v_{i,0.90}^{L} }{ \sigma_i^{\mathrm{feature}} },$

and

$x_{i,\mathrm{vel90},R} = \frac{ v_{i,0.90}^{R} }{ \sigma_i^{\mathrm{feature}} }.$

These correspond to:

- `normalized_median_absolute_velocity_left`

- `normalized_median_absolute_velocity_right`

- `normalized_p90_absolute_velocity_left`

- `normalized_p90_absolute_velocity_right`

Because timestamps are converted to seconds, the unnormalized velocity values have units of signal-units per second.

---

## 4. Leakage-Safe Feature Extraction

Selector inputs use only information observable for a real missing interval: gap geometry, the masked recording, valid finite context and boundary observations, their timestamps, and the supplied feature-scale floor.

They never use concealed gap values, reconstruction errors, oracle labels, selector targets, or dataset/group identifiers. Concealed values of artificial gaps are retained solely to evaluate reconstruction quality.

---

## 5. Raw Numerators Retained for Fold-Specific Normalization

`selector_feature_values(...)` additionally stores the raw numerators underlying the amplitude-normalized features.

These values are **auxiliary reproducibility quantities**, not additional members of the 16-dimensional selector feature vector.

| Normalized feature | Retained raw numerator |
|---|---|
| `normalized_boundary_jump` | `raw_absolute_boundary_difference` |
| `normalized_mean_difference_right_minus_left` | `raw_mean_difference_right_minus_left` |
| `local_std_over_scale` | `raw_local_std` |
| `local_range_over_scale` | `raw_local_range` |
| `normalized_trend_before` | `raw_trend_before` |
| `normalized_trend_after` | `raw_trend_after` |
| `normalized_trend_difference` | `raw_trend_difference` |
| `normalized_median_absolute_velocity_left` | `raw_median_absolute_velocity_left` |
| `normalized_median_absolute_velocity_right` | `raw_median_absolute_velocity_right` |
| `normalized_p90_absolute_velocity_left` | `raw_p90_absolute_velocity_left` |
| `normalized_p90_absolute_velocity_right` | `raw_p90_absolute_velocity_right` |

The implementation also retains diagnostic quantities such as local IQR, local standard deviation, local range, left and right context means, numbers of valid context samples, gap length in samples, sampling rate, unnormalized slopes, and unnormalized velocity summaries.

---

## 6. Relationship to Random Forest Training and Runtime Selection

The domain-specific Random Forest uses the 16 selector features to predict the expected nRMSE of each available imputation method for a gap. It selects the method with the lowest predicted nRMSE.

During training, the model learns this relationship from artificially masked gaps, for which the actual reconstruction errors of all applicable methods are known. At runtime, it uses only observable gap context; it does not predict missing signal values directly.

The duration feature is converted from the canonical millisecond representation to the unit expected by the domain-specific selector before inference.

---

## 7. Implementation Reference

### Shared feature extraction

- [`src/gap_imputation_benchmark/benchmark/features.py`](../src/gap_imputation_benchmark/benchmark/features.py)

  - `extract_basic_gap_features(...)`

  - `_temporal_side_diagnostics(...)`

  - `selector_feature_values(...)`

  - `RAW_FEATURE_NUMERATOR_COLUMNS`

### Domain-specific selector input formats

- [`src/gap_imputation_benchmark/domains/eyetracking/adapter.py`](../src/gap_imputation_benchmark/domains/eyetracking/adapter.py)

  - `EYE_TRACKING_DOMAIN`

- [`src/gap_imputation_benchmark/domains/weather/adapter.py`](../src/gap_imputation_benchmark/domains/weather/adapter.py)

  - `WEATHER_FEATURE_COLUMNS`

- [`src/gap_imputation_benchmark/domains/traffic/adapter.py`](../src/gap_imputation_benchmark/domains/traffic/adapter.py)

  - `TRAFFIC_FEATURE_COLUMNS`

### Benchmark and training conversion

- [`src/gap_imputation_benchmark/domains/weather/workflow.py`](../src/gap_imputation_benchmark/domains/weather/workflow.py)

  - `weather_feature_row(...)`

- [`src/gap_imputation_benchmark/domains/traffic/workflow.py`](../src/gap_imputation_benchmark/domains/traffic/workflow.py)

  - `traffic_feature_row(...)`

### Runtime conversion and selection

- [`src/gap_imputation_benchmark/algorithm/missing_values.py`](../src/gap_imputation_benchmark/algorithm/missing_values.py)

The documentation and implementation should be kept synchronized. The implementation is the authoritative executable specification of the feature values used by the benchmark and runtime selector.
