# Benchmark and evaluation methods

This document specifies the intended benchmark and evaluation protocol.
For any specific published run, the corresponding strict domain configuration
and generated `metadata.json` are the authoritative records of the settings
actually used. If they differ from this document or from historical outputs,
use those run-specific records.

The benchmark evaluates univariate horizontal gaze position, hourly air
temperature, and five-minute traffic flow. Its aim is not to train one
cross-domain model; it evaluates the same gap-level selection architecture in
three domain-specific settings with separate candidate portfolios and selector
artifacts. It does not claim that a selector trained in one domain is valid for
another domain.

## How to use this specification

Read this document together with the frozen configuration under `configs/` and
the matching `metadata.json` files in `benchmarks/`, `results/`, and
`artifacts/`. The feature-level contract is specified in
[selector_features.md](selector_features.md), and external source-data
obligations are in `data/`.

## 1. Problem formulation

Let a univariate recording be $X=(x_1,\ldots,x_n)$, and let
$g_i=[s_i,e_i)$ be a contiguous interval of $n_i=e_i-s_i$ samples.
At runtime, the values inside a real gap are unknown. The selector therefore
uses only observable samples around the gap to choose a reconstruction method;
it never uses hidden values, oracle labels, reconstruction errors, or dataset,
participant, station, or district identifiers as model features.

For a domain $d$, let $M_d$ be its portfolio of applicable candidate
methods. Rather than directly predicting the missing signal values, a
domain-specific native multi-output Random Forest predicts one expected
normalized reconstruction error for each candidate method. The recommendation
is the method with the smallest predicted error:

$$
\hat m_i = \arg\min_{m \in M_d} \hat e_{i,m}.
$$

At deployment, candidates are attempted in ascending predicted-error order.
If the highest-ranked candidate is not applicable or cannot return the required
number of values, the next candidate is used. If none succeeds, the interval
remains missing and the reason is retained by the preprocessing provenance
layer.

## 2. Common artificial-gap protocol

Supervised targets are obtained exclusively from originally observed portions
of a recording. A candidate interval is temporarily masked, while its original
values are retained separately as offline ground truth. Every candidate method
receives the identical masked recording for the same artificial gap, so method
errors are paired at gap level.

Natural missing or invalid observations remain missing. They are neither
selected as artificial-gap targets nor used as ground truth. Eligible artificial
gaps must satisfy all of the following:

- all target values inside the gap are finite and originally observed;
- the immediately adjacent boundary observations are observed and finite;
- each side of the requested context contains at least 80% valid, finite
  observations;
- requested context length is $\max(n_i, 2)$ samples on each side;
- the selected context/gap windows do not overlap within the domain's sampling
  unit; and
- the domain-specific sampling procedure can draw an admissible location within
  its configured attempt limit.

Gap duration is sampled in five strata in every domain. Balancing prevents the
benchmark from being dominated by the much more common short gaps. Failed draws
and gaps for which not every candidate yields a finite score are retained in
the benchmark outputs as exclusions rather than silently discarded.

## 3. Candidate reconstruction methods

Seven reconstruction methods are shared by all domains. Each method operates only on the masked recording and the predefined local context of the respective gap. A method is considered applicable only if its required observations are available and valid; otherwise, it returns an explicit non-applicable outcome rather than silently falling back to another reconstruction method.

| Method                         | Reconstruction principle                                                                                                                                                                                                                                                                                   |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Forward fill**               | Assigns the last valid observation immediately before the gap to every missing sample.                                                                                                                                                                                                                     |
| **Nearest boundary**           | Assigns each missing sample the value of the temporally closer valid gap boundary. Distance is measured using timestamps when valid timestamps are available and otherwise by sample position; ties are assigned to the left boundary.                                                                     |
| **Linear interpolation**       | Connects the valid observations immediately before and after the gap by a straight line and evaluates this line at all missing positions.                                                                                                                                                                  |
| **PCHIP**                      | Applies piecewise cubic Hermite interpolation to the valid finite support points in the predefined left and right contexts. PCHIP provides a smooth, shape-preserving interpolation and limits the overshoot that can occur with unconstrained cubic interpolation.                                        |
| **Local natural cubic spline** | Fits a natural cubic spline to all valid finite support points in the predefined left and right contexts and evaluates the spline within the gap. The natural boundary condition sets the second derivative to zero at the outer ends of the fitted support interval.                                      |
| **Polyfit-BIC**                | Fits local polynomial models of degrees 1, 2, and 3 to the valid finite context points after centering and scaling the time coordinate. The degree with the smallest Bayesian information criterion (BIC) is selected and evaluated at the missing positions.                                              |
| **Template reconstruction**    | Combines a linear interpolation between the two gap boundaries with short-term variation extracted from fully observed neighboring segments of the same length as the gap. The local variation is detrended and smoothly tapered to zero at both gap boundaries before being added to the linear baseline. |

### Template reconstruction

First, a **linear baseline** is constructed between the valid observations immediately before and after the gap. Let the two observed boundary values be denoted by $x_{\mathrm{left}}$ and $x_{\mathrm{right}}$. For the $i$-th missing sample, the linear baseline is defined as

$$
b_i
=
x_{\mathrm{left}}
+
\frac{i}{L+1}
\left(
x_{\mathrm{right}}-x_{\mathrm{left}}
\right),
\qquad i=1,\ldots,L.
$$

This baseline represents the large-scale transition between the observed values immediately before and after the gap.


Second, the template is **detrended**. To express the relative position within the template, define

$$
\tau_i
=
\begin{cases}
\dfrac{i-1}{L-1}, & L>1,\\[6pt]
0, & L=1.
\end{cases}
$$

The linear trend connecting the first and last template values is then

$$
\ell_i^{(q)}
=
q_1
+
\tau_i
\left(
q_L-q_1
\right),
$$

and the residual template movement is

$$
r_i
=
q_i-\ell_i^{(q)}.
$$

Thus, \(r_i\) contains only the local variation of the neighboring template after removing its overall level change.

Finally, this residual movement is smoothly added to the linear baseline. The tapering weight is defined as

$$
w_i
=
\sin^2\!\left(
\pi \tau_i
\right),
$$

giving the final reconstruction

$$
\boxed{
\hat{x}_i
=
b_i
+
w_i r_i
}
\qquad
i=1,\ldots,L.
$$

The tapering weight approaches zero at both ends of the gap and is largest toward its center. Consequently, the reconstruction remains close to the boundary-constrained linear baseline near the gap edges, while the local variation obtained from the neighboring template primarily affects the interior of the missing interval. For \(L=1\), \(r_i=0\), so the method reduces naturally to the linear boundary baseline.


### Seasonal-periodic reconstruction

Weather and Traffic additionally include a seasonal-periodic method. Instead of using only the immediate local context, this method searches for complete reference segments at seasonally corresponding positions. Candidate references are considered in increasing seasonal distance from the target gap. At most three complete references are retained from the past and three from the future, with offsets of up to eight seasonal periods. A reference segment is accepted only if every corresponding sample exists and is valid and finite. At least three complete references in total are required; otherwise, the method is considered not applicable.

The accepted reference segments are aligned sample by sample with the target gap and aggregated pointwise. The offline benchmark uses references from both temporal directions. This allows future observations to contribute to reconstruction and therefore represents an offline setting; a causal online deployment would use the same procedure in a past-only configuration.

For **Weather**, seasonal correspondence is defined by calendar year. Each target timestamp is mapped to the same calendar date and hour in surrounding years. Up to three complete reference segments from each temporal direction are retained and aggregated pointwise by their arithmetic mean. Calendar-based year shifts avoid the one-day displacement that would arise from repeatedly applying a fixed 365-day offset across leap years. For February 29, a candidate year without the corresponding calendar date is skipped.

For **Traffic**, seasonal correspondence is defined by a fixed period of seven days. References therefore correspond to the same weekday and five-minute clock position in surrounding weeks. Up to three complete references from each temporal direction are retained and aggregated pointwise by their median, reducing the influence of individual atypical reference weeks.

The candidate registry defines the seven shared reconstruction methods and extends this common portfolio with the domain-specific seasonal-periodic configurations for Weather and Traffic. The corresponding implementation is provided in `src/gap_imputation_benchmark/imputers/registry.py`, `src/gap_imputation_benchmark/benchmark/imputers.py`, and the domain adapters.


## 4. Error metric and oracle

For artificial gap $i$ and method $m$, reconstruction quality is measured
as normalized root mean squared error:

$$
\operatorname{nRMSE}_{i,m} =
\frac{\operatorname{RMSE}_{i,m}}{\sigma_i},
\qquad
\sigma_i = \max\left(\operatorname{IQR}(C_i),
0.05\operatorname{IQR}(O_i)\right),
$$

where $C_i$ is the union of valid left and right context observations and
$O_i$ contains valid observations in the recording outside the artificial
gap. The second term avoids an unstable denominator when the local context is
nearly constant. Gaps with a non-finite or non-positive normalization scale are
excluded from method-score learning and evaluation.

The offline oracle selects the candidate with the lowest observed nRMSE for a
gap. It uses retained ground truth and is not deployable. It is reported only
as an upper reference for the possible benefit of gap-specific selection.

## 5. Selector features and training targets

Each eligible gap is represented by 16 features calculated from its observable
context. They cover gap duration, left/right context completeness, boundary and
mean differences, local variation, linear trends and fit quality, and absolute
velocity summaries. Amplitude-dependent features are normalized with a
fold-local scale floor learned as the first percentile of positive context-IQR
values in the corresponding training partition. No additional z-score feature
standardization is applied before Random Forest fitting. Undefined values are
median-imputed using statistics estimated on the training partition only.

The exact definitions, units, and domain-specific duration conversion are in
[selector_features.md](selector_features.md). Eye Tracking uses gap duration
in milliseconds, Weather in hours, and Traffic in minutes; the other 15 feature
semantics are shared.

The target vector is the method-specific nRMSE vector for a gap. Random Forest
hyperparameters are selected within the outer training data only. The final
selector artifact is fitted after evaluation on all eligible gaps in its domain;
it is a deployment artifact and not an additional unbiased performance result.

## 6. Domain protocols

### Eye Tracking

The Eye-Tracking benchmark uses GazeBase, GazeBaseVR, ZuCo 2.0, and Pedrotti
reading-task recordings. Native coordinate systems are retained: GazeBase and
GazeBaseVR use degrees, while ZuCo and Pedrotti use pixels. Gap-local robust
normalization permits method comparison without converting coordinates to a
common unit.

The protocol selects 10 participants per dataset. GazeBase, GazeBaseVR, and
ZuCo contribute two recordings per participant; Pedrotti contributes 40 trials
per participant. The configuration requests 400 gaps per dataset in five
duration strata from 0 to 250 ms, for 1,600 requested gaps across the domain
before exclusions. The portfolio contains the seven common methods.

The frozen settings are in `configs/eye_tracking_final.toml`; source selection,
task restrictions, and data details are in [data/eye_tracking.md](../data/eye_tracking.md).

### Weather

The Weather benchmark uses DWD historical hourly air temperature (`TT_TU`) for
16 stations, restricted to 2000-01-01 00:00 through 2025-12-31 23:00. Each
station-year is a sampling unit. Four non-overlapping gaps are requested from
each of five strata: 1-6, 7-24, 25-72, 73-168, and 169-440 hours. This yields
20 requested gaps per station-year and 8,320 requested gaps in total before
exclusions.

The portfolio contains the seven shared methods plus calendar-aligned seasonal
reconstruction. The frozen settings are in `configs/weather_final.toml`; input
preparation, station selection, and source details are in
[data/weather.md](../data/weather.md).

### Traffic

The Traffic benchmark uses five-minute LargeST traffic-flow data for PeMS
Districts 3, 4, 7, and 11 from 2017 through 2021. Before sampling, sensors are
retained only when valid coverage is at least 95% and no natural gap exceeds 24
hours; zero flow is valid. Each district-year contributes 100 deterministically
selected sensor-year units.

Every selected unit contributes one non-overlapping gap in each of five
duration strata: 1-3, 4-12, 13-36, 37-144, and 145-288 five-minute steps
(5-15, 20-60, 65-180, 185-720, and 725-1,440 minutes). The protocol therefore
requests 10,000 gaps before exclusions. The portfolio contains the seven
shared methods plus weekly seasonal reconstruction.

The frozen settings are in `configs/traffic_final.toml`; sensor auditing,
selected-panel construction, and source details are in
[data/traffic.md](../data/traffic.md).

## 7. Nested group-wise evaluation

Generalization is assessed with nested validation that withholds complete
groups, not individual gaps. The held-out unit is a dataset for Eye Tracking,
a station for Weather, and a district for Traffic.

For each outer held-out group:

1. all gaps from that group are reserved for outer testing;
2. feature-scale floors, median imputation statistics, and Random Forest
   hyperparameters are fit or selected using only the remaining groups;
3. the selected model predicts one nRMSE per method for every held-out gap;
4. the method with the smallest predicted nRMSE is scored against hidden
   ground truth; and
5. per-gap predictions, selected-method frequencies, scale floors,
   hyperparameter results, and group summaries are written to the result
   directory.

Reported domain performance is a macro-average: nRMSE is averaged within each
held-out group and then averaged across groups. Larger datasets, stations, or
districts therefore cannot dominate the domain result merely because they
contribute more gaps.

Three comparison levels are retained:

| Level | Definition | Deployment status |
| --- | --- | --- |
| Best fixed method | One candidate chosen by its domain-wide mean nRMSE. | Deployable baseline. |
| Adaptive selector | Lowest predicted nRMSE from the leakage-safe outer-fold selector. | Deployable policy. |
| Offline oracle | Lowest observed nRMSE using hidden target values. | Not deployable. |

## 8. Exclusions and output artifacts

The workflow does not treat missing scores as successful performance. Its
output contract distinguishes requested, sampled, evaluable, and
excluded gaps through the following files:

| File | Role |
| --- | --- |
| `input_manifest.csv` | Source units included in the domain workflow. |
| `selected_recordings.csv` | Deterministically selected recordings, station-years, or sensor-years. |
| `coverage_table.csv` | Requested, generated, successful, and shortfall counts by relevant group and stratum. |
| `excluded_gaps.csv` | Candidate intervals not retained, with reasons where available. |
| `learnable_gap_table.csv` | Per-gap geometry, observable features, applicability, errors, and oracle labels for eligible benchmark gaps. |
| `metadata.json` | Frozen protocol, methods, feature contract, random seed, and aggregate counts. |

Nested-evaluation results additionally retain per-gap predictions, per-group
summaries, selected-method counts, fold-local feature-scale floors, and
hyperparameter-tuning tables. These artifacts make it possible to inspect
sampling, exclusions, candidate applicability, and selector behavior without
access to raw study data.

## 9. Reproduction entry points

The canonical reproduction sequence, including the domain-specific notebook
paths and command-line entry points, is documented in
[REPRODUCE.md](../REPRODUCE.md). External raw-data requirements remain in the
individual files below `data/`.
