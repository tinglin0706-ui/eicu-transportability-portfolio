# Technical Guide

## Research question

The analysis examines whether representation-based patient similarity is associated with the external performance of a mortality model trained at one hospital. Hospital-level correlations describe associations; they do not establish that similarity causes a change in performance.

## Cohort and outcome

The cohort builder selects adults aged at least 18 years, with a known hospital, an ICU discharge offset of at least 2,880 minutes, and a resolvable hospital discharge status. Ages reported as greater than 89 are encoded as 90. The outcome is in-hospital mortality after the 48-hour observation period, rather than death within 48 hours.

Patients observed at more than one hospital are excluded. Among remaining patients with multiple stays, one stay is retained using hospital discharge year, when available, and the smallest stay ID as a deterministic tie-breaker. This procedure does not establish that the selected record is the first chronological ICU stay. Hospitals must have at least 1,000 eligible retained stays; the minimum positive-count setting is zero. These are code-defined settings, not independently reverified sample counts.

## Data preparation

The extraction scripts read selected eICU tables and standardize events to `ID`, `t`, `variable_name`, and `variable_value`. Event offsets are in minutes from ICU admission. The mortality input-preparation script retains timed events in `[0, 2880)` and untimed static records, removes identifier-like variables using explicit name patterns, and resolves duplicate patient-time-variable records by retaining the first occurrence.

The plain FIDDLE population file contains `ID` and `mortality_LABEL`. A separate metadata file supplies hospital membership for evaluation. Label and hospital metadata are not appended as predictors by the model-loading code. The identifier-name filter is a specific safeguard, not a comprehensive guarantee against clinical information leakage.

FIDDLE produces a static matrix `S.npz`, a temporal tensor `X.npz`, and `IDs.csv`. Arrays use the PyData `sparse` format. The feature loader flattens the temporal tensor along time and feature dimensions, then concatenates static features. Metadata are aligned to `IDs.csv` through validated patient-ID joins.

## Prediction and evaluation

The XGBoost model uses the unreduced FIDDLE feature matrix. The SVD representation is used for similarity, not for prediction. The retained configuration specifies 300 estimators, depth 8, learning rate 0.05, and random seed 42. Class weighting is calculated from source training labels.

Source evaluation uses five stratified folds, reduced when necessary to accommodate class counts. In each outer training fold, 25% is reserved for threshold selection. The model is fit on the remaining training subset; the F1-maximizing validation threshold is then applied to the untouched outer test fold. External evaluation uses a model refit on all source patients and the mean of the fold-specific thresholds. Consequently, the CV and final models have different training-set sizes, which should be considered when interpreting their performance difference.

For metric `m`, performance drop is `mean source CV(m) - external(m)`. Positive values indicate lower external performance. AUPRC is implemented as scikit-learn average precision. It is sensitive to outcome prevalence, so hospital AUPRC differences are not pure measures of population shift.

The portfolio version exports target predictions from this same refitted model for downstream similarity analysis. It passes the newly calculated threshold and CV baselines between stages, avoiding reuse of historical constants when the pipeline is rerun.

## Representation and similarity

TruncatedSVD is fitted to pooled features across source and target hospitals, with 500 components in the retained research configuration. Neither mortality labels nor hospital identifiers enter the SVD matrix. Target feature distributions nevertheless contribute to the representation. This is a transductive analysis, and the package does not establish performance under source-only preprocessing or representation fitting. The provenance of the original FIDDLE fitting configuration also needs to be considered when assessing the external-validation setting.

For target patient `i`, define:

- `d(i)`: mean cosine similarity to source patients who died.
- `s(i)`: mean cosine similarity to source patients who survived.
- `lean(i) = d(i) - s(i)`.
- `absolute_lean(i) = |d(i) - s(i)|`.

Absolute lean measures separation between the two reference similarities. A larger value does not necessarily mean greater overall similarity to source patients or a higher probability of death.

A class-balanced hospital mean averages the mean score among target decedents and the mean score among target survivors with equal weight. It differs from the patient-level `balanced_mean` score `(d(i) + s(i)) / 2`. Equal weighting removes direct weighting by the two target group sizes; it does not make the estimate independent of prevalence or group-specific sampling uncertainty.

Computing `d(i)` and `s(i)` for an individual target patient does not require that patient's outcome. Splitting target patients by outcome, class-balancing their hospital summaries, outcome-aligned scores, and the within-hospital outcome-reference control do require target outcomes. Those analyses cannot be treated as prospective deployment features.

## Statistical summaries

The correlation script computes Pearson and Spearman associations across target hospitals. Its BH adjustment groups Spearman tests by reference, aggregation, and performance metric, correcting across score metrics within each family. This is not a global correction across every exploratory analysis. The inferential sample size is the number of target hospitals, not the number of patients or patient pairs. Correlations from a single source hospital do not demonstrate generalization across source hospitals.

The retained figure script displays class-balanced absolute lean and separate target-decedent and target-survivor summaries against whole-hospital AUPRC drop. Clinical findings and interpretation belong to the accompanying poster; no historical research estimates are reproduced in this package.

## AI assistance and author responsibility

AI coding assistants were used during implementation. I designed the analysis, reviewed the code, and verified the outputs.
