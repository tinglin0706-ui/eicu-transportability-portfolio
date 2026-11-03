# Patient Similarity and Cross-Hospital Transportability

**Ting Lin · University of Michigan**

This code portfolio implements a study of how patient similarity relates to the cross-hospital performance of an ICU mortality prediction model. It includes cohort construction, clinical event extraction, FIDDLE input preparation, source-hospital model evaluation, and representation-based similarity analysis.

The accompanying poster presents the research findings. This package provides the implementation and a synthetic demonstration; it contains no original eICU records or study results.

## Portfolio documents

- [Software Portfolio (PDF)](portfolio/Ting_Lin_eICU_Software_Portfolio.pdf): research context, selected code, methods, and verification.
- [Contribution Statement (PDF)](portfolio/Ting_Lin_eICU_Contribution_Statement.pdf): contributions to the software and companion poster, research supervision, and third-party attribution.

Repository: [tinglin0706-ui/eicu-transportability-portfolio](https://github.com/tinglin0706-ui/eicu-transportability-portfolio).

## Where to start

| File | Purpose |
| --- | --- |
| [Contribution statement](CONTRIBUTION.md) | Applicant responsibilities, supervision, and reused software |
| [Technical guide](docs/TECHNICAL_GUIDE.md) | Study design, pipeline, and interpretation |
| [Reproduction guide](docs/REPRODUCIBILITY.md) | Environment, inputs, and commands |
| [`cross_hospital_transportability.py`](scripts/4_ML_models/cross_hospital_transportability.py) | Source-only model fitting and external evaluation |
| [`similarity_by_outcome_class.py`](scripts/4_ML_models/similarity_by_outcome_class.py) | Similarity to source decedents and survivors |
| [`run_analysis.py`](run_analysis.py) | Connect the retained analysis stages |

## Run the synthetic demonstration

Python 3.10 was used for the local verification environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/synthetic_demo.py --outdir demo_output
```

The demonstration generates fictional patient-table records and preconstructed FIDDLE-format feature arrays. It exercises cohort filtering, mortality label extraction, population preparation, model evaluation, SVD, and similarity analysis. The synthetic features bypass raw clinical-table extraction and FIDDLE transformation. Demonstration statistics have no clinical interpretation.

## Analysis design

The retained analysis trains an XGBoost mortality model at source hospital 73 and evaluates it at other hospitals. Stratified cross-validation provides the source performance baseline; a validation subset within each training fold determines the F1 threshold. The mean threshold is fixed for external evaluation.

A separate pooled TruncatedSVD representation supports cosine similarity calculations. Each target patient is compared with source decedents and source survivors. Hospital summaries are related to the difference between source cross-validation performance and external performance. Outcome-stratified and class-balanced summaries use target outcomes and are retrospective analyses.

## Package structure

```text
scripts/
  0_cohort/          Adult eligibility, deduplication, hospital filtering
  1_data_extraction/ Clinical event and outcome extraction
  2_population/     Mortality population construction
  3_apply_FIDDLE/    Event-window filtering and FIDDLE input preparation
  4_ML_models/       XGBoost evaluation, SVD, and outcome-class similarity
  5_transportability/ Hospital summaries, correlations, and figures
configs/            Explicit analysis settings
examples/           Synthetic demonstration
docs/               Methods, reproduction, and verification
```

Historical model comparisons, backups, cluster-specific job files, and notebooks with stored research outputs are omitted. Scripts retain their original stage-based organization to make the research workflow easy to follow.

## Attribution and scope

FIDDLE is external software by Tang and colleagues. The extraction and population workflow adapts the associated eICU experiment code. See [third-party attribution](THIRD_PARTY_NOTICES.md) for the paper and source repositories.

The full study requires authorized eICU access and a compatible FIDDLE installation. Local checks use synthetic inputs and do not establish numerical reproduction of the poster. The [verification record](docs/VERIFICATION.md) states what was tested.
