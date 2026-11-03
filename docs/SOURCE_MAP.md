# Source Map

The portfolio retains the original stage and file names where practical. Paths below are relative to the supplied research directory and to this package. Selection and editing did not overwrite the original files.

| Portfolio file | Original research file |
| --- | --- |
| `scripts/0_cohort/01_build_cohort.py` | `scripts/0_cohort/01_build_cohort.py` |
| `scripts/0_cohort/02_deduplicate_patients.py` | `scripts/0_cohort/02_deduplicate_patients.py` |
| `scripts/0_cohort/03_hospital_eligibility_filter.py` | `scripts/0_cohort/03_hospital_eligibility_filter.py` |
| `scripts/0_cohort/04_extract_labels.py` | `scripts/0_cohort/04_extract_labels.py` |
| `scripts/0_cohort/run_pipeline.py` | `scripts/0_cohort/run_pipeline.py` |
| `scripts/1_data_extraction/extract_medication_sample.py` | `scripts/1_data_extraction/extract_medication_sample.py` |
| `scripts/1_data_extraction/extract_nurseCharting_sample.py` | `scripts/1_data_extraction/extract_nurseCharting_sample.py` |
| `scripts/1_data_extraction/extract_other_tables_sample.py` | `scripts/1_data_extraction/extract_other_tables_sample.py` |
| `scripts/1_data_extraction/extract_pivoted_sample.py` | `scripts/1_data_extraction/extract_pivoted_sample.py` |
| `scripts/1_data_extraction/extract_resp_IO_sample.py` | `scripts/1_data_extraction/extract_resp_IO_sample.py` |
| `scripts/1_data_extraction/extract_respiratoryCharting_sample.py` | `scripts/1_data_extraction/extract_respiratoryCharting_sample.py` |
| `scripts/1_data_extraction/extract_infusionDrug_sample.py` | `scripts/1_data_extraction/extract_infusionDrug_sample.py` |
| `scripts/1_data_extraction/generate_labels_sample.py` | `scripts/1_data_extraction/generate_labels_sample.py` |
| `scripts/2_population/population_mortality_sample.py` | `scripts/2_population/population_mortality_sample.py` |
| `scripts/3_apply_FIDDLE/prepare_data_pooled_sample.py` | `scripts/3_apply_FIDDLE/prepare_data_pooled_sample.py` |
| `scripts/4_ML_models/pooled_svd_embeddings.py` | `scripts/4_ML_models/pooled_svd_embeddings.py` |
| `scripts/4_ML_models/cross_hospital_transportability.py` | `scripts/4_ML_models/cross_hospital_transportability.py` |
| `scripts/4_ML_models/similarity_by_outcome_class.py` | `scripts/4_ML_models/similarity_by_outcome_class.py` |
| `scripts/5_transportability/balanced_similarity_drop_correlation.py` | `scripts/5_transportability/balanced_similarity_drop_correlation.py` |
| `scripts/5_transportability/poster_absolute_lean_auprc_figures.py` | `scripts/5_transportability/poster_absolute_lean_auprc_figures.py` |
| `scripts/4_ML_models/similarity_utils.py` | `scripts/4_ML_models/similarity_tail_analysis.py` |

The shared similarity helper module contains functions extracted from the original `similarity_tail_analysis.py`; its additional model-fitting experiments are not bundled. The package adds `run_analysis.py`, `prepare_data.py`, and `examples/synthetic_demo.py` for portfolio navigation and verification.

Extraction and mortality population code adapt the FIDDLE experiment workflow; see [third-party attribution](../THIRD_PARTY_NOTICES.md). Implementation changes are summarized in the [reproduction guide](REPRODUCIBILITY.md).
