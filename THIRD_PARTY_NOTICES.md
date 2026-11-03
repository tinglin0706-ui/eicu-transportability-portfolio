# Third-Party Attribution

## FIDDLE

Tang S, Davarmanesh P, Song Y, Koutra D, Sjoding MW, Wiens J. *Democratizing EHR analyses with FIDDLE: a flexible data-driven preprocessing pipeline for structured clinical data.* Journal of the American Medical Informatics Association. 2020;27(12):1921–1934. [doi:10.1093/jamia/ocaa139](https://doi.org/10.1093/jamia/ocaa139).

- [FIDDLE software](https://github.com/MLD3/FIDDLE)
- [FIDDLE experiment code](https://github.com/MLD3/FIDDLE-experiments)
- [eICU extraction and population workflow](https://github.com/MLD3/FIDDLE-experiments/tree/77483adf4327e87cbea4963252db873829cad813/eicu_experiments/1_data_extraction)

The applicant identified these resources as the basis for label extraction and FIDDLE preprocessing. The experiment repository contains `generate_labels.ipynb`, `population_mortality.ipynb`, and extraction code corresponding to the adapted workflow in this package. The submitted versions add cohort selection, command-line interfaces, validation, and project integration; the existence of these adaptations does not imply original authorship of the upstream workflow.

| Portfolio component | Upstream relationship |
| --- | --- |
| `scripts/1_data_extraction/` | Adapted eICU clinical-table and outcome-extraction workflow |
| `scripts/2_population/population_mortality_sample.py` | Adapted mortality population preparation |
| `scripts/3_apply_FIDDLE/prepare_data_pooled_sample.py` | Project preparation layer for the external FIDDLE pipeline |
| FIDDLE transformation | External dependency; implementation is not bundled |

Attribution was checked against FIDDLE revision `7dd6d98819ccf1602e0ab84a54815f07064f411e` and FIDDLE-experiments revision `77483adf4327e87cbea4963252db873829cad813` on November 3, 2026. These are inspected reference revisions, not evidence of the revisions used in the original research environment. An exact historical upstream version was not supplied.

FIDDLE has an MIT license. No repository-level license file was identified in the inspected FIDDLE-experiments tree; the FIDDLE license should not be assumed to cover the separate experiment repository. This portfolio does not assign a blanket software license to adapted upstream files.

## Other dependencies

The analysis uses NumPy, pandas, SciPy, scikit-learn, XGBoost, PyData sparse, matplotlib, PyArrow, joblib, numba, and tqdm. Their implementations are external dependencies and are not bundled. See `requirements.txt` and the verification environment for the versions used in local checks.
