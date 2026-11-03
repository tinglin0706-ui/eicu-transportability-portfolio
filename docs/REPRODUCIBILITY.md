# Reproduction Guide

## Local demonstration

Install `requirements.txt` in a Python 3.10 virtual environment, then run:

```bash
python examples/synthetic_demo.py --outdir demo_output
```

Use a new output directory for each run. The demonstration creates 487 fictional patient-table rows, including deliberately ineligible and repeated records. After cohort filtering and deduplication, 480 patients remain across six fictional hospital groups. Hospital 73 supplies 80 source records and five target hospitals supply 400 records. Hospital identifiers are used as software inputs only; the fictional records do not describe real hospitals.

The demonstration uses 4 SVD components, 30 trees, and depth 2 to reduce runtime. It retains the five-fold evaluation design and source-derived threshold selection. It generates synthetic `S.npz` and `X.npz` arrays directly. Raw clinical-table extraction and the FIDDLE transformation are not exercised by this demonstration.

Generated files include cohort audits, population files, CV metrics, target predictions, embeddings, hospital correlations, figures, and `demo_summary.json`. All are synthetic. They remain outside the submitted source package when local verification is performed.

## Authorized-data workflow

The full workflow requires the eICU tables and the FIDDLE implementation used by the project. FIDDLE is not installed by `requirements.txt`. The original FIDDLE revision and complete research environment were not present in the supplied project directory, so exact historical reproduction is not claimed.

The preparation wrapper exposes each stage separately. Replace the two paths below with local authorized locations:

```bash
export EICU_DATA_DIR="/path/to/authorized/eicu"
export EICU_RUN_DIR="/path/to/private/run"

python prepare_data.py --stage cohort --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
python prepare_data.py --stage extract --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
python prepare_data.py --stage labels --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
python prepare_data.py --stage population --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
python prepare_data.py --stage prepare --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
```

After installing a compatible FIDDLE implementation, the following stage invokes the project-recorded mortality configuration: 48-hour observation window, 4-hour bins, `theta_1=0.03`, `theta_2=0.01`, `theta_freq=1`, and min/max/mean statistics. These settings are taken from the retained project launcher; a directory name alone should not be used to infer the threshold value.

```bash
python prepare_data.py --stage fiddle --eicu-data-dir "$EICU_DATA_DIR" --outdir "$EICU_RUN_DIR"
```

Run the analysis after feature generation:

```bash
python run_analysis.py \
  --config configs/mortality.json \
  --fiddle-output-dir "$EICU_RUN_DIR/features/mortality_48h/FIDDLE_output_theta1_003" \
  --population-file "$EICU_RUN_DIR/population/mortality_48h_with_metadata.csv" \
  --outdir "$EICU_RUN_DIR/analysis"
```

The wrapper reads new CV baselines and the selected threshold from the current transportability output and passes them to downstream stages. The individual analysis stages require run-specific thresholds and CV baselines; the wrapper supplies these automatically. Its figure stage is restricted to source hospital 73 because retained figure labels refer to that source.

## Input and output contracts

| Stage | Required input | Main output |
| --- | --- | --- |
| Cohort | `patient.csv.gz` with patient, hospital, age, stay-duration, and discharge-status fields | Final cohort and exclusion audits |
| Extraction | Final cohort IDs and the corresponding clinical tables | Events: `ID`, `t`, `variable_name`, `variable_value` |
| Labels | Patient table and final cohort | `icustays.csv`, `mortality.csv` |
| Population | Label outputs | `mortality_48h.csv` and `_with_metadata.csv` |
| Input preparation | Event files and mortality population | Time-filtered `input_data.csv` |
| FIDDLE | Input events and plain population | PyData-sparse `S.npz`, `X.npz`, `IDs.csv` |
| Analysis | FIDDLE outputs and population metadata | Predictions, embeddings, hospital metrics, correlations, figures |

The `partition` column inherited from label extraction is metadata; the XGBoost evaluation code creates its own source-hospital folds rather than using those partitions. Intermediate records, embeddings, and predictions generated from original data should remain in the authorized research environment.

## Portfolio changes

The retained scripts were selected from the supplied project directory. Chinese instructional comments and verbose debugging notes were removed; module descriptions and reviewer documentation are in English. Research paths were replaced with package-relative defaults or explicit command-line inputs. Backups, old model comparisons, cluster account settings, and notebooks with stored outputs were omitted.

The main integration changes are direct export of predictions from the evaluated source model, extraction of shared similarity helpers, a generic `predicted_probability_source` column, and passing run-specific thresholds and baselines between stages. The label-extraction script is limited to mortality in this package, with discharge-status normalization matched to the cohort builder. The synthetic example and the two top-level launchers were added during portfolio preparation; they should not be interpreted as evidence of additional original research experiments.
