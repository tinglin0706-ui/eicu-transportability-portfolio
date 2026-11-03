"""Exercise the retained workflow with fictional records and feature arrays."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_analysis import run_analysis, run_script


def create_records(outdir):
    rng = np.random.default_rng(42)
    records, features = [], {}
    for index, hospital in enumerate([73, 101, 102, 103, 104, 105]):
        for patient in range(80):
            identifier = 10000 + index * 1000 + patient
            latent = rng.normal(size=12)
            latent[0] += index * 0.12
            risk = 1 / (1 + np.exp(-1.5 * latent[0] + 0.7 * latent[1] + 0.8))
            outcome = int(rng.random() < risk)
            records.append(
                {
                    "patientunitstayid": identifier,
                    "uniquepid": f"SYNTHETIC_{identifier}",
                    "patienthealthsystemstayid": identifier + 900000,
                    "hospitalid": hospital,
                    "gender": "Female" if patient % 2 else "Male",
                    "age": "> 89" if patient == 0 else str(18 + patient % 70),
                    "unitdischargeoffset": 4000,
                    "hospitaldischargestatus": "Expired" if outcome else "Alive",
                    "hospitaldischargeyear": 2015,
                }
            )
            features[identifier] = latent
    raw = pd.DataFrame(records)
    death_index = raw.index[raw["hospitaldischargestatus"].eq("Expired")][0]
    raw.loc[death_index, "hospitaldischargestatus"] = " expired "
    additions = []
    for i, change in enumerate(
        [
            {"age": "17"},
            {"unitdischargeoffset": 2879},
            {"hospitaldischargestatus": None},
            {"age": "not recorded"},
        ]
    ):
        row = records[5].copy()
        row.update(patientunitstayid=99000 + i, uniquepid=f"EXCLUDED_{i}", **change)
        additions.append(row)
    # A repeat stay is removed; both records of a cross-hospital patient are excluded.
    repeat = records[5].copy()
    repeat["patientunitstayid"] = 99900
    additions.append(repeat)
    for i, hospital in enumerate([73, 101]):
        row = records[6].copy()
        row.update(
            patientunitstayid=99910 + i, uniquepid="CROSS_HOSPITAL", hospitalid=hospital
        )
        additions.append(row)
    raw = pd.concat([raw, pd.DataFrame(additions)], ignore_index=True)
    rawdir = outdir / "synthetic_raw"
    rawdir.mkdir(parents=True)
    raw.to_csv(rawdir / "patient.csv.gz", index=False, compression="gzip")
    return rawdir, features


def prepare_inputs(outdir, rawdir, features):
    cohort = outdir / "cohort"
    run_script(
        "scripts/0_cohort/01_build_cohort.py",
        "--patient-file",
        rawdir / "patient.csv.gz",
        "--output-dir",
        cohort / "eligible",
    )
    run_script(
        "scripts/0_cohort/02_deduplicate_patients.py",
        "--input",
        cohort / "eligible/adult_mortality_48h_cohort.csv.gz",
        "--output-dir",
        cohort / "unique",
    )
    run_script(
        "scripts/0_cohort/03_hospital_eligibility_filter.py",
        "--input",
        cohort / "unique/adult_mortality_48h_one_stay_per_patient.csv.gz",
        "--output-dir",
        cohort / "final",
        "--min-stays",
        50,
        "--min-positives",
        0,
    )
    final = cohort / "final/adult_mortality_48h_final_cohort.csv.gz"
    run_script(
        "scripts/0_cohort/04_extract_labels.py",
        "--input",
        final,
        "--output",
        outdir / "labels/cohort_labels.csv",
    )
    retained = pd.read_csv(final)
    assert len(retained) == 480
    assert retained["uniquepid"].is_unique
    assert retained["age"].eq(90).sum() == 6
    assert set(retained["patientunitstayid"]) == set(features)

    run_script(
        "scripts/1_data_extraction/generate_labels_sample.py",
        "--eicu_path",
        rawdir,
        "--sample_ids_path",
        final,
        "--labels_path",
        outdir / "labels",
    )
    run_script(
        "scripts/2_population/population_mortality_sample.py",
        "--labels_path",
        outdir / "labels",
        "--population_path",
        outdir / "population",
    )
    population = outdir / "population/mortality_48h_with_metadata.csv"
    metadata = pd.read_csv(population).sample(frac=1, random_state=19)
    metadata.to_csv(population, index=False)
    expected = retained.set_index("patientunitstayid")["mortality_LABEL"].sort_index()
    actual = metadata.set_index("ID")["mortality_LABEL"].sort_index()
    np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())

    events = []
    for identifier in features:
        events.extend(
            [
                (identifier, np.nan, "synthetic_static", "1"),
                (identifier, 0, "synthetic_measurement", "2"),
                (identifier, 0, "synthetic_measurement", "2"),
                (identifier, 2880, "after_window", "3"),
                (identifier, -1, "before_window", "3"),
                (identifier, 60, "hospitalid", "73"),
            ]
        )
    extracted = outdir / "events"
    extracted.mkdir()
    pd.DataFrame(
        events, columns=["ID", "t", "variable_name", "variable_value"]
    ).to_parquet(extracted / "synthetic_events.parquet", index=False)
    run_script(
        "scripts/3_apply_FIDDLE/prepare_data_pooled_sample.py",
        "--extracted_path",
        extracted,
        "--population_path",
        outdir / "population",
        "--features_root",
        outdir / "prepared",
        "--summary_path",
        outdir / "audit",
    )
    prepared = pd.read_csv(outdir / "prepared/mortality_48h/input_data.csv")
    assert len(prepared) == 960
    assert set(prepared["variable_name"]) == {
        "synthetic_static",
        "synthetic_measurement",
    }

    fiddle = outdir / "synthetic_features"
    fiddle.mkdir()
    identifiers = sorted(features, reverse=True)
    values = np.array([features[x] for x in identifiers], dtype=np.float32)
    pd.DataFrame({"ID": identifiers}).to_csv(fiddle / "IDs.csv", index=False)
    sparse.save_npz(fiddle / "S.npz", sparse.COO.from_numpy(values[:, :4]))
    sparse.save_npz(
        fiddle / "X.npz", sparse.COO.from_numpy(values[:, 4:].reshape(480, 2, 4))
    )
    return fiddle, population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "demo_output")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error("Choose an empty output directory to keep runs separate.")
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / ".matplotlib"))
    rawdir, features = create_records(outdir)
    fiddle, population = prepare_inputs(outdir, rawdir, features)
    config = json.loads((ROOT / "configs/mortality.json").read_text())
    config.update(svd_components=4, n_estimators=30, max_depth=2)
    analysis = run_analysis(config, fiddle, population, outdir / "analysis")
    scores = pd.read_csv(
        analysis
        / "outcome_similarity/mortality_48h_source73_svd4_similarity_by_outcome_class.csv.gz"
    )
    assert len(scores) == 400
    assert scores["ID"].is_unique
    assert scores["predicted_probability_source"].between(0, 1).all()
    assert scores["similarity_to_source_decedents"].between(-1.00001, 1.00001).all()
    transport = pd.read_csv(
        analysis / "transportability/mortality_48h_source73_transportability.csv"
    )
    assert len(transport) == 5
    for metric in ["f1", "auroc", "auprc"]:
        np.testing.assert_allclose(
            transport[f"difference_{metric}"],
            transport[f"cv_mean_{metric}"] - transport[f"external_{metric}"],
        )
    summary = {
        "data_type": "synthetic",
        "retained_patients": 480,
        "source_patients": 80,
        "target_patients": 400,
        "target_hospitals": 5,
        "checks": "cohort exclusions, age coding, label alignment, event filtering, ID alignment, prediction export, similarity bounds, performance differences",
        "full_eicu_reproduction": False,
    }
    (outdir / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Synthetic workflow checks passed.")


if __name__ == "__main__":
    main()
