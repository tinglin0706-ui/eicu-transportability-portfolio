"""Prepare an authorized eICU dataset one stage at a time."""

import argparse
from pathlib import Path
import subprocess
import sys

from run_analysis import run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["cohort", "extract", "labels", "population", "prepare", "fiddle"],
    )
    parser.add_argument("--eicu-data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-stays", type=int, default=1000)
    args = parser.parse_args()
    raw, out = args.eicu_data_dir.resolve(), args.outdir.resolve()
    cohort = out / "cohort/final/adult_mortality_48h_final_cohort.csv.gz"
    labels, events, population = out / "labels", out / "events", out / "population"
    if args.stage == "cohort":
        run_script(
            "scripts/0_cohort/01_build_cohort.py",
            "--patient-file",
            raw / "patient.csv.gz",
            "--output-dir",
            out / "cohort/eligible",
        )
        run_script(
            "scripts/0_cohort/02_deduplicate_patients.py",
            "--input",
            out / "cohort/eligible/adult_mortality_48h_cohort.csv.gz",
            "--output-dir",
            out / "cohort/unique",
        )
        run_script(
            "scripts/0_cohort/03_hospital_eligibility_filter.py",
            "--input",
            out / "cohort/unique/adult_mortality_48h_one_stay_per_patient.csv.gz",
            "--output-dir",
            out / "cohort/final",
            "--min-stays",
            args.min_stays,
            "--min-positives",
            0,
        )
        run_script(
            "scripts/0_cohort/04_extract_labels.py",
            "--input",
            cohort,
            "--output",
            labels / "cohort_labels.csv",
        )
    elif args.stage == "extract":
        for name in [
            "medication",
            "nurseCharting",
            "other_tables",
            "resp_IO",
            "respiratoryCharting",
            "infusionDrug",
        ]:
            run_script(
                f"scripts/1_data_extraction/extract_{name}_sample.py",
                "--eicu_path",
                raw,
                "--sample_ids_path",
                cohort,
                "--save_path",
                events,
            )
        for table in ["vitalPeriodic", "vitalAperiodic"]:
            run_script(
                "scripts/1_data_extraction/extract_pivoted_sample.py",
                table,
                "--eicu_path",
                raw,
                "--sample_ids_path",
                cohort,
                "--save_path",
                events,
            )
    elif args.stage == "labels":
        run_script(
            "scripts/1_data_extraction/generate_labels_sample.py",
            "--eicu_path",
            raw,
            "--sample_ids_path",
            cohort,
            "--labels_path",
            labels,
        )
    elif args.stage == "population":
        run_script(
            "scripts/2_population/population_mortality_sample.py",
            "--labels_path",
            labels,
            "--population_path",
            population,
        )
    elif args.stage == "prepare":
        run_script(
            "scripts/3_apply_FIDDLE/prepare_data_pooled_sample.py",
            "--extracted_path",
            events,
            "--population_path",
            population,
            "--features_root",
            out / "prepared",
            "--summary_path",
            out / "audit",
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "FIDDLE.run",
                f"--data_fname={out / 'prepared/mortality_48h/input_data.csv'}",
                f"--population_fname={population / 'mortality_48h.csv'}",
                f"--output_dir={out / 'features/mortality_48h/FIDDLE_output_theta1_003'}",
                "--T=2880.0",
                "--dt=240.0",
                "--theta_1=0.03",
                "--theta_2=0.01",
                "--theta_freq=1",
                "--stats_functions",
                "min",
                "max",
                "mean",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
