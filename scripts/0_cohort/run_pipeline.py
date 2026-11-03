"""Build the adult cohort, deduplicate patients, filter hospitals, and extract labels."""

import os


import argparse
import subprocess
import sys
from pathlib import Path


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


MIN_STAYS = 1000
MIN_POSITIVES = 0


SCRIPT_DIR = Path(__file__).resolve().parent

RAW_PATIENT_FILE = Path(str(EICU_ROOT / "patient.csv.gz"))

COHORT_ROOT = Path(str(PORTFOLIO_ROOT / "data/cohort"))
LABELS_ROOT = Path(str(PORTFOLIO_ROOT / "data/labels"))

STEP1_DIR = COHORT_ROOT / "adult_mortality_48h_all"
STEP2_DIR = COHORT_ROOT / "adult_mortality_48h_patient_unique"
STEP3_DIR = COHORT_ROOT / "adult_mortality_48h_final"

STEP1_COHORT = STEP1_DIR / "adult_mortality_48h_cohort.csv.gz"
STEP2_COHORT = STEP2_DIR / "adult_mortality_48h_one_stay_per_patient.csv.gz"
STEP3_COHORT = STEP3_DIR / "adult_mortality_48h_final_cohort.csv.gz"
LABELS_OUTPUT = LABELS_ROOT / "adult_mortality_48h_labels.csv"


def run_step(description, cmd):
    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING: {description}", flush=True)
    print("=" * 70, flush=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(
            f"\nSTEP FAILED: {description} (exit code {result.returncode})", flush=True
        )
        sys.exit(result.returncode)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the full cohort pipeline.")

    parser.add_argument("--patient-file", type=Path, default=RAW_PATIENT_FILE)
    parser.add_argument("--min-stays", type=int, default=MIN_STAYS)
    parser.add_argument("--min-positives", type=int, default=MIN_POSITIVES)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip step 1 and reuse whatever is already at STEP1_COHORT.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip step 4 (label extraction).",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.skip_build:
        print(f"\nSkipping step 1 (--skip-build). Reusing: {STEP1_COHORT}", flush=True)
        if not STEP1_COHORT.exists():
            print(
                f"ERROR: {STEP1_COHORT} does not exist. Remove --skip-build.",
                flush=True,
            )
            sys.exit(1)
    else:
        run_step(
            "Step 1/4 - build adult 48h cohort",
            [
                sys.executable,
                str(SCRIPT_DIR / "01_build_cohort.py"),
                "--patient-file",
                str(args.patient_file),
                "--output-dir",
                str(STEP1_DIR),
            ],
        )

    run_step(
        "Step 2/4 - remove cross-hospital patients and deduplicate",
        [
            sys.executable,
            str(SCRIPT_DIR / "02_deduplicate_patients.py"),
            "--input",
            str(STEP1_COHORT),
            "--output-dir",
            str(STEP2_DIR),
        ],
    )

    run_step(
        f"Step 3/4 - hospital eligibility filter "
        f"(min_stays={args.min_stays}, min_positives={args.min_positives})",
        [
            sys.executable,
            str(SCRIPT_DIR / "03_hospital_eligibility_filter.py"),
            "--input",
            str(STEP2_COHORT),
            "--output-dir",
            str(STEP3_DIR),
            "--min-stays",
            str(args.min_stays),
            "--min-positives",
            str(args.min_positives),
        ],
    )

    if args.skip_labels:
        print("\nSkipping step 4 (--skip-labels).", flush=True)
    else:
        run_step(
            "Step 4/4 - extract labels",
            [
                sys.executable,
                str(SCRIPT_DIR / "04_extract_labels.py"),
                "--input",
                str(STEP3_COHORT),
                "--output",
                str(LABELS_OUTPUT),
            ],
        )

    print("\n" + "=" * 70, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"  min_stays:      {args.min_stays}", flush=True)
    print(f"  min_positives:  {args.min_positives}", flush=True)
    print(f"  final cohort:   {STEP3_COHORT}", flush=True)
    print(
        f"  labels:         {LABELS_OUTPUT if not args.skip_labels else '(skipped)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
