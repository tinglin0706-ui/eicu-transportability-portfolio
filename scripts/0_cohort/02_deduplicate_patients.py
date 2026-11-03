"""Exclude patients observed across hospitals; retain one stay per remaining patient.

Stay selection uses discharge year and stay ID, not an inferred admission chronology."""


import argparse
from pathlib import Path

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_COLUMNS = {
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "mortality_LABEL",
}


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100 * numerator / denominator


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Remove cross-hospital patients and keep one stay per remaining patient."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            str(
                PORTFOLIO_ROOT
                / "data/cohort/adult_mortality_48h_all/adult_mortality_48h_cohort.csv.gz"
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            str(PORTFOLIO_ROOT / "data/cohort/adult_mortality_48h_patient_unique")
        ),
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    output_cohort = args.output_dir / "adult_mortality_48h_one_stay_per_patient.csv.gz"
    output_checkpoint = args.output_dir / "deduplication_checkpoint.csv"
    output_cross_patients = args.output_dir / "excluded_cross_hospital_patients.csv"
    output_removed_stays = args.output_dir / "removed_stays.csv.gz"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.input)

    missing_columns = REQUIRED_COLUMNS.difference(cohort.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if cohort["patientunitstayid"].isna().any():
        raise ValueError("patientunitstayid contains missing values.")

    if not cohort["patientunitstayid"].is_unique:
        duplicated = cohort.loc[
            cohort["patientunitstayid"].duplicated(keep=False), "patientunitstayid"
        ]
        raise ValueError(
            f"patientunitstayid is not unique. Examples: {duplicated.head(10).tolist()}"
        )

    if cohort["uniquepid"].isna().any():
        raise ValueError("uniquepid contains missing values.")

    if cohort["hospitalid"].isna().any():
        raise ValueError("hospitalid contains missing values.")

    if not cohort["mortality_LABEL"].isin([0, 1]).all():
        raise ValueError("mortality_LABEL must contain only 0 and 1.")

    cohort["hospitalid"] = pd.to_numeric(cohort["hospitalid"], errors="raise").astype(
        int
    )
    cohort["patientunitstayid"] = pd.to_numeric(
        cohort["patientunitstayid"], errors="raise"
    ).astype(int)
    cohort["uniquepid"] = cohort["uniquepid"].astype("string").str.strip()

    if cohort["uniquepid"].eq("").any():
        raise ValueError("uniquepid contains blank strings.")

    n_stays_before = len(cohort)
    n_patients_before = cohort["uniquepid"].nunique()
    n_hospitals_before = cohort["hospitalid"].nunique()

    patient_hospital_counts = cohort.groupby("uniquepid", observed=True)[
        "hospitalid"
    ].nunique()
    cross_hospital_ids = patient_hospital_counts.loc[patient_hospital_counts > 1].index
    cross_hospital_mask = cohort["uniquepid"].isin(cross_hospital_ids)

    cross_hospital_stays = cohort.loc[cross_hospital_mask].copy()
    after_cross_hospital = cohort.loc[~cross_hospital_mask].copy()

    n_cross_patients = len(cross_hospital_ids)
    n_cross_stays = len(cross_hospital_stays)
    n_stays_after_cross = len(after_cross_hospital)
    n_patients_after_cross = after_cross_hospital["uniquepid"].nunique()

    cross_patient_summary = (
        cross_hospital_stays.groupby("uniquepid", observed=True)
        .agg(
            n_stays=("patientunitstayid", "size"),
            n_hospitals=("hospitalid", "nunique"),
            hospitalids=(
                "hospitalid",
                lambda x: ",".join(map(str, sorted(x.unique()))),
            ),
        )
        .reset_index()
        .sort_values(["n_hospitals", "n_stays", "uniquepid"])
    )
    cross_patient_summary.to_csv(output_cross_patients, index=False)

    sort_columns = ["uniquepid"]

    if "hospitaldischargeyear" in after_cross_hospital.columns:
        after_cross_hospital["_sort_year"] = pd.to_numeric(
            after_cross_hospital["hospitaldischargeyear"], errors="coerce"
        ).fillna(float("inf"))
        sort_columns.append("_sort_year")

    sort_columns.append("patientunitstayid")

    ordered = after_cross_hospital.sort_values(
        sort_columns, ascending=True, kind="mergesort"
    )
    keep_mask = ~ordered.duplicated(subset="uniquepid", keep="first")

    deduplicated = ordered.loc[keep_mask].copy()
    repeated_stays_removed = ordered.loc[~keep_mask].copy()

    deduplicated = deduplicated.drop(columns=["_sort_year"], errors="ignore")
    repeated_stays_removed = repeated_stays_removed.drop(
        columns=["_sort_year"], errors="ignore"
    )

    n_first_stay_removed = len(repeated_stays_removed)
    n_stays_after_dedup = len(deduplicated)
    n_patients_after_dedup = deduplicated["uniquepid"].nunique()

    n_total_removed = n_stays_before - n_stays_after_dedup
    overall_removal_pct = pct(n_total_removed, n_stays_before)

    cross_hospital_stays["removal_reason"] = "patient_appears_in_multiple_hospitals"
    repeated_stays_removed["removal_reason"] = "additional_eligible_stay"

    removed_stays = pd.concat(
        [cross_hospital_stays, repeated_stays_removed], ignore_index=True
    )
    removed_stays = removed_stays.sort_values(
        ["removal_reason", "uniquepid", "patientunitstayid"]
    )
    removed_stays.to_csv(output_removed_stays, index=False, compression="gzip")

    if not deduplicated["patientunitstayid"].is_unique:
        raise AssertionError("Final patientunitstayid is not unique.")

    if not deduplicated["uniquepid"].is_unique:
        raise AssertionError("Final uniquepid is not unique.")

    remaining_cross_hospital = (
        deduplicated.groupby("uniquepid", observed=True)["hospitalid"]
        .nunique()
        .gt(1)
        .sum()
    )
    if remaining_cross_hospital != 0:
        raise AssertionError(
            f"{remaining_cross_hospital} cross-hospital patients remain."
        )

    if n_stays_after_dedup != n_patients_after_dedup:
        raise AssertionError(
            "Final number of stays does not equal number of unique patients."
        )

    if n_cross_stays + n_first_stay_removed != n_total_removed:
        raise AssertionError("Removal counts do not reconcile.")

    deduplicated = deduplicated.sort_values(
        ["hospitalid", "patientunitstayid"]
    ).reset_index(drop=True)
    deduplicated.to_csv(output_cohort, index=False, compression="gzip")

    checkpoint = pd.DataFrame(
        [
            {
                "stays_before": n_stays_before,
                "unique_patients_before": n_patients_before,
                "hospitals_before": n_hospitals_before,
                "cross_hospital_patients_excluded": n_cross_patients,
                "cross_hospital_stays_excluded": n_cross_stays,
                "stays_after_cross_hospital_exclusion": n_stays_after_cross,
                "patients_after_cross_hospital_exclusion": n_patients_after_cross,
                "additional_stays_removed_for_one_per_patient": n_first_stay_removed,
                "stays_after_deduplication": n_stays_after_dedup,
                "unique_patients_after_deduplication": n_patients_after_dedup,
                "total_stays_removed": n_total_removed,
                "overall_removal_proportion": n_total_removed / n_stays_before,
            }
        ]
    )
    checkpoint.to_csv(output_checkpoint, index=False)

    print("\nPATIENT DEDUPLICATION CHECKPOINT", flush=True)
    print("=" * 55, flush=True)
    print("\nOriginal eligible cohort", flush=True)
    print(f"  Stays:                          {n_stays_before:,}", flush=True)
    print(f"  Unique patients:                {n_patients_before:,}", flush=True)
    print(f"  Hospitals:                      {n_hospitals_before:,}", flush=True)
    print("\nCross-hospital exclusion", flush=True)
    print(f"  Cross-hospital patients:        {n_cross_patients:,}", flush=True)
    print(f"  Their stays excluded:           {n_cross_stays:,}", flush=True)
    print(f"  Stays remaining:                {n_stays_after_cross:,}", flush=True)
    print(f"  Patients remaining:             {n_patients_after_cross:,}", flush=True)
    print("\nOne eligible stay per patient", flush=True)
    print(f"  Additional stays removed:       {n_first_stay_removed:,}", flush=True)
    print(f"  Final stays:                    {n_stays_after_dedup:,}", flush=True)
    print(f"  Final unique patients:          {n_patients_after_dedup:,}", flush=True)
    print("\nOverall", flush=True)
    print(f"  Total stays removed:            {n_total_removed:,}", flush=True)
    print(f"  Overall deletion proportion:    {overall_removal_pct:.2f}%", flush=True)
    print("\nSelection rule", flush=True)
    print("  Coarse hospitaldischargeyear, then minimum patientunitstayid", flush=True)
    print("  hospitaladmitoffset was not treated as calendar chronology", flush=True)
    print("\nSaved files", flush=True)
    print(f"  Cohort:             {output_cohort}", flush=True)
    print(f"  Checkpoint:         {output_checkpoint}", flush=True)
    print(f"  Cross-hospital IDs: {output_cross_patients}", flush=True)
    print(f"  Removed stays:      {output_removed_stays}", flush=True)
    print("\nALL PATIENT DEDUPLICATION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
