"""Apply minimum hospital cohort-size and outcome-count requirements."""


import argparse
from pathlib import Path

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_COLUMNS = {
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "age",
    "mortality_LABEL",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Filter to hospitals meeting minimum stay/positive-case criteria."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            str(
                PORTFOLIO_ROOT
                / "data/cohort/adult_mortality_48h_patient_unique/adult_mortality_48h_one_stay_per_patient.csv.gz"
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(PORTFOLIO_ROOT / "data/cohort/adult_mortality_48h_final")),
    )
    parser.add_argument(
        "--min-stays",
        type=int,
        default=100,
        help="Minimum number of stays a hospital must have to be kept.",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=25,
        help="Minimum number of positive (mortality) cases a hospital must have to be kept.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    min_stays = args.min_stays
    min_positives = args.min_positives

    output_cohort = args.output_dir / "adult_mortality_48h_final_cohort.csv.gz"
    output_manifest = args.output_dir / "adult_mortality_48h_final_manifest.csv.gz"
    output_hospital_summary = args.output_dir / "hospital_eligibility_summary.csv"
    output_checkpoint = args.output_dir / "hospital_filtering_checkpoint.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.input)

    missing_columns = REQUIRED_COLUMNS.difference(cohort.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if cohort["patientunitstayid"].isna().any():
        raise ValueError("patientunitstayid contains missing values.")
    if not cohort["patientunitstayid"].is_unique:
        raise ValueError("Input patientunitstayid is not unique.")
    if cohort["uniquepid"].isna().any():
        raise ValueError("uniquepid contains missing values.")

    cohort["uniquepid"] = cohort["uniquepid"].astype("string").str.strip()

    if cohort["uniquepid"].eq("").any():
        raise ValueError("uniquepid contains blank strings.")
    if not cohort["uniquepid"].is_unique:
        raise ValueError(
            "Input cohort is not patient-unique. "
            "Run 02_deduplicate_patients.py first."
        )

    cohort["patientunitstayid"] = pd.to_numeric(
        cohort["patientunitstayid"], errors="raise"
    ).astype(int)
    cohort["hospitalid"] = pd.to_numeric(cohort["hospitalid"], errors="raise").astype(
        int
    )
    cohort["mortality_LABEL"] = pd.to_numeric(
        cohort["mortality_LABEL"], errors="raise"
    ).astype(int)

    if not cohort["mortality_LABEL"].isin([0, 1]).all():
        raise ValueError("mortality_LABEL must contain only 0 and 1.")
    if cohort["age"].isna().any():
        raise ValueError("age contains missing values.")
    if not cohort["age"].ge(18).all():
        raise ValueError("Patients younger than 18 remain in the cohort.")

    hospital_summary = (
        cohort.groupby("hospitalid", observed=True)
        .agg(
            n_stays=("patientunitstayid", "size"),
            n_unique_patients=("uniquepid", "nunique"),
            n_positive=("mortality_LABEL", "sum"),
        )
        .reset_index()
    )

    hospital_summary["n_negative"] = (
        hospital_summary["n_stays"] - hospital_summary["n_positive"]
    )
    hospital_summary["prevalence"] = (
        hospital_summary["n_positive"] / hospital_summary["n_stays"]
    )
    hospital_summary["meets_min_stays"] = hospital_summary["n_stays"] >= min_stays
    hospital_summary["meets_min_positives"] = (
        hospital_summary["n_positive"] >= min_positives
    )
    hospital_summary["has_negative"] = hospital_summary["n_negative"] > 0
    hospital_summary["eligible"] = (
        hospital_summary["meets_min_stays"]
        & hospital_summary["meets_min_positives"]
        & hospital_summary["has_negative"]
    )

    def exclusion_reason(row):
        reasons = []
        if not row["meets_min_stays"]:
            reasons.append(f"n_stays<{min_stays}")
        if not row["meets_min_positives"]:
            reasons.append(f"n_positive<{min_positives}")
        if not row["has_negative"]:
            reasons.append("n_negative=0")
        if not reasons:
            return "eligible"
        return ";".join(reasons)

    hospital_summary["eligibility_reason"] = hospital_summary.apply(
        exclusion_reason, axis=1
    )
    hospital_summary = hospital_summary.sort_values(
        ["eligible", "n_stays", "hospitalid"], ascending=[False, False, True]
    ).reset_index(drop=True)
    hospital_summary.to_csv(output_hospital_summary, index=False)

    eligible_hospitalids = hospital_summary.loc[
        hospital_summary["eligible"], "hospitalid"
    ]
    final_cohort = cohort.loc[cohort["hospitalid"].isin(eligible_hospitalids)].copy()

    prevalence_map = hospital_summary.set_index("hospitalid")["prevalence"]
    final_cohort["hospital_prevalence"] = final_cohort["hospitalid"].map(prevalence_map)

    if final_cohort.empty:
        raise AssertionError("No hospitals passed the eligibility criteria.")
    if not final_cohort["patientunitstayid"].is_unique:
        raise AssertionError("Final patientunitstayid is not unique.")
    if not final_cohort["uniquepid"].is_unique:
        raise AssertionError("Final uniquepid is not unique.")
    if final_cohort["hospital_prevalence"].isna().any():
        raise AssertionError("hospital_prevalence contains missing values.")

    final_check = final_cohort.groupby("hospitalid", observed=True).agg(
        n_stays=("patientunitstayid", "size"),
        n_unique_patients=("uniquepid", "nunique"),
        n_positive=("mortality_LABEL", "sum"),
    )
    final_check["n_negative"] = final_check["n_stays"] - final_check["n_positive"]

    if not final_check["n_stays"].ge(min_stays).all():
        raise AssertionError(f"A final hospital has fewer than {min_stays} stays.")
    if not final_check["n_positive"].ge(min_positives).all():
        raise AssertionError(
            f"A final hospital has fewer than {min_positives} positives."
        )
    if not final_check["n_negative"].gt(0).all():
        raise AssertionError("A final hospital has no negative cases.")
    if not (final_check["n_stays"] == final_check["n_unique_patients"]).all():
        raise AssertionError(
            "At least one final hospital has multiple stays per patient."
        )

    final_hospitalids = set(final_cohort["hospitalid"].unique())
    expected_hospitalids = set(eligible_hospitalids.tolist())
    if final_hospitalids != expected_hospitalids:
        raise AssertionError("Final hospital IDs do not match eligible hospital IDs.")

    final_cohort = final_cohort.sort_values(
        ["hospitalid", "patientunitstayid"]
    ).reset_index(drop=True)
    final_cohort.to_csv(output_cohort, index=False, compression="gzip")

    final_manifest = (
        final_cohort[
            [
                "patientunitstayid",
                "uniquepid",
                "hospitalid",
                "mortality_LABEL",
                "age",
                "hospital_prevalence",
            ]
        ]
        .rename(columns={"mortality_LABEL": "label"})
        .copy()
    )
    final_manifest.to_csv(output_manifest, index=False, compression="gzip")

    n_hospitals_before = cohort["hospitalid"].nunique()
    n_stays_before = len(cohort)
    n_hospitals_final = final_cohort["hospitalid"].nunique()
    n_stays_final = len(final_cohort)
    n_patients_final = final_cohort["uniquepid"].nunique()
    n_hospitals_removed = n_hospitals_before - n_hospitals_final
    n_stays_removed = n_stays_before - n_stays_final

    eligible_summary = hospital_summary.loc[hospital_summary["eligible"]].copy()
    prevalence_min = eligible_summary["prevalence"].min()
    prevalence_median = eligible_summary["prevalence"].median()
    prevalence_max = eligible_summary["prevalence"].max()

    checkpoint = pd.DataFrame(
        [
            {
                "minimum_stays": min_stays,
                "minimum_positives": min_positives,
                "hospitals_before_filtering": n_hospitals_before,
                "stays_before_filtering": n_stays_before,
                "hospitals_excluded": n_hospitals_removed,
                "stays_excluded": n_stays_removed,
                "final_hospitals": n_hospitals_final,
                "final_stays": n_stays_final,
                "final_unique_patients": n_patients_final,
                "prevalence_min": prevalence_min,
                "prevalence_median": prevalence_median,
                "prevalence_max": prevalence_max,
            }
        ]
    )
    checkpoint.to_csv(output_checkpoint, index=False)

    print("\nHOSPITAL ELIGIBILITY CHECKPOINT", flush=True)
    print("=" * 60, flush=True)
    print("\nInput patient-unique cohort", flush=True)
    print(f"  Hospitals:                      {n_hospitals_before:,}", flush=True)
    print(f"  Stays:                          {n_stays_before:,}", flush=True)
    print(
        f"  Unique patients:                {cohort['uniquepid'].nunique():,}",
        flush=True,
    )
    print("\nEligibility criteria", flush=True)
    print(f"  Minimum stays:                  {min_stays:,}", flush=True)
    print(f"  Minimum positive cases:         {min_positives:,}", flush=True)
    print("  Negative cases required:        Yes", flush=True)
    print("\nHospital filtering", flush=True)
    print(f"  Hospitals excluded:             {n_hospitals_removed:,}", flush=True)
    print(f"  Stays excluded:                 {n_stays_removed:,}", flush=True)
    print(f"  Final hospitals:                {n_hospitals_final:,}", flush=True)
    print(f"  Final stays:                    {n_stays_final:,}", flush=True)
    print(f"  Final unique patients:          {n_patients_final:,}", flush=True)
    print("\nFinal hospital prevalence", flush=True)
    print(f"  Minimum:                        {prevalence_min:.4f}", flush=True)
    print(f"  Median:                         {prevalence_median:.4f}", flush=True)
    print(f"  Maximum:                        {prevalence_max:.4f}", flush=True)
    print("\nFinal hospital size", flush=True)
    print(
        f"  Minimum stays:                  {eligible_summary['n_stays'].min():,}",
        flush=True,
    )
    print(
        f"  Median stays:                   {eligible_summary['n_stays'].median():,.1f}",
        flush=True,
    )
    print(
        f"  Maximum stays:                  {eligible_summary['n_stays'].max():,}",
        flush=True,
    )
    print("\nFinal positive counts", flush=True)
    print(
        f"  Minimum positives:              {eligible_summary['n_positive'].min():,}",
        flush=True,
    )
    print(
        f"  Median positives:               {eligible_summary['n_positive'].median():,.1f}",
        flush=True,
    )
    print(
        f"  Maximum positives:              {eligible_summary['n_positive'].max():,}",
        flush=True,
    )
    print("\nSaved files", flush=True)
    print(f"  Final cohort:       {output_cohort}", flush=True)
    print(f"  Final manifest:     {output_manifest}", flush=True)
    print(f"  Hospital summary:   {output_hospital_summary}", flush=True)
    print(f"  Checkpoint:         {output_checkpoint}", flush=True)
    print("\nNO SAMPLING WAS PERFORMED", flush=True)
    print("ALL HOSPITAL ELIGIBILITY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
