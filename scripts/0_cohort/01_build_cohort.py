"""Select adults with at least 48 hours in the ICU and known hospital mortality."""

import os


import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


REQUIRED_COLUMNS = {
    "patientunitstayid",
    "hospitalid",
    "age",
    "unitdischargeoffset",
    "hospitaldischargestatus",
}


def parse_age(series):
    age_text = (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "> 89": "90",
                ">89": "90",
            }
        )
    )
    return pd.to_numeric(age_text, errors="coerce")


def make_mortality_label(series):
    status = series.astype("string").str.strip().str.lower()

    label = pd.Series(pd.NA, index=series.index, dtype="Int64")
    label.loc[status.eq("alive")] = 0
    label.loc[status.eq("expired")] = 1

    return label


def proportional_allocation(counts, sample_size):
    ideal = counts / counts.sum() * sample_size
    allocation = np.floor(ideal).astype(int)

    remaining = sample_size - int(allocation.sum())

    while remaining > 0:
        capacity = counts - allocation
        candidates = capacity[capacity > 0]

        if candidates.empty:
            raise RuntimeError("Unable to allocate the requested sample size.")

        remainder = ideal[candidates.index] - allocation[candidates.index]
        selected_class = remainder.idxmax()

        allocation.loc[selected_class] += 1
        remaining -= 1

    return allocation


def stratified_sample_hospital(hospital_df, n, label_col, rng):
    if len(hospital_df) < n:
        raise ValueError(
            f"Hospital {hospital_df['hospitalid'].iloc[0]} has "
            f"{len(hospital_df)} eligible stays, fewer than requested n={n}."
        )

    if len(hospital_df) == n:
        return hospital_df.copy()

    counts = hospital_df[label_col].value_counts().sort_index()
    allocation = proportional_allocation(counts, n)

    selected_indices = []

    for label_value, class_n in allocation.items():
        class_indices = hospital_df.index[
            hospital_df[label_col].eq(label_value)
        ].to_numpy()

        chosen = rng.choice(class_indices, size=int(class_n), replace=False)
        selected_indices.extend(chosen.tolist())

    return hospital_df.loc[selected_indices].copy()


def summarize_by_hospital(df, prefix):
    summary = (
        df.groupby("hospitalid", as_index=False)
        .agg(
            n_stays=("patientunitstayid", "size"),
            n_positive=("mortality_LABEL", "sum"),
        )
        .sort_values("hospitalid")
        .reset_index(drop=True)
    )

    summary["n_negative"] = summary["n_stays"] - summary["n_positive"]
    summary["mortality_prevalence"] = summary["n_positive"] / summary["n_stays"]
    summary.insert(0, "cohort", prefix)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build the adult eICU mortality-48h cohort before "
            "extraction/FIDDLE. Optionally create a separate "
            "outcome-stratified per-hospital sample."
        )
    )

    parser.add_argument(
        "--patient-file",
        type=Path,
        default=Path(str(EICU_ROOT / "patient.csv.gz")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(PORTFOLIO_ROOT / "data/cohort/adult_mortality_48h_all")),
    )
    parser.add_argument("--minimum-age", type=float, default=18)
    parser.add_argument("--prediction-hours", type=float, default=48)
    parser.add_argument(
        "--sample-per-hospital",
        type=int,
        default=None,
        help=(
            "Optional fixed sample size per hospital. "
            "Do not set this for the full FIDDLE cohort."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.patient_file}", flush=True)

    patient = pd.read_csv(args.patient_file, compression="infer", low_memory=False)

    missing_columns = REQUIRED_COLUMNS - set(patient.columns)
    if missing_columns:
        raise ValueError(f"Missing required patient columns: {sorted(missing_columns)}")

    if patient["patientunitstayid"].duplicated().any():
        duplicated = patient.loc[
            patient["patientunitstayid"].duplicated(keep=False),
            "patientunitstayid",
        ]
        raise ValueError(
            f"patientunitstayid is not unique. Examples: {duplicated.head(10).tolist()}"
        )

    audit_rows = []

    def record(stage, frame):
        audit_rows.append(
            {
                "stage": stage,
                "n_stays": len(frame),
                "n_hospitals": frame["hospitalid"].nunique(),
            }
        )
        print(
            f"{stage}: stays={len(frame):,}; hospitals={frame['hospitalid'].nunique():,}",
            flush=True,
        )

    cohort = patient.copy()
    record("raw_patient_table", cohort)

    cohort["age_original"] = cohort["age"]
    cohort["age"] = parse_age(cohort["age"])

    n_age_missing = int(cohort["age"].isna().sum())
    n_minors = int(cohort["age"].lt(args.minimum_age).sum())

    print(f"Unparseable/missing age: {n_age_missing:,}", flush=True)
    print(f"Age < {args.minimum_age:g}: {n_minors:,}", flush=True)

    cohort = cohort.loc[cohort["age"].ge(args.minimum_age)].copy()
    record("adult_age_ge_18", cohort)

    cohort["hospitalid"] = pd.to_numeric(cohort["hospitalid"], errors="coerce")
    cohort = cohort.dropna(subset=["hospitalid"]).copy()
    cohort["hospitalid"] = cohort["hospitalid"].astype(int)
    record("known_hospital", cohort)

    cohort["unitdischargeoffset"] = pd.to_numeric(
        cohort["unitdischargeoffset"], errors="coerce"
    )

    minimum_offset = args.prediction_hours * 60
    cohort = cohort.loc[cohort["unitdischargeoffset"].ge(minimum_offset)].copy()
    record(f"icu_stay_at_least_{args.prediction_hours:g}h", cohort)

    cohort["mortality_LABEL"] = make_mortality_label(cohort["hospitaldischargestatus"])
    cohort = cohort.loc[cohort["mortality_LABEL"].notna()].copy()
    cohort["mortality_LABEL"] = cohort["mortality_LABEL"].astype(int)
    record("known_hospital_mortality", cohort)

    cohort = cohort.sort_values(["hospitalid", "patientunitstayid"]).reset_index(
        drop=True
    )

    cohort_path = args.output_dir / "adult_mortality_48h_cohort.csv.gz"
    ids_path = args.output_dir / "adult_mortality_48h_ids.csv"
    population_path = args.output_dir / "adult_mortality_48h_population.csv"
    summary_path = args.output_dir / "adult_mortality_48h_hospital_summary.csv"
    audit_path = args.output_dir / "adult_mortality_48h_filter_audit.csv"

    cohort.to_csv(cohort_path, index=False, compression="gzip")
    cohort[["patientunitstayid", "hospitalid"]].to_csv(ids_path, index=False)
    cohort[["patientunitstayid", "hospitalid", "age", "mortality_LABEL"]].rename(
        columns={"patientunitstayid": "ID"}
    ).to_csv(population_path, index=False)

    full_summary = summarize_by_hospital(cohort, "full_adult_eligible")
    full_summary.to_csv(summary_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print("\nFull adult cohort saved:", flush=True)
    print(cohort_path, flush=True)
    print(ids_path, flush=True)
    print(population_path, flush=True)
    print(summary_path, flush=True)
    print(audit_path, flush=True)

    print(
        f"\nFinal cohort: n={len(cohort):,}; "
        f"positive={cohort['mortality_LABEL'].sum():,}; "
        f"prevalence={cohort['mortality_LABEL'].mean():.4f}",
        flush=True,
    )

    if args.sample_per_hospital is None:
        print(
            "\nNo fixed-size sampling requested. "
            "Use the full adult cohort for extraction/FIDDLE.",
            flush=True,
        )
        return

    eligible_hospitals = full_summary.loc[
        full_summary["n_stays"].ge(args.sample_per_hospital), "hospitalid"
    ].tolist()

    sampling_pool = cohort.loc[cohort["hospitalid"].isin(eligible_hospitals)].copy()

    rng = np.random.default_rng(args.random_state)
    sampled_groups = []

    for hospitalid, hospital_df in sampling_pool.groupby("hospitalid", sort=True):
        sampled = stratified_sample_hospital(
            hospital_df=hospital_df,
            n=args.sample_per_hospital,
            label_col="mortality_LABEL",
            rng=rng,
        )
        sampled_groups.append(sampled)

        print(
            f"Stratified sample hospital={hospitalid}: "
            f"n={len(sampled):,}; "
            f"positive={sampled['mortality_LABEL'].sum():,}; "
            f"prevalence={sampled['mortality_LABEL'].mean():.4f}",
            flush=True,
        )

    sampled_cohort = (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["hospitalid", "patientunitstayid"])
        .reset_index(drop=True)
    )

    sample_stem = f"adult_mortality_48h_stratified_n{args.sample_per_hospital}"

    sampled_path = args.output_dir / f"{sample_stem}.csv.gz"
    sampled_ids_path = args.output_dir / f"{sample_stem}_ids.csv"
    sampled_summary_path = args.output_dir / f"{sample_stem}_hospital_summary.csv"

    sampled_cohort.to_csv(sampled_path, index=False, compression="gzip")
    sampled_cohort[["patientunitstayid", "hospitalid"]].to_csv(
        sampled_ids_path, index=False
    )

    sampled_summary = summarize_by_hospital(
        sampled_cohort, f"stratified_n{args.sample_per_hospital}"
    )
    comparison = full_summary.merge(
        sampled_summary, on="hospitalid", how="inner", suffixes=("_full", "_sample")
    )
    comparison.to_csv(sampled_summary_path, index=False)

    print("\nOptional stratified sample saved:", flush=True)
    print(sampled_path, flush=True)
    print(sampled_ids_path, flush=True)
    print(sampled_summary_path, flush=True)


if __name__ == "__main__":
    main()
