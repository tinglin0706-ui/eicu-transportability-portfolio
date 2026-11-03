"""Validate cohort eligibility and create mortality population files for FIDDLE."""

from pathlib import Path


import argparse
import os

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


def parse_age(series):
    """Convert age values to numeric; encode censored ages above 89 as 90."""
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


def make_population_mortality(
    labels_path,
    population_path,
    prediction_time_hours,
):
    os.makedirs(population_path, exist_ok=True)

    icustays_path = os.path.join(labels_path, "icustays.csv")
    mortality_path = os.path.join(labels_path, "mortality.csv")

    examples = pd.read_csv(icustays_path)
    labels = pd.read_csv(mortality_path)

    print("Original population:", examples["ICUStayID"].nunique())

    examples["age"] = parse_age(examples["age"])

    if examples["age"].isna().any() or not examples["age"].ge(18).all():
        bad = examples.loc[
            examples["age"].isna() | examples["age"].lt(18),
            "ICUStayID",
        ]
        raise ValueError(
            "Found stays with missing/non-adult age. This should have been "
            "filtered upstream by 01_build_cohort.py. Example ICUStayIDs: "
            f"{bad.head(10).tolist()}"
        )

    examples = examples.rename(columns={"ICUStayID": "ID"})
    examples["ID"] = examples["ID"].astype(int)
    labels["ID"] = labels["ID"].astype(int)

    examples = examples.merge(
        labels[
            ["ID", "ICUDischargeTime", "hospitaldischargestatus", "mortality_LABEL"]
        ],
        on="ID",
        how="inner",
        validate="one_to_one",
    )

    if not examples["mortality_LABEL"].isin([0.0, 1.0]).all():
        bad = examples.loc[
            ~examples["mortality_LABEL"].isin([0.0, 1.0]),
            "ID",
        ]
        raise ValueError(
            "Found stays with unknown mortality_LABEL. This should have been "
            "filtered upstream. Example IDs: "
            f"{bad.head(10).tolist()}"
        )

    cutoff_min = prediction_time_hours * 60

    examples["ICUDischargeTime"] = pd.to_numeric(
        examples["ICUDischargeTime"],
        errors="coerce",
    )

    if (
        examples["ICUDischargeTime"].isna().any()
        or not examples["ICUDischargeTime"].ge(cutoff_min).all()
    ):
        bad = examples.loc[
            examples["ICUDischargeTime"].isna()
            | examples["ICUDischargeTime"].lt(cutoff_min),
            "ID",
        ]
        raise ValueError(
            f"Found stays with ICUDischargeTime below the {prediction_time_hours}h "
            "cutoff. This should have been filtered upstream. Example IDs: "
            f"{bad.head(10).tolist()}"
        )

    print(
        "Population after eligibility checks "
        "(age/label/48h -- all pre-validated, not re-filtered):",
        examples["ID"].nunique(),
    )

    if examples["ID"].duplicated().any():
        duplicated_ids = examples.loc[
            examples["ID"].duplicated(keep=False),
            "ID",
        ].unique()

        raise ValueError(
            "Duplicate IDs found in final mortality population. "
            f"Example duplicated IDs: {duplicated_ids[:10].tolist()}"
        )

    pop = examples[["ID", "mortality_LABEL"]].copy()
    pop["mortality_LABEL"] = pop["mortality_LABEL"].astype(int)

    out_path = os.path.join(
        population_path,
        f"mortality_{int(prediction_time_hours)}h.csv",
    )

    pop.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print("Final population rows:", len(pop))
    print("Unique final IDs:", pop["ID"].nunique())
    print("Mortality positives:", int(pop["mortality_LABEL"].sum()))
    print(
        "Mortality negatives:",
        int((pop["mortality_LABEL"] == 0).sum()),
    )
    print("Mortality prevalence:", pop["mortality_LABEL"].mean())

    audit_cols = [
        "ID",
        "mortality_LABEL",
        "PatientID",
        "AdmissionID",
        "hospitalid",
        "gender",
        "age",
        "partition",
        "ICUDischargeTime",
        "hospitaldischargestatus",
    ]

    existing_audit_cols = [
        column for column in audit_cols if column in examples.columns
    ]

    audit = examples[existing_audit_cols].copy()

    audit_path = os.path.join(
        population_path,
        f"mortality_{int(prediction_time_hours)}h_with_metadata.csv",
    )

    audit.to_csv(audit_path, index=False)
    print(f"Saved audit file {audit_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_path",
        default=(str(PORTFOLIO_ROOT / "data/labels/final_1000_0")),
    )

    parser.add_argument(
        "--population_path",
        default=(str(PORTFOLIO_ROOT / "data/population/final_1000_0")),
    )

    parser.add_argument(
        "--prediction_time_hours",
        type=float,
        default=48.0,
    )

    args = parser.parse_args()

    make_population_mortality(
        labels_path=args.labels_path,
        population_path=args.population_path,
        prediction_time_hours=args.prediction_time_hours,
    )


if __name__ == "__main__":
    main()
