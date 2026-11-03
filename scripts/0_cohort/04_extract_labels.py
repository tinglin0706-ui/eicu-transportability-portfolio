"""Export mortality labels from the final hospital-filtered cohort."""


import argparse
from pathlib import Path

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract a slim ID/hospitalid/label file from a cohort file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            str(
                PORTFOLIO_ROOT
                / "data/cohort/adult_mortality_48h_final/adult_mortality_48h_final_cohort.csv.gz"
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            str(PORTFOLIO_ROOT / "data/labels/adult_mortality_48h_labels.csv")
        ),
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    cohort = pd.read_csv(
        args.input,
        usecols=[
            "patientunitstayid",
            "hospitalid",
            "age",
            "unitdischargeoffset",
            "mortality_LABEL",
        ],
    )

    assert cohort["patientunitstayid"].notna().all()
    assert cohort["patientunitstayid"].is_unique
    assert cohort["hospitalid"].notna().all()
    assert cohort["age"].ge(18).all()
    assert cohort["unitdischargeoffset"].ge(48 * 60).all()
    assert cohort["mortality_LABEL"].isin([0, 1]).all()

    labels = (
        cohort[["patientunitstayid", "hospitalid", "mortality_LABEL"]]
        .rename(columns={"patientunitstayid": "ID"})
        .sort_values(["hospitalid", "ID"])
        .reset_index(drop=True)
    )

    labels["ID"] = labels["ID"].astype(int)
    labels["hospitalid"] = labels["hospitalid"].astype(int)
    labels["mortality_LABEL"] = labels["mortality_LABEL"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Stays: {len(labels):,}")
    print(f"Hospitals: {labels['hospitalid'].nunique():,}")
    print(f"Positive labels: {labels['mortality_LABEL'].sum():,}")
    print(f"Mortality prevalence: {labels['mortality_LABEL'].mean():.2%}")
    print("\nLabel counts:")
    print(labels["mortality_LABEL"].value_counts().sort_index().to_string())
    print("\nLABEL EXTRACTION PASSED")


if __name__ == "__main__":
    main()
