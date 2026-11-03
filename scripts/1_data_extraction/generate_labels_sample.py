"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


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


def make_icustays_and_mortality(eicu_path, sample_ids_path, labels_path):
    patient_path = os.path.join(eicu_path, "patient.csv.gz")

    cohort = pd.read_csv(sample_ids_path)
    sample_ids = set(cohort["patientunitstayid"].astype(int))

    df_P = pd.read_csv(patient_path)
    df_P = df_P[df_P["patientunitstayid"].isin(sample_ids)].copy()

    print("Sample patient rows:", len(df_P))

    df_ID = df_P[
        [
            "uniquepid",
            "patienthealthsystemstayid",
            "patientunitstayid",
            "hospitalid",
            "gender",
            "age",
        ]
    ].copy()

    df_ID = df_ID.rename(
        columns={
            "uniquepid": "PatientID",
            "patienthealthsystemstayid": "AdmissionID",
            "patientunitstayid": "ICUStayID",
        }
    )

    df_ID["age"] = parse_age(df_ID["age"])

    patients = df_ID["PatientID"].dropna().unique()

    if len(patients) >= 3:
        sss1 = ShuffleSplit(1, test_size=0.3, random_state=0)
        sss2 = ShuffleSplit(1, test_size=0.5, random_state=0)

        train_idx, val_test_idx = next(sss1.split(patients, patients))
        patients_val_test = patients[val_test_idx]

        val_idx_inner, test_idx_inner = next(
            sss2.split(patients_val_test, patients_val_test)
        )

        val_idx = val_test_idx[val_idx_inner]
        test_idx = val_test_idx[test_idx_inner]

        df_ID["partition"] = "null"
        df_ID.loc[df_ID["PatientID"].isin(patients[train_idx]), "partition"] = "train"
        df_ID.loc[df_ID["PatientID"].isin(patients[val_idx]), "partition"] = "val"
        df_ID.loc[df_ID["PatientID"].isin(patients[test_idx]), "partition"] = "test"

        c = Counter(df_ID["partition"])
        print(
            "Patient splits:",
            len(train_idx),
            len(val_idx),
            len(test_idx),
            "/",
            len(patients),
        )
        print("ICU stay splits:", c["train"], c["val"], c["test"], "/", len(df_ID))
    else:
        df_ID["partition"] = "train"
        print("Too few patients for train/val/test split; assigned all to train.")

    icustays_path = os.path.join(labels_path, "icustays.csv")
    df_ID.to_csv(icustays_path, index=False, quoting=2)
    print(f"Saved {icustays_path}")

    df_mort = df_P[
        [
            "patientunitstayid",
            "unitdischargeoffset",
            "hospitaldischargestatus",
        ]
    ].copy()

    df_mort = df_mort.rename(
        columns={
            "patientunitstayid": "ID",
            "unitdischargeoffset": "ICUDischargeTime",
        }
    )

    status = df_mort["hospitaldischargestatus"].astype("string").str.strip().str.lower()
    df_mort["mortality_LABEL"] = status.map({"alive": 0, "expired": 1}).astype("Int64")

    mortality_path = os.path.join(labels_path, "mortality.csv")
    df_mort.to_csv(mortality_path, index=False)
    print(f"Saved {mortality_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eicu_path",
        default=str(EICU_ROOT),
    )

    parser.add_argument(
        "--sample_ids_path",
        default=str(PORTFOLIO_ROOT / "data/cohort/sample_1000_ids.csv"),
    )

    parser.add_argument(
        "--labels_path",
        default=str(PORTFOLIO_ROOT / "data/labels/final_1000_0"),
    )

    args = parser.parse_args()

    os.makedirs(args.labels_path, exist_ok=True)

    make_icustays_and_mortality(
        eicu_path=args.eicu_path,
        sample_ids_path=args.sample_ids_path,
        labels_path=args.labels_path,
    )


if __name__ == "__main__":
    main()
