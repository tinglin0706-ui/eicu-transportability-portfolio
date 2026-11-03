"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


def load_sample_ids(sample_ids_path):
    cohort = pd.read_csv(sample_ids_path)
    ids = set(cohort["patientunitstayid"].dropna().astype(int))
    print(f"Loaded {len(ids)} sampled patientunitstayid values")
    return ids


def save_df(df, save_path, fname):
    os.makedirs(save_path, exist_ok=True)

    parquet_path = os.path.join(save_path, f"{fname}.parquet")
    pickle_path = os.path.join(save_path, f"{fname}.pickle")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path}")
    except Exception as e:
        print(f"Could not save parquet for {fname}: {e}")
        df.to_pickle(pickle_path)
        print(f"Saved pickle: {pickle_path}")

    print(
        f"{fname}: rows={len(df)}, unique IDs={df['ID'].nunique() if 'ID' in df.columns else 'NA'}"
    )


def extract_patient(eicu_path, save_path, sample_ids):
    fname = "patient"
    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    cols = [
        "patientunitstayid",
        "gender",
        "age",
        "ethnicity",
        "hospitalid",
        "wardid",
        "apacheadmissiondx",
        "admissionheight",
        "hospitaladmittime24",
        "hospitaladmitoffset",
        "hospitaladmitsource",
        "unittype",
        "unitadmittime24",
        "unitadmitsource",
        "unitvisitnumber",
        "unitstaytype",
        "admissionweight",
        "uniquepid",
    ]

    df = pd.read_csv(input_file, usecols=lambda c: c in cols)
    df = df[df["patientunitstayid"].isin(sample_ids)].copy()

    df = df.rename(columns={"patientunitstayid": "ID"})
    df = df.set_index("ID")
    df.columns.name = "variable_name"

    df = df.stack(dropna=True)
    df.name = "variable_value"
    df = df.reset_index()

    df["t"] = np.nan
    df = df[["ID", "t", "variable_name", "variable_value"]]

    save_df(df, save_path, fname)


def extract_lab(eicu_path, save_path, sample_ids, chunksize):
    fname = "lab"
    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    chunks = []

    for df in tqdm(pd.read_csv(input_file, chunksize=chunksize), desc=fname):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        if "labresulttext" in df.columns:
            value_col = "labresulttext"
        elif "labresult" in df.columns:
            value_col = "labresult"
        elif "labothervaluetext" in df.columns:
            value_col = "labothervaluetext"
        else:
            raise ValueError("No recognizable lab value column found.")

        df = df[["patientunitstayid", "labresultoffset", "labname", value_col]].copy()

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "labresultoffset": "t",
                "labname": "variable_name",
                value_col: "variable_value",
            }
        )

        df["variable_value"] = pd.to_numeric(df["variable_value"], errors="ignore")
        df = df[["ID", "t", "variable_name", "variable_value"]]

        chunks.append(df.reset_index(drop=True))

    if len(chunks) == 0:
        print("No lab rows found for sampled IDs.")
        return

    df_out = pd.concat(chunks, ignore_index=True)
    save_df(df_out, save_path, fname)


def extract_past_history(eicu_path, save_path, sample_ids, chunksize):
    fname = "pastHistory"
    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    chunks = []

    for df in tqdm(pd.read_csv(input_file, chunksize=chunksize), desc=fname):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        df = df[["patientunitstayid", "pasthistoryoffset", "pasthistorypath"]].copy()

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "pasthistoryoffset": "t",
                "pasthistorypath": "variable_name",
            }
        )

        df["variable_value"] = 1
        df = df[["ID", "t", "variable_name", "variable_value"]]

        chunks.append(df.reset_index(drop=True))

    if len(chunks) == 0:
        print("No pastHistory rows found for sampled IDs.")
        return

    df_out = pd.concat(chunks, ignore_index=True)
    save_df(df_out, save_path, fname)


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
        "--save_path",
        default=str(PORTFOLIO_ROOT / "data/extracted/final_1000_0"),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000000,
    )

    args = parser.parse_args()

    sample_ids = load_sample_ids(args.sample_ids_path)

    extract_patient(args.eicu_path, args.save_path, sample_ids)
    extract_lab(args.eicu_path, args.save_path, sample_ids, args.chunksize)
    extract_past_history(args.eicu_path, args.save_path, sample_ids, args.chunksize)


if __name__ == "__main__":
    main()
