"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


CONFIG = {
    "n_rows": {
        "respiratoryCare": None,
        "intakeOutput": None,
    }
}


def load_sample_ids(sample_ids_path):
    cohort = pd.read_csv(sample_ids_path)
    ids = set(cohort["patientunitstayid"].dropna().astype(int))
    print(f"Loaded {len(ids)} sampled patientunitstayid values")
    return ids


def read_table_chunks(eicu_path, fname, chunksize):
    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    for df in tqdm(pd.read_csv(input_file, chunksize=chunksize), desc=fname):
        yield df


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

    print(f"{fname}: rows={len(df)}")
    print(f"{fname}: unique IDs={df['ID'].nunique() if 'ID' in df.columns else 'NA'}")


def extract_respiratory_care(eicu_path, save_path, sample_ids, chunksize):
    fname = "respiratoryCare"
    chunks = []

    for df in read_table_chunks(eicu_path, fname, chunksize):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        for c in [
            "respcarestatusoffset",
            "ventstartoffset",
            "ventendoffset",
            "priorventstartoffset",
            "priorventendoffset",
        ]:
            if c in df.columns:
                df[c] = df[c].replace({0: np.nan})

        drop_cols = [
            "respcareid",
            "currenthistoryseqnum",
            "ventendoffset",
            "priorventendoffset",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "respcarestatusoffset": "t",
            }
        )

        df = df.set_index(["ID", "t"])
        df.columns.name = "variable_name"
        df = df.stack(dropna=True)
        df.name = "variable_value"
        df = df.reset_index()

        is_vent_start = df["variable_name"].eq("ventstartoffset")
        valid_vent_start = df["t"] >= pd.to_numeric(
            df["variable_value"], errors="coerce"
        )
        df = df[(~is_vent_start) | (is_vent_start & valid_vent_start)]

        df["variable_value"] = pd.to_numeric(df["variable_value"], errors="ignore")
        df = df[["ID", "t", "variable_name", "variable_value"]]

        chunks.append(df.reset_index(drop=True))

    if len(chunks) == 0:
        print("No respiratoryCare rows found for sampled IDs.")
        return

    df_out = pd.concat(chunks, ignore_index=True)
    save_df(df_out, save_path, fname)


def extract_intake_output(eicu_path, save_path, sample_ids, chunksize):
    fname = "intakeOutput"
    chunks = []

    for df in read_table_chunks(eicu_path, fname, chunksize):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        drop_cols = [
            "intakeoutputid",
            "intakeoutputentryoffset",
            "celllabel",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "intakeoutputoffset": "t",
            }
        )

        total_cols = [
            "intaketotal",
            "outputtotal",
            "dialysistotal",
            "nettotal",
        ]
        total_cols = [c for c in total_cols if c in df.columns]

        pieces = []

        if total_cols:
            df_totals = df[["ID", "t"] + total_cols].copy()
            df_totals = df_totals.set_index(["ID", "t"])
            df_totals.columns.name = "variable_name"
            df_totals = df_totals.stack(dropna=True)
            df_totals.name = "variable_value"
            df_totals = df_totals.reset_index()
            pieces.append(df_totals)

        if {"cellpath", "cellvaluetext"}.issubset(df.columns):
            df_cells = df[
                ["ID", "t", "cellpath", "cellvaluenumeric", "cellvaluetext"]
            ].copy()
            df_cells = df_cells.rename(columns={"cellpath": "variable_name"})

            df_cells["variable_value"] = pd.to_numeric(
                df_cells["cellvaluetext"],
                errors="ignore",
            )

            has_numeric = "cellvaluenumeric" in df_cells.columns
            if has_numeric:
                missing_text = df_cells["cellvaluetext"].isna()
                df_cells.loc[missing_text, "variable_value"] = df_cells.loc[
                    missing_text, "cellvaluenumeric"
                ]

            df_cells = df_cells[["ID", "t", "variable_name", "variable_value"]]
            df_cells = df_cells.dropna(subset=["variable_name", "variable_value"])
            pieces.append(df_cells)

        if not pieces:
            continue

        df_out = pd.concat(pieces, ignore_index=True)
        df_out = df_out[["ID", "t", "variable_name", "variable_value"]]
        chunks.append(df_out.reset_index(drop=True))

    if len(chunks) == 0:
        print("No intakeOutput rows found for sampled IDs.")
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

    extract_respiratory_care(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids=sample_ids,
        chunksize=args.chunksize,
    )

    extract_intake_output(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids=sample_ids,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
