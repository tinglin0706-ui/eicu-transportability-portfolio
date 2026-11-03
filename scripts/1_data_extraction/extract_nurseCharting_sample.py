"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


CONFIG = {
    "n_rows": {
        "nurseCharting": 151_604_232,
    }
}


def load_sample_ids(sample_ids_path):
    cohort = pd.read_csv(sample_ids_path)
    ids = set(cohort["patientunitstayid"].dropna().astype(int))
    print(f"Loaded {len(ids)} sampled patientunitstayid values")
    return ids


def read_events(eicu_path, fname, chunksize):
    n_rows = CONFIG["n_rows"][fname]
    total = n_rows // chunksize + 1

    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    with tqdm(desc=fname, total=total) as pbar:
        for df in pd.read_csv(input_file, chunksize=chunksize):
            pbar.update()
            yield df


def clean_nurse_charting_chunk(df, sample_ids):

    df = df[df["patientunitstayid"].isin(sample_ids)]

    if df.empty:
        return None

    drop_cols = [
        "nursingchartid",
        "nursingchartentryoffset",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.rename(
        columns={
            "patientunitstayid": "ID",
            "nursingchartoffset": "t",
        }
    )

    name_cols = [
        "nursingchartcelltypecat",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
    ]

    df[name_cols] = df[name_cols].fillna("missing").astype(str)

    df["variable_name"] = df[name_cols].agg("|".join, axis=1)

    raw_value = df["nursingchartvalue"]
    numeric_value = pd.to_numeric(raw_value, errors="coerce")
    mixed_value = raw_value.where(numeric_value.isna(), numeric_value)

    df["variable_value"] = mixed_value.astype("string").fillna("missing")

    df = df[["ID", "t", "variable_name", "variable_value"]].copy()

    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    df = df.dropna(subset=["ID", "t"])

    if df.empty:
        return None

    df["ID"] = df["ID"].astype("int64")
    df["t"] = df["t"].astype("int64")
    df["variable_name"] = df["variable_name"].astype("string")
    df["variable_value"] = df["variable_value"].astype("string")

    return df.reset_index(drop=True)


def extract_nurse_charting(eicu_path, save_path, sample_ids_path, chunksize):
    os.makedirs(save_path, exist_ok=True)

    sample_ids = load_sample_ids(sample_ids_path)

    parquet_path = os.path.join(save_path, "nurseCharting.parquet")
    tmp_path = os.path.join(save_path, "nurseCharting.parquet.tmp")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    schema = pa.schema(
        [
            ("ID", pa.int64()),
            ("t", pa.int64()),
            ("variable_name", pa.string()),
            ("variable_value", pa.string()),
        ]
    )

    writer = None
    n_rows_saved = 0
    unique_ids = set()
    n_chunks_written = 0

    try:
        for df in read_events(eicu_path, "nurseCharting", chunksize):
            df_clean = clean_nurse_charting_chunk(df, sample_ids)

            if df_clean is None or df_clean.empty:
                continue

            table = pa.Table.from_pandas(
                df_clean,
                schema=schema,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema=schema,
                    compression="snappy",
                )

            writer.write_table(table)

            n_rows_saved += len(df_clean)
            unique_ids.update(df_clean["ID"].unique().tolist())
            n_chunks_written += 1

            print(
                f"Wrote chunk {n_chunks_written}: "
                f"rows_this_chunk={len(df_clean)}, "
                f"total_rows_saved={n_rows_saved}, "
                f"unique_ids_so_far={len(unique_ids)}",
                flush=True,
            )

    finally:
        if writer is not None:
            writer.close()

    if n_rows_saved == 0:
        print("No nurseCharting rows found for sampled IDs.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    os.replace(tmp_path, parquet_path)

    print(f"Saved parquet: {parquet_path}")
    print(f"Rows saved: {n_rows_saved}")
    print(f"Unique IDs in nurseCharting output: {len(unique_ids)}")
    print(f"Chunks written: {n_chunks_written}")


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
        default=500000,
    )

    args = parser.parse_args()

    extract_nurse_charting(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids_path=args.sample_ids_path,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
