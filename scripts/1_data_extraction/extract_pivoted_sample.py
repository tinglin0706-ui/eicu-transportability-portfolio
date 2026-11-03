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
        "vitalPeriodic": 146_671_642,
        "vitalAperiodic": 25_075_074,
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


def clean_pivoted_chunk(df, sample_ids):
    df = df[df["patientunitstayid"].isin(sample_ids)]

    if df.empty:
        return None

    df = df.set_index(["patientunitstayid", "observationoffset"])

    drop_cols = [
        c for c in df.columns if c.lower().endswith("id") and c != "patientunitstayid"
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.columns.name = "variable_name"

    df = df.stack(future_stack=True)
    df.name = "variable_value"
    df = df.reset_index()

    df = df.rename(
        columns={
            "patientunitstayid": "ID",
            "observationoffset": "t",
        }
    )

    df["variable_value"] = pd.to_numeric(
        df["variable_value"],
        errors="coerce",
    )

    df = df.dropna(subset=["variable_value"])
    df = df[["ID", "t", "variable_name", "variable_value"]]

    if df.empty:
        return None

    df["ID"] = df["ID"].astype("int64")
    df["t"] = df["t"].astype("int64")
    df["variable_name"] = df["variable_name"].astype("string")
    df["variable_value"] = df["variable_value"].astype("float64")

    return df.reset_index(drop=True)


def stream_filter_to_parquet(eicu_path, fname, sample_ids, chunksize, tmp_path):
    """
    Stage 1: scan the full (huge) source table chunk by chunk, keep only rows
    for sampled patients, write each chunk straight to disk. Memory use here
    stays roughly constant regardless of how many chunks we've read, unlike
    accumulating every chunk's output in a Python list.
    """
    schema = pa.schema(
        [
            ("ID", pa.int64()),
            ("t", pa.int64()),
            ("variable_name", pa.string()),
            ("variable_value", pa.float64()),
        ]
    )

    writer = None
    n_rows_written = 0

    try:
        for df in read_events(eicu_path, fname, chunksize):
            df_clean = clean_pivoted_chunk(df, sample_ids)

            if df_clean is None or df_clean.empty:
                continue

            table = pa.Table.from_pandas(df_clean, schema=schema, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(tmp_path, schema=schema, compression="snappy")

            writer.write_table(table)
            n_rows_written += len(df_clean)
    finally:
        if writer is not None:
            writer.close()

    return n_rows_written


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
    print(f"{fname}: unique IDs={df['ID'].nunique()}")


def extract_pivoted(eicu_path, save_path, sample_ids_path, fname, chunksize):
    if fname not in CONFIG["n_rows"]:
        raise ValueError(
            f"Unknown filename: {fname}. Expected one of {list(CONFIG['n_rows'].keys())}"
        )

    os.makedirs(save_path, exist_ok=True)
    sample_ids = load_sample_ids(sample_ids_path)

    tmp_path = os.path.join(save_path, f"{fname}_prededup.parquet.tmp")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"Stage 1/2: streaming filter of {fname} to {tmp_path}", flush=True)
    n_rows_written = stream_filter_to_parquet(
        eicu_path, fname, sample_ids, chunksize, tmp_path
    )

    if n_rows_written == 0:
        print(f"No {fname} rows found for sampled IDs.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    print(
        f"Stage 1/2 complete: {n_rows_written:,} rows written before dedup.",
        flush=True,
    )

    print(f"Stage 2/2: loading filtered rows for dedup ({tmp_path})", flush=True)
    df_out = pd.read_parquet(tmp_path)

    df_out = df_out.groupby(["ID", "t", "variable_name"], as_index=False)[
        "variable_value"
    ].median()

    os.remove(tmp_path)

    save_df(df_out, save_path, fname)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename",
        choices=["vitalPeriodic", "vitalAperiodic"],
        help="Which pivoted vital table to extract",
    )

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

    extract_pivoted(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids_path=args.sample_ids_path,
        fname=args.filename,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
