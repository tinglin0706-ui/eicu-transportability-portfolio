"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
import pandas as pd
from tqdm import tqdm


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
EICU_ROOT = Path(os.environ.get("EICU_DATA_DIR", str(PORTFOLIO_ROOT / "data" / "raw")))


CONFIG = {
    "n_rows": {
        "medication": 7_301_853,
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


def extract_medication(eicu_path, save_path, sample_ids_path, chunksize):
    os.makedirs(save_path, exist_ok=True)

    sample_ids = load_sample_ids(sample_ids_path)
    chunks = []

    for df in read_events(eicu_path, "medication", chunksize):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        df["drughiclseqno"] = df["drughiclseqno"].astype("Int64")
        df = df.dropna(subset=["drugname", "drughiclseqno"], how="all")

        if df.empty:
            continue

        df.loc[:, "drugnameid"] = df[["drugname", "drughiclseqno"]].apply(
            lambda x: "{}|{}".format(x.iloc[0], x.iloc[1]),
            axis=1,
        )

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "drugstartoffset": "t",
            }
        )

        df = df.set_index(["ID", "t", "drugnameid"])[
            ["dosage", "routeadmin", "frequency"]
        ]

        df.columns.name = "property"
        df = df.stack(dropna=True)
        df.name = "variable_value"
        df = df.reset_index()

        df["variable_name"] = df[["drugnameid", "property"]].apply(
            lambda x: "|".join(x.astype(str)),
            axis=1,
        )

        df["variable_value"] = pd.to_numeric(
            df["variable_value"],
            errors="ignore",
        )

        df = df[["ID", "t", "variable_name", "variable_value"]]
        chunks.append(df.reset_index(drop=True))

    if len(chunks) == 0:
        print("No medication rows found for sampled IDs.")
        return

    df_out = pd.concat(chunks, ignore_index=True)

    parquet_path = os.path.join(save_path, "medication.parquet")
    pickle_path = os.path.join(save_path, "medication.pickle")

    try:
        df_out.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path}")
    except Exception as e:
        print(f"Could not save parquet: {e}")
        df_out.to_pickle(pickle_path)
        print(f"Saved pickle: {pickle_path}")

    print(f"Rows saved: {len(df_out)}")
    print(f"Unique IDs in medication output: {df_out['ID'].nunique()}")


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
        default=100000,
    )

    args = parser.parse_args()

    extract_medication(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids_path=args.sample_ids_path,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
