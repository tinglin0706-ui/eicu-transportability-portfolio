"""Project-specific eICU event extraction adapted from the FIDDLE experiment workflow."""

from pathlib import Path


import argparse
import os
import pandas as pd
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

    print(f"{fname}: rows={len(df)}")
    print(f"{fname}: unique IDs={df['ID'].nunique()}")


def extract_infusion_drug(eicu_path, save_path, sample_ids_path, chunksize):
    fname = "infusionDrug"
    input_file = os.path.join(eicu_path, f"{fname}.csv.gz")

    sample_ids = load_sample_ids(sample_ids_path)
    chunks = []

    for df in tqdm(pd.read_csv(input_file, chunksize=chunksize), desc=fname):
        df = df[df["patientunitstayid"].isin(sample_ids)]

        if df.empty:
            continue

        df = df.drop(columns=[c for c in ["infusiondrugid"] if c in df.columns])

        df = df.dropna(subset=["drugname"], how="all")

        if df.empty:
            continue

        df = df.rename(
            columns={
                "patientunitstayid": "ID",
                "infusionoffset": "t",
            }
        )

        value_cols = [
            "drugrate",
            "infusionrate",
            "drugamount",
            "volumeoffluid",
            "patientweight",
        ]

        existing_value_cols = [c for c in value_cols if c in df.columns]

        df["drugname"] = df["drugname"].fillna("missing").astype(str)

        df = df.set_index(["ID", "t", "drugname"])[existing_value_cols]
        df.columns.name = "property"

        df = df.stack(dropna=True)
        df.name = "variable_value"
        df = df.reset_index()

        df["variable_name"] = df[["drugname", "property"]].apply(
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
        print("No infusionDrug rows found for sampled IDs.")
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

    extract_infusion_drug(
        eicu_path=args.eicu_path,
        save_path=args.save_path,
        sample_ids_path=args.sample_ids_path,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
