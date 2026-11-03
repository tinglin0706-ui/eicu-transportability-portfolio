"""Prepare time-windowed event data and exclude identifier-like variables for FIDDLE."""

from pathlib import Path


import argparse
import os
import pickle
import re

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


BAD_VARIABLE_PATTERNS = [
    "uniquepid",
    "patienthealthsystemstayid",
    "hospitalid",
    "wardid",
    "hospitaladmittime24",
    "unitadmittime24",
    "hospitaldischargetime24",
    "unitdischargetime24",
]


TASK_SPECS = [("mortality_48h", "mortality_48h.csv", 48.0)]


def read_event_file(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    if path.endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported file type: {path}")


def filter_bad_variables(data):
    pattern = "|".join(re.escape(x) for x in BAD_VARIABLE_PATTERNS)

    variable_name = data["variable_name"].astype(str)

    bad_mask = variable_name.str.contains(
        pattern,
        case=False,
        na=False,
        regex=True,
    )

    bad_rows = data.loc[bad_mask].copy()
    clean_data = data.loc[~bad_mask].copy()

    return clean_data, bad_rows


def prepare_one_task(task_name, population_file, T, extracted_path, features_root):
    output_dir = os.path.join(features_root, task_name)
    os.makedirs(output_dir, exist_ok=True)

    pop = pd.read_csv(population_file)
    pop["ID"] = pop["ID"].astype(int)
    pop_ids = set(pop["ID"])

    print("\n======================================")
    print(f"TASK: {task_name}")
    print(f"Population file: {population_file}")
    print(f"Extracted path: {extracted_path}")
    print(f"Output dir: {output_dir}")
    print(f"N IDs: {len(pop_ids)}")
    print(f"T: {T}h")
    print("======================================")

    chunks = []

    for filename in sorted(os.listdir(extracted_path)):
        if not (filename.endswith(".pickle") or filename.endswith(".parquet")):
            continue

        path = os.path.join(extracted_path, filename)
        print(f"Reading {filename}", flush=True)

        df = read_event_file(path)

        required = {"ID", "t", "variable_name", "variable_value"}
        if not required.issubset(df.columns):
            print(f"Skipping {filename}: missing {required - set(df.columns)}")
            continue

        df = df[["ID", "t", "variable_name", "variable_value"]].copy()

        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
        df = df.dropna(subset=["ID"])
        df["ID"] = df["ID"].astype(int)

        df = df[df["ID"].isin(pop_ids)]

        if df.empty:
            continue

        df["t"] = pd.to_numeric(df["t"], errors="coerce")

        df = df[((df["t"] >= 0) & (df["t"] < T * 60)) | (df["t"].isna())].copy()

        if df.empty:
            continue

        df["variable_name"] = df["variable_name"].astype(str)
        df["variable_value"] = df["variable_value"].astype(str)

        chunks.append(df)

    if len(chunks) == 0:
        raise RuntimeError(f"No rows after filtering for {task_name}")

    data = pd.concat(chunks, ignore_index=True)

    n_rows_before = len(data)

    data, bad_rows = filter_bad_variables(data)

    n_bad_rows_removed = len(bad_rows)
    n_rows_after = len(data)

    print(f"Rows before bad feature filtering: {n_rows_before}")
    print(f"Bad feature rows removed: {n_bad_rows_removed}")
    print(f"Rows after bad feature filtering: {n_rows_after}")

    if n_bad_rows_removed > 0:
        print("Top removed bad variable_names:")
        print(bad_rows["variable_name"].astype(str).value_counts().head(30).to_string())

    data = data.drop_duplicates(
        subset=["ID", "t", "variable_name"],
        keep="first",
    )

    data = data.sort_values(["ID", "t", "variable_name"])

    output_file = os.path.join(output_dir, "input_data.csv")
    data.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print("Rows after duplicate removal:", len(data))
    print("Unique IDs:", data["ID"].nunique())
    print("Unique variables:", data["variable_name"].nunique())
    print("t min:", data["t"].min())
    print("t max:", data["t"].max())

    return {
        "task": task_name,
        "population_file": population_file,
        "T_hours": T,
        "n_population_ids": len(pop_ids),
        "n_rows_before": n_rows_before,
        "n_bad_rows_removed": n_bad_rows_removed,
        "n_rows_after": n_rows_after,
        "n_rows_after_duplicate_removal": len(data),
        "n_unique_ids": data["ID"].nunique(),
        "n_unique_variables": data["variable_name"].nunique(),
        "bad_variable_patterns": ";".join(BAD_VARIABLE_PATTERNS),
        "output_file": output_file,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extracted_path",
        default=str(PORTFOLIO_ROOT / "data/extracted/final_1000_0"),
    )

    parser.add_argument(
        "--population_path",
        default=str(PORTFOLIO_ROOT / "data/population/final_1000_0"),
    )

    parser.add_argument(
        "--features_root",
        default=str(PORTFOLIO_ROOT / "data/features_pooled/final_1000_0"),
    )

    parser.add_argument(
        "--summary_path",
        default=str(PORTFOLIO_ROOT / "data/summary/final_1000_0"),
    )

    args = parser.parse_args()

    os.makedirs(args.features_root, exist_ok=True)
    os.makedirs(args.summary_path, exist_ok=True)

    summary_rows = []

    for task_name, population_fname, T in TASK_SPECS:
        population_file = os.path.join(args.population_path, population_fname)

        if not os.path.exists(population_file):
            print(f"Skipping {task_name}: missing {population_file}")
            continue

        row = prepare_one_task(
            task_name=task_name,
            population_file=population_file,
            T=T,
            extracted_path=args.extracted_path,
            features_root=args.features_root,
        )

        summary_rows.append(row)

    if len(summary_rows) == 0:
        raise RuntimeError(
            f"No tasks were prepared. Check population_path: {args.population_path}"
        )

    summary = pd.DataFrame(summary_rows)

    summary_file = os.path.join(
        args.summary_path,
        "bad_feature_filter_summary_pooled.csv",
    )

    summary.to_csv(summary_file, index=False)

    print("\nSaved pooled bad feature filter summary:")
    print(summary_file)
    print(summary)


if __name__ == "__main__":
    main()
