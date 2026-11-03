"""Fit pooled TruncatedSVD to static and flattened temporal FIDDLE features.

Labels and hospital IDs are excluded from the feature matrix. Representation fitting
uses all hospitals; this is a pooled, transductive representation analysis."""


from pathlib import Path
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import sparse as pydata_sparse
from scipy import sparse as scipy_sparse
from sklearn.decomposition import TruncatedSVD


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


PROJECT_ROOT = PORTFOLIO_ROOT


DEFAULT_FIDDLE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features_pooled"
    / "final_1000_0"
    / "mortality_48h"
    / "FIDDLE_output_theta1_003"
)
DEFAULT_POPULATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "population"
    / "final_1000_0"
    / "mortality_48h_with_metadata.csv"
)
DEFAULT_OUTDIR = (
    PROJECT_ROOT / "results" / "final_1000_0" / "patient_embeddings" / "pooled_svd"
)


def find_column(dataframe, candidates, description):
    """Find a column without depending on capitalization."""
    lower_to_original = {
        str(column).strip().lower(): column for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def normalize_id_series(series, column_name):
    """Normalize eICU IDs while rejecting missing or non-numeric values."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        examples = series.loc[numeric.isna()].head(5).tolist()
        raise ValueError(
            f"{column_name} contains missing/non-numeric values. "
            f"Examples: {examples}"
        )

    return numeric.astype("int64")


def read_fiddle_ids(ids_path):
    ids = pd.read_csv(ids_path)

    if ids.empty:
        raise ValueError(f"No rows found in {ids_path}")

    try:
        id_column = find_column(
            ids,
            ["ID", "patientunitstayid", "icustay_id"],
            "patient ID column in IDs.csv",
        )
    except ValueError:
        non_index_columns = [
            column
            for column in ids.columns
            if not str(column).lower().startswith("unnamed")
        ]

        if len(non_index_columns) != 1:
            raise

        id_column = non_index_columns[0]

    result = pd.DataFrame({"ID": normalize_id_series(ids[id_column], id_column)})

    if result["ID"].duplicated().any():
        duplicate_ids = result.loc[result["ID"].duplicated(), "ID"].head(5).tolist()
        raise ValueError(f"FIDDLE IDs are not unique. Examples: {duplicate_ids}")

    return result


def read_patient_metadata(population_path, fiddle_ids):
    population = pd.read_csv(population_path)

    id_column = find_column(
        population,
        ["ID", "patientunitstayid", "icustay_id"],
        "patient ID column in population file",
    )
    hospital_column = find_column(
        population,
        ["hospitalid", "hospital_id"],
        "hospital ID column in population file",
    )

    metadata = population[[id_column, hospital_column]].copy()
    metadata.columns = ["ID", "hospitalid"]
    metadata["ID"] = normalize_id_series(metadata["ID"], "ID")

    if metadata["ID"].duplicated().any():
        duplicate_ids = metadata.loc[metadata["ID"].duplicated(), "ID"].head(5).tolist()
        raise ValueError(
            "Population metadata has duplicate patient IDs. "
            f"Examples: {duplicate_ids}"
        )

    row_info = fiddle_ids.merge(
        metadata,
        on="ID",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if row_info["hospitalid"].isna().any():
        missing_ids = (
            row_info.loc[row_info["hospitalid"].isna(), "ID"].head(10).tolist()
        )
        raise ValueError(
            "Some FIDDLE IDs have no hospital ID in the population file. "
            f"Examples: {missing_ids}"
        )

    return row_info


def load_pydata_sparse_npz(path):
    """
    Load an npz written by the PyData `sparse` library (FIDDLE's format:
    keys data/coords/shape/fill_value), NOT scipy.sparse.load_npz.
    """
    return pydata_sparse.load_npz(str(path))


def flatten_temporal_matrix(x_matrix, n_patients):
    """Flatten temporal features while preserving the patient axis."""
    if x_matrix.ndim != 3:
        raise ValueError(
            f"Expected a 3D (N, L, D) temporal array, got shape {x_matrix.shape}"
        )

    n, l, d = x_matrix.shape

    if n != n_patients:
        raise ValueError(
            "Cannot align X.npz with IDs.csv: "
            f"X shape={x_matrix.shape}, n_patients={n_patients}"
        )

    flattened = x_matrix.reshape((n, l * d))

    return flattened.tocsr()


def load_fiddle_matrix(fiddle_output_dir, feature_set):
    ids_path = fiddle_output_dir / "IDs.csv"
    x_path = fiddle_output_dir / "X.npz"
    s_path = fiddle_output_dir / "S.npz"

    required_paths = [ids_path]
    if feature_set in {"temporal", "all"}:
        required_paths.append(x_path)
    if feature_set in {"static", "all"}:
        required_paths.append(s_path)

    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required FIDDLE output(s): "
            + ", ".join(str(path) for path in missing_paths)
        )

    ids = read_fiddle_ids(ids_path)
    n_patients = len(ids)
    matrix_parts = []
    audit = {
        "n_patients": n_patients,
        "feature_set": feature_set,
    }

    if feature_set in {"static", "all"}:
        s_matrix_pydata = load_pydata_sparse_npz(s_path)

        if s_matrix_pydata.ndim != 2:
            raise ValueError(
                f"Expected a 2D static (N, D_static) array, "
                f"got shape {s_matrix_pydata.shape}"
            )

        s_matrix = s_matrix_pydata.tocsr()

        if s_matrix.shape[0] != n_patients:
            raise ValueError(
                "Cannot align S.npz with IDs.csv: "
                f"S shape={s_matrix.shape}, n_patients={n_patients}"
            )
        matrix_parts.append(s_matrix)
        audit["static_shape"] = list(s_matrix.shape)

    if feature_set in {"temporal", "all"}:
        x_saved = load_pydata_sparse_npz(x_path)
        audit["temporal_saved_shape"] = list(x_saved.shape)

        x_flat = flatten_temporal_matrix(x_saved, n_patients)
        matrix_parts.append(x_flat)
        audit["temporal_flat_shape"] = list(x_flat.shape)

    feature_matrix = scipy_sparse.hstack(
        matrix_parts,
        format="csr",
        dtype=np.float32,
    )

    audit["combined_shape"] = list(feature_matrix.shape)
    audit["combined_nnz"] = int(feature_matrix.nnz)
    audit["combined_density"] = float(
        feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])
    )

    return ids, feature_matrix, audit


def save_embedding_outputs(
    embeddings,
    row_info,
    svd,
    task,
    n_components,
    outdir,
    output_format,
):
    prefix = f"{task}_pooled_svd{n_components}"

    if output_format in {"npz", "both"}:
        np.savez_compressed(
            outdir / f"{prefix}_patient_embeddings.npz",
            embeddings=embeddings.astype(np.float32, copy=False),
        )

    if output_format in {"csv", "both"}:
        component_columns = [f"svd_{index:03d}" for index in range(1, n_components + 1)]
        embedding_table = pd.concat(
            [
                row_info.reset_index(drop=True),
                pd.DataFrame(
                    embeddings,
                    columns=component_columns,
                ),
            ],
            axis=1,
        )
        embedding_table.to_csv(
            outdir / f"{prefix}_patient_embeddings.csv.gz",
            index=False,
            compression="gzip",
        )

    explained_variance = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": (svd.explained_variance_ratio_),
        }
    )
    explained_variance["cumulative_explained_variance_ratio"] = explained_variance[
        "explained_variance_ratio"
    ].cumsum()
    explained_variance.to_csv(
        outdir / f"{prefix}_explained_variance.csv",
        index=False,
    )

    joblib.dump(
        svd,
        outdir / f"{prefix}_model.joblib",
        compress=3,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit pooled TruncatedSVD on FIDDLE features and save "
            "one embedding vector per patient."
        )
    )
    parser.add_argument("--task", default="mortality_48h")
    parser.add_argument(
        "--fiddle-output-dir",
        type=Path,
        default=DEFAULT_FIDDLE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--population-file",
        type=Path,
        default=DEFAULT_POPULATION_FILE,
        help=(
            "Must be the '_with_metadata' population file (has hospitalid), "
            "not the plain mortality_48h.csv used to feed FIDDLE itself "
            "(that one only has ID + label by design)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=[500],
        help=("Fit each requested dimension separately; the study setting is 500."),
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "temporal", "static"],
        default="all",
        help=(
            "'all' concatenates static S and flattened temporal X; "
            "'temporal' uses X only; 'static' uses S only."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["npz", "csv", "both"],
        default="both",
    )
    parser.add_argument("--n-iter", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    requested_components = sorted(set(args.n_components))
    if not requested_components or requested_components[0] < 1:
        raise ValueError("--n-components must contain positive integers")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FIDDLE output: {args.fiddle_output_dir}")
    ids, feature_matrix, input_audit = load_fiddle_matrix(
        args.fiddle_output_dir,
        args.feature_set,
    )
    row_info = read_patient_metadata(
        args.population_file,
        ids,
    )

    if len(row_info) != feature_matrix.shape[0]:
        raise RuntimeError("Internal alignment error between metadata and feature rows")

    row_info.to_csv(
        args.outdir / f"{args.task}_patient_row_metadata.csv",
        index=False,
    )

    hospital_counts = (
        row_info.groupby("hospitalid", dropna=False)
        .size()
        .rename("n_patients")
        .reset_index()
        .sort_values("hospitalid")
    )
    hospital_counts.to_csv(
        args.outdir / f"{args.task}_hospital_patient_counts.csv",
        index=False,
    )

    max_allowed_components = min(feature_matrix.shape) - 1
    too_large = [
        value for value in requested_components if value > max_allowed_components
    ]
    if too_large:
        raise ValueError(
            f"Requested n_components={too_large}, but the maximum "
            f"allowed for matrix shape {feature_matrix.shape} is "
            f"{max_allowed_components}"
        )

    summary_rows = []

    print(
        f"Patients={feature_matrix.shape[0]:,}, "
        f"features={feature_matrix.shape[1]:,}, "
        f"hospitals={row_info['hospitalid'].nunique():,}, "
        f"feature_set={args.feature_set}"
    )

    for n_components in requested_components:
        print(f"\nFitting pooled TruncatedSVD({n_components})...")
        start_time = time.time()

        svd = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=args.n_iter,
            random_state=args.random_state,
        )
        embeddings = svd.fit_transform(feature_matrix)
        elapsed_seconds = time.time() - start_time
        explained_variance_sum = float(svd.explained_variance_ratio_.sum())

        save_embedding_outputs(
            embeddings=embeddings,
            row_info=row_info,
            svd=svd,
            task=args.task,
            n_components=n_components,
            outdir=args.outdir,
            output_format=args.output_format,
        )

        summary_rows.append(
            {
                "task": args.task,
                "feature_set": args.feature_set,
                "n_patients": feature_matrix.shape[0],
                "n_hospitals": row_info["hospitalid"].nunique(),
                "n_input_features": feature_matrix.shape[1],
                "n_components": n_components,
                "explained_variance_ratio_sum": (explained_variance_sum),
                "fit_seconds": elapsed_seconds,
                "random_state": args.random_state,
                "n_iter": args.n_iter,
            }
        )

        print(
            f"Saved SVD-{n_components}: "
            f"cumulative explained variance="
            f"{explained_variance_sum:.4f}, "
            f"fit time={elapsed_seconds:.1f}s"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.outdir / f"{args.task}_pooled_svd_summary.csv",
        index=False,
    )

    run_config = {
        "task": args.task,
        "fiddle_output_dir": str(args.fiddle_output_dir.resolve()),
        "population_file": str(args.population_file.resolve()),
        "outdir": str(args.outdir.resolve()),
        "n_components": requested_components,
        "feature_set": args.feature_set,
        "output_format": args.output_format,
        "n_iter": args.n_iter,
        "random_state": args.random_state,
        "input_audit": input_audit,
    }
    with (args.outdir / f"{args.task}_pooled_svd_run_config.json").open(
        "w", encoding="utf-8"
    ) as file_handle:
        json.dump(run_config, file_handle, indent=2)

    print("\nCompleted.")
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs: {args.outdir}")


if __name__ == "__main__":
    main()
