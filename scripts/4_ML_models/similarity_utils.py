"""Shared embedding loading, similarity, and binning utilities."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_DIR = (
    PROJECT_ROOT / "results/final_1000_0/patient_embeddings/pooled_svd"
)
DEFAULT_POPULATION_FILE = (
    PROJECT_ROOT / "data/population/final_1000_0/mortality_48h_with_metadata.csv"
)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def assign_deciles(similarity_values, n_bins):
    """Bin 1 is the LEAST similar. Returns integer bins starting at 1."""
    ranks = pd.Series(similarity_values).rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    return (bins + 1).to_numpy()


def mean_similarity_to_reference(
    query_vectors,
    reference_vectors,
    chunk_size,
    exclude_self=False,
):
    """Mean cosine similarity from each query patient to all reference patients.

    exclude_self=True is for the within-hospital control, where query and
    reference are the same patients and the diagonal (a patient compared with
    themselves, similarity 1.0) has to come out.
    """
    n_query = query_vectors.shape[0]
    n_reference = reference_vectors.shape[0]
    means = np.empty(n_query, dtype=float)

    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        block = cosine_similarity(query_vectors[start:stop], reference_vectors)

        if exclude_self:

            for local_index, global_index in enumerate(range(start, stop)):
                block[local_index, global_index] = np.nan
            means[start:stop] = np.nanmean(block, axis=1)
        else:
            means[start:stop] = block.mean(axis=1)

    if exclude_self and n_reference <= 1:
        means[:] = np.nan

    return means


def load_embeddings(embeddings_dir, task, n_components):
    path = embeddings_dir / f"{task}_pooled_svd{n_components}_patient_embeddings.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")

    frame = pd.read_csv(path)
    svd_columns = [c for c in frame.columns if c.startswith("svd_")]
    if not svd_columns:
        raise ValueError(f"No svd_* columns in {path}")
    if "ID" not in frame.columns:
        raise ValueError(f"No ID column in {path}")

    frame["ID"] = frame["ID"].astype("int64")
    return frame, svd_columns
