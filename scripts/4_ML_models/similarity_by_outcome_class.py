"""Compare cosine similarity to source decedents and survivors separately.

The within-hospital reference is an outcome-informed exploratory diagnostic.
All performance summaries use predictions from the same source model."""


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr, wilcoxon

import matplotlib


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = PORTFOLIO_ROOT
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "4_ML_models"
sys.path.insert(0, str(SCRIPTS_DIR))

from pooled_svd_embeddings import read_patient_metadata
from cross_hospital_transportability import normalize_hospital_id
from similarity_utils import (
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_POPULATION_FILE,
    assign_deciles,
    load_embeddings,
    mean_similarity_to_reference,
)


DEFAULT_OUTDIR = (
    PROJECT_ROOT / "results" / "final_1000_0" / "similarity_by_outcome_class"
)


def wilson_confidence_interval(n_correct, n, confidence=0.95):
    if n == 0:
        return np.nan, np.nan

    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = n_correct / n
    denominator = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))

    low = (center - margin) / denominator
    high = (center + margin) / denominator
    return float(max(0.0, low)), float(min(1.0, high))


def similarity_to_source_classes(
    target_vectors,
    source_decedent_vectors,
    source_survivor_vectors,
    chunk_size,
):
    to_decedents = mean_similarity_to_reference(
        target_vectors,
        source_decedent_vectors,
        chunk_size,
        exclude_self=False,
    )
    to_survivors = mean_similarity_to_reference(
        target_vectors,
        source_survivor_vectors,
        chunk_size,
        exclude_self=False,
    )
    return to_decedents, to_survivors


def similarity_to_own_hospital_classes(
    decedent_vectors,
    survivor_vectors,
    chunk_size,
):
    n_decedents = decedent_vectors.shape[0]
    n_survivors = survivor_vectors.shape[0]

    decedent_to_decedents = mean_similarity_to_reference(
        decedent_vectors,
        decedent_vectors,
        chunk_size,
        exclude_self=True,
    )

    decedent_to_survivors = mean_similarity_to_reference(
        decedent_vectors,
        survivor_vectors,
        chunk_size,
        exclude_self=False,
    )

    survivor_to_decedents = mean_similarity_to_reference(
        survivor_vectors,
        decedent_vectors,
        chunk_size,
        exclude_self=False,
    )

    survivor_to_survivors = mean_similarity_to_reference(
        survivor_vectors,
        survivor_vectors,
        chunk_size,
        exclude_self=True,
    )

    assert n_decedents > 1 and n_survivors > 1

    return (
        decedent_to_decedents,
        decedent_to_survivors,
        survivor_to_decedents,
        survivor_to_survivors,
    )


def direct_similarity_comparison(
    frame,
    similarity_to_decedents_col,
    similarity_to_survivors_col,
    reference_name,
):
    rows = []

    for outcome_value, outcome_name in [(1, "decedents"), (0, "survivors")]:
        piece = frame.loc[frame["label"] == outcome_value].dropna(
            subset=[
                similarity_to_decedents_col,
                similarity_to_survivors_col,
            ]
        )

        sim_death = piece[similarity_to_decedents_col].to_numpy()
        sim_survive = piece[similarity_to_survivors_col].to_numpy()
        difference = sim_death - sim_survive

        if len(difference) > 0 and np.any(difference != 0):
            statistic, p_value = wilcoxon(
                sim_death,
                sim_survive,
                alternative="two-sided",
            )
        else:
            statistic, p_value = np.nan, np.nan

        rows.append(
            {
                "reference": reference_name,
                "actual_outcome": outcome_name,
                "n": int(len(piece)),
                "mean_similarity_to_decedents": float(np.mean(sim_death)),
                "median_similarity_to_decedents": float(np.median(sim_death)),
                "mean_similarity_to_survivors": float(np.mean(sim_survive)),
                "median_similarity_to_survivors": float(np.median(sim_survive)),
                "mean_decedent_lean": float(np.mean(difference)),
                "median_decedent_lean": float(np.median(difference)),
                "fraction_decedent_lean_positive": float(np.mean(difference > 0)),
                "paired_wilcoxon_statistic": float(statistic),
                "paired_wilcoxon_p": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def compare_lean_between_outcomes(frame, lean_column, reference_name):
    usable = frame.dropna(subset=[lean_column, "label"])
    death_lean = usable.loc[usable["label"] == 1, lean_column].to_numpy()
    survive_lean = usable.loc[usable["label"] == 0, lean_column].to_numpy()

    if len(death_lean) == 0 or len(survive_lean) == 0:
        return {
            "reference": reference_name,
            "n_decedents": len(death_lean),
            "n_survivors": len(survive_lean),
            "mannwhitney_u": np.nan,
            "mannwhitney_p": np.nan,
            "rank_biserial": np.nan,
            "spearman_r_with_label": np.nan,
            "spearman_p_with_label": np.nan,
        }

    u_statistic, u_p = mannwhitneyu(
        death_lean,
        survive_lean,
        alternative="two-sided",
    )

    rank_biserial = 2 * u_statistic / (len(death_lean) * len(survive_lean)) - 1

    spearman_r, spearman_p = spearmanr(usable[lean_column], usable["label"])

    return {
        "reference": reference_name,
        "n_decedents": int(len(death_lean)),
        "n_survivors": int(len(survive_lean)),
        "mean_lean_decedents": float(np.mean(death_lean)),
        "mean_lean_survivors": float(np.mean(survive_lean)),
        "median_lean_decedents": float(np.median(death_lean)),
        "median_lean_survivors": float(np.median(survive_lean)),
        "mannwhitney_u": float(u_statistic),
        "mannwhitney_p": float(u_p),
        "rank_biserial": float(rank_biserial),
        "spearman_r_with_label": float(spearman_r),
        "spearman_p_with_label": float(spearman_p),
    }


def bin_proportions(
    frame,
    threshold,
    outcome_value,
    similarity_column,
    n_bins,
    bin_label,
    bin_within_hospital=False,
):
    subset = (
        frame.loc[frame["label"] == outcome_value]
        .dropna(subset=[similarity_column, "predicted_probability_source"])
        .copy()
    )

    if len(subset) == 0:
        return pd.DataFrame(), {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "n_bins": 0,
        }

    if bin_within_hospital:

        subset[bin_label] = np.nan
        for hospital, index in subset.groupby("hospitalid").groups.items():
            if len(index) < n_bins:
                print(
                    f"  Hospital {hospital}: only {len(index)} patients in "
                    f"this outcome subset; excluded from {n_bins}-bin analysis"
                )
                continue
            subset.loc[index, bin_label] = assign_deciles(
                subset.loc[index, similarity_column].to_numpy(), n_bins
            )
        subset = subset.dropna(subset=[bin_label]).copy()
    else:
        subset[bin_label] = assign_deciles(subset[similarity_column].to_numpy(), n_bins)

    if outcome_value == 1:
        subset["correct"] = (
            subset["predicted_probability_source"] >= threshold
        ).astype(int)
    else:
        subset["correct"] = (subset["predicted_probability_source"] < threshold).astype(
            int
        )

    rows = []
    for bin_value in sorted(subset[bin_label].unique()):
        piece = subset.loc[subset[bin_label] == bin_value]
        n = len(piece)
        n_correct = int(piece["correct"].sum())
        ci_low, ci_high = wilson_confidence_interval(n_correct, n)

        rows.append(
            {
                bin_label: int(bin_value),
                "n": int(n),
                "n_correct": n_correct,
                "proportion_correct": n_correct / n,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_predicted_probability_source": float(
                    piece["predicted_probability_source"].mean()
                ),
                "mean_decedent_lean": float(piece[similarity_column].mean()),
                "n_hospitals": int(piece["hospitalid"].nunique()),
            }
        )

    table = pd.DataFrame(rows)

    if len(table) >= 3 and table["proportion_correct"].nunique() > 1:
        r, p = spearmanr(table[bin_label], table["proportion_correct"])
    else:
        r, p = np.nan, np.nan

    return table, {
        "spearman_r": float(r) if r == r else np.nan,
        "spearman_p": float(p) if p == p else np.nan,
        "n_bins": int(len(table)),
    }


def plot_performance(
    frame,
    lean_column,
    threshold,
    bin_within_hospital,
    out_path,
    title,
):
    usable = frame.dropna(subset=[lean_column, "predicted_probability_source", "label"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    survivors = usable.loc[usable["label"] == 0]
    decedents = usable.loc[usable["label"] == 1]

    axes[0].scatter(
        survivors[lean_column],
        survivors["predicted_probability_source"],
        s=4,
        alpha=0.20,
        color="#1f77b4",
        label="survived",
    )
    axes[0].scatter(
        decedents[lean_column],
        decedents["predicted_probability_source"],
        s=6,
        alpha=0.35,
        color="#d62728",
        label="died",
    )
    axes[0].axhline(
        threshold,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="hospital 73 threshold",
    )
    axes[0].set_xlabel(lean_column)
    axes[0].set_ylabel("Hospital 73 predicted probability")
    axes[0].set_title("73-model probability vs decedent_lean", fontsize=10)
    axes[0].legend(fontsize=8, markerscale=2)
    axes[0].grid(alpha=0.3)

    for outcome_value, color, label in [
        (1, "#d62728", "recall among decedents"),
        (0, "#1f77b4", "specificity among survivors"),
    ]:
        table, _ = bin_proportions(
            usable,
            threshold,
            outcome_value,
            lean_column,
            10,
            "bin",
            bin_within_hospital=bin_within_hospital,
        )
        if len(table):
            axes[1].plot(
                table["bin"],
                table["proportion_correct"],
                marker="o",
                color=color,
                label=label,
            )

    axes[1].set_xlabel("decedent_lean decile")
    axes[1].set_ylabel("Proportion correct")
    axes[1].set_title("73-model performance by decedent_lean", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    figure.suptitle(title, fontsize=12, y=1.02)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute each target patient's mean cosine similarity to "
            "hospital 73 decedents and survivors separately."
        )
    )
    parser.add_argument("--source-hospital", default="73")
    parser.add_argument("--task", default="mortality_48h")
    parser.add_argument("--population-file", type=Path, default=DEFAULT_POPULATION_FILE)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--n-components", type=int, default=500)
    parser.add_argument("--label-col", default="mortality_LABEL")

    parser.add_argument(
        "--to-source-patient-level",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "final_1000_0"
            / "transportability"
            / "mortality_48h_source73_predictions.csv.gz"
        ),
        help="Patient-level predictions exported by cross_hospital_transportability.py.",
    )

    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    source_hospital = normalize_hospital_id(args.source_hospital)
    prefix = f"{args.task}_source{source_hospital}_svd{args.n_components}"

    print(f"Loading svd{args.n_components} embeddings")
    embeddings, svd_columns = load_embeddings(
        args.embeddings_dir,
        args.task,
        args.n_components,
    )
    embeddings["ID"] = embeddings["ID"].astype("int64")
    embedding_lookup = embeddings.set_index("ID")

    print("Loading population metadata")

    fiddle_ids = embeddings[["ID"]].copy()
    row_info = read_patient_metadata(args.population_file, fiddle_ids)

    if args.label_col not in row_info.columns:
        population = pd.read_csv(args.population_file)
        population["ID"] = population["ID"].astype("int64")
        row_info = row_info.merge(
            population[["ID", args.label_col]],
            on="ID",
            how="left",
            validate="one_to_one",
        )

    required_metadata = {"ID", "hospitalid", args.label_col}
    missing_metadata = required_metadata - set(row_info.columns)
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {sorted(missing_metadata)}")
    if row_info[list(required_metadata)].isna().any().any():
        raise ValueError("Missing ID, hospital, or outcome values in metadata")

    row_info["ID"] = row_info["ID"].astype("int64")
    row_info[args.label_col] = row_info[args.label_col].astype(int)
    row_info["hospitalid_norm"] = row_info["hospitalid"].apply(normalize_hospital_id)
    row_info = row_info.set_index("ID")

    labels_all = row_info[args.label_col]
    hospitals_all = row_info["hospitalid_norm"]

    print(
        f"\nSource hospital {source_hospital}: building outcome-specific "
        "reference groups"
    )
    source_ids = row_info.index[hospitals_all == source_hospital]
    source_decedent_ids = source_ids[labels_all.loc[source_ids] == 1]
    source_survivor_ids = source_ids[labels_all.loc[source_ids] == 0]

    if len(source_decedent_ids) == 0 or len(source_survivor_ids) == 0:
        raise ValueError(
            "Source hospital must contain at least one decedent and survivor"
        )

    print(
        f"  decedents={len(source_decedent_ids)}, "
        f"survivors={len(source_survivor_ids)}"
    )

    source_decedent_vectors = embedding_lookup.loc[
        source_decedent_ids, svd_columns
    ].to_numpy()
    source_survivor_vectors = embedding_lookup.loc[
        source_survivor_ids, svd_columns
    ].to_numpy()

    target_ids = row_info.index[hospitals_all != source_hospital]
    target_vectors = embedding_lookup.loc[target_ids, svd_columns].to_numpy()

    print(
        f"Computing to-source similarities for {len(target_ids):,} "
        "target patients..."
    )
    to_decedents, to_survivors = similarity_to_source_classes(
        target_vectors,
        source_decedent_vectors,
        source_survivor_vectors,
        args.chunk_size,
    )

    to_source_frame = pd.DataFrame(
        {
            "ID": target_ids.to_numpy(dtype="int64"),
            "hospitalid": hospitals_all.loc[target_ids].to_numpy(),
            "label": labels_all.loc[target_ids].to_numpy(),
            "similarity_to_source_decedents": to_decedents,
            "similarity_to_source_survivors": to_survivors,
        }
    )
    to_source_frame["decedent_lean_to_source"] = (
        to_source_frame["similarity_to_source_decedents"]
        - to_source_frame["similarity_to_source_survivors"]
    )

    print("\nComputing within-hospital outcome-specific similarities...")
    within_rows = []
    target_hospitals = sorted(hospitals_all.loc[target_ids].unique())

    for hospital in target_hospitals:
        hospital_ids = row_info.index[hospitals_all == hospital]
        hospital_labels = labels_all.loc[hospital_ids]
        decedent_ids = hospital_ids[hospital_labels == 1]
        survivor_ids = hospital_ids[hospital_labels == 0]

        if len(decedent_ids) < 2 or len(survivor_ids) < 2:
            print(
                f"  Hospital {hospital}: skipped "
                f"(decedents={len(decedent_ids)}, survivors={len(survivor_ids)})"
            )
            continue

        decedent_vectors = embedding_lookup.loc[decedent_ids, svd_columns].to_numpy()
        survivor_vectors = embedding_lookup.loc[survivor_ids, svd_columns].to_numpy()

        (
            decedent_to_decedents,
            decedent_to_survivors,
            survivor_to_decedents,
            survivor_to_survivors,
        ) = similarity_to_own_hospital_classes(
            decedent_vectors,
            survivor_vectors,
            args.chunk_size,
        )

        within_rows.append(
            pd.DataFrame(
                {
                    "ID": decedent_ids.to_numpy(dtype="int64"),
                    "similarity_to_own_decedents": decedent_to_decedents,
                    "similarity_to_own_survivors": decedent_to_survivors,
                }
            )
        )
        within_rows.append(
            pd.DataFrame(
                {
                    "ID": survivor_ids.to_numpy(dtype="int64"),
                    "similarity_to_own_decedents": survivor_to_decedents,
                    "similarity_to_own_survivors": survivor_to_survivors,
                }
            )
        )
        print(
            f"  Hospital {hospital}: decedents={len(decedent_ids)}, "
            f"survivors={len(survivor_ids)}"
        )

    if within_rows:
        within_frame = pd.concat(within_rows, ignore_index=True)
        if within_frame["ID"].duplicated().any():
            raise ValueError("Duplicate patient IDs in within-hospital results")
        within_frame["decedent_lean_within_hospital"] = (
            within_frame["similarity_to_own_decedents"]
            - within_frame["similarity_to_own_survivors"]
        )
        combined = to_source_frame.merge(
            within_frame,
            on="ID",
            how="left",
            validate="one_to_one",
        )
    else:
        print("Warning: no hospital had enough patients for within-hospital control")
        combined = to_source_frame.copy()

    if not args.to_source_patient_level.exists():
        raise FileNotFoundError(
            f"Missing hospital 73 prediction file: {args.to_source_patient_level}"
        )

    source_predictions = pd.read_csv(args.to_source_patient_level)[
        ["ID", "predicted_probability"]
    ].copy()
    source_predictions["ID"] = source_predictions["ID"].astype("int64")
    source_predictions = source_predictions.rename(
        columns={"predicted_probability": "predicted_probability_source"}
    )
    if source_predictions["ID"].duplicated().any():
        raise ValueError("Duplicate patient IDs in hospital 73 prediction file")

    combined = combined.merge(
        source_predictions,
        on="ID",
        how="left",
        validate="one_to_one",
    )
    missing_predictions = int(combined["predicted_probability_source"].isna().sum())
    if missing_predictions:
        raise ValueError(
            f"{missing_predictions} target patients lack hospital 73 predictions"
        )

    combined_path = args.outdir / f"{prefix}_similarity_by_outcome_class.csv.gz"
    combined.to_csv(combined_path, index=False, compression="gzip")
    print(f"\nSaved patient-level file: {combined_path}")

    direct_tables = []
    separation_rows = []

    to_source_direct = direct_similarity_comparison(
        combined,
        "similarity_to_source_decedents",
        "similarity_to_source_survivors",
        "to_source",
    )
    direct_tables.append(to_source_direct)
    separation_rows.append(
        compare_lean_between_outcomes(
            combined,
            "decedent_lean_to_source",
            "to_source",
        )
    )

    if "decedent_lean_within_hospital" in combined.columns:
        within_direct = direct_similarity_comparison(
            combined,
            "similarity_to_own_decedents",
            "similarity_to_own_survivors",
            "within_hospital_exploratory",
        )
        direct_tables.append(within_direct)
        separation_rows.append(
            compare_lean_between_outcomes(
                combined,
                "decedent_lean_within_hospital",
                "within_hospital_exploratory",
            )
        )

    direct_table = pd.concat(direct_tables, ignore_index=True)
    separation_table = pd.DataFrame(separation_rows)

    direct_table.to_csv(
        args.outdir / f"{prefix}_direct_similarity_comparison.csv",
        index=False,
    )
    separation_table.to_csv(
        args.outdir / f"{prefix}_decedent_lean_outcome_separation.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("DIRECT COMPARISON: SIMILARITY TO DECEDENTS VS SURVIVORS")
    print("=" * 78)
    print(direct_table.round(6).to_string(index=False))

    print("\n" + "=" * 78)
    print("DOES DECEDENT_LEAN SEPARATE ACTUAL OUTCOMES?")
    print("=" * 78)
    print(separation_table.round(6).to_string(index=False))

    all_bins = []
    all_trends = []

    analyses = [
        (
            "decedent_lean_to_source",
            "to_source",
            False,
        )
    ]
    if "decedent_lean_within_hospital" in combined.columns:
        analyses.append(
            (
                "decedent_lean_within_hospital",
                "within_hospital_exploratory",
                True,
            )
        )

    for lean_column, analysis_name, bin_within_hospital in analyses:
        for n_bins, bin_tag in [(4, "quartile"), (args.n_bins, "decile")]:
            for outcome_value, outcome_name in [
                (1, "decedents_recall"),
                (0, "survivors_specificity"),
            ]:
                print(
                    f"\n--- {analysis_name}, {bin_tag}, {outcome_name}; "
                    "hospital 73 model + hospital 73 threshold ---"
                )
                table, trend = bin_proportions(
                    combined,
                    args.threshold,
                    outcome_value,
                    lean_column,
                    n_bins,
                    "bin",
                    bin_within_hospital=bin_within_hospital,
                )

                if len(table):
                    print(table.round(6).to_string(index=False))
                    print(
                        f"Trend: r={trend['spearman_r']:.4f}, "
                        f"p={trend['spearman_p']:.4g}"
                    )
                    table["similarity_definition"] = analysis_name
                    table["outcome_metric"] = outcome_name
                    table["bin_type"] = bin_tag
                    all_bins.append(table)
                    all_trends.append(
                        {
                            "similarity_definition": analysis_name,
                            "outcome_metric": outcome_name,
                            "bin_type": bin_tag,
                            **trend,
                        }
                    )

        plot_performance(
            combined,
            lean_column,
            args.threshold,
            bin_within_hospital,
            args.outdir / f"{prefix}_{analysis_name}_decedent_lean.png",
            f"decedent_lean ({analysis_name}); hospital 73 model",
        )

    if all_bins:
        pd.concat(all_bins, ignore_index=True).to_csv(
            args.outdir / f"{prefix}_decedent_lean_bins.csv",
            index=False,
        )
    if all_trends:
        pd.DataFrame(all_trends).to_csv(
            args.outdir / f"{prefix}_decedent_lean_trends.csv",
            index=False,
        )

    print(f"\nAll outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
