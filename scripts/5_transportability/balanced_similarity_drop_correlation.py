"""Relate hospital-level similarity scores to source-minus-target performance.

Class-balanced summaries give equal weight to target decedents and survivors.
Spearman tests use hospitals as observations, with within-family BH correction."""


import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import matplotlib


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


PROJECT_ROOT = PORTFOLIO_ROOT

DEFAULT_PATIENT_LEVEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_1000_0"
    / "similarity_by_outcome_class"
    / "mortality_48h_source73_svd500_similarity_by_outcome_class.csv.gz"
)

DEFAULT_OUTDIR = (
    PROJECT_ROOT / "results" / "final_1000_0" / "balanced_similarity_drop_correlation"
)


SCORE_LABELS = {
    "balanced_mean": "Balanced mean",
    "lean": "Lean (death - survive)",
    "maximum_similarity": "Maximum similarity",
    "minimum_similarity": "Minimum similarity",
    "absolute_lean": "Absolute lean",
    "normalized_lean": "Normalized lean",
    "correct_class_similarity": "Correct-class similarity",
    "outcome_aligned_lean": "Outcome-aligned lean",
}

PERFORMANCE_LABELS = {
    "f1_drop": "F1 drop",
    "auroc_drop": "AUROC drop",
    "auprc_drop": "AUPRC drop",
}

REFERENCE_LABEL = "Similarity to hospital 73 outcome classes"


def normalize_hospital_id(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return np.nan
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def load_patient_level(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing patient-level file: {path}")

    frame = pd.read_csv(path)
    required = {
        "ID",
        "hospitalid",
        "label",
        "predicted_probability_source",
        "similarity_to_source_decedents",
        "similarity_to_source_survivors",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["hospitalid"] = frame["hospitalid"].apply(normalize_hospital_id)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1")

    numeric_columns = [
        "predicted_probability_source",
        "similarity_to_source_decedents",
        "similarity_to_source_survivors",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[list(required)].isna().any().any():
        missing_counts = frame[list(required)].isna().sum()
        raise ValueError(
            "Required columns contain missing values:\n"
            + missing_counts[missing_counts > 0].to_string()
        )
    if frame["ID"].duplicated().any():
        raise ValueError("Patient-level file contains duplicate IDs")
    return frame


def add_scalar_scores(frame, death_col, survive_col, prefix):
    result = frame.copy()
    death_similarity = result[death_col].astype(float)
    survive_similarity = result[survive_col].astype(float)
    label = result["label"].astype(int)

    denominator = death_similarity.abs() + survive_similarity.abs()
    normalized_lean = np.divide(
        death_similarity - survive_similarity,
        denominator,
        out=np.full(len(result), np.nan, dtype=float),
        where=denominator.to_numpy() > 0,
    )

    correct_similarity = np.where(
        label.to_numpy() == 1,
        death_similarity.to_numpy(),
        survive_similarity.to_numpy(),
    )
    incorrect_similarity = np.where(
        label.to_numpy() == 1,
        survive_similarity.to_numpy(),
        death_similarity.to_numpy(),
    )

    formulas = {
        "balanced_mean": (death_similarity + survive_similarity) / 2.0,
        "lean": death_similarity - survive_similarity,
        "maximum_similarity": np.maximum(death_similarity, survive_similarity),
        "minimum_similarity": np.minimum(death_similarity, survive_similarity),
        "absolute_lean": (death_similarity - survive_similarity).abs(),
        "normalized_lean": normalized_lean,
        "correct_class_similarity": correct_similarity,
        "outcome_aligned_lean": correct_similarity - incorrect_similarity,
    }

    column_mapping = {}
    for score_name, values in formulas.items():
        column_name = f"{prefix}_{score_name}"
        result[column_name] = values
        column_mapping[score_name] = column_name
    return result, column_mapping


def compute_hospital_performance(
    frame,
    threshold,
    source_cv_f1,
    source_cv_auroc,
    source_cv_auprc,
):
    rows = []
    for hospital, piece in frame.groupby("hospitalid", sort=True):
        labels = piece["label"].to_numpy(dtype=int)
        probabilities = piece["predicted_probability_source"].to_numpy(dtype=float)
        predictions = (probabilities >= threshold).astype(int)

        true_positive = int(((labels == 1) & (predictions == 1)).sum())
        false_positive = int(((labels == 0) & (predictions == 1)).sum())
        false_negative = int(((labels == 1) & (predictions == 0)).sum())
        true_negative = int(((labels == 0) & (predictions == 0)).sum())

        has_both_classes = len(np.unique(labels)) == 2
        external_f1 = float(f1_score(labels, predictions, zero_division=0))
        external_auroc = (
            float(roc_auc_score(labels, probabilities)) if has_both_classes else np.nan
        )
        external_auprc = (
            float(average_precision_score(labels, probabilities))
            if has_both_classes
            else np.nan
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else np.nan
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive > 0
            else np.nan
        )

        rows.append(
            {
                "hospitalid": hospital,
                "n_patients": int(len(piece)),
                "n_decedents": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "mean_predicted_probability": float(probabilities.mean()),
                "predicted_positive_rate": float(predictions.mean()),
                "threshold": float(threshold),
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "tn": true_negative,
                "external_f1": external_f1,
                "external_auroc": external_auroc,
                "external_auprc": external_auprc,
                "external_recall": float(recall),
                "external_specificity": float(specificity),
                "external_brier": float(np.mean((probabilities - labels) ** 2)),
                "f1_drop": float(source_cv_f1 - external_f1),
                "auroc_drop": float(source_cv_auroc - external_auroc),
                "auprc_drop": float(source_cv_auprc - external_auprc),
            }
        )
    return pd.DataFrame(rows)


def aggregate_patient_scores(frame, score_columns, reference_name):
    rows = []
    for hospital, piece in frame.groupby("hospitalid", sort=True):
        for score_name, score_column in score_columns.items():
            valid = piece[["label", score_column]].dropna()
            if len(valid) == 0:
                continue

            class_means = valid.groupby("label")[score_column].mean()
            class_medians = valid.groupby("label")[score_column].median()
            if set(class_means.index) == {0, 1}:
                class_balanced_mean = float(class_means.loc[[0, 1]].mean())
                class_balanced_median = float(class_medians.loc[[0, 1]].mean())
            else:
                class_balanced_mean = np.nan
                class_balanced_median = np.nan

            aggregation_values = {
                "overall_mean": float(valid[score_column].mean()),
                "overall_median": float(valid[score_column].median()),
                "class_balanced_mean": class_balanced_mean,
                "class_balanced_median": class_balanced_median,
            }
            for aggregation_name, value in aggregation_values.items():
                rows.append(
                    {
                        "reference": reference_name,
                        "hospitalid": hospital,
                        "score_metric": score_name,
                        "aggregation": aggregation_name,
                        "hospital_score": value,
                        "n_patients_used": int(len(valid)),
                        "n_decedents_used": int(valid["label"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def safe_correlations(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(pair)
    if n < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {
            "n_hospitals": n,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

    spearman = spearmanr(pair["x"], pair["y"])
    pearson = pearsonr(pair["x"], pair["y"])
    return {
        "n_hospitals": n,
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def benjamini_hochberg(p_values):
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if len(valid) == 0:
        return result

    ordered = valid.sort_values()
    m = len(ordered)
    raw_adjusted = ordered.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[ordered.index] = adjusted
    return result


def compute_correlations(hospital_scores, hospital_performance):
    merged = hospital_scores.merge(
        hospital_performance,
        on="hospitalid",
        how="inner",
        validate="many_to_one",
    )

    performance_metrics = ["f1_drop", "auroc_drop", "auprc_drop"]
    rows = []
    group_columns = ["reference", "aggregation", "score_metric"]
    for keys, piece in merged.groupby(group_columns, sort=True):
        reference, aggregation, score_metric = keys
        for performance_metric in performance_metrics:
            rows.append(
                {
                    "reference": reference,
                    "aggregation": aggregation,
                    "score_metric": score_metric,
                    "performance_metric": performance_metric,
                    **safe_correlations(
                        piece["hospital_score"],
                        piece[performance_metric],
                    ),
                }
            )

    correlations = pd.DataFrame(rows)
    correlations["spearman_q_bh"] = correlations.groupby(
        ["reference", "aggregation", "performance_metric"],
        group_keys=False,
    )["spearman_p"].transform(benjamini_hochberg)
    return correlations, merged


def summarize_numeric_distribution(values):
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) == 0:
        return {
            "n_hospitals": 0,
            "minimum": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "q3": np.nan,
            "maximum": np.nan,
            "range": np.nan,
            "standard_deviation": np.nan,
            "iqr": np.nan,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    minimum = float(clean.min())
    maximum = float(clean.max())
    return {
        "n_hospitals": int(len(clean)),
        "minimum": minimum,
        "q1": q1,
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "q3": q3,
        "maximum": maximum,
        "range": maximum - minimum,
        "standard_deviation": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "iqr": q3 - q1,
    }


def summarize_hospital_scores(hospital_scores):
    rows = []
    for (aggregation, score_metric), piece in hospital_scores.groupby(
        ["aggregation", "score_metric"],
        sort=True,
    ):
        valid = piece.dropna(subset=["hospital_score"]).copy()
        minimum = valid["hospital_score"].min()
        maximum = valid["hospital_score"].max()
        rows.append(
            {
                "aggregation": aggregation,
                "score_metric": score_metric,
                "minimum_hospitalid": ",".join(
                    valid.loc[
                        valid["hospital_score"] == minimum,
                        "hospitalid",
                    ].astype(str)
                ),
                "maximum_hospitalid": ",".join(
                    valid.loc[
                        valid["hospital_score"] == maximum,
                        "hospitalid",
                    ].astype(str)
                ),
                **summarize_numeric_distribution(valid["hospital_score"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_hospital_performance(hospital_performance):
    metric_names = [
        "external_f1",
        "external_auroc",
        "external_auprc",
        "f1_drop",
        "auroc_drop",
        "auprc_drop",
        "prevalence",
        "mean_predicted_probability",
        "predicted_positive_rate",
    ]
    rows = []
    for metric_name in metric_names:
        if metric_name not in hospital_performance.columns:
            continue
        valid = hospital_performance[["hospitalid", metric_name]].dropna().copy()
        minimum = valid[metric_name].min()
        maximum = valid[metric_name].max()
        rows.append(
            {
                "performance_metric": metric_name,
                "minimum_hospitalid": ",".join(
                    valid.loc[
                        valid[metric_name] == minimum,
                        "hospitalid",
                    ].astype(str)
                ),
                "maximum_hospitalid": ",".join(
                    valid.loc[
                        valid[metric_name] == maximum,
                        "hospitalid",
                    ].astype(str)
                ),
                **summarize_numeric_distribution(valid[metric_name]),
            }
        )
    return pd.DataFrame(rows)


def remove_stale_within_hospital_outputs(outdir):
    stale_paths = list(
        outdir.glob("correlation_heatmap_within_hospital_exploratory_*.png")
    )
    for path in stale_paths:
        path.unlink()
    return len(stale_paths)


def set_poster_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "grid.color": "#d9dee7",
            "grid.alpha": 0.55,
        }
    )


def plot_correlation_heatmap(correlations, reference, aggregation, out_path):
    piece = correlations.loc[
        (correlations["reference"] == reference)
        & (correlations["aggregation"] == aggregation)
    ].copy()
    if len(piece) == 0:
        return

    score_order = [name for name in SCORE_LABELS if name in set(piece["score_metric"])]
    performance_order = ["f1_drop", "auroc_drop", "auprc_drop"]
    rho = piece.pivot(
        index="score_metric",
        columns="performance_metric",
        values="spearman_r",
    ).reindex(index=score_order, columns=performance_order)
    p_values = piece.pivot(
        index="score_metric",
        columns="performance_metric",
        values="spearman_p",
    ).reindex(index=score_order, columns=performance_order)
    q_values = piece.pivot(
        index="score_metric",
        columns="performance_metric",
        values="spearman_q_bh",
    ).reindex(index=score_order, columns=performance_order)

    figure_height = max(6.0, 0.72 * len(score_order) + 2.2)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    image = axis.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for row_index, score_name in enumerate(score_order):
        for column_index, performance_name in enumerate(performance_order):
            value = rho.loc[score_name, performance_name]
            p_value = p_values.loc[score_name, performance_name]
            q_value = q_values.loc[score_name, performance_name]
            if pd.isna(value):
                annotation = "NA"
            else:
                annotation = f"ρ={value:.2f}\np={p_value:.3g}\nq={q_value:.3g}"
            text_color = (
                "white" if pd.notna(value) and abs(value) >= 0.55 else "#182230"
            )
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="semibold"
                if pd.notna(value) and abs(value) >= 0.55
                else "normal",
            )

    axis.set_xticks(range(len(performance_order)))
    axis.set_xticklabels([PERFORMANCE_LABELS[name] for name in performance_order])
    axis.set_yticks(range(len(score_order)))
    axis.set_yticklabels([SCORE_LABELS[name] for name in score_order])
    axis.tick_params(length=0)
    axis.set_title(
        f"Similarity scores vs model-performance drop\n"
        f"{REFERENCE_LABEL}; {aggregation.replace('_', ' ')}",
        pad=16,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Spearman ρ")
    figure.text(
        0.5,
        0.015,
        "For similarity-level and outcome-aligned scores, negative ρ means a higher score "
        "is associated with a smaller drop. Raw lean metrics are directional. "
        "Hospitals are the observations.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def draw_scatter_panel(
    axis,
    merged,
    score_metric,
    performance_metric,
    reference,
    aggregation,
    prevalence_normalizer,
):
    piece = merged.loc[
        (merged["reference"] == reference)
        & (merged["aggregation"] == aggregation)
        & (merged["score_metric"] == score_metric)
    ].dropna(subset=["hospital_score", performance_metric])

    if len(piece) == 0:
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center")
        axis.set_axis_off()
        return None

    patient_counts = piece["n_patients"].to_numpy(dtype=float)
    if patient_counts.max() > patient_counts.min():
        sizes = 65 + 145 * (
            (patient_counts - patient_counts.min())
            / (patient_counts.max() - patient_counts.min())
        )
    else:
        sizes = np.full(len(piece), 110.0)

    scatter = axis.scatter(
        piece["hospital_score"],
        piece[performance_metric],
        c=piece["prevalence"],
        cmap="viridis",
        norm=prevalence_normalizer,
        s=sizes,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.92,
        zorder=3,
    )

    x = piece["hospital_score"].to_numpy(dtype=float)
    y = piece[performance_metric].to_numpy(dtype=float)
    if len(piece) >= 2 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        axis.plot(x_line, intercept + slope * x_line, color="#44546a", linewidth=1.7)

    for _, row in piece.iterrows():
        axis.annotate(
            str(row["hospitalid"]),
            (row["hospital_score"], row[performance_metric]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f2937",
        )

    statistics = safe_correlations(piece["hospital_score"], piece[performance_metric])
    axis.text(
        0.03,
        0.96,
        f"Spearman ρ={statistics['spearman_r']:.2f}\n"
        f"p={statistics['spearman_p']:.3g}, n={statistics['n_hospitals']}",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d5dae3",
            "alpha": 0.92,
        },
    )
    axis.set_xlabel(SCORE_LABELS[score_metric])
    axis.set_ylabel(PERFORMANCE_LABELS[performance_metric])
    axis.set_title(
        f"{SCORE_LABELS[score_metric]} vs {PERFORMANCE_LABELS[performance_metric]}"
    )
    axis.grid(True)
    return scatter


def plot_primary_scatter_grid(merged, reference, aggregation, out_path):
    piece = merged.loc[
        (merged["reference"] == reference) & (merged["aggregation"] == aggregation)
    ]
    if len(piece) == 0:
        return

    prevalence_min = float(piece["prevalence"].min())
    prevalence_max = float(piece["prevalence"].max())
    if prevalence_min == prevalence_max:
        prevalence_min -= 0.01
        prevalence_max += 0.01
    prevalence_normalizer = Normalize(vmin=prevalence_min, vmax=prevalence_max)

    panels = [
        ("balanced_mean", "auroc_drop"),
        ("balanced_mean", "f1_drop"),
        ("maximum_similarity", "auroc_drop"),
        ("lean", "auroc_drop"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    scatter = None
    for axis, (score_metric, performance_metric) in zip(axes.ravel(), panels):
        current = draw_scatter_panel(
            axis,
            merged,
            score_metric,
            performance_metric,
            reference,
            aggregation,
            prevalence_normalizer,
        )
        if current is not None:
            scatter = current

    figure.suptitle(
        "Outcome-class similarity scores and hospital-73 model transportability",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        f"{REFERENCE_LABEL}; {aggregation.replace('_', ' ')}",
        ha="center",
        fontsize=11,
        color="#4b5563",
    )
    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Target-hospital mortality prevalence")
    figure.text(
        0.5,
        0.015,
        "Each point is one target hospital; labels are hospital IDs; point size reflects cohort size.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.08, right=0.88, bottom=0.08, top=0.91, hspace=0.30, wspace=0.25
    )
    figure.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_all_score_scatter_grid(
    merged,
    performance_metric,
    reference,
    aggregation,
    out_path,
):
    piece = merged.loc[
        (merged["reference"] == reference) & (merged["aggregation"] == aggregation)
    ]
    if len(piece) == 0:
        return

    prevalence_min = float(piece["prevalence"].min())
    prevalence_max = float(piece["prevalence"].max())
    if prevalence_min == prevalence_max:
        prevalence_min -= 0.01
        prevalence_max += 0.01
    prevalence_normalizer = Normalize(vmin=prevalence_min, vmax=prevalence_max)

    available_scores = [
        score_name
        for score_name in SCORE_LABELS
        if score_name in set(piece["score_metric"])
    ]
    n_columns = 4
    n_rows = int(math.ceil(len(available_scores) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.1 * n_columns, 4.2 * n_rows),
        squeeze=False,
    )
    scatter = None
    for axis, score_name in zip(axes.ravel(), available_scores):
        current = draw_scatter_panel(
            axis,
            merged,
            score_name,
            performance_metric,
            reference,
            aggregation,
            prevalence_normalizer,
        )
        if current is not None:
            scatter = current
    for axis in axes.ravel()[len(available_scores) :]:
        axis.set_axis_off()

    figure.suptitle(
        f"All scalar similarity definitions vs {PERFORMANCE_LABELS[performance_metric]}",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )
    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes.ravel().tolist(),
            fraction=0.018,
            pad=0.015,
        )
        colorbar.set_label("Target-hospital mortality prevalence")
    figure.subplots_adjust(
        left=0.06, right=0.91, bottom=0.08, top=0.91, hspace=0.36, wspace=0.28
    )
    figure.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_similarity_score_distributions(hospital_scores, aggregation, out_path):
    piece = hospital_scores.loc[hospital_scores["aggregation"] == aggregation].copy()
    if len(piece) == 0:
        return

    available_scores = [
        score_name
        for score_name in SCORE_LABELS
        if score_name in set(piece["score_metric"])
    ]
    n_columns = 4
    n_rows = int(math.ceil(len(available_scores) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.1 * n_columns, 3.9 * n_rows),
        squeeze=False,
    )

    color_map = plt.get_cmap("viridis")
    for axis, score_name in zip(axes.ravel(), available_scores):
        score_piece = piece.loc[piece["score_metric"] == score_name].copy()
        score_piece = score_piece.dropna(subset=["hospital_score"])
        score_piece = score_piece.sort_values("hospital_score")
        values = score_piece["hospital_score"].to_numpy(dtype=float)
        if len(values) == 0:
            axis.set_axis_off()
            continue

        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        vertical_offsets = np.linspace(-0.075, 0.075, len(score_piece))
        axis.hlines(0, values.min(), values.max(), color="#aeb7c4", linewidth=2.0)
        axis.hlines(0, q1, q3, color="#334155", linewidth=7.0, alpha=0.78)
        axis.scatter(
            [median],
            [0],
            marker="D",
            s=72,
            color="#f97316",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="median",
        )
        colors = color_map(np.linspace(0.12, 0.88, len(score_piece)))
        axis.scatter(
            values,
            vertical_offsets,
            s=62,
            c=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for (_, row), y_offset in zip(score_piece.iterrows(), vertical_offsets):
            axis.annotate(
                str(row["hospitalid"]),
                (row["hospital_score"], y_offset),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#374151",
            )

        axis.set_yticks([])
        axis.set_xlabel("Hospital-level score")
        axis.set_title(SCORE_LABELS[score_name])
        axis.grid(True, axis="x")
        axis.text(
            0.02,
            0.93,
            f"min={values.min():.4f}\nmedian={median:.4f}\nmax={values.max():.4f}\n"
            f"range={values.max() - values.min():.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d5dae3",
                "alpha": 0.92,
            },
        )

    for axis in axes.ravel()[len(available_scores) :]:
        axis.set_axis_off()

    figure.suptitle(
        "Distribution and range of hospital-level similarity scores",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        0.965,
        f"To hospital 73 outcome classes; {aggregation.replace('_', ' ')}; "
        "each labeled point is one target hospital",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    figure.text(
        0.5,
        0.015,
        "Thin line: min-max | thick line: IQR | orange diamond: median",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.985,
        bottom=0.08,
        top=0.90,
        hspace=0.38,
        wspace=0.24,
    )
    figure.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_performance_distributions(hospital_performance, out_path):
    panels = [
        ("external_f1", "External F1"),
        ("external_auroc", "External AUROC"),
        ("external_auprc", "External AUPRC"),
        ("f1_drop", "F1 drop"),
        ("auroc_drop", "AUROC drop"),
        ("auprc_drop", "AUPRC drop"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.2), squeeze=False)
    color_map = plt.get_cmap("plasma")

    for axis, (metric_name, display_name) in zip(axes.ravel(), panels):
        piece = hospital_performance[["hospitalid", metric_name]].dropna().copy()
        piece = piece.sort_values(metric_name)
        values = piece[metric_name].to_numpy(dtype=float)
        if len(values) == 0:
            axis.set_axis_off()
            continue

        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        vertical_offsets = np.linspace(-0.075, 0.075, len(piece))
        axis.hlines(0, values.min(), values.max(), color="#aeb7c4", linewidth=2.0)
        axis.hlines(0, q1, q3, color="#334155", linewidth=7.0, alpha=0.78)
        axis.scatter(
            [median],
            [0],
            marker="D",
            s=72,
            color="#10b981",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        colors = color_map(np.linspace(0.12, 0.88, len(piece)))
        axis.scatter(
            values,
            vertical_offsets,
            s=64,
            c=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for (_, row), y_offset in zip(piece.iterrows(), vertical_offsets):
            axis.annotate(
                str(row["hospitalid"]),
                (row[metric_name], y_offset),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=8,
                color="#374151",
            )

        axis.set_yticks([])
        axis.set_xlabel(display_name)
        axis.set_title(display_name)
        axis.grid(True, axis="x")
        axis.text(
            0.02,
            0.93,
            f"min={values.min():.4f}\nmedian={median:.4f}\nmax={values.max():.4f}\n"
            f"range={values.max() - values.min():.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d5dae3",
                "alpha": 0.92,
            },
        )

    figure.suptitle(
        "Distribution and range of target-hospital model performance",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.015,
        "Each labeled point is one target hospital; drop = hospital 73 CV baseline - target performance.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.91,
        hspace=0.34,
        wspace=0.24,
    )
    figure.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Combine similarity-to-decedents and similarity-to-survivors "
            "into patient scalar scores, aggregate by hospital, and "
            "correlate them with model-performance drop."
        )
    )
    parser.add_argument(
        "--patient-level-file",
        type=Path,
        default=DEFAULT_PATIENT_LEVEL_FILE,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--source-cv-f1", type=float, required=True)
    parser.add_argument(
        "--source-cv-auroc",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--source-cv-auprc",
        type=float,
        required=True,
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_removed = remove_stale_within_hospital_outputs(args.outdir)
    if n_removed:
        print(
            f"Removed {n_removed} stale within-hospital figure(s) from output directory."
        )
    set_poster_style()

    print(
        f"Loading patient-level outcome-class similarities: {args.patient_level_file}"
    )
    frame = load_patient_level(args.patient_level_file)
    print(
        f"n_patients={len(frame):,}, "
        f"n_target_hospitals={frame['hospitalid'].nunique()}"
    )
    print(
        f"Hospital 73 CV baselines: F1={args.source_cv_f1:.6f}, "
        f"AUROC={args.source_cv_auroc:.6f}, "
        f"AUPRC={args.source_cv_auprc:.6f}; "
        f"threshold={args.threshold:.9f}"
    )

    own_hospital_columns = [
        column
        for column in frame.columns
        if column.startswith("similarity_to_own_") or "within_hospital" in column
    ]
    if own_hospital_columns:
        frame = frame.drop(columns=own_hospital_columns)
        print(
            "Removed old own-hospital exploratory columns: "
            + ", ".join(own_hospital_columns)
        )

    frame, to_source_scores = add_scalar_scores(
        frame,
        "similarity_to_source_decedents",
        "similarity_to_source_survivors",
        "to_source",
    )

    hospital_scores = aggregate_patient_scores(
        frame,
        to_source_scores,
        "to_source",
    )
    hospital_performance = compute_hospital_performance(
        frame,
        args.threshold,
        args.source_cv_f1,
        args.source_cv_auroc,
        args.source_cv_auprc,
    )
    correlations, merged = compute_correlations(
        hospital_scores,
        hospital_performance,
    )
    score_distributions = summarize_hospital_scores(hospital_scores)
    performance_distributions = summarize_hospital_performance(hospital_performance)

    patient_path = args.outdir / "patient_scalar_similarity_scores.csv.gz"
    performance_path = args.outdir / "hospital_model_performance_and_drop.csv"
    hospital_scores_path = args.outdir / "hospital_scalar_similarity_scores_long.csv"
    correlations_path = args.outdir / "similarity_score_drop_correlations.csv"
    merged_path = args.outdir / "hospital_scores_with_performance_long.csv"
    score_distribution_path = (
        args.outdir / "hospital_similarity_score_distribution_summary.csv"
    )
    performance_distribution_path = (
        args.outdir / "hospital_performance_distribution_summary.csv"
    )

    frame.to_csv(patient_path, index=False, compression="gzip")
    hospital_performance.to_csv(performance_path, index=False)
    hospital_scores.to_csv(hospital_scores_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    merged.to_csv(merged_path, index=False)
    score_distributions.to_csv(score_distribution_path, index=False)
    performance_distributions.to_csv(
        performance_distribution_path,
        index=False,
    )

    references = sorted(correlations["reference"].unique())
    aggregations = sorted(correlations["aggregation"].unique())
    for reference in references:
        for aggregation in aggregations:
            figure_tag = f"{reference}_{aggregation}"
            plot_correlation_heatmap(
                correlations,
                reference,
                aggregation,
                args.outdir / f"correlation_heatmap_{figure_tag}.png",
            )

    primary_reference = "to_source"
    primary_aggregation = "overall_median"
    plot_primary_scatter_grid(
        merged,
        primary_reference,
        primary_aggregation,
        args.outdir / "POSTER_primary_similarity_vs_drop.png",
    )
    for performance_metric in ["f1_drop", "auroc_drop", "auprc_drop"]:
        plot_all_score_scatter_grid(
            merged,
            performance_metric,
            primary_reference,
            primary_aggregation,
            args.outdir / f"all_scores_vs_{performance_metric}.png",
        )

    for aggregation in aggregations:
        plot_similarity_score_distributions(
            hospital_scores,
            aggregation,
            args.outdir / f"hospital_similarity_distributions_{aggregation}.png",
        )
    plot_performance_distributions(
        hospital_performance,
        args.outdir / "hospital_performance_distributions.png",
    )

    primary_results = correlations.loc[
        (correlations["reference"] == primary_reference)
        & (correlations["aggregation"] == primary_aggregation)
    ].copy()
    primary_results["abs_spearman_r"] = primary_results["spearman_r"].abs()
    primary_results = primary_results.sort_values(
        ["performance_metric", "abs_spearman_r"],
        ascending=[True, False],
    )

    print("\n" + "=" * 110)
    print("PRIMARY RESULTS: TO-SOURCE SCORES, OVERALL MEDIAN ACROSS PATIENTS")
    print("=" * 110)
    print(
        primary_results[
            [
                "score_metric",
                "performance_metric",
                "n_hospitals",
                "spearman_r",
                "spearman_p",
                "spearman_q_bh",
                "pearson_r",
                "pearson_p",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("HOSPITAL-LEVEL SIMILARITY RANGE: TO-SOURCE, OVERALL MEDIAN")
    print("=" * 110)
    primary_score_distributions = score_distributions.loc[
        score_distributions["aggregation"] == primary_aggregation
    ].copy()
    primary_score_distributions["score_order"] = primary_score_distributions[
        "score_metric"
    ].map({name: index for index, name in enumerate(SCORE_LABELS)})
    primary_score_distributions = primary_score_distributions.sort_values("score_order")
    print(
        primary_score_distributions[
            [
                "score_metric",
                "minimum_hospitalid",
                "maximum_hospitalid",
                "n_hospitals",
                "minimum",
                "q1",
                "median",
                "mean",
                "q3",
                "maximum",
                "range",
                "standard_deviation",
                "iqr",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("HOSPITAL-LEVEL PERFORMANCE RANGE")
    print("=" * 110)
    print(performance_distributions.round(6).to_string(index=False))

    print("\nSaved:")
    print(f"  Patient scalar scores: {patient_path}")
    print(f"  Hospital performance/drop: {performance_path}")
    print(f"  Hospital similarity scores: {hospital_scores_path}")
    print(f"  Correlations: {correlations_path}")
    print(f"  Similarity range summary: {score_distribution_path}")
    print(f"  Performance range summary: {performance_distribution_path}")
    print(f"  Poster figure: {args.outdir / 'POSTER_primary_similarity_vs_drop.png'}")
    print(
        "  Similarity distributions: "
        f"{args.outdir / 'hospital_similarity_distributions_overall_median.png'}"
    )
    print(
        "  Performance distributions: "
        f"{args.outdir / 'hospital_performance_distributions.png'}"
    )
    print(f"  All outputs: {args.outdir}")


if __name__ == "__main__":
    main()
