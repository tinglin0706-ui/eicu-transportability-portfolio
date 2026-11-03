"""Plot class-balanced and outcome-stratified absolute lean against AUPRC drop.

Absolute lean is the absolute difference between mean cosine similarity to
source decedents and source survivors; it is not overall source similarity."""


import argparse
from pathlib import Path

import matplotlib


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

import balanced_similarity_drop_correlation as base


DEFAULT_OUTDIR = (
    base.PROJECT_ROOT
    / "results"
    / "final_1000_0"
    / "poster_absolute_lean_auprc_figures"
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create three poster panels in two output figures for absolute-lean "
            "similarity versus target-hospital AUPRC drop."
        )
    )
    parser.add_argument(
        "--patient-level-file",
        type=Path,
        default=base.DEFAULT_PATIENT_LEVEL_FILE,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Hospital 73 source-selected threshold; shared by every target.",
    )
    parser.add_argument(
        "--source-cv-f1",
        type=float,
        required=True,
    )
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


def add_absolute_lean_similarity(frame):
    result = frame.copy()
    similarity_to_decedents = pd.to_numeric(
        result["similarity_to_source_decedents"], errors="raise"
    )
    similarity_to_survivors = pd.to_numeric(
        result["similarity_to_source_survivors"], errors="raise"
    )
    result["absolute_lean_similarity"] = (
        similarity_to_decedents - similarity_to_survivors
    ).abs()
    return result


def summarize_absolute_lean_by_hospital(frame):
    rows = []
    for hospitalid, hospital_frame in frame.groupby("hospitalid", sort=True):
        decedents = hospital_frame.loc[
            hospital_frame["label"] == 1, "absolute_lean_similarity"
        ].dropna()
        survivors = hospital_frame.loc[
            hospital_frame["label"] == 0, "absolute_lean_similarity"
        ].dropna()

        decedent_mean = float(decedents.mean()) if len(decedents) else np.nan
        survivor_mean = float(survivors.mean()) if len(survivors) else np.nan
        class_balanced_mean = (
            float((decedent_mean + survivor_mean) / 2.0)
            if np.isfinite(decedent_mean) and np.isfinite(survivor_mean)
            else np.nan
        )

        rows.append(
            {
                "hospitalid": hospitalid,
                "decedents_mean_absolute_lean_similarity": decedent_mean,
                "survivors_mean_absolute_lean_similarity": survivor_mean,
                "class_balanced_mean_absolute_lean_similarity": (class_balanced_mean),
                "n_decedents": int(len(decedents)),
                "n_survivors": int(len(survivors)),
                "n_patients": int(len(hospital_frame)),
            }
        )
    return pd.DataFrame(rows)


def scale_marker_sizes(counts):
    values = pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0:
        return values
    if np.nanmax(values) == np.nanmin(values):
        return np.full(len(values), 145.0)
    return 85.0 + 180.0 * (
        (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
    )


def format_p_value(value):
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def annotate_hospitals_without_overlap(axis, plot_data, x_column):
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_box = axis.get_window_extent(renderer=renderer)

    point_locations = axis.transData.transform(
        np.column_stack(
            [
                plot_data[x_column].to_numpy(dtype=float),
                plot_data["auprc_drop"].to_numpy(dtype=float),
            ]
        )
    )
    point_boxes = [
        Bbox.from_extents(x - 7, y - 7, x + 7, y + 7) for x, y in point_locations
    ]
    placed_label_boxes = []

    x_min = float(plot_data[x_column].min())
    x_max = float(plot_data[x_column].max())
    y_min = float(plot_data["auprc_drop"].min())
    y_max = float(plot_data["auprc_drop"].max())
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)

    for point_index, (_, row) in enumerate(plot_data.iterrows()):
        x_value = float(row[x_column])
        y_value = float(row["auprc_drop"])
        x_fraction = (x_value - x_min) / x_span
        y_fraction = (y_value - y_min) / y_span
        horizontal_sign = -1 if x_fraction > 0.70 else 1
        vertical_sign = -1 if y_fraction > 0.78 else 1

        candidate_offsets = [
            (7 * horizontal_sign, 7 * vertical_sign),
            (7 * horizontal_sign, -9 * vertical_sign),
            (-8 * horizontal_sign, 7 * vertical_sign),
            (-8 * horizontal_sign, -9 * vertical_sign),
            (13 * horizontal_sign, 1),
            (-13 * horizontal_sign, 1),
            (1, 13 * vertical_sign),
            (1, -15 * vertical_sign),
            (16 * horizontal_sign, 13 * vertical_sign),
            (-16 * horizontal_sign, -15 * vertical_sign),
        ]

        chosen_box = None
        for x_offset, y_offset in candidate_offsets:
            annotation = axis.annotate(
                str(row["hospitalid"]),
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if x_offset >= 0 else "right",
                va="bottom" if y_offset >= 0 else "top",
                fontsize=9.5,
                color="#1f2937",
                annotation_clip=True,
                zorder=4,
            )
            figure.canvas.draw()
            label_box = annotation.get_window_extent(renderer=renderer).expanded(
                1.04, 1.10
            )
            inside_axis = (
                label_box.x0 >= axis_box.x0
                and label_box.x1 <= axis_box.x1
                and label_box.y0 >= axis_box.y0
                and label_box.y1 <= axis_box.y1
            )
            overlaps_label = any(
                label_box.overlaps(other_box) for other_box in placed_label_boxes
            )
            overlaps_other_point = any(
                label_box.overlaps(point_box)
                for other_index, point_box in enumerate(point_boxes)
                if other_index != point_index
            )
            if inside_axis and not overlaps_label and not overlaps_other_point:
                chosen_box = label_box
                break
            annotation.remove()

        if chosen_box is None:
            x_offset = -7 if x_fraction > 0.5 else 7
            y_offset = -9 if y_fraction > 0.5 else 7
            annotation = axis.annotate(
                str(row["hospitalid"]),
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="right" if x_offset < 0 else "left",
                va="top" if y_offset < 0 else "bottom",
                fontsize=9.5,
                color="#1f2937",
                annotation_clip=True,
                zorder=4,
            )
            figure.canvas.draw()
            chosen_box = annotation.get_window_extent(renderer=renderer).expanded(
                1.04, 1.10
            )
        placed_label_boxes.append(chosen_box)


def draw_scatter_panel(
    axis,
    hospital_table,
    x_column,
    count_column,
    panel_title,
    x_label,
    color_norm,
    x_limits=None,
    y_limits=None,
):
    plot_data = hospital_table.dropna(
        subset=[x_column, "auprc_drop", "prevalence"]
    ).copy()
    if len(plot_data) < 3:
        raise ValueError(
            f"Need at least 3 hospitals for {panel_title}; found {len(plot_data)}"
        )

    scatter = axis.scatter(
        plot_data[x_column],
        plot_data["auprc_drop"],
        c=plot_data["prevalence"],
        cmap="viridis",
        norm=color_norm,
        s=scale_marker_sizes(plot_data[count_column]),
        edgecolor="white",
        linewidth=1.3,
        alpha=0.94,
        zorder=3,
    )

    x_values = plot_data[x_column].to_numpy(dtype=float)
    y_values = plot_data["auprc_drop"].to_numpy(dtype=float)
    if np.unique(x_values).size >= 2:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 200)
        axis.plot(
            x_line,
            intercept + slope * x_line,
            color="#334155",
            linewidth=2.4,
            zorder=2,
        )

    statistics = base.safe_correlations(plot_data[x_column], plot_data["auprc_drop"])
    axis.text(
        0.5,
        1.015,
        f"Spearman rho = {statistics['spearman_r']:.2f}   |   "
        f"p = {format_p_value(statistics['spearman_p'])}   |   "
        f"n = {statistics['n_hospitals']} hospitals",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#334155",
        clip_on=False,
    )

    axis.set_xlabel(x_label, fontsize=13, labelpad=10)
    axis.set_ylabel(
        "AUPRC drop\n(source hospital CV - target hospital)",
        fontsize=12,
        labelpad=10,
    )
    axis.set_title(panel_title, fontsize=17, fontweight="bold", pad=40)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="y", color="#dbe2ea", linewidth=0.65, alpha=0.50)
    if x_limits is None:
        axis.margins(x=0.12)
    else:
        axis.set_xlim(*x_limits)
    if y_limits is None:
        axis.margins(y=0.09)
    else:
        axis.set_ylim(*y_limits)
    annotate_hospitals_without_overlap(axis, plot_data, x_column)
    return scatter, statistics


def plot_outcome_stratified_figure(hospital_table, output_path):
    prevalence_min = float(hospital_table["prevalence"].min())
    prevalence_max = float(hospital_table["prevalence"].max())
    if prevalence_min == prevalence_max:
        prevalence_min -= 0.01
        prevalence_max += 0.01
    color_norm = Normalize(vmin=prevalence_min, vmax=prevalence_max)

    decedent_x = pd.to_numeric(
        hospital_table["decedents_mean_absolute_lean_similarity"],
        errors="coerce",
    ).dropna()
    survivor_x = pd.to_numeric(
        hospital_table["survivors_mean_absolute_lean_similarity"],
        errors="coerce",
    ).dropna()
    combined_x = pd.concat([decedent_x, survivor_x], ignore_index=True)
    combined_y = pd.to_numeric(hospital_table["auprc_drop"], errors="coerce").dropna()
    x_span = float(combined_x.max() - combined_x.min())
    y_span = float(combined_y.max() - combined_y.min())
    x_padding = 0.10 * x_span if x_span > 0 else 0.01
    y_padding = 0.10 * y_span if y_span > 0 else 0.01
    shared_x_limits = (
        float(combined_x.min() - x_padding),
        float(combined_x.max() + x_padding),
    )
    shared_y_limits = (
        float(combined_y.min() - y_padding),
        float(combined_y.max() + y_padding),
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 10.3),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.84,
        top=0.83,
        bottom=0.11,
        hspace=0.40,
    )
    decedent_scatter, decedent_statistics = draw_scatter_panel(
        axis=axes[0],
        hospital_table=hospital_table,
        x_column="decedents_mean_absolute_lean_similarity",
        count_column="n_patients",
        panel_title="A. Patients who died",
        x_label="",
        color_norm=color_norm,
        x_limits=shared_x_limits,
        y_limits=shared_y_limits,
    )
    _, survivor_statistics = draw_scatter_panel(
        axis=axes[1],
        hospital_table=hospital_table,
        x_column="survivors_mean_absolute_lean_similarity",
        count_column="n_patients",
        panel_title="B. Patients who survived",
        x_label="",
        color_norm=color_norm,
        x_limits=shared_x_limits,
        y_limits=shared_y_limits,
    )
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    figure.suptitle(
        "Outcome-Stratified Absolute Lean Similarity vs AUPRC Drop",
        fontsize=22,
        fontweight="bold",
        y=0.965,
    )
    figure.supxlabel(
        "Mean absolute lean",
        x=0.485,
        y=0.060,
        fontsize=13,
    )
    figure.supylabel(
        "AUPRC drop\n(source hospital CV - target hospital)",
        x=0.035,
        fontsize=12,
    )
    colorbar_axis = figure.add_axes([0.875, 0.11, 0.025, 0.72])
    colorbar = figure.colorbar(decedent_scatter, cax=colorbar_axis)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_label("Target mortality rate", fontsize=11)
    figure.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return decedent_statistics, survivor_statistics


def plot_class_balanced_figure(hospital_table, output_path):
    prevalence_min = float(hospital_table["prevalence"].min())
    prevalence_max = float(hospital_table["prevalence"].max())
    if prevalence_min == prevalence_max:
        prevalence_min -= 0.01
        prevalence_max += 0.01
    color_norm = Normalize(vmin=prevalence_min, vmax=prevalence_max)

    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    figure.subplots_adjust(left=0.14, right=0.84, top=0.76, bottom=0.18)
    scatter, statistics = draw_scatter_panel(
        axis=axis,
        hospital_table=hospital_table,
        x_column="class_balanced_mean_absolute_lean_similarity",
        count_column="n_patients",
        panel_title=("Class-Balanced Absolute Lean Similarity vs AUPRC Drop"),
        x_label="Class-balanced mean absolute lean similarity",
        color_norm=color_norm,
    )
    colorbar_axis = figure.add_axes([0.87, 0.18, 0.025, 0.58])
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_label("Target mortality rate", fontsize=11)
    figure.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return statistics


def main():
    args = build_arg_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    base.set_poster_style()

    print(f"Loading patient-level file: {args.patient_level_file}")
    patient_frame = base.load_patient_level(args.patient_level_file)
    patient_frame = add_absolute_lean_similarity(patient_frame)

    similarity_summary = summarize_absolute_lean_by_hospital(patient_frame)
    performance_summary = base.compute_hospital_performance(
        patient_frame,
        args.threshold,
        args.source_cv_f1,
        args.source_cv_auroc,
        args.source_cv_auprc,
    )
    hospital_table = similarity_summary.merge(
        performance_summary.drop(columns=["n_patients", "n_decedents"]),
        on="hospitalid",
        how="inner",
        validate="one_to_one",
    )

    outcome_path = args.outdir / "poster_absolute_lean_auprc_outcome_stratified.png"
    class_balanced_path = args.outdir / "poster_absolute_lean_auprc_class_balanced.png"
    decedent_statistics, survivor_statistics = plot_outcome_stratified_figure(
        hospital_table, outcome_path
    )
    class_balanced_statistics = plot_class_balanced_figure(
        hospital_table, class_balanced_path
    )

    correlation_rows = []
    for analysis, statistics in [
        ("target_decedents_mean", decedent_statistics),
        ("target_survivors_mean", survivor_statistics),
        ("class_balanced_mean", class_balanced_statistics),
    ]:
        correlation_rows.append(
            {
                "analysis": analysis,
                "similarity_metric": "absolute_lean_similarity",
                "performance_metric": "auprc_drop",
                **statistics,
            }
        )

    print(f"Saved: {outcome_path}")
    print(f"Saved: {class_balanced_path}")

    hospital_table.to_csv(
        args.outdir / "poster_absolute_lean_auprc_plot_data.csv", index=False
    )
    correlation_summary = pd.DataFrame(correlation_rows)
    correlation_summary.to_csv(
        args.outdir / "poster_absolute_lean_auprc_correlations.csv", index=False
    )

    print("\nPoster correlations:")
    print(
        correlation_summary[
            [
                "analysis",
                "n_hospitals",
                "spearman_r",
                "spearman_p",
                "pearson_r",
                "pearson_p",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nAll outputs: {args.outdir}")


if __name__ == "__main__":
    main()
