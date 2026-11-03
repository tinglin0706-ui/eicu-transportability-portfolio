"""Run the mortality transportability and patient-similarity analysis."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent


def run_script(relative_path, *arguments):
    command = [sys.executable, str(ROOT / relative_path), *map(str, arguments)]
    subprocess.run(command, check=True, cwd=ROOT)


def run_analysis(config, fiddle_dir, population_file, outdir):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fiddle_dir = Path(fiddle_dir).resolve()
    population_file = Path(population_file).resolve()
    source = str(config["source_hospital"])
    task = "mortality_48h"
    dimension = int(config["svd_components"])
    transport = outdir / "transportability"
    embeddings = outdir / "embeddings"
    similarity = outdir / "outcome_similarity"
    correlations = outdir / "correlations"
    figures = outdir / "figures"

    run_script(
        "scripts/4_ML_models/cross_hospital_transportability.py",
        "--source-hospital",
        source,
        "--fiddle-output-dir",
        fiddle_dir,
        "--population-file",
        population_file,
        "--outdir",
        transport,
        "--n-splits",
        config["cv_folds"],
        "--val-size",
        config["validation_fraction"],
        "--random-state",
        config["random_state"],
        "--n-estimators",
        config["n_estimators"],
        "--max-depth",
        config["max_depth"],
        "--learning-rate",
        config["learning_rate"],
    )
    run_script(
        "scripts/4_ML_models/pooled_svd_embeddings.py",
        "--fiddle-output-dir",
        fiddle_dir,
        "--population-file",
        population_file,
        "--outdir",
        embeddings,
        "--n-components",
        dimension,
        "--random-state",
        config["random_state"],
    )
    metrics = pd.read_csv(transport / f"{task}_source{source}_transportability.csv")
    threshold = float(metrics["mean_cv_threshold"].iloc[0])
    baseline_arguments = []
    for metric in ("f1", "auroc", "auprc"):
        baseline_arguments.extend(
            [f"--source-cv-{metric}", str(float(metrics[f"cv_mean_{metric}"].iloc[0]))]
        )
    run_script(
        "scripts/4_ML_models/similarity_by_outcome_class.py",
        "--source-hospital",
        source,
        "--population-file",
        population_file,
        "--embeddings-dir",
        embeddings,
        "--n-components",
        dimension,
        "--to-source-patient-level",
        transport / f"{task}_source{source}_predictions.csv.gz",
        "--threshold",
        threshold,
        "--outdir",
        similarity,
    )
    patient_scores = (
        similarity
        / f"{task}_source{source}_svd{dimension}_similarity_by_outcome_class.csv.gz"
    )
    for script, directory in [
        ("balanced_similarity_drop_correlation.py", correlations),
        ("poster_absolute_lean_auprc_figures.py", figures),
    ]:
        run_script(
            f"scripts/5_transportability/{script}",
            "--patient-level-file",
            patient_scores,
            "--outdir",
            directory,
            "--threshold",
            threshold,
            *baseline_arguments,
        )
    resolved = dict(
        config,
        task=task,
        fiddle_output_dir=str(fiddle_dir),
        population_file=str(population_file),
        outdir=str(outdir),
        selected_threshold=threshold,
    )
    (outdir / "analysis_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mortality.json")
    parser.add_argument("--fiddle-output-dir", type=Path, required=True)
    parser.add_argument("--population-file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Figure labels in the retained research scripts refer to source hospital 73.
    if str(config["source_hospital"]) != "73":
        parser.error("This curated analysis is configured for source hospital 73.")
    run_analysis(config, args.fiddle_output_dir, args.population_file, args.outdir)


if __name__ == "__main__":
    main()
