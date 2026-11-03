"""Evaluate source-hospital cross-validation and external hospital performance.

Thresholds are selected within source training folds. The final source model
is evaluated at a fixed threshold on each target hospital."""


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]


PROJECT_ROOT = PORTFOLIO_ROOT

sys.path.append(str(PROJECT_ROOT / "scripts" / "4_ML_models"))
from pooled_svd_embeddings import load_fiddle_matrix, read_patient_metadata


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
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "final_1000_0" / "transportability"


def normalize_hospital_id(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "null"}:
        return np.nan
    try:
        return str(int(float(value)))
    except Exception:
        return value


def choose_best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.zeros_like(denominator, dtype=float)
    valid = denominator > 0
    f1[valid] = 2 * precision[valid] * recall[valid] / denominator[valid]
    return float(thresholds[int(np.nanargmax(f1))])


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "auroc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "auprc": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_xgb_model(
    scale_pos_weight, random_state, n_estimators, max_depth, learning_rate
):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def run_source_hospital_cv(
    x_source,
    y_source,
    n_splits,
    val_size,
    random_state,
    n_estimators,
    max_depth,
    learning_rate,
):
    """5-fold CV on the source hospital. Returns per-fold metrics and
    the mean tuned threshold (used later for external evaluation)."""

    n_positive = int(y_source.sum())
    n_negative = int(len(y_source) - n_positive)
    actual_splits = min(n_splits, n_positive, n_negative)

    if actual_splits < 2:
        raise ValueError(
            "Source hospital does not have enough positive/negative cases for CV."
        )

    cv = StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=random_state
    )

    fold_rows = []

    for fold, (outer_train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(y_source)), y_source)
    ):
        y_outer_train = y_source[outer_train_idx]

        model_train_local_idx, val_local_idx = train_test_split(
            np.arange(len(outer_train_idx)),
            test_size=val_size,
            random_state=random_state + fold,
            stratify=y_outer_train,
        )
        model_train_idx = outer_train_idx[model_train_local_idx]
        val_idx = outer_train_idx[val_local_idx]

        x_train, y_train = x_source[model_train_idx], y_source[model_train_idx]
        x_val, y_val = x_source[val_idx], y_source[val_idx]
        x_test, y_test = x_source[test_idx], y_source[test_idx]

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = make_xgb_model(
            scale_pos_weight,
            random_state + fold,
            n_estimators,
            max_depth,
            learning_rate,
        )
        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)[:, 1]
        tuned_threshold = choose_best_f1_threshold(y_val, val_prob)

        test_prob = model.predict_proba(x_test)[:, 1]
        metrics = calculate_metrics(y_test, test_prob, tuned_threshold)
        metrics["fold"] = fold
        fold_rows.append(metrics)

        print(
            f"  Fold {fold}: n_train={len(y_train)}, n_val={len(y_val)}, "
            f"n_test={len(y_test)}, threshold={tuned_threshold:.3f}, "
            f"F1={metrics['f1']:.3f}, AUROC={metrics['auroc']:.3f}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_rows)
    mean_threshold = float(fold_df["threshold"].mean())

    return fold_df, mean_threshold


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cross-hospital transportability experiment."
    )
    parser.add_argument(
        "--source-hospital", required=True, help="hospitalid to use as the source."
    )
    parser.add_argument("--task", default="mortality_48h")
    parser.add_argument(
        "--fiddle-output-dir", type=Path, default=DEFAULT_FIDDLE_OUTPUT_DIR
    )
    parser.add_argument("--population-file", type=Path, default=DEFAULT_POPULATION_FILE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--label-col", default="mortality_LABEL")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    source_hospital = normalize_hospital_id(args.source_hospital)

    print(f"Loading task: {args.task}")
    ids, feature_matrix, input_audit = load_fiddle_matrix(
        args.fiddle_output_dir,
        feature_set="all",
    )
    row_info = read_patient_metadata(args.population_file, ids)

    if args.label_col not in row_info.columns:

        pop = pd.read_csv(args.population_file)
        pop["ID"] = pop["ID"].astype("int64")
        row_info = row_info.merge(
            pop[["ID", args.label_col]], on="ID", how="left", validate="one_to_one"
        )

    if row_info[args.label_col].isna().any():
        raise ValueError(f"Missing {args.label_col} for some patients after merge.")

    row_info["hospitalid_norm"] = row_info["hospitalid"].apply(normalize_hospital_id)
    row_info[args.label_col] = row_info[args.label_col].astype(int)

    y_all = row_info[args.label_col].to_numpy()
    hospital_all = row_info["hospitalid_norm"].to_numpy()

    source_mask = hospital_all == str(source_hospital)
    if source_mask.sum() == 0:
        raise ValueError(f"No rows found for source hospital {source_hospital}")

    x_source = feature_matrix[source_mask]
    y_source = y_all[source_mask]

    print(
        f"Source hospital {source_hospital}: "
        f"n={len(y_source)}, positive={int(y_source.sum())}, "
        f"prevalence={y_source.mean():.4f}"
    )

    print("\nRunning 5-fold CV on source hospital...")
    fold_df, mean_threshold = run_source_hospital_cv(
        x_source,
        y_source,
        args.n_splits,
        args.val_size,
        args.random_state,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
    )

    cv_summary = (
        fold_df[["f1", "auroc", "auprc", "precision", "recall"]].mean().to_dict()
    )
    cv_summary = {f"cv_mean_{k}": v for k, v in cv_summary.items()}
    print(f"\nCV mean: {cv_summary}")
    print(f"Mean tuned threshold across folds: {mean_threshold:.4f}")

    fold_df.to_csv(
        args.outdir / f"{args.task}_source{source_hospital}_cv_folds.csv", index=False
    )

    print(f"\nRefitting on ALL source hospital ({source_hospital}) data...")
    scale_pos_weight = (y_source == 0).sum() / max((y_source == 1).sum(), 1)
    final_model = make_xgb_model(
        scale_pos_weight,
        args.random_state,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
    )
    final_model.fit(x_source, y_source)

    other_hospitals = sorted(set(hospital_all[~source_mask]))
    print(f"\nEvaluating on {len(other_hospitals)} other hospitals...")

    result_rows = []
    prediction_rows = []

    for target_hospital in other_hospitals:
        target_mask = hospital_all == target_hospital
        x_target = feature_matrix[target_mask]
        y_target = y_all[target_mask]

        target_prob = final_model.predict_proba(x_target)[:, 1]
        prediction_rows.append(
            pd.DataFrame(
                {
                    "ID": row_info.loc[target_mask, "ID"].to_numpy(),
                    "hospitalid": hospital_all[target_mask],
                    "label": y_target,
                    "predicted_probability": target_prob,
                }
            )
        )
        target_metrics = calculate_metrics(y_target, target_prob, mean_threshold)

        row = {
            "source_hospital": source_hospital,
            "target_hospital": target_hospital,
            "mean_cv_threshold": mean_threshold,
        }
        row.update(cv_summary)
        row.update({f"external_{k}": v for k, v in target_metrics.items()})

        for metric in ["f1", "auroc", "auprc"]:
            row[f"difference_{metric}"] = (
                row[f"cv_mean_{metric}"] - row[f"external_{metric}"]
            )

        result_rows.append(row)

        print(
            f"  {source_hospital} -> {target_hospital}: "
            f"n={target_metrics['n']}, "
            f"external_F1={target_metrics['f1']:.3f}, "
            f"diff_F1={row['difference_f1']:.3f}, "
            f"external_AUROC={target_metrics['auroc']:.3f}, "
            f"diff_AUROC={row['difference_auroc']:.3f}"
        )

    if not result_rows:
        raise ValueError("At least one target hospital is required.")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.outdir / f"{args.task}_source{source_hospital}_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    results_df = pd.DataFrame(result_rows)
    results_path = (
        args.outdir / f"{args.task}_source{source_hospital}_transportability.csv"
    )
    results_df.to_csv(results_path, index=False)

    run_config = {
        "task": args.task,
        "source_hospital": source_hospital,
        "n_splits": args.n_splits,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "mean_cv_threshold": mean_threshold,
        "feature_matrix_shape": list(feature_matrix.shape),
        "input_audit": input_audit,
    }
    with (args.outdir / f"{args.task}_source{source_hospital}_run_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(
        results_df[
            [
                "target_hospital",
                "external_f1",
                "difference_f1",
                "external_auroc",
                "difference_auroc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
