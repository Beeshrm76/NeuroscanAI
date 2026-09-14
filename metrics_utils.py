"""
Shared metric computation used by evaluate_all.py and external_validation.py,
so both report metrics the exact same way and results are directly
comparable.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_and_save_metrics(y_true, y_pred, class_names, output_dir, run_name):
    """
    Computes accuracy, precision, recall, F1 (macro + weighted), a full
    per-class report, and a confusion matrix for one model/ensemble run.
    Saves a per-class CSV and a confusion matrix PNG under output_dir.
    Returns a flat dict summary suitable for appending to a comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Per-class CSV (precision/recall/f1/support for every class)
    per_class_path = os.path.join(output_dir, f"{run_name}_per_class_metrics.csv")
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for cls in class_names:
            row = report_dict.get(cls, {})
            writer.writerow([
                cls,
                round(row.get("precision", 0), 4),
                round(row.get("recall", 0), 4),
                round(row.get("f1-score", 0), 4),
                int(row.get("support", 0)),
            ])

    # Confusion matrix (raw counts) as CSV + PNG heatmap
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_csv_path = os.path.join(output_dir, f"{run_name}_confusion_matrix.csv")
    with open(cm_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for cls, row in zip(class_names, cm):
            writer.writerow([cls] + list(row))

    _save_confusion_matrix_png(
        cm, class_names,
        os.path.join(output_dir, f"{run_name}_confusion_matrix.png"),
        title=f"Confusion Matrix - {run_name}"
    )

    summary = {
        "run_name": run_name,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "n_samples": int(len(y_true)),
    }
    return summary


def _save_confusion_matrix_png(cm, class_names, out_path, title):
    n = len(class_names)
    # Scale figure with class count so labels stay legible for ~40+ classes.
    fig_size = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_table(summaries, output_dir, filename="summary_metrics.csv"):
    """Write a comparison table across all evaluated models/ensembles."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if not summaries:
        return path
    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return path
