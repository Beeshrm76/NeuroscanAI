"""
Confidence calibration check.

WHY THIS FILE EXISTS
---------------------
`confidence < 50% = UNCERTAIN` in app.py is just an arbitrary threshold.
It has never been checked against the actual test set to see whether, say,
predictions the model is "90% confident" about are actually correct ~90%
of the time. A model can be badly overconfident (says 95% and is wrong
often) or underconfident. This script checks that directly by:

  1. Running the ensemble over the test set.
  2. Bucketing predictions into confidence bins (0-10%, 10-20%, ... 90-100%).
  3. For each bin, computing the ACTUAL accuracy of predictions in that bin.
  4. Plotting a reliability diagram (ideal = diagonal line) and computing
     Expected Calibration Error (ECE) - lower is better, 0 is perfect.
  5. Explicitly comparing accuracy of high-confidence (>=50%) vs
     low-confidence (<50%) predictions, to test whether the app's current
     threshold is actually doing meaningful uncertainty separation.

WHY THIS FILE CHANGED
-----------------------
ensemble_predict_generator() now builds its own per-model, correctly
preprocessed generator internally (each ensemble member needs its own
preprocess_input, not one shared generic rescale) - so it takes the test
directory PATH, not a pre-built generator. Passing the old
preprocess.get_data_generators() test_gen here would silently feed every
model the wrong preprocessing again.

Run this AFTER evaluate_all.py, using models trained on the patient-level
split.

Usage:
    python confidence_calibration.py
Outputs land in: evaluation_results/calibration/
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ensemble_utils import load_ensemble_models, ensemble_predict_generator

OUTPUT_DIR = "evaluation_results/calibration"
TEST_DIR = "dataset/Testing"
N_BINS = 10
THRESHOLD_TO_CHECK = 50.0  # matches app.py's current cutoff


def compute_calibration(y_true, y_pred, y_proba, n_bins=N_BINS):
    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count == 0:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
        bin_counts.append(int(count))

    n = len(confidences)
    ece = 0.0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        if count == 0 or np.isnan(acc):
            continue
        ece += (count / n) * abs(acc - conf)

    return bin_edges, bin_accs, bin_confs, bin_counts, ece


def plot_reliability_diagram(bin_edges, bin_accs, ece, out_path, title):
    n_bins = len(bin_accs)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    accs_plot = [a if not np.isnan(a) else 0 for a in bin_accs]
    ax.bar(bin_centers, accs_plot, width=1.0 / n_bins, edgecolor='black',
           alpha=0.7, label='Observed accuracy')
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{title}\nExpected Calibration Error (ECE) = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def check_threshold_separation(y_true, y_pred, y_proba, threshold_pct, output_dir, run_name):
    confidences = np.max(y_proba, axis=1) * 100
    correct = (y_pred == y_true).astype(int)

    high_mask = confidences >= threshold_pct
    low_mask = ~high_mask

    high_acc = correct[high_mask].mean() if high_mask.sum() else float('nan')
    low_acc = correct[low_mask].mean() if low_mask.sum() else float('nan')

    print(f"\n[{run_name}] Threshold check at {threshold_pct:.0f}%:")
    print(f"  High-confidence group (>= {threshold_pct:.0f}%): "
          f"{high_mask.sum()} predictions, accuracy = {high_acc*100:.2f}%")
    print(f"  Low-confidence group  (<  {threshold_pct:.0f}%): "
          f"{low_mask.sum()} predictions, accuracy = {low_acc*100:.2f}%")
    if not np.isnan(high_acc) and not np.isnan(low_acc):
        if high_acc > low_acc:
            print(f"  -> Threshold is doing its job: high-confidence predictions "
                  f"ARE more reliable ({(high_acc-low_acc)*100:.1f} pts higher accuracy).")
        else:
            print(f"  -> WARNING: high-confidence predictions are NOT more accurate "
                  f"than low-confidence ones. The 50% cutoff is not a meaningful "
                  f"reliability signal for this model - do not present it to users "
                  f"as a trust indicator without revisiting calibration.")

    path = os.path.join(output_dir, f"{run_name}_threshold_check.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "threshold_pct", "n_predictions", "accuracy"])
        writer.writerow(["high_confidence", threshold_pct, int(high_mask.sum()), round(float(high_acc), 4)])
        writer.writerow(["low_confidence", threshold_pct, int(low_mask.sum()), round(float(low_acc), 4)])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Evaluating calibration for the ensemble (DenseNet121 + MobileNetV2 + VGG16)...")
    ensemble_models = load_ensemble_models()
    # Pass the TEST DIRECTORY, not a generator - ensemble_predict_generator
    # builds its own correctly-preprocessed generator per model internally.
    y_true, y_pred, y_proba = ensemble_predict_generator(ensemble_models, TEST_DIR)

    bin_edges, bin_accs, bin_confs, bin_counts, ece = compute_calibration(y_true, y_pred, y_proba)

    print(f"\nExpected Calibration Error (ECE): {ece:.4f} (0 = perfectly calibrated)")
    for i in range(N_BINS):
        lo, hi = bin_edges[i] * 100, bin_edges[i + 1] * 100
        acc = bin_accs[i]
        acc_str = f"{acc*100:.1f}%" if not np.isnan(acc) else "n/a (empty bin)"
        print(f"  Confidence [{lo:5.1f}-{hi:5.1f}%): n={bin_counts[i]:4d}  observed accuracy = {acc_str}")

    plot_reliability_diagram(
        bin_edges, bin_accs, ece,
        os.path.join(OUTPUT_DIR, "ensemble_reliability_diagram.png"),
        "Reliability Diagram - Ensemble"
    )

    check_threshold_separation(y_true, y_pred, y_proba, THRESHOLD_TO_CHECK, OUTPUT_DIR, "ensemble")

    print(f"\nSaved reliability diagram and threshold check to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()