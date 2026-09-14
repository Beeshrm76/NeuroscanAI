"""
External validation - the strongest test of generalization.

WHY THIS FILE EXISTS
---------------------
Every number produced by evaluate_all.py, confidence_calibration.py, and
robustness_test.py comes from data that ultimately originated from the same
source/scanner/collection process as your training data, even after fixing
patient-level leakage. A model can still overfit to dataset-wide quirks
(specific scanners, preprocessing pipeline, label conventions) that a
patient-level split alone won't catch. This script evaluates your ensemble
(and optionally individual models) against a SEPARATE MRI dataset that was
never touched during training, validation, or model selection.

YOU must supply this dataset yourself - I cannot fetch or fabricate real
patient MRI data. Good candidates are other public brain-tumor MRI
datasets on Kaggle/TCIA with a DIFFERENT source than the one you trained
on. Point EXTERNAL_DATASET_DIR below at it. It must be organized the same
way as dataset/Testing (one subfolder per class).

Note on class names: your external dataset almost certainly won't use the
exact same folder/class names as this project (e.g. "glioma" vs
"Astrocitoma T1"). Set CLASS_NAME_MAP below to map the external dataset's
folder names onto this project's class names for any classes that overlap.
Classes with no mapping are reported separately and excluded from the
accuracy/F1 numbers (they can't be scored against labels the model was
never trained to produce).

Usage:
    python external_validation.py
Outputs land in: evaluation_results/external/
"""

import os
import numpy as np
from PIL import Image

from ensemble_utils import load_ensemble_models, ensemble_predict_proba, ENSEMBLE_MODEL_PREPROCESS, ENSEMBLE_MODEL_IMG_SIZE
from metrics_utils import compute_and_save_metrics

# ----------------------------------------------------------------------
# CONFIG - fill these in for your external dataset
# ----------------------------------------------------------------------
EXTERNAL_DATASET_DIR = "external_dataset"  # one subfolder per class, like dataset/Testing
OUTPUT_DIR = "evaluation_results/external"

# Map external folder name -> this project's class name (from
# dataset/Training folder names). Only classes present in this map are
# scored. Example:
# CLASS_NAME_MAP = {
#     "glioma_tumor": "Glioblastoma",
#     "meningioma_tumor": "meningioma",
#     "no_tumor": "_NORMAL T1",
# }
CLASS_NAME_MAP = {
    # "<external_folder_name>": "<this_project_class_name>",
}


def get_project_class_names():
    train_dir = "dataset/Training"
    return sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )


def load_external_items(external_dir, class_map):
    items = []
    skipped_classes = []
    for folder in sorted(os.listdir(external_dir)):
        folder_path = os.path.join(external_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in class_map:
            skipped_classes.append(folder)
            continue
        mapped_class = class_map[folder]
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for f in files:
            items.append((os.path.join(folder_path, f), mapped_class))
    return items, skipped_classes


def main():
    if not os.path.isdir(EXTERNAL_DATASET_DIR):
        print(f"EXTERNAL_DATASET_DIR '{EXTERNAL_DATASET_DIR}' not found.")
        print("Download a separate, independent MRI dataset, point "
              "EXTERNAL_DATASET_DIR at it, fill in CLASS_NAME_MAP, and re-run.")
        return

    if not CLASS_NAME_MAP:
        print("CLASS_NAME_MAP is empty - external folder names have not been "
              "mapped to this project's class names yet. Edit "
              "external_validation.py and fill in CLASS_NAME_MAP before running.")
        return

    project_classes = get_project_class_names()
    class_to_idx = {c: i for i, c in enumerate(project_classes)}

    items, skipped = load_external_items(EXTERNAL_DATASET_DIR, CLASS_NAME_MAP)
    if skipped:
        print(f"NOTE: {len(skipped)} external folder(s) have no entry in "
              f"CLASS_NAME_MAP and were skipped: {skipped}")
    if not items:
        print("No usable images found after applying CLASS_NAME_MAP. Nothing to evaluate.")
        return

    print(f"Loaded {len(items)} external images across "
          f"{len(set(c for _, c in items))} mapped classes.")

    models = load_ensemble_models()
    model_names = list(models.keys())

    y_true, y_pred = [], []
    for filepath, mapped_class in items:
        base_img = Image.open(filepath).convert("RGB")
        x_by_model = {}
        for name in model_names:
            img_size = ENSEMBLE_MODEL_IMG_SIZE[name]
            preprocess_fn = ENSEMBLE_MODEL_PREPROCESS[name]
            resized = base_img.resize(img_size)
            arr = np.expand_dims(np.array(resized).astype(np.float32), axis=0)
            x_by_model[name] = preprocess_fn(arr)
        proba = ensemble_predict_proba(models, x_by_model)
        y_pred.append(int(np.argmax(proba)))
        y_true.append(class_to_idx[mapped_class])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = compute_and_save_metrics(
        y_true, y_pred, project_classes, OUTPUT_DIR, "external_ensemble"
    )

    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION RESULTS (independent dataset)")
    print("=" * 70)
    print(f"  Accuracy:          {summary['accuracy']*100:.2f}%")
    print(f"  Precision (macro): {summary['precision_macro']:.4f}")
    print(f"  Recall (macro):    {summary['recall_macro']:.4f}")
    print(f"  F1 (macro):        {summary['f1_macro']:.4f}")
    print(f"  N samples:         {summary['n_samples']}")
    print(f"\nCompare this against the internal test-set numbers from "
          f"evaluate_all.py. A large drop here (vs internal test accuracy) "
          f"indicates the model is overfitting to this dataset's specific "
          f"source/scanner/preprocessing rather than truly generalizing.")
    print(f"Full report saved under '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()