"""
Robustness testing under realistic image perturbations.

WHY THIS FILE EXISTS
---------------------
A model that only performs well on clean, perfectly-formatted test images
can still fail badly in the real world, where uploaded MRI images may have
slightly different brightness/contrast (different scanner/export settings),
minor rotation (patient positioning, or a photographed screen), compression
noise, or slight blur. This script re-runs the ensemble on the SAME test
set after applying each perturbation, so you can see how much accuracy
degrades under each realistic condition rather than assuming the clean-test
number generalizes.

Run this AFTER evaluate_all.py, on models trained on the patient-level
split.

Usage:
    python robustness_test.py
Outputs land in: evaluation_results/robustness/
"""

import os
import csv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ensemble_utils import load_ensemble_models, ensemble_predict_proba, ENSEMBLE_MODEL_PREPROCESS, ENSEMBLE_MODEL_IMG_SIZE

TEST_DIR = "dataset/Testing"
OUTPUT_DIR = "evaluation_results/robustness"

# Cap how many images per class to test with, to keep runtime reasonable
# for a many-class dataset. Set to None to use the full test set.
MAX_IMAGES_PER_CLASS = 30


def list_test_images(test_dir):
    """Returns list of (filepath, class_name)."""
    items = []
    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if MAX_IMAGES_PER_CLASS:
            files = files[:MAX_IMAGES_PER_CLASS]
        for f in files:
            items.append((os.path.join(class_dir, f), class_name))
    return items


# ----------------------------------------------------------------------
# Perturbations - each takes a PIL Image and returns a perturbed PIL Image
# ----------------------------------------------------------------------

def perturb_brightness_up(img):
    return ImageEnhance.Brightness(img).enhance(1.3)


def perturb_brightness_down(img):
    return ImageEnhance.Brightness(img).enhance(0.7)


def perturb_contrast_up(img):
    return ImageEnhance.Contrast(img).enhance(1.3)


def perturb_contrast_down(img):
    return ImageEnhance.Contrast(img).enhance(0.7)


def perturb_rotation(img):
    return img.rotate(10, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def perturb_gaussian_noise(img):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 12, arr.shape)  # sigma=12 on 0-255 scale
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def perturb_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


PERTURBATIONS = {
    "baseline": lambda img: img,
    "brightness_up": perturb_brightness_up,
    "brightness_down": perturb_brightness_down,
    "contrast_up": perturb_contrast_up,
    "contrast_down": perturb_contrast_down,
    "rotation_10deg": perturb_rotation,
    "gaussian_noise": perturb_gaussian_noise,
    "gaussian_blur": perturb_blur,
}


def load_and_preprocess(filepath, perturb_fn, model_names):
    """
    Load, perturb (at a fixed base resolution), then produce a SEPARATE
    correctly-preprocessed array per ensemble model - since DenseNet121,
    MobileNetV2, and VGG16 each require their own preprocess_input, a
    single shared /255.0 array silently corrupts predictions from models
    that don't expect that convention.
    """
    base_img = Image.open(filepath).convert("RGB")
    base_img = perturb_fn(base_img)

    x_by_model = {}
    for name in model_names:
        img_size = ENSEMBLE_MODEL_IMG_SIZE[name]
        preprocess_fn = ENSEMBLE_MODEL_PREPROCESS[name]
        resized = base_img.resize(img_size)
        arr = np.expand_dims(np.array(resized).astype(np.float32), axis=0)
        x_by_model[name] = preprocess_fn(arr)
    return x_by_model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    items = list_test_images(TEST_DIR)
    if not items:
        print(f"No test images found under {TEST_DIR}. Run split_data.py first.")
        return

    class_names = sorted({c for _, c in items})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"Loaded {len(items)} test images across {len(class_names)} classes "
          f"(capped at {MAX_IMAGES_PER_CLASS} per class).")

    models = load_ensemble_models()
    model_names = list(models.keys())

    results_rows = []
    for perturbation_name, fn in PERTURBATIONS.items():
        correct = 0
        total = 0
        for filepath, true_class in items:
            x_by_model = load_and_preprocess(filepath, fn, model_names)
            proba = ensemble_predict_proba(models, x_by_model)
            pred_idx = int(np.argmax(proba))
            true_idx = class_to_idx[true_class]
            if pred_idx == true_idx:
                correct += 1
            total += 1
        accuracy = correct / total if total else float('nan')
        print(f"  {perturbation_name:18s}: accuracy = {accuracy*100:.2f}%  (n={total})")
        results_rows.append({
            "perturbation": perturbation_name,
            "accuracy": round(accuracy, 4),
            "n_samples": total,
        })

    baseline_acc = next(r["accuracy"] for r in results_rows if r["perturbation"] == "baseline")
    for row in results_rows:
        row["accuracy_drop_vs_baseline"] = round(baseline_acc - row["accuracy"], 4)

    out_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", "accuracy", "n_samples", "accuracy_drop_vs_baseline"])
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nResults saved to {out_path}")
    print("\nLargest accuracy drops (most concerning perturbations first):")
    for row in sorted(results_rows, key=lambda r: -r["accuracy_drop_vs_baseline"])[:3]:
        if row["perturbation"] == "baseline":
            continue
        print(f"  {row['perturbation']}: -{row['accuracy_drop_vs_baseline']*100:.2f} pts vs baseline")


if __name__ == "__main__":
    main()