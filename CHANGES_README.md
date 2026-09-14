# What changed and why

This addresses the priority list, in order. Files not listed here (templates
CSS, gradcam.py's standalone helper, DICOM handling, PDF report generation)
were left untouched.

## 1. Data leakage fix (MUST DO - do this first)

**`split_data.py`** - completely rewritten. Old version randomly shuffled
individual images 80/20 into Training/Testing. New version:
- Groups images by an *inferred patient ID* per class, then splits at the
  **patient** level into Training (70%) / Validation (15%) / Testing (15%),
  so no patient's images can appear in more than one split.
- Has a `--inspect` dry-run mode that prints the inferred filename -> patient
  grouping so you can sanity-check it before touching any files.
- Writes `dataset/split_manifest.csv` (filename, class, patient_id, split)
  for auditing.
- Automatically runs a leakage self-check after splitting and prints
  PASS/FAIL.

**IMPORTANT - ACTION NEEDED FROM YOU:** I don't have your real filenames
(the dataset folders in your zip were emptied, matching your screenshots).
The patient-ID extraction regex is a best-effort guess. Before running the
real split:

```
python split_data.py --inspect
```

Read the output. If the grouping looks wrong, send me 5-10 real filenames
from one `dataset/Training/<class>/` folder (you can see folder names in
your screenshots, e.g. `_NORMAL T1`, `Astrocitoma T1`) and I'll fix the
regex. If your source images genuinely have no patient identifier at all,
that's a real limitation to state explicitly in your report rather than
silently keeping the old random split.

**`preprocess.py`** - rewritten. It used to carve a validation set out of
Training with a second random `validation_split=0.2` inside
`ImageDataGenerator` - a second, independent leakage point (same patient's
images could land in both the "training" and "validation" subset). It now
just loads the three pre-split, patient-grouped folders
(`dataset/Training`, `dataset/Validation`, `dataset/Testing`) that
`split_data.py` produces, with no further random splitting. Also added a
sanity check that class folders match across all three splits.

**`train.py`** - rewritten. It had its own separate, independent
`validation_split=0.2` (not even going through `preprocess.py`) - same
leakage bug in a third place. Now uses the shared `get_data_generators()`
from `preprocess.py` like every other training script.

**`train_multi.py`** and **`train_vgg16.py`** - no changes needed; they
already call `preprocess.get_data_generators()`, so they're automatically
fixed by the `preprocess.py` change.

You will need to **re-run `split_data.py` and retrain all 5 models** on the
new splits. Evaluating your existing `.h5` models against a newly-split test
set does not fix anything - those models may have partially memorized
patients that are now sitting in the new Testing set.

## 2. Three separate datasets (Training / Validation / Testing)

Already the target structure; `split_data.py` now produces exactly this
(`dataset/Training`, `dataset/Validation`, `dataset/Testing`).

## 3 & 4. Proper metrics for all 5 models + confusion matrix

**`metrics_utils.py`** (new) - shared helper: computes accuracy, precision,
recall, F1 (macro + weighted), full per-class report, and confusion matrix
(CSV + PNG heatmap) for any set of predictions.

**`evaluate_all.py`** - rewritten. Evaluates all 5 models
(VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0) with full
metrics via `metrics_utils.py`. Outputs land in `evaluation_results/`:
- `<model>_per_class_metrics.csv`
- `<model>_confusion_matrix.csv` and `.png`
- `summary_metrics.csv` - one row per model + ensemble, for easy comparison

## 5. Real ensemble evaluation

**`ensemble_utils.py`** (new) - single source of truth for the ensemble:
averages the softmax probability vectors of the 3 models `app.py` actually
serves (DenseNet121, MobileNetV2, VGG16) and takes the argmax. Used by both
`app.py` and every evaluation script, so the number you report is guaranteed
to be the same ensemble users actually get.

**`app.py`** - `predict()` now computes a real `'Ensemble (Average)'` entry
in the results dict (previously it only showed 3 independent model cards,
despite the UI calling them "Ensemble Diagnostic Results"). No template
changes were required for it to render - `templates/index.html` loops over
`results.items()` generically - but I added a small visual highlight
(cyan border + ★) so the ensemble card stands out from individual models.

**`evaluate_all.py`** also evaluates this same ensemble on the test set with
full metrics, saved as `Ensemble_DenseNet_MobileNet_VGG16_*` files.

## 6. Confidence calibration (strongly recommended)

**`confidence_calibration.py`** (new) - checks whether the `<50% =
uncertain` cutoff is actually meaningful: bins predictions by confidence,
computes real accuracy per bin, plots a reliability diagram, computes
Expected Calibration Error (ECE), and explicitly compares accuracy of
high-confidence vs low-confidence predictions. If it turns out
high-confidence predictions aren't actually more accurate, the script prints
a warning - that would mean the 50% threshold shouldn't be presented to
users as a trust signal without further work (e.g. temperature scaling).

## 7. Robustness testing (strongly recommended)

**`robustness_test.py`** (new) - re-runs the ensemble on the test set after
applying brightness up/down, contrast up/down, 10° rotation, Gaussian noise,
and Gaussian blur, and reports accuracy under each condition vs baseline, so
you can see which realistic distortions hurt the model most.

## 8. External validation (strongly recommended)

**`external_validation.py`** (new) - evaluates the ensemble against a
completely separate MRI dataset you provide (not used in training/model
selection). You'll need to point `EXTERNAL_DATASET_DIR` at a downloaded
external dataset and fill in `CLASS_NAME_MAP` to map its folder names onto
this project's class names. I can't fetch or fabricate real patient MRI data
myself, so this one requires you to source the dataset.

## Suggested run order

```
python split_data.py --inspect        # verify patient grouping first!
python split_data.py                  # real patient-level split
python train_multi.py                 # retrain all 5 architectures
python train_vgg16.py                 # (if you keep this as a separate run)
python evaluate_all.py                # full metrics + confusion matrices + ensemble
python confidence_calibration.py      # is the 50% threshold meaningful?
python robustness_test.py             # accuracy under realistic distortions
python external_validation.py         # after you source an external dataset
```

## Not changed (out of scope of your list)

- `gradcam.py` has an unrelated bug (`cv2.colormap(...)` isn't a real
  OpenCV function) in `save_and_display_gradcam()`, but that function is
  never actually called anywhere (`app.py` has its own inline Grad-CAM
  implementation that doesn't use this file), so it's dead code. Flagging it
  in case you use that function later.
