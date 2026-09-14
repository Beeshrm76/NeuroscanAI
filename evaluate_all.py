"""
Full evaluation of all 5 trained models AND the real ensemble on the final
held-out Testing set.

WHY THIS FILE CHANGED (again)
-------------------------------
The previous version used ONE shared test generator (from
preprocess.get_data_generators(), generic rescale=1./255, fixed 150x150)
for every model. That was fine back when every model actually used that
same generic preprocessing - but train_multi.py has since been corrected
to give each architecture its OWN preprocess_input function and, for
ResNet50, its own input size (224x224 instead of 150x150, since ResNet50's
deeper downsampling stages need the larger input to retain useful spatial
features - see model_performance_report.md for the full explanation).

Evaluating with the old shared generator caused two failures:
  - VGG16 was evaluated with the WRONG preprocessing (generic rescale
    instead of vgg16 preprocess_input), producing a meaningless ~2%
    "accuracy" that has nothing to do with how well the model actually
    performs - it's an artifact of mismatched preprocessing between
    training and evaluation, not a real result.
  - ResNet50 crashed outright with a shape mismatch, because its saved
    model expects (224, 224, 3) input but the shared generator only
    produces (150, 150, 3).

This version builds a SEPARATE test generator per model, using that
model's own correct preprocess_input function and input size, mirroring
exactly what train_multi.py used during training. This is the only way
evaluation numbers are meaningful - a model must be evaluated with the
same preprocessing convention it was trained under.

Run this only AFTER re-running split_data.py (patient-level/deduplicated
split) and retraining all models on the new splits with corrected
preprocessing - evaluating old models trained on the leaky split or with
broken preprocessing against a new test set doesn't fix anything.

Usage:
    python evaluate_all.py
Outputs land in: evaluation_results/
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from ensemble_utils import ENSEMBLE_MODEL_PATHS, load_ensemble_models, ensemble_predict_generator
from metrics_utils import compute_and_save_metrics, write_summary_table

OUTPUT_DIR = "evaluation_results"
TEST_DIR = "dataset/Testing"
BATCH_SIZE = 32

# Each entry: (display name, model file path, preprocess_input fn, input size)
# Must exactly match what train_multi.py used to train each model.
MODELS_TO_EVALUATE = [
    ('VGG16', 'models/vgg16_model.h5', vgg16_preprocess, (150, 150)),
    ('ResNet50', 'models/resnet50_model.h5', resnet50_preprocess, (224, 224)),
    ('MobileNetV2', 'models/mobilenet_model.h5', mobilenet_preprocess, (150, 150)),
    ('DenseNet121', 'models/densenet_model.h5', densenet_preprocess, (150, 150)),
    ('EfficientNetB0', 'models/efficientnet_model.h5', efficientnet_preprocess, (150, 150)),
]


def make_test_generator(preprocess_fn, img_size):
    """Build a test generator using a specific model's own preprocessing
    and input size, not the shared generic one."""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )


def main():
    # Only used to get the canonical class name list/order - any generator
    # reports the same class_indices regardless of preprocessing/size.
    _probe_gen = make_test_generator(vgg16_preprocess, (150, 150))
    class_names = list(_probe_gen.class_indices.keys())
    del _probe_gen

    print("=" * 70)
    print("     INDIVIDUAL MODEL EVALUATION (Accuracy, Precision, Recall,")
    print("     F1, Macro-F1, Per-Class Report, Confusion Matrix)")
    print("=" * 70)

    summaries = []

    for name, path, preprocess_fn, img_size in MODELS_TO_EVALUATE:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: model file not found at {path}")
            continue

        # Build a generator matching THIS model's own training setup.
        test_gen = make_test_generator(preprocess_fn, img_size)

        model = load_model(path)
        steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
        y_proba = model.predict(test_gen, steps=steps, verbose=0)[:test_gen.samples]
        y_pred = np.argmax(y_proba, axis=1)
        y_true = test_gen.classes[:test_gen.samples]

        summary = compute_and_save_metrics(y_true, y_pred, class_names, OUTPUT_DIR, name)
        summaries.append(summary)

        print(f"\n{name}: (input size {img_size})")
        print(f"  Accuracy:         {summary['accuracy']*100:.2f}%")
        print(f"  Precision (macro): {summary['precision_macro']:.4f}")
        print(f"  Recall (macro):    {summary['recall_macro']:.4f}")
        print(f"  F1 (macro):        {summary['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {summary['f1_weighted']:.4f}")

        # Free memory before loading the next big model.
        del model

    # ------------------------------------------------------------------
    # Real ensemble evaluation - averages probabilities of the models
    # app.py actually serves to users.
    #
    # IMPORTANT: if the ensemble mixes models with DIFFERENT input sizes
    # (e.g. ResNet50 at 224x224 alongside others at 150x150),
    # ensemble_utils.py's generator-based prediction needs to feed each
    # model its own correctly-sized, correctly-preprocessed batch before
    # averaging probabilities - averaging raw model outputs is fine
    # (softmax outputs are always num_classes-length regardless of input
    # size), but each model must receive ITS OWN correctly prepared input
    # first. Check ensemble_utils.py separately if the ensemble result
    # looks wrong the same way the individual models did here.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("     ENSEMBLE EVALUATION (matches the models app.py serves)")
    print(f"     Models in ensemble: {list(ENSEMBLE_MODEL_PATHS.keys())}")
    print("=" * 70)

    missing = [p for p in ENSEMBLE_MODEL_PATHS.values() if not os.path.exists(p)]
    if missing:
        print(f"[SKIP ENSEMBLE] Missing model file(s): {missing}")
    else:
        ensemble_models = load_ensemble_models()
        # ensemble_predict_generator now builds its OWN correctly
        # preprocessed, correctly sized generator per model internally,
        # so it takes the test directory path, not a pre-built generator.
        y_true, y_pred, y_proba = ensemble_predict_generator(ensemble_models, TEST_DIR)
        ensemble_summary = compute_and_save_metrics(
            y_true, y_pred, class_names, OUTPUT_DIR, "Ensemble_DenseNet_MobileNet_VGG16"
        )
        summaries.append(ensemble_summary)

        print(f"\nEnsemble:")
        print(f"  Accuracy:          {ensemble_summary['accuracy']*100:.2f}%")
        print(f"  Precision (macro): {ensemble_summary['precision_macro']:.4f}")
        print(f"  Recall (macro):    {ensemble_summary['recall_macro']:.4f}")
        print(f"  F1 (macro):        {ensemble_summary['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {ensemble_summary['f1_weighted']:.4f}")

        ensemble_models.clear()

    # ------------------------------------------------------------------
    summary_path = write_summary_table(summaries, OUTPUT_DIR)
    print("\n" + "=" * 70)
    print(f"All metrics, per-class reports, and confusion matrices saved under "
          f"'{OUTPUT_DIR}/'.")
    print(f"Comparison table across all models + ensemble: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()