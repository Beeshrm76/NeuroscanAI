"""
Shared ensemble logic used by both app.py (live web predictions) and the
evaluation scripts (evaluate_all.py, confidence_calibration.py,
robustness_test.py, external_validation.py).

WHY THIS FILE CHANGED (all-5 weighted ensemble + top-3 display)
------------------------------------------------------------------
Two SEPARATE things now happen, not one combined selection+ensemble
step:

1. THE ENSEMBLE PREDICTION uses ALL 5 models, always - not a dynamic
   subset. Each model's probability vector is combined using a WEIGHTED
   average, where the weight is that model's own overall test-set
   ACCURACY (a fixed number from evaluate_all.py, MODEL_ACCURACY below),
   normalized across all 5 so the weights sum to 1. A generally more
   reliable model (e.g. ResNet50 at 85.45%) always counts for more in
   the final ensemble prediction than a generally less reliable one
   (e.g. DenseNet121 at 68.50%), regardless of confidence on any single
   image.

2. THE TOP-3-BY-CONFIDENCE LIST is purely informational now - it does
   NOT feed into the ensemble prediction above. It's just "of these 5
   individual model predictions, here are the 3 that were most
   confident on this specific image", shown alongside the ensemble
   result so a user can see which individual models were most sure of
   themselves, without that affecting the actual combined answer.

IMPORTANT CAVEATS:
- MODEL_ACCURACY below must be kept in sync manually with whatever
  evaluate_all.py's summary_metrics.csv actually reports after your most
  recent training run - it is NOT computed automatically here.
- Because the ensemble now always uses all 5 (weighted), a chronically
  weak model (e.g. DenseNet121) still has SOME influence on every
  prediction, just proportionally less than stronger models - it can no
  longer be excluded entirely the way the earlier dynamic-selection
  version could exclude it. If a much weaker model exists, this softens
  but doesn't remove its drag on the ensemble.
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# All 5 trained architectures - the ensemble always combines all 5.
ENSEMBLE_MODEL_PATHS = {
    'VGG16': 'models/vgg16_model.h5',
    'ResNet50': 'models/resnet50_model.h5',
    'MobileNetV2': 'models/mobilenet_model.h5',
    'DenseNet121': 'models/densenet_model.h5',
    'EfficientNetB0': 'models/efficientnet_model.h5',
}

ENSEMBLE_MODEL_PREPROCESS = {
    'VGG16': vgg16_preprocess,
    'ResNet50': resnet50_preprocess,
    'MobileNetV2': mobilenet_preprocess,
    'DenseNet121': densenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
}

ENSEMBLE_MODEL_IMG_SIZE = {
    'VGG16': (150, 150),
    'ResNet50': (224, 224),
    'MobileNetV2': (150, 150),
    'DenseNet121': (150, 150),
    'EfficientNetB0': (150, 150),
}

# Overall test-set accuracy per model, from evaluate_all.py's most recent
# run (summary_metrics.csv). Used as ensemble combination weights across
# ALL 5 models. Update these manually after any retraining run.
MODEL_ACCURACY = {
    'VGG16': 0.7302,
    'ResNet50': 0.8545,
    'MobileNetV2': 0.7316,
    'DenseNet121': 0.6850,
    'EfficientNetB0': 0.7020,
}

TOP_K_DISPLAY = 3  # how many individual models' confidences to surface for display


def load_ensemble_models(model_paths=None):
    """Load the models that make up the ensemble. Returns {name: model}."""
    model_paths = model_paths or ENSEMBLE_MODEL_PATHS
    return {name: load_model(path) for name, path in model_paths.items()}


def weighted_ensemble_proba(individual_probas: dict, weights: dict = None):
    """
    Combine ALL models' probability vectors using accuracy-based weights,
    normalized to sum to 1 across whichever models are present in
    `individual_probas`. Returns the weighted-average probability vector,
    shape (num_classes,).
    """
    weights = weights or MODEL_ACCURACY
    names = list(individual_probas.keys())
    raw_weights = np.array([weights[name] for name in names], dtype=np.float64)
    norm_weights = raw_weights / raw_weights.sum()
    stacked = np.array([individual_probas[name] for name in names])
    return np.average(stacked, axis=0, weights=norm_weights), dict(zip(names, norm_weights))


def top_k_by_confidence(individual_probas: dict, k: int = TOP_K_DISPLAY):
    """
    Given {model_name: probability_vector} for ONE input, return a list
    of (model_name, confidence) tuples for the k models with the highest
    individual confidence on this image, sorted highest first. PURELY
    INFORMATIONAL - does not affect the ensemble prediction.
    """
    confidences = {name: float(np.max(proba)) for name, proba in individual_probas.items()}
    ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]


def ensemble_predict_proba(models: dict, x_by_model: dict, weights: dict = None,
                            top_k_display: int = TOP_K_DISPLAY):
    """
    Run all 5 models on a single input, combine ALL of them into one
    weighted-average ensemble prediction (weighted by MODEL_ACCURACY),
    and separately report the top-k most individually-confident models
    for display purposes only.

    Returns (proba_weighted, weights_used, top_k_confidences):
      proba_weighted    - weighted-average probability vector over ALL
                           models, shape (num_classes,)
      weights_used       - {model_name: normalized_weight} for all models
                            actually used in the ensemble
      top_k_confidences  - list of (model_name, confidence) for the top-k
                            most confident individual models on this
                            image, informational only
    """
    individual_probas = {
        name: model.predict(x_by_model[name], verbose=0)[0]
        for name, model in models.items()
    }
    proba_weighted, weights_used = weighted_ensemble_proba(individual_probas, weights=weights)
    top_k_confidences = top_k_by_confidence(individual_probas, k=top_k_display)
    return proba_weighted, weights_used, top_k_confidences


def ensemble_predict_generator(models: dict, test_dir, batch_size=32, weights: dict = None):
    """
    Run the all-5, accuracy-weighted ensemble over the test set on disk,
    building a SEPARATE correctly-preprocessed generator for each model,
    then combining ALL 5 models' probabilities per test image using the
    same accuracy weights used in production - so this evaluation is a
    true measurement of what app.py actually serves.

    `test_dir` is the path to the test folder on disk (e.g.
    'dataset/Testing').

    Returns (y_true, y_pred, y_proba):
      y_true  - ground-truth class indices, shape (N,)
      y_pred  - ensemble argmax predictions, shape (N,)
      y_proba - ensemble weighted-average probabilities, shape (N, num_classes)
    """
    weights = weights or MODEL_ACCURACY
    per_model_preds = {}
    y_true = None

    for name, model in models.items():
        preprocess_fn = ENSEMBLE_MODEL_PREPROCESS[name]
        img_size = ENSEMBLE_MODEL_IMG_SIZE[name]

        datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        gen = datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
        )

        steps = int(np.ceil(gen.samples / gen.batch_size))
        preds = model.predict(gen, steps=steps, verbose=0)
        preds = preds[:gen.samples]
        per_model_preds[name] = preds

        if y_true is None:
            y_true = gen.classes[:gen.samples]

    names = list(per_model_preds.keys())
    raw_weights = np.array([weights[name] for name in names], dtype=np.float64)
    norm_weights = raw_weights / raw_weights.sum()

    # Weighted average across all samples at once: stack into
    # (n_models, n_samples, n_classes), then weight along the model axis.
    stacked = np.array([per_model_preds[name] for name in names])  # (n_models, N, C)
    y_proba = np.tensordot(norm_weights, stacked, axes=(0, 0))  # (N, C)
    y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba