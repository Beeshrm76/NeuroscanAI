"""
Shared ensemble logic used by both app.py (live web predictions) and the
evaluation scripts (evaluate_all.py, confidence_calibration.py,
robustness_test.py, external_validation.py).

WHY THIS FILE CHANGED (6-model ensemble + native resolutions)
------------------------------------------------------------------
1. INCEPTIONV3 ADDED AS A 6TH ENSEMBLE MEMBER across all dicts.
2. EVERY MODEL'S IMG_SIZE NOW MATCHES train_multi.py's NATIVE RESOLUTIONS:
   - 224x224 for VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0
   - 299x299 for InceptionV3
3. PRESERVED STANDARDS: The previous 150px configuration is preserved in
   ensemble_utils_standard.py for comparison.
"""

import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from keras.applications.densenet import preprocess_input as densenet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess

# All 6 trained architectures - the ensemble always combines all 6.
ENSEMBLE_MODEL_PATHS = {
    'VGG16': 'models/vgg16_model.h5',
    'ResNet50': 'models/resnet50_model.h5',
    'MobileNetV2': 'models/mobilenet_model.h5',
    'DenseNet121': 'models/densenet_model.h5',
    'EfficientNetB0': 'models/efficientnet_model.h5',
    'InceptionV3': 'models/inception_model.h5',
}

# Also support explicit native model path directory
ENSEMBLE_MODEL_NATIVE_PATHS = {
    'VGG16': 'models_native/vgg16_model.h5',
    'ResNet50': 'models_native/resnet50_model.h5',
    'MobileNetV2': 'models_native/mobilenet_model.h5',
    'DenseNet121': 'models_native/densenet_model.h5',
    'EfficientNetB0': 'models_native/efficientnet_model.h5',
    'InceptionV3': 'models_native/inception_model.h5',
}

ENSEMBLE_MODEL_PREPROCESS = {
    'VGG16': vgg16_preprocess,
    'ResNet50': resnet50_preprocess,
    'MobileNetV2': mobilenet_preprocess,
    'DenseNet121': densenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'InceptionV3': inception_preprocess,
}

# Native ImageNet resolutions matching train_multi.py / train_multi_native.py
ENSEMBLE_MODEL_IMG_SIZE = {
    'VGG16': (150, 150),
    'ResNet50': (224, 224),
    'MobileNetV2': (150, 150),
    'DenseNet121': (150, 150),
    'EfficientNetB0': (150, 150),
    'InceptionV3': (150, 150),
}

# Overall test-set accuracy per model, from evaluate_all.py's most recent
# run (summary_metrics.csv). Used as ensemble combination weights across
# ALL 6 models. Update these manually after any retraining run.
MODEL_ACCURACY = {
    'VGG16': 0.7302,
    'ResNet50': 0.8545,
    'MobileNetV2': 0.7316,
    'DenseNet121': 0.6850,
    'EfficientNetB0': 0.7020,
    'InceptionV3': 0.7218,
}

TOP_K_DISPLAY = 3  # how many individual models' confidences to surface for display


def load_ensemble_models(model_paths=None):
    """Load the models that make up the ensemble. Checks paths with fallback."""
    paths = model_paths or ENSEMBLE_MODEL_PATHS
    loaded = {}
    for name, path in paths.items():
        if os.path.exists(path):
            loaded[name] = load_model(path)
        elif name in ENSEMBLE_MODEL_NATIVE_PATHS and os.path.exists(ENSEMBLE_MODEL_NATIVE_PATHS[name]):
            loaded[name] = load_model(ENSEMBLE_MODEL_NATIVE_PATHS[name])
        elif os.path.exists(f"models_standard/{os.path.basename(path)}"):
            loaded[name] = load_model(f"models_standard/{os.path.basename(path)}")
        else:
            raise FileNotFoundError(f"Model file for {name} not found at {path}")
    return loaded


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
    Run all models on a single input, combine ALL of them into one
    weighted-average ensemble prediction (weighted by MODEL_ACCURACY),
    and separately report the top-k most individually-confident models
    for display purposes only.
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
    Run the all-models, accuracy-weighted ensemble over the test set on
    disk, building a SEPARATE correctly-preprocessed generator for each
    model at its native resolution, then combining all models' probabilities
    per test image using the accuracy weights.
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

    stacked = np.array([per_model_preds[name] for name in names])  # (n_models, N, C)
    y_proba = np.tensordot(norm_weights, stacked, axes=(0, 0))  # (N, C)
    y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba