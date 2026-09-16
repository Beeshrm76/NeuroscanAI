import os
import sys
import time
import shutil

# Enable ANSI escape sequences on Windows
os.system('')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# WHY THIS FILE CHANGED (round 2): fine-tuning + class weighting
# -----------------------------------------------------------------
# Two cheap, legitimate levers were left untried after the preprocessing
# fix:
#
# 1. FULLY FROZEN BACKBONES. Every base_model layer was frozen for the
#    entire run, so training only ever adjusted the small custom head
#    (GlobalAveragePooling2D -> Dense(256) -> Dense(num_classes)). That
#    caps how well ImageNet-generic features can be reshaped toward
#    MRI-specific tumor features. Standard transfer-learning practice is
#    a SECOND phase: unfreeze the last ~20% of backbone layers and
#    continue training at a much lower learning rate (1e-5 vs the head's
#    1e-4), so the backbone adapts slightly without catastrophically
#    forgetting its pretrained weights. This is implemented below as
#    STAGE 1 (frozen head, existing behavior) -> STAGE 2 (fine-tune).
#
# 2. NO CLASS WEIGHTING. Class sizes in this dataset range from 18 to
#    367 images (~20x imbalance per the manifest). Unweighted
#    categorical_crossentropy lets the loss be dominated by the largest
#    classes, so the model can reach a deceptively OK-looking overall
#    accuracy while doing poorly on rare classes - which is exactly what
#    macro-F1 in evaluate_all.py would expose. compute_class_weight
#    ('balanced') is now computed once from the training generator's
#    labels and passed to BOTH fit() calls (frozen + fine-tune stages),
#    so under-represented classes contribute proportionally more to the
#    loss.
#
# Both changes are applied identically to every architecture in
# MODEL_CONFIGS so results stay comparable across the ensemble.

FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_FRACTION = 0.20  # unfreeze the last 20% of backbone layers

# WHY THIS FILE CHANGED
# ----------------------
# The previous version reused preprocess.get_data_generators() for every
# architecture, which applies one shared `rescale=1./255` to all five
# models. That is correct for a from-scratch CNN, but every ImageNet-
# pretrained backbone here expects its OWN specific input convention:
#   - EfficientNet expects raw 0-255 input (it rescales internally) ->
#     pre-scaling to 0-1 broke it completely (flat ~8% accuracy, never
#     learned, across all 25 epochs in the previous run).
#   - ResNet50 expects Caffe-style preprocessing (BGR order, per-channel
#     mean subtraction, unscaled) -> the mismatch is the most likely
#     reason it only reached ~21% val accuracy.
#   - VGG16 expects a similar mean-centered input -> likely capped its
#     climb to ~45% val accuracy despite still improving at epoch 25.
#   - MobileNetV2 and DenseNet121 happened to tolerate the mismatch well
#     enough to still train reasonably (~70% and ~65%), but are not
#     guaranteed to be at their true ceiling either.
#
# This script now builds a SEPARATE pair of train/val ImageDataGenerators
# per architecture, each using that architecture's own
# `keras.applications.<name>.preprocess_input` function, so every frozen
# pretrained backbone receives the input distribution it was actually
# trained on. Train/Validation/Testing folder paths and the underlying
# patient/content-deduplicated split are unchanged - only the pixel
# preprocessing changes here.

DEFAULT_IMG_SIZE = (150, 150)
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/Training'
VAL_DIR = 'dataset/Validation'

MODELS_DIR = 'models_standard'
HISTORY_DIR = 'training_history_standard'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Each entry: (model builder fn, correct preprocess_input fn, input size)
# ResNet50 uses 224x224 (its native ImageNet training resolution) instead
# of the shared 150x150 default. At 150x150, ResNet50's deeper stack of
# stride-2 downsampling stages collapses spatial resolution to almost
# nothing by the final feature map, which independently cripples feature
# quality regardless of preprocessing - this showed up as near-identical,
# still-stuck accuracy (~15-16%) even after fixing preprocess_input.
MODEL_CONFIGS = {
    'vgg16': (
        lambda: VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
        vgg16_preprocess,
        (150, 150),
    ),
    'resnet50': (
        lambda: ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        resnet50_preprocess,
        (224, 224),
    ),
    'mobilenet': (
        lambda: MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
        mobilenet_preprocess,
        (150, 150),
    ),
    'densenet': (
        lambda: DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
        densenet_preprocess,
        (150, 150),
    ),
    'efficientnet': (
        lambda: EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
        efficientnet_preprocess,
        (150, 150),
    ),
}


def make_generators(preprocess_fn, img_size):
    """Build train/val generators using the architecture-specific
    preprocess_input function and input resolution instead of one
    shared generic rescale + fixed size for every model."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=BATCH_SIZE, class_mode='categorical',
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=img_size, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False,
    )
    return train_gen, val_gen


# Only need num_classes once - any generator will report the same count
_probe_gen, _ = make_generators(vgg16_preprocess, DEFAULT_IMG_SIZE)
num_classes = _probe_gen.num_classes
print(f"Detected {num_classes} classes for training.")

# Class weights computed ONCE from the training labels (class membership
# doesn't depend on which architecture's preprocessing is applied, so any
# probe generator's .classes is representative of the real, imbalanced
# training set). 'balanced' reweights inversely proportional to class
# frequency: rare classes get a higher weight so they aren't drowned out
# by the largest classes in the loss.
class_indices = _probe_gen.classes
unique_classes = np.unique(class_indices)
class_weight_values = compute_class_weight(
    class_weight='balanced', classes=unique_classes, y=class_indices
)
class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weight_values)}
print(f"Computed class weights (balanced): {class_weight_dict}")
del _probe_gen


def unfreeze_top_fraction(base_model, fraction):
    """Unfreeze the last `fraction` of the backbone's layers for fine-
    tuning, keeping the earlier (more generic, e.g. edge/texture) layers
    frozen. BatchNorm layers are kept frozen regardless, since fine-
    tuning BatchNorm statistics on a much smaller dataset than ImageNet
    tends to destabilize training rather than help it."""
    n_layers = len(base_model.layers)
    n_unfreeze = max(1, int(n_layers * fraction))
    cutoff = n_layers - n_unfreeze
    unfrozen_count = 0
    for i, layer in enumerate(base_model.layers):
        if i < cutoff:
            layer.trainable = False
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_count += 1
    print(f"  Unfroze {unfrozen_count}/{n_layers} backbone layers "
          f"(last {fraction*100:.0f}%, BatchNorm layers kept frozen).")


class SingleLineProgressBar(tf.keras.callbacks.Callback):
    """Clean single-line progress bar: exactly 1 bar per epoch (e.g. 25 epochs = 25 bars).
    Sub-steps are hidden, parameters dynamically update in place via \\r, and each epoch
    finalizes on its single line ending with \\n.
    """
    def __init__(self, steps_per_epoch=None, bar_width=10):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.bar_width = bar_width

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.total_epochs = self.params.get('epochs', 25) if self.params else 25
        self.steps = self.steps_per_epoch or (self.params.get('steps') if self.params else None) or 97
        self.start_time = time.time()
        self.last_update = 0

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        now = time.time()
        if step < self.steps and (now - self.last_update) < 0.08:
            return
        self.last_update = now
        elapsed = now - self.start_time
        rate = elapsed / max(step, 1)
        eta = rate * max(self.steps - step, 0)

        prog = min(step / max(self.steps, 1), 1.0)
        filled = int(self.bar_width * prog)
        bar = '=' * filled + ('>' if filled < self.bar_width else '=') + '.' * (self.bar_width - filled - 1 if filled < self.bar_width else 0)

        loss_val = logs.get('loss', 0.0) if logs else 0.0
        acc_val = logs.get('accuracy', logs.get('acc', 0.0)) if logs else 0.0

        if step == self.steps:
            eta_str = 'validating...'
        else:
            eta_str = f'ETA {int(eta)}s' if eta < 60 else f'ETA {int(eta//60)}m{int(eta%60):02d}s'

        pct = int(prog * 100)
        # Sub-step (e.g. 67/97) is hidden; parameters update dynamically
        line = f"Epoch {self.epoch:02d}/{self.total_epochs:02d} [{bar}] {pct:>2d}% - {eta_str} - loss: {loss_val:.4f} - acc: {acc_val:.4f}"
        cols = shutil.get_terminal_size((80, 20)).columns
        display_line = line[:cols - 1]
        sys.stdout.write(f"\r{display_line:<{min(cols - 1, 72)}}")
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        loss_val = logs.get('loss', 0.0) if logs else 0.0
        acc_val = logs.get('accuracy', logs.get('acc', 0.0)) if logs else 0.0
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy', logs.get('val_acc'))

        bar = '=' * self.bar_width
        cols = shutil.get_terminal_size((80, 20)).columns
        if val_loss is not None and val_acc is not None:
            summary = (f"Epoch {self.epoch:02d}/{self.total_epochs:02d} [{bar}] {elapsed:.0f}s - "
                       f"loss: {loss_val:.4f} - acc: {acc_val:.4f} - "
                       f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            if len(summary) >= cols:
                summary = (f"Epoch {self.epoch:02d}/{self.total_epochs:02d} ({elapsed:.0f}s) "
                           f"loss: {loss_val:.4f} acc: {acc_val:.4f} - "
                           f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")
            if len(summary) >= cols:
                summary = (f"Epoch {self.epoch:02d}/{self.total_epochs:02d} ({elapsed:.0f}s) "
                           f"loss: {loss_val:.3f} acc: {acc_val:.3f} | "
                           f"val_loss: {val_loss:.3f} val_acc: {val_acc:.3f}")
        else:
            summary = (f"Epoch {self.epoch:02d}/{self.total_epochs:02d} [{bar}] {elapsed:.0f}s - "
                       f"loss: {loss_val:.4f} - acc: {acc_val:.4f}")

        # Finalize this epoch's single line with \n so next epoch starts on next line
        sys.stdout.write(f"\r{summary[:cols-1].ljust(min(cols-1, len(summary)))}\n")
        sys.stdout.flush()


for model_name, (build_base_model, preprocess_fn, img_size) in MODEL_CONFIGS.items():
    print(f"\n==========================================")
    print(f"   TRAINING {model_name.upper()} MODEL (corrected preprocessing, input size {img_size})")
    print(f"==========================================")

    train_gen, val_gen = make_generators(preprocess_fn, img_size)

    base_model = build_base_model()
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    # --- STAGE 1: frozen backbone, train the head only ---
    print(f"\n--- {model_name.upper()} Stage 1: training classification head "
          f"(backbone frozen) ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        verbose=0,
        callbacks=[early_stopping, SingleLineProgressBar(steps_per_epoch=len(train_gen))],
        class_weight=class_weight_dict,
    )

    # --- STAGE 2: fine-tune the top fraction of the backbone at a low LR ---
    print(f"\n--- {model_name.upper()} Stage 2: fine-tuning top "
          f"{FINE_TUNE_UNFREEZE_FRACTION*100:.0f}% of backbone (lr={FINE_TUNE_LR}) ---")
    unfreeze_top_fraction(base_model, FINE_TUNE_UNFREEZE_FRACTION)

    # Must re-compile after changing any layer's trainable flag for it to
    # take effect, and use a much lower LR so pretrained weights shift
    # gently instead of being overwritten by large gradient steps.
    model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    fine_tune_early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    fine_tune_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        verbose=0,
        callbacks=[fine_tune_early_stopping, SingleLineProgressBar(steps_per_epoch=len(train_gen))],
        class_weight=class_weight_dict,
    )

    save_path = f"{MODELS_DIR}/{model_name}_model.h5"
    model.save(save_path)
    print(f"--- {model_name.upper()} model successfully saved to {save_path} "
          f"(after frozen-head training + fine-tuning) ---")

print("\nAll models have finished training with corrected, architecture-specific "
      "preprocessing, class-balanced loss, and backbone fine-tuning!")