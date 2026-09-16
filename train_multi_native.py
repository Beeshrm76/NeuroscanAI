import os
import sys
import time
import shutil
import json

# Enable ANSI escape sequences on Windows
os.system('')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, InceptionV3
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# WHY THIS FILE CHANGED (consolidated to 6 models + native resolutions + persisted history)
# -----------------------------------------------------------------------
# 1. INCEPTIONV3 FOLDED IN. It was previously trained by a separate,
#    near-duplicate script (train_inception.py). There is now ONE training
#    script. train_inception_standard.py and train_vgg16_standard.py preserve
#    the old standalone scripts, but active training lives here under
#    MODEL_CONFIGS['inception'] / ['vgg16'].
#
# 2. EVERY MODEL USES ITS OWN NATIVE IMAGENET RESOLUTION:
#      - VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0: 224x224
#      - InceptionV3: 299x299
#
# 3. TRAINING HISTORY IS SAVED TO DISK (JSON). Both stages' per-epoch
#    loss/accuracy/val_loss/val_accuracy dicts are saved to
#    training_history/<model>_history.json and training_history_native/.

FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_FRACTION = 0.20  # unfreeze the last 20% of backbone layers

BATCH_SIZE = 32
TRAIN_DIR = 'dataset/Training'
VAL_DIR = 'dataset/Validation'
HISTORY_DIR = 'training_history'
HISTORY_NATIVE_DIR = 'training_history_native'
MODELS_DIR = 'models'
MODELS_NATIVE_DIR = 'models_native'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_NATIVE_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(HISTORY_NATIVE_DIR, exist_ok=True)

MODEL_CONFIGS = {
    'vgg16': (
        lambda: VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        vgg16_preprocess,
        (224, 224),
    ),
    'resnet50': (
        lambda: ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        resnet50_preprocess,
        (224, 224),
    ),
    'mobilenet': (
        lambda: MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        mobilenet_preprocess,
        (224, 224),
    ),
    'densenet': (
        lambda: DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        densenet_preprocess,
        (224, 224),
    ),
    'efficientnet': (
        lambda: EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        efficientnet_preprocess,
        (224, 224),
    ),
    'inception': (
        lambda: InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        inception_preprocess,
        (299, 299),
    ),
}


def make_generators(preprocess_fn, img_size):
    """Build train/val generators using architecture-specific preprocessing
    and native resolution."""
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


# Probe once for class count and balanced weights
_probe_preprocess, _probe_size = vgg16_preprocess, MODEL_CONFIGS['vgg16'][2]
_probe_gen, _ = make_generators(_probe_preprocess, _probe_size)
num_classes = _probe_gen.num_classes
print(f"Detected {num_classes} classes for training.")

class_indices = _probe_gen.classes
unique_classes = np.unique(class_indices)
class_weight_values = compute_class_weight(
    class_weight='balanced', classes=unique_classes, y=class_indices
)
class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weight_values)}
print(f"Computed class weights (balanced): {class_weight_dict}")
del _probe_gen


def unfreeze_top_fraction(base_model, fraction):
    """Unfreeze the last `fraction` of backbone layers for fine-tuning,
    keeping BatchNorm layers frozen to protect moving statistics."""
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


def save_history(model_name, history, fine_tune_history):
    """Persist both stages' per-epoch loss/accuracy/val_loss/val_accuracy
    to disk as JSON in both training_history and training_history_native."""
    history_data = {
        'stage1': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'stage2': {k: [float(v) for v in vals] for k, vals in fine_tune_history.history.items()},
    }
    for hdir in [HISTORY_DIR, HISTORY_NATIVE_DIR]:
        path = os.path.join(hdir, f"{model_name}_history.json")
        with open(path, 'w') as f:
            json.dump(history_data, f, indent=2)
    print(f"  Saved training history to {os.path.join(HISTORY_DIR, f'{model_name}_history.json')}")


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

        sys.stdout.write(f"\r{summary[:cols-1].ljust(min(cols-1, len(summary)))}\n")
        sys.stdout.flush()


def train_all_models():
    for model_name, (build_base_model, preprocess_fn, img_size) in MODEL_CONFIGS.items():
        print(f"\n==========================================")
        print(f"   TRAINING {model_name.upper()} MODEL (native resolution {img_size})")
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
        print(f"\n--- {model_name.upper()} Stage 1: training classification head (backbone frozen) ---")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=25,
            verbose=0,
            callbacks=[early_stopping, SingleLineProgressBar(steps_per_epoch=len(train_gen))],
            class_weight=class_weight_dict,
        )

        # --- STAGE 2: fine-tune the top fraction of the backbone at low LR ---
        print(f"\n--- {model_name.upper()} Stage 2: fine-tuning top {FINE_TUNE_UNFREEZE_FRACTION*100:.0f}% of backbone (lr={FINE_TUNE_LR}) ---")
        unfreeze_top_fraction(base_model, FINE_TUNE_UNFREEZE_FRACTION)

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

        save_path = f"models/{model_name}_model.h5"
        native_save_path = f"models_native/{model_name}_model.h5"
        model.save(save_path)
        model.save(native_save_path)
        save_history(model_name, history, fine_tune_history)
        print(f"--- {model_name.upper()} model saved to {save_path} and {native_save_path} ---")

    print("\nAll 6 models have finished training at their native resolutions, "
          "with corrected architecture-specific preprocessing, class-balanced "
          "loss, backbone fine-tuning, and per-model history saved under "
          f"'{HISTORY_DIR}/' and '{HISTORY_NATIVE_DIR}/'.")
    print("\nNEXT STEPS:")
    print("  1. Re-run evaluate_all.py or evaluate_all_native.py.")
    print("  2. Update MODEL_ACCURACY in ensemble_utils.py / ensemble_utils_native.py")
    print("     with the fresh numbers from evaluation_results/summary_metrics.csv.")


if __name__ == '__main__':
    train_all_models()