import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# WHY THIS FILE CHANGED
# ----------------------
# The old version only had Training/ and Testing/ folders on disk, and then
# used `validation_split=0.2` inside ImageDataGenerator.flow_from_directory()
# to carve a validation set OUT of Training at random, per image. That is a
# second, less obvious source of the exact same leakage problem described in
# split_data.py: if a patient has several images in Training, some of that
# patient's images could land in "training subset" and others in "validation
# subset" of the same generator, so validation accuracy during training was
# already partly optimistic before the model ever touched Testing.
#
# split_data.py now writes three ALREADY patient-grouped folders on disk:
#   dataset/Training/, dataset/Validation/, dataset/Testing/
# so this file just loads each folder as its own generator and no longer
# performs any further random splitting.

train_dir = 'dataset/Training'
val_dir = 'dataset/Validation'
test_dir = 'dataset/Testing'
IMG_SIZE = (150, 150)
BATCH_SIZE = 32


def get_data_generators():
    # Augmentation only makes sense for the training set. Validation and
    # test sets are only rescaled, never augmented, so that reported metrics
    # reflect real, un-augmented images.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    # Sanity check: class sets and index mapping must match across all three
    # splits, or metrics/labels will silently misalign later on.
    if train_generator.class_indices != val_generator.class_indices:
        raise ValueError(
            "Training and Validation class_indices do not match. Did every "
            "class folder get created under both dataset/Training and "
            "dataset/Validation? Re-run split_data.py."
        )
    if train_generator.class_indices != test_generator.class_indices:
        raise ValueError(
            "Training and Testing class_indices do not match. Did every "
            "class folder get created under both dataset/Training and "
            "dataset/Testing? Re-run split_data.py."
        )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print(f"Successfully loaded classes: {list(train_gen.class_indices.keys())}")
    print(f"Train samples: {train_gen.samples} | "
          f"Val samples: {val_gen.samples} | "
          f"Test samples: {test_gen.samples}")
