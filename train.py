import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from preprocess import get_data_generators

# WHY THIS FILE CHANGED
# ----------------------
# This script used to build its OWN ImageDataGenerator with
# `validation_split=0.2` directly on dataset/Training, completely separate
# from preprocess.py. That was a second independent place where images from
# the same patient could be randomly split between "training" and
# "validation" subsets. It now reuses preprocess.get_data_generators(),
# which loads the three patient-grouped, pre-split folders
# (Training/Validation/Testing) written by split_data.py, so every training
# script in this project now shares one single, leakage-checked split.

IMG_SIZE = (150, 150)

os.makedirs('models', exist_ok=True)

# 1. Load the shared, patient-safe generators
train_generator, val_generator, test_generator = get_data_generators()

# Class weighting: classes in this dataset range from 18 to 367 images
# (~20x imbalance). Without this, categorical_crossentropy is dominated
# by the largest classes and this from-scratch CNN (with no pretrained
# features to fall back on) is especially exposed to that imbalance.
unique_classes = np.unique(train_generator.classes)
class_weight_values = compute_class_weight(
    class_weight='balanced', classes=unique_classes, y=train_generator.classes
)
class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weight_values)}
print(f"Computed class weights (balanced): {class_weight_dict}")

# 2. CNN Model Architecture
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Callbacks
checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Train Model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight_dict
)

print("Training complete. Best model saved to models/best_model.h5")