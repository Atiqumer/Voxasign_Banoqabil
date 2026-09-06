"""
VoxaSign PSL V2 Training

Uses the engineered 115-feature representation.

Important:
    StandardScaler is fitted ONLY on training data.
    Validation and test data are transformed using
    the training scaler.
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_DIR = ROOT / "data" / "landmarks" / "v2"

LANDMARK_DIR = ROOT / "data" / "landmarks"

OUTPUT_DIR = ROOT / "output"

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


MODEL_PATH = OUTPUT_DIR / "psl_static_model_v2.keras"

SCALER_PATH = OUTPUT_DIR / "psl_v2_scaler.npz"

CONFIG_PATH = OUTPUT_DIR / "psl_v2_config.json"


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("VoxaSign PSL V2 Training")
print("=" * 70)

print()
print("Loading V2 features...")

X_train = np.load(
    FEATURE_DIR / "X_train_v2.npy"
)

X_val = np.load(
    FEATURE_DIR / "X_validation_v2.npy"
)

X_test = np.load(
    FEATURE_DIR / "X_test_v2.npy"
)

y_train = np.load(
    LANDMARK_DIR / "y_train.npy"
)

y_val = np.load(
    LANDMARK_DIR / "y_validation.npy"
)

y_test = np.load(
    LANDMARK_DIR / "y_test.npy"
)


print()
print("TRAIN:")
print("X:", X_train.shape)
print("y:", y_train.shape)

print()
print("VALIDATION:")
print("X:", X_val.shape)
print("y:", y_val.shape)

print()
print("TEST:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# ============================================================
# Standardization
# ============================================================

print()
print("=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

print("Fitting scaler on TRAINING data only...")

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


print("Train:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)


# Save scaler

np.savez(
    SCALER_PATH,
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32),
)

print()
print("Scaler saved to:")
print(SCALER_PATH)


# ============================================================
# Number of classes
# ============================================================

NUM_CLASSES = int(
    max(
        np.max(y_train),
        np.max(y_val),
        np.max(y_test),
    ) + 1
)

INPUT_SIZE = X_train_scaled.shape[1]

print()
print("Number of classes:", NUM_CLASSES)
print("Input features:", INPUT_SIZE)


# ============================================================
# Model
# ============================================================

print()
print("=" * 70)
print("BUILDING V2 MODEL")
print("=" * 70)


model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(INPUT_SIZE,)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    ),
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],
)


model.summary()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]


# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("TRAINING V2")
print("=" * 70)

history = model.fit(
    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val,
    ),

    epochs=100,

    batch_size=64,

    callbacks=callbacks,

    verbose=1,
)


# ============================================================
# Final test evaluation
# ============================================================

print()
print("=" * 70)
print("FINAL V2 TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=1,
)


print()
print("Test Loss:", round(float(test_loss), 4))

print(
    "Test Accuracy:",
    round(float(test_accuracy) * 100, 2),
    "%"
)


# ============================================================
# Save training history
# ============================================================

history_path = REPORT_DIR / "training_history_v2.json"

history_json = {
    key: [
        float(value)
        for value in values
    ]

    for key, values in history.history.items()
}


with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_json,
        f,
        indent=4,
    )


# ============================================================
# Save config
# ============================================================

config = {
    "version": "V2",

    "input_features": INPUT_SIZE,

    "num_classes": NUM_CLASSES,

    "architecture": [
        256,
        128,
        64,
        NUM_CLASSES,
    ],

    "dropout": [
        0.30,
        0.25,
        0.20,
    ],

    "batch_size": 64,

    "max_epochs": 100,

    "seed": SEED,

    "test_loss": float(test_loss),

    "test_accuracy": float(test_accuracy),
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


print()
print("=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Scaler:")
print(SCALER_PATH)

print()
print("History:")
print(history_path)

print()
print("Config:")
print(CONFIG_PATH)