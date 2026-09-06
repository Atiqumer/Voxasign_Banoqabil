from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from config import (
    LANDMARK_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    PSL_CLASSES,
    INPUT_FEATURES,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        LANDMARK_DIR / "X_train.npy"
    )

    y_train = np.load(
        LANDMARK_DIR / "y_train.npy"
    )

    X_val = np.load(
        LANDMARK_DIR / "X_validation.npy"
    )

    y_val = np.load(
        LANDMARK_DIR / "y_validation.npy"
    )

    X_test = np.load(
        LANDMARK_DIR / "X_test.npy"
    )

    y_test = np.load(
        LANDMARK_DIR / "y_test.npy"
    )

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Train:      X={X_train.shape}, "
        f"y={y_train.shape}"
    )

    print(
        f"Validation: X={X_val.shape}, "
        f"y={y_val.shape}"
    )

    print(
        f"Test:       X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y_train):

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            weights
        )
    }

    print("\nClass weights:")

    for class_id, weight in class_weights.items():

        print(
            f"{class_id:2d} "
            f"{PSL_CLASSES[class_id]:15s} "
            f"{weight:.3f}"
        )

    return class_weights


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(INPUT_FEATURES,)
            ),

            tf.keras.layers.Dense(
                256,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.30
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.25
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.15
            ),

            tf.keras.layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            ),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():

    best_model_path = (
        CHECKPOINT_DIR /
        "best_psl_model.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return callbacks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VoxaSign PSL Static Model v1")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    assert X_train.shape[1] == INPUT_FEATURES
    assert X_val.shape[1] == INPUT_FEATURES
    assert X_test.shape[1] == INPUT_FEATURES

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        y_train
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    model.summary()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=create_callbacks(),

        verbose=1,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_accuracy = (
        model.evaluate(
            X_test,
            y_test,
            verbose=1,
        )
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        OUTPUT_DIR /
        "psl_static_model_v1.keras"
    )

    model.save(
        final_model_path
    )

    print(
        f"\nModel saved to:\n"
        f"{final_model_path}"
    )

    # --------------------------------------------------------
    # Save class map
    # --------------------------------------------------------

    class_map = {
        str(index): name
        for index, name in enumerate(
            PSL_CLASSES
        )
    }

    with open(
        OUTPUT_DIR / "class_map.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_map,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        REPORT_DIR /
        "training_history.json"
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history_data,
            f,
            indent=2
        )

    print(
        f"Training history saved to:\n"
        f"{history_path}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()