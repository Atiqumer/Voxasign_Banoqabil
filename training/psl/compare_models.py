"""
VoxaSign PSL V1 vs V2 Comparison

V1:
    63 normalized landmark coordinates

V2:
    115 engineered features

Both models are evaluated on the SAME test set.
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

LANDMARK_DIR = ROOT / "data" / "landmarks"

V2_FEATURE_DIR = (
    LANDMARK_DIR / "v2"
)

OUTPUT_DIR = ROOT / "output"

REPORT_DIR = ROOT / "reports"

COMPARE_DIR = (
    REPORT_DIR / "model_comparison"
)

COMPARE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


V1_MODEL = (
    OUTPUT_DIR /
    "psl_static_model_v1.keras"
)

V2_MODEL = (
    OUTPUT_DIR /
    "psl_static_model_v2.keras"
)

V2_SCALER = (
    OUTPUT_DIR /
    "psl_v2_scaler.npz"
)


# ============================================================
# Load test labels
# ============================================================

y_test = np.load(
    LANDMARK_DIR /
    "y_test.npy"
)


# ============================================================
# V1
# ============================================================

print("=" * 70)
print("VoxaSign PSL V1 vs V2")
print("=" * 70)


print()
print("Loading V1 model...")

v1_model = tf.keras.models.load_model(
    V1_MODEL
)

print("V1 loaded.")


X_test_v1 = np.load(
    LANDMARK_DIR /
    "X_test.npy"
)


print()
print("Evaluating V1...")

v1_prob = v1_model.predict(
    X_test_v1,
    verbose=0
)

v1_pred = np.argmax(
    v1_prob,
    axis=1
)


v1_accuracy = accuracy_score(
    y_test,
    v1_pred
)

v1_macro_f1 = f1_score(
    y_test,
    v1_pred,
    average="macro"
)

v1_weighted_f1 = f1_score(
    y_test,
    v1_pred,
    average="weighted"
)


# ============================================================
# V2
# ============================================================

print()
print("Loading V2 model...")

v2_model = tf.keras.models.load_model(
    V2_MODEL
)

print("V2 loaded.")


X_test_v2 = np.load(
    V2_FEATURE_DIR /
    "X_test_v2.npy"
)


# Load V2 scaler

scaler_data = np.load(
    V2_SCALER
)

mean = scaler_data["mean"]

scale = scaler_data["scale"]

X_test_v2_scaled = (
    X_test_v2 - mean
) / np.where(
    scale == 0,
    1.0,
    scale
)


print()
print("Evaluating V2...")

v2_prob = v2_model.predict(
    X_test_v2_scaled,
    verbose=0
)

v2_pred = np.argmax(
    v2_prob,
    axis=1
)


v2_accuracy = accuracy_score(
    y_test,
    v2_pred
)

v2_macro_f1 = f1_score(
    y_test,
    v2_pred,
    average="macro"
)

v2_weighted_f1 = f1_score(
    y_test,
    v2_pred,
    average="weighted"
)


# ============================================================
# Comparison
# ============================================================

accuracy_difference = (
    v2_accuracy -
    v1_accuracy
)

macro_f1_difference = (
    v2_macro_f1 -
    v1_macro_f1
)

weighted_f1_difference = (
    v2_weighted_f1 -
    v1_weighted_f1
)


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Metric':<20}"
    f"{'V1':>12}"
    f"{'V2':>12}"
    f"{'Change':>12}"
)

print("-" * 56)

print(
    f"{'Accuracy':<20}"
    f"{v1_accuracy * 100:>11.2f}%"
    f"{v2_accuracy * 100:>11.2f}%"
    f"{accuracy_difference * 100:>+11.2f}%"
)

print(
    f"{'Macro F1':<20}"
    f"{v1_macro_f1 * 100:>11.2f}%"
    f"{v2_macro_f1 * 100:>11.2f}%"
    f"{macro_f1_difference * 100:>+11.2f}%"
)

print(
    f"{'Weighted F1':<20}"
    f"{v1_weighted_f1 * 100:>11.2f}%"
    f"{v2_weighted_f1 * 100:>11.2f}%"
    f"{weighted_f1_difference * 100:>+11.2f}%"
)


# ============================================================
# Daal / Tuey comparison
# ============================================================

CLASS_NAMES = [
    "1-Hay",
    "Ain",
    "Alif",
    "Bay",
    "Byeh",
    "Chay",
    "Cyeh",
    "Daal",
    "Dal",
    "Dochahay",
    "Fay",
    "Gaaf",
    "Ghain",
    "Hamza",
    "Kaf",
    "Khay",
    "Kiaf",
    "Lam",
    "Meem",
    "Nuun",
    "Nuungh",
    "Pay",
    "Ray",
    "Say",
    "Seen",
    "Sheen",
    "Suad",
    "Taay",
    "Tay",
    "Tuey",
    "Wao",
    "Zaal",
    "Zaey",
    "Zay",
    "Zuad",
    "Zuey",
]


def class_accuracy(y_true, y_pred, class_index):

    mask = y_true == class_index

    if np.sum(mask) == 0:
        return 0.0

    return np.mean(
        y_pred[mask] == class_index
    )


daal_index = CLASS_NAMES.index("Daal")

tuey_index = CLASS_NAMES.index("Tuey")


v1_daal = class_accuracy(
    y_test,
    v1_pred,
    daal_index
)

v2_daal = class_accuracy(
    y_test,
    v2_pred,
    daal_index
)

v1_tuey = class_accuracy(
    y_test,
    v1_pred,
    tuey_index
)

v2_tuey = class_accuracy(
    y_test,
    v2_pred,
    tuey_index
)


print()
print("=" * 70)
print("HARD CLASS COMPARISON")
print("=" * 70)

print()

print(
    f"{'Class':<15}"
    f"{'V1':>12}"
    f"{'V2':>12}"
    f"{'Change':>12}"
)

print("-" * 51)

print(
    f"{'Daal':<15}"
    f"{v1_daal * 100:>11.2f}%"
    f"{v2_daal * 100:>11.2f}%"
    f"{(v2_daal-v1_daal)*100:>+11.2f}%"
)

print(
    f"{'Tuey':<15}"
    f"{v1_tuey * 100:>11.2f}%"
    f"{v2_tuey * 100:>11.2f}%"
    f"{(v2_tuey-v1_tuey)*100:>+11.2f}%"
)


# ============================================================
# Recommendation
# ============================================================

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if v2_macro_f1 > v1_macro_f1:
    print("V2 has better Macro F1 than V1.")

elif v2_macro_f1 < v1_macro_f1:
    print("V1 has better Macro F1 than V2.")

else:
    print("V1 and V2 have equal Macro F1.")


if v2_accuracy > v1_accuracy:
    print("V2 has better overall accuracy.")

elif v2_accuracy < v1_accuracy:
    print("V1 has better overall accuracy.")

else:
    print("V1 and V2 have equal accuracy.")


# ============================================================
# Save JSON
# ============================================================

comparison = {

    "v1": {
        "accuracy": float(v1_accuracy),
        "macro_f1": float(v1_macro_f1),
        "weighted_f1": float(v1_weighted_f1),
        "daal_accuracy": float(v1_daal),
        "tuey_accuracy": float(v1_tuey),
    },

    "v2": {
        "accuracy": float(v2_accuracy),
        "macro_f1": float(v2_macro_f1),
        "weighted_f1": float(v2_weighted_f1),
        "daal_accuracy": float(v2_daal),
        "tuey_accuracy": float(v2_tuey),
    },

    "difference": {
        "accuracy": float(accuracy_difference),
        "macro_f1": float(macro_f1_difference),
        "weighted_f1": float(weighted_f1_difference),
        "daal": float(v2_daal - v1_daal),
        "tuey": float(v2_tuey - v1_tuey),
    },
}


comparison_path = (
    COMPARE_DIR /
    "v1_vs_v2.json"
)


with open(
    comparison_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=4,
    )


print()
print("Comparison saved to:")
print(comparison_path)

print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)