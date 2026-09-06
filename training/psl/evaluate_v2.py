"""
VoxaSign PSL V2 Evaluation

Produces:

- Accuracy
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix
- Per-class accuracy
- Weakest classes
- Best classes
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_DIR = ROOT / "data" / "landmarks" / "v2"

LANDMARK_DIR = ROOT / "data" / "landmarks"

OUTPUT_DIR = ROOT / "output"

REPORT_DIR = ROOT / "reports"

EVAL_DIR = REPORT_DIR / "v2"

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    OUTPUT_DIR /
    "psl_static_model_v2.keras"
)

SCALER_PATH = (
    OUTPUT_DIR /
    "psl_v2_scaler.npz"
)


# ============================================================
# Class names
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


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("VoxaSign PSL V2 Evaluation")
print("=" * 70)

print()
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")


X_test = np.load(
    FEATURE_DIR / "X_test_v2.npy"
)

y_test = np.load(
    LANDMARK_DIR / "y_test.npy"
)


# ============================================================
# Load scaler
# ============================================================

scaler_data = np.load(
    SCALER_PATH
)

mean = scaler_data["mean"]

scale = scaler_data["scale"]

X_test_scaled = (
    X_test - mean
) / np.where(
    scale == 0,
    1.0,
    scale
)


print()
print("Test samples:", len(y_test))

print()
print("Generating predictions...")


# ============================================================
# Predictions
# ============================================================

probabilities = model.predict(
    X_test_scaled,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# Overall metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)


print()
print("=" * 70)
print("OVERALL V2 RESULTS")
print("=" * 70)

print(
    f"Accuracy:     {accuracy * 100:.2f}%"
)

print(
    f"Macro F1:     {macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1:  {weighted_f1 * 100:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    y_test,
    predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_path = (
    EVAL_DIR /
    "classification_report_v2.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(CLASS_NAMES)))
)


plt.figure(
    figsize=(16, 14)
)

plt.imshow(cm)

plt.title(
    "VoxaSign PSL V2 Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=90
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.colorbar()

plt.tight_layout()

cm_path = (
    EVAL_DIR /
    "confusion_matrix_v2.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Per-class accuracy
# ============================================================

per_class = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    mask = y_test == class_index

    total = int(np.sum(mask))

    correct = int(
        np.sum(
            predictions[mask] == class_index
        )
    )

    if total > 0:
        class_accuracy = (
            correct / total
        )
    else:
        class_accuracy = 0.0

    per_class[class_name] = {
        "correct": correct,
        "total": total,
        "accuracy": class_accuracy,
    }


print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)


for class_name, result in per_class.items():

    print(
        f"{class_name:<15} "
        f"{result['correct']:>2}/"
        f"{result['total']:<2} = "
        f"{result['accuracy'] * 100:>6.2f}%"
    )


# ============================================================
# Weakest / best classes
# ============================================================

sorted_classes = sorted(
    per_class.items(),
    key=lambda x: x[1]["accuracy"]
)


print()
print("=" * 70)
print("5 WEAKEST CLASSES")
print("=" * 70)

for name, result in sorted_classes[:5]:

    print(
        f"{name:<15} "
        f"{result['accuracy'] * 100:.2f}%"
    )


print()
print("=" * 70)
print("5 BEST CLASSES")
print("=" * 70)

for name, result in sorted_classes[-5:][::-1]:

    print(
        f"{name:<15} "
        f"{result['accuracy'] * 100:.2f}%"
    )


# ============================================================
# Save JSON
# ============================================================

evaluation = {
    "version": "V2",

    "test_samples": int(len(y_test)),

    "accuracy": float(accuracy),

    "macro_f1": float(macro_f1),

    "weighted_f1": float(weighted_f1),

    "per_class": per_class,

    "weakest_classes": [
        {
            "class": name,
            "accuracy": float(result["accuracy"]),
        }

        for name, result in sorted_classes[:5]
    ],

    "best_classes": [
        {
            "class": name,
            "accuracy": float(result["accuracy"]),
        }

        for name, result in sorted_classes[-5:][::-1]
    ],
}


json_path = (
    EVAL_DIR /
    "evaluation_results_v2.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation,
        f,
        indent=4,
    )


print()
print("=" * 70)
print("V2 EVALUATION COMPLETE")
print("=" * 70)

print()
print("Classification report:")
print(report_path)

print()
print("Confusion matrix:")
print(cm_path)

print()
print("Evaluation JSON:")
print(json_path)