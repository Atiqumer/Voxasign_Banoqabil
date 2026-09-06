from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from config import (
    LANDMARK_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    PSL_CLASSES,
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    OUTPUT_DIR / "psl_static_model_v1.keras"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():

    X_test = np.load(
        LANDMARK_DIR / "X_test.npy"
    )

    y_test = np.load(
        LANDMARK_DIR / "y_test.npy"
    )

    return X_test, y_test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VoxaSign PSL Model v1 Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print(
        f"\nTest samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
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

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(PSL_CLASSES))),
        target_names=PSL_CLASSES,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORT_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VoxaSign PSL Static Model v1\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Macro F1: {macro_f1 * 100:.2f}%\n"
        )

        f.write(
            f"Weighted F1: {weighted_f1 * 100:.2f}%\n\n"
        )

        f.write(
            report
        )

    print(
        f"Report saved to:\n{report_path}"
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(PSL_CLASSES)))
    )

    fig, ax = plt.subplots(
        figsize=(18, 16)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=PSL_CLASSES
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=True,
    )

    ax.set_title(
        "VoxaSign PSL Static Model v1 — Confusion Matrix"
    )

    plt.tight_layout()

    cm_path = (
        REPORT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n{cm_path}"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    class_results = []

    for class_id, class_name in enumerate(
        PSL_CLASSES
    ):

        mask = (
            y_test == class_id
        )

        total = np.sum(mask)

        correct = np.sum(
            y_pred[mask] == class_id
        )

        class_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        class_results.append(
            {
                "class_id": class_id,
                "class": class_name,
                "correct": int(correct),
                "total": int(total),
                "accuracy": float(
                    class_accuracy
                ),
            }
        )

        print(
            f"{class_name:15s} "
            f"{correct:2d}/{total:2d} "
            f"= {class_accuracy * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {
        "model": "psl_static_model_v1",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": class_results,
    }

    results_path = (
        REPORT_DIR /
        "evaluation_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nEvaluation JSON saved to:\n"
        f"{results_path}"
    )

    # --------------------------------------------------------
    # Weakest classes
    # --------------------------------------------------------

    sorted_classes = sorted(
        class_results,
        key=lambda x: x["accuracy"]
    )

    print("\n" + "=" * 70)
    print("5 WEAKEST CLASSES")
    print("=" * 70)

    for item in sorted_classes[:5]:

        print(
            f"{item['class']:15s} "
            f"{item['accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Best classes
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5 BEST CLASSES")
    print("=" * 70)

    for item in sorted_classes[-5:][::-1]:

        print(
            f"{item['class']:15s} "
            f"{item['accuracy'] * 100:.2f}%"
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()