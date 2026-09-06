import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output" / "psl_static_model_v2.keras"
SCALER_PATH = BASE_DIR / "output" / "psl_v2_scaler.npz"
CONFIG_PATH = BASE_DIR / "output" / "psl_v2_config.json"
CLASS_MAP_PATH = BASE_DIR / "output" / "class_map.json"

X_TEST_PATH = BASE_DIR / "data" / "landmarks" / "v2" / "X_test_v2.npy"
Y_TEST_PATH = BASE_DIR / "data" / "landmarks" / "y_test.npy"

REPORT_DIR = BASE_DIR / "reports" / "tuey_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPORT_DIR / "tuey_predictions.csv"
JSON_PATH = REPORT_DIR / "tuey_confusion.json"
TXT_PATH = REPORT_DIR / "tuey_confusion.txt"


# ============================================================
# HELPERS
# ============================================================

def heading(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_class_map(class_map):
    """
    Supports either:
        {"0": "1-Hay", "1": "Ain", ...}

    or:
        {"1-Hay": 0, "Ain": 1, ...}
    """

    if not class_map:
        raise RuntimeError("Class map is empty.")

    first_key = next(iter(class_map))
    first_value = class_map[first_key]

    if isinstance(first_value, str):
        return {int(k): v for k, v in class_map.items()}

    return {int(v): k for k, v in class_map.items()}


def load_scaler(path):
    """
    Load scaler saved with np.savez.

    Expected keys may include:
        mean
        scale

    or:
        mean_
        scale_
    """

    data = np.load(path)

    print("Scaler keys:", list(data.keys()))

    mean = None
    scale = None

    for key in ["mean", "mean_", "center"]:
        if key in data:
            mean = data[key]
            break

    for key in ["scale", "scale_", "std"]:
        if key in data:
            scale = data[key]
            break

    if mean is None or scale is None:
        raise RuntimeError(
            "Could not find scaler mean/scale inside psl_v2_scaler.npz"
        )

    return mean, scale


def apply_scaler(X, mean, scale):
    return (X - mean) / scale


# ============================================================
# MAIN
# ============================================================

def main():

    heading("VoxaSign PSL — Tuey Confusion Analysis V2 VERIFIED")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    heading("CHECKING REQUIRED FILES")

    required = [
        MODEL_PATH,
        SCALER_PATH,
        CONFIG_PATH,
        CLASS_MAP_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

        print(f"OK: {path}")

    # --------------------------------------------------------
    # CLASS MAP
    # --------------------------------------------------------

    heading("LOADING CLASS MAP")

    raw_class_map = load_json(CLASS_MAP_PATH)
    class_map = normalize_class_map(raw_class_map)

    print(f"Number of classes: {len(class_map)}")

    inverse_map = class_map

    tuey_id = None
    bay_id = None
    daal_id = None

    for idx, name in inverse_map.items():

        if name == "Tuey":
            tuey_id = idx

        elif name == "Bay":
            bay_id = idx

        elif name == "Daal":
            daal_id = idx

    if tuey_id is None:
        raise RuntimeError("Tuey not found in class map.")

    print(f"Tuey = {tuey_id}")
    print(f"Bay  = {bay_id}")
    print(f"Daal = {daal_id}")

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    heading("LOADING V2 CONFIG")

    config = load_json(CONFIG_PATH)

    print(json.dumps(config, indent=2))

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    heading("LOADING TEST DATA")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"Raw X_test shape: {X_test.shape}")
    print(f"y_test shape:     {y_test.shape}")

    if len(X_test) != len(y_test):
        raise RuntimeError("X_test and y_test have different sample counts.")

    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------

    heading("LOADING V2 SCALER")

    mean, scale = load_scaler(SCALER_PATH)

    print(f"Scaler mean shape:  {mean.shape}")
    print(f"Scaler scale shape: {scale.shape}")

    if X_test.shape[1] != len(mean):
        raise RuntimeError(
            f"Feature mismatch:\n"
            f"X_test features = {X_test.shape[1]}\n"
            f"Scaler features = {len(mean)}"
        )

    # --------------------------------------------------------
    # APPLY V2 PREPROCESSING
    # --------------------------------------------------------

    heading("APPLYING V2 PREPROCESSING")

    X_test_scaled = apply_scaler(X_test.astype(np.float32), mean, scale)

    print(f"Scaled X_test shape: {X_test_scaled.shape}")
    print(
        f"Scaled mean: {X_test_scaled.mean():.6f}"
    )
    print(
        f"Scaled std:  {X_test_scaled.std():.6f}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    heading("LOADING V2 MODEL")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    if model.input_shape[-1] != X_test_scaled.shape[1]:
        raise RuntimeError(
            f"Model/input feature mismatch:\n"
            f"Model expects {model.input_shape[-1]}\n"
            f"Data contains {X_test_scaled.shape[1]}"
        )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    heading("GENERATING VERIFIED PREDICTIONS")

    probabilities = model.predict(
        X_test_scaled,
        verbose=1
    )

    predictions = np.argmax(probabilities, axis=1)

    overall_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"VERIFIED OVERALL TEST ACCURACY: "
        f"{overall_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    heading("V2 SANITY CHECK")

    expected_accuracy = 0.9423076923076923

    difference = abs(
        overall_accuracy - expected_accuracy
    )

    print(
        f"Expected V2 accuracy: "
        f"{expected_accuracy * 100:.2f}%"
    )

    print(
        f"Current accuracy:      "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference * 100:.4f} percentage points"
    )

    if difference <= 0.01:
        print()
        print("SUCCESS: V2 preprocessing appears consistent.")
    else:
        print()
        print(
            "WARNING: Accuracy does NOT match the original V2 evaluation."
        )
        print(
            "Do NOT use this confusion analysis for V3 yet."
        )

    # --------------------------------------------------------
    # TUEY SAMPLES
    # --------------------------------------------------------

    heading("TUEY TEST SAMPLES")

    tuey_indices = np.where(
        y_test == tuey_id
    )[0]

    tuey_total = len(tuey_indices)

    tuey_correct = np.sum(
        predictions[tuey_indices] == tuey_id
    )

    tuey_accuracy = (
        tuey_correct / tuey_total
        if tuey_total > 0
        else 0
    )

    print(f"Tuey test samples: {tuey_total}")
    print(
        f"Tuey correct: "
        f"{tuey_correct}/{tuey_total}"
    )

    print(
        f"Tuey recall: "
        f"{tuey_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TUEY -> OTHER
    # --------------------------------------------------------

    heading("TUEY → OTHER CLASS CONFUSION")

    tuey_predictions = {}

    for idx in tuey_indices:

        pred_id = int(predictions[idx])

        tuey_predictions[pred_id] = (
            tuey_predictions.get(pred_id, 0) + 1
        )

    for pred_id, count in sorted(
        tuey_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        name = inverse_map[pred_id]

        print(
            f"TUEY -> {name:<15} "
            f"{count}/{tuey_total} "
            f"({count / tuey_total * 100:6.2f}%)"
        )

    # --------------------------------------------------------
    # OTHER -> TUEY
    # --------------------------------------------------------

    heading("OTHER CLASS → TUEY CONFUSION")

    other_to_tuey = {}

    for idx in range(len(y_test)):

        true_id = int(y_test[idx])
        pred_id = int(predictions[idx])

        if (
            pred_id == tuey_id
            and true_id != tuey_id
        ):

            other_to_tuey[true_id] = (
                other_to_tuey.get(true_id, 0) + 1
            )

    if other_to_tuey:

        for true_id, count in sorted(
            other_to_tuey.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(
                f"{inverse_map[true_id]:<15}"
                f" -> Tuey       {count}"
            )

    else:

        print("No other classes predicted as Tuey.")

    # --------------------------------------------------------
    # TUEY METRICS
    # --------------------------------------------------------

    heading("TUEY METRICS")

    tuey_precision = precision_score(
        y_test == tuey_id,
        predictions == tuey_id,
        zero_division=0
    )

    tuey_recall = recall_score(
        y_test == tuey_id,
        predictions == tuey_id,
        zero_division=0
    )

    tuey_f1 = f1_score(
        y_test == tuey_id,
        predictions == tuey_id,
        zero_division=0
    )

    print(
        f"Precision: {tuey_precision * 100:.2f}%"
    )

    print(
        f"Recall:    {tuey_recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  {tuey_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CONFUSION PAIRS
    # --------------------------------------------------------

    heading("TOP CONFUSION PAIRS")

    confusion_pairs = {}

    for true_id, pred_id in zip(
        y_test,
        predictions
    ):

        true_id = int(true_id)
        pred_id = int(pred_id)

        if true_id == pred_id:
            continue

        pair = (
            inverse_map[true_id],
            inverse_map[pred_id]
        )

        confusion_pairs[pair] = (
            confusion_pairs.get(pair, 0) + 1
        )

    for (true_name, pred_name), count in sorted(
        confusion_pairs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:

        print(
            f"{true_name:<15} -> "
            f"{pred_name:<15} {count}"
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    heading("TUEY CONFIDENCE ANALYSIS")

    tuey_confidences = []

    for idx in tuey_indices:

        confidence = float(
            probabilities[idx][predictions[idx]]
        )

        tuey_confidences.append(confidence)

    tuey_confidences = np.array(
        tuey_confidences
    )

    print(
        f"Mean confidence:   "
        f"{tuey_confidences.mean() * 100:.2f}%"
    )

    print(
        f"Median confidence: "
        f"{np.median(tuey_confidences) * 100:.2f}%"
    )

    print(
        f"Minimum confidence:"
        f"{tuey_confidences.min() * 100:.2f}%"
    )

    print(
        f"Maximum confidence:"
        f"{tuey_confidences.max() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # INDIVIDUAL TUEY
    # --------------------------------------------------------

    heading("INDIVIDUAL TUEY PREDICTIONS")

    records = []

    for idx in tuey_indices:

        pred_id = int(predictions[idx])

        confidence = float(
            probabilities[idx][pred_id]
        )

        correct = (
            pred_id == tuey_id
        )

        print(
            f"Index {idx:4d} | "
            f"True=Tuey       | "
            f"Predicted={inverse_map[pred_id]:<12} | "
            f"Confidence={confidence * 100:6.2f}% | "
            f"{'CORRECT' if correct else 'WRONG'}"
        )

        records.append({
            "test_index": int(idx),
            "true_class": "Tuey",
            "predicted_class": inverse_map[pred_id],
            "predicted_id": pred_id,
            "confidence": confidence,
            "correct": bool(correct),
        })

    # --------------------------------------------------------
    # ALTERNATIVE PREDICTIONS
    # --------------------------------------------------------

    heading("TUEY SAMPLE ALTERNATIVE PREDICTIONS")

    for idx in tuey_indices:

        print()
        print(f"Test index {idx}:")

        probs = probabilities[idx]

        top_indices = np.argsort(
            probs
        )[::-1][:5]

        for class_id in top_indices:

            print(
                f"  {inverse_map[int(class_id)]:<15}"
                f"{probs[class_id] * 100:6.2f}%"
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    heading("SAVING CSV")

    import csv

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_index",
                "true_class",
                "predicted_class",
                "predicted_id",
                "confidence",
                "correct",
            ]
        )

        writer.writeheader()

        writer.writerows(records)

    print(f"Saved: {CSV_PATH}")

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    heading("SAVING JSON")

    results = {
        "overall_accuracy": float(
            overall_accuracy
        ),
        "expected_v2_accuracy": expected_accuracy,
        "accuracy_difference": float(
            difference
        ),
        "preprocessing_verified": bool(
            difference <= 0.01
        ),
        "test_samples": int(len(y_test)),
        "tuey_id": int(tuey_id),
        "tuey_total": int(tuey_total),
        "tuey_correct": int(tuey_correct),
        "tuey_accuracy": float(tuey_accuracy),
        "tuey_precision": float(tuey_precision),
        "tuey_recall": float(tuey_recall),
        "tuey_f1": float(tuey_f1),
        "tuey_confusion": {
            inverse_map[k]: int(v)
            for k, v in tuey_predictions.items()
        },
        "other_to_tuey": {
            inverse_map[k]: int(v)
            for k, v in other_to_tuey.items()
        },
        "tuey_mean_confidence": float(
            tuey_confidences.mean()
        ),
        "tuey_median_confidence": float(
            np.median(tuey_confidences)
        ),
        "tuey_min_confidence": float(
            tuey_confidences.min()
        ),
        "tuey_max_confidence": float(
            tuey_confidences.max()
        ),
        "individual_predictions": records,
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(f"Saved: {JSON_PATH}")

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    heading("SAVING TEXT REPORT")

    lines = []

    lines.append(
        "VoxaSign PSL — Tuey Confusion Analysis V2 VERIFIED"
    )

    lines.append("=" * 60)

    lines.append(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    lines.append(
        f"Expected V2 accuracy: "
        f"{expected_accuracy * 100:.2f}%"
    )

    lines.append(
        f"Preprocessing verified: "
        f"{difference <= 0.01}"
    )

    lines.append("")

    lines.append(
        f"Tuey samples: {tuey_total}"
    )

    lines.append(
        f"Tuey correct: {tuey_correct}/{tuey_total}"
    )

    lines.append(
        f"Tuey recall: {tuey_recall * 100:.2f}%"
    )

    lines.append(
        f"Tuey precision: {tuey_precision * 100:.2f}%"
    )

    lines.append(
        f"Tuey F1: {tuey_f1 * 100:.2f}%"
    )

    lines.append("")

    lines.append("Tuey confusion:")

    for pred_id, count in sorted(
        tuey_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        lines.append(
            f"Tuey -> {inverse_map[pred_id]}: "
            f"{count}"
        )

    lines.append("")

    lines.append("Other -> Tuey:")

    for true_id, count in sorted(
        other_to_tuey.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        lines.append(
            f"{inverse_map[true_id]} -> Tuey: "
            f"{count}"
        )

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(f"Saved: {TXT_PATH}")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    heading("ANALYSIS COMPLETE")

    print(
        f"Verified V2 accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Tuey recall: "
        f"{tuey_recall * 100:.2f}%"
    )

    print(
        f"Tuey F1: "
        f"{tuey_f1 * 100:.2f}%"
    )

    print()
    print("IMPORTANT:")

    if difference <= 0.01:

        print(
            "V2 preprocessing is consistent."
        )

        print(
            "We can now safely use the Tuey confusion "
            "results to design V3."
        )

    else:

        print(
            "V2 preprocessing mismatch detected."
        )

        print(
            "DO NOT train V3 yet."
        )


if __name__ == "__main__":
    main()