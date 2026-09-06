"""Evaluate the current PSL V2 model on independent Mendeley Tuey/Daal images.

This is an audit only: it does not change the training set, model, scaler, or
class map. It confirms external-data compatibility before any retraining work.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from extract_landmarks import create_landmarker, process_image
from feature_engineering import extract_features
from v2_preprocessing import load_and_scale_v2_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "output" / "psl_static_model_v2.keras"
SCALER_PATH = ROOT / "output" / "psl_v2_scaler.npz"
CLASS_MAP_PATH = ROOT / "output" / "class_map.json"
REPORT_PATH = ROOT / "reports" / "external_mendeley_tuey_daal_audit.csv"

def load_class_map() -> dict[int, str]:
    with open(CLASS_MAP_PATH, encoding="utf-8") as file:
        return {int(key): str(value) for key, value in json.load(file).items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PSL V2 on Mendeley raw Tuey and Daal images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the dataset's PSL folder (the folder containing Urdu class folders).",
    )
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand-detection and presence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--tuey-folder",
        default="Toay'n ط",
        help="Name of the Tuey (ط) folder under --dataset-root.",
    )
    parser.add_argument(
        "--daal-folder",
        default="Daal ڈ",
        help="Name of the Daal (ڈ) folder under --dataset-root.",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_root}")

    target_folders = {"Tuey": args.tuey_folder, "Daal": args.daal_folder}
    samples: list[tuple[str, Path]] = []
    for expected, folder_name in target_folders.items():
        folder = args.dataset_root / folder_name
        images = (
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if folder.is_dir()
            else []
        )
        if not images:
            raise FileNotFoundError(f"No supported images found for {expected}: {folder}")
        samples.extend((expected, image) for image in images)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_map = load_class_map()
    if not 0.0 < args.hand_confidence <= 1.0:
        raise ValueError("--hand-confidence must be greater than 0 and at most 1.")

    landmarker = create_landmarker(
        min_detection_confidence=args.hand_confidence,
        min_presence_confidence=args.hand_confidence,
    )
    rows: list[dict[str, object]] = []

    try:
        for expected, image_path in samples:
            base_features = process_image(image_path, landmarker)
            if base_features is None:
                rows.append(
                    {
                        "expected": expected,
                        "predicted": "NO_HAND_DETECTED",
                        "confidence": "",
                        "correct": False,
                        "image_path": str(image_path),
                    }
                )
                continue

            features = extract_features(base_features.reshape(1, -1))
            scaled = load_and_scale_v2_features(features, SCALER_PATH)
            probabilities = model.predict(scaled, verbose=0)[0]
            predicted_id = int(np.argmax(probabilities))
            predicted = class_map[predicted_id]
            rows.append(
                {
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": float(probabilities[predicted_id]),
                    "correct": predicted == expected,
                    "image_path": str(image_path),
                }
            )
    finally:
        landmarker.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("External PSL V2 audit: Tuey/Daal")
    print("=" * 48)
    print(f"MediaPipe hand confidence threshold: {args.hand_confidence:.2f}")
    for expected in target_folders:
        subset = [row for row in rows if row["expected"] == expected]
        correct = sum(bool(row["correct"]) for row in subset)
        predictions = Counter(str(row["predicted"]) for row in subset)
        print(f"{expected}: {correct}/{len(subset)} correct ({correct / len(subset):.2%})")
        print(f"  Predictions: {dict(predictions)}")

    total_correct = sum(bool(row["correct"]) for row in rows)
    print(f"Overall: {total_correct}/{len(rows)} correct ({total_correct / len(rows):.2%})")
    print(f"Saved per-image report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
