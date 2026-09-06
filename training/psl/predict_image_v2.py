"""Run the verified VoxaSign PSL V2 model on one image.

This uses the same landmark normalization, V2 feature extraction, and saved
training-only scaler as the canonical offline evaluation.
"""

from __future__ import annotations

import argparse
import json
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


def load_class_map() -> dict[int, str]:
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as file:
        class_map = json.load(file)

    try:
        return {int(class_id): str(name) for class_id, name in class_map.items()}
    except (AttributeError, ValueError) as exc:
        raise ValueError("class_map.json must map class IDs to class names.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a PSL class from one hand-sign image using V2."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    print(f"Image: {args.image}")
    print("Detecting one hand with MediaPipe...")

    landmarker = create_landmarker()
    try:
        base_features = process_image(args.image, landmarker)
    finally:
        landmarker.close()

    if base_features is None:
        raise RuntimeError(
            "No usable hand landmarks were detected. Use an image with one "
            "fully visible hand and good lighting."
        )

    v2_features = extract_features(base_features.reshape(1, -1))
    scaled_features = load_and_scale_v2_features(v2_features, SCALER_PATH)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(scaled_features, verbose=0)[0]
    class_map = load_class_map()

    top_k = min(args.top_k, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print()
    print("VoxaSign PSL V2 prediction")
    print("=" * 32)
    for rank, class_id in enumerate(top_indices, start=1):
        print(
            f"{rank}. {class_map[int(class_id)]:<15} "
            f"{probabilities[class_id] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
