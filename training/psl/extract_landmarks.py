from pathlib import Path
import json

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from config import (
    METADATA_DIR,
    LANDMARK_DIR,
    PSL_CLASSES,
    NUM_LANDMARKS,
    INPUT_FEATURES,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAND_LANDMARKER_PATH = (
    PROJECT_ROOT / "hand_landmarker.task"
)

METADATA_FILE = (
    METADATA_DIR / "psl_static_metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_landmarks(landmarks):
    """
    Convert 21 MediaPipe landmarks into 63 normalized values.

    Steps:
    1. Make wrist the origin.
    2. Normalize by maximum absolute coordinate.

    Returns:
        np.ndarray with shape (63,)
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Wrist-relative coordinates
    # --------------------------------------------------------

    wrist = points[0]

    points = points - wrist

    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(points)
    )

    if max_value > 0:
        points = points / max_value

    return points.flatten()


# ============================================================
# MEDIAPIPE
# ============================================================

def create_landmarker(
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_presence_confidence: float = MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
):

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                HAND_LANDMARKER_PATH
            )
        ),
        running_mode=(
            VisionRunningMode.IMAGE
        ),
        num_hands=1,
        min_hand_detection_confidence=(
            min_detection_confidence
        ),
        min_hand_presence_confidence=(
            min_presence_confidence
        ),
        min_tracking_confidence=(
            min_tracking_confidence
        ),
    )

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path,
    landmarker
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.hand_landmarks:
        return None

    # First detected hand
    landmarks = result.hand_landmarks[0]

    if len(landmarks) != NUM_LANDMARKS:
        return None

    features = normalize_landmarks(
        landmarks
    )

    if features.shape != (INPUT_FEATURES,):
        return None

    return features


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    df,
    split,
    landmarker,
):

    split_df = df[
        df["split"] == split
    ].reset_index(drop=True)

    X = []
    y = []

    failed = []

    print()
    print("=" * 70)
    print(f"Processing {split.upper()}")
    print("=" * 70)

    total = len(split_df)

    for index, row in split_df.iterrows():

        image_path = Path(
            row["path"]
        )

        features = process_image(
            image_path,
            landmarker
        )

        if features is None:

            failed.append(
                {
                    "path": str(image_path),
                    "class": row["class"],
                    "subject_id": row["subject_id"],
                }
            )

            continue

        X.append(features)

        y.append(
            PSL_CLASSES.index(
                row["class"]
            )
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == total
        ):
            print(
                f"\rProcessed "
                f"{index + 1}/{total}",
                end=""
            )

    print()

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print(
        f"Successful: {len(X)}"
    )

    print(
        f"Failed:     {len(failed)}"
    )

    return X, y, failed


# ============================================================
# SAVE
# ============================================================

def save_split(
    split,
    X,
    y
):

    LANDMARK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        LANDMARK_DIR / f"X_{split}.npy",
        X
    )

    np.save(
        LANDMARK_DIR / f"y_{split}.npy",
        y
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VoxaSign PSL Landmark Extraction")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not HAND_LANDMARKER_PATH.exists():

        raise FileNotFoundError(
            "hand_landmarker.task not found at:\n"
            f"{HAND_LANDMARKER_PATH}"
        )

    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "Dataset metadata not found:\n"
            f"{METADATA_FILE}\n\n"
            "Run prepare_dataset.py first."
        )

    df = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"\nMetadata rows: {len(df)}"
    )

    print(
        f"Classes: {df['class'].nunique()}"
    )

    # --------------------------------------------------------
    # Create MediaPipe detector
    # --------------------------------------------------------

    print(
        "\nLoading MediaPipe HandLandmarker..."
    )

    landmarker = create_landmarker()

    print("MediaPipe loaded successfully.")

    all_failures = {}

    try:

        for split in [
            "train",
            "validation",
            "test",
        ]:

            X, y, failed = process_split(
                df,
                split,
                landmarker
            )

            save_split(
                split,
                X,
                y
            )

            all_failures[split] = failed

            print(
                f"\n{split.upper()} shape:"
            )

            print(
                f"X = {X.shape}"
            )

            print(
                f"y = {y.shape}"
            )

    finally:

        landmarker.close()

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    with open(
        LANDMARK_DIR / "class_names.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            PSL_CLASSES,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save failures
    # --------------------------------------------------------

    failure_file = (
        LANDMARK_DIR /
        "failed_images.json"
    )

    with open(
        failure_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_failures,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        X = np.load(
            LANDMARK_DIR /
            f"X_{split}.npy"
        )

        y = np.load(
            LANDMARK_DIR /
            f"y_{split}.npy"
        )

        print(
            f"{split:12} "
            f"X={X.shape} "
            f"y={y.shape}"
        )

    print(
        "\nSaved to:"
    )

    print(
        LANDMARK_DIR
    )


if __name__ == "__main__":
    main()
