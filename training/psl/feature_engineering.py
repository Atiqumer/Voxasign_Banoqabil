"""
VoxaSign PSL V2 Feature Engineering

Input:
    data/landmarks/X_train.npy
    data/landmarks/X_validation.npy
    data/landmarks/X_test.npy

Current input:
    63 values = 21 MediaPipe landmarks × (x, y, z)

V2 adds:
    - Original 63 coordinates
    - Bone lengths
    - Joint angles
    - Wrist-to-fingertip distances
    - Fingertip pair distances
    - Palm geometry

IMPORTANT:
    All calculations are performed independently per sample.
    No information from validation/test is used to fit anything.
"""

from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parent

LANDMARK_DIR = ROOT / "data" / "landmarks"
FEATURE_DIR = LANDMARK_DIR / "v2"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MediaPipe hand landmark indices
# ============================================================

WRIST = 0

THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

FINGERS = [
    THUMB,
    INDEX,
    MIDDLE,
    RING,
    PINKY,
]


# ============================================================
# Utility functions
# ============================================================

def reshape_landmarks(X):
    """
    Convert:
        (N, 63)

    into:
        (N, 21, 3)
    """
    return X.reshape(-1, 21, 3)


def safe_norm(v):
    return np.linalg.norm(v, axis=-1)


def safe_angle(a, b, c):
    """
    Calculate angle ABC.

    a, b, c:
        (..., 3)

    Returns:
        (...,)
    """

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba, axis=-1, keepdims=True)
    bc_norm = np.linalg.norm(bc, axis=-1, keepdims=True)

    ba_norm = np.maximum(ba_norm, 1e-8)
    bc_norm = np.maximum(bc_norm, 1e-8)

    ba_unit = ba / ba_norm
    bc_unit = bc / bc_norm

    cosine = np.sum(ba_unit * bc_unit, axis=-1)

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.arccos(cosine)


# ============================================================
# Feature extraction
# ============================================================

def extract_features(X):
    """
    Generate V2 features.

    Original:
        63 coordinates

    Additional:
        Bone lengths
        Joint angles
        Wrist-fingertip distances
        Fingertip pair distances
        Palm distances

    Returns:
        numpy array
    """

    landmarks = reshape_landmarks(X)

    features = []

    # --------------------------------------------------------
    # 1. Original 63 landmark coordinates
    # --------------------------------------------------------

    features.append(landmarks.reshape(len(landmarks), -1))

    # --------------------------------------------------------
    # 2. Bone lengths
    # --------------------------------------------------------

    # Wrist -> first joint of each finger
    bone_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    bone_features = []

    for a, b in bone_pairs:
        distance = safe_norm(
            landmarks[:, a] - landmarks[:, b]
        )
        bone_features.append(distance)

    bone_features = np.stack(bone_features, axis=1)

    features.append(bone_features)

    # --------------------------------------------------------
    # 3. Joint angles
    # --------------------------------------------------------

    angle_triplets = [
        # Thumb
        (1, 2, 3),
        (2, 3, 4),

        # Index
        (5, 6, 7),
        (6, 7, 8),

        # Middle
        (9, 10, 11),
        (10, 11, 12),

        # Ring
        (13, 14, 15),
        (14, 15, 16),

        # Pinky
        (17, 18, 19),
        (18, 19, 20),
    ]

    angle_features = []

    for a, b, c in angle_triplets:
        angle = safe_angle(
            landmarks[:, a],
            landmarks[:, b],
            landmarks[:, c],
        )

        # Convert radians to normalized range [0, 1]
        angle = angle / np.pi

        angle_features.append(angle)

    angle_features = np.stack(angle_features, axis=1)

    features.append(angle_features)

    # --------------------------------------------------------
    # 4. Wrist -> fingertip distances
    # --------------------------------------------------------

    fingertips = [4, 8, 12, 16, 20]

    wrist_distances = []

    for fingertip in fingertips:
        distance = safe_norm(
            landmarks[:, fingertip] -
            landmarks[:, WRIST]
        )

        wrist_distances.append(distance)

    wrist_distances = np.stack(
        wrist_distances,
        axis=1
    )

    features.append(wrist_distances)

    # --------------------------------------------------------
    # 5. Fingertip-to-fingertip distances
    # --------------------------------------------------------

    fingertip_pairs = [
        (4, 8),
        (4, 12),
        (4, 16),
        (4, 20),

        (8, 12),
        (8, 16),
        (8, 20),

        (12, 16),
        (12, 20),

        (16, 20),
    ]

    fingertip_distances = []

    for a, b in fingertip_pairs:
        distance = safe_norm(
            landmarks[:, a] -
            landmarks[:, b]
        )

        fingertip_distances.append(distance)

    fingertip_distances = np.stack(
        fingertip_distances,
        axis=1
    )

    features.append(fingertip_distances)

    # --------------------------------------------------------
    # 6. Palm geometry
    # --------------------------------------------------------

    palm_pairs = [
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),

        (5, 9),
        (9, 13),
        (13, 17),
    ]

    palm_features = []

    for a, b in palm_pairs:
        distance = safe_norm(
            landmarks[:, a] -
            landmarks[:, b]
        )

        palm_features.append(distance)

    palm_features = np.stack(
        palm_features,
        axis=1
    )

    features.append(palm_features)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_features = np.concatenate(
        features,
        axis=1
    )

    # Safety check
    final_features = np.nan_to_num(
        final_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return final_features.astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("VoxaSign PSL V2 Feature Engineering")
    print("=" * 70)

    splits = [
        "train",
        "validation",
        "test",
    ]

    metadata = {}

    for split in splits:

        input_path = LANDMARK_DIR / f"X_{split}.npy"
        output_path = FEATURE_DIR / f"X_{split}_v2.npy"

        print()
        print(f"Processing {split.upper()}...")
        print("-" * 70)

        X = np.load(input_path)

        print("Original shape:", X.shape)

        X_v2 = extract_features(X)

        print("V2 shape:", X_v2.shape)

        np.save(output_path, X_v2)

        metadata[split] = {
            "original_shape": list(X.shape),
            "v2_shape": list(X_v2.shape),
        }

        print("Saved:", output_path)

    # Save feature description
    feature_info = {
        "original_features": 63,
        "bone_features": 20,
        "angle_features": 10,
        "wrist_fingertip_features": 5,
        "fingertip_pair_features": 10,
        "palm_features": 7,
        "total_features": 115,
        "splits": metadata,
    }

    info_path = FEATURE_DIR / "feature_info.json"

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            feature_info,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print()
    print("Feature breakdown:")
    print("Original coordinates       : 63")
    print("Bone lengths               : 20")
    print("Joint angles               : 10")
    print("Wrist → fingertip          : 5")
    print("Fingertip distances        : 10")
    print("Palm geometry              : 7")
    print("-" * 40)
    print("TOTAL V2 FEATURES         : 115")

    print()
    print("Saved to:")
    print(FEATURE_DIR)


if __name__ == "__main__":
    main()