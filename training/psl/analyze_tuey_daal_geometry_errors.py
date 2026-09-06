"""
VoxaSign PSL — Tuey vs Daal Geometry Error Analysis

This script DOES NOT retrain any model.

It compares:

    1. Existing V2 neural model predictions
    2. V2 landmark features + geometry features using 1-NN

It analyzes:

    - V2 correct / wrong
    - Geometry correct / wrong
    - V2 wrong -> Geometry correct
    - V2 correct -> Geometry wrong
    - Both wrong
    - Tuey -> Daal errors
    - Daal -> Tuey errors
    - Feature differences between classes
    - Individual test samples
    - Subject-level error concentration
    - Copies original images into visualization folders

IMPORTANT:
This is an analysis experiment only.
It does NOT modify the V2 model or V2 training data.

IMPORTANT:
Metadata is only attached when test metadata can be safely aligned
with X_test. We do NOT guess positional alignment when lengths differ.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "output" / "psl_static_model_v2.keras"

X_TRAIN_V2_PATH = (
    ROOT / "data" / "landmarks" / "v2" / "X_train_v2.npy"
)

X_VAL_V2_PATH = (
    ROOT / "data" / "landmarks" / "v2" / "X_validation_v2.npy"
)

X_TEST_V2_PATH = (
    ROOT / "data" / "landmarks" / "v2" / "X_test_v2.npy"
)

Y_TRAIN_PATH = (
    ROOT / "data" / "landmarks" / "y_train.npy"
)

Y_VAL_PATH = (
    ROOT / "data" / "landmarks" / "y_validation.npy"
)

Y_TEST_PATH = (
    ROOT / "data" / "landmarks" / "y_test.npy"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "psl_static_metadata.csv"
)

CLASS_MAP_PATH = (
    ROOT / "output" / "class_map.json"
)

GEOMETRY_INFO_PATH = (
    ROOT
    / "reports"
    / "tuey_daal_geometry"
    / "geometry_feature_info.json"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "tuey_daal_geometry_errors"
)

VIS_DIR = OUTPUT_DIR / "visualizations"


# ============================================================================
# TARGET CLASSES
# ============================================================================

TUEY_NAME = "Tuey"
DAAL_NAME = "Daal"

TARGET_NAMES = [
    DAAL_NAME,
    TUEY_NAME,
]


# ============================================================================
# UTILITIES
# ============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    print(f"OK: {path}")


def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def safe_copy(
    src: str | Path,
    destination: Path,
) -> bool:
    """
    Copy an image if it exists.
    """

    if src is None:
        return False

    src_string = str(src).strip()

    if not src_string:
        return False

    src = Path(src_string)

    if not src.exists():
        return False

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        shutil.copy2(
            src,
            destination / src.name,
        )

        return True

    except Exception:
        return False


# ============================================================================
# CLASS MAP
# ============================================================================


def load_class_map():

    banner("LOADING CLASS MAP")

    class_map = load_json(
        CLASS_MAP_PATH
    )

    id_to_name = {}

    for key, value in class_map.items():

        # Format:
        #
        # {
        #     "0": "Alif",
        #     "1": "Bay",
        # }

        if (
            isinstance(key, str)
            and key.isdigit()
        ):

            class_id = int(key)
            class_name = str(value)

            id_to_name[class_id] = (
                class_name
            )

        # Format:
        #
        # {
        #     "Alif": 0,
        #     "Bay": 1,
        # }

        elif isinstance(value, int):

            class_id = int(value)
            class_name = str(key)

            id_to_name[class_id] = (
                class_name
            )

    if not id_to_name:

        raise RuntimeError(
            "Could not understand class_map.json format."
        )

    name_to_id = {
        name: idx
        for idx, name in id_to_name.items()
    }

    if TUEY_NAME not in name_to_id:

        raise RuntimeError(
            "Tuey not found in class map."
        )

    if DAAL_NAME not in name_to_id:

        raise RuntimeError(
            "Daal not found in class map."
        )

    tuey_id = name_to_id[TUEY_NAME]
    daal_id = name_to_id[DAAL_NAME]

    print(
        f"Number of classes: "
        f"{len(id_to_name)}"
    )

    print(
        f"{TUEY_NAME} = {tuey_id}"
    )

    print(
        f"{DAAL_NAME} = {daal_id}"
    )

    return (
        id_to_name,
        name_to_id,
        tuey_id,
        daal_id,
    )


# ============================================================================
# GEOMETRY FEATURES
# ============================================================================


def normalize_landmarks(
    coords: np.ndarray,
) -> np.ndarray:

    """
    Normalize 21 hand landmarks.

    Input:
        (N, 63)

    Output:
        (N, 63)
    """

    coords = coords.reshape(
        -1,
        21,
        3,
    ).copy()

    wrist = coords[
        :,
        0:1,
        :,
    ]

    relative = coords - wrist

    scale = np.linalg.norm(
        relative[:, 9, :],
        axis=1,
        keepdims=True,
    )

    scale = np.maximum(
        scale,
        1e-6,
    )

    normalized = (
        relative
        / scale[:, None, :]
    )

    return normalized.reshape(
        -1,
        63,
    )


def distance(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    return np.linalg.norm(
        a - b,
        axis=1,
    )


def angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:

    ba = a - b
    bc = c - b

    numerator = np.sum(
        ba * bc,
        axis=1,
    )

    denominator = (
        np.linalg.norm(
            ba,
            axis=1,
        )
        *
        np.linalg.norm(
            bc,
            axis=1,
        )
    )

    denominator = np.maximum(
        denominator,
        1e-8,
    )

    cos_value = (
        numerator / denominator
    )

    cos_value = np.clip(
        cos_value,
        -1.0,
        1.0,
    )

    return np.arccos(
        cos_value
    )


def build_geometry_features(
    X: np.ndarray,
):

    """
    Build deterministic geometry features
    from the first 63 V2 features.

    Returns:
        geometry_features
        feature_names
    """

    base = X[:, :63]

    landmarks = base.reshape(
        -1,
        21,
        3,
    )

    features = []
    names = []

    # ------------------------------------------------------------------
    # Raw relative coordinates
    # ------------------------------------------------------------------

    relative = (
        landmarks
        - landmarks[:, 0:1, :]
    )

    for landmark_idx in range(21):

        for axis_idx, axis_name in enumerate(
            ["x", "y", "z"]
        ):

            features.append(
                relative[
                    :,
                    landmark_idx,
                    axis_idx,
                ]
            )

            names.append(
                f"relative_landmark_"
                f"{landmark_idx}_"
                f"{axis_name}"
            )

    # ------------------------------------------------------------------
    # Normalized coordinates
    # ------------------------------------------------------------------

    normalized = normalize_landmarks(
        base
    )

    normalized = normalized.reshape(
        -1,
        21,
        3,
    )

    for landmark_idx in range(21):

        for axis_idx, axis_name in enumerate(
            ["x", "y", "z"]
        ):

            features.append(
                normalized[
                    :,
                    landmark_idx,
                    axis_idx,
                ]
            )

            names.append(
                f"normalized_landmark_"
                f"{landmark_idx}_"
                f"{axis_name}"
            )

    # ------------------------------------------------------------------
    # Landmark distances
    # ------------------------------------------------------------------

    distance_pairs = [
        (0, 4),
        (0, 8),
        (0, 12),
        (0, 16),
        (0, 20),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (4, 12),
        (4, 16),
        (4, 20),
        (8, 16),
        (8, 20),
        (12, 20),
    ]

    for a, b in distance_pairs:

        features.append(
            distance(
                normalized[:, a, :],
                normalized[:, b, :],
            )
        )

        names.append(
            f"tip_distance_{a}_{b}"
        )

    # ------------------------------------------------------------------
    # Finger segment lengths
    # ------------------------------------------------------------------

    fingers = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    for finger_name, points in fingers.items():

        for segment_idx in range(
            len(points) - 1
        ):

            a = points[segment_idx]
            b = points[segment_idx + 1]

            features.append(
                distance(
                    normalized[:, a, :],
                    normalized[:, b, :],
                )
            )

            names.append(
                f"{finger_name}_segment_"
                f"{segment_idx + 1}"
            )

    # ------------------------------------------------------------------
    # Wrist to fingertip distances
    # ------------------------------------------------------------------

    fingertips = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    for finger_name, point_idx in (
        fingertips.items()
    ):

        features.append(
            distance(
                normalized[:, 0, :],
                normalized[:, point_idx, :],
            )
        )

        names.append(
            f"wrist_to_tip_{point_idx}"
        )

    # ------------------------------------------------------------------
    # Joint angles
    # ------------------------------------------------------------------

    angle_triplets = [
        (1, 2, 3),
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20),
    ]

    for angle_idx, (
        a,
        b,
        c,
    ) in enumerate(angle_triplets):

        features.append(
            angle(
                normalized[:, a, :],
                normalized[:, b, :],
                normalized[:, c, :],
            )
        )

        names.append(
            f"joint_angle_{angle_idx + 1}"
        )

    # ------------------------------------------------------------------
    # Thumb segment angles
    # ------------------------------------------------------------------

    thumb_angles = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
    ]

    for idx, (
        a,
        b,
        c,
    ) in enumerate(thumb_angles):

        features.append(
            angle(
                normalized[:, a, :],
                normalized[:, b, :],
                normalized[:, c, :],
            )
        )

        names.append(
            f"thumb_joint_angle_{idx + 1}"
        )

    geometry = np.column_stack(
        features
    )

    return (
        geometry,
        names,
    )


# ============================================================================
# LOAD DATA
# ============================================================================


def load_data():

    banner("LOADING LANDMARK DATA")

    X_train = np.load(
        X_TRAIN_V2_PATH
    )

    X_val = np.load(
        X_VAL_V2_PATH
    )

    X_test = np.load(
        X_TEST_V2_PATH
    )

    y_train = np.load(
        Y_TRAIN_PATH
    )

    y_val = np.load(
        Y_VAL_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    print(
        f"Train:      "
        f"X={X_train.shape} "
        f"y={y_train.shape}"
    )

    print(
        f"Validation: "
        f"X={X_val.shape} "
        f"y={y_val.shape}"
    )

    print(
        f"Test:       "
        f"X={X_test.shape} "
        f"y={y_test.shape}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================================
# LOAD METADATA
# ============================================================================


def load_metadata():

    banner("LOADING TEST METADATA")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows: "
        f"{len(metadata)}"
    )

    print(
        f"Metadata columns: "
        f"{list(metadata.columns)}"
    )

    failed_path = (
        ROOT
        / "data"
        / "landmarks"
        / "failed_images.json"
    )

    if failed_path.exists():

        try:

            failed_data = load_json(
                failed_path
            )

            failed_paths = set()

            if isinstance(
                failed_data,
                list,
            ):

                for item in failed_data:

                    if isinstance(
                        item,
                        str,
                    ):

                        failed_paths.add(
                            item
                        )

                    elif isinstance(
                        item,
                        dict,
                    ):

                        for key in [
                            "path",
                            "filename",
                            "image",
                        ]:

                            if key in item:

                                failed_paths.add(
                                    str(
                                        item[key]
                                    )
                                )

            elif isinstance(
                failed_data,
                dict,
            ):

                for key in [
                    "failed_images",
                    "images",
                    "paths",
                ]:

                    value = failed_data.get(
                        key
                    )

                    if isinstance(
                        value,
                        list,
                    ):

                        failed_paths.update(
                            str(x)
                            for x in value
                        )

            if failed_paths:

                before = len(metadata)

                metadata = metadata[
                    ~metadata[
                        "path"
                    ]
                    .astype(str)
                    .isin(failed_paths)
                ].reset_index(
                    drop=True
                )

                print(
                    f"Removed "
                    f"{before - len(metadata)} "
                    f"failed metadata rows."
                )

        except Exception as exc:

            print(
                "WARNING: could not process "
                f"failed_images.json: {exc}"
            )

    return metadata


# ============================================================================
# BUILD TEST METADATA ALIGNMENT
# ============================================================================


def build_test_records(
    metadata: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
):

    """
    Safely align test metadata with X_test.

    Priority:

    1. split == test/testing AND exact length match
    2. entire metadata exact length match

    Otherwise:
        return None

    We intentionally DO NOT guess alignment.
    """

    if metadata is None:
        return None

    test_metadata = metadata.copy()

    # ---------------------------------------------------------------
    # Try split-based alignment
    # ---------------------------------------------------------------

    if "split" in test_metadata.columns:

        split_values = (
            test_metadata[
                "split"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        candidates = test_metadata[
            split_values.isin(
                [
                    "test",
                    "testing",
                ]
            )
        ].copy()

        print()
        print(
            f"Test metadata candidates: "
            f"{len(candidates)}"
        )

        if len(candidates) == len(
            X_test
        ):

            test_metadata = (
                candidates
                .reset_index(
                    drop=True
                )
            )

            print(
                "Test metadata aligned "
                "using split='test'."
            )

        elif len(test_metadata) == len(
            X_test
        ):

            test_metadata = (
                test_metadata
                .reset_index(
                    drop=True
                )
            )

            print(
                "Metadata aligned "
                "positionally."
            )

        else:

            print()
            print(
                "WARNING: test metadata "
                "cannot be safely aligned."
            )

            print(
                f"X_test rows: "
                f"{len(X_test)}"
            )

            print(
                f"Metadata rows: "
                f"{len(test_metadata)}"
            )

            print(
                f"Test split rows: "
                f"{len(candidates)}"
            )

            print(
                "Image paths will NOT be attached "
                "to predictions."
            )

            return None

    elif len(test_metadata) == len(
        X_test
    ):

        test_metadata = (
            test_metadata
            .reset_index(
                drop=True
            )
        )

        print(
            "Test metadata aligned "
            "positionally."
        )

    else:

        print()
        print(
            "WARNING: metadata/test "
            "alignment unavailable."
        )

        return None

    records = test_metadata.copy()

    records["array_index"] = np.arange(
        len(records)
    )

    records["true_id"] = y_test

    return records


# ============================================================================
# V2 MODEL PREDICTIONS
# ============================================================================


def generate_v2_predictions(
    X_test,
    y_test,
):

    banner("LOADING V2 MODEL")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(
        "Model loaded successfully."
    )

    banner(
        "GENERATING V2 PREDICTIONS"
    )

    probabilities = model.predict(
        X_test,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    # IMPORTANT:
    # Calculate outside the f-string.
    # This avoids the original SyntaxError.

    overall_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"Overall V2 test accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    return (
        predictions,
        probabilities,
        confidences,
    )


# ============================================================================
# GEOMETRY 1-NN
# ============================================================================


def generate_geometry_predictions(
    X_train,
    y_train,
    X_test,
):

    banner(
        "BUILDING GEOMETRY FEATURES"
    )

    geometry_train, geometry_names = (
        build_geometry_features(
            X_train
        )
    )

    geometry_test, _ = (
        build_geometry_features(
            X_test
        )
    )

    print(
        f"Geometry train shape: "
        f"{geometry_train.shape}"
    )

    print(
        f"Geometry test shape: "
        f"{geometry_test.shape}"
    )

    # ---------------------------------------------------------------
    # V2 + Geometry
    # ---------------------------------------------------------------

    combined_train = np.concatenate(
        [
            X_train,
            geometry_train,
        ],
        axis=1,
    )

    combined_test = np.concatenate(
        [
            X_test,
            geometry_test,
        ],
        axis=1,
    )

    print(
        f"V2 + Geometry train shape: "
        f"{combined_train.shape}"
    )

    print(
        f"V2 + Geometry test shape: "
        f"{combined_test.shape}"
    )

    # ---------------------------------------------------------------
    # Scale
    # ---------------------------------------------------------------

    scaler = StandardScaler()

    combined_train = (
        scaler.fit_transform(
            combined_train
        )
    )

    combined_test = (
        scaler.transform(
            combined_test
        )
    )

    # ---------------------------------------------------------------
    # 1-NN
    # ---------------------------------------------------------------

    banner(
        "GEOMETRY 1-NN PREDICTIONS"
    )

    model = KNeighborsClassifier(
        n_neighbors=1,
        metric="euclidean",
    )

    model.fit(
        combined_train,
        y_train,
    )

    predictions = model.predict(
        combined_test
    )

    distances, _ = model.kneighbors(
        combined_test,
        n_neighbors=1,
    )

    nearest_distance = distances[
        :,
        0,
    ]

    return (
        predictions,
        nearest_distance,
        geometry_names,
    )


# ============================================================================
# TARGET FILTER
# ============================================================================


def filter_target(
    y_true,
    tuey_id,
    daal_id,
):

    return np.isin(
        y_true,
        [
            tuey_id,
            daal_id,
        ],
    )


# ============================================================================
# FEATURE COMPARISON
# ============================================================================


def analyze_feature_differences(
    X_test,
    y_test,
    geometry_names,
    tuey_id,
    daal_id,
):

    banner(
        "FEATURE DIFFERENCE ANALYSIS"
    )

    geometry_test, _ = (
        build_geometry_features(
            X_test
        )
    )

    mask = np.isin(
        y_test,
        [
            tuey_id,
            daal_id,
        ],
    )

    X_target = X_test[mask]
    G_target = geometry_test[mask]
    y_target = y_test[mask]

    rows = []

    # ---------------------------------------------------------------
    # V2 features
    # ---------------------------------------------------------------

    for feature_idx in range(
        X_target.shape[1]
    ):

        tuey_values = X_target[
            y_target == tuey_id,
            feature_idx,
        ]

        daal_values = X_target[
            y_target == daal_id,
            feature_idx,
        ]

        if len(tuey_values) == 0:
            continue

        if len(daal_values) == 0:
            continue

        tuey_mean = np.mean(
            tuey_values
        )

        daal_mean = np.mean(
            daal_values
        )

        tuey_std = np.std(
            tuey_values
        )

        daal_std = np.std(
            daal_values
        )

        pooled_std = np.sqrt(
            (
                tuey_std ** 2
                + daal_std ** 2
            )
            / 2
        )

        effect = abs(
            tuey_mean - daal_mean
        ) / max(
            pooled_std,
            1e-8,
        )

        rows.append(
            {
                "feature_set": "V2",
                "feature_index": feature_idx,
                "feature": (
                    f"V2_Feature_{feature_idx}"
                ),
                "tuey_mean": tuey_mean,
                "daal_mean": daal_mean,
                "absolute_difference": abs(
                    tuey_mean
                    - daal_mean
                ),
                "effect_size": effect,
            }
        )

    # ---------------------------------------------------------------
    # Geometry features
    # ---------------------------------------------------------------

    for feature_idx in range(
        G_target.shape[1]
    ):

        tuey_values = G_target[
            y_target == tuey_id,
            feature_idx,
        ]

        daal_values = G_target[
            y_target == daal_id,
            feature_idx,
        ]

        if len(tuey_values) == 0:
            continue

        if len(daal_values) == 0:
            continue

        tuey_mean = np.mean(
            tuey_values
        )

        daal_mean = np.mean(
            daal_values
        )

        tuey_std = np.std(
            tuey_values
        )

        daal_std = np.std(
            daal_values
        )

        pooled_std = np.sqrt(
            (
                tuey_std ** 2
                + daal_std ** 2
            )
            / 2
        )

        effect = abs(
            tuey_mean - daal_mean
        ) / max(
            pooled_std,
            1e-8,
        )

        rows.append(
            {
                "feature_set": "Geometry",
                "feature_index": feature_idx,
                "feature": geometry_names[
                    feature_idx
                ],
                "tuey_mean": tuey_mean,
                "daal_mean": daal_mean,
                "absolute_difference": abs(
                    tuey_mean
                    - daal_mean
                ),
                "effect_size": effect,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = (
            result
            .sort_values(
                "effect_size",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return result


# ============================================================================
# SAMPLE CLASSIFICATION
# ============================================================================


def build_sample_comparison(
    y_test,
    v2_predictions,
    geometry_predictions,
    v2_confidences,
    geometry_distances,
    metadata,
    tuey_id,
    daal_id,
    id_to_name,
):

    banner(
        "BUILDING SAMPLE COMPARISON"
    )

    rows = []

    for i in range(
        len(y_test)
    ):

        true_id = int(
            y_test[i]
        )

        if true_id not in [
            tuey_id,
            daal_id,
        ]:
            continue

        v2_id = int(
            v2_predictions[i]
        )

        geometry_id = int(
            geometry_predictions[i]
        )

        v2_correct = (
            v2_id == true_id
        )

        geometry_correct = (
            geometry_id == true_id
        )

        if (
            v2_correct
            and geometry_correct
        ):

            category = (
                "V2 correct / Geometry correct"
            )

        elif (
            v2_correct
            and not geometry_correct
        ):

            category = (
                "V2 correct / Geometry wrong"
            )

        elif (
            not v2_correct
            and geometry_correct
        ):

            category = (
                "V2 wrong / Geometry correct"
            )

        else:

            category = "Both wrong"

        row = {
            "test_index": i,

            "true_class": id_to_name.get(
                true_id,
                str(true_id),
            ),

            "v2_prediction": id_to_name.get(
                v2_id,
                str(v2_id),
            ),

            "geometry_prediction": (
                id_to_name.get(
                    geometry_id,
                    str(geometry_id),
                )
            ),

            "v2_correct": v2_correct,

            "geometry_correct": (
                geometry_correct
            ),

            "category": category,

            "v2_confidence": float(
                v2_confidences[i]
            ),

            "geometry_nearest_distance": float(
                geometry_distances[i]
            ),
        }

        # -----------------------------------------------------------
        # Metadata
        # -----------------------------------------------------------

        if metadata is not None:

            if i < len(metadata):

                meta_row = metadata.iloc[
                    i
                ]

                for column in [
                    "path",
                    "filename",
                    "subject_id",
                    "source",
                    "class",
                    "split",
                ]:

                    if column in (
                        metadata.columns
                    ):

                        value = meta_row[
                            column
                        ]

                        if pd.notna(
                            value
                        ):

                            row[column] = (
                                value
                            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================


def print_category_analysis(
    df,
):

    banner(
        "ERROR CATEGORY ANALYSIS"
    )

    if len(df) == 0:

        print(
            "No Tuey/Daal samples found."
        )

        return {}

    counts = (
        df["category"]
        .value_counts()
        .to_dict()
    )

    print()

    categories = [
        "V2 correct / Geometry correct",
        "V2 correct / Geometry wrong",
        "V2 wrong / Geometry correct",
        "Both wrong",
    ]

    for category in categories:

        count = counts.get(
            category,
            0,
        )

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"{category:<38} "
            f"{count:>3} "
            f"({percentage:6.2f}%)"
        )

    return counts


# ============================================================================
# CONFUSION ANALYSIS
# ============================================================================


def print_confusion_analysis(
    df,
):

    banner(
        "TUEY / DAAL ERROR ANALYSIS"
    )

    if len(df) == 0:

        print(
            "No target samples."
        )

        return

    print()
    print("V2:")

    v2_pairs = (
        df.groupby(
            [
                "true_class",
                "v2_prediction",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        v2_pairs.to_string()
    )

    print()
    print(
        "V2 + Geometry 1-NN:"
    )

    geometry_pairs = (
        df.groupby(
            [
                "true_class",
                "geometry_prediction",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        geometry_pairs.to_string()
    )


# ============================================================================
# SUBJECT ANALYSIS
# ============================================================================


def analyze_subjects(
    df,
):

    banner(
        "SUBJECT-LEVEL ERROR ANALYSIS"
    )

    if "subject_id" not in (
        df.columns
    ):

        print(
            "Subject ID unavailable."
        )

        return pd.DataFrame()

    working = df.copy()

    working["subject_id"] = (
        working["subject_id"]
        .astype(str)
    )

    result_rows = []

    for subject_id, group in (
        working.groupby(
            "subject_id"
        )
    ):

        v2_errors = int(
            (~group["v2_correct"])
            .sum()
        )

        geometry_errors = int(
            (~group["geometry_correct"])
            .sum()
        )

        both_wrong = int(
            (
                group["category"]
                == "Both wrong"
            ).sum()
        )

        geometry_fixes = int(
            (
                group["category"]
                == "V2 wrong / Geometry correct"
            ).sum()
        )

        geometry_breaks = int(
            (
                group["category"]
                == "V2 correct / Geometry wrong"
            ).sum()
        )

        result_rows.append(
            {
                "subject_id": subject_id,
                "samples": len(group),
                "v2_errors": v2_errors,
                "geometry_errors": (
                    geometry_errors
                ),
                "geometry_fixes": (
                    geometry_fixes
                ),
                "geometry_breaks": (
                    geometry_breaks
                ),
                "both_wrong": both_wrong,
            }
        )

    result = pd.DataFrame(
        result_rows
    )

    if len(result):

        result = (
            result
            .sort_values(
                [
                    "both_wrong",
                    "geometry_errors",
                    "v2_errors",
                ],
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print(
            result.to_string(
                index=False
            )
        )

    return result


# ============================================================================
# VISUALIZATION COPYING
# ============================================================================


def copy_visualizations(
    df,
):

    banner(
        "COPYING ORIGINAL IMAGES"
    )

    folders = [
        "v2_correct_geometry_correct",
        "v2_correct_geometry_wrong",
        "v2_wrong_geometry_correct",
        "both_wrong",
        "tuey_as_daal",
        "daal_as_tuey",
        "all_target_samples",
    ]

    for folder in folders:

        (
            VIS_DIR / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    counters = {
        folder: 0
        for folder in folders
    }

    if "path" not in df.columns:

        print(
            "WARNING: image paths are "
            "unavailable because metadata "
            "was not safely aligned."
        )

        return counters

    for _, row in df.iterrows():

        path = row.get(
            "path"
        )

        if path is None:
            continue

        if pd.isna(path):
            continue

        path = Path(
            str(path)
        )

        if not path.exists():

            print(
                f"WARNING: image not found: "
                f"{path}"
            )

            continue

        category = row[
            "category"
        ]

        if category == (
            "V2 correct / Geometry correct"
        ):

            folder = (
                "v2_correct_geometry_correct"
            )

        elif category == (
            "V2 correct / Geometry wrong"
        ):

            folder = (
                "v2_correct_geometry_wrong"
            )

        elif category == (
            "V2 wrong / Geometry correct"
        ):

            folder = (
                "v2_wrong_geometry_correct"
            )

        else:

            folder = "both_wrong"

        if safe_copy(
            path,
            VIS_DIR / folder,
        ):

            counters[folder] += 1

        # -----------------------------------------------------------
        # Tuey -> Daal
        # -----------------------------------------------------------

        if (
            row["true_class"]
            == TUEY_NAME
            and row["v2_prediction"]
            == DAAL_NAME
        ):

            if safe_copy(
                path,
                VIS_DIR / "tuey_as_daal",
            ):

                counters[
                    "tuey_as_daal"
                ] += 1

        # -----------------------------------------------------------
        # Daal -> Tuey
        # -----------------------------------------------------------

        if (
            row["true_class"]
            == DAAL_NAME
            and row["v2_prediction"]
            == TUEY_NAME
        ):

            if safe_copy(
                path,
                VIS_DIR / "daal_as_tuey",
            ):

                counters[
                    "daal_as_tuey"
                ] += 1

        # -----------------------------------------------------------
        # All target samples
        # -----------------------------------------------------------

        if safe_copy(
            path,
            VIS_DIR / "all_target_samples",
        ):

            counters[
                "all_target_samples"
            ] += 1

    print()

    for folder in folders:

        print(
            f"{folder:<38}: "
            f"{counters[folder]}"
        )

    return counters


# ============================================================================
# METRICS
# ============================================================================


def calculate_metrics(
    y_true,
    predictions,
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ============================================================================
# MAIN
# ============================================================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner(
        "VoxaSign PSL — "
        "Tuey vs Daal Geometry Error Analysis"
    )

    print()

    print(
        "This script DOES NOT retrain any model."
    )

    print(
        "It compares the existing V2 model "
        "with V2 + Geometry 1-NN."
    )

    # ==================================================================
    # FILES
    # ==================================================================

    banner(
        "CHECKING REQUIRED FILES"
    )

    required_files = [
        MODEL_PATH,
        X_TRAIN_V2_PATH,
        X_VAL_V2_PATH,
        X_TEST_V2_PATH,
        Y_TRAIN_PATH,
        Y_VAL_PATH,
        Y_TEST_PATH,
        METADATA_PATH,
        CLASS_MAP_PATH,
    ]

    for path in required_files:
        check_file(path)

    # ==================================================================
    # CLASS MAP
    # ==================================================================

    (
        id_to_name,
        name_to_id,
        tuey_id,
        daal_id,
    ) = load_class_map()

    # ==================================================================
    # DATA
    # ==================================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_data()

    # ==================================================================
    # METADATA
    # ==================================================================

    metadata = load_metadata()

    test_metadata = build_test_records(
        metadata,
        X_test,
        y_test,
    )

    # ==================================================================
    # V2 PREDICTIONS
    # ==================================================================

    (
        v2_predictions,
        v2_probabilities,
        v2_confidences,
    ) = generate_v2_predictions(
        X_test,
        y_test,
    )

    # ==================================================================
    # GEOMETRY PREDICTIONS
    # ==================================================================

    (
        geometry_predictions,
        geometry_distances,
        geometry_names,
    ) = generate_geometry_predictions(
        X_train,
        y_train,
        X_test,
    )

    # ==================================================================
    # TARGET FILTER
    # ==================================================================

    mask = filter_target(
        y_test,
        tuey_id,
        daal_id,
    )

    target_indices = np.where(
        mask
    )[0]

    y_target = y_test[
        mask
    ]

    v2_target = (
        v2_predictions[mask]
    )

    geometry_target = (
        geometry_predictions[mask]
    )

    v2_conf_target = (
        v2_confidences[mask]
    )

    geometry_distance_target = (
        geometry_distances[mask]
    )

    print()
    print(
        f"Tuey test samples: "
        f"{int((y_target == tuey_id).sum())}"
    )

    print(
        f"Daal test samples: "
        f"{int((y_target == daal_id).sum())}"
    )

    # ==================================================================
    # METRICS
    # ==================================================================

    banner(
        "TUEY / DAAL METRICS"
    )

    # Daal = 0
    # Tuey = 1

    y_binary = (
        y_target == tuey_id
    ).astype(int)

    v2_binary = (
        v2_target == tuey_id
    ).astype(int)

    geometry_binary = (
        geometry_target == tuey_id
    ).astype(int)

    v2_metrics = calculate_metrics(
        y_binary,
        v2_binary,
    )

    geometry_metrics = calculate_metrics(
        y_binary,
        geometry_binary,
    )

    print()
    print("V2:")

    print(
        f"Accuracy : "
        f"{v2_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{v2_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{v2_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{v2_metrics['f1'] * 100:.2f}%"
    )

    print()
    print(
        "V2 + Geometry 1-NN:"
    )

    print(
        f"Accuracy : "
        f"{geometry_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{geometry_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{geometry_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1       : "
        f"{geometry_metrics['f1'] * 100:.2f}%"
    )

    # ==================================================================
    # SAMPLE COMPARISON
    # ==================================================================

    comparison = build_sample_comparison(
        y_test,
        v2_predictions,
        geometry_predictions,
        v2_confidences,
        geometry_distances,
        test_metadata,
        tuey_id,
        daal_id,
        id_to_name,
    )

    # ==================================================================
    # CATEGORIES
    # ==================================================================

    category_counts = (
        print_category_analysis(
            comparison
        )
    )

    # ==================================================================
    # CONFUSION
    # ==================================================================

    print_confusion_analysis(
        comparison
    )

    # ==================================================================
    # FEATURE DIFFERENCES
    # ==================================================================

    feature_df = (
        analyze_feature_differences(
            X_test,
            y_test,
            geometry_names,
            tuey_id,
            daal_id,
        )
    )

    feature_path = (
        OUTPUT_DIR
        / "feature_comparison.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False,
    )

    print()
    print(
        f"Saved: {feature_path}"
    )

    banner(
        "TOP FEATURES FOR TUEY / DAAL"
    )

    if len(feature_df):

        print(
            feature_df.head(
                30
            ).to_string(
                index=False
            )
        )

    # ==================================================================
    # SUBJECT ANALYSIS
    # ==================================================================

    subject_df = analyze_subjects(
        comparison
    )

    subject_path = (
        OUTPUT_DIR
        / "subject_error_analysis.csv"
    )

    if len(subject_df):

        subject_df.to_csv(
            subject_path,
            index=False,
        )

        print(
            f"Saved: {subject_path}"
        )

    # ==================================================================
    # VISUALIZATIONS
    # ==================================================================

    copy_counts = (
        copy_visualizations(
            comparison
        )
    )

    # ==================================================================
    # DIRECT ERROR COUNTS
    # ==================================================================

    tuey_to_daal_v2 = int(
        (
            (y_test == tuey_id)
            &
            (
                v2_predictions
                == daal_id
            )
        ).sum()
    )

    daal_to_tuey_v2 = int(
        (
            (y_test == daal_id)
            &
            (
                v2_predictions
                == tuey_id
            )
        ).sum()
    )

    tuey_to_daal_geometry = int(
        (
            (y_test == tuey_id)
            &
            (
                geometry_predictions
                == daal_id
            )
        ).sum()
    )

    daal_to_tuey_geometry = int(
        (
            (y_test == daal_id)
            &
            (
                geometry_predictions
                == tuey_id
            )
        ).sum()
    )

    # ==================================================================
    # GEOMETRY FIXES / BREAKS
    # ==================================================================

    if len(comparison):

        geometry_fixes = int(
            (
                comparison[
                    "category"
                ]
                ==
                "V2 wrong / Geometry correct"
            ).sum()
        )

        geometry_breaks = int(
            (
                comparison[
                    "category"
                ]
                ==
                "V2 correct / Geometry wrong"
            ).sum()
        )

        both_wrong = int(
            (
                comparison[
                    "category"
                ]
                == "Both wrong"
            ).sum()
        )

        both_correct = int(
            (
                comparison[
                    "category"
                ]
                ==
                "V2 correct / Geometry correct"
            ).sum()
        )

    else:

        geometry_fixes = 0
        geometry_breaks = 0
        both_wrong = 0
        both_correct = 0

    # ==================================================================
    # INDIVIDUAL SAMPLES
    # ==================================================================

    banner(
        "INDIVIDUAL TUEY / DAAL SAMPLES"
    )

    display_columns = [
        "test_index",
        "true_class",
        "v2_prediction",
        "geometry_prediction",
        "category",
        "v2_confidence",
        "geometry_nearest_distance",
        "subject_id",
        "filename",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in comparison.columns
    ]

    if len(comparison):

        print(
            comparison[
                available_columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No Tuey/Daal samples."
        )

    # ==================================================================
    # SAVE SAMPLE CSV
    # ==================================================================

    comparison_path = (
        OUTPUT_DIR
        / "sample_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print(
        f"Saved: {comparison_path}"
    )

    # ==================================================================
    # CLASSIFICATION REPORTS
    # ==================================================================

    banner(
        "V2 CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_binary,
            v2_binary,
            target_names=[
                DAAL_NAME,
                TUEY_NAME,
            ],
            zero_division=0,
        )
    )

    banner(
        "V2 + GEOMETRY CLASSIFICATION REPORT"
    )

    geometry_report = (
        classification_report(
            y_binary,
            geometry_binary,
            target_names=[
                DAAL_NAME,
                TUEY_NAME,
            ],
            zero_division=0,
        )
    )

    print(
        geometry_report
    )

    # ==================================================================
    # FINAL INTERPRETATION
    # ==================================================================

    banner(
        "INTERPRETATION"
    )

    print()

    print(
        f"V2 F1: "
        f"{v2_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"V2 + Geometry F1: "
        f"{geometry_metrics['f1'] * 100:.2f}%"
    )

    improvement = (
        geometry_metrics["f1"]
        - v2_metrics["f1"]
    )

    print(
        f"F1 difference: "
        f"{improvement * 100:+.2f} "
        f"percentage points"
    )

    print()

    print(
        f"V2 Tuey -> Daal: "
        f"{tuey_to_daal_v2}"
    )

    print(
        f"V2 Daal -> Tuey: "
        f"{daal_to_tuey_v2}"
    )

    print()

    print(
        f"Geometry Tuey -> Daal: "
        f"{tuey_to_daal_geometry}"
    )

    print(
        f"Geometry Daal -> Tuey: "
        f"{daal_to_tuey_geometry}"
    )

    print()

    print(
        f"Geometry fixes V2 errors: "
        f"{geometry_fixes}"
    )

    print(
        f"Geometry introduces errors: "
        f"{geometry_breaks}"
    )

    print(
        f"Both models correct: "
        f"{both_correct}"
    )

    print(
        f"Both models wrong: "
        f"{both_wrong}"
    )

    print()

    if geometry_fixes > geometry_breaks:

        print("POSITIVE:")

        print(
            "Geometry fixes more V2 errors "
            "than it introduces."
        )

    elif geometry_fixes == geometry_breaks:

        print("NEUTRAL:")

        print(
            "Geometry fixes and introduces "
            "the same number of errors."
        )

    else:

        print("WARNING:")

        print(
            "Geometry introduces more errors "
            "than it fixes."
        )

    print()

    print("IMPORTANT:")

    print(
        "The current test set contains only "
        f"{len(comparison)} Tuey/Daal samples."
    )

    print(
        "Therefore these results should be "
        "treated as experimental evidence, "
        "not final V3 validation."
    )

    print()

    if test_metadata is None:

        print(
            "Metadata could NOT be safely aligned "
            "with X_test."
        )

        print(
            "Therefore image copying and "
            "subject-level analysis are unavailable."
        )

    else:

        print(
            "Test metadata was safely aligned "
            "with X_test."
        )

    print()

    print(
        "DO NOT RETRAIN THE FULL V3 MODEL YET."
    )

    # ==================================================================
    # JSON REPORT
    # ==================================================================

    report = {
        "experiment": (
            "Tuey vs Daal "
            "Geometry Error Analysis"
        ),

        "retrained": False,

        "test_target_samples": int(
            len(comparison)
        ),

        "classes": {
            "Tuey": int(tuey_id),
            "Daal": int(daal_id),
        },

        "metrics": {
            "v2": v2_metrics,

            "v2_plus_geometry_1nn": (
                geometry_metrics
            ),

            "f1_difference": float(
                improvement
            ),
        },

        "confusion": {
            "v2": {
                "tuey_to_daal": (
                    tuey_to_daal_v2
                ),
                "daal_to_tuey": (
                    daal_to_tuey_v2
                ),
            },

            "geometry": {
                "tuey_to_daal": (
                    tuey_to_daal_geometry
                ),
                "daal_to_tuey": (
                    daal_to_tuey_geometry
                ),
            },
        },

        "error_categories": {
            "both_correct": both_correct,

            "v2_wrong_geometry_correct": (
                geometry_fixes
            ),

            "v2_correct_geometry_wrong": (
                geometry_breaks
            ),

            "both_wrong": both_wrong,
        },

        "visualization_counts": (
            copy_counts
        ),

        "metadata_alignment": (
            test_metadata is not None
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "geometry_error_analysis.json"
    )

    save_json(
        json_path,
        report,
    )

    print()
    print(
        f"Saved: {json_path}"
    )

    # ==================================================================
    # TEXT REPORT
    # ==================================================================

    text_path = (
        OUTPUT_DIR
        / "geometry_error_analysis.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "VoxaSign PSL — "
            "Tuey vs Daal Geometry "
            "Error Analysis\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "V2 metrics\n"
        )

        f.write(
            json.dumps(
                v2_metrics,
                indent=4,
            )
        )

        f.write(
            "\n\n"
            "V2 + Geometry 1-NN metrics\n"
        )

        f.write(
            json.dumps(
                geometry_metrics,
                indent=4,
            )
        )

        f.write(
            "\n\nF1 difference\n"
        )

        f.write(
            f"{improvement * 100:+.2f} "
            "percentage points\n"
        )

        f.write(
            "\n\nConfusion\n"
        )

        f.write(
            json.dumps(
                report["confusion"],
                indent=4,
            )
        )

        f.write(
            "\n\nError categories\n"
        )

        f.write(
            json.dumps(
                report[
                    "error_categories"
                ],
                indent=4,
            )
        )

        f.write(
            "\n\nMetadata alignment\n"
        )

        f.write(
            str(
                test_metadata is not None
            )
        )

        f.write(
            "\n\nIMPORTANT\n"
        )

        f.write(
            "Metadata/landmark alignment is "
            "not guessed.\n"
        )

        f.write(
            "The experiment uses a small "
            "Tuey/Daal test set.\n"
        )

        f.write(
            "Do not treat this as final "
            "V3 validation.\n"
        )

    print(
        f"Saved: {text_path}"
    )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    banner(
        "ANALYSIS COMPLETE"
    )

    print(
        f"Tuey/Daal samples analyzed: "
        f"{len(comparison)}"
    )

    print(
        f"V2 F1: "
        f"{v2_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"V2 + Geometry F1: "
        f"{geometry_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Geometry fixes: "
        f"{geometry_fixes}"
    )

    print(
        f"Geometry breaks: "
        f"{geometry_breaks}"
    )

    print()
    print("Reports:")

    print(
        f"  {text_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {comparison_path}"
    )

    print(
        f"  {feature_path}"
    )

    if len(subject_df):

        print(
            f"  {subject_path}"
        )

    print()
    print(
        "Visualizations:"
    )

    for folder in [
        "v2_correct_geometry_correct",
        "v2_correct_geometry_wrong",
        "v2_wrong_geometry_correct",
        "both_wrong",
        "tuey_as_daal",
        "daal_as_tuey",
        "all_target_samples",
    ]:

        print(
            f"  {VIS_DIR / folder}"
        )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Send me the COMPLETE terminal output."
    )

    print(
        "Especially the ERROR CATEGORY ANALYSIS "
        "and INDIVIDUAL TUEY / DAAL SAMPLES."
    )

    print(
        "\nDO NOT RETRAIN V3 YET."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()