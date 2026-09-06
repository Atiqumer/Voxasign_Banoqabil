"""
VoxaSign PSL — Tuey vs Daal Geometric Feature Experiment

PURPOSE
-------
This script does NOT retrain the 36-class VoxaSign model.

It compares three feature representations for the difficult Tuey/Daal pair:

1. V2 features
2. Geometry-only features
3. V2 + Geometry features

The geometry features are derived from the first 63 V2 features,
which correspond to the original 21 MediaPipe landmarks × (x,y,z).

The experiment evaluates:
- Logistic Regression
- SVM RBF
- Random Forest
- 1-NN

It also performs subject-aware GroupKFold validation when metadata
is correctly aligned.

Outputs:
    reports/tuey_daal_geometry/
        tuey_daal_geometry.txt
        tuey_daal_geometry.json
        feature_importance.csv
        test_predictions.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

LANDMARK_DIR = ROOT / "data" / "landmarks"
V2_DIR = LANDMARK_DIR / "v2"
METADATA_PATH = ROOT / "data" / "metadata" / "psl_static_metadata.csv"
CLASS_MAP_PATH = ROOT / "output" / "class_map.json"

REPORT_DIR = ROOT / "reports" / "tuey_daal_geometry"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET CLASSES
# ============================================================

TUEY_NAME = "Tuey"
DAAL_NAME = "Daal"

TUEY_ID = 29
DAAL_ID = 7


# ============================================================
# LANDMARK DEFINITIONS
# MediaPipe hand landmarks:
#
# 0  Wrist
# 1  Thumb CMC
# 2  Thumb MCP
# 3  Thumb IP
# 4  Thumb Tip
#
# 5  Index MCP
# 6  Index PIP
# 7  Index DIP
# 8  Index Tip
#
# 9  Middle MCP
# 10 Middle PIP
# 11 Middle DIP
# 12 Middle Tip
#
# 13 Ring MCP
# 14 Ring PIP
# 15 Ring DIP
# 16 Ring Tip
#
# 17 Pinky MCP
# 18 Pinky PIP
# 19 Pinky DIP
# 20 Pinky Tip
# ============================================================

WRIST = 0

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGERTIPS = [4, 8, 12, 16, 20]

MCP_POINTS = [1, 5, 9, 13, 17]


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def landmark_array(x_row: np.ndarray) -> np.ndarray:
    """
    Convert first 63 V2 features into:
        shape = (21, 3)

    Assumption:
        feature layout = x,y,z repeated for each landmark.
    """
    raw = np.asarray(x_row[:63], dtype=np.float64)

    if raw.shape[0] != 63:
        raise ValueError(
            f"Expected at least 63 raw landmark features, got {raw.shape[0]}"
        )

    return raw.reshape(21, 3)


def safe_norm(v):
    return float(np.linalg.norm(v))


def safe_angle(a, b):
    """
    Angle between vectors a and b in degrees.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    cosine = np.dot(a, b) / (na * nb)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


# ============================================================
# GEOMETRIC FEATURE EXTRACTION
# ============================================================

def extract_geometry_features(x_row: np.ndarray):
    """
    Build explicit geometric features from the 21 hand landmarks.

    Features include:

    A. Wrist-relative coordinates
    B. Scale-normalized coordinates
    C. Wrist-to-fingertip distances
    D. Fingertip pairwise distances
    E. Finger segment lengths
    F. Finger joint angles
    G. MCP-to-tip distances
    H. Angles between major finger directions
    """

    pts = landmark_array(x_row)

    wrist = pts[WRIST]

    features = []
    names = []

    # --------------------------------------------------------
    # A. Wrist-relative coordinates
    # --------------------------------------------------------

    relative = pts - wrist

    for i in range(21):
        for axis, axis_name in enumerate(["x", "y", "z"]):
            features.append(relative[i, axis])
            names.append(
                f"relative_landmark_{i}_{axis_name}"
            )

    # --------------------------------------------------------
    # B. Scale normalization
    #
    # Use wrist -> middle MCP as a stable hand-size reference.
    # --------------------------------------------------------

    scale = np.linalg.norm(pts[9] - wrist)

    if scale < 1e-8:
        scale = 1.0

    normalized = relative / scale

    for i in range(21):
        for axis, axis_name in enumerate(["x", "y", "z"]):
            features.append(normalized[i, axis])
            names.append(
                f"normalized_landmark_{i}_{axis_name}"
            )

    # --------------------------------------------------------
    # C. Wrist -> fingertip distances
    # --------------------------------------------------------

    for tip in FINGERTIPS:
        d = np.linalg.norm(pts[tip] - wrist)

        features.append(d / scale)
        names.append(
            f"wrist_to_tip_{tip}"
        )

    # --------------------------------------------------------
    # D. Pairwise fingertip distances
    # --------------------------------------------------------

    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):

            p1 = pts[FINGERTIPS[i]]
            p2 = pts[FINGERTIPS[j]]

            d = np.linalg.norm(p1 - p2)

            features.append(d / scale)

            names.append(
                f"tip_distance_{FINGERTIPS[i]}_{FINGERTIPS[j]}"
            )

    # --------------------------------------------------------
    # E. Finger segment lengths
    # --------------------------------------------------------

    for finger_name, chain in FINGERS.items():

        for i in range(len(chain) - 1):

            p1 = pts[chain[i]]
            p2 = pts[chain[i + 1]]

            d = np.linalg.norm(p2 - p1)

            features.append(d / scale)

            names.append(
                f"{finger_name}_segment_{i + 1}"
            )

    # --------------------------------------------------------
    # F. Finger joint angles
    # --------------------------------------------------------

    for finger_name, chain in FINGERS.items():

        # chain has 4 points
        # calculate angle at the two internal joints

        for i in range(1, len(chain) - 1):

            prev_p = pts[chain[i - 1]]
            curr_p = pts[chain[i]]
            next_p = pts[chain[i + 1]]

            v1 = prev_p - curr_p
            v2 = next_p - curr_p

            angle = safe_angle(v1, v2)

            features.append(angle / 180.0)

            names.append(
                f"{finger_name}_joint_angle_{i}"
            )

    # --------------------------------------------------------
    # G. MCP -> fingertip distances
    # --------------------------------------------------------

    for finger_name, chain in FINGERS.items():

        mcp = pts[chain[0]]
        tip = pts[chain[-1]]

        d = np.linalg.norm(tip - mcp)

        features.append(d / scale)

        names.append(
            f"{finger_name}_mcp_to_tip"
        )

    # --------------------------------------------------------
    # H. Finger direction angles
    #
    # Compare each finger direction against the middle finger.
    # --------------------------------------------------------

    middle_vec = pts[12] - pts[9]

    for finger_name, chain in FINGERS.items():

        if finger_name == "middle":
            continue

        vec = pts[chain[-1]] - pts[chain[0]]

        angle = safe_angle(vec, middle_vec)

        features.append(angle / 180.0)

        names.append(
            f"{finger_name}_vs_middle_angle"
        )

    return np.asarray(features, dtype=np.float64), names


def build_geometry_dataset(X):
    """
    Extract geometry features for every sample.
    """

    all_features = []
    feature_names = None

    for row in X:

        feats, names = extract_geometry_features(row)

        all_features.append(feats)

        if feature_names is None:
            feature_names = names

    return np.asarray(all_features), feature_names


# ============================================================
# METADATA
# ============================================================

def load_split_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    print(f"Metadata rows: {len(df)}")
    print(f"Metadata columns: {list(df.columns)}")

    # Remove failed images if the project has the file.
    failed_path = LANDMARK_DIR / "failed_images.json"

    if failed_path.exists():

        try:
            failed = load_json(failed_path)

            if isinstance(failed, dict):
                failed_paths = set(failed.keys())
            elif isinstance(failed, list):
                failed_paths = set(str(x) for x in failed)
            else:
                failed_paths = set()

            if failed_paths:

                before = len(df)

                df = df[
                    ~df["path"].astype(str).isin(failed_paths)
                ].copy()

                print(
                    f"Removed {before - len(df)} failed images."
                )

        except Exception as e:
            print(
                f"WARNING: Could not process failed_images.json: {e}"
            )

    return df


def get_split_metadata(df, split_name):

    split_df = df[
        df["split"].astype(str).str.lower() == split_name.lower()
    ].copy()

    return split_df.reset_index(drop=True)


# ============================================================
# MODEL FACTORY
# ============================================================

def make_models():

    return {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

        "SVM RBF": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "1-NN": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=1
                )
            )
        ]),
    }


# ============================================================
# BINARY FILTER
# ============================================================

def filter_binary(X, y, groups=None):

    mask = np.isin(
        y,
        [TUEY_ID, DAAL_ID]
    )

    X_filtered = X[mask]
    y_filtered = y[mask]

    if groups is not None:
        groups_filtered = np.asarray(groups)[mask]
    else:
        groups_filtered = None

    # Encode:
    # Tuey = 1
    # Daal = 0

    y_binary = np.where(
        y_filtered == TUEY_ID,
        1,
        0
    )

    return (
        X_filtered,
        y_binary,
        groups_filtered
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def group_cross_validation(
    model,
    X,
    y,
    groups
):

    unique_groups = np.unique(groups)

    # Need at least 2 groups.
    if len(unique_groups) < 2:
        raise ValueError(
            "Not enough unique subjects for GroupKFold."
        )

    n_splits = min(
        5,
        len(unique_groups)
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    scores = []

    for train_idx, val_idx in splitter.split(
        X,
        y,
        groups
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(X_val)

        scores.append(
            accuracy_score(
                y_val,
                pred
            )
        )

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        scores
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_test_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        pred
    )

    cm = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    )

    return (
        metrics,
        pred,
        cm,
        model
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "VoxaSign PSL — Tuey vs Daal Geometric Feature Experiment"
    )

    print(
        """
This experiment DOES NOT retrain the 36-class VoxaSign model.

It compares:

    1. V2 features
    2. Geometry-only features
    3. V2 + Geometry features

Target:

    Tuey
    Daal
"""
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print_header("CHECKING REQUIRED FILES")

    required_files = [

        V2_DIR / "X_train_v2.npy",
        V2_DIR / "X_validation_v2.npy",
        V2_DIR / "X_test_v2.npy",

        LANDMARK_DIR / "y_train.npy",
        LANDMARK_DIR / "y_validation.npy",
        LANDMARK_DIR / "y_test.npy",

        METADATA_PATH,
        CLASS_MAP_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        print(f"OK: {path}")

    # --------------------------------------------------------
    # Load class map
    # --------------------------------------------------------

    print_header("LOADING CLASS MAP")

    class_map = load_json(
        CLASS_MAP_PATH
    )

    print(
        f"Number of classes: {len(class_map)}"
    )

    print(
        f"Tuey = {TUEY_ID}"
    )

    print(
        f"Daal = {DAAL_ID}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print_header("LOADING V2 DATA")

    X_train = np.load(
        V2_DIR / "X_train_v2.npy"
    )

    X_val = np.load(
        V2_DIR / "X_validation_v2.npy"
    )

    X_test = np.load(
        V2_DIR / "X_test_v2.npy"
    )

    y_train = np.load(
        LANDMARK_DIR / "y_train.npy"
    )

    y_val = np.load(
        LANDMARK_DIR / "y_validation.npy"
    )

    y_test = np.load(
        LANDMARK_DIR / "y_test.npy"
    )

    print(
        f"Train:      X={X_train.shape} y={y_train.shape}"
    )

    print(
        f"Validation: X={X_val.shape} y={y_val.shape}"
    )

    print(
        f"Test:       X={X_test.shape} y={y_test.shape}"
    )

    if X_train.shape[1] < 63:

        raise RuntimeError(
            "V2 feature matrix has fewer than 63 features. "
            "Cannot extract raw 21-landmark geometry."
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print_header(
        "LOADING SUBJECT METADATA"
    )

    metadata = load_split_metadata()

    train_meta = get_split_metadata(
        metadata,
        "train"
    )

    val_meta = get_split_metadata(
        metadata,
        "validation"
    )

    test_meta = get_split_metadata(
        metadata,
        "test"
    )

    print(
        f"Train metadata:      {len(train_meta)}"
    )

    print(
        f"Validation metadata: {len(val_meta)}"
    )

    print(
        f"Test metadata:       {len(test_meta)}"
    )

    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    print_header(
        "CHECKING METADATA / ARRAY ALIGNMENT"
    )

    alignment_ok = (
        len(train_meta) == len(X_train)
        and
        len(val_meta) == len(X_val)
        and
        len(test_meta) == len(X_test)
    )

    if alignment_ok:

        print(
            "OK: metadata appears aligned with landmark arrays."
        )

    else:

        print(
            "WARNING: metadata lengths do not match arrays."
        )

        print(
            f"Train: X={len(X_train)}, metadata={len(train_meta)}"
        )

        print(
            f"Val:   X={len(X_val)}, metadata={len(val_meta)}"
        )

        print(
            f"Test:  X={len(X_test)}, metadata={len(test_meta)}"
        )

        print(
            """
Subject-aware validation will NOT be used because metadata
alignment cannot be guaranteed.
"""
        )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    if alignment_ok:

        train_groups = train_meta[
            "subject_id"
        ].astype(str).values

        val_groups = val_meta[
            "subject_id"
        ].astype(str).values

        test_groups = test_meta[
            "subject_id"
        ].astype(str).values

    else:

        train_groups = None
        val_groups = None
        test_groups = None

    # --------------------------------------------------------
    # Build geometry
    # --------------------------------------------------------

    print_header(
        "EXTRACTING GEOMETRIC FEATURES"
    )

    print(
        "Building geometry features from the first 63 V2 features..."
    )

    G_train, geometry_names = build_geometry_dataset(
        X_train
    )

    G_val, _ = build_geometry_dataset(
        X_val
    )

    G_test, _ = build_geometry_dataset(
        X_test
    )

    print(
        f"Geometry train shape: {G_train.shape}"
    )

    print(
        f"Geometry val shape:   {G_val.shape}"
    )

    print(
        f"Geometry test shape:  {G_test.shape}"
    )

    print(
        f"Geometry feature count: {len(geometry_names)}"
    )

    # --------------------------------------------------------
    # Save geometry feature names
    # --------------------------------------------------------

    feature_info = {
        "source": "first_63_V2_features",
        "landmarks": 21,
        "coordinates_per_landmark": 3,
        "geometry_feature_count": len(
            geometry_names
        ),
        "features": geometry_names
    }

    save_json(
        REPORT_DIR / "geometry_feature_info.json",
        feature_info
    )

    # --------------------------------------------------------
    # Create V2 + geometry
    # --------------------------------------------------------

    V2G_train = np.concatenate(
        [
            X_train,
            G_train
        ],
        axis=1
    )

    V2G_val = np.concatenate(
        [
            X_val,
            G_val
        ],
        axis=1
    )

    V2G_test = np.concatenate(
        [
            X_test,
            G_test
        ],
        axis=1
    )

    print(
        f"V2 + Geometry train shape: {V2G_train.shape}"
    )

    # --------------------------------------------------------
    # Binary filtering
    # --------------------------------------------------------

    print_header(
        "FILTERING TUEY / DAAL"
    )

    Xtr_v2, ytr, groups_tr = filter_binary(
        X_train,
        y_train,
        train_groups
    )

    Xva_v2, yva, groups_va = filter_binary(
        X_val,
        y_val,
        val_groups
    )

    Xte_v2, yte, groups_te = filter_binary(
        X_test,
        y_test,
        test_groups
    )

    Gtr, _, _ = filter_binary(
        G_train,
        y_train,
        train_groups
    )

    Gva, _, _ = filter_binary(
        G_val,
        y_val,
        val_groups
    )

    Gte, _, _ = filter_binary(
        G_test,
        y_test,
        test_groups
    )

    V2Gtr, _, _ = filter_binary(
        V2G_train,
        y_train,
        train_groups
    )

    V2Gva, _, _ = filter_binary(
        V2G_val,
        y_val,
        val_groups
    )

    V2Gte, _, _ = filter_binary(
        V2G_test,
        y_test,
        test_groups
    )

    print(
        f"Train: Tuey={np.sum(ytr == 1)} | "
        f"Daal={np.sum(ytr == 0)}"
    )

    print(
        f"Validation: Tuey={np.sum(yva == 1)} | "
        f"Daal={np.sum(yva == 0)}"
    )

    print(
        f"Test: Tuey={np.sum(yte == 1)} | "
        f"Daal={np.sum(yte == 0)}"
    )

    # --------------------------------------------------------
    # Feature sets
    # --------------------------------------------------------

    feature_sets = {

        "V2": (
            Xtr_v2,
            Xte_v2
        ),

        "Geometry Only": (
            Gtr,
            Gte
        ),

        "V2 + Geometry": (
            V2Gtr,
            V2Gte
        )
    }

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    all_results = {}

    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

    print_header(
        "SUBJECT-AWARE CROSS-VALIDATION"
    )

    for feature_set_name, (
        Xtr,
        Xte
    ) in feature_sets.items():

        print()
        print(
            f"### {feature_set_name}"
        )

        all_results[
            feature_set_name
        ] = {}

        for model_name, model in make_models().items():

            if groups_tr is not None:

                try:

                    mean_score, std_score, fold_scores = (
                        group_cross_validation(
                            model,
                            Xtr,
                            ytr,
                            groups_tr
                        )
                    )

                    print(
                        f"{model_name:22s}: "
                        f"{mean_score * 100:.2f}% "
                        f"+/- {std_score * 100:.2f}%"
                    )

                    all_results[
                        feature_set_name
                    ][model_name] = {
                        "cv_accuracy": mean_score,
                        "cv_std": std_score,
                        "fold_scores": fold_scores,
                        "validation_type": "GroupKFold"
                    }

                except Exception as e:

                    print(
                        f"{model_name:22s}: CV FAILED — {e}"
                    )

            else:

                print(
                    f"{model_name:22s}: "
                    "SKIPPED — metadata alignment unavailable"
                )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print_header(
        "HELD-OUT TEST EVALUATION"
    )

    test_rows = []

    best_overall = None

    for feature_set_name, (
        Xtr,
        Xte
    ) in feature_sets.items():

        print()
        print(
            f"### {feature_set_name}"
        )

        for model_name, model in make_models().items():

            metrics, pred, cm, fitted = evaluate_test_model(
                model,
                Xtr,
                ytr,
                Xte,
                yte
            )

            print(
                f"{model_name:22s}: "
                f"Accuracy={metrics['accuracy'] * 100:.2f}% | "
                f"F1={metrics['f1'] * 100:.2f}%"
            )

            result = {
                "feature_set": feature_set_name,
                "model": model_name,
                **metrics,
                "confusion_matrix": cm.tolist()
            }

            test_rows.append(
                result
            )

            if (
                best_overall is None
                or metrics["f1"]
                > best_overall["f1"]
            ):

                best_overall = result

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    print_header(
        "RESULT COMPARISON"
    )

    results_df = pd.DataFrame(
        test_rows
    )

    print(
        results_df[
            [
                "feature_set",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    print_header(
        "BEST TEST CONFIGURATION"
    )

    print(
        f"Feature set: {best_overall['feature_set']}"
    )

    print(
        f"Model:       {best_overall['model']}"
    )

    print(
        f"Accuracy:    {best_overall['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:   {best_overall['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:      {best_overall['recall'] * 100:.2f}%"
    )

    print(
        f"F1:          {best_overall['f1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Best model detailed evaluation
    # --------------------------------------------------------

    print_header(
        "BEST MODEL CLASSIFICATION REPORT"
    )

    best_fs = best_overall[
        "feature_set"
    ]

    best_model_name = best_overall[
        "model"
    ]

    Xtr_best, Xte_best = feature_sets[
        best_fs
    ]

    best_model = make_models()[
        best_model_name
    ]

    best_model.fit(
        Xtr_best,
        ytr
    )

    best_pred = best_model.predict(
        Xte_best
    )

    report = classification_report(
        yte,
        best_pred,
        target_names=[
            DAAL_NAME,
            TUEY_NAME
        ],
        zero_division=0
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Test prediction CSV
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "test_index": np.arange(
            len(yte)
        ),
        "true_class": [
            TUEY_NAME if x == 1 else DAAL_NAME
            for x in yte
        ],
        "predicted_class": [
            TUEY_NAME if x == 1 else DAAL_NAME
            for x in best_pred
        ],
        "correct": (
            yte == best_pred
        )
    })

    if alignment_ok:

        binary_test_meta = test_meta[
            np.isin(
                y_test,
                [TUEY_ID, DAAL_ID]
            )
        ].reset_index(drop=True)

        if len(binary_test_meta) == len(
            prediction_df
        ):

            prediction_df[
                "subject_id"
            ] = binary_test_meta[
                "subject_id"
            ].values

            prediction_df[
                "filename"
            ] = binary_test_meta[
                "filename"
            ].values

            prediction_df[
                "source"
            ] = binary_test_meta[
                "source"
            ].values

    prediction_path = (
        REPORT_DIR /
        "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Saved: {prediction_path}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print_header(
        "FEATURE IMPORTANCE"
    )

    try:

        # Only directly interpretable for RF.
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf.fit(
            Xtr_best,
            ytr
        )

        importances = rf.feature_importances_

        if best_fs == "V2":

            names = [
                f"V2_Feature_{i}"
                for i in range(
                    Xtr_best.shape[1]
                )
            ]

        elif best_fs == "Geometry Only":

            names = geometry_names

        else:

            names = (
                [
                    f"V2_Feature_{i}"
                    for i in range(
                        X_train.shape[1]
                    )
                ]
                +
                geometry_names
            )

        importance_df = pd.DataFrame({
            "feature": names,
            "importance": importances
        }).sort_values(
            "importance",
            ascending=False
        )

        importance_path = (
            REPORT_DIR /
            "feature_importance.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False
        )

        print(
            importance_df.head(
                30
            ).to_string(
                index=False
            )
        )

        print(
            f"\nSaved: {importance_path}"
        )

    except Exception as e:

        print(
            f"Feature importance failed: {e}"
        )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print_header(
        "INTERPRETATION"
    )

    v2_results = results_df[
        results_df["feature_set"] == "V2"
    ]

    geo_results = results_df[
        results_df["feature_set"] == "Geometry Only"
    ]

    combined_results = results_df[
        results_df["feature_set"] == "V2 + Geometry"
    ]

    best_v2 = (
        v2_results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    best_geo = (
        geo_results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    best_combined = (
        combined_results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print(
        f"""
Best V2:
    {best_v2['model']}
    Accuracy: {best_v2['accuracy'] * 100:.2f}%
    F1:       {best_v2['f1'] * 100:.2f}%

Best Geometry:
    {best_geo['model']}
    Accuracy: {best_geo['accuracy'] * 100:.2f}%
    F1:       {best_geo['f1'] * 100:.2f}%

Best V2 + Geometry:
    {best_combined['model']}
    Accuracy: {best_combined['accuracy'] * 100:.2f}%
    F1:       {best_combined['f1'] * 100:.2f}%
"""
    )

    improvement = (
        best_combined["f1"]
        -
        best_v2["f1"]
    )

    print(
        f"V2 + Geometry F1 improvement: "
        f"{improvement * 100:+.2f} percentage points"
    )

    if improvement >= 0.05:

        conclusion = (
            "STRONG POSITIVE RESULT: "
            "geometry substantially improves Tuey/Daal "
            "classification. Geometry should be considered "
            "for V3."
        )

    elif improvement >= 0.02:

        conclusion = (
            "POSITIVE RESULT: geometry provides a meaningful "
            "improvement. Continue with a targeted V3 experiment."
        )

    elif improvement > 0:

        conclusion = (
            "SMALL POSITIVE RESULT: geometry helps slightly. "
            "More targeted feature selection is recommended."
        )

    else:

        conclusion = (
            "NO POSITIVE RESULT: geometry does not improve "
            "the current representation. Do not add all geometry "
            "features to V3 yet."
        )

    print(
        f"\n{conclusion}"
    )

    print(
        """
IMPORTANT:

This experiment does NOT modify:
    - psl_static_model_v2.keras
    - psl_static_model_v1.keras
    - class_map.json
    - V2 training data

It is purely an analysis experiment.
"""
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_json = {

        "experiment": "Tuey vs Daal Geometry",

        "target_classes": {
            "Tuey": TUEY_ID,
            "Daal": DAAL_ID
        },

        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_validation": list(X_val.shape),
            "X_test": list(X_test.shape),
            "geometry_train": list(G_train.shape),
            "geometry_validation": list(G_val.shape),
            "geometry_test": list(G_test.shape)
        },

        "feature_counts": {
            "v2": int(X_train.shape[1]),
            "geometry": int(G_train.shape[1]),
            "v2_plus_geometry": int(
                V2G_train.shape[1]
            )
        },

        "test_results": test_rows,

        "best_configuration": best_overall,

        "interpretation": {
            "v2_best_f1": float(
                best_v2["f1"]
            ),
            "geometry_best_f1": float(
                best_geo["f1"]
            ),
            "combined_best_f1": float(
                best_combined["f1"]
            ),
            "combined_minus_v2_f1": float(
                improvement
            ),
            "conclusion": conclusion
        }
    }

    json_path = (
        REPORT_DIR /
        "tuey_daal_geometry.json"
    )

    save_json(
        json_path,
        output_json
    )

    # --------------------------------------------------------
    # Save TXT report
    # --------------------------------------------------------

    txt_path = (
        REPORT_DIR /
        "tuey_daal_geometry.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VoxaSign PSL — Tuey vs Daal "
            "Geometric Feature Experiment\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"V2 best model: {best_v2['model']}\n"
        )

        f.write(
            f"V2 accuracy: {best_v2['accuracy']:.6f}\n"
        )

        f.write(
            f"V2 F1: {best_v2['f1']:.6f}\n\n"
        )

        f.write(
            f"Geometry best model: {best_geo['model']}\n"
        )

        f.write(
            f"Geometry accuracy: "
            f"{best_geo['accuracy']:.6f}\n"
        )

        f.write(
            f"Geometry F1: "
            f"{best_geo['f1']:.6f}\n\n"
        )

        f.write(
            f"V2 + Geometry best model: "
            f"{best_combined['model']}\n"
        )

        f.write(
            f"V2 + Geometry accuracy: "
            f"{best_combined['accuracy']:.6f}\n"
        )

        f.write(
            f"V2 + Geometry F1: "
            f"{best_combined['f1']:.6f}\n\n"
        )

        f.write(
            f"F1 improvement: "
            f"{improvement:+.6f}\n\n"
        )

        f.write(
            conclusion + "\n"
        )

    print(
        f"Saved: {txt_path}"
    )

    print_header(
        "ANALYSIS COMPLETE"
    )

    print(
        "Run complete."
    )

    print(
        "\nReports:"
    )

    print(
        f"  {txt_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {prediction_path}"
    )

    print(
        f"  {REPORT_DIR / 'geometry_feature_info.json'}"
    )

    print(
        f"  {REPORT_DIR / 'feature_importance.csv'}"
    )

    print(
        "\nDO NOT RETRAIN V3 YET."
    )

    print(
        "Send me the COMPLETE terminal output."
    )


if __name__ == "__main__":
    main()