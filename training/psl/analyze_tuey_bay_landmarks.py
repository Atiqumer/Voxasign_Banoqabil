"""
VoxaSign PSL — Tuey vs Bay Landmark Analysis
================================================

Purpose
-------
Analyze why Tuey is being confused with Bay in V2.

This script:
1. Loads V2 landmark features.
2. Identifies Tuey and Bay class IDs.
3. Extracts Tuey/Bay samples from train/validation/test.
4. Computes feature statistics.
5. Measures class centroid separation.
6. Measures within-class and between-class distances.
7. Tests Tuey-vs-Bay separability using:
   - 1-NN
   - Logistic Regression
   - Random Forest
8. Performs PCA analysis.
9. Performs feature importance analysis.
10. Checks subject-level generalization.
11. Saves TXT, JSON, CSV and PCA outputs.

IMPORTANT
---------
This is an ANALYSIS script only.

DO NOT retrain V3 based only on this script.
Use the results to decide the V3 strategy.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

LANDMARK_DIR = ROOT / "data" / "landmarks"
V2_DIR = LANDMARK_DIR / "v2"

METADATA_PATH = ROOT / "data" / "metadata" / "psl_static_metadata.csv"
CLASS_MAP_PATH = ROOT / "output" / "class_map.json"

REPORT_DIR = ROOT / "reports" / "tuey_bay_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

X_TRAIN_PATH = V2_DIR / "X_train_v2.npy"
X_VAL_PATH = V2_DIR / "X_validation_v2.npy"
X_TEST_PATH = V2_DIR / "X_test_v2.npy"

Y_TRAIN_PATH = LANDMARK_DIR / "y_train.npy"
Y_VAL_PATH = LANDMARK_DIR / "y_validation.npy"
Y_TEST_PATH = LANDMARK_DIR / "y_test.npy"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_npy(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return np.load(path, allow_pickle=True)


def load_class_map():
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Class map not found:\n{CLASS_MAP_PATH}"
        )

    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def normalize_class_map(data):
    """
    Handles common formats:

    {
        "0": "1-Hay",
        "1": "Ain"
    }

    OR

    {
        "1-Hay": 0,
        "Ain": 1
    }
    """

    if not data:
        raise ValueError("class_map.json is empty.")

    first_key = next(iter(data.keys()))
    first_value = data[first_key]

    if isinstance(first_value, str):
        # ID -> class
        return {int(k): v for k, v in data.items()}

    if isinstance(first_value, int):
        # class -> ID
        return {int(v): k for k, v in data.items()}

    raise ValueError(
        "Unsupported class_map.json format."
    )


def feature_name(index):
    """
    V2 has 115 features.

    The first 63 are raw x/y/z landmark coordinates.

    The remaining features are derived V2 geometric features.
    """

    if index < 63:
        landmark = index // 3
        coord_index = index % 3

        coord = ["x", "y", "z"][coord_index]

        return f"Landmark {landmark:2d} {coord}"

    return f"V2 Feature {index}"


def safe_mean_distance(A, B):
    """
    Mean Euclidean distance between two feature matrices.
    """

    if len(A) == 0 or len(B) == 0:
        return float("nan")

    # Avoid huge memory usage.
    distances = []

    for x in A:
        d = np.linalg.norm(B - x, axis=1)
        distances.append(np.mean(d))

    return float(np.mean(distances))


def pairwise_mean_distance(X):
    """
    Mean pairwise distance within one class.
    """

    n = len(X)

    if n < 2:
        return float("nan")

    distances = []

    for i in range(n):
        d = np.linalg.norm(X[i + 1:] - X[i], axis=1)

        if len(d):
            distances.extend(d.tolist())

    return float(np.mean(distances))


def get_class_indices(y, class_id):
    return np.where(y == class_id)[0]


def get_class_data(X, y, class_id):
    idx = get_class_indices(y, class_id)
    return X[idx]


def normalize_features(X):
    """
    Normalize each sample using the same type of scale-independent
    representation expected by landmark-based models.

    For analysis we standardize features globally afterward.
    """

    X = np.asarray(X, dtype=np.float32)

    # Replace invalid values.
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "VoxaSign PSL — Tuey vs Bay Landmark Analysis"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print_header("CHECKING REQUIRED FILES")

    required = [
        X_TRAIN_PATH,
        X_VAL_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_VAL_PATH,
        Y_TEST_PATH,
        METADATA_PATH,
        CLASS_MAP_PATH,
    ]

    for path in required:

        if path.exists():
            print(f"OK: {path}")

        else:
            print(f"MISSING: {path}")
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # LOAD CLASS MAP
    # --------------------------------------------------------

    print_header("LOADING CLASS MAP")

    raw_class_map = load_class_map()
    class_map = normalize_class_map(raw_class_map)

    reverse_map = {
        name: class_id
        for class_id, name in class_map.items()
    }

    print(f"Number of classes: {len(class_map)}")

    if "Tuey" not in reverse_map:
        raise RuntimeError(
            "Tuey was not found in class_map.json"
        )

    if "Bay" not in reverse_map:
        raise RuntimeError(
            "Bay was not found in class_map.json"
        )

    tuey_id = reverse_map["Tuey"]
    bay_id = reverse_map["Bay"]

    print(f"Tuey = {tuey_id}")
    print(f"Bay  = {bay_id}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print_header("LOADING LANDMARK DATA")

    X_train = normalize_features(
        load_npy(X_TRAIN_PATH)
    )

    X_val = normalize_features(
        load_npy(X_VAL_PATH)
    )

    X_test = normalize_features(
        load_npy(X_TEST_PATH)
    )

    y_train = load_npy(Y_TRAIN_PATH).astype(int)
    y_val = load_npy(Y_VAL_PATH).astype(int)
    y_test = load_npy(Y_TEST_PATH).astype(int)

    print(f"Train:      X={X_train.shape} y={y_train.shape}")
    print(f"Validation: X={X_val.shape} y={y_val.shape}")
    print(f"Test:       X={X_test.shape} y={y_test.shape}")

    # --------------------------------------------------------
    # CHECK FEATURE DIMENSIONS
    # --------------------------------------------------------

    feature_count = X_train.shape[1]

    print()
    print(f"Feature count: {feature_count}")

    # --------------------------------------------------------
    # EXTRACT TUEY / BAY
    # --------------------------------------------------------

    print_header("TUEY / BAY SAMPLE COUNTS")

    datasets = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

    class_data = {}

    for split_name, (X, y) in datasets.items():

        tuey = get_class_data(X, y, tuey_id)
        bay = get_class_data(X, y, bay_id)

        class_data[split_name] = {
            "Tuey": tuey,
            "Bay": bay,
        }

        print(
            f"{split_name.capitalize():12s}: "
            f"Tuey={len(tuey):3d} | "
            f"Bay={len(bay):3d}"
        )

    X_tuey_train = class_data["train"]["Tuey"]
    X_bay_train = class_data["train"]["Bay"]

    X_tuey_val = class_data["validation"]["Tuey"]
    X_bay_val = class_data["validation"]["Bay"]

    X_tuey_test = class_data["test"]["Tuey"]
    X_bay_test = class_data["test"]["Bay"]

    if len(X_tuey_train) == 0 or len(X_bay_train) == 0:
        raise RuntimeError(
            "No Tuey/Bay training samples found."
        )

    # --------------------------------------------------------
    # COMBINED BINARY DATASET
    # --------------------------------------------------------

    X_binary = np.vstack([
        X_tuey_train,
        X_bay_train,
    ])

    y_binary = np.concatenate([
        np.ones(len(X_tuey_train), dtype=int),
        np.zeros(len(X_bay_train), dtype=int),
    ])

    # 1 = Tuey
    # 0 = Bay

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    print_header("BASIC FEATURE STATISTICS")

    tuey_mean = np.mean(X_tuey_train, axis=0)
    bay_mean = np.mean(X_bay_train, axis=0)

    tuey_std = np.std(X_tuey_train, axis=0)
    bay_std = np.std(X_bay_train, axis=0)

    mean_difference = np.abs(
        tuey_mean - bay_mean
    )

    pooled_std = np.sqrt(
        (tuey_std ** 2 + bay_std ** 2) / 2
    )

    effect_size = np.divide(
        mean_difference,
        pooled_std + 1e-8
    )

    top_effect_indices = np.argsort(
        effect_size
    )[::-1][:20]

    print()
    print("TOP 20 FEATURE EFFECT SIZES")
    print("-" * 78)

    feature_rows = []

    for rank, idx in enumerate(
        top_effect_indices,
        start=1
    ):

        row = {
            "rank": rank,
            "feature_index": int(idx),
            "feature_name": feature_name(idx),
            "tuey_mean": float(tuey_mean[idx]),
            "bay_mean": float(bay_mean[idx]),
            "absolute_difference": float(
                mean_difference[idx]
            ),
            "pooled_std": float(
                pooled_std[idx]
            ),
            "effect_size": float(
                effect_size[idx]
            ),
        }

        feature_rows.append(row)

        print(
            f"{rank:2d}. "
            f"Feature {idx:3d} | "
            f"{feature_name(idx):20s} | "
            f"Effect={effect_size[idx]:.5f}"
        )

    # --------------------------------------------------------
    # CENTROID DISTANCE
    # --------------------------------------------------------

    print_header("LANDMARK / FEATURE DISTANCE ANALYSIS")

    tuey_centroid = np.mean(
        X_tuey_train,
        axis=0
    )

    bay_centroid = np.mean(
        X_bay_train,
        axis=0
    )

    centroid_distance = float(
        np.linalg.norm(
            tuey_centroid - bay_centroid
        )
    )

    tuey_within = pairwise_mean_distance(
        X_tuey_train
    )

    bay_within = pairwise_mean_distance(
        X_bay_train
    )

    tuey_to_bay = safe_mean_distance(
        X_tuey_train,
        X_bay_train
    )

    bay_to_tuey = safe_mean_distance(
        X_bay_train,
        X_tuey_train
    )

    between_class_distance = (
        tuey_to_bay + bay_to_tuey
    ) / 2

    average_within = (
        tuey_within + bay_within
    ) / 2

    separation_ratio = (
        between_class_distance /
        (average_within + 1e-8)
    )

    print(
        f"Tuey centroid ↔ Bay centroid: "
        f"{centroid_distance:.4f}"
    )

    print(
        f"Tuey within-class distance:    "
        f"{tuey_within:.4f}"
    )

    print(
        f"Bay within-class distance:     "
        f"{bay_within:.4f}"
    )

    print(
        f"Tuey ↔ Bay mean distance:       "
        f"{between_class_distance:.4f}"
    )

    print(
        f"Separation ratio:               "
        f"{separation_ratio:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFIER SEPARABILITY
    # --------------------------------------------------------

    print_header("TUEY / BAY SEPARABILITY")

    scaler = StandardScaler()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    classifier_results = {}

    # --------------------------------------------------------
    # 1-NN
    # --------------------------------------------------------

    knn = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=1
            )
        ),
    ])

    knn_scores = cross_val_score(
        knn,
        X_binary,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    classifier_results["1-NN"] = {
        "mean": float(np.mean(knn_scores)),
        "std": float(np.std(knn_scores)),
        "scores": knn_scores.tolist(),
    }

    print(
        f"1-NN                  : "
        f"{np.mean(knn_scores) * 100:.2f}% "
        f"+/- {np.std(knn_scores) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=42,
            )
        ),
    ])

    logistic_scores = cross_val_score(
        logistic,
        X_binary,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    classifier_results["Logistic Regression"] = {
        "mean": float(np.mean(logistic_scores)),
        "std": float(np.std(logistic_scores)),
        "scores": logistic_scores.tolist(),
    }

    print(
        f"Logistic Regression   : "
        f"{np.mean(logistic_scores) * 100:.2f}% "
        f"+/- {np.std(logistic_scores) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf_scores = cross_val_score(
        rf,
        X_binary,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    classifier_results["Random Forest"] = {
        "mean": float(np.mean(rf_scores)),
        "std": float(np.std(rf_scores)),
        "scores": rf_scores.tolist(),
    }

    print(
        f"Random Forest         : "
        f"{np.mean(rf_scores) * 100:.2f}% "
        f"+/- {np.std(rf_scores) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------------

    print_header("RANDOM FOREST FEATURE IMPORTANCE")

    rf.fit(X_binary, y_binary)

    importances = rf.feature_importances_

    top_rf_indices = np.argsort(
        importances
    )[::-1][:20]

    rf_rows = []

    for rank, idx in enumerate(
        top_rf_indices,
        start=1
    ):

        row = {
            "rank": rank,
            "feature_index": int(idx),
            "feature_name": feature_name(idx),
            "importance": float(
                importances[idx]
            ),
        }

        rf_rows.append(row)

        print(
            f"{rank:2d}. "
            f"Feature {idx:3d} | "
            f"{feature_name(idx):20s} | "
            f"Importance={importances[idx]:.5f}"
        )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST BINARY EVALUATION
    # --------------------------------------------------------

    print_header(
        "BINARY TEST EVALUATION"
    )

    X_test_binary = np.vstack([
        X_tuey_test,
        X_bay_test,
    ])

    y_test_binary = np.concatenate([
        np.ones(len(X_tuey_test), dtype=int),
        np.zeros(len(X_bay_test), dtype=int),
    ])

    logistic.fit(
        X_binary,
        y_binary
    )

    binary_predictions = logistic.predict(
        X_test_binary
    )

    binary_accuracy = accuracy_score(
        y_test_binary,
        binary_predictions
    )

    binary_precision = precision_score(
        y_test_binary,
        binary_predictions,
        zero_division=0,
    )

    binary_recall = recall_score(
        y_test_binary,
        binary_predictions,
        zero_division=0,
    )

    binary_f1 = f1_score(
        y_test_binary,
        binary_predictions,
        zero_division=0,
    )

    print(
        f"Accuracy:  {binary_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {binary_precision * 100:.2f}%"
    )

    print(
        f"Recall:    {binary_recall * 100:.2f}%"
    )

    print(
        f"F1:        {binary_f1 * 100:.2f}%"
    )

    binary_cm = confusion_matrix(
        y_test_binary,
        binary_predictions,
    )

    print()
    print("Confusion matrix:")
    print(binary_cm)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print_header("PCA ANALYSIS")

    X_pca_input = np.vstack([
        X_tuey_train,
        X_bay_train,
        X_tuey_test,
        X_bay_test,
    ])

    pca_labels = np.concatenate([
        np.ones(len(X_tuey_train), dtype=int),
        np.zeros(len(X_bay_train), dtype=int),
        np.ones(len(X_tuey_test), dtype=int),
        np.zeros(len(X_bay_test), dtype=int),
    ])

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    X_scaled_pca = StandardScaler().fit_transform(
        X_pca_input
    )

    X_pca = pca.fit_transform(
        X_scaled_pca
    )

    explained = pca.explained_variance_ratio_

    print(
        f"PC1 explained variance: "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained[1] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PCA PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    tuey_mask = pca_labels == 1
    bay_mask = pca_labels == 0

    plt.scatter(
        X_pca[bay_mask, 0],
        X_pca[bay_mask, 1],
        label="Bay",
        alpha=0.65,
        s=35,
    )

    plt.scatter(
        X_pca[tuey_mask, 0],
        X_pca[tuey_mask, 1],
        label="Tuey",
        alpha=0.65,
        s=35,
    )

    plt.xlabel(
        f"PC1 ({explained[0] * 100:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained[1] * 100:.1f}% variance)"
    )

    plt.title(
        "VoxaSign PSL — Tuey vs Bay PCA"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    pca_path = REPORT_DIR / "tuey_bay_pca.png"

    plt.savefig(
        pca_path,
        dpi=180,
    )

    plt.close()

    print(
        f"PCA plot saved: {pca_path}"
    )

    # --------------------------------------------------------
    # SUBJECT ANALYSIS
    # --------------------------------------------------------

    print_header(
        "SUBJECT-LEVEL ANALYSIS"
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    print(
        "Metadata columns:"
    )

    print(
        metadata.columns.tolist()
    )

    # --------------------------------------------------------
    # DETERMINE SUBJECT SPLITS
    # --------------------------------------------------------

    split_subjects = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = metadata[
            metadata["split"] == split_name
        ]

        subjects = sorted(
            subset["subject_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        split_subjects[split_name] = subjects

        print(
            f"{split_name.capitalize():12s}: "
            f"{len(subjects)} subjects"
        )

    # --------------------------------------------------------
    # SUBJECT CLASS COVERAGE
    # --------------------------------------------------------

    subject_class_counts = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = metadata[
            metadata["split"] == split_name
        ]

        subset = subset[
            subset["class"].isin(
                ["Tuey", "Bay"]
            )
        ]

        counts = (
            subset
            .groupby(
                "subject_id"
            )["class"]
            .nunique()
        )

        subject_class_counts[
            split_name
        ] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    # --------------------------------------------------------
    # FEATURE DISTRIBUTION TABLE
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        feature_rows
    )

    feature_csv = (
        REPORT_DIR /
        "tuey_bay_feature_comparison.csv"
    )

    feature_df.to_csv(
        feature_csv,
        index=False,
    )

    print()
    print(
        f"Feature comparison saved: "
        f"{feature_csv}"
    )

    # --------------------------------------------------------
    # RF FEATURE TABLE
    # --------------------------------------------------------

    rf_df = pd.DataFrame(
        rf_rows
    )

    rf_csv = (
        REPORT_DIR /
        "tuey_bay_rf_features.csv"
    )

    rf_df.to_csv(
        rf_csv,
        index=False,
    )

    print(
        f"RF feature importance saved: "
        f"{rf_csv}"
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print_header("INTERPRETATION")

    if separation_ratio < 1.0:

        interpretation = (
            "Tuey and Bay have substantial landmark overlap. "
            "Their between-class distance is not larger than "
            "their average within-class variation."
        )

    elif separation_ratio < 1.5:

        interpretation = (
            "Tuey and Bay are partially separable, but "
            "there is meaningful landmark overlap."
        )

    else:

        interpretation = (
            "Tuey and Bay appear reasonably separable "
            "in the current V2 feature representation."
        )

    print(interpretation)

    best_classifier = max(
        classifier_results.items(),
        key=lambda item: item[1]["mean"]
    )

    print()
    print(
        f"Best binary classifier: "
        f"{best_classifier[0]}"
    )

    print(
        f"Cross-validation accuracy: "
        f"{best_classifier[1]['mean'] * 100:.2f}%"
    )

    print()

    if (
        binary_accuracy >= 0.90
    ):
        recommendation = (
            "Tuey/Bay are separable using the current "
            "features. A targeted V3 classifier or "
            "loss strategy may be useful."
        )

    elif (
        binary_accuracy >= 0.75
    ):
        recommendation = (
            "Tuey/Bay have moderate separability. "
            "Investigate geometric features and "
            "targeted augmentation before major model changes."
        )

    else:
        recommendation = (
            "Tuey/Bay are difficult to separate using "
            "the current feature representation. "
            "Inspect original images and landmark extraction "
            "before retraining."
        )

    print(recommendation)

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    results = {
        "version": "Tuey-Bay Analysis V1",

        "classes": {
            "Tuey": tuey_id,
            "Bay": bay_id,
        },

        "dataset_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_val.shape),
            "test": list(X_test.shape),
        },

        "sample_counts": {
            split: {
                "Tuey": len(data["Tuey"]),
                "Bay": len(data["Bay"]),
            }
            for split, data in class_data.items()
        },

        "distance_analysis": {
            "centroid_distance": centroid_distance,
            "tuey_within_class": tuey_within,
            "bay_within_class": bay_within,
            "tuey_bay_mean_distance": between_class_distance,
            "separation_ratio": separation_ratio,
        },

        "binary_test_metrics": {
            "accuracy": float(binary_accuracy),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "confusion_matrix": binary_cm.tolist(),
        },

        "classifier_separability": classifier_results,

        "pca": {
            "pc1_variance": float(explained[0]),
            "pc2_variance": float(explained[1]),
        },

        "top_effect_features": feature_rows,

        "top_random_forest_features": rf_rows,

        "subject_analysis": {
            "split_subject_counts": {
                k: len(v)
                for k, v in split_subjects.items()
            },
            "class_counts": subject_class_counts,
        },

        "interpretation": interpretation,

        "recommendation": recommendation,
    }

    json_path = (
        REPORT_DIR /
        "tuey_bay_analysis.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"JSON saved: {json_path}"
    )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    txt_path = (
        REPORT_DIR /
        "tuey_bay_analysis.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "VoxaSign PSL — Tuey vs Bay Landmark Analysis\n"
        )

        f.write(
            "=" * 78 + "\n\n"
        )

        f.write(
            "CLASS IDS\n"
        )

        f.write(
            f"Tuey = {tuey_id}\n"
        )

        f.write(
            f"Bay  = {bay_id}\n\n"
        )

        f.write(
            "DATASET SHAPES\n"
        )

        f.write(
            f"Train      : {X_train.shape}\n"
        )

        f.write(
            f"Validation : {X_val.shape}\n"
        )

        f.write(
            f"Test       : {X_test.shape}\n\n"
        )

        f.write(
            "SAMPLE COUNTS\n"
        )

        for split, data in class_data.items():

            f.write(
                f"{split}: "
                f"Tuey={len(data['Tuey'])}, "
                f"Bay={len(data['Bay'])}\n"
            )

        f.write("\n")

        f.write(
            "DISTANCE ANALYSIS\n"
        )

        f.write(
            f"Centroid distance: "
            f"{centroid_distance:.6f}\n"
        )

        f.write(
            f"Tuey within-class: "
            f"{tuey_within:.6f}\n"
        )

        f.write(
            f"Bay within-class: "
            f"{bay_within:.6f}\n"
        )

        f.write(
            f"Tuey-Bay mean distance: "
            f"{between_class_distance:.6f}\n"
        )

        f.write(
            f"Separation ratio: "
            f"{separation_ratio:.6f}\n\n"
        )

        f.write(
            "BINARY TEST RESULTS\n"
        )

        f.write(
            f"Accuracy: "
            f"{binary_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Precision: "
            f"{binary_precision * 100:.2f}%\n"
        )

        f.write(
            f"Recall: "
            f"{binary_recall * 100:.2f}%\n"
        )

        f.write(
            f"F1: "
            f"{binary_f1 * 100:.2f}%\n\n"
        )

        f.write(
            "CLASSIFIER SEPARABILITY\n"
        )

        for name, result in classifier_results.items():

            f.write(
                f"{name}: "
                f"{result['mean'] * 100:.2f}% "
                f"+/- "
                f"{result['std'] * 100:.2f}%\n"
            )

        f.write("\n")

        f.write(
            "TOP EFFECT-SIZE FEATURES\n"
        )

        for row in feature_rows:

            f.write(
                f"{row['rank']:2d}. "
                f"Feature {row['feature_index']:3d} "
                f"{row['feature_name']:20s} "
                f"effect={row['effect_size']:.6f}\n"
            )

        f.write("\n")

        f.write(
            "TOP RANDOM FOREST FEATURES\n"
        )

        for row in rf_rows:

            f.write(
                f"{row['rank']:2d}. "
                f"Feature {row['feature_index']:3d} "
                f"{row['feature_name']:20s} "
                f"importance={row['importance']:.6f}\n"
            )

        f.write("\n")

        f.write(
            "PCA\n"
        )

        f.write(
            f"PC1 variance: "
            f"{explained[0] * 100:.4f}%\n"
        )

        f.write(
            f"PC2 variance: "
            f"{explained[1] * 100:.4f}%\n\n"
        )

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            interpretation + "\n\n"
        )

        f.write(
            "RECOMMENDATION\n"
        )

        f.write(
            recommendation + "\n"
        )

    print(
        f"Text report saved: {txt_path}"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_header(
        "ANALYSIS COMPLETE"
    )

    print(
        f"Tuey training samples: "
        f"{len(X_tuey_train)}"
    )

    print(
        f"Bay training samples:  "
        f"{len(X_bay_train)}"
    )

    print(
        f"Tuey test samples:     "
        f"{len(X_tuey_test)}"
    )

    print(
        f"Bay test samples:      "
        f"{len(X_bay_test)}"
    )

    print()

    print(
        f"Tuey ↔ Bay centroid distance: "
        f"{centroid_distance:.4f}"
    )

    print(
        f"Separation ratio: "
        f"{separation_ratio:.4f}"
    )

    print(
        f"Binary test accuracy: "
        f"{binary_accuracy * 100:.2f}%"
    )

    print()

    print(
        "Reports:"
    )

    print(
        f"  {txt_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {feature_csv}"
    )

    print(
        f"  {rf_csv}"
    )

    print(
        f"  {pca_path}"
    )

    print()
    print(
        "DO NOT RETRAIN V3 YET."
    )

    print(
        "Send me the complete terminal output."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()