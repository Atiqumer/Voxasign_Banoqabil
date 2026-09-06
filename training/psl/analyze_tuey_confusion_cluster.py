import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "landmarks"
V2_DIR = DATA_DIR / "v2"
METADATA_DIR = BASE_DIR / "data" / "metadata"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "reports" / "tuey_cluster_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP_PATH = OUTPUT_DIR / "class_map.json"

X_TRAIN_PATH = V2_DIR / "X_train_v2.npy"
X_VAL_PATH = V2_DIR / "X_validation_v2.npy"
X_TEST_PATH = V2_DIR / "X_test_v2.npy"

Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
Y_VAL_PATH = DATA_DIR / "y_validation.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"

METADATA_PATH = METADATA_DIR / "psl_static_metadata.csv"


# Target confusion cluster
TARGET_CLASSES = [
    "Tuey",
    "Bay",
    "Daal",
    "Tay",
    "Chay",
    "Say",
]


# ============================================================================
# HELPERS
# ============================================================================

def separator(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    print(f"OK: {path}")


def load_class_map():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # Handle either:
    # {"Tuey": 29, ...}
    # or {"29": "Tuey", ...}
    if all(isinstance(v, int) for v in class_map.values()):
        name_to_id = class_map
        id_to_name = {v: k for k, v in class_map.items()}
    else:
        id_to_name = {int(k): v for k, v in class_map.items()}
        name_to_id = {v: int(k) for k, v in class_map.items()}

    return name_to_id, id_to_name


def feature_name(index):
    """
    V2 has 115 features.

    First 63:
        21 landmarks × 3 coordinates
        x, y, z

    Remaining features are engineered V2 features.
    """

    if index < 63:
        landmark = index // 3
        coord = ["x", "y", "z"][index % 3]

        return f"Landmark {landmark:2d} {coord}"

    return f"V2 Feature {index}"


def safe_effect_size(x1, x2):
    """
    Cohen-style standardized difference.
    """

    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    pooled = np.sqrt((var1 + var2) / 2)

    if pooled == 0:
        return 0.0

    return abs(mean1 - mean2) / pooled


def pairwise_distance_stats(X):
    """
    Calculate centroid and within-class distances.
    """

    centroid = np.mean(X, axis=0)

    distances = np.linalg.norm(X - centroid, axis=1)

    return {
        "centroid": centroid,
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    separator("VoxaSign PSL — Tuey Confusion Cluster Analysis")

    print("\nTarget cluster:\n")

    for cls in TARGET_CLASSES:
        print(f"    {cls}")

    # ------------------------------------------------------------------------
    # CHECK FILES
    # ------------------------------------------------------------------------

    separator("CHECKING REQUIRED FILES")

    required_files = [
        X_TRAIN_PATH,
        X_VAL_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_VAL_PATH,
        Y_TEST_PATH,
        CLASS_MAP_PATH,
    ]

    for path in required_files:
        check_file(path)

    # ------------------------------------------------------------------------
    # CLASS MAP
    # ------------------------------------------------------------------------

    separator("LOADING CLASS MAP")

    name_to_id, id_to_name = load_class_map()

    print(f"Number of classes: {len(name_to_id)}")

    target_ids = {}

    print("\nTarget class IDs:")

    for cls in TARGET_CLASSES:

        if cls not in name_to_id:
            raise ValueError(
                f"Class '{cls}' not found in class_map.json"
            )

        target_ids[cls] = name_to_id[cls]

        print(
            f"{cls:<10} = {target_ids[cls]}"
        )

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    separator("LOADING V2 LANDMARK DATA")

    X_train = np.load(X_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)
    X_test = np.load(X_TEST_PATH)

    y_train = np.load(Y_TRAIN_PATH)
    y_val = np.load(Y_VAL_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(
        f"Train:      X={X_train.shape} y={y_train.shape}"
    )

    print(
        f"Validation: X={X_val.shape} y={y_val.shape}"
    )

    print(
        f"Test:       X={X_test.shape} y={y_test.shape}"
    )

    print(f"\nFeature count: {X_train.shape[1]}")

    # ------------------------------------------------------------------------
    # FILTER TARGET CLUSTER
    # ------------------------------------------------------------------------

    separator("FILTERING TARGET CLUSTER")

    target_id_list = list(target_ids.values())

    train_mask = np.isin(y_train, target_id_list)
    val_mask = np.isin(y_val, target_id_list)
    test_mask = np.isin(y_test, target_id_list)

    X_train_cluster = X_train[train_mask]
    y_train_cluster = y_train[train_mask]

    X_val_cluster = X_val[val_mask]
    y_val_cluster = y_val[val_mask]

    X_test_cluster = X_test[test_mask]
    y_test_cluster = y_test[test_mask]

    print(
        f"Train cluster samples:      {len(y_train_cluster)}"
    )

    print(
        f"Validation cluster samples: {len(y_val_cluster)}"
    )

    print(
        f"Test cluster samples:       {len(y_test_cluster)}"
    )

    # ------------------------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------------------------

    print("\nCluster distribution:")

    for cls in TARGET_CLASSES:

        cid = target_ids[cls]

        train_count = np.sum(y_train_cluster == cid)
        val_count = np.sum(y_val_cluster == cid)
        test_count = np.sum(y_test_cluster == cid)

        print(
            f"{cls:<10} "
            f"Train={train_count:3d} | "
            f"Val={val_count:3d} | "
            f"Test={test_count:3d}"
        )

    # ------------------------------------------------------------------------
    # FEATURE EFFECT ANALYSIS
    # ------------------------------------------------------------------------

    separator("FEATURE SEPARABILITY ANALYSIS")

    effects = []

    for feature_idx in range(X_train_cluster.shape[1]):

        values_by_class = []

        for cls in TARGET_CLASSES:

            cid = target_ids[cls]

            values = X_train_cluster[
                y_train_cluster == cid,
                feature_idx
            ]

            values_by_class.append(values)

        # Mean pairwise effect
        pair_effects = []

        for i in range(len(values_by_class)):
            for j in range(i + 1, len(values_by_class)):

                effect = safe_effect_size(
                    values_by_class[i],
                    values_by_class[j]
                )

                pair_effects.append(effect)

        mean_effect = np.mean(pair_effects)

        effects.append(
            (
                feature_idx,
                mean_effect
            )
        )

    effects.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTOP 25 FEATURES BY BETWEEN-CLASS EFFECT")
    print("-" * 78)

    for rank, (feature_idx, effect) in enumerate(
        effects[:25],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Feature {feature_idx:3d} | "
            f"{feature_name(feature_idx):<28} | "
            f"Effect={effect:.5f}"
        )

    # ------------------------------------------------------------------------
    # STANDARDIZE
    # ------------------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_cluster
    )

    X_test_scaled = scaler.transform(
        X_test_cluster
    )

    # ------------------------------------------------------------------------
    # CLASSIFIER FUNCTION
    # ------------------------------------------------------------------------

    def evaluate_classifier(name, model):

        model.fit(
            X_train_scaled,
            y_train_cluster
        )

        predictions = model.predict(
            X_test_scaled
        )

        accuracy = accuracy_score(
            y_test_cluster,
            predictions
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test_cluster,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        print(f"\n--- {name} ---")
        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"F1       : {f1 * 100:.2f}%")

        return {
            "model": model,
            "predictions": predictions,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------------
    # TEST CLASSIFIERS
    # ------------------------------------------------------------------------

    separator("6-CLASS SEPARABILITY")

    classifiers = {}

    classifiers["1-NN"] = evaluate_classifier(
        "1-NN",
        KNeighborsClassifier(
            n_neighbors=1
        )
    )

    classifiers["Logistic Regression"] = evaluate_classifier(
        "Logistic Regression",
        LogisticRegression(
            max_iter=3000,
            C=1.0
        )
    )

    classifiers["SVM RBF"] = evaluate_classifier(
        "SVM RBF",
        SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale"
        )
    )

    classifiers["Random Forest"] = evaluate_classifier(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )

    # ------------------------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------------------------

    separator("CROSS-VALIDATION ON TRAINING CLUSTER")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_models = {
        "1-NN": KNeighborsClassifier(
            n_neighbors=1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            C=1.0
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

    cv_results = {}

    for name, model in cv_models.items():

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train_cluster,
            cv=cv,
            scoring="accuracy"
        )

        mean_score = scores.mean()
        std_score = scores.std()

        cv_results[name] = {
            "mean": float(mean_score),
            "std": float(std_score),
        }

        print(
            f"{name:<22}: "
            f"{mean_score * 100:.2f}% +/- "
            f"{std_score * 100:.2f}%"
        )

    best_cv_model = max(
        cv_results,
        key=lambda x: cv_results[x]["mean"]
    )

    print(
        f"\nBest CV model: {best_cv_model}"
    )

    # ------------------------------------------------------------------------
    # DETAILED TEST RESULTS
    # ------------------------------------------------------------------------

    best_predictions = classifiers[
        best_cv_model
    ]["predictions"]

    separator(
        f"DETAILED TEST RESULTS — {best_cv_model}"
    )

    report = classification_report(
        y_test_cluster,
        best_predictions,
        labels=target_id_list,
        target_names=TARGET_CLASSES,
        zero_division=0
    )

    print(report)

    # ------------------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------------------

    separator("6 × 6 CONFUSION MATRIX")

    cm = confusion_matrix(
        y_test_cluster,
        best_predictions,
        labels=target_id_list
    )

    print(
        "      " +
        " ".join(
            f"{cls:>6}"
            for cls in TARGET_CLASSES
        )
    )

    for i, cls in enumerate(TARGET_CLASSES):

        row = " ".join(
            f"{value:6d}"
            for value in cm[i]
        )

        print(
            f"{cls:<6}{row}"
        )

    # ------------------------------------------------------------------------
    # PAIRWISE CONFUSION
    # ------------------------------------------------------------------------

    separator("PAIRWISE CONFUSION ANALYSIS")

    pairwise_confusions = []

    for i, true_cls in enumerate(TARGET_CLASSES):

        for j, pred_cls in enumerate(TARGET_CLASSES):

            if i == j:
                continue

            count = int(cm[i, j])

            if count > 0:

                pairwise_confusions.append(
                    {
                        "true": true_cls,
                        "predicted": pred_cls,
                        "count": count,
                    }
                )

    pairwise_confusions.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    if pairwise_confusions:

        for item in pairwise_confusions:

            print(
                f"{item['true']:<10} -> "
                f"{item['predicted']:<10} "
                f"{item['count']}"
            )

    else:
        print("No confusion found.")

    # ------------------------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # ------------------------------------------------------------------------

    separator("RANDOM FOREST FEATURE IMPORTANCE")

    rf = classifiers["Random Forest"]["model"]

    importances = rf.feature_importances_

    rf_features = []

    for idx, importance in enumerate(importances):

        rf_features.append(
            {
                "feature_index": idx,
                "feature_name": feature_name(idx),
                "importance": float(importance),
            }
        )

    rf_features.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    for rank, item in enumerate(
        rf_features[:25],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Feature {item['feature_index']:3d} | "
            f"{item['feature_name']:<28} | "
            f"Importance={item['importance']:.5f}"
        )

    rf_csv_path = (
        REPORT_DIR /
        "tuey_cluster_rf_features.csv"
    )

    pd.DataFrame(
        rf_features
    ).to_csv(
        rf_csv_path,
        index=False
    )

    print(
        f"\nRF feature importance saved: "
        f"{rf_csv_path}"
    )

    # ------------------------------------------------------------------------
    # PAIRWISE CSV
    # ------------------------------------------------------------------------

    pairwise_csv_path = (
        REPORT_DIR /
        "tuey_cluster_pairwise.csv"
    )

    pd.DataFrame(
        pairwise_confusions
    ).to_csv(
        pairwise_csv_path,
        index=False
    )

    print(
        f"Pairwise confusion saved: "
        f"{pairwise_csv_path}"
    )

    # ------------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------------

    separator("PCA ANALYSIS")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_test_scaled
    )

    print(
        f"PC1 explained variance: "
        f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
    )

    plt.figure(figsize=(10, 8))

    for cls in TARGET_CLASSES:

        cid = target_ids[cls]

        mask = y_test_cluster == cid

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=cls,
            alpha=0.75
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        "VoxaSign PSL — Tuey Confusion Cluster PCA"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    pca_path = (
        REPORT_DIR /
        "tuey_cluster_pca.png"
    )

    plt.savefig(
        pca_path,
        dpi=200
    )

    plt.close()

    print(
        f"PCA plot saved: {pca_path}"
    )

    # ------------------------------------------------------------------------
    # CONFUSION MATRIX IMAGE
    # ------------------------------------------------------------------------

    separator("SAVING CONFUSION MATRIX")

    plt.figure(figsize=(9, 8))

    plt.imshow(cm)

    plt.colorbar()

    plt.xticks(
        range(len(TARGET_CLASSES)),
        TARGET_CLASSES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(TARGET_CLASSES)),
        TARGET_CLASSES
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        "VoxaSign PSL — Tuey Confusion Cluster"
    )

    plt.tight_layout()

    cm_path = (
        REPORT_DIR /
        "tuey_cluster_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=200
    )

    plt.close()

    print(
        f"Confusion matrix saved: {cm_path}"
    )

    # ------------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------------

    separator("INTERPRETATION")

    best_test_accuracy = classifiers[
        best_cv_model
    ]["accuracy"]

    largest_confusion = (
        pairwise_confusions[0]
        if pairwise_confusions
        else None
    )

    print(
        f"Best classifier: {best_cv_model}"
    )

    print(
        f"Cross-validation accuracy: "
        f"{cv_results[best_cv_model]['mean'] * 100:.2f}%"
    )

    print(
        f"6-class test accuracy: "
        f"{best_test_accuracy * 100:.2f}%"
    )

    if largest_confusion:

        print(
            f"\nLargest confusion: "
            f"{largest_confusion['true']} -> "
            f"{largest_confusion['predicted']} "
            f"({largest_confusion['count']} samples)"
        )

    print(
        "\nThe six-class cluster should be investigated "
        "as a targeted specialist problem."
    )

    print(
        "\nThis supports using a targeted specialist "
        "classifier inside V3 rather than immediately "
        "retraining the entire 36-class model."
    )

    # ------------------------------------------------------------------------
    # JSON REPORT
    # ------------------------------------------------------------------------

    results = {

        "target_classes": TARGET_CLASSES,

        "target_ids": target_ids,

        "dataset": {
            "train_cluster_samples":
                int(len(y_train_cluster)),
            "validation_cluster_samples":
                int(len(y_val_cluster)),
            "test_cluster_samples":
                int(len(y_test_cluster)),
            "feature_count":
                int(X_train_cluster.shape[1]),
        },

        "distribution": {},

        "feature_effects": [
            {
                "feature_index": int(idx),
                "feature_name": feature_name(idx),
                "effect": float(effect),
            }
            for idx, effect in effects[:25]
        ],

        "test_results": {
            name: {
                "accuracy": value["accuracy"],
                "precision": value["precision"],
                "recall": value["recall"],
                "f1": value["f1"],
            }
            for name, value in classifiers.items()
        },

        "cross_validation": cv_results,

        "best_cv_model": best_cv_model,

        "confusion_matrix": cm.tolist(),

        "pairwise_confusions": pairwise_confusions,

        "random_forest_features": rf_features[:25],

        "pca": {
            "pc1_explained_variance":
                float(pca.explained_variance_ratio_[0]),
            "pc2_explained_variance":
                float(pca.explained_variance_ratio_[1]),
        },
    }

    for cls in TARGET_CLASSES:

        cid = target_ids[cls]

        results["distribution"][cls] = {
            "train": int(
                np.sum(y_train_cluster == cid)
            ),
            "validation": int(
                np.sum(y_val_cluster == cid)
            ),
            "test": int(
                np.sum(y_test_cluster == cid)
            ),
        }

    # ------------------------------------------------------------------------
    # SAVE JSON
    # ------------------------------------------------------------------------

    separator("SAVING JSON REPORT")

    json_path = (
        REPORT_DIR /
        "tuey_cluster_analysis.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"JSON saved: {json_path}"
    )

    # ------------------------------------------------------------------------
    # TEXT REPORT
    # ------------------------------------------------------------------------

    separator("SAVING TEXT REPORT")

    txt_path = (
        REPORT_DIR /
        "tuey_cluster_analysis.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VoxaSign PSL — Tuey Confusion Cluster Analysis\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "Target cluster:\n"
        )

        for cls in TARGET_CLASSES:
            f.write(
                f"  {cls}\n"
            )

        f.write("\n")

        f.write(
            f"Best model: {best_cv_model}\n"
        )

        f.write(
            f"CV accuracy: "
            f"{cv_results[best_cv_model]['mean'] * 100:.2f}%\n"
        )

        f.write(
            f"Test accuracy: "
            f"{best_test_accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification report:\n"
        )

        f.write(
            report
        )

        f.write(
            "\nPairwise confusion:\n"
        )

        for item in pairwise_confusions:

            f.write(
                f"{item['true']} -> "
                f"{item['predicted']}: "
                f"{item['count']}\n"
            )

    print(
        f"Text report saved: {txt_path}"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    separator("ANALYSIS COMPLETE")

    print(
        "Target cluster: "
        + ", ".join(TARGET_CLASSES)
    )

    print(
        f"Best model: {best_cv_model}"
    )

    print(
        f"Cross-validation: "
        f"{cv_results[best_cv_model]['mean'] * 100:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{best_test_accuracy * 100:.2f}%"
    )

    print("\nReports:")

    print(f"  {txt_path}")
    print(f"  {json_path}")
    print(f"  {pairwise_csv_path}")
    print(f"  {rf_csv_path}")
    print(f"  {pca_path}")
    print(f"  {cm_path}")

    print(
        "\nDO NOT RETRAIN V3 YET."
    )


if __name__ == "__main__":
    main()