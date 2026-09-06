from pathlib import Path
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = [
    "1-Hay", "Ain", "Alif", "Bay", "Byeh", "Chay", "Cyeh",
    "Daal", "Dal", "Dochahay", "Fay", "Gaaf", "Ghain", "Hamza",
    "Kaf", "Khay", "Kiaf", "Lam", "Meem", "Nuun", "Nuungh",
    "Pay", "Ray", "Say", "Seen", "Sheen", "Suad", "Taay",
    "Tay", "Tuey", "Wao", "Zaal", "Zaey", "Zay", "Zuad", "Zuey"
]

DAAL_ID = CLASS_NAMES.index("Daal")
TUEY_ID = CLASS_NAMES.index("Tuey")


DATA_DIR = Path("data/landmarks")
OUTPUT_DIR = Path("reports/landmark_analysis")


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path, allow_pickle=True)


def mean_pairwise_distance(X1, X2, max_pairs=50000):
    rng = np.random.default_rng(42)

    total = len(X1) * len(X2)

    if total <= max_pairs:
        i = np.repeat(np.arange(len(X1)), len(X2))
        j = np.tile(np.arange(len(X2)), len(X1))
    else:
        i = rng.integers(0, len(X1), max_pairs)
        j = rng.integers(0, len(X2), max_pairs)

    return float(np.mean(np.linalg.norm(X1[i] - X2[j], axis=1)))


def within_class_distance(X, max_pairs=50000):
    if len(X) < 2:
        return 0.0

    rng = np.random.default_rng(42)
    total = len(X) * (len(X) - 1) // 2

    if total <= max_pairs:
        i, j = np.triu_indices(len(X), k=1)
    else:
        i = rng.integers(0, len(X), max_pairs)
        j = rng.integers(0, len(X), max_pairs)

        same = i == j
        while np.any(same):
            j[same] = rng.integers(
                0, len(X), np.sum(same)
            )
            same = i == j

    return float(np.mean(np.linalg.norm(X[i] - X[j], axis=1)))


def evaluate_classifier(name, model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    model.fit(X, y)
    pred = model.predict(X)

    return {
        "name": name,
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "train_accuracy": float(
            accuracy_score(y, pred)
        ),
        "train_f1": float(
            f1_score(y, pred, average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=[DAAL_ID, TUEY_ID]
        ).tolist()
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("VoxaSign PSL Landmark Analysis")
    print("=" * 70)
    print()

    X_train = load("X_train.npy").astype(np.float32)
    y_train = load("y_train.npy").astype(int)

    X_val = load("X_validation.npy").astype(np.float32)
    y_val = load("y_validation.npy").astype(int)

    X_test = load("X_test.npy").astype(np.float32)
    y_test = load("y_test.npy").astype(int)

    print("DATASET SHAPES")
    print("-" * 70)
    print(f"Train:      X={X_train.shape} y={y_train.shape}")
    print(f"Validation: X={X_val.shape} y={y_val.shape}")
    print(f"Test:       X={X_test.shape} y={y_test.shape}")
    print()

    print("CLASS MAPPING")
    print("-" * 70)
    print(f"Daal = {DAAL_ID}")
    print(f"Tuey = {TUEY_ID}")
    print()

    # ------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------

    print("DAAL / TUEY COUNTS")
    print("-" * 70)

    for name, y in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test)
    ]:

        daal_count = int(np.sum(y == DAAL_ID))
        tuey_count = int(np.sum(y == TUEY_ID))

        print(
            f"{name:12s}: "
            f"Daal={daal_count}, "
            f"Tuey={tuey_count}"
        )

    print()

    # ------------------------------------------------------------
    # TRAIN Daal/Tuey
    # ------------------------------------------------------------

    train_mask = np.isin(
        y_train,
        [DAAL_ID, TUEY_ID]
    )

    X_dt = X_train[train_mask]
    y_dt = y_train[train_mask]

    daal_train = X_dt[y_dt == DAAL_ID]
    tuey_train = X_dt[y_dt == TUEY_ID]

    # ------------------------------------------------------------
    # Distance analysis
    # ------------------------------------------------------------

    daal_centroid = daal_train.mean(axis=0)
    tuey_centroid = tuey_train.mean(axis=0)

    centroid_distance = float(
        np.linalg.norm(
            daal_centroid - tuey_centroid
        )
    )

    daal_within = within_class_distance(
        daal_train
    )

    tuey_within = within_class_distance(
        tuey_train
    )

    between_distance = mean_pairwise_distance(
        daal_train,
        tuey_train
    )

    pooled_within = (
        daal_within + tuey_within
    ) / 2

    separation_ratio = (
        between_distance / pooled_within
        if pooled_within > 0
        else 0
    )

    print("LANDMARK DISTANCE ANALYSIS")
    print("-" * 70)
    print(
        f"Daal centroid ↔ Tuey centroid: "
        f"{centroid_distance:.4f}"
    )

    print(
        f"Daal within-class distance:    "
        f"{daal_within:.4f}"
    )

    print(
        f"Tuey within-class distance:    "
        f"{tuey_within:.4f}"
    )

    print(
        f"Daal ↔ Tuey mean distance:     "
        f"{between_distance:.4f}"
    )

    print(
        f"Separation ratio:               "
        f"{separation_ratio:.4f}"
    )

    print()

    # ------------------------------------------------------------
    # Classical classifiers
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_dt)

    models = [

        (
            "1-NN",
            KNeighborsClassifier(
                n_neighbors=1
            )
        ),

        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]

    results = {}

    print("DAAL / TUEY SEPARABILITY")
    print("-" * 70)

    for name, model in models:

        result = evaluate_classifier(
            name,
            model,
            X_scaled,
            y_dt
        )

        results[name] = result

        print(
            f"{name:22s}: "
            f"{result['cv_accuracy'] * 100:.2f}% "
            f"+/- "
            f"{result['cv_std'] * 100:.2f}%"
        )

    print()

    # ------------------------------------------------------------
    # Random Forest importance
    # ------------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_scaled,
        y_dt
    )

    importances = rf.feature_importances_

    top = np.argsort(
        importances
    )[::-1][:15]

    print("TOP LANDMARK FEATURES")
    print("-" * 70)

    feature_info = []

    for idx in top:

        landmark = idx // 3

        coordinate = [
            "x",
            "y",
            "z"
        ][idx % 3]

        importance = float(
            importances[idx]
        )

        feature_info.append({
            "feature": int(idx),
            "landmark": int(landmark),
            "coordinate": coordinate,
            "importance": importance
        })

        print(
            f"Feature {idx:2d} | "
            f"Landmark {landmark:2d} | "
            f"{coordinate} | "
            f"{importance:.5f}"
        )

    print()

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    plt.figure(
        figsize=(9, 7)
    )

    daal_mask = y_dt == DAAL_ID
    tuey_mask = y_dt == TUEY_ID

    plt.scatter(
        X_pca[daal_mask, 0],
        X_pca[daal_mask, 1],
        label="Daal",
        alpha=0.65
    )

    plt.scatter(
        X_pca[tuey_mask, 0],
        X_pca[tuey_mask, 1],
        label="Tuey",
        alpha=0.65
    )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    plt.title(
        "VoxaSign PSL — Daal vs Tuey"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    pca_path = (
        OUTPUT_DIR /
        "daal_tuey_pca.png"
    )

    plt.savefig(
        pca_path,
        dpi=180
    )

    plt.close()

    # ------------------------------------------------------------
    # Test-set binary separability
    # ------------------------------------------------------------

    test_mask = np.isin(
        y_test,
        [DAAL_ID, TUEY_ID]
    )

    X_test_dt = X_test[test_mask]
    y_test_dt = y_test[test_mask]

    X_test_scaled = scaler.transform(
        X_test_dt
    )

    print("UNSEEN TEST SUBJECT CHECK")
    print("-" * 70)

    for name, model in models:

        model.fit(
            X_scaled,
            y_dt
        )

        pred = model.predict(
            X_test_scaled
        )

        acc = accuracy_score(
            y_test_dt,
            pred
        )

        print(
            f"{name:22s}: "
            f"{acc * 100:.2f}%"
        )

    print()

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------

    best_cv = max(
        r["cv_accuracy"]
        for r in results.values()
    )

    if best_cv >= 0.95:

        interpretation = (
            "Daal and Tuey are strongly separable in the "
            "current 63-dimensional landmark representation. "
            "The information is present in the landmarks. "
            "The next V2 investigation should focus on the "
            "36-class decision boundary, model training, "
            "and confusion between visually similar classes."
        )

    elif best_cv >= 0.85:

        interpretation = (
            "Daal and Tuey are reasonably separable but have "
            "meaningful overlap. V2 should investigate both "
            "feature representation and classifier improvements."
        )

    else:

        interpretation = (
            "Daal and Tuey have substantial overlap in the "
            "current landmark representation. Increasing model "
            "capacity alone is unlikely to solve the problem. "
            "Feature representation, hand orientation, landmark "
            "quality, and additional high-quality data should "
            "be investigated."
        )

    print("INTERPRETATION")
    print("-" * 70)
    print(interpretation)
    print()

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    report = {

        "class_mapping": {
            str(i): name
            for i, name in enumerate(
                CLASS_NAMES
            )
        },

        "target_classes": {
            "Daal": DAAL_ID,
            "Tuey": TUEY_ID
        },

        "dataset_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_val.shape),
            "test": list(X_test.shape)
        },

        "distance_analysis": {
            "centroid_distance":
                centroid_distance,

            "daal_within_class":
                daal_within,

            "tuey_within_class":
                tuey_within,

            "between_class":
                between_distance,

            "separation_ratio":
                separation_ratio
        },

        "classifiers": results,

        "feature_importance":
            feature_info,

        "pca": {
            "pc1_variance":
                float(pca.explained_variance_ratio_[0]),

            "pc2_variance":
                float(pca.explained_variance_ratio_[1])
        },

        "interpretation":
            interpretation
    }

    # Remove fitted model objects if any accidentally appear.
    for result in report["classifiers"].values():
        result.pop("fitted_model", None)

    json_path = (
        OUTPUT_DIR /
        "landmark_analysis.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    txt_path = (
        OUTPUT_DIR /
        "landmark_analysis.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VoxaSign PSL Landmark Analysis\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Daal class ID: {DAAL_ID}\n"
        )

        f.write(
            f"Tuey class ID: {TUEY_ID}\n\n"
        )

        f.write(
            f"Daal centroid distance: "
            f"{centroid_distance:.6f}\n"
        )

        f.write(
            f"Daal within-class: "
            f"{daal_within:.6f}\n"
        )

        f.write(
            f"Tuey within-class: "
            f"{tuey_within:.6f}\n"
        )

        f.write(
            f"Between-class: "
            f"{between_distance:.6f}\n"
        )

        f.write(
            f"Separation ratio: "
            f"{separation_ratio:.6f}\n\n"
        )

        f.write(
            "Classifier results:\n"
        )

        for name, result in results.items():

            f.write(
                f"{name}: "
                f"{result['cv_accuracy']*100:.2f}% "
                f"+/- "
                f"{result['cv_std']*100:.2f}%\n"
            )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            interpretation + "\n"
        )

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"JSON: {json_path}"
    )

    print(
        f"TXT:  {txt_path}"
    )

    print(
        f"PCA:  {pca_path}"
    )


if __name__ == "__main__":
    main()