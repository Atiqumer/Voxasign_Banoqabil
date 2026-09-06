import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from v2_preprocessing import load_and_scale_v2_features


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output" / "psl_static_model_v2.keras"
SCALER_PATH = BASE_DIR / "output" / "psl_v2_scaler.npz"

X_TEST_PATH = BASE_DIR / "data" / "landmarks" / "v2" / "X_test_v2.npy"
Y_TEST_PATH = BASE_DIR / "data" / "landmarks" / "y_test.npy"

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "metadata"
    / "psl_static_metadata.csv"
)

CLASS_MAP_PATH = BASE_DIR / "output" / "class_map.json"

OUTPUT_DIR = BASE_DIR / "reports" / "tuey_daal_samples"

CSV_PATH = OUTPUT_DIR / "tuey_daal_test_samples.csv"
SUBJECT_CSV_PATH = OUTPUT_DIR / "tuey_daal_subject_analysis.csv"
REPORT_PATH = OUTPUT_DIR / "tuey_daal_confusion_report.txt"
JSON_PATH = OUTPUT_DIR / "tuey_daal_analysis.json"

VIS_DIR = OUTPUT_DIR / "visualizations"

# Classes we are investigating
TUEY_NAME = "Tuey"
DAAL_NAME = "Daal"


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{name}: {path}"
        )

    print(f"OK: {path}")


def load_class_map():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle either:
    # {"Tuey": 29, "Daal": 7}
    #
    # or:
    # {"0": "1-Hay", "1": "Ain", ...}
    #
    # or potentially {"classes": [...]}

    if isinstance(data, dict):

        if "classes" in data:
            classes = data["classes"]

            if isinstance(classes, list):
                id_to_class = {
                    i: str(name)
                    for i, name in enumerate(classes)
                }
            else:
                id_to_class = {}

        else:
            # Determine whether dictionary is name -> id
            # or id -> name
            name_to_id = {}
            id_to_name = {}

            for key, value in data.items():
                try:
                    key_int = int(key)
                except (ValueError, TypeError):
                    key_int = None

                if isinstance(value, int):
                    name_to_id[str(key)] = int(value)

                elif isinstance(value, str) and key_int is not None:
                    id_to_name[key_int] = value

            if name_to_id:
                id_to_class = {
                    class_id: class_name
                    for class_name, class_id in name_to_id.items()
                }

            elif id_to_name:
                id_to_class = id_to_name

            else:
                id_to_class = {}

    else:
        raise ValueError("Unsupported class_map.json format.")

    if not id_to_class:
        raise ValueError(
            "Could not determine class mapping from class_map.json."
        )

    return id_to_class


def find_test_metadata(metadata):
    """
    The landmark arrays contain only successful landmark extractions.

    Therefore metadata rows must correspond to the successfully extracted
    samples, not simply the raw metadata rows.

    We use split == 'test' and then remove failed images using the same
    failed_images.json information if available.
    """

    test_metadata = metadata[
        metadata["split"].astype(str).str.lower() == "test"
    ].copy()

    failed_path = BASE_DIR / "data" / "landmarks" / "failed_images.json"

    if failed_path.exists():

        print(f"Found failed-image record: {failed_path}")

        with open(failed_path, "r", encoding="utf-8") as f:
            failed = json.load(f)

        failed_test = failed.get("test", [])

        failed_paths = set()

        for item in failed_test:

            if isinstance(item, dict):
                path = item.get("path")

                if path:
                    failed_paths.add(
                        str(Path(path)).replace("\\", "/").lower()
                    )

            elif isinstance(item, str):
                failed_paths.add(
                    str(Path(item)).replace("\\", "/").lower()
                )

        if failed_paths:

            normalized = (
                test_metadata["path"]
                .astype(str)
                .map(lambda x: str(Path(x)).replace("\\", "/").lower())
            )

            before = len(test_metadata)

            test_metadata = test_metadata[
                ~normalized.isin(failed_paths)
            ].copy()

            removed = before - len(test_metadata)

            print(
                f"Removed {removed} failed test images from metadata."
            )

    test_metadata = test_metadata.reset_index(drop=True)

    return test_metadata


def normalize_path(path):
    return str(path).replace("\\", "/").lower()


def locate_image(path_string):
    """
    Try to locate the original image.

    Metadata paths can be absolute or relative.
    """

    if not path_string:
        return None

    original = Path(str(path_string))

    candidates = []

    if original.is_absolute():
        candidates.append(original)

    else:
        candidates.append(BASE_DIR / original)

        # Also try current project directory
        candidates.append(BASE_DIR / "data" / original)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Search by filename if direct path doesn't work
    filename = original.name

    if filename:
        try:
            matches = list(BASE_DIR.rglob(filename))

            if matches:
                return matches[0]

        except Exception:
            pass

    return None


def save_image_copy(row, destination_dir):

    source_path = locate_image(row["path"])

    if source_path is None:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)

    index = int(row["test_index"])
    true_class = str(row["true_class"])
    predicted_class = str(row["predicted_class"])
    subject_id = str(row["subject_id"])

    safe_name = (
        f"idx_{index:03d}"
        f"__subject_{subject_id}"
        f"__true_{true_class}"
        f"__pred_{predicted_class}"
        f"__{source_path.name}"
    )

    destination = destination_dir / safe_name

    try:
        shutil.copy2(source_path, destination)
        return str(destination)

    except Exception as e:
        print(
            f"WARNING: Could not copy image {source_path}: {e}"
        )
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------------

    print_header(
        "VoxaSign PSL — Tuey vs Daal Test Sample Analysis"
    )

    print()
    print("This script DOES NOT retrain the model.")
    print("It analyzes existing V2 predictions only.")

    # ------------------------------------------------------------------------
    # CHECK FILES
    # ------------------------------------------------------------------------

    print_header("CHECKING REQUIRED FILES")

    check_file(MODEL_PATH, "V2 model")
    check_file(SCALER_PATH, "V2 scaler")
    check_file(X_TEST_PATH, "X_test_v2")
    check_file(Y_TEST_PATH, "y_test")
    check_file(METADATA_PATH, "metadata")
    check_file(CLASS_MAP_PATH, "class_map")

    # ------------------------------------------------------------------------
    # CLASS MAP
    # ------------------------------------------------------------------------

    print_header("LOADING CLASS MAP")

    id_to_class = load_class_map()

    class_to_id = {
        name: int(class_id)
        for class_id, name in id_to_class.items()
    }

    if TUEY_NAME not in class_to_id:
        raise ValueError("Tuey not found in class_map.json.")

    if DAAL_NAME not in class_to_id:
        raise ValueError("Daal not found in class_map.json.")

    tuey_id = class_to_id[TUEY_NAME]
    daal_id = class_to_id[DAAL_NAME]

    print(f"Tuey = {tuey_id}")
    print(f"Daal = {daal_id}")

    # ------------------------------------------------------------------------
    # LOAD TEST DATA
    # ------------------------------------------------------------------------

    print_header("LOADING TEST DATA")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    if len(X_test) != len(y_test):
        raise RuntimeError(
            "X_test and y_test have different numbers of samples."
        )

    # ------------------------------------------------------------------------
    # LOAD METADATA
    # ------------------------------------------------------------------------

    print_header("LOADING TEST METADATA")

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Total metadata rows: {len(metadata)}")
    print(f"Metadata columns: {metadata.columns.tolist()}")

    required_columns = [
        "path",
        "class",
        "subject_id",
        "source",
        "filename",
        "split",
    ]

    for column in required_columns:
        if column not in metadata.columns:
            raise ValueError(
                f"Required metadata column missing: {column}"
            )

    test_metadata = find_test_metadata(metadata)

    print(
        f"Usable test metadata rows after failed-image filtering: "
        f"{len(test_metadata)}"
    )

    if len(test_metadata) != len(X_test):
        print()
        print("WARNING:")
        print(
            f"Metadata rows = {len(test_metadata)}"
        )
        print(
            f"Test samples  = {len(X_test)}"
        )
        print()
        print(
            "The metadata/test ordering may not match exactly."
        )
        print(
            "The script will continue, but inspect the mapping carefully."
        )

    # ------------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------------

    print_header("LOADING V2 MODEL")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # ------------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------------

    print_header("GENERATING TEST PREDICTIONS")

    X_test_scaled = load_and_scale_v2_features(
        X_test,
        SCALER_PATH,
    )

    print("Applied saved training-only V2 scaler.")

    probabilities = model.predict(
        X_test_scaled,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    overall_accuracy = np.mean(
        predictions == y_test
    )

    print()
    print(
        f"Overall V2 test accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------------------
    # BUILD SAMPLE RECORDS
    # ------------------------------------------------------------------------

    print_header("BUILDING TEST SAMPLE RECORDS")

    records = []

    for i in range(len(X_test)):

        true_id = int(y_test[i])
        pred_id = int(predictions[i])

        true_name = id_to_class.get(
            true_id,
            f"Class_{true_id}"
        )

        pred_name = id_to_class.get(
            pred_id,
            f"Class_{pred_id}"
        )

        if i < len(test_metadata):

            meta = test_metadata.iloc[i]

            record = {
                "test_index": i,
                "true_id": true_id,
                "true_class": true_name,
                "predicted_id": pred_id,
                "predicted_class": pred_name,
                "correct": bool(true_id == pred_id),
                "confidence": float(confidence[i]),
                "path": str(meta["path"]),
                "class_metadata": str(meta["class"]),
                "subject_id": str(meta["subject_id"]),
                "source": str(meta["source"]),
                "filename": str(meta["filename"]),
                "split": str(meta["split"]),
            }

        else:

            record = {
                "test_index": i,
                "true_id": true_id,
                "true_class": true_name,
                "predicted_id": pred_id,
                "predicted_class": pred_name,
                "correct": bool(true_id == pred_id),
                "confidence": float(confidence[i]),
                "path": "",
                "class_metadata": "",
                "subject_id": "",
                "source": "",
                "filename": "",
                "split": "test",
            }

        records.append(record)

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------------
    # FILTER TUEY / DAAL
    # ------------------------------------------------------------------------

    print_header("TUEY / DAAL TEST SAMPLES")

    target_mask = (
        df["true_class"].isin([TUEY_NAME, DAAL_NAME])
        |
        df["predicted_class"].isin([TUEY_NAME, DAAL_NAME])
    )

    target_df = df[target_mask].copy()

    print(
        f"Total samples involving Tuey or Daal: "
        f"{len(target_df)}"
    )

    # ------------------------------------------------------------------------
    # DIRECT CONFUSION
    # ------------------------------------------------------------------------

    print_header("TUEY ↔ DAAL CONFUSION")

    tuey_to_daal = df[
        (df["true_class"] == TUEY_NAME)
        &
        (df["predicted_class"] == DAAL_NAME)
    ]

    daal_to_tuey = df[
        (df["true_class"] == DAAL_NAME)
        &
        (df["predicted_class"] == TUEY_NAME)
    ]

    tuey_correct = df[
        (df["true_class"] == TUEY_NAME)
        &
        (df["predicted_class"] == TUEY_NAME)
    ]

    daal_correct = df[
        (df["true_class"] == DAAL_NAME)
        &
        (df["predicted_class"] == DAAL_NAME)
    ]

    print(
        f"Tuey → Tuey : {len(tuey_correct)}"
    )

    print(
        f"Tuey → Daal : {len(tuey_to_daal)}"
    )

    print(
        f"Daal → Daal : {len(daal_correct)}"
    )

    print(
        f"Daal → Tuey : {len(daal_to_tuey)}"
    )

    # ------------------------------------------------------------------------
    # PRINT MISCLASSIFIED SAMPLES
    # ------------------------------------------------------------------------

    print_header("MISCLASSIFIED TUEY / DAAL SAMPLES")

    errors = pd.concat(
        [
            tuey_to_daal,
            daal_to_tuey
        ],
        ignore_index=True
    )

    if len(errors) == 0:

        print("No direct Tuey ↔ Daal errors found.")

    else:

        display_columns = [
            "test_index",
            "true_class",
            "predicted_class",
            "confidence",
            "subject_id",
            "source",
            "filename",
            "path",
        ]

        print(
            errors[display_columns].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # SUBJECT ANALYSIS
    # ------------------------------------------------------------------------

    print_header("SUBJECT-LEVEL ERROR ANALYSIS")

    target_subject_df = target_df.copy()

    if len(target_subject_df) > 0:

        subject_records = []

        for subject_id, group in target_subject_df.groupby(
            "subject_id"
        ):

            tuey_total = np.sum(
                group["true_class"] == TUEY_NAME
            )

            daal_total = np.sum(
                group["true_class"] == DAAL_NAME
            )

            tuey_to_daal_count = np.sum(
                (group["true_class"] == TUEY_NAME)
                &
                (group["predicted_class"] == DAAL_NAME)
            )

            daal_to_tuey_count = np.sum(
                (group["true_class"] == DAAL_NAME)
                &
                (group["predicted_class"] == TUEY_NAME)
            )

            subject_records.append(
                {
                    "subject_id": subject_id,
                    "total_target_samples": len(group),
                    "tuey_samples": int(tuey_total),
                    "daal_samples": int(daal_total),
                    "tuey_to_daal": int(tuey_to_daal_count),
                    "daal_to_tuey": int(daal_to_tuey_count),
                    "total_direct_errors": int(
                        tuey_to_daal_count
                        +
                        daal_to_tuey_count
                    ),
                }
            )

        subject_df = pd.DataFrame(subject_records)

        subject_df = subject_df.sort_values(
            "total_direct_errors",
            ascending=False
        )

        print(
            subject_df.to_string(
                index=False
            )
        )

        subject_df.to_csv(
            SUBJECT_CSV_PATH,
            index=False
        )

        print()
        print(
            f"Subject analysis saved: {SUBJECT_CSV_PATH}"
        )

    else:

        subject_df = pd.DataFrame()

        print(
            "No Tuey/Daal target samples available."
        )

    # ------------------------------------------------------------------------
    # SOURCE ANALYSIS
    # ------------------------------------------------------------------------

    print_header("SOURCE ANALYSIS")

    if len(target_df) > 0:

        source_table = (
            target_df
            .groupby(
                ["true_class", "source"]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        print(
            source_table.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # CONFIDENCE ANALYSIS
    # ------------------------------------------------------------------------

    print_header("CONFIDENCE ANALYSIS")

    if len(errors) > 0:

        print(
            f"Mean error confidence: "
            f"{errors['confidence'].mean() * 100:.2f}%"
        )

        print(
            f"Median error confidence: "
            f"{errors['confidence'].median() * 100:.2f}%"
        )

        print(
            f"Minimum error confidence: "
            f"{errors['confidence'].min() * 100:.2f}%"
        )

        print(
            f"Maximum error confidence: "
            f"{errors['confidence'].max() * 100:.2f}%"
        )

    # ------------------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------------------

    print_header("SAVING SAMPLE CSV")

    target_df.to_csv(
        CSV_PATH,
        index=False
    )

    print(
        f"Saved: {CSV_PATH}"
    )

    # ------------------------------------------------------------------------
    # COPY VISUAL SAMPLES
    # ------------------------------------------------------------------------

    print_header("COPYING ORIGINAL IMAGES")

    folders = {
        "tuey_correct": tuey_correct,
        "tuey_as_daal": tuey_to_daal,
        "daal_correct": daal_correct,
        "daal_as_tuey": daal_to_tuey,
    }

    copied_counts = {}

    for folder_name, folder_df in folders.items():

        destination = VIS_DIR / folder_name

        count = 0

        for _, row in folder_df.iterrows():

            result = save_image_copy(
                row,
                destination
            )

            if result is not None:
                count += 1

        copied_counts[folder_name] = count

        print(
            f"{folder_name:18s}: "
            f"{count}/{len(folder_df)} images copied"
        )

    # ------------------------------------------------------------------------
    # ALL TARGET IMAGE SAMPLES
    # ------------------------------------------------------------------------

    all_target_dir = VIS_DIR / "all_tuey_daal"

    copied_all = 0

    for _, row in target_df.iterrows():

        result = save_image_copy(
            row,
            all_target_dir
        )

        if result is not None:
            copied_all += 1

    print(
        f"all_tuey_daal      : "
        f"{copied_all}/{len(target_df)} images copied"
    )

    # ------------------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------------------------

    print_header("TUEY / DAAL CLASSIFICATION")

    target_ids = [
        tuey_id,
        daal_id
    ]

    target_mask_numeric = np.isin(
        y_test,
        target_ids
    )

    target_true = y_test[target_mask_numeric]
    target_pred = predictions[target_mask_numeric]

    if len(target_true) > 0:

        report = classification_report(
            target_true,
            target_pred,
            labels=target_ids,
            target_names=[
                TUEY_NAME,
                DAAL_NAME
            ],
            zero_division=0
        )

        print(report)

        cm = confusion_matrix(
            target_true,
            target_pred,
            labels=target_ids
        )

        print("Tuey / Daal confusion matrix:")
        print()
        print(
            pd.DataFrame(
                cm,
                index=[
                    "True Tuey",
                    "True Daal"
                ],
                columns=[
                    "Pred Tuey",
                    "Pred Daal"
                ]
            )
        )

    else:

        report = ""
        cm = np.zeros((2, 2), dtype=int)

    # ------------------------------------------------------------------------
    # ERROR CONCENTRATION
    # ------------------------------------------------------------------------

    print_header("ERROR CONCENTRATION")

    if len(errors) > 0:

        error_subject_counts = (
            errors
            .groupby("subject_id")
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            "Direct Tuey ↔ Daal errors by subject:"
        )

        print(
            error_subject_counts.to_string()
        )

        top_subject = error_subject_counts.index[0]
        top_count = int(
            error_subject_counts.iloc[0]
        )

        print()
        print(
            f"Most affected subject: "
            f"{top_subject}"
        )

        print(
            f"Direct errors from subject: "
            f"{top_count}"
        )

    else:

        top_subject = None
        top_count = 0

        print(
            "No direct Tuey ↔ Daal errors."
        )

    # ------------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------------

    print_header("INTERPRETATION")

    tuey_total = len(
        df[df["true_class"] == TUEY_NAME]
    )

    daal_total = len(
        df[df["true_class"] == DAAL_NAME]
    )

    tuey_recall = (
        len(tuey_correct) / tuey_total
        if tuey_total > 0
        else 0
    )

    daal_recall = (
        len(daal_correct) / daal_total
        if daal_total > 0
        else 0
    )

    print(
        f"Tuey test recall: "
        f"{tuey_recall * 100:.2f}%"
    )

    print(
        f"Daal test recall: "
        f"{daal_recall * 100:.2f}%"
    )

    print()

    if len(tuey_to_daal) > 0:
        print(
            f"Tuey → Daal errors: "
            f"{len(tuey_to_daal)}"
        )

    if len(daal_to_tuey) > 0:
        print(
            f"Daal → Tuey errors: "
            f"{len(daal_to_tuey)}"
        )

    print()

    if len(errors) == 0:

        interpretation = (
            "No direct Tuey/Daal confusion was observed "
            "in the V2 test predictions."
        )

    elif len(errors) <= 3:

        interpretation = (
            "Tuey/Daal confusion exists but is limited. "
            "Inspect the individual samples before changing "
            "the model."
        )

    else:

        interpretation = (
            "Tuey/Daal confusion is significant. "
            "The next step should be visual and subject-level "
            "inspection before V3 retraining."
        )

    print(interpretation)

    if top_subject is not None:

        print()

        print(
            "Important:"
        )

        print(
            "If errors are concentrated in a small number "
            "of subjects, investigate subject-specific "
            "variation before changing the architecture."
        )

    print()
    print(
        "DO NOT RETRAIN V3 YET."
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    print_header("SAVING JSON REPORT")

    json_data = {
        "version": "Tuey-Daal Analysis V1",
        "model": str(MODEL_PATH),
        "test_samples": int(len(X_test)),
        "overall_accuracy": float(overall_accuracy),

        "classes": {
            "Tuey": int(tuey_id),
            "Daal": int(daal_id),
        },

        "test_counts": {
            "Tuey": int(tuey_total),
            "Daal": int(daal_total),
        },

        "recall": {
            "Tuey": float(tuey_recall),
            "Daal": float(daal_recall),
        },

        "confusion": {
            "Tuey_to_Tuey": int(len(tuey_correct)),
            "Tuey_to_Daal": int(len(tuey_to_daal)),
            "Daal_to_Daal": int(len(daal_correct)),
            "Daal_to_Tuey": int(len(daal_to_tuey)),
        },

        "error_count": int(len(errors)),

        "error_subject_counts": {
            str(k): int(v)
            for k, v in error_subject_counts.items()
        }
        if len(errors) > 0
        else {},

        "top_error_subject": (
            str(top_subject)
            if top_subject is not None
            else None
        ),

        "top_error_subject_count": int(top_count),

        "copied_images": copied_counts,

        "interpretation": interpretation,
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_data,
            f,
            indent=4
        )

    print(
        f"Saved: {JSON_PATH}"
    )

    # ------------------------------------------------------------------------
    # TEXT REPORT
    # ------------------------------------------------------------------------

    print_header("SAVING TEXT REPORT")

    report_lines = []

    report_lines.append(
        "VoxaSign PSL — Tuey vs Daal Test Sample Analysis"
    )

    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append(
        f"Model: {MODEL_PATH}"
    )

    report_lines.append(
        f"Test samples: {len(X_test)}"
    )

    report_lines.append(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        "TUEY / DAAL RESULTS"
    )

    report_lines.append("-" * 70)

    report_lines.append(
        f"Tuey total: {tuey_total}"
    )

    report_lines.append(
        f"Tuey correct: {len(tuey_correct)}"
    )

    report_lines.append(
        f"Tuey recall: {tuey_recall * 100:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        f"Daal total: {daal_total}"
    )

    report_lines.append(
        f"Daal correct: {len(daal_correct)}"
    )

    report_lines.append(
        f"Daal recall: {daal_recall * 100:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        f"Tuey -> Daal: {len(tuey_to_daal)}"
    )

    report_lines.append(
        f"Daal -> Tuey: {len(daal_to_tuey)}"
    )

    report_lines.append("")

    report_lines.append(
        "DIRECT ERROR SUBJECT COUNTS"
    )

    report_lines.append("-" * 70)

    if len(errors) > 0:

        for subject, count in error_subject_counts.items():

            report_lines.append(
                f"{subject}: {count}"
            )

    else:

        report_lines.append(
            "No direct errors."
        )

    report_lines.append("")

    report_lines.append(
        "CLASSIFICATION REPORT"
    )

    report_lines.append("-" * 70)
    report_lines.append(report)

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append("-" * 70)
    report_lines.append(interpretation)

    report_lines.append("")

    report_lines.append(
        "Next step: visually inspect the copied Tuey/Daal "
        "samples and determine whether errors are related "
        "to image quality, hand pose, landmark representation, "
        "or subject variation."
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    print(
        f"Saved: {REPORT_PATH}"
    )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print_header("ANALYSIS COMPLETE")

    print(
        f"Tuey recall: {tuey_recall * 100:.2f}%"
    )

    print(
        f"Daal recall: {daal_recall * 100:.2f}%"
    )

    print(
        f"Tuey → Daal: {len(tuey_to_daal)}"
    )

    print(
        f"Daal → Tuey: {len(daal_to_tuey)}"
    )

    print(
        f"Total direct errors: {len(errors)}"
    )

    print()

    print("Reports:")
    print(f"  {CSV_PATH}")
    print(f"  {SUBJECT_CSV_PATH}")
    print(f"  {REPORT_PATH}")
    print(f"  {JSON_PATH}")

    print()

    print("Visual folders:")
    print(f"  {VIS_DIR / 'tuey_correct'}")
    print(f"  {VIS_DIR / 'tuey_as_daal'}")
    print(f"  {VIS_DIR / 'daal_correct'}")
    print(f"  {VIS_DIR / 'daal_as_tuey'}")

    print()
    print(
        "NEXT STEP:"
    )
    print(
        "Send me the COMPLETE terminal output from this script."
    )
    print(
        "Then we will decide whether V3 needs feature changes, "
        "specialist classification, or targeted data work."
    )


if __name__ == "__main__":
    main()
