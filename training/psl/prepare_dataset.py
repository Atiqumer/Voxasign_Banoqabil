from pathlib import Path
import re
import random
import pandas as pd

from config import (
    DATASET_ROOT,
    METADATA_DIR,
    PSL_CLASSES,
    RANDOM_SEED,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_base_class(folder_name: str):
    if folder_name.endswith("-Original"):
        return folder_name[:-9]

    if folder_name.endswith("-Augmented"):
        return folder_name[:-10]

    return folder_name


def extract_subject_id(filename: str):
    match = re.match(r"s(\d+)-", filename)

    if match:
        return match.group(1)

    return None


def collect_images():

    records = []

    for folder in DATASET_ROOT.iterdir():

        if not folder.is_dir():
            continue

        base_class = get_base_class(folder.name)

        if base_class not in PSL_CLASSES:
            continue

        if not (
            folder.name.endswith("-Original")
            or folder.name.endswith("-Augmented")
        ):
            continue

        source_type = (
            "original"
            if folder.name.endswith("-Original")
            else "augmented"
        )

        for file in folder.iterdir():

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            subject_id = extract_subject_id(file.name)

            if subject_id is None:
                print(f"WARNING: Could not extract subject: {file}")
                continue

            records.append(
                {
                    "path": str(file.resolve()),
                    "class": base_class,
                    "subject_id": subject_id,
                    "source": source_type,
                    "filename": file.name,
                }
            )

    return pd.DataFrame(records)


def find_complete_subjects(df):

    coverage = (
        df.groupby("subject_id")["class"]
        .nunique()
    )

    complete = sorted(
        coverage[
            coverage == len(PSL_CLASSES)
        ].index.tolist()
    )

    return complete


def create_subject_split(subjects):

    subjects = list(subjects)

    random.seed(RANDOM_SEED)

    random.shuffle(subjects)

    required = (
        TRAIN_SUBJECTS
        + VAL_SUBJECTS
        + TEST_SUBJECTS
    )

    if len(subjects) < required:
        raise ValueError(
            f"Need {required} subjects but only "
            f"{len(subjects)} complete subjects found."
        )

    subjects = subjects[:required]

    train = subjects[:TRAIN_SUBJECTS]

    val = subjects[
        TRAIN_SUBJECTS:
        TRAIN_SUBJECTS + VAL_SUBJECTS
    ]

    test = subjects[
        TRAIN_SUBJECTS + VAL_SUBJECTS:
    ]

    return train, val, test


def assign_split(row, train, val, test):

    subject = row["subject_id"]

    if subject in train:

        # Training can contain original + augmented.
        return "train"

    if subject in val:

        # Validation must use original only.
        if row["source"] == "original":
            return "validation"

        return "exclude"

    if subject in test:

        # Test must use original only.
        if row["source"] == "original":
            return "test"

        return "exclude"

    return "exclude"


def main():

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("VoxaSign PSL Dataset Preparation")
    print("=" * 70)

    df = collect_images()

    print(f"\nTotal images found: {len(df)}")

    complete_subjects = find_complete_subjects(df)

    print(
        f"Complete subjects: "
        f"{len(complete_subjects)}"
    )

    train, val, test = create_subject_split(
        complete_subjects
    )

    print("\nSubject split:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    df["split"] = df.apply(
        lambda row: assign_split(
            row,
            train,
            val,
            test
        ),
        axis=1
    )

    df = df[
        df["split"] != "exclude"
    ].copy()

    # Save complete metadata
    metadata_file = (
        METADATA_DIR /
        "psl_static_metadata.csv"
    )

    df.to_csv(
        metadata_file,
        index=False
    )

    # Save subject lists
    pd.DataFrame(
        {"subject_id": train}
    ).to_csv(
        METADATA_DIR / "train_subjects.csv",
        index=False
    )

    pd.DataFrame(
        {"subject_id": val}
    ).to_csv(
        METADATA_DIR / "validation_subjects.csv",
        index=False
    )

    pd.DataFrame(
        {"subject_id": test}
    ).to_csv(
        METADATA_DIR / "test_subjects.csv",
        index=False
    )

    print("\nImages per split:")

    print(
        df.groupby(
            ["split", "source"]
        ).size()
    )

    print("\nClasses per split:")

    print(
        df.groupby(
            ["split", "class"]
        ).size()
        .unstack(fill_value=0)
    )

    print(
        f"\nMetadata saved to:\n"
        f"{metadata_file}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()