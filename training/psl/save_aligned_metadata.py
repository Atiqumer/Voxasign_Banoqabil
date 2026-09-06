"""Save exact metadata mappings for the V2 landmark arrays."""

from pathlib import Path

import numpy as np
import pandas as pd

from metadata_alignment import aligned_split_metadata


ROOT = Path(__file__).resolve().parent
LANDMARK_DIR = ROOT / "data" / "landmarks"
V2_DIR = LANDMARK_DIR / "v2"
METADATA_PATH = ROOT / "data" / "metadata" / "psl_static_metadata.csv"
FAILED_IMAGES_PATH = LANDMARK_DIR / "failed_images.json"


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)

    for split in ("train", "validation", "test"):
        features = np.load(V2_DIR / f"X_{split}_v2.npy")
        labels = np.load(LANDMARK_DIR / f"y_{split}.npy")

        if len(features) != len(labels):
            raise ValueError(
                f"Feature and label counts differ for '{split}'."
            )

        aligned = aligned_split_metadata(
            metadata=metadata,
            split=split,
            expected_rows=len(features),
            failed_images_path=FAILED_IMAGES_PATH,
            labels=labels,
        )

        output_path = V2_DIR / f"{split}_metadata_v2.csv"
        aligned.to_csv(output_path, index=False)
        print(f"Saved {len(aligned)} rows: {output_path}")


if __name__ == "__main__":
    main()
