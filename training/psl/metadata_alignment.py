"""Exact metadata alignment for landmark arrays.

Landmark extraction skips images where MediaPipe finds no usable hand.  This
module reproduces that skip list so each returned metadata row maps to the
same row in an ``X_*.npy`` and ``y_*.npy`` array.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_path(path: object) -> str:
    """Normalize Windows/POSIX path strings for reliable comparison."""
    return str(path).replace("\\", "/").strip().lower()


def _failed_paths_for_split(
    failed_images_path: Path,
    split: str,
) -> set[str]:
    if not failed_images_path.exists():
        return set()

    with open(failed_images_path, "r", encoding="utf-8") as file:
        failed_images = json.load(file)

    if not isinstance(failed_images, dict):
        raise ValueError(
            "failed_images.json must map split names to failed-image lists."
        )

    entries = failed_images.get(split, [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Failed-image entries for split '{split}' must be a list."
        )

    paths = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("path"):
            paths.add(normalize_path(entry["path"]))
        elif isinstance(entry, str):
            paths.add(normalize_path(entry))

    return paths


def aligned_split_metadata(
    metadata: pd.DataFrame,
    split: str,
    expected_rows: int,
    failed_images_path: Path,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return metadata in the exact successful-extraction array order."""
    if "split" not in metadata.columns or "path" not in metadata.columns:
        raise ValueError("Metadata must contain 'split' and 'path' columns.")

    split = split.lower()
    candidates = metadata[
        metadata["split"].astype(str).str.strip().str.lower() == split
    ].copy()

    failed_paths = _failed_paths_for_split(failed_images_path, split)
    if failed_paths:
        candidates = candidates[
            ~candidates["path"].map(normalize_path).isin(failed_paths)
        ].copy()

    candidates = candidates.reset_index(drop=True)

    if len(candidates) != expected_rows:
        raise ValueError(
            f"Metadata alignment failed for '{split}': expected "
            f"{expected_rows} successful rows, found {len(candidates)}."
        )

    candidates.insert(0, "array_index", np.arange(expected_rows, dtype=int))

    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != expected_rows:
            raise ValueError(
                f"Label count for '{split}' does not match expected rows."
            )
        candidates["class_id"] = labels.astype(int)

    return candidates
