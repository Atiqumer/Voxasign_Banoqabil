"""Canonical preprocessing for the saved VoxaSign PSL V2 model.

The V2 model was trained on 115 engineered features standardized with a
``StandardScaler`` fitted on the training split.  All saved-model inference
must call this module rather than sending raw V2 features to the model.
"""

from pathlib import Path

import numpy as np


def load_and_scale_v2_features(
    features: np.ndarray,
    scaler_path: Path,
) -> np.ndarray:
    """Apply the persisted training-only V2 StandardScaler to features.

    The operation exactly matches ``evaluate_v2.py`` and handles any zero
    scale entries defensively, matching scikit-learn's effective behavior.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"V2 scaler not found: {scaler_path}")

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D V2 feature array, got shape {features.shape}."
        )

    with np.load(scaler_path) as scaler_data:
        required_keys = {"mean", "scale"}
        missing_keys = required_keys.difference(scaler_data.files)
        if missing_keys:
            raise KeyError(
                f"V2 scaler is missing required keys: {sorted(missing_keys)}"
            )

        mean = np.asarray(scaler_data["mean"], dtype=np.float32)
        scale = np.asarray(scaler_data["scale"], dtype=np.float32)

    if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
        raise ValueError(
            "V2 scaler mean and scale must be one-dimensional arrays "
            "with identical shapes."
        )

    if features.shape[1] != mean.shape[0]:
        raise ValueError(
            "V2 feature count does not match the saved scaler: "
            f"features={features.shape[1]}, scaler={mean.shape[0]}."
        )

    safe_scale = np.where(scale == 0, np.float32(1.0), scale)
    return (features - mean) / safe_scale
