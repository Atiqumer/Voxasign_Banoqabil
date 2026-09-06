"""Export non-model PSL V2 assets for TensorFlow.js browser inference."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
OUTPUT_DIR = ROOT / "output"
WEB_DIR = PROJECT_ROOT / "web_model" / "psl_v2"


def main() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(OUTPUT_DIR / "psl_v2_scaler.npz") as scaler:
        payload = {
            "version": "PSL-V2",
            "feature_count": 115,
            "mean": scaler["mean"].astype(float).tolist(),
            "scale": scaler["scale"].astype(float).tolist(),
        }

    with open(WEB_DIR / "scaler.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    with open(OUTPUT_DIR / "class_map.json", "r", encoding="utf-8") as source:
        class_map = json.load(source)
    with open(WEB_DIR / "class_map.json", "w", encoding="utf-8") as file:
        json.dump(class_map, file, indent=2)

    print(f"Saved PSL browser assets to: {WEB_DIR}")


if __name__ == "__main__":
    main()
