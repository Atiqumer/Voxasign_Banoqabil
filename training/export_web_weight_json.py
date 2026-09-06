"""Create browser-safe JSON copies of VoxaSign TensorFlow.js weight binaries.

Some Windows download managers intercept .bin requests from localhost. The
application reads these Base64 JSON assets instead, then reconstructs the
identical ArrayBuffer in memory for TensorFlow.js.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_FILES = (
    PROJECT_ROOT / "web_model" / "group1-shard1of1.bin",
    PROJECT_ROOT / "web_model" / "psl_v2" / "group1-shard1of1.bin",
)


def main() -> None:
    for source in WEIGHT_FILES:
        if not source.exists():
            raise FileNotFoundError(f"Missing TensorFlow.js weights: {source}")

        target = source.with_name("weights.json")
        payload = {
            "encoding": "base64",
            "byte_length": source.stat().st_size,
            "data": base64.b64encode(source.read_bytes()).decode("ascii"),
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        print(f"Saved {target} ({payload['byte_length']} weight bytes)")


if __name__ == "__main__":
    main()
