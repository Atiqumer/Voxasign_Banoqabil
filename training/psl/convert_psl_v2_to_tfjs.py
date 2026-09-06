"""Export the fixed VoxaSign PSL V2 Dense model to TensorFlow.js Layers.

This intentionally avoids the tensorflowjs Python package: its current
converter pulls unrelated JAX and Decision Forest dependencies that conflict
with the verified training environment. It supports only the saved V2
architecture (Dense, BatchNormalization, and Dropout layers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_PATH = ROOT / "output" / "psl_static_model_v2.keras"
WEB_DIR = PROJECT_ROOT / "web_model" / "psl_v2"


def dense_config(layer: tf.keras.layers.Dense) -> dict:
    return {
        "class_name": "Dense",
        "config": {
            "name": layer.name,
            "trainable": True,
            "dtype": "float32",
            "units": layer.units,
            "activation": layer.activation.__name__,
            "use_bias": layer.use_bias,
        },
    }


def batch_normalization_config(
    layer: tf.keras.layers.BatchNormalization,
) -> dict:
    return {
        "class_name": "BatchNormalization",
        "config": {
            "name": layer.name,
            "trainable": True,
            "dtype": "float32",
            "axis": layer.axis,
            "momentum": layer.momentum,
            "epsilon": layer.epsilon,
            "center": layer.center,
            "scale": layer.scale,
        },
    }


def dropout_config(layer: tf.keras.layers.Dropout) -> dict:
    return {
        "class_name": "Dropout",
        "config": {
            "name": layer.name,
            "trainable": True,
            "dtype": "float32",
            "rate": layer.rate,
            "noise_shape": None,
            "seed": layer.seed,
        },
    }


def model_topology(model: tf.keras.Sequential) -> dict:
    layers = [
        {
            "class_name": "InputLayer",
            "config": {
                "batch_input_shape": [None, 115],
                "dtype": "float32",
                "sparse": False,
                "ragged": False,
                "name": "input_layer",
            },
        }
    ]

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layers.append(dense_config(layer))
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layers.append(batch_normalization_config(layer))
        elif isinstance(layer, tf.keras.layers.Dropout):
            layers.append(dropout_config(layer))
        else:
            raise TypeError(f"Unsupported V2 layer: {layer.__class__.__name__}")

    return {
        "keras_version": "2.21.0",
        "backend": "tensorflow",
        "model_config": {
            "class_name": "Sequential",
            "config": {"name": model.name, "layers": layers},
        },
    }


def write_weights(model: tf.keras.Sequential, output_path: Path) -> list[dict]:
    specs: list[dict] = []

    with open(output_path, "wb") as binary_file:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                continue

            if isinstance(layer, tf.keras.layers.Dense):
                names = ["kernel", "bias"]
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                names = ["gamma", "beta", "moving_mean", "moving_variance"]
            else:
                raise TypeError(f"Unsupported V2 layer: {layer.__class__.__name__}")

            values = layer.get_weights()
            if len(values) != len(names):
                raise ValueError(f"Unexpected weight count for layer '{layer.name}'.")

            for name, value in zip(names, values):
                value = np.asarray(value, dtype="<f4")
                specs.append(
                    {
                        "name": f"{layer.name}/{name}",
                        "shape": list(value.shape),
                        "dtype": "float32",
                    }
                )
                binary_file.write(value.tobytes(order="C"))

    return specs


def main() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if model.input_shape != (None, 115) or model.output_shape != (None, 36):
        raise ValueError(
            f"Unexpected V2 model shape: input={model.input_shape}, "
            f"output={model.output_shape}."
        )

    weight_file = "group1-shard1of1.bin"
    weight_specs = write_weights(model, WEB_DIR / weight_file)
    payload = {
        "format": "layers-model",
        "generatedBy": "VoxaSign PSL V2 direct exporter",
        "convertedBy": "VoxaSign",
        "modelTopology": model_topology(model),
        "weightsManifest": [{"paths": [weight_file], "weights": weight_specs}],
    }

    with open(WEB_DIR / "model.json", "w", encoding="utf-8") as file:
        json.dump(payload, file)

    print(f"Saved PSL TensorFlow.js model to: {WEB_DIR}")
    print(f"Weights: {len(weight_specs)} tensors")


if __name__ == "__main__":
    main()
