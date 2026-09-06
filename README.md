# VoxaSign

Browser-based, on-device static sign recognition for American Sign Language (ASL) and Pakistan Sign Language (PSL). VoxaSign uses MediaPipe hand landmarks and TensorFlow.js models; camera frames and uploaded photos stay in the browser.

## Current capabilities

| Mode | Classes | Input | Status |
| --- | --- | --- | --- |
| ASL | A-Z plus an internal Blank class | 63 normalized hand-landmark values | Existing browser model |
| PSL V2 | 36 static Urdu PSL signs | 115 landmark and geometry features | Verified V2 browser model |

The Studio provides live-camera and photo-upload modes, manual or auto word entry, confidence feedback, text-to-speech, backspace, clear, session stats, and camera switching on supported devices.

## Run the web app

Do not open the HTML files directly. Start a local server from the repository root:

```powershell
python -m http.server 8000
```

Then open [the Studio](http://localhost:8000/studio.html). Select **ASL** or **PSL** at the top of the page before using a camera or uploading an image.

If Internet Download Manager intercepts TensorFlow.js `.bin` files, VoxaSign uses the included `weights.json` assets instead. Do not open model-weight files in the browser directly.

## Inference pipelines

### ASL

```text
Image or webcam → MediaPipe 21 landmarks → wrist-relative 63 values
→ max-absolute normalization → ASL TensorFlow.js model → label/confidence
```

ASL hides low-confidence predictions and its Blank class. Automatic entry requires a stable high-confidence prediction held for 1.5 seconds.

### PSL V2

```text
Image or webcam → MediaPipe 21 landmarks → wrist-relative normalization
→ 115 features → saved V2 scaler → PSL TensorFlow.js model → label/confidence
```

The 115 PSL features are 63 normalized coordinates, 20 bone lengths, 10 joint angles, 5 wrist-to-tip distances, 10 fingertip distances, and 7 palm-geometry distances. PSL displays its top prediction even when uncertain, so a low score is visible instead of appearing as no result.

## PSL V2 quality and scope

The V2 model was evaluated on the held-out UAlpha40 static PSL test split:

- Accuracy: **94.23%** (416 samples)
- Macro F1: **93.71%**
- Weighted F1: **94.08%**

Known limitation: **Tuey (ط)** and **Daal (ڈ)** are a difficult pair. Tuey recall on the held-out test subset is low and is commonly confused with Daal. Treat low-confidence results for this pair as uncertain; do not present them as a corrected model behavior.

Both language modes are static-sign recognition only. Motion-defined signs require a separate sequence/video model and are not implemented yet.

## Project layout

```text
VoxaSign/
├── studio.html                         # Main dual-language browser UI
├── script.js                           # MediaPipe, model loading, inference
├── psl_v2_features.js                  # Browser PSL 115-feature contract
├── web_model/
│   ├── model.json                      # ASL TensorFlow.js model
│   ├── weights.json                    # ASL browser-safe weight asset
│   └── psl_v2/                         # PSL V2 model, scaler, classes, weights
├── training/
│   ├── export_web_weight_json.py       # Create browser-safe weight JSON files
│   └── psl/
│       ├── extract_landmarks.py        # Shared MediaPipe landmark extraction
│       ├── feature_engineering.py      # PSL V2 feature generation
│       ├── train_v2.py / evaluate_v2.py
│       ├── predict_image_v2.py         # One-image PSL V2 inference
│       ├── export_psl_web_assets.py    # Export scaler and class map
│       └── convert_psl_v2_to_tfjs.py   # Export V2 Keras model for the browser
├── signspeak_ui.py                     # Legacy Python ASL desktop UI
└── VOXASIGN_PROJECT_CONTEXT.md         # Detailed project history and handoff
```

## PSL developer workflow

The PSL training pipeline uses UAlpha40 data and a subject-aware split. Run these commands from an activated environment only when the verified V2 Keras model has changed:

```powershell
python training\psl\export_psl_web_assets.py
python training\psl\convert_psl_v2_to_tfjs.py
python training\export_web_weight_json.py
```

The final command creates JSON/Base64 versions of TensorFlow.js weights for local browser use on systems where download managers intercept `.bin` files.

To evaluate a single PSL image through the verified Python pipeline:

```powershell
python training\psl\predict_image_v2.py --image "C:\path\to\sign.jpg"
```

## Important engineering rules

- Keep the PSL 115-feature order and saved scaler together with the V2 model.
- Do not train on the test split or re-use generated report images as data.
- Do not merge a PSL dataset merely because class names match; verify landmark extraction and sign convention first.
- Generated checkpoints, audit reports, and report visualizations are ignored by Git.

## Technology

- TensorFlow.js
- MediaPipe Tasks Vision
- TensorFlow/Keras and NumPy for PSL training/evaluation
- OpenCV for Python image handling
- Web Speech API for browser text-to-speech

## License

See the repository license or project owner for licensing terms.
