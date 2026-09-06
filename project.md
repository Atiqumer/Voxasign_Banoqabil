# VoxaSign Project Handoff

## Purpose

VoxaSign is a browser-based static sign recognizer with separate ASL and PSL modes. It performs landmark extraction and inference locally in the browser, then optionally builds text and speaks it through the Web Speech API.

## Current release state

- Branch: `main`
- Browser UI: ASL and PSL selector in `studio.html`
- PSL V2 browser assets: `web_model/psl_v2/`
- ASL browser assets: `web_model/model.json` and `web_model/weights.json`
- PSL V2 test evaluation: 94.23% accuracy, macro F1 93.71%, weighted F1 94.08% on 416 held-out UAlpha40 samples.

## Model contracts

### ASL

- 21 MediaPipe landmarks → 63 wrist-relative, max-normalized values.
- Labels: A-Z with an internal `Blank` class.
- Browser hides low-confidence and Blank predictions.

### PSL V2

- 21 MediaPipe landmarks → 115 features → saved standard scaler → 36-class model.
- Feature order is implemented in both `training/psl/feature_engineering.py` and `psl_v2_features.js`; these must stay aligned.
- Browser class map and scaler are in `web_model/psl_v2/`.
- Browser-safe Base64 weight assets are used because some Windows download managers intercept `.bin` requests from localhost.

## Known limitations

- Static signs only; dynamic or motion-defined gestures need a separate temporal model.
- PSL Tuey (ط) and Daal (ڈ) remain a known confusion pair. Do not claim they are fixed.
- A tested Mendeley PSL image dataset could not be used because MediaPipe detected no hands.
- A tested 7,242-image PSL JPG dataset produced only 2.39% accuracy under the UAlpha40 label mapping, indicating incompatible sign conventions. Do not merge it into V2.
- The repository has an ASL deployment model but no included ASL training/test dataset, so its per-letter accuracy is not currently reproducible.

## How to run

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/studio.html`, hard-refresh after frontend changes, and select the intended language mode.

## Safe next work

1. Preserve the verified PSL V2 model as the release baseline.
2. Find or collect PSL data verified against the same UAlpha40/FESF sign convention before retraining Tuey/Daal.
3. Restore or obtain the ASL training/test dataset, then produce a per-letter evaluation before modifying the ASL model.
4. Build dynamic-sign recognition separately with landmark sequences; do not mix it into static V2.

## Git hygiene

- Keep source code, model assets required by the browser, scalers, class maps, and evaluated model outputs.
- Do not commit generated checkpoints, report visualizations, or temporary audit CSVs; `.gitignore` covers them.
- Keep `VOXASIGN_PROJECT_CONTEXT.md` as the detailed historical record. This file is the concise current-state handoff.
- Netlify builds only the generated `deploy/` folder. It contains the browser runtime assets, not training data or checkpoints.
