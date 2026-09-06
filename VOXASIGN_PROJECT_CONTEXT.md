# VoxaSign --- Complete Project Context & Codex Handoff

> **Purpose:** Give Codex the development history, current state,
> verified results, unresolved issues, and next steps for the VoxaSign
> FYP.
>
> **Current product direction:** A real-time **Android application**
> using on-device MediaPipe hand landmarks and a verified TensorFlow
> Lite model.

------------------------------------------------------------------------

## 1. Project Overview

**VoxaSign** is a Pakistan Sign Language (PSL) recognition Final Year
Project.

The project recognizes PSL alphabet hand signs and converts them into
useful output such as:

-   recognized letters/text
-   accumulated words
-   text-to-speech
-   real-time camera feedback

The project started around Python/OpenCV-style experimentation and is
now intended to finish as an Android application.

### Intended Android inference pipeline

``` text
Android Camera
    ↓
MediaPipe Hand Landmarker
    ↓
Exact VoxaSign Feature Extraction
    ↓
Exact Training Preprocessing / Scaling
    ↓
TensorFlow Lite Model
    ↓
PSL Class Prediction
    ↓
Prediction Stabilization
    ↓
Text / Word Builder
    ↓
Android Text-to-Speech
```

Prefer **on-device inference**. Do not introduce a Python/FastAPI
backend unless a future feature genuinely requires it.

------------------------------------------------------------------------

## 2. Current Development Environment

Main training directory:

``` text
D:\voxasign\training\psl
```

Development has been done on Windows using VS Code / PowerShell and a
Python virtual environment.

TensorFlow currently runs on CPU on native Windows. TensorFlow's warning
that CUDA GPU is unavailable on native Windows for TensorFlow \>= 2.11
is known and is not a blocker.

------------------------------------------------------------------------

## 3. Dataset

Dataset: **UAlpha40**, used for Urdu alphabet / Pakistan Sign Language
recognition.

The current static-sign model uses **36 classes**.

Static class names currently used:

``` text
1-Hay
Ain
Alif
Bay
Byeh
Chay
Cyeh
Daal
Dal
Dochahay
Fay
Gaaf
Ghain
Hamza
Kaf
Khay
Kiaf
Lam
Meem
Nuun
Nuungh
Pay
Ray
Say
Seen
Sheen
Suad
Taay
Tay
Tuey
Wao
Zaal
Zaey
Zay
Zuad
Zuey
```

Subject split previously established:

``` text
Train subjects:      56
Validation subjects: 8
Test subjects:       12
Total complete:      76
```

Earlier image counts:

``` text
Original training images:  1,942
Augmented training images: 8,573
Training metadata rows:   10,515
Validation metadata rows:    288
Test metadata rows:          432
```

Landmark extraction failed for some images, so final landmark arrays
contain fewer rows.

------------------------------------------------------------------------

## 4. Metadata

Metadata file:

``` text
data/metadata/psl_static_metadata.csv
```

Columns:

``` text
path
class
subject_id
source
filename
split
```

Examples of original filenames include:

``` text
s0251-13daal.jpg
s0252-13daal.jpg
s0251-23tuey.JPG
s0252-23tuey.JPG
```

The labels stored in `y_train.npy`, `y_validation.npy`, and `y_test.npy`
are integer class IDs, not class-name strings.

Known IDs:

``` text
Daal = 7
Tuey = 29
Bay = 3
Tay = 28
Chay = 5
Say = 23
```

------------------------------------------------------------------------

## 5. Current Project Structure

Observed important structure:

``` text
training/psl/
├── checkpoints/
├── data/
│   ├── landmarks/
│   │   ├── v2/
│   │   │   ├── feature_info.json
│   │   │   ├── X_train_v2.npy
│   │   │   ├── X_validation_v2.npy
│   │   │   └── X_test_v2.npy
│   │   ├── y_train.npy
│   │   ├── y_validation.npy
│   │   ├── y_test.npy
│   │   └── failed_images.json
│   └── metadata/
│       └── psl_static_metadata.csv
├── output/
│   ├── class_map.json
│   ├── psl_static_model_v1.keras
│   ├── psl_static_model_v2.keras
│   ├── psl_v2_config.json
│   └── psl_v2_scaler.npz
└── reports/
    ├── landmark_analysis/
    ├── model_comparison/
    ├── v2/
    ├── tuey_analysis/
    ├── tuey_bay_analysis/
    ├── tuey_cluster_analysis/
    ├── tuey_daal_samples/
    ├── tuey_daal_geometry/
    └── tuey_daal_geometry_errors/
```

### Base landmark arrays

``` text
X_train.npy       = (9244, 63)
X_validation.npy  = (277, 63)
X_test.npy        = (416, 63)
```

The 63 base features represent:

``` text
21 MediaPipe hand landmarks × x/y/z = 63
```

### V2 arrays

``` text
X_train_v2.npy       = (9244, 115)
X_validation_v2.npy  = (277, 115)
X_test_v2.npy        = (416, 115)
```

The first 63 V2 features correspond to the landmark coordinates. The
remaining features are engineered features.

------------------------------------------------------------------------

## 6. V1 Model

Official V1 test results:

``` text
Accuracy:    89.42%
Macro F1:    89.24%
Weighted F1: 89.56%
Loss:        0.2606
```

Known weak classes included:

``` text
Daal: 45.45%
Tuey: 54.55%
```

Several classes achieved 100%.

Saved model:

``` text
output/psl_static_model_v1.keras
```

------------------------------------------------------------------------

## 7. V2 Model

Saved artifacts:

``` text
output/psl_static_model_v2.keras
output/psl_v2_config.json
output/psl_v2_scaler.npz
```

Authoritative V2 report:

``` text
reports/v2/evaluation_results_v2.json
```

Official V2 results:

``` text
Accuracy:    94.23%
Macro F1:    93.71%
Weighted F1: 94.08%
Test samples: 416
```

Comparison:

``` text
V1 accuracy: 89.42%
V2 accuracy: 94.23%
Improvement: +4.81 percentage points
```

Daal:

``` text
V1: 45.45%
V2: 81.82%
Change: +36.36 percentage points
```

Tuey:

``` text
V1: 54.55%
V2 official: 27.27%
```

Tuey became the primary class requiring investigation.

------------------------------------------------------------------------

## 8. CRITICAL V2 Inference Issue

This is currently the most important unresolved technical issue.

Several later analysis scripts load:

``` text
output/psl_static_model_v2.keras
```

and directly call prediction on:

``` text
data/landmarks/v2/X_test_v2.npy
```

Those scripts report:

``` text
Overall test accuracy: 19.71%
```

This is incompatible with the official V2 evaluation:

``` text
94.23%
```

Therefore, the direct diagnostic inference pipeline is almost certainly
not reproducing the preprocessing used during V2 training/evaluation.

A likely cause is failure to apply:

``` text
output/psl_v2_scaler.npz
```

or another preprocessing transformation.

**Do not guess the scaler format. Inspect the training/evaluation code
and NPZ keys first.**

### Required acceptance test

Create one canonical saved-model evaluation pipeline and verify:

``` text
saved V2 model
+ saved V2 preprocessing
+ X_test_v2
→ approximately 94.23% accuracy on 416 test samples
```

Until this works:

-   do not trust the 19.71% diagnostic result;
-   do not use those raw neural predictions to justify V3;
-   do not port the current raw V2 inference implementation to Android.

------------------------------------------------------------------------

## 9. Initial Daal vs Tuey Landmark Analysis

An initial binary analysis produced:

``` text
Train: Daal=280, Tuey=246
Validation: Daal=8, Tuey=8
Test: Daal=11, Tuey=11
```

Geometry/statistical values:

``` text
Daal centroid ↔ Tuey centroid: 0.2019
Daal within-class distance:    1.2125
Tuey within-class distance:    1.0450
Daal ↔ Tuey mean distance:     1.1502
Separation ratio:              1.0190
```

Training cross-validation:

``` text
1-NN:                90.30% ± 1.53%
Logistic Regression: 76.81% ± 4.60%
Random Forest:       89.54% ± 3.02%
```

Unseen test-subject binary check:

``` text
1-NN:                59.09%
Logistic Regression: 59.09%
Random Forest:       59.09%
```

Interpretation:

Training separation was considerably better than unseen-subject
generalization, suggesting subject/pose variation rather than simply
insufficient classifier capacity.

------------------------------------------------------------------------

## 10. Tuey vs Bay Analysis

Targeted V2-feature analysis:

``` text
Train: Tuey=246, Bay=327
Validation: Tuey=8, Bay=8
Test: Tuey=11, Bay=12
Feature count: 115
```

Separation ratio:

``` text
2.3197
```

Training CV:

``` text
1-NN:                100%
Logistic Regression: 100%
Random Forest:       100%
```

Binary held-out test:

``` text
100%
```

Confusion matrix:

``` text
[[12, 0],
 [ 0,11]]
```

Interpretation:

Tuey and Bay are highly separable in V2 feature space. Therefore, later
neural confusion between Tuey and Bay does not by itself prove that the
features are incapable of distinguishing the signs.

------------------------------------------------------------------------

## 11. Six-Class Confusion Cluster

Classes:

``` text
Tuey
Bay
Daal
Tay
Chay
Say
```

Data:

``` text
Train:      1517
Validation: 45
Test:       64
```

Held-out test results:

``` text
1-NN:                84.38%
Logistic Regression: 81.25%
SVM RBF:             84.38%
Random Forest:       82.81%
```

Training CV:

``` text
1-NN:                95.85%
Logistic Regression: 94.66%
SVM RBF:             93.87%
Random Forest:       95.98%
```

A Random Forest test confusion matrix showed the difficult residual
pair:

``` text
Tuey -> Daal: 6
Daal -> Tuey: 5
```

while Bay, Tay, Chay, and Say were separated correctly in that
experiment.

This narrowed the targeted investigation toward **Tuey ↔ Daal**.

------------------------------------------------------------------------

## 12. Tuey/Daal Test Image Investigation

A previous script successfully aligned test metadata after removing 16
failed test images:

``` text
432 raw test metadata rows
- 16 failed landmark extractions
= 416 usable test rows
```

It identified examples such as:

``` text
Tuey -> Daal
subject 294
s0294-23tuey.JPG

Daal -> Tuey
subject 289
s0289-13daal.jpg

Daal -> Tuey
subject 334
s0334-13daal.JPG
```

Visual inspection did not provide sufficient evidence to declare these
labels wrong.

The more defensible interpretation was subject/pose/orientation
variation.

These neural predictions were produced by a script affected by the
unresolved raw V2 inference mismatch, so the images remain useful for
qualitative investigation but the diagnostic neural metrics are not
authoritative.

------------------------------------------------------------------------

## 13. Geometry Experiments

Geometry features were created from the first 63 landmark features.

Feature concepts included:

-   wrist-relative landmark coordinates
-   normalized landmark coordinates
-   fingertip distances
-   finger segment lengths
-   wrist-to-tip distances
-   joint angles
-   thumb joint angles

One experiment produced approximately:

``` text
Geometry features: 175
V2 + Geometry:     290
```

A later implementation produced:

``` text
Geometry features: 174
V2 + Geometry:     289
```

This difference must be standardized before production use.

### Earlier held-out Tuey/Daal experiment

V2 features:

``` text
Logistic: 59.09%
SVM:      54.55%
RF:       50.00%
1-NN:     68.18%
```

Geometry only:

``` text
Logistic: 59.09%
SVM:      54.55%
RF:       63.64%
1-NN:     59.09%
```

V2 + Geometry:

``` text
Logistic: 59.09%
SVM:      54.55%
RF:       54.55%
1-NN:     72.73%
```

Best experimental configuration:

``` text
V2 + Geometry + 1-NN

Accuracy:  72.73%
Precision: 72.73%
Recall:    72.73%
F1:        72.73%
```

The test contains only 22 Tuey/Daal samples, so this is experimental
evidence rather than final validation.

------------------------------------------------------------------------

## 14. Latest Geometry Error Analysis

Latest run of:

``` text
analyze_tuey_daal_geometry_errors.py
```

successfully executed.

Shapes:

``` text
Train:      X=(9244, 115)
Validation: X=(277, 115)
Test:       X=(416, 115)

Geometry train: (9244, 174)
Geometry test:  (416, 174)

Combined train: (9244, 289)
Combined test:  (416, 289)
```

Target samples:

``` text
Daal: 11
Tuey: 11
Total: 22
```

Raw V2 diagnostic:

``` text
Accuracy:  68.18%
Precision: 75.00%
Recall:    54.55%
F1:        63.16%
```

V2 + Geometry 1-NN:

``` text
Accuracy:  63.64%
Precision: 66.67%
Recall:    54.55%
F1:        60.00%
```

Difference:

``` text
-3.16 F1 percentage points
```

Error categories:

``` text
V2 correct / Geometry correct: 5
V2 correct / Geometry wrong:   3
V2 wrong / Geometry correct:   9
Both wrong:                    5
```

Geometry direct confusion:

``` text
Tuey -> Daal: 5
Daal -> Tuey: 3
```

The experiment shows geometry changes useful decisions, but the current
geometry 1-NN classifier is **not better overall** than the raw V2
diagnostic on this subset.

Do not automatically add geometry as a production correction layer.

------------------------------------------------------------------------

## 15. Interesting Geometry Features

Latest effect-size analysis found potentially discriminative features
such as:

``` text
tip_distance_16_20
tip_distance_4_16
joint_angle_8
relative_landmark_17_z
relative_landmark_18_z
relative_landmark_13_z
relative_landmark_19_z
normalized_landmark_9_y
wrist_to_tip_16
tip_distance_0_16
joint_angle_3
```

These suggest that relative finger geometry and depth/orientation may
contain useful Tuey/Daal information.

However, do not select production features solely from this 22-sample
test subset.

------------------------------------------------------------------------

## 16. Metadata Alignment Problem

Latest geometry-error script reported:

``` text
Metadata rows: 11235
Test metadata candidates: 432
X_test rows: 416
```

It correctly refused to guess positional alignment.

Therefore, in that run:

``` text
Subject ID unavailable
Image paths unavailable
```

Another earlier script successfully removed the 16 failed test images
and obtained exactly 416 aligned rows.

### Required improvement

Future extraction should save exact aligned metadata beside every array,
e.g.:

``` text
data/landmarks/v2/train_metadata_v2.csv
data/landmarks/v2/validation_metadata_v2.csv
data/landmarks/v2/test_metadata_v2.csv
```

Every metadata row must correspond exactly to the same row of `X_*.npy`
and `y_*.npy`.

This is needed for:

-   subject-aware validation
-   reliable error analysis
-   image copying
-   reproducibility
-   academic reporting

------------------------------------------------------------------------

## 17. Current Conclusions

The defensible conclusions are:

1.  V2 officially improved overall test accuracy from **89.42% to
    94.23%**.
2.  V2 officially improved macro F1 from **89.24% to 93.71%**.
3.  Daal improved strongly in the official V2 evaluation.
4.  Tuey became the main official V2 weakness.
5.  Tuey/Bay are separable in targeted V2-feature experiments.
6.  Tuey/Daal appear to be a more meaningful hard pair in
    subject-independent experiments.
7.  Geometry contains potentially useful information.
8.  Geometry is not sufficiently validated to become the final solution
    yet.
9.  The raw V2 inference scripts are currently inconsistent with the
    official V2 evaluation.
10. Metadata alignment still needs to be standardized.
11. **Do not retrain V3 yet.**

------------------------------------------------------------------------

## 18. Immediate Next Steps

### Step 1 --- Inspect V2 preprocessing

Inspect:

``` text
output/psl_v2_scaler.npz
output/psl_v2_config.json
data/landmarks/v2/feature_info.json
```

Also inspect the V2 training/evaluation source code.

Determine exactly how the scaler/preprocessing was applied.

### Step 2 --- Create canonical saved-model evaluation

Create one script such as:

``` text
evaluate_v2_saved_model.py
```

It should load:

-   `X_test_v2.npy`
-   `y_test.npy`
-   saved scaler/preprocessing
-   `psl_static_model_v2.keras`

Acceptance criterion:

``` text
~94.23% accuracy on 416 test samples
```

### Step 3 --- Fix metadata alignment

Make exact metadata-to-array mapping persistent.

### Step 4 --- Re-run authoritative error analysis

Only after canonical V2 inference works, repeat:

-   full confusion matrix
-   per-class metrics
-   Tuey errors
-   Daal errors
-   Tuey/Daal analysis
-   subject-level analysis

### Step 5 --- Decide whether V3 is needed

Potential options:

-   improved unified feature set
-   selected geometry features
-   targeted augmentation
-   class weighting
-   hard-example mining
-   architecture adjustment
-   specialist Tuey/Daal classifier

Do not choose one until validation supports it.

------------------------------------------------------------------------

## 19. Android App Plan

The final application should be developed after the inference contract
is verified.

### Phase 1 --- Freeze inference specification

Document:

``` text
MediaPipe landmark format
landmark order
feature order
feature count
normalization
scaling
model input shape
model output shape
class-map order
camera mirroring assumptions
```

### Phase 2 --- Convert model to TensorFlow Lite

Pipeline:

``` text
verified .keras
→ .tflite
→ Python TFLite test
→ compare Keras vs TFLite predictions
→ Android integration
```

Do not consider conversion complete merely because a `.tflite` file is
produced.

The TFLite model should reproduce essentially the same predictions as
the verified Keras model.

### Phase 3 --- Create Android project

Recommended:

-   Kotlin
-   Jetpack Compose
-   CameraX

### Phase 4 --- MediaPipe

Use MediaPipe Hand Landmarker to extract 21 hand landmarks in real time.

Check:

-   x/y/z ordering
-   handedness
-   mirroring
-   camera orientation
-   normalization
-   landmark ordering

### Phase 5 --- Kotlin feature extractor

Port the final Python feature extraction to Kotlin.

Create parity tests:

``` text
same saved landmark input
→ Python feature vector
→ Kotlin feature vector
```

They should match within a small numerical tolerance.

### Phase 6 --- TFLite inference

Bundle required assets:

``` text
model.tflite
class_map.json
scaler/preprocessing constants
```

Inference should run on-device.

### Phase 7 --- Prediction stabilization

Do not append a letter every video frame.

Use logic such as:

``` text
frame predictions
→ confidence threshold
→ short rolling window
→ majority/weighted vote
→ stable sign
→ cooldown/release
→ append letter
```

This prevents prediction flicker and repeated letters.

### Phase 8 --- Text and speech

Recognition screen can contain:

-   live camera preview
-   hand overlay
-   predicted PSL letter
-   confidence
-   accumulated text
-   delete
-   clear
-   speak

Use Android `TextToSpeech` for speech output.

------------------------------------------------------------------------

## 20. Android MVP

A good first MVP is:

``` text
Camera
+ hand landmark detection
+ verified feature extraction
+ TFLite prediction
+ stable PSL alphabet output
+ text accumulation
+ text-to-speech
```

Authentication, cloud sync, dashboards, etc. are optional unless
explicitly required by the FYP.

------------------------------------------------------------------------

## 21. Real-World Android Evaluation

Dataset accuracy alone is not sufficient.

Test the final app with:

-   unseen people
-   different backgrounds
-   different lighting
-   different hand-camera distances
-   different camera angles
-   supported left/right hands
-   no-hand frames
-   partially visible hands

Measure:

-   sign accuracy
-   false predictions
-   confidence
-   latency
-   FPS
-   stabilization delay

Record the Android device used.

------------------------------------------------------------------------

## 22. FYP Evidence to Preserve

Do not delete existing experiment artifacts.

Important reports include:

``` text
reports/v2/classification_report_v2.txt
reports/v2/confusion_matrix_v2.png
reports/v2/evaluation_results_v2.json
reports/model_comparison/v1_vs_v2.json
reports/tuey_analysis/
reports/tuey_bay_analysis/
reports/tuey_cluster_analysis/
reports/tuey_daal_samples/
reports/tuey_daal_geometry/
reports/tuey_daal_geometry_errors/
```

These can support:

-   methodology
-   feature engineering
-   model comparison
-   confusion analysis
-   ablation/targeted experiments
-   limitations
-   future work

------------------------------------------------------------------------

## 23. Recommended Final Evaluation Table

Eventually maintain something like:

  ----------------------------------------------------------------------------
  Version    Main change       Accuracy      Macro F1   Tuey Recall Status
  ---------- ------------ ------------- ------------- ------------- ----------
  V1         Baseline            89.42%        89.24%        54.55% Complete

  V2         Engineered          94.23%        93.71%        27.27% Best
             features                                               official
                                                                    model so
                                                                    far

  V3         TBD                    TBD           TBD           TBD Do not
                                                                    train yet

  TFLite     Final mobile           TBD           TBD           TBD Pending
             conversion                                             

  Android    Real-world             TBD           ---           TBD Pending
  live test  camera                                                 
  ----------------------------------------------------------------------------

Never fill TBD values without actually running the corresponding
evaluation.

------------------------------------------------------------------------

## 24. Codex Instructions

When continuing this project in Codex:

1.  Read this file before changing the ML pipeline.
2.  Inspect the actual repository files before assuming implementation
    details.
3.  Preserve V1 and V2 artifacts.
4.  Do not retrain V3 unless explicitly requested after validation.
5.  Treat `reports/v2/evaluation_results_v2.json` as the authoritative
    V2 evaluation until canonical saved-model reproduction is
    established.
6.  Resolve the **94.23% vs 19.71%** mismatch before Android model
    integration.
7.  Never fit preprocessing on the test set.
8.  Preserve subject-independent train/validation/test separation.
9.  Preserve class-map ordering.
10. Save new experiments under new filenames/directories.
11. Record configuration and seeds where relevant.
12. Do not guess metadata alignment.
13. Keep experimental geometry logic separate from production inference
    until validated.
14. Never report binary Tuey/Daal accuracy as overall 36-class accuracy.
15. Validate Keras → TFLite conversion numerically.
16. Verify Python ↔ Kotlin feature parity.
17. Prefer on-device Android inference.
18. Keep the final pipeline reproducible for the FYP report.

------------------------------------------------------------------------

## 25. Recommended Next Codex Prompt

After placing this file in the repository, use:

> Read `VOXASIGN_PROJECT_CONTEXT.md` completely first, then inspect the
> existing VoxaSign repository. Do not retrain anything yet. Our first
> priority is to reproduce the official V2 test accuracy of
> approximately 94.23% from the saved `psl_static_model_v2.keras`,
> `psl_v2_scaler.npz`, config, feature arrays, and existing
> training/evaluation code. Find the exact preprocessing used during V2
> training, explain why the recent direct inference scripts produce
> 19.71%, and create/fix a canonical saved-model evaluation script.
> Preserve all existing V1/V2 artifacts.

------------------------------------------------------------------------

## 26. Current Status

### Completed / substantially completed

-   dataset organization
-   subject-based splitting
-   MediaPipe landmark extraction
-   V1 training/evaluation
-   V2 feature engineering
-   V2 training/evaluation
-   V1 vs V2 comparison
-   per-class analysis
-   Tuey-focused investigation
-   Tuey/Bay separability experiment
-   confusion-cluster experiment
-   Tuey/Daal sample investigation
-   geometry experiments
-   diagnostic report generation

### Unresolved

-   canonical V2 inference reproduction
-   exact scaler/preprocessing verification
-   exact metadata ↔ landmark alignment
-   authoritative post-fix Tuey/Daal diagnosis
-   V3 decision

### Remaining product work

-   final model freeze
-   TFLite conversion
-   Keras/TFLite parity testing
-   Android project
-   CameraX
-   MediaPipe Android integration
-   Kotlin feature extraction
-   Python/Kotlin parity tests
-   TFLite inference
-   prediction stabilization
-   text/word builder
-   text-to-speech
-   live Android evaluation
-   final FYP documentation/demo

------------------------------------------------------------------------

## 27. Most Important Warning

**Official V2 accuracy is 94.23%, while some later direct saved-model
scripts report only 19.71%.**

Do not build Android inference around the incorrect diagnostic pipeline.

The production path must reproduce:

``` text
MediaPipe landmarks
        ↓
EXACT training feature extraction
        ↓
EXACT training preprocessing/scaling
        ↓
verified model
        ↓
correct class-map ordering
```

Only after the Python saved-model pipeline reproduces the official
result should it be ported to TFLite and Android.

------------------------------------------------------------------------

## 28. Final Product Vision

``` text
User opens VoxaSign
        ↓
Starts recognition
        ↓
Camera sees a PSL hand sign
        ↓
MediaPipe detects hand landmarks
        ↓
On-device model recognizes the sign
        ↓
Stable PSL letter appears
        ↓
Letters form text
        ↓
User can hear the text using speech
```

The final FYP should demonstrate both:

**ML quality** - controlled subject splits - reproducible
preprocessing - model comparisons - per-class metrics - error analysis

**Product quality** - usable Android camera application - real-time
on-device inference - stable predictions - text building - speech output

------------------------------------------------------------------------

**End of VoxaSign Codex handoff.**
