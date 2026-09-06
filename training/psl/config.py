from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CHANGE THIS to your actual UAlpha40 location
DATASET_ROOT = Path(
    r"C:\Users\atiqu\Downloads\Compressed\UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language_3\UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"
)

DATA_DIR = Path(__file__).resolve().parent / "data"

METADATA_DIR = DATA_DIR / "metadata"
LANDMARK_DIR = DATA_DIR / "landmarks"

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ============================================================
# DATASET
# ============================================================

PSL_CLASSES = [
    "1-Hay",
    "Ain",
    "Alif",
    "Bay",
    "Byeh",
    "Chay",
    "Cyeh",
    "Daal",
    "Dal",
    "Dochahay",
    "Fay",
    "Gaaf",
    "Ghain",
    "Hamza",
    "Kaf",
    "Khay",
    "Kiaf",
    "Lam",
    "Meem",
    "Nuun",
    "Nuungh",
    "Pay",
    "Ray",
    "Say",
    "Seen",
    "Sheen",
    "Suad",
    "Taay",
    "Tay",
    "Tuey",
    "Wao",
    "Zaal",
    "Zaey",
    "Zay",
    "Zuad",
    "Zuey",
]

NUM_CLASSES = len(PSL_CLASSES)


# ============================================================
# SUBJECT SPLIT
# ============================================================

# We have 76 complete subjects.
TRAIN_SUBJECTS = 56
VAL_SUBJECTS = 8
TEST_SUBJECTS = 12


# ============================================================
# MEDIAPIPE
# ============================================================

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3

INPUT_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK


# ============================================================
# TRAINING
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

VALIDATION_SPLIT = None