from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dane
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "MO_106"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"

# Wyniki / artefakty
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
STATISTICAL_TESTS_DIR = OUTPUTS_DIR / "statistical_tests"

# Split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 13

# Obrazy
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224

# Trening
BATCH_SIZE = 32
NUM_WORKERS = 4

# Normalizacja ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model
NUM_CLASSES = 106
MODEL_NAME = "resnet18"
USE_PRETRAINED = False

# Tuned training
TUNED_NUM_EPOCHS = 25
TUNED_LEARNING_RATE = 3e-4
TUNED_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6