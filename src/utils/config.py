from pathlib import Path
import torch

# ----------------------------------------------------
# Project Paths
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "models"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Training
# ----------------------------------------------------

IMAGE_SIZE = 512

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-4

NUM_CLASSES = 5

NUM_WORKERS = 0

SEED = 42

# ----------------------------------------------------
# Device
# ----------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)