import logging
import os
from datetime import datetime

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED = True
MODEL_NAME = "resnet18"

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 5
SEED = 42

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logger settings
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file_path = f"learning_{current_time}.log"
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logs_dir = os.path.join(BASE_DIR, "logs", log_file_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=logs_dir,
)
logger = logging.getLogger("logs") 