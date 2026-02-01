import logging
import os
from datetime import datetime

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "resnet18"

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 5
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_SET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_SET_DIR, "train")
VAL_DIR = os.path.join(DATA_SET_DIR, "validation")
TEST_DIR = os.path.join(DATA_SET_DIR, "test")

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")
CLASS_TO_IDX_PATH = os.path.join(WEIGHTS_DIR, "class_to_idx.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file_path = f"learning_{current_time}.log"
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logs_dir = os.path.join(BASE_DIR, "logs", log_file_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=logs_dir,
)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logger = logging.getLogger("logs") 
