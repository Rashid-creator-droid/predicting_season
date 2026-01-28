import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 4
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.0005

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "last_model.pth")

CLASSES = ["winter", "spring", "summer", "autumn"]