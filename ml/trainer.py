import json
from typing import List

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from PIL import ImageFile

from .models.models import get_model
from .loaders import DatasetManager
from .transforms import get_train_transforms, get_val_transforms
from .config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TRAIN_DIR,
    VAL_DIR,
    BEST_MODEL_PATH,
    DEVICE,
    NUM_CLASSES,
    MODEL_NAME,
    PRETRAINED,
    EARLY_STOPPING_PATIENCE,
    logger
)


class Trainer:
    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        device: torch.device = DEVICE,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        train_dir: str = TRAIN_DIR,
        val_dir: str = VAL_DIR,
        best_model_path: str = BEST_MODEL_PATH,
        seed: int = SEED,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.best_model_path = best_model_path
        self.early_stopping_patience = early_stopping_patience

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model = get_model(model_name, num_classes, pretrained).to(self.device)

        dataset_manager = DatasetManager(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size
        )
        self.train_loader, self.val_loader = dataset_manager.get_dataloaders(
            get_train_transforms(),
            get_val_transforms()
        )

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_val_acc = 0.0


    def train(self):
        train_loss_list: List = []
        val_loss_list: List = []
        train_acc_list: List = []
        val_acc_list: List = []

        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} - train"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / max(1, len(self.train_loader))
            train_acc = correct / max(1, total)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            self.model.eval()
            val_running_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} - val"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_running_loss / max(1, len(self.val_loader))
            val_acc = correct / max(1, total)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}] Train Acc{train_acc:.4f} Val Acc{val_acc:.4f}",
                epoch + 1,
                self.epochs,
                train_acc,
                val_acc,
            )
            print(f"Epoch [{epoch + 1}/{self.epochs}] Train Acc{train_acc:.4f} Val Acc{val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"Saved best model to {self.best_model_path} (val_acc={val_acc:.4f})")
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stopping_patience and no_improve >= self.early_stopping_patience:
                logger.info(f"Early stopping")
                break

        dataset = ImageFolder(root=TRAIN_DIR)
        with open("class_to_idx.json", "w") as f:
            json.dump(dataset.class_to_idx, f)



def quick_train(learning_rate: float = LEARNING_RATE, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    trainer = Trainer(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    trainer.train()

if __name__ == "__main__":
    quick_train()
    
