import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.resnet import build_model
from loaders import get_dataloaders
from transforms import get_train_transforms, get_val_transforms
from config import NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_DIR, VAL_DIR, BEST_MODEL_PATH, LAST_MODEL_PATH, DEVICE


def train():
    
    device = DEVICE

    model = build_model(num_classes=NUM_CLASSES).to(device)

    train_loader, val_loader = get_dataloaders(
        TRAIN_DIR,
        VAL_DIR,
        get_train_transforms(),
        get_val_transforms(),
        batch_size=BATCH_SIZE
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total


        model.eval()
        correct, total = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        torch.save(model.state_dict(), LAST_MODEL_PATH)

if __name__ == "__main__":
    train()