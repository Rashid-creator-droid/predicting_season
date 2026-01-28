from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from config import BATCH_SIZE


def get_dataloaders(
    train_dir,
    val_dir,
    train_transforms,
    val_transforms,
    batch_size=BATCH_SIZE,
):
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader