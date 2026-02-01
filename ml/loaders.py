from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ml.config import BATCH_SIZE, TEST_DIR, TRAIN_DIR, VAL_DIR
from .transforms import get_pred_transforms


class DatasetManager:
    def __init__(
        self,
        train_dir: str = TRAIN_DIR, 
        val_dir: str = VAL_DIR, 
        test_dir: str = TEST_DIR, 
        batch_size: int = BATCH_SIZE,
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def get_dataloaders(self, train_transforms, val_transforms) -> Tuple[DataLoader, DataLoader]:
        train_dataset = ImageFolder(self.train_dir, transform=train_transforms)
        val_dataset = ImageFolder(self.val_dir, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def get_test_loader(self) -> DataLoader:
        if not self.test_dir:
            raise ValueError("test_dir was not initialization")

        test_dataset = ImageFolder(root=self.test_dir, transform=get_pred_transforms())
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0,
        )

        return test_loader
