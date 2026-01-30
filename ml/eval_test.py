import base64
import io
import logging
from typing import Tuple, List

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DEVICE

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device = DEVICE) -> None:
        self.model = model
        self.device = device

    def evaluate(self, dataloader) -> Tuple[np.ndarray, str, float]:
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        logger.info(f"Accuracy: {accuracy:.4f}")
        return cm, report, accuracy


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> str:
    with plt.style.context('ggplot'):
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", xticklabels=classes, yticklabels=classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return img_base64

