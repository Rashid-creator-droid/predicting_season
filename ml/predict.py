import json
from typing import Tuple

import torch
from PIL import Image

from .config import (
    BEST_MODEL_PATH, 
    CLASS_TO_IDX_PATH, 
    DEVICE, 
    MODEL_NAME,
    NUM_CLASSES, 
    PRETRAINED,
)
from .eval_test import Evaluator
from .loaders import DatasetManager
from .models import get_model
from .transforms import get_pred_transforms


class SeasonPredictor:
    def __init__(
        self, 
        class_map_path: str = None, 
        model_path: str = BEST_MODEL_PATH, 
        device: torch.device = None,
    ) -> None:
        if class_map_path is None:
            class_map_path = CLASS_TO_IDX_PATH
        self.device = device or DEVICE

        self.model = get_model(
            MODEL_NAME, 
            num_classes=NUM_CLASSES, 
            pretrained=PRETRAINED,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transforms = get_pred_transforms()

        with open(class_map_path) as f:
            class_to_idx = json.load(f)

        self.class_names = [None] * len(class_to_idx)
        for cls, idx in class_to_idx.items():
            self.class_names[idx] = cls

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        x = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu()
            top_prob, top_class = torch.topk(probabilities, 1)
        return self.class_names[top_class.item()], top_prob.item()

    def test_model(self) -> Tuple:
        test_loader = DatasetManager().get_test_loader()
        evaluator = Evaluator(self.model, self.device)
        cm, report, accuracy = evaluator.evaluate(test_loader)
        plot = evaluator.plot_confusion_matrix(cm, classes=self.class_names)
        return plot, accuracy
