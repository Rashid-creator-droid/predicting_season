import torch.nn as nn
from torchvision import models


def get_model(model_name: str = "resnet18", num_classes: int = 4) -> nn.Module:
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    return model
