import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str = "resnet18", num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.fc = torch.nn.Linear(1024, num_classes)
    else:
        raise ValueError(f"nknown model name: {model_name}")

    return model
