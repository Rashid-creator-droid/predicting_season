import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 4):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
