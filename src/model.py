import torch.nn as nn
from torchvision import models

def build_model(model_name, num_classes, pretrained=True, freeze_backbone=True):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unknown model name")

    return model
