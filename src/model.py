import torch.nn as nn
from torchvision.models import resnet18

from config import NUM_CLASSES, USE_PRETRAINED


def build_resnet18(num_classes=NUM_CLASSES, use_pretrained=USE_PRETRAINED):
    model = resnet18(weights="DEFAULT" if use_pretrained else None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)