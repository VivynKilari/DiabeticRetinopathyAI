import torch.nn as nn
import timm

from src.utils.config import NUM_CLASSES


class DRClassifier(nn.Module):

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained
        )

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        return self.backbone(x)


def freeze_backbone(model):

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.classifier.parameters():
        param.requires_grad = True