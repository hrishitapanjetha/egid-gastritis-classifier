import torch.nn as nn
from torchvision import models

from config import TASKS

BACKBONES = {
    "resnet18": lambda: (models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), "fc"),
    "resnet50": lambda: (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), "fc"),
    "efficientnet_b0": lambda: (models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), "classifier"),
    "densenet121": lambda: (models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1), "classifier"),
    "vit_b_16": lambda: (models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), "heads"),
    "swin_t": lambda: (models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1), "head"),
}


class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        backbone, head_attr = BACKBONES[backbone_name]()

        head_module = getattr(backbone, head_attr)
        if isinstance(head_module, nn.Sequential):
            in_features = head_module[-1].in_features
        else:
            in_features = head_module.in_features
        setattr(backbone, head_attr, nn.Identity())

        self.backbone = backbone
        self.heads = nn.ModuleDict({
            task: nn.Linear(in_features, len(classes)) for task, classes in TASKS.items()
        })

    def forward(self, x):
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}
