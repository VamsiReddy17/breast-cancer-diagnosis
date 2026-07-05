import torch
import torch.nn as nn
import torchvision.models as models
from config.config import DL_CONFIG

class HistopathologyCNN(nn.Module):
    def __init__(self, model_name=None, pretrained=True):
        super(HistopathologyCNN, self).__init__()
        if model_name is None:
            model_name = DL_CONFIG["backbone"]
        self.model_name = model_name
        
        if model_name == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50()
                
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2) # Output logits for Benign (0) vs Malignant (1)
            )
            
        elif model_name == "efficientnet_b0":
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            else:
                self.backbone = models.efficientnet_b0()
                
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        return self.backbone(x)
