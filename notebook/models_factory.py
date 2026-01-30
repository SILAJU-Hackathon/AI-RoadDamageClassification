import torch
import torch.nn as nn
from torchvision import models

# --- Shared Components ---
class RoadAttention(nn.Module):
    def __init__(self, in_channels, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.spatial_weight = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.spatial_weight(x))
        return x * (1 + mask) if self.use_residual else x * mask

class PotholeRefinement(nn.Module):
    def __init__(self, in_channels, version='resnet'):
        super().__init__()
        self.version = version
        if version == 'resnet':
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
        else: # convnext version
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.GroupNorm(1, in_channels),
                nn.GELU()
            )
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.GELU() if version != 'resnet' else nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.spatial_refine(x)
        gate = self.channel_gate(feat)
        return x + (feat * gate) if self.version != 'resnet' else feat * gate

# --- Specific Classifiers ---

class ResNet18Pothole(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
    def forward(self, x):
        return self.backbone(x)

class ModifResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.road_attention = RoadAttention(512)
        self.pothole_refine = PotholeRefinement(512, version='resnet')
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.pothole_refine(self.road_attention(self.backbone(x)))
        return self.classifier(x)

class ModifConvNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None).features
        self.road_attention = RoadAttention(768, use_residual=True)
        self.pothole_refine = PotholeRefinement(768, version='convnext')
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LayerNorm(768),
            nn.Dropout(0.4), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.pothole_refine(self.road_attention(self.backbone(x)))
        return self.classifier(x)

class ModifEffNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None).features
        self.in_features = 1280
        
        self.road_attention = RoadAttention(in_channels=self.in_features)
        self.pothole_refine = PotholeRefinement(in_channels=self.in_features,version="effnet")
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.in_features), 
            nn.Dropout(0.4),
            nn.Linear(self.in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.road_attention(x)
        x = self.pothole_refine(x)
        return self.classifier(x)
    
class ModifEffNetB3(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None).features
        self.in_features = 1536
        
        self.road_attention = RoadAttention(in_channels=self.in_features)
        self.pothole_refine = PotholeRefinement(in_channels=self.in_features,version="effnet")
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.in_features), 
            nn.Dropout(0.4),
            nn.Linear(self.in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.road_attention(x)
        x = self.pothole_refine(x)
        return self.classifier(x)
    
class ModifMobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None).features
        self.in_features = 960
        
        self.road_attention = RoadAttention(in_channels=self.in_features)
        self.pothole_refine = PotholeRefinement(in_channels=self.in_features,version="mobnet")
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.in_features), 
            nn.Dropout(0.4),
            nn.Linear(self.in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.road_attention(x)
        x = self.pothole_refine(x)
        return self.classifier(x)