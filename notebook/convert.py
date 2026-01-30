import torch
import torchvision.models as models
from torch import nn
import onnx
import onnxruntime as ort
import numpy as np
import os

class RoadAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_weight = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.spatial_weight(x))
        return x * (1 + mask) 

class PotholeRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(1, in_channels), 
            nn.GELU() 
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.spatial_refine(x)
        gate = self.channel_gate(feat)
        return x + (feat * gate)

class PotholeClassifier(nn.Module):
    def __init__(self, num_classes, backbone_type="tiny"):
        super().__init__()
        # Robustly select backbone
        if backbone_type == "tiny":
            self.backbone = models.convnext_tiny(weights=None).features
            self.in_features = 768
        elif backbone_type == "small":
            self.backbone = models.convnext_small(weights=None).features
            self.in_features = 768
        
        self.road_attention = RoadAttention(in_channels=self.in_features)
        self.pothole_refine = PotholeRefinement(in_channels=self.in_features)
        
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

# --- Robust Conversion Function ---

def convert_to_onnx_robust(checkpoint_path, output_name, class_names):
    device = torch.device("cpu")
    print(f"[*] Loading checkpoint: {checkpoint_path}")
    
    # 1. Initialize Model
    model = PotholeClassifier(num_classes=len(class_names), backbone_type="tiny")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    # 2. Export to ONNX
    print(f"[*] Exporting to {output_name}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        export_params=True,
        opset_version=13, # Use 13 for better support of newer layers
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 3. Add Metadata (Labels) to ONNX file
    # This makes the file "self-documenting"
    onnx_model = onnx.load(output_name)
    meta = onnx_model.metadata_props.add()
    meta.key = "labels"
    meta.value = ",".join(class_names)
    onnx.save(onnx_model, output_name)

    # 4. Numerical Verification
    print("[*] Verifying numerical accuracy...")
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    ort_sess = ort.InferenceSession(output_name)
    ort_outs = ort_sess.run(None, {'input': dummy_input.numpy()})

    # Compare results
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)
    print("[OK] ONNX and PyTorch outputs match!")

if __name__ == "__main__":
    # Ensure these are in the order your training set used (alphabetical by default)
    CLASSES = ["Fair", "Good", "Poor", "Very Poor"]
    
    CKPT = "notebook/save_model/97.36_modif_convnext-s_checkpoint.pth"
    OUT = "notebook/save_model/road_quality_convnext_s.onnx"
    
    if os.path.exists(CKPT):
        convert_to_onnx_robust(CKPT, OUT, CLASSES)
    else:
        print(f"[!] Checkpoint not found at {CKPT}")