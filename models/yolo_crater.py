# models/yolo_crater.py

import torch
import torch.nn as nn

# --- Basic Building Blocks ---

class Conv(nn.Module):
    """
    Standard convolution block: Conv2d -> BatchNorm -> SiLU (Swish)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1):
        super(Conv, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard bottleneck layer with optional shortcut
    """
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add:
            y = y + x
        return y

# --- YOLOv5-inspired Model ---

class YOLOv5(nn.Module):
    def __init__(self, num_classes=1):
        """
        A simplified YOLOv5 model for crater detection.
        The model predicts bounding boxes and objectness scores for each grid cell.
        
        num_classes: number of classes (set to 1 if only 'crater' is detected)
        """
        super(YOLOv5, self).__init__()
        # A simple backbone (this is a reduced version compared to YOLOv5)
        self.stem = Conv(3, 32, kernel_size=3, stride=1)
        self.layer1 = nn.Sequential(
            Conv(32, 64, kernel_size=3, stride=2),
            Bottleneck(64, 64, shortcut=True)
        )
        self.layer2 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=2),
            Bottleneck(128, 128, shortcut=True),
            Bottleneck(128, 128, shortcut=True)
        )
        self.layer3 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            Bottleneck(256, 256, shortcut=True),
            Bottleneck(256, 256, shortcut=True),
            Bottleneck(256, 256, shortcut=True)
        )
        self.layer4 = nn.Sequential(
            Conv(256, 512, kernel_size=3, stride=2),
            Bottleneck(512, 512, shortcut=True),
            Bottleneck(512, 512, shortcut=True),
            Bottleneck(512, 512, shortcut=True),
            Bottleneck(512, 512, shortcut=True)
        )
        
        # Detection head:
        # We'll assume that at the final feature map (from layer4), we predict 3 bounding boxes per grid cell.
        # For each bounding box, we need to predict 5 values (x, y, w, h, objectness) plus class scores.
        self.num_bbox = 3
        self.num_classes = num_classes
        self.head_channels = self.num_bbox * (5 + self.num_classes)
        
        # 1x1 convolution to produce the final predictions.
        self.head_conv = nn.Conv2d(512, self.head_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.stem(x)      # [B, 32, H, W]
        x = self.layer1(x)    # [B, 64, H/2, W/2]
        x = self.layer2(x)    # [B, 128, H/4, W/4]
        x = self.layer3(x)    # [B, 256, H/8, W/8]
        x = self.layer4(x)    # [B, 512, H/16, W/16]
        out = self.head_conv(x)  # [B, num_bbox*(5+num_classes), H/16, W/16]
        
        # Reshape output to [B, num_bbox, (5+num_classes), grid_h, grid_w]
        B, _, grid_h, grid_w = out.shape
        out = out.view(B, self.num_bbox, 5 + self.num_classes, grid_h, grid_w)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_bbox, grid_h, grid_w, (5+num_classes)]
        return out

if __name__ == "__main__":
    # Quick test to verify the model's output shape.
    model = YOLOv5(num_classes=1)
    x = torch.randn(1, 3, 640, 640)  # Example input (batch of 1, 640x640 RGB image)
    out = model(x)
    print("Output shape:", out.shape)
    # Expected shape: [1, 3, 40, 40, 6] for 640x640 input (since 640/16=40)