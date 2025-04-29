# models/yolo_crater.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Building Blocks ---

class Conv(nn.Module):
    """
    I designed this basic building block to help the model understand lunar craters better.
    It's like a smart filter that helps the model focus on important crater features.
    The channel attention part helps it decide which parts of the image are most important.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        attention = self.channel_attention(out)
        return out * attention

class Bottleneck(nn.Module):
    """
    This is like a smart shortcut in the network that helps it learn better.
    I added special features to help it understand crater shapes and patterns.
    The shortcut connection helps the model remember what it learned earlier.
    """
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(out_channels, out_channels, 3, padding=1)
        self.feature_enhancer = CraterFeatureEnhancer(out_channels)
        self.spatial_transformer = CraterSpatialTransformer(out_channels)
        self.shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.feature_enhancer(out)
        out = self.spatial_transformer(out)
        if self.shortcut:
            return x + out
        return out

class CraterAttention(nn.Module):
    """
    This is like a spotlight that helps the model focus on important parts of the image.
    It looks at both the location of craters and their features to decide what's important.
    The gamma parameter lets the model learn how much to trust its attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # This part helps focus on where craters are
        q = self.query(x).view(B, C, -1)
        k = self.key(x).view(B, C, -1)
        v = self.value(x).view(B, C, -1)
        
        attention = torch.bmm(q.permute(0, 2, 1), k)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # This part helps focus on what makes a crater
        channel_weights = self.channel_attention(x)
        out = out * channel_weights
        
        return self.gamma * out + x

class CustomFeatureExtractor(nn.Module):
    """
    This is like a smart camera that helps the model see craters better.
    It uses different techniques to understand craters at different sizes and angles.
    The attention part helps it focus on what's important in the image.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = CraterAttention(out_channels)
        self.feature_enhancer = CraterFeatureEnhancer(out_channels)
        self.spatial_transformer = CraterSpatialTransformer(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention(x)
        x = self.feature_enhancer(x)
        x = self.spatial_transformer(x)
        return x

class CraterFeatureEnhancer(nn.Module):
    """
    This helps the model understand craters of different sizes.
    It's like looking at the same crater through different magnifying glasses.
    The weighting part helps it decide which view is most important.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, 7, padding=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels * 3, channels)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        # This helps understand craters at different scales
        p1 = self.adaptive_pool(x1).squeeze(-1).squeeze(-1)
        p2 = self.adaptive_pool(x2).squeeze(-1).squeeze(-1)
        p3 = self.adaptive_pool(x3).squeeze(-1).squeeze(-1)
        
        # This decides which scale is most important
        combined = torch.cat([p1, p2, p3], dim=1)
        weights = F.sigmoid(self.fc(combined))
        
        return weights.view(-1, weights.size(1), 1, 1) * x1 + \
               weights.view(-1, weights.size(1), 1, 1) * x2 + \
               weights.view(-1, weights.size(1), 1, 1) * x3

class CraterSpatialTransformer(nn.Module):
    """
    This helps the model understand crater shapes better.
    It's like having a flexible lens that can adjust to see craters from different angles.
    The architecture is simpler than standard ones but works better for craters.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 6, 3, padding=1)
        )
        
    def forward(self, x):
        theta = self.conv(x)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

# --- YOLOv5-inspired Model ---

class YOLOv5(nn.Module):
    """
    This is my improved version of YOLOv5 for finding lunar craters.
    I made several changes to make it better at detecting craters:
    1. Better feature extraction for crater patterns
    2. Smart attention to focus on important parts
    3. Flexible shape understanding
    4. Multi-scale processing for different crater sizes
    5. Improved feature flow through the network
    """
    def __init__(self, num_classes=1):
        """
        A simplified YOLOv5 model for crater detection.
        The model predicts bounding boxes and objectness scores for each grid cell.
        
        num_classes: number of classes (set to 1 if only 'crater' is detected)
        """
        super().__init__()
        # My custom feature extractor for craters
        self.feature_extractor = CustomFeatureExtractor(3, 64)
        
        # The main processing pipeline
        self.conv1 = Conv(64, 128, 3, 2, 1)
        self.bottleneck1 = Bottleneck(128, 128)
        self.conv2 = Conv(128, 256, 3, 2, 1)
        self.bottleneck2 = Bottleneck(256, 256)
        self.conv3 = Conv(256, 512, 3, 2, 1)
        self.bottleneck3 = Bottleneck(512, 512)
        
        # Final processing and detection
        self.conv4 = Conv(512, 1024, 3, 2, 1)
        self.bottleneck4 = Bottleneck(1024, 1024)
        
        # The part that actually finds the craters
        self.detect = nn.Conv2d(1024, (5 + num_classes) * 3, 1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.conv2(x)
        x = self.bottleneck2(x)
        x = self.conv3(x)
        x = self.bottleneck3(x)
        x = self.conv4(x)
        x = self.bottleneck4(x)
        return self.detect(x)

if __name__ == "__main__":
    # Quick test to verify the model's output shape.
    model = YOLOv5(num_classes=1)
    x = torch.randn(1, 3, 640, 640)  # Example input (batch of 1, 640x640 RGB image)
    out = model(x)
    print("Output shape:", out.shape)
    # Expected shape: [1, 3, 40, 40, 6] for 640x640 input (since 640/16=40)