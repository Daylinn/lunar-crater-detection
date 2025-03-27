import torch
import torch.nn as nn
import torch.nn.functional as F

class CraterAttention(nn.Module):
    """Custom attention mechanism for crater detection"""
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
    def forward(self, x):
        # Reshape for attention
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Self-attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm(x + ffn_out)
        
        # Reshape back
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions and crater attention"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(nn.Sequential(
            Conv(self.c, self.c, 3),
            CraterAttention(self.c),
            Conv(self.c, self.c, 3)
        ) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 with crater attention"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attention = CraterAttention(c_ * 4)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = torch.cat((x, y1, y2, y3), 1)
        y = self.attention(y)
        return self.cv2(y)

class CraterYOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Backbone with crater-specific features
        self.conv1 = Conv(3, 32, 3, 2)  # 320x320
        self.conv2 = Conv(32, 64, 3, 2)  # 160x160
        self.c2f1 = C2f(64, 64, n=1)
        self.conv3 = Conv(64, 128, 3, 2)  # 80x80
        self.c2f2 = C2f(128, 128, n=2)
        self.conv4 = Conv(128, 256, 3, 2)  # 40x40
        self.c2f3 = C2f(256, 256, n=2)
        self.conv5 = Conv(256, 512, 3, 2)  # 20x20
        self.sppf = SPPF(512, 512)
        
        # Neck (FPN) with crater attention
        self.lateral1 = Conv(512, 256, 1)
        self.lateral2 = Conv(256, 256, 1)
        self.lateral3 = Conv(128, 128, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.fpn1 = C2f(512, 256, n=1)
        self.fpn2 = C2f(512, 256, n=1)
        self.fpn3 = C2f(256, 128, n=1)
        
        # Crater-specific detection heads
        self.detect1 = nn.Sequential(
            Conv(128, 256, 3),
            CraterAttention(256),
            nn.Conv2d(256, num_classes + 4, 1)
        )  # Small objects (80x80)
        
        self.detect2 = nn.Sequential(
            Conv(256, 512, 3),
            CraterAttention(512),
            nn.Conv2d(512, num_classes + 4, 1)
        )  # Medium objects (40x40)
        
        self.detect3 = nn.Sequential(
            Conv(256, 512, 3),
            CraterAttention(512),
            nn.Conv2d(512, num_classes + 4, 1)
        )  # Large objects (20x20)
        
    def forward(self, x):
        # Backbone
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = self.c2f1(x2)
        x3 = self.conv3(x2)
        x3 = self.c2f2(x3)
        x4 = self.conv4(x3)
        x4 = self.c2f3(x4)
        x5 = self.conv5(x4)
        x5 = self.sppf(x5)
        
        # FPN
        p5 = self.lateral1(x5)
        p4 = torch.cat([self.lateral2(x4), self.upsample(p5)], dim=1)
        p4 = self.fpn2(p4)
        p3 = torch.cat([self.lateral3(x3), self.upsample(p4)], dim=1)
        p3 = self.fpn3(p3)
        
        # Detection heads
        out1 = self.detect1(p3)  # Small objects
        out2 = self.detect2(p4)  # Medium objects
        out3 = self.detect3(p5)  # Large objects
        
        return [out1, out2, out3]

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p 