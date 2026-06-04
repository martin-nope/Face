"""
Attention mechanisms for feature enhancement.

Includes:
- SE (Squeeze-and-Excitation) : channel attention
- CBAM : channel + spatial attention
- Coordinate Attention : position-aware attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-Excitation module.

    Learns channel-wise importance via global average pooling.
    Reference: "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention from CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention sequentially.
    Reference: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CoordinateAttention(nn.Module):
    """Coordinate Attention for mobile networks.

    Embeds position information into channel attention.
    Reference: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_dim = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # Pool along height and width
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x)  # (B, C, 1, W)

        # Concatenate and transform
        x_cat = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)  # (B, C, H+W, 1)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        # Split back
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Apply attention
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()

        return identity * x_h * x_w


class ECALayer(nn.Module):
    """Efficient Channel Attention.

    Uses 1D convolution instead of MLP - more efficient.
    Reference: "ECA-Net: Efficient Channel Attention" (CVPR 2020)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        kernel_size = int(
            abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma)
        )
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SelectiveAttention(nn.Module):
    """Dynamic attention selection module.

    Learns to combine multiple attention mechanisms.
    """

    def __init__(self, channels: int, use_se: bool = True, use_ca: bool = False):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.use_se = use_se
        self.use_ca = use_ca

        if use_se:
            self.se = SEModule(channels)
        if use_ca:
            self.ca = CoordinateAttention(channels, channels)

        # Fusion weights
        num_atts = sum([use_se, use_ca])
        self.fusion = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2, bias=False),
                nn.Softmax(dim=1),
            )
            if num_atts > 1
            else None
        )

        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        if self.use_se:
            outputs.append(self.se(x))
        if self.use_ca:
            outputs.append(self.ca(x))

        if len(outputs) == 1:
            return x + self.gate * (outputs[0] - x)  # Residual

        # Weighted combination
        weights = self.fusion(x).view(-1, 2, 1, 1, 1)
        out = weights[:, 0] * outputs[0] + weights[:, 1] * outputs[1]
        return x + self.gate * (out - x)  # Residual connection
