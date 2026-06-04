"""
ASPP (Atrous Spatial Pyramid Pooling) and multi-scale feature modules.

These modules enhance the receptive field of CNN features by capturing
context at multiple scales using dilated convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Provides multi-scale receptive fields using different atrous rates.
    Reference: "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
                Atrous Convolution, and Fully Connected CRFs" (TPAMI 2017)
    """

    def __init__(
        self, in_channels: int, out_channels: int = 256, rates: list = [6, 12, 18]
    ):
        super().__init__()

        # 1x1 conv branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Atrous conv branches
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global pooling branch
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv
        total_channels = out_channels * (2 + len(rates))  # 1x1 + atrous + global
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # 1x1 branch
        out1 = self.branch1x1(x)

        # Atrous branches
        atrous_outs = [branch(x) for branch in self.branches]

        # Global branch - upsample to match size
        global_out = self.branch_global(x)
        global_out = F.interpolate(
            global_out, size=(h, w), mode="bilinear", align_corners=False
        )

        # Concatenate all branches
        concat = torch.cat([out1, *atrous_outs, global_out], dim=1)

        return self.fusion(concat)


class RFBModule(nn.Module):
    """Receptive Field Block.

    Multi-branch structure with different kernel sizes and dilations
    to enhance receptive field diversity.
    Reference: "Receptive Field Block Net for Accurate and Fast Object Detection" (ECCV 2018)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations: list = [1, 3, 5],
        base_channels: int = 64,
    ):
        super().__init__()

        # Reduce channels first
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Multi-branch with different dilations
        self.branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                nn.Conv2d(
                    base_channels, base_channels, 3, padding=d, dilation=d, bias=False
                ),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # Extra branch with larger kernel
        self.branch_large = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion
        total = base_channels * (len(dilations) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.conv_reduce(x)

        # Collect branch outputs
        outs = [branch(reduced) for branch in self.branches]
        outs.append(self.branch_large(reduced))

        concat = torch.cat(outs, dim=1)
        out = self.fusion(concat)

        # Residual connection
        out = out + self.shortcut(x)
        return self.act(out)


class DenseASPP(nn.Module):
    """Dense ASPP for better multi-scale feature extraction.

    Densely connects atrous convolution layers for more comprehensive
    context aggregation.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 256, rates: list = [3, 6, 12, 18]
    ):
        super().__init__()

        self.num_rates = len(rates)
        base_channels = out_channels // (self.num_rates + 1)

        # Initial reduction
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Dense atrous layers
        self.atrous_layers = nn.ModuleList()
        current_channels = base_channels
        for rate in rates:
            layer = nn.Sequential(
                nn.Conv2d(
                    current_channels,
                    base_channels,
                    3,
                    padding=rate,
                    dilation=rate,
                    bias=False,
                ),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.atrous_layers.append(layer)
            current_channels += base_channels

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)

        features = [x]
        for layer in self.atrous_layers:
            # Concatenate all previous features
            concat = torch.cat(features, dim=1)
            out = layer(concat)
            features.append(out)

        # Final concatenation and projection
        final_concat = torch.cat(features, dim=1)
        return self.project(final_concat)


class ScaleAwareModule(nn.Module):
    """Scale-Aware Feature Enhancement.

    Adapts feature extraction based on estimated object scale.
    Uses scale estimation to weight different receptive fields.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Scale estimation
        self.scale_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),  # 3 scale bins
            nn.Softmax(dim=1),
        )

        # Multi-scale convs
        self.conv_small = nn.Conv2d(channels, channels // 3, 3, padding=1)
        self.conv_medium = nn.Conv2d(channels, channels // 3, 5, padding=2)
        self.conv_large = nn.Conv2d(channels, channels // 3, 7, padding=3, dilation=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estimate scale distribution
        scale_weights = self.scale_est(x)  # (B, 3, 1, 1)

        # Get multi-scale features
        f_small = self.conv_small(x)
        f_medium = self.conv_medium(x)
        f_large = self.conv_large(x)

        # Weighted combination
        f_multi = torch.cat([f_small, f_medium, f_large], dim=1)
        B, C, H, W = f_multi.shape

        # Apply scale weights
        scale_weights = scale_weights.view(B, 3, 1, 1)
        f_small_w = f_small * scale_weights[:, 0:1]
        f_medium_w = f_medium * scale_weights[:, 1:2]
        f_large_w = f_large * scale_weights[:, 2:3]

        f_out = torch.cat([f_small_w, f_medium_w, f_large_w], dim=1)
        f_out = self.fusion(f_out)

        # Residual
        return x + self.gate * f_out
