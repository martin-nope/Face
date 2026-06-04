"""
Enhanced Feature Pyramid Network with Attention and Multi-scale Modules.

Combines FPN with modern enhancements:
- BiFPN-style bidirectional connections
- Attention modules at each level
- ASPP for expanded receptive field
- PANet bottom-up path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aspp import ASPPModule, RFBModule, ScaleAwareModule
from model.attention import CBAM, CoordinateAttention, ECALayer, SEModule


class BiFPNBlock(nn.Module):
    """Bidirectional FPN block (EfficientDet-style).

    Adds extra lateral connections in both directions for richer
    feature fusion. Reference: EfficientDet (CVPR 2020)
    """

    def __init__(self, channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Weights for weighted feature fusion
        self.weights_td = nn.Parameter(torch.ones(num_levels - 1, 2))
        self.weights_bu = nn.Parameter(torch.ones(num_levels - 1, 2))

        # Conv layers after fusion
        self.convs_td = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels - 1)
            ]
        )

        self.convs_bu = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels - 1)
            ]
        )

    def forward(self, features):
        """Bidirectional feature fusion.

        features: list of [P3, P4, P5]
        """
        td_features = list(features)

        # Top-down path (with lateral connection)
        for i in range(self.num_levels - 1, 0, -1):
            prev = td_features[i]
            curr = td_features[i - 1]

            # Resize prev to match curr
            prev_up = F.interpolate(prev, size=curr.shape[2:], mode="nearest")

            # Weighted fusion
            weights = F.softmax(self.weights_td[i - 1], dim=0)
            fused = weights[0] * curr + weights[1] * prev_up
            td_features[i - 1] = self.convs_td[i - 1](fused)

        # Bottom-up path
        bu_features = [td_features[0]]
        for i in range(self.num_levels - 1):
            curr = bu_features[-1]
            next_feat = td_features[i + 1]

            # Downsample curr
            curr_down = F.max_pool2d(curr, kernel_size=3, stride=2, padding=1)

            # Weighted fusion
            weights = F.softmax(self.weights_bu[i], dim=0)
            fused = weights[0] * next_feat + weights[1] * curr_down
            bu_features.append(self.convs_bu[i](fused))

        return bu_features


class ASPP_FPN(nn.Module):
    """FPN with ASPP enhancement at P5.

    Adds dilated convolutions to capture larger context at coarse level."""

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        assert len(in_channels_list) == 3

        # Standard lateral connections
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # ASPP at P5
        self.aspp = ASPPModule(out_channels, out_channels, rates=[6, 12, 18])

        # Smoothing
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        c3, c4, c5 = features

        # Lateral connections with ASPP at P5
        p5_lat = self.lateral_c5(c5)
        p5 = self.aspp(p5_lat)

        p4_lat = self.lateral_c4(c4)
        p4 = p4_lat + F.interpolate(p5, size=c4.shape[2:], mode="nearest")

        p3_lat = self.lateral_c3(c3)
        p3 = p3_lat + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        return p3, p4, p5


class AttentionFPN(nn.Module):
    """FPN with attention modules at each pyramid level."""

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int = 256,
        attention_type: str = "se",
    ):
        """
        Parameters
        ----------
        in_channels_list : list of 3 ints
            Channels for [C3, C4, C5]
        out_channels : int
            FPN output channels
        attention_type : str
            "se", "cbam", "eca", "ca"
        """
        super().__init__()
        assert len(in_channels_list) == 3

        # Lateral connections
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Attention modules for each level
        attention_dict = {
            "se": SEModule,
            "cbam": CBAM,
            "eca": ECALayer,
            "ca": CoordinateAttention,
        }

        attn_class = attention_dict.get(attention_type, SEModule)

        self.attn_p3 = attn_class(out_channels)
        self.attn_p4 = attn_class(out_channels)
        self.attn_p5 = attn_class(out_channels)

        # Smoothing with extra conv
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.smooth_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        c3, c4, c5 = features

        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        # Apply attention
        p3 = self.attn_p3(p3)
        p4 = self.attn_p4(p4)
        p5 = self.attn_p5(p5)

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        return p3, p4, p5


class EnhancedFPN(nn.Module):
    """Fully Enhanced FPN - Combination of best features.

    Includes:
    - BiFPN-style bidirectional connections
    - Attention modules
    - RFB for receptive field enhancement
    - PANet bottom-up path
    """

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int = 256,
        use_bifpn: bool = True,
        use_attention: bool = True,
        attention_type: str = "se",
    ):
        super().__init__()
        assert len(in_channels_list) == 3

        # Initial lateral connections
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # RFB for enhanced receptive fields
        self.rfb_p3 = RFBModule(out_channels, out_channels, dilations=[1, 3, 5])
        self.rfb_p4 = RFBModule(out_channels, out_channels, dilations=[1, 3, 5])
        self.rfb_p5 = RFBModule(out_channels, out_channels, dilations=[3, 5, 7])

        # BiFPN blocks
        if use_bifpn:
            self.bifpn = BiFPNBlock(out_channels, num_levels=3)
        else:
            # Standard FPN smoothing
            self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.use_bifpn = use_bifpn

        # Attention modules
        if use_attention:
            attention_dict = {
                "se": SEModule,
                "cbam": CBAM,
                "eca": ECALayer,
                "ca": CoordinateAttention,
            }
            attn_class = attention_dict.get(attention_type, SEModule)
            self.attn_p3 = attn_class(out_channels)
            self.attn_p4 = attn_class(out_channels)
            self.attn_p5 = attn_class(out_channels)
        self.use_attention = use_attention

        # PA-Net bottom-up path (extra lateral up)
        self.pan_conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pan_conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        c3, c4, c5 = features

        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)

        # Top-down
        p4 = p4 + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        # RFB enhancement
        p3 = self.rfb_p3(p3)
        p4 = self.rfb_p4(p4)
        p5 = self.rfb_p5(p5)

        # BiFPN or standard smoothing
        if self.use_bifpn:
            features = [p3, p4, p5]
            p3, p4, p5 = self.bifpn(features)
        else:
            p3 = self.smooth_p3(p3)
            p4 = self.smooth_p4(p4)
            p5 = self.smooth_p5(p5)

        # Bottom-up path (PANet)
        p4 = p4 + self.pan_conv3(p3)
        p5 = p5 + self.pan_conv4(p4)

        # Attention
        if self.use_attention:
            p3 = self.attn_p3(p3)
            p4 = self.attn_p4(p4)
            p5 = self.attn_p5(p5)

        return p3, p4, p5


class SimpleFPN_v2(nn.Module):
    """Enhanced version of original FPN with selective enhancements."""

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int = 256,
        use_rfb: bool = False,
        use_attention: bool = False,
        attention_type: str = "se",
    ):
        """
        Parameters
        ----------
        in_channels_list : list
            [C3_channels, C4_channels, C5_channels]
        out_channels : int
            FPN output channels
        use_rfb : bool
            Add RFB modules for larger receptive fields
        use_attention : bool
            Add channel/spatial attention
        attention_type : str
            Type of attention: "se", "cbam", "eca"
        """
        super().__init__()

        # Lateral connections
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Smoothing
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Optional RFB
        self.use_rfb = use_rfb
        if use_rfb:
            self.rfb_p3 = RFBModule(out_channels, out_channels)
            self.rfb_p4 = RFBModule(out_channels, out_channels)
            self.rfb_p5 = RFBModule(out_channels, out_channels)

        # Optional Attention
        self.use_attention = use_attention
        if use_attention:
            self.attn_p3 = (
                SEModule(out_channels)
                if attention_type == "se"
                else CBAM(out_channels)
                if attention_type == "cbam"
                else ECALayer(out_channels)
            )
            self.attn_p4 = (
                SEModule(out_channels)
                if attention_type == "se"
                else CBAM(out_channels)
                if attention_type == "cbam"
                else ECALayer(out_channels)
            )
            self.attn_p5 = (
                SEModule(out_channels)
                if attention_type == "se"
                else CBAM(out_channels)
                if attention_type == "cbam"
                else ECALayer(out_channels)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

    def forward(self, features):
        c3, c4, c5 = features

        # Standard top-down FPN
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        # Optional RFB before smoothing
        if self.use_rfb:
            p3 = self.rfb_p3(p3)
            p4 = self.rfb_p4(p4)
            p5 = self.rfb_p5(p5)

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        # Optional attention
        if self.use_attention:
            p3 = self.attn_p3(p3)
            p4 = self.attn_p4(p4)
            p5 = self.attn_p5(p5)

        return p3, p4, p5
