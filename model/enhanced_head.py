"""
Enhanced Detection Head with Decoupled Branches.

Designed after YOLOX/Tood - separates classification and regression
towers for better task-specific feature learning.
"""

import math

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """Conv + BN + Activation block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "mish":
            self.act = (
                nn.Mish(inplace=True) if hasattr(nn, "Mish") else nn.ReLU(inplace=True)
            )
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Conv + Pointwise Conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        self.dw = ConvBnAct(in_ch, in_ch, kernel, stride, groups=in_ch, act=act)
        self.pw = ConvBnAct(in_ch, out_ch, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class DecoupledHead(nn.Module):
    """Decoupled Detection Head (YOLOX-style).

    Separates classification and regression into independent towers
    with more conv layers for better task-specific learning.

    Architecture:
        stem                    stem
         ↓                       ↓
        conv tower  ──────→     conv tower
         (cls_tower)            (reg_tower)
         ↓                       ↓
        cls pred                reg pred
                                 ↓
                                obj pred (optional)
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 1,
        num_classes: int = 1,
        num_landmarks: int = 0,
        tower_depth: int = 4,
        stacked_convs: int = 2,
        use_obj: bool = True,
        use_dw: bool = True,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Input channels from FPN
        num_anchors : int
            Anchors per location (usually 1 for anchor-free, or 3+ for anchor-based)
        num_classes : int
            Number of classes
        num_landmarks : int
            Number of landmark points (0 to disable)
        tower_depth : int
            Number of conv layers in shared stem
        stacked_convs : int
            Number of task-specific conv layers
        use_obj : bool
            Use objectness prediction (like YOLO)
        use_dw : bool
            Use depthwise separable convs for efficiency
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_landmarks = num_landmarks
        self.use_obj = use_obj
        Conv = DWConv if use_dw else ConvBnAct

        # Shared stem
        self.stem = nn.ModuleList(
            [Conv(in_channels, in_channels) for _ in range(tower_depth)]
        )

        # Classification tower
        self.cls_tower = nn.ModuleList(
            [Conv(in_channels, in_channels) for _ in range(stacked_convs)]
        )
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)

        # Regression tower
        self.reg_tower = nn.ModuleList(
            [Conv(in_channels, in_channels) for _ in range(stacked_convs)]
        )
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # Objectness (optional, like YOLO)
        if use_obj:
            self.obj_pred = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)

        # Landmark tower (if enabled)
        if num_landmarks > 0:
            self.lmk_tower = nn.ModuleList(
                [Conv(in_channels, in_channels) for _ in range(stacked_convs)]
            )
            self.lmk_pred = nn.Conv2d(
                in_channels, num_anchors * num_landmarks * 2, kernel_size=1
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Classification bias for focal loss stability
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, bias_val)
        if self.use_obj:
            nn.init.constant_(self.obj_pred.bias, bias_val)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor (B, C, H, W)

        Returns
        -------
        cls_pred : (B, N, num_classes)
        reg_pred : (B, N, 4)
        obj_pred : (B, N, 1) or None
        lmk_pred : (B, N, num_landmarks*2) or None
        """
        B, C, H, W = x.shape

        # Shared stem
        feat = x
        for conv in self.stem:
            feat = conv(feat)

        # Classification branch
        cls_feat = feat
        for conv in self.cls_tower:
            cls_feat = conv(cls_feat)
        cls = self.cls_pred(cls_feat)  # (B, A*C, H, W)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(B, -1, self.num_classes)

        # Regression branch
        reg_feat = feat
        for conv in self.reg_tower:
            reg_feat = conv(reg_feat)
        reg = self.reg_pred(reg_feat)  # (B, A*4, H, W)
        reg = reg.permute(0, 2, 3, 1).contiguous()
        reg = reg.view(B, -1, 4)

        # Objectness
        obj = None
        if self.use_obj:
            obj = self.obj_pred(reg_feat)
            obj = obj.permute(0, 2, 3, 1).contiguous()
            obj = obj.view(B, -1, 1)

        # Landmarks
        lmk = None
        if self.num_landmarks > 0:
            lmk_feat = feat
            for conv in self.lmk_tower:
                lmk_feat = conv(lmk_feat)
            lmk = self.lmk_pred(lmk_feat)
            lmk = lmk.permute(0, 2, 3, 1).contiguous()
            lmk = lmk.view(B, -1, self.num_landmarks * 2)

        return cls, reg, obj, lmk


class EnhancedDetectionHead(nn.Module):
    """Multi-scale Detection Head with Enhanced Features.

    Combines:
    - Decoupled classification/regression towers
    - Dynamic label assignment
    - Task-aligned predictions
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 1,
        num_classes: int = 1,
        num_landmarks: int = 0,
        tower_depth: int = 4,
        use_obj: bool = True,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_landmarks = num_landmarks

        # One decoupled head per FPN level
        self.heads = nn.ModuleList(
            [
                DecoupledHead(
                    in_channels=in_channels,
                    num_anchors=num_anchors,
                    num_classes=num_classes,
                    num_landmarks=num_landmarks,
                    tower_depth=tower_depth,
                    use_obj=use_obj,
                    use_dw=True,
                )
                for _ in range(3)  # P3, P4, P5
            ]
        )

    def forward(self, fpn_features):
        """Apply heads to each FPN level.

        Parameters
        ----------
        fpn_features : tuple of (P3, P4, P5)

        Returns
        -------
        cls_preds : (B, total_anchors, num_classes)
        reg_preds : (B, total_anchors, 4)
        obj_preds : (B, total_anchors, 1) or None
        lmk_preds : (B, total_anchors, num_landmarks*2) or None
        """
        all_cls = []
        all_reg = []
        all_obj = []
        all_lmk = []

        assert len(fpn_features) == len(self.heads), (
            f"Expected {len(self.heads)} FPN features, got {len(fpn_features)}"
        )

        for feat, head in zip(fpn_features, self.heads):
            cls, reg, obj, lmk = head(feat)
            all_cls.append(cls)
            all_reg.append(reg)
            if obj is not None:
                all_obj.append(obj)
            if lmk is not None:
                all_lmk.append(lmk)

        cls_preds = torch.cat(all_cls, dim=1)
        reg_preds = torch.cat(all_reg, dim=1)

        obj_preds = torch.cat(all_obj, dim=1) if len(all_obj) > 0 else None
        lmk_preds = torch.cat(all_lmk, dim=1) if len(all_lmk) > 0 else None

        return cls_preds, reg_preds, obj_preds, lmk_preds


class DyHead(nn.Module):
    """Dynamic Head for Object Detection (TOOD-style).

    Uses attention mechanisms to dynamically fuse task-specific features.
    Reference: "TOOD: Task-aligned One-stage Object Detection" (ICCV 2021)
    """

    def __init__(self, channels: int = 256, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Scale-aware attention
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_levels),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_conv = ConvBnAct(channels, channels, kernel=5)

        # Task-aware attention (simplified)
        self.task_conv = nn.Sequential(
            ConvBnAct(channels, channels // 4), ConvBnAct(channels // 4, channels)
        )

    def forward(self, features: list):
        """Dynamic feature fusion across FPN levels.

        Parameters
        ----------
        features : list of FPN feature tensors

        Returns
        -------
        enhanced : list of enhanced features
        """
        B, C, _, _ = features[0].shape

        # Get scale weights
        base_feat = features[0]
        scale_weights = self.scale_attn(base_feat)  # (B, num_levels)

        # Apply scale attention and spatial enhancement
        enhanced = []
        for i, feat in enumerate(features):
            # Scale weighting
            weight = scale_weights[:, i : i + 1].view(B, 1, 1, 1)
            scaled = feat * weight

            # Spatial attention
            spatial = self.spatial_conv(scaled)

            # Task-aware
            task = self.task_conv(spatial)

            # Residual
            enhanced.append(feat + task)

        return enhanced
