"""
Enhanced Face Detection Model.

Combines all CNN enhancements into a single flexible architecture.
"""

import torch
import torch.nn as nn

from model.backbone import MobileBackbone
from model.enhanced_head import DyHead, EnhancedDetectionHead
from model.enhanced_neck import ASPP_FPN, AttentionFPN, EnhancedFPN, SimpleFPN_v2
from model.head import DetectionHead
from model.neck import FPN


class EnhancedFaceDetectionModel(nn.Module):
    """Enhanced Face Detection Model with flexible configuration.

    Supports various combinations of modern CNN techniques:
    - Enhanced backbones
    - Attention-based FPN
    - Decoupled detection heads
    - Dynamic feature fusion

    Example configurations:

    # Basic (original)
    EnhancedFaceDetectionModel(cfg, variant='basic')

    # With attention FPN
    EnhancedFaceDetectionModel(cfg, variant='attention', attention_type='se')

    # Full enhancement
    EnhancedFaceDetectionModel(cfg, variant='full', use_decoupled_head=True)
    """

    def __init__(self, cfg: dict, variant: str = "basic"):
        """
        Parameters
        ----------
        cfg : dict
            Full config dict
        variant : str
            'basic' - Original architecture
            'attention' - FPN with attention
            'aspp' - FPN with ASPP
            'enhanced' - FPN with RFB + attention
            'full' - Everything: BiFPN + attention + decoupled head + DyHead
        """
        super().__init__()

        # Extract config
        width_mult = cfg["model"].get("backbone_width_mult", 1.0)
        fpn_ch = cfg["model"].get("fpn_channels", 256)
        num_anchors = cfg["model"].get("num_anchors", 3)
        num_classes = cfg["model"].get("num_classes", 1)
        landmark_enabled = cfg["model"].get("landmark_enabled", False)
        num_landmarks = 5 if landmark_enabled else 0

        # Backbone
        self.backbone = MobileBackbone(width_mult=width_mult)
        backbone_ch = self.backbone.out_channels

        # Neck variant
        if variant == "basic":
            self.neck = FPN(backbone_ch, out_channels=fpn_ch)

        elif variant == "attention":
            attention_type = cfg.get("attention_type", "se")
            self.neck = AttentionFPN(
                backbone_ch, out_channels=fpn_ch, attention_type=attention_type
            )

        elif variant == "aspp":
            self.neck = ASPP_FPN(backbone_ch, out_channels=fpn_ch)

        elif variant == "enhanced":
            attention_type = cfg.get("attention_type", "se")
            use_rfb = cfg.get("use_rfb", True)
            use_attention = cfg.get("use_attention", True)
            self.neck = SimpleFPN_v2(
                backbone_ch,
                out_channels=fpn_ch,
                use_rfb=use_rfb,
                use_attention=use_attention,
                attention_type=attention_type,
            )

        elif variant == "full":
            attention_type = cfg.get("attention_type", "se")
            use_bifpn = cfg.get("use_bifpn", True)
            use_attention = cfg.get("use_attention", True)
            self.neck = EnhancedFPN(
                backbone_ch,
                out_channels=fpn_ch,
                use_bifpn=use_bifpn,
                use_attention=use_attention,
                attention_type=attention_type,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant

        # Head variant
        use_decoupled = cfg.get("use_decoupled_head", False)
        use_dyhead = cfg.get("use_dyhead", False)

        if use_decoupled:
            self.head = EnhancedDetectionHead(
                in_channels=fpn_ch,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_landmarks=num_landmarks,
                tower_depth=4,
                use_obj=cfg.get("use_obj", True),
            )
        else:
            self.head = DetectionHead(
                in_channels=fpn_ch,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_landmarks=num_landmarks,
                tower_depth=4,
            )

        # Optional dynamic head
        if use_dyhead:
            self.dyhead = DyHead(channels=fpn_ch, num_levels=3)
        else:
            self.dyhead = None

    def forward(self, x):
        """Forward pass.

        Returns
        -------
        cls_preds : (B, total_anchors, num_classes)
        reg_preds : (B, total_anchors, 4)
        obj_preds : (B, total_anchors, 1) or None
        lmk_preds : (B, total_anchors, num_landmarks*2) or None
        """
        # Backbone
        features = self.backbone(x)

        # FPN
        fpn_out = self.neck(features)

        # Dynamic head enhancement
        if self.dyhead is not None:
            fpn_out = list(fpn_out)
            fpn_out = self.dyhead(fpn_out)
            fpn_out = tuple(fpn_out)

        # Detection head
        if hasattr(self.head, "use_obj") and self.head.use_obj:
            cls, reg, obj, lmk = self.head(fpn_out)
            return cls, reg, obj, lmk
        else:
            cls, reg, lmk = self.head(fpn_out)
            return cls, reg, None, lmk

    def get_param_groups(self, lr: float, weight_decay: float):
        """Get parameter groups with different learning rates.

        Typically backbone uses lower LR than detection head.
        """
        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": lr * 0.1,  # Lower LR for pretrained backbone
                "weight_decay": weight_decay,
            },
            {"params": [], "lr": lr, "weight_decay": weight_decay},
        ]

        # Collect neck + head params
        neck_head_params = []
        for module in [self.neck, self.head]:
            if module is not None:
                neck_head_params.extend(list(module.parameters()))
        if self.dyhead is not None:
            neck_head_params.extend(list(self.dyhead.parameters()))

        param_groups[1]["params"] = neck_head_params

        return param_groups


class ModelRegistry:
    """Registry of model configurations for easy switching."""

    CONFIGS = {
        # Vanilla - original architecture
        "vanilla": {
            "variant": "basic",
            "attention_type": None,
            "use_rfb": False,
            "use_attention": False,
            "use_decoupled_head": False,
            "use_dyhead": False,
            "use_obj": False,
        },
        # Lite - minimal enhancements for mobile
        "lite": {
            "variant": "enhanced",
            "attention_type": "se",
            "use_rfb": False,
            "use_attention": True,
            "use_decoupled_head": False,
            "use_dyhead": False,
            "use_obj": False,
        },
        # Standard - good balance of speed/accuracy
        "standard": {
            "variant": "enhanced",
            "attention_type": "se",
            "use_rfb": True,
            "use_attention": True,
            "use_decoupled_head": False,
            "use_dyhead": False,
            "use_obj": False,
        },
        # Pro - full enhancements
        "pro": {
            "variant": "full",
            "attention_type": "cbam",
            "use_bifpn": True,
            "use_rfb": True,
            "use_attention": True,
            "use_decoupled_head": True,
            "use_dyhead": True,
            "use_obj": True,
        },
        # Max - everything including heavy modules
        "max": {
            "variant": "full",
            "attention_type": "cbam",
            "use_bifpn": True,
            "use_rfb": True,
            "use_attention": True,
            "use_decoupled_head": True,
            "use_dyhead": True,
            "use_obj": True,
        },
    }

    @classmethod
    def list_configs(cls):
        """List available model configs."""
        print("Available model configurations:")
        for name, config in cls.CONFIGS.items():
            print(f"  - {name}: {config}")

    @classmethod
    def get_config(cls, name: str):
        """Get a model configuration by name."""
        if name not in cls.CONFIGS:
            raise ValueError(
                f"Unknown config: {name}. Available: {list(cls.CONFIGS.keys())}"
            )
        return cls.CONFIGS[name].copy()

    @classmethod
    def create_model(cls, cfg: dict, config_name: str = "standard"):
        """Create a model with predefined configuration."""
        model_cfg = cls.get_config(config_name)
        # Merge with user config (user config takes priority)
        merged_cfg = {**model_cfg, **cfg}
        return EnhancedFaceDetectionModel(merged_cfg, variant=model_cfg["variant"])


def print_model_info(model: nn.Module):
    """Print model information."""
    print("=" * 60)
    print("Enhanced Face Detection Model")
    print("=" * 60)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()

    # Module sizes
    print("Module sizes:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {params:>10,} params")

    print("=" * 60)


# Test
if __name__ == "__main__":
    # Example config
    cfg = {
        "model": {
            "image_size": 640,
            "num_classes": 1,
            "backbone_width_mult": 1.0,
            "fpn_channels": 256,
            "num_anchors": 3,
            "landmark_enabled": False,
        }
    }

    # List available configs
    ModelRegistry.list_configs()
    print()

    # Test different variants
    for variant in ["vanilla", "lite", "standard", "pro"]:
        print(f"\n{'=' * 40}")
        print(f"Testing {variant} variant")
        print("=" * 40)

        model = ModelRegistry.create_model(cfg, variant)
        print_model_info(model)

        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            cls, reg, obj, lmk = model(x)
        print(f"Output shapes:")
        print(f"  cls: {tuple(cls.shape)}")
        print(f"  reg: {tuple(reg.shape)}")
        print(f"  obj: {obj.shape if obj is not None else None}")
        print(f"  lmk: {lmk.shape if lmk is not None else None}")
