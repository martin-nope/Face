"""
Unified Model Interface - Drop-in replacement for original FaceDetectionModel.

This module provides backward compatibility while allowing easy switching
to enhanced architectures.
"""

import torch
import torch.nn as nn

# Support both original and enhanced models
try:
    from training.train import FaceDetectionModel as OriginalModel
except ImportError:
    OriginalModel = None

from model.enhanced_model import EnhancedFaceDetectionModel, ModelRegistry


class UnifiedFaceDetectionModel(nn.Module):
    """Unified interface that supports both original and enhanced models.

    This is a drop-in replacement for the original FaceDetectionModel
    that can seamlessly switch between architectures via config.

    Usage:
        # Original behavior (backward compatible)
        model = UnifiedFaceDetectionModel(cfg)

        # Enhanced with specific variant
        cfg['model_variant'] = 'pro'
        model = UnifiedFaceDetectionModel(cfg)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # Check if enhanced model is requested
        variant = cfg.get("model_variant", "vanilla")

        if variant == "vanilla" or variant is None:
            # Use original model for backward compatibility
            if OriginalModel is not None:
                self.model = OriginalModel(cfg)
            else:
                # Fallback to enhanced vanilla if original not available
                self.model = EnhancedFaceDetectionModel(cfg, variant="basic")
        else:
            # Use enhanced model
            self.model = ModelRegistry.create_model(cfg, config_name=variant)

        self.variant = variant

        # Expose subnets for compatibility
        if hasattr(self.model, "backbone"):
            self.backbone = self.model.backbone
        if hasattr(self.model, "neck"):
            self.neck = self.model.neck
        if hasattr(self.model, "head"):
            self.head = self.model.head

    def forward(self, x):
        """Forward pass - returns (cls_preds, reg_preds, lmk_preds) for compatibility.

        Original model returns: cls, reg, lmk
        Enhanced model returns: cls, reg, obj, lmk

        We normalize to: cls, reg, lmk (dropping obj if present)
        """
        outputs = self.model(x)

        # Handle both old and new output formats
        if len(outputs) == 3:
            # Original: cls, reg, lmk
            return outputs
        elif len(outputs) == 4:
            # Enhanced: cls, reg, obj, lmk - drop obj for compatibility
            cls, reg, obj, lmk = outputs
            return cls, reg, lmk
        else:
            raise ValueError(f"Unexpected output format with {len(outputs)} tensors")

    def get_param_groups(self, lr: float, weight_decay: float):
        """Get parameter groups for optimizer.

        Enhanced models may have different LR schedules for backbone vs head.
        """
        if hasattr(self.model, "get_param_groups"):
            return self.model.get_param_groups(lr, weight_decay)
        else:
            # Default: single group
            return [
                {"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}
            ]

    def __repr__(self):
        return (
            f"UnifiedFaceDetectionModel(variant={self.variant}, "
            f"params={sum(p.numel() for p in self.parameters()):,})"
        )


def create_model(cfg: dict, variant: str = None):
    """Factory function to create appropriate model.

    Parameters
    ----------
    cfg : dict
        Full configuration
    variant : str, optional
        Override variant from config. One of:
        - 'vanilla': Original architecture
        - 'lite': Light enhancements (SE attention)
        - 'standard': Balanced enhancements (RFB + SE)
        - 'pro': Full enhancements (BiFPN + CBAM + Decoupled)
        - 'max': Maximum enhancements

    Returns
    -------
    model : UnifiedFaceDetectionModel or OriginalModel
    """
    cfg = cfg.copy()
    if variant is not None:
        cfg["model_variant"] = variant

    return UnifiedFaceDetectionModel(cfg)


def list_available_models():
    """List all available model variants."""
    print("=" * 60)
    print("Available Model Variants")
    print("=" * 60)
    print()
    print("Legacy:")
    print("  vanilla    - Original FaceDetectionModel")
    print()
    print("Enhanced (Recommended):")
    ModelRegistry.list_configs()
    print()
    print("=" * 60)


# For direct import compatibility
FaceDetectionModel = UnifiedFaceDetectionModel


if __name__ == "__main__":
    # Test all variants
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

    list_available_models()
    print()

    # Test each variant
    for variant in ["vanilla", "lite", "standard", "pro"]:
        print(f"\nTesting {variant}...")
        try:
            model = create_model(cfg, variant=variant)
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                cls, reg, lmk = model(x)
            print(f"  ✓ {variant}: cls={tuple(cls.shape)}, reg={tuple(reg.shape)}")
        except Exception as e:
            print(f"  ✗ {variant}: {e}")
