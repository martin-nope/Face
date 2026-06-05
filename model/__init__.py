"""Face detection model architecture package.

Includes:
- Backbone: MobileNetV2 with various widths
- Neck: FPN, EnhancedFPN, AttentionFPN, BiFPN, ASPP_FPN
- Head: DetectionHead, EnhancedDetectionHead (decoupled)
- Attention: SE, CBAM, ECA, Coordinate Attention
- Multi-scale: ASPP, RFB, DenseASPP
- Loss: Focal, GIoU, CIoU, Smooth L1
"""

from model.aspp import ASPPModule, DenseASPP, RFBModule, ScaleAwareModule

# Enhanced modules
from model.attention import (
    CBAM,
    ChannelAttention,
    CoordinateAttention,
    ECALayer,
    SelectiveAttention,
    SEModule,
    SpatialAttention,
)
from model.backbone import MobileBackbone
from model.enhanced_head import (
    ConvBnAct,
    DecoupledHead,
    DWConv,
    DyHead,
    EnhancedDetectionHead,
)
from model.enhanced_model import (
    EnhancedFaceDetectionModel,
    ModelRegistry,
    print_model_info,
)
from model.enhanced_neck import (
    ASPP_FPN,
    AttentionFPN,
    BiFPNBlock,
    EnhancedFPN,
    SimpleFPN_v2,
)
from model.head import DetectionHead
from model.loss import FocalLoss, MultiTaskLoss, RegressionLoss
from model.neck import FPN
from model.unified_model import (
    FaceDetectionModel,
    UnifiedFaceDetectionModel,
    create_model,
    list_available_models,
)

__all__ = [
    # Original
    "MobileBackbone",
    "FPN",
    "DetectionHead",
    "MultiTaskLoss",
    "FocalLoss",
    "RegressionLoss",
    # Attention
    "SEModule",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "CoordinateAttention",
    "ECALayer",
    "SelectiveAttention",
    # Multi-scale
    "ASPPModule",
    "RFBModule",
    "DenseASPP",
    "ScaleAwareModule",
    # Enhanced necks
    "EnhancedFPN",
    "AttentionFPN",
    "ASPP_FPN",
    "SimpleFPN_v2",
    "BiFPNBlock",
    # Enhanced heads
    "EnhancedDetectionHead",
    "DecoupledHead",
    "DyHead",
    "ConvBnAct",
    "DWConv",
    # Model wrappers
    "EnhancedFaceDetectionModel",
    "ModelRegistry",
    "print_model_info",
    # Unified model
    "UnifiedFaceDetectionModel",
    "create_model",
    "list_available_models",
    "FaceDetectionModel",
]
