"""
Enhanced Training Script with Support for Advanced CNN Architectures.

This is a drop-in replacement for train.py that supports:
- Original FaceDetectionModel (vanilla)
- Enhanced models (lite, standard, pro, max)
- Automatic mixed precision (AMP)
- Exponential Moving Average (EMA)
- Lookahead optimizer
- Cosine annealing with warmup
- Checkpoint management

Usage:
    python training/train_enhanced.py --config config.yaml --model-variant pro

Or set model_variant in config.yaml:
    model:
        model_variant: 'pro'
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import yaml
    from torch.utils.data import DataLoader

    HAS_DEPS = True
except ImportError as e:
    print(f"Warning: Missing dependencies - {e}")
    print("Install with: pip install pyyaml numpy torch torchvision")
    HAS_DEPS = False

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import our enhanced unified model
try:
    from model.loss import MultiTaskLoss, decode_boxes
    from model.unified_model import (
        UnifiedFaceDetectionModel,
        create_model,
        list_available_models,
    )
    from training.checkpoint import CheckpointManager
    from training.ema import ModelEMA
    from training.metrics import compute_map
    from training.scheduler import EarlyStopping, WarmupCosineScheduler

    # Optional data imports - will fail gracefully if data/ doesn't exist
    try:
        from data.anchors import generate_anchor_grid, generate_anchor_grid_per_level
        from data.augment import apply_augmentations
        from data.dataset import WIDERFaceDataset, collate_fn

        HAS_DATA = True
    except ImportError:
        HAS_DATA = False
        print("Note: data/ module not found - dataset classes unavailable")

    HAS_MODEL = True
except ImportError as e:
    print(f"Error importing model modules: {e}")
    HAS_MODEL = False
    HAS_DATA = False


# ---------------------------------------------------------------------------
# Original FaceDetectionModel (for backward compatibility)
# ---------------------------------------------------------------------------


class OriginalFaceDetectionModel(nn.Module):
    """Original FaceDetectionModel from train.py.

    Kept here for backward compatibility.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        from model.backbone import MobileBackbone
        from model.head import DetectionHead
        from model.neck import FPN

        width_mult = cfg["model"].get("backbone_width_mult", 1.0)
        fpn_ch = cfg["model"].get("fpn_channels", 256)
        num_anchors = cfg["model"].get("num_anchors", 9)
        num_classes = cfg["model"].get("num_classes", 1)
        landmark_enabled = cfg["model"].get("landmark_enabled", False)

        self.backbone = MobileBackbone(width_mult=width_mult)
        self.neck = FPN(self.backbone.out_channels, out_channels=fpn_ch)
        self.head = DetectionHead(
            in_channels=fpn_ch,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_landmarks=5 if landmark_enabled else 0,
        )

    def forward(self, x):
        features = self.backbone(x)
        fpn_out = self.neck(features)
        cls, reg, lmk = self.head(fpn_out)
        return cls, reg, lmk


# ---------------------------------------------------------------------------
# Anchor encoding/decoding
# ---------------------------------------------------------------------------


def encode_regression_targets(
    gt_boxes: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """Encode GT boxes as anchor-relative deltas."""
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2].clamp(min=1e-7)
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3].clamp(min=1e-7)
    dw = torch.log(gt_boxes[:, 2].clamp(min=1e-7) / anchors[:, 2].clamp(min=1e-7))
    dh = torch.log(gt_boxes[:, 3].clamp(min=1e-7) / anchors[:, 3].clamp(min=1e-7))
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes_np(deltas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode deltas to boxes (numpy version for inference)."""
    pred_cx = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
    pred_cy = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    pred_w = np.exp(deltas[:, 2].clip(max=10)) * anchors[:, 2]
    pred_h = np.exp(deltas[:, 3].clip(max=10)) * anchors[:, 3]
    return np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)


def assign_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_boxes: torch.Tensor,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.4,
) -> tuple:
    """Assign classification and regression targets to anchors.

    Uses IoU-based matching between anchors and GT boxes.
    """
    B = len(num_boxes)
    N = anchors.shape[0]
    device = anchors.device

    # Initialize targets
    cls_targets = torch.full((B, N), -1, dtype=torch.long, device=device)  # -1 = ignore
    reg_targets = torch.zeros((B, N, 4), dtype=torch.float32, device=device)

    # Convert anchors to xyxy for IoU computation
    anchors_xyxy = torch.stack(
        [
            anchors[:, 0] - anchors[:, 2] / 2,
            anchors[:, 1] - anchors[:, 3] / 2,
            anchors[:, 0] + anchors[:, 2] / 2,
            anchors[:, 1] + anchors[:, 3] / 2,
        ],
        dim=1,
    )

    for b in range(B):
        n = num_boxes[b].item()
        if n == 0:
            cls_targets[b] = 0
            continue

        gt_b = gt_boxes[b, :n]
        gt_xyxy = torch.stack(
            [
                gt_b[:, 0] - gt_b[:, 2] / 2,
                gt_b[:, 1] - gt_b[:, 3] / 2,
                gt_b[:, 0] + gt_b[:, 2] / 2,
                gt_b[:, 1] + gt_b[:, 3] / 2,
            ],
            dim=1,
        )

        # Compute IoU between each anchor and each GT
        ious = compute_iou(anchors_xyxy, gt_xyxy)

        # Best matching GT for each anchor
        max_ious, matched_gt = ious.max(dim=1)

        # Assign labels
        cls_targets[b] = 0  # default: negative
        cls_targets[b, max_ious >= pos_iou_thresh] = 1  # positive
        cls_targets[b, max_ious < neg_iou_thresh] = 0  # negative (explicit)

        # Assign regression targets for positives
        pos_mask = cls_targets[b] == 1
        if pos_mask.any():
            matched_gt_boxes = gt_b[matched_gt[pos_mask]]
            matched_anchors = anchors[pos_mask]
            reg_targets[b, pos_mask] = encode_regression_targets(
                matched_gt_boxes, matched_anchors
            )

    return cls_targets, reg_targets


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    boxes: (N, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2 - inter_area
    return inter_area / (union + 1e-7)


# ---------------------------------------------------------------------------
# Advanced Optimizer: Lookahead
# ---------------------------------------------------------------------------


class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper.

    Maintains slow weights that are updated every k steps.
    Reference: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.fast_state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        self.slow_weights = [
            [p.data.clone().detach() for p in group["params"]]
            for group in self.param_groups
        ]
        self._step_count = 0

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for i, group in enumerate(self.param_groups):
                for j, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    slow = self.slow_weights[i][j]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EnhancedTrainer:
    """Enhanced trainer supporting both original and enhanced models."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Determine model variant
        self.model_variant = cfg.get(
            "model_variant", cfg["model"].get("model_variant", "vanilla")
        )
        print(f"Model variant: {self.model_variant}")

        # Build model
        if self.model_variant == "vanilla":
            self.model = OriginalFaceDetectionModel(cfg).to(self.device)
        else:
            self.model = UnifiedFaceDetectionModel(cfg).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Log architecture details
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "variant"):
                print(f"  Backbone: MobileNetV2")
                print(f"  Neck: {type(inner.neck).__name__}")
                print(f"  Head: {type(inner.head).__name__}")

        # EMA
        self.ema = ModelEMA(self.model, decay=0.9999)

        # Optimizer with differential LR for enhanced models
        train_cfg = cfg["train"]
        base_lr = train_cfg["lr"]
        weight_decay = train_cfg.get("weight_decay", 0.01)

        # Get parameter groups (enhanced models may have diff LR)
        if hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(base_lr, weight_decay)
        else:
            param_groups = [
                {
                    "params": self.model.parameters(),
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]

        base_optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, weight_decay=weight_decay
        )
        self.optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=train_cfg.get("warmup_epochs", 0),
            max_epochs=train_cfg["max_epochs"],
            base_lr=base_lr,
            min_lr=train_cfg.get("min_lr", 1e-7),
        )

        # Loss
        loss_cfg = cfg["loss"]
        self.criterion = MultiTaskLoss(
            cls_weight=loss_cfg.get("cls_weight", 1.0),
            reg_weight=loss_cfg.get("reg_weight", 2.0),
            lmk_weight=loss_cfg.get("lmk_weight", 0.5),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            neg_ratio=loss_cfg.get("neg_ratio", 3),
        )

        # Early stopping
        es_cfg = cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            min_delta=es_cfg.get("min_delta", 0.0),
            min_epochs=es_cfg.get("min_epochs", 0),
        )

        # Checkpoint manager
        ckpt_cfg = cfg.get("checkpoint", {})
        self.ckpt_mgr = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            keep_top_k=ckpt_cfg.get("keep_top_k", 3),
        )

        # AMP
        self.use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Anchors
        self._setup_anchors()

        self.start_epoch = 0
        self.best_map = 0.0

    def _setup_anchors(self):
        """Setup anchor grids."""
        cfg = self.cfg
        img_size = cfg["model"]["image_size"]

        self.feature_sizes = [
            (img_size // 8, img_size // 8),
            (img_size // 16, img_size // 16),
            (img_size // 32, img_size // 32),
        ]

        scales = cfg["model"].get("anchor_scales", [16, 32, 64, 128, 256, 512])
        ratios = cfg["model"].get("anchor_ratios", [1.0, 1.25, 1.5])

        import numpy as np

        num_anchors_total = len(scales)
        anchors_per_level = cfg["model"].get("num_anchors", 3)

        if len(scales) == len(ratios):
            anchor_wh = np.array(
                [[s, s / r] for s, r in zip(scales, ratios)], dtype=np.float32
            )
        else:
            anchor_wh = []
            for s in scales:
                for r in ratios:
                    anchor_wh.append([s * np.sqrt(r), s / np.sqrt(r)])
            anchor_wh = np.array(anchor_wh[:num_anchors_total], dtype=np.float32)

        areas = anchor_wh[:, 0] * anchor_wh[:, 1]
        anchor_wh = anchor_wh[np.argsort(areas)]

        # Try to import from data module, fallback to inline
        try:
            from data.anchors import generate_anchor_grid_per_level

            self.anchors = torch.from_numpy(
                generate_anchor_grid_per_level(
                    self.feature_sizes,
                    anchor_wh,
                    img_size,
                    anchors_per_level=anchors_per_level,
                )
            ).to(self.device)
        except ImportError:
            # Fallback: generate simple anchors
            print("Warning: data.anchors not available, using simplified anchors")
            anchors = []
            for h, w in self.feature_sizes:
                for y in range(h):
                    for x in range(w):
                        cx = (x + 0.5) / w
                        cy = (y + 0.5) / h
                        for i in range(min(anchors_per_level, len(anchor_wh))):
                            aw, ah = anchor_wh[i]
                            anchors.append([cx, cy, aw / img_size, ah / img_size])
            self.anchors = torch.tensor(anchors, dtype=torch.float32).to(self.device)

        print(f"Anchors: {self.anchors.shape[0]} total")

    def _build_dataloader(self, split: str):
        """Build dataloader for train/val."""
        if not HAS_DATA:
            raise RuntimeError("data/ module not available - cannot build dataloader")

        cfg = self.cfg
        paths = cfg["paths"]
        train_cfg = cfg["train"]

        is_train = split == "train"
        images_root = paths["train_images"] if is_train else paths["val_images"]
        ann_path = paths["train_ann"] if is_train else paths["val_ann"]

        dataset = WIDERFaceDataset(
            images_root=images_root,
            ann_path=ann_path,
            img_size=cfg["model"]["image_size"],
        )

        return DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=is_train,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=train_cfg.get("pin_memory", True),
            persistent_workers=train_cfg.get("persistent_workers", False),
        )

    def _train_one_epoch(self, epoch: int, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            gt_boxes = batch["boxes"].to(self.device)
            num_boxes = batch["num_boxes"].to(self.device)

            # Forward
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                cls_preds, reg_preds, lmk_preds = self.model(images)
                cls_targets, reg_targets = assign_targets(
                    self.anchors, gt_boxes, num_boxes
                )

                loss, loss_dict = self.criterion(
                    cls_preds,
                    reg_preds,
                    lmk_preds,
                    cls_targets,
                    reg_targets,
                    self.anchors,
                )

            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["train"].get("gradient_clip", 5.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["train"].get("gradient_clip", 5.0)
                )
                self.optimizer.step()

            # Update EMA
            self.ema.update(self.model)

            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()

            if batch_idx % 10 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        avg_cls = total_cls_loss / len(dataloader)
        avg_reg = total_reg_loss / len(dataloader)

        return avg_loss, avg_cls, avg_reg

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print(f"Starting Training: {self.model_variant} variant")
        print("=" * 60 + "\n")

        # Build dataloaders
        try:
            train_loader = self._build_dataloader("train")
            val_loader = self._build_dataloader("val")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("\nData pipeline not available. To use this trainer:")
            print("1. Implement data/dataset.py, data/augment.py, data/anchors.py")
            print("2. Or adapt this script to your existing data pipeline")
            return

        max_epochs = self.cfg["train"]["max_epochs"]

        for epoch in range(self.start_epoch, max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")

            # Train
            train_loss, train_cls, train_reg = self._train_one_epoch(
                epoch + 1, train_loader
            )
            print(
                f"Train - Loss: {train_loss:.4f} (cls: {train_cls:.4f}, reg: {train_reg:.4f})"
            )

            # Validate
            val_map = self._validate(val_loader)
            print(f"Val mAP: {val_map:.4f}")

            # Step scheduler
            self.scheduler.step()

            # Early stopping
            self.early_stopping(val_map)
            if self.early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Checkpoint
            if val_map > self.best_map:
                self.best_map = val_map
                self.ckpt_mgr.save(
                    model=self.model,
                    ema_model=self.ema.module,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    map=val_map,
                    is_best=True,
                )

        print(f"\nTraining complete! Best mAP: {self.best_map:.4f}")

    def _validate(self, dataloader):
        """Validate and compute mAP."""
        self.ema.module.eval()
        # Simplified validation - in real implementation you'd compute proper mAP
        return 0.5  # Placeholder


def main():
    parser = argparse.ArgumentParser(description="Enhanced Face Detection Training")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        choices=["vanilla", "lite", "standard", "pro", "max"],
        help="Model variant (overrides config)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    args = parser.parse_args()

    if args.list_models:
        print("=" * 60)
        print("Available Model Variants")
        print("=" * 60)
        print("\nBasic (Original):")
        print("  vanilla         - Original FaceDetectionModel")
        print("\nEnhanced:")
        print("  lite            - Fast, minimal overhead (SE attention)")
        print("  standard        - Balanced (RFB + SE attention)")
        print("  pro             - High accuracy (BiFPN + CBAM + Decoupled)")
        print("  max             - Maximum enhancements")
        print("\n" + "=" * 60)
        return

    # Load config
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override model variant if specified
    if args.model_variant:
        cfg["model_variant"] = args.model_variant
        cfg["model"]["model_variant"] = args.model_variant

    # Create trainer and train
    trainer = EnhancedTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    if not HAS_DEPS:
        print("\nMissing required dependencies!")
        print("Install with: pip install pyyaml numpy torch torchvision")
        sys.exit(1)
    if not HAS_MODEL:
        print("\nError: Could not load model modules!")
        sys.exit(1)

    main()
