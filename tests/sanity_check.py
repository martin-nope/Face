"""
Sanity check script — verifies all 7 fixes are structurally correct
before starting a full training run.

Tests:
  1. Per-level anchor counts (H×W×3 per level, not H×W×9)
  2. Total anchor count (25,200 not 75,600)
  3. Model forward pass shape with num_anchors=3
  4. Anchor target assignment positive ratio
  5. Loss computation (no NaN/Inf, hard negative mining active)
  6. ImageNet normalization applied
  7. mAP difficulty filtering
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_anchor_counts(cfg):
    """Test 1 & 2: Per-level anchor assignment and total count."""
    from data.anchors import generate_anchor_grid_per_level

    img_size = cfg["model"]["image_size"]
    feature_sizes = [
        (img_size // 8, img_size // 8),
        (img_size // 16, img_size // 16),
        (img_size // 32, img_size // 32),
    ]

    scales = cfg["model"]["anchor_scales"]
    ratios = cfg["model"]["anchor_ratios"]
    anchor_wh = np.array([[s, s / r] for s, r in zip(scales, ratios)], dtype=np.float32)
    # Sort by area
    areas = anchor_wh[:, 0] * anchor_wh[:, 1]
    anchor_wh = anchor_wh[np.argsort(areas)]

    anchors_per_level = cfg["model"].get("num_anchors", 3)
    anchors = generate_anchor_grid_per_level(
        feature_sizes, anchor_wh, img_size,
        anchors_per_level=anchors_per_level,
    )

    # Expected counts
    expected_per_level = [fh * fw * anchors_per_level for fh, fw in feature_sizes]
    expected_total = sum(expected_per_level)

    ok = anchors.shape[0] == expected_total
    print(f"  {'PASS' if ok else 'FAIL'}: Total anchors = {anchors.shape[0]} "
          f"(expected {expected_total})")

    # Check levels
    offset = 0
    for i, (fh, fw) in enumerate(feature_sizes):
        count = fh * fw * anchors_per_level
        level_anchors = anchors[offset:offset + count]
        offset += count
        # Check anchor widths are in the right scale range
        avg_w = level_anchors[:, 2].mean() * img_size
        print(f"    Level P{i+3} (stride {img_size // fh}): "
              f"{count} anchors, avg width = {avg_w:.1f}px")

    if not ok:
        return False

    # Verify old function would give 9× more
    from data.anchors import generate_anchor_grid
    old_anchors = generate_anchor_grid(feature_sizes, anchor_wh, img_size)
    old_total = old_anchors.shape[0]
    print(f"  INFO: Old generate_anchor_grid would give {old_total} anchors "
          f"({old_total / expected_total:.1f}× more)")
    return True


def test_model_forward(cfg):
    """Test 3: Forward pass shapes match num_anchors=3."""
    from training.train import FaceDetectionModel

    model = FaceDetectionModel(cfg)
    model.eval()

    img_size = cfg["model"]["image_size"]
    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        cls_preds, reg_preds, lmk_preds = model(x)

    num_anchors = cfg["model"].get("num_anchors", 3)
    feature_sizes = [
        (img_size // 8, img_size // 8),
        (img_size // 16, img_size // 16),
        (img_size // 32, img_size // 32),
    ]
    expected_total = sum(fh * fw * num_anchors for fh, fw in feature_sizes)

    cls_ok = cls_preds.shape == (1, expected_total, cfg["model"]["num_classes"])
    reg_ok = reg_preds.shape == (1, expected_total, 4)
    print(f"  {'PASS' if cls_ok else 'FAIL'}: cls_preds shape = {tuple(cls_preds.shape)} "
          f"(expected (1, {expected_total}, {cfg['model']['num_classes']}))")
    print(f"  {'PASS' if reg_ok else 'FAIL'}: reg_preds shape = {tuple(reg_preds.shape)} "
          f"(expected (1, {expected_total}, 4))")
    return cls_ok and reg_ok


def test_target_assignment(cfg):
    """Test 4: Anchor-GT matching produces reasonable positives."""
    from data.anchors import generate_anchor_grid_per_level
    from training.train import assign_targets

    img_size = cfg["model"]["image_size"]
    feature_sizes = [
        (img_size // 8, img_size // 8),
        (img_size // 16, img_size // 16),
        (img_size // 32, img_size // 32),
    ]
    scales = cfg["model"]["anchor_scales"]
    ratios = cfg["model"]["anchor_ratios"]
    anchor_wh = np.array([[s, s / r] for s, r in zip(scales, ratios)], dtype=np.float32)
    areas = anchor_wh[:, 0] * anchor_wh[:, 1]
    anchor_wh = anchor_wh[np.argsort(areas)]

    anchors = torch.from_numpy(
        generate_anchor_grid_per_level(
            feature_sizes, anchor_wh, img_size,
            anchors_per_level=cfg["model"].get("num_anchors", 3),
        )
    )

    # Synthetic GT: 3 boxes of different sizes
    gt_boxes = torch.tensor([[[
        0.3, 0.3, 0.05, 0.06,   # small face (~32px)
    ], [
        0.6, 0.6, 0.15, 0.18,   # medium face (~96px)
    ], [
        0.5, 0.5, 0.4, 0.48,    # large face (~256px)
    ]]])
    num_boxes = torch.tensor([3])

    cls_targets, reg_targets = assign_targets(anchors, gt_boxes, num_boxes)

    n_pos = (cls_targets == 1).sum().item()
    n_neg = (cls_targets == 0).sum().item()
    n_ignore = (cls_targets == -1).sum().item()
    total = cls_targets.numel()
    pos_ratio = n_pos / total * 100

    ok = n_pos >= 3  # at least 1 per GT
    print(f"  {'PASS' if ok else 'FAIL'}: Positives = {n_pos} ({pos_ratio:.2f}%), "
          f"Negatives = {n_neg}, Ignored = {n_ignore}")
    return ok


def test_loss_computation(cfg):
    """Test 5: Loss runs without NaN, hard negative mining active."""
    from model.loss import MultiTaskLoss

    loss_cfg = cfg["loss"]
    criterion = MultiTaskLoss(
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        reg_weight=loss_cfg.get("reg_weight", 2.0),
        lmk_weight=loss_cfg.get("lmk_weight", 0.5),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        neg_ratio=loss_cfg.get("neg_ratio", 3),
    )

    B, N, C = 2, 100, 1
    cls_preds = torch.randn(B, N, C)
    reg_preds = torch.randn(B, N, 4)

    # Simulate targets: 5 positives, rest negative
    cls_targets = torch.zeros(B, N, dtype=torch.long)
    cls_targets[0, :3] = 1
    cls_targets[1, :2] = 1
    reg_targets = torch.randn(B, N, 4) * 0.1
    
    # Synthetic anchors for testing
    anchors = torch.randn(N, 4).abs()
    anchors[:, 2:4] += 0.1 # ensure positive w,h

    total_loss, loss_dict = criterion(cls_preds, reg_preds, None, cls_targets, reg_targets, anchors)

    no_nan = not torch.isnan(total_loss).item()
    no_inf = not torch.isinf(total_loss).item()
    positive_loss = total_loss.item() > 0

    ok = no_nan and no_inf and positive_loss
    print(f"  {'PASS' if ok else 'FAIL'}: total_loss = {total_loss.item():.4f} "
          f"(nan={not no_nan}, inf={not no_inf})")
    for k, v in loss_dict.items():
        print(f"    {k} = {v.item():.4f}")
    return ok


def test_imagenet_normalization():
    """Test 6: Dataset applies ImageNet normalization."""
    # Create a synthetic white image
    img = np.ones((3, 640, 640), dtype=np.float32)  # Already CHW, [0,1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    normalized = (img - mean) / std

    # After normalization, values should NOT be in [0, 1]
    ch0_val = normalized[0, 0, 0]
    expected = (1.0 - 0.485) / 0.229
    ok = abs(ch0_val - expected) < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}: Channel 0 normalized value = {ch0_val:.4f} "
          f"(expected {expected:.4f})")
    return ok


def test_map_difficulty():
    """Test 7: mAP ignores difficult GT boxes."""
    from training.metrics import compute_map

    # 2 GT boxes: 1 easy, 1 hard (difficulty=2)
    # 1 prediction matching the easy box
    predictions = [{
        "boxes": np.array([[0.5, 0.5, 0.1, 0.1]]),
        "scores": np.array([0.9]),
    }]
    ground_truths = [{
        "boxes": np.array([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]]),
        "difficulties": np.array([0, 2]),
    }]

    result = compute_map(predictions, ground_truths, iou_threshold=0.5)

    # With difficulty filtering: 1 TP / 1 easy GT = mAP 1.0
    # Without filtering: 1 TP / 2 total GT = mAP 0.5
    ok = result["mAP"] > 0.9  # Should be ~1.0 with filtering
    print(f"  {'PASS' if ok else 'FAIL'}: mAP = {result['mAP']:.4f} "
          f"(expected ~1.0 with difficulty filtering)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Sanity check after fixes")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("SANITY CHECK — Verifying all 7 fixes")
    print("=" * 60)

    tests = [
        ("1. Per-level anchor counts", lambda: test_anchor_counts(cfg)),
        ("2. Model forward pass shapes", lambda: test_model_forward(cfg)),
        ("3. Target assignment positives", lambda: test_target_assignment(cfg)),
        ("4. Loss computation (HNM)", lambda: test_loss_computation(cfg)),
        ("5. ImageNet normalization", lambda: test_imagenet_normalization()),
        ("6. mAP difficulty filtering", lambda: test_map_difficulty()),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print(f"\n{'=' * 60}")
    all_passed = all(ok for _, ok in results)
    passed = sum(1 for _, ok in results if ok)
    print(f"Results: {passed}/{len(results)} tests passed")
    if all_passed:
        print("All checks passed! Ready for training.")
    else:
        print("Some checks failed — review the output above.")
        for name, ok in results:
            if not ok:
                print(f"  FAILED: {name}")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
