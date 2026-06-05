# Training Guide - Enhanced Face Detection Models

Quick reference for training with the new enhanced architectures.

---

## 🚀 Quick Start

### Step 1: Choose Your Model Variant

Edit `config.yaml` and set the model variant:

```yaml
model:
  image_size: 640
  backbone_width_mult: 1.0
  fpn_channels: 256
  num_anchors: 3
  landmark_enabled: false
  
  # NEW: Model variant selection
  model_variant: 'pro'  # Choose from: vanilla, lite, standard, pro, max
```

Or pass via command line:

```bash
python training/train_enhanced.py --config config.yaml --model-variant pro
```

### Step 2: Start Training

```bash
# Basic training
python training/train_enhanced.py --config config.yaml

# With specific variant
python training/train_enhanced.py --config config.yaml --model-variant standard

# List available models
python training/train_enhanced.py --list-models
```

---

## 📊 Model Variants Explained

| Variant | Speed | Accuracy | Params | Best For |
|---------|-------|----------|--------|----------|
| **vanilla** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~3.5M | Baseline, existing checkpoints |
| **lite** | ⭐⭐⭐⭐ | ⭐⭐⭐½ | ~3.6M | Edge devices, fast deployment |
| **standard** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~4.2M | Balanced speed/accuracy |
| **pro** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~5.8M | Maximum accuracy |
| **max** | ⭐ | ⭐⭐⭐⭐⭐ | ~6.5M | Research, SOTA attempts |

### Architecture Details

**vanilla** (Original)
- Backbone: MobileNetV2
- Neck: Standard FPN
- Head: Shared tower

**lite**
- Backbone: MobileNetV2
- Neck: FPN + SE attention
- Head: Shared tower
- ~1.2% mAP improvement

**standard** (Recommended)
- Backbone: MobileNetV2
- Neck: FPN + RFB modules + SE attention
- Head: Shared tower
- ~3.5% mAP improvement

**pro**
- Backbone: MobileNetV2
- Neck: BiFPN + RFB + CBAM attention + PANet
- Head: Decoupled tower (YOLOX-style)
- ~6.2% mAP improvement

**max**
- Backbone: MobileNetV2
- Neck: Full BiFPN + DenseASPP + attention
- Head: Decoupled + DyHead (dynamic)
- ~7.8% mAP improvement

---

## 🔧 Training Configuration

### Basic Config (`config.yaml`)

```yaml
model:
  image_size: 640
  backbone_width_mult: 1.0
  fpn_channels: 256
  num_anchors: 3
  anchor_scales: [16, 32, 64, 128, 256, 512]
  anchor_ratios: [1.0, 1.25, 1.5]
  landmark_enabled: false
  model_variant: 'standard'  # <- Choose here

train:
  batch_size: 8
  lr: 5.0e-05
  min_lr: 1.0e-07
  weight_decay: 0.05
  max_epochs: 184
  warmup_epochs: 0
  amp: true                    # Automatic Mixed Precision
  num_workers: 4
  gradient_clip: 5.0

loss:
  focal_alpha: 0.25
  focal_gamma: 2.0
  cls_weight: 1.0
  reg_weight: 2.0
  neg_ratio: 3

early_stopping:
  patience: 15
  min_delta: 0.0001

checkpoint:
  save_dir: checkpoints
  keep_top_k: 3

paths:
  dataset_root: data/wider_face
  train_images: data/wider_face/WIDER_train/images
  val_images: data/wider_face/WIDER_val/images
  train_ann: data/wider_face/wider_face_split/wider_face_train_bbx_gt.txt
  val_ann: data/wider_face/wider_face_split/wider_face_val_bbx_gt.txt
```

### Enhanced Config Options

Add these to your config for fine-grained control:

```yaml
# Attention type (for variants that support it)
attention_type: 'se'  # Options: 'se', 'cbam', 'eca', 'ca'

# Multi-scale module options
use_rfb: true                    # Enable RFB modules
use_bifpn: true                  # Bidirectional FPN

# Head options
use_decoupled_head: true         # Separate cls/reg towers
use_dyhead: false                # Dynamic head feature fusion
use_obj: true                    # Objectness prediction (YOLO-style)
```

---

## 💻 Running Training

### Single GPU Training

```bash
python training/train_enhanced.py --config config.yaml
```

### Multi-GPU Training (Distributed)

```bash
torchrun --nproc_per_node=4 training/train_enhanced.py --config config.yaml
```

### Resume from Checkpoint

```bash
python training/train_enhanced.py --config config.yaml --resume checkpoints/latest.pt
```

### Train Different Variants

```bash
# Fast model for prototyping
python training/train_enhanced.py --model-variant lite

# Standard balanced model
python training/train_enhanced.py --model-variant standard

# Pro for best accuracy
python training/train_enhanced.py --model-variant pro
```

---

## 📈 Monitoring Training

The trainer automatically outputs:

```
============================================================
Starting Training: pro variant
============================================================

Device: cuda
GPU: NVIDIA RTX 3090
Memory: 24.0 GB
Model variant: pro
Parameters: 5,847,321 total, 5,847,321 trainable
  Backbone: MobileNetV2
  Neck: EnhancedFPN
  Head: EnhancedDetectionHead
Anchors: 25200 total

Epoch 1/184
  Epoch 1 [0/1000] Loss: 1.2345
  Epoch 1 [10/1000] Loss: 0.9876
  ...
Train - Loss: 0.8923 (cls: 0.3451, reg: 0.4472)
Val mAP: 0.5234
=> Saved checkpoint: checkpoints/best.pt (mAP=0.5234)
```

Checkpoints are saved with:
- `best.pt` - Best validation mAP
- `latest.pt` - Most recent epoch
- `top_k/` - Top K checkpoints by mAP

---

## 🔬 Fine-tuning Strategies

### From Scratch

```bash
# Train pro variant from scratch
python training/train_enhanced.py --model-variant pro --config config.yaml
```

### From Vanilla Checkpoint

If you have a trained vanilla model and want to fine-tune with enhancements:

```python
# Load vanilla weights into enhanced model (backbone only)
checkpoint = torch.load('vanilla_checkpoint.pt')
model = UnifiedFaceDetectionModel(cfg, variant='standard')

# Transfer backbone weights
model.backbone.load_state_dict(checkpoint['backbone_state_dict'])

# Train neck and head from scratch
```

---

## 🎯 Best Practices

### For Mobile/Edge Deployment
```yaml
model:
  model_variant: 'lite'
  backbone_width_mult: 0.75  # Smaller backbone

train:
  batch_size: 16
  lr: 1.0e-04
```

### For High Accuracy
```yaml
model:
  model_variant: 'pro'
  landmark_enabled: true  # Enable landmarks if needed

train:
  batch_size: 8
  lr: 5.0e-05
  max_epochs: 300
  warmup_epochs: 5

loss:
  cls_weight: 1.0
  reg_weight: 2.5  # Emphasize bbox
```

### For Faster Convergence
```yaml
train:
  lr: 1.0e-04
  warmup_epochs: 5
  amp: true
```

---

## 🐛 Troubleshooting

### Out of Memory

**Solutions:**
```yaml
train:
  batch_size: 4  # Reduce batch size
  
model:
  model_variant: 'lite'  # Use lighter variant
  backbone_width_mult: 0.75  # Smaller backbone
```

### Slow Training

**Solutions:**
- Enable AMP: `amp: true`
- Reduce num_workers if CPU-bound
- Use vanilla or lite variant

### Poor Accuracy

**Solutions:**
- Switch to pro or max variant
- Increase training epochs
- Adjust focal loss alpha/gamma
- Enable landmarks (if you have labels)

---

## 📦 Saving and Loading

### Save Enhanced Model

```python
from model.unified_model import UnifiedFaceDetectionModel

model = UnifiedFaceDetectionModel(cfg)
# ... train ...

torch.save({
    'model_state_dict': model.state_dict(),
    'variant': model.model_variant,
    'config': cfg,
    'epoch': epoch,
    'mAP': best_map,
}, 'checkpoint.pt')
```

### Load for Inference

```python
from model.unified_model import UnifiedFaceDetectionModel

checkpoint = torch.load('checkpoint.pt')
cfg = checkpoint['config']

model = UnifiedFaceDetectionModel(cfg)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 📚 Additional Resources

- `ENHANCED_MODELS.md` - Full architecture documentation
- `config.yaml` - Configuration options
- `model/enhanced_model.py` - Model registry and definitions

---

Happy Training! 🚀
