# Getting Started

A step-by-step guide to train and deploy the custom face detection & recognition system.

---

## ⚡ Quick Start — Run the Model Now

If you already have a trained checkpoint (e.g., `checkpoints/latest.pt`) and want to see the system in action immediately:

### 1. Live Webcam Recognition
Detect, track, and recognize faces in real-time from your webcam.

```bash
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt
```

> Tip: If you still see boxes around non-faces, raise the minimum confidence with `--conf-threshold 0.5` (default is now 0.4).
>
> Tip: When you add new crops to `known_students/`, the system will auto-update `encodings.pkl` on startup (no manual rebuild required).
>
> If you want to skip auto-updating (e.g., for speed), add `--no-encoding-update`.

*Press `q` to quit the window.*

### 2. Single Image Inference
Run detection on a specific image and save the result to the `outputs/` directory.
```bash
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --image path/to/your/image.jpg
```

---

## Prerequisites

- **Python 3.10+** with pip
- **CUDA GPU** with ≥6 GB VRAM (recommended for training)
- **Virtual Environment**: Always run scripts using your virtual environment (e.g., `.\venv\Scripts\python.exe` on Windows) so that PyTorch can correctly detect your CUDA GPU. Avoid using the global `py -3` launcher.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Step 1 — Download WIDER FACE Dataset

Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset and organise it as follows:

```
data/wider_face/
├── WIDER_train/
│   └── images/
│       ├── 0--Parade/
│       ├── 1--Handshaking/
│       └── ...
├── WIDER_val/
│   └── images/
│       └── ...
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt
    └── wider_face_val_bbx_gt.txt
```

**Direct download links:**
| File | URL |
|---|---|
| Training images | https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip |
| Validation images | https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip |
| Annotations | https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip |

Extract all three into `data/wider_face/`.

---

## Step 2 — Compute Dataset-Specific Anchors

Run K-means clustering on WIDER FACE bounding box dimensions to generate anchors tuned for face aspect ratios. This replaces generic anchors with data-driven ones.

```bash
venv\Scripts\python.exe data/anchors.py --config config.yaml --update-config
```

This will print the computed anchor sizes and automatically update `config.yaml` with the new values.

---

## Step 3 — Train the Model

Launch the full training loop with automatic mixed precision (FP16), cosine warmup scheduler, and checkpoint management.

> **IMPORTANT**: On Windows, do NOT use `py -3` or the global `python` command, as it will ignore your virtual environment and run on the CPU. Always provide the explicit path to your virtual environment's Python executable.

```bash
venv\Scripts\python.exe training/train.py --config config.yaml
```

**What to expect:**
- Training logs per-component losses: `cls_loss`, `reg_loss`, `lmk_loss`
- Validation mAP@0.5 is computed at the end of every epoch
- Checkpoints are saved to `checkpoints/` (top-3 by mAP + latest)
- Early stopping triggers after 15 epochs without mAP improvement
- Auto-resumes from `checkpoints/latest.pt` if training is interrupted

**Tuning tips** — edit `config.yaml`:
- Reduce `train.batch_size` if you hit OOM errors
- Adjust `train.max_epochs` for longer/shorter training
- Set `model.landmark_enabled: true` if you add landmark-annotated data

---

## Step 4 — Test Inference

Benchmark FPS, visualise detections, and test edge cases.

```bash
# FPS benchmark (CPU + GPU)
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --benchmark

# Detect on a single image
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --image path/to/photo.jpg

# GT vs predicted comparison grid
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --compare

# Edge case tests (dark, partial, noise)
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --edge-cases
```

Results are saved to the `outputs/` directory.

---

## Step 5 — Build Face Encodings

Enroll known identities by saving face crops, then build embeddings for recognition.

### 5a. Capture enrollment photos

```bash
venv\Scripts\python.exe app/capture_photo.py --checkpoint checkpoints/latest.pt
```

- Enter the person's name when prompted
- Press `s` to save a crop (quality gate enforces blur / confidence / size thresholds)
- Captures 5 crops per person, saved to `known_students/{name}/`

### 5b. Build the encoding database

```bash
venv\Scripts\python.exe app/build_encodings.py --config config.yaml
```

- Extracts InsightFace embeddings from all saved crops
- Saves to `encodings.pkl` as `dict[name → List[embedding]]`
- Use `--rebuild` to force reprocessing of all crops

---

## Step 6 — Live Recognition

Run the webcam pipeline with real-time detection + tracking + recognition.

```bash
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt
```

- Detection by our custom model
- SORT tracker assigns persistent track IDs across frames
- InsightFace embedding runs every 5 frames per track (not every frame → higher FPS)
- Press `q` to quit

---

## Optional — Export & Profile

### Export to ONNX / TorchScript

```bash
venv\Scripts\python.exe inference/export.py --checkpoint checkpoints/latest.pt --format both --verify
```

### Profile bottlenecks

```bash
venv\Scripts\python.exe inference/profiler.py --checkpoint checkpoints/latest.pt --batch-sweep
```

Outputs a Chrome trace JSON — open `chrome://tracing` and load the file to visualise per-operator timing.

---

## Project Structure Reference

```
new/
├── config.yaml              # All hyperparameters (single source of truth)
├── requirements.txt         # Python dependencies
├── data/                    # Data pipeline
│   ├── dataset.py           #   WIDER FACE loader + difficulty filtering
│   ├── augment.py           #   Mosaic, flip, jitter, crop, cutout
│   └── anchors.py           #   K-means anchor clustering
├── model/                   # Model architecture
│   ├── backbone.py          #   MobileNetV2 depthwise-separable CNN
│   ├── neck.py              #   Feature Pyramid Network
│   ├── head.py              #   Classification + regression + landmark heads
│   └── loss.py              #   Focal Loss + GIoU Loss + hard neg mining
├── training/                # Training loop
│   ├── train.py             #   AMP training with component-wise logging
│   ├── scheduler.py         #   Warmup cosine annealing + early stopping
│   ├── checkpoint.py        #   Top-3 by mAP checkpoint manager
│   └── metrics.py           #   mAP@0.5 + per-difficulty PR curves
├── inference/               # Inference
│   ├── detector.py          #   Soft-NMS + Platt calibration
│   ├── export.py            #   ONNX + TorchScript export
│   ├── inference_test.py    #   FPS bench + comparison grids
│   └── profiler.py          #   torch.profiler analysis
└── app/                     # Application layer
    ├── capture_photo.py     #   Quality-gated face enrollment
    ├── face_recog.py        #   SORT tracker + live recognition
    └── build_encodings.py   #   Incremental encoding manager
```
