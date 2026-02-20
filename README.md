# YOLO12 Aerial Detection

Config-driven YOLO12 training pipeline for aerial/drone object detection. Supports 3 training modes, 3 model sizes, AABB and OBB detection, multi-GPU training, and multi-format export.

## Features

- **3 Training Modes:** Raw (from scratch), Pretrained (COCO), Transfer Learning
- **3 Model Sizes:** Nano (2.6M), Small (9.3M), Medium (20.2M)
- **OBB Support:** Oriented bounding box detection with AABB-to-OBB converter
- **Config-Driven:** One YAML config controls everything - no code changes needed
- **Multi-GPU:** Single or multi-GPU training
- **Export:** ONNX, TensorRT, NCNN (Raspberry Pi), TFLite (edge devices)

## Project Structure

```
yolo12-aerial-detection/
├── architectures/           # Model architecture definitions (6 files)
│   ├── yolo12n.yaml         # Nano - AABB detection
│   ├── yolo12s.yaml         # Small - AABB detection
│   ├── yolo12m.yaml         # Medium - AABB detection
│   ├── yolo12n_obb.yaml     # Nano - OBB detection
│   ├── yolo12s_obb.yaml     # Small - OBB detection
│   └── yolo12m_obb.yaml     # Medium - OBB detection
├── configs/                 # Training configurations (15 files)
│   ├── raw_n/s/m.yaml       # Train from scratch
│   ├── pretrained_n/s/m.yaml# Fine-tune COCO weights
│   ├── transfer_n/s/m.yaml  # Transfer learning
│   ├── raw_obb_n/s/m.yaml   # OBB from scratch
│   └── transfer_obb_n/s/m.yaml # OBB transfer learning
├── datasets/
│   └── example/             # Example dataset structure
│       ├── data.yaml
│       ├── train/images/
│       ├── train/labels/
│       ├── val/images/
│       └── val/labels/
├── tools/                   # Utility scripts
│   ├── split_dataset.py     # Split raw data into train/val
│   ├── resplit_dataset.py   # Resplit by frame ranges
│   ├── validate_dataset.py  # Validate dataset before training
│   └── convert_aabb_to_obb.py  # Convert AABB labels to OBB
├── training/
│   ├── scripts/train_yolo.py   # Training engine
│   └── results/                # Training outputs (auto-created)
├── start_training.sh        # Training launcher
├── export_model.sh          # Model export script
└── requirements.txt
```

## Setup

```bash
# Clone repository
git clone https://github.com/sergiibertov/yolo12-aerial-detection.git
cd yolo12-aerial-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Dataset

### YOLO Format

Your dataset must follow this structure:

```
datasets/my_dataset/
├── data.yaml
├── train/
│   ├── images/    # .jpg, .png, etc.
│   └── labels/    # .txt files (one per image)
└── val/
    ├── images/
    └── labels/
```

### Label Format

**AABB (standard):** `class_id center_x center_y width height` (normalized 0-1)

**OBB (oriented):** `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized 0-1)

### data.yaml Template

```yaml
path: .
train: train/images
val: val/images

nc: 2
names: ['car', 'truck']

# For OBB datasets, add:
# task: obb
```

### Split Raw Data

```bash
python tools/split_dataset.py \
  --source /path/to/raw/images_and_labels \
  --output datasets/my_dataset \
  --ratio 0.8 \
  --classes "car,truck"
```

### Validate Dataset

```bash
python tools/validate_dataset.py --data datasets/my_dataset/data.yaml
```

## Training

### 1. Update Config

Edit the config file to point to your dataset:

```yaml
# In configs/raw_s.yaml (or whichever config you choose)
data:
  path: ../datasets/my_dataset/data.yaml
nc: 2  # Update to match your number of classes
```

### 2. Start Training

**Raw training (from scratch) - best for thermal/grayscale:**
```bash
./start_training.sh --config raw_n.yaml --gpu 0
./start_training.sh --config raw_s.yaml --gpu 0
```

**Pretrained (COCO weights) - best for RGB imagery:**
```bash
./start_training.sh --config pretrained_s.yaml --gpu 0
```

**Transfer learning - best when you have a previously trained model:**
```bash
# First update model path in transfer_s.yaml, then:
./start_training.sh --config transfer_s.yaml --gpu 0
```

**With custom run name:**
```bash
./start_training.sh --config raw_s.yaml --gpu 0 --name "experiment_v1"
```

**Resume training:**
```bash
./start_training.sh --config raw_s.yaml --resume training/results/<run>/weights/last.pt --gpu 0
```

**Multi-GPU:**
```bash
./start_training.sh --config raw_s.yaml --gpu 0,1
```

## OBB Training

### Convert AABB Dataset to OBB

```bash
# Convert to new directory
python tools/convert_aabb_to_obb.py datasets/my_dataset --output datasets/my_dataset_obb

# With debug visualization
python tools/convert_aabb_to_obb.py datasets/my_dataset --output datasets/my_dataset_obb --visualize
```

### Train OBB Model

```bash
# Update data path in config, then:
./start_training.sh --config raw_obb_s.yaml --gpu 0
```

**OBB rules:**
- Architecture must use `OBB` head (not `Detect`)
- `data.yaml` must contain `task: obb`
- Cannot transfer from AABB model to OBB model (incompatible heads)

## Export

```bash
# ONNX + TensorRT (default, 640px)
./export_model.sh training/results/<run>/weights/best.pt

# Custom size
./export_model.sh training/results/<run>/weights/best.pt 320

# NCNN (Raspberry Pi)
./export_model.sh training/results/<run>/weights/best.pt 320 --ncnn

# TFLite (edge devices)
./export_model.sh training/results/<run>/weights/best.pt 256 --tflite

# INT8 quantization
./export_model.sh training/results/<run>/weights/best.pt 320 --ncnn --int8
```

## Configuration System

Each config YAML controls the entire training run. Key sections:

```
Core         -> training_mode, model, nc, epochs
Batch        -> batch_size, val_batch_size
Dataset      -> data.path, data.image_size, data.cache
Optimizer    -> learning_rate, lrf, weight_decay
Augmentation -> hsv_h/s/v, mosaic, mixup, degrees
Hardware     -> device, benchmark
Validation   -> interval, iou_threshold
```

## Training Modes

| Mode         | Use Case                        | Weights          | LR    |
|:-------------|:--------------------------------|:-----------------|:------|
| `raw`        | Thermal/grayscale, custom domain | None (scratch)  | 0.001 |
| `pretrained` | RGB real-world imagery          | COCO weights     | 0.001 |
| `transfer`   | New dataset, same sensor domain | Your trained .pt | 0.001 |

> **WARNING:** LR 0.01 for pretrained causes NaN loss. Always use 0.001.

## Augmentation Presets

**RGB** (full color augmentation):

```yaml
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.6
mosaic: 1.0
mixup: 0.2
```

**Thermal / Grayscale** (no color augmentation):

```yaml
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.2
mosaic: 1.0
mixup: 0.0
```

## Batch Size Guidelines

Reference values for a 32 GB GPU at 640 px image size.

Formula for other image sizes: `batch_new = batch_640 / (new_size / 640)^2`

| Model  | Batch @ 640 |
|:-------|:-----------:|
| Nano   | 160         |
| Small  | 64          |
| Medium | 32          |

| Image Size | Small Batch | Memory Multiplier |
|:-----------|:-----------:|:-----------------:|
| 640        | 64          | 1.0x              |
| 832        | 38          | 1.7x              |
| 960        | 24          | 2.25x             |

## Epoch Guidelines

Official YOLO12 trains from scratch on COCO (330K images) for 600 epochs across all model sizes. Our configs use 800 for raw as a safety margin for smaller custom datasets. All model sizes (Nano/Small/Medium) use the same epoch count — this matches the official approach.

| Mode         | Epochs | Reasoning                                              |
|:-------------|:------:|:-------------------------------------------------------|
| Raw          | 800    | From scratch, no prior knowledge                       |
| Raw OBB      | 800    | From scratch + rotation learning                       |
| Pretrained   | 300    | COCO features transfer, adapting to new domain         |
| Transfer     | 300    | Domain features already learned, fine-tuning only      |
| Transfer OBB | 300    | Same as transfer                                       |

Early stopping is enabled in all configs (`patience`: 150 for raw, 80 for pretrained, 60 for transfer). Training stops automatically when mAP plateaus — the epoch count is a maximum, not a target.

**Adjusting epochs for your dataset:**

| Dataset Size      | Recommendation                                |
|:------------------|:----------------------------------------------|
| < 1,000 images    | Reduce by ~30% (risk of overfitting)          |
| 1,000 - 10,000    | Default values should work well               |
| > 10,000 images   | Can increase to 1000-1200 for raw             |
| Still improving?   | Increase epochs if mAP is rising at the limit |

## Available Configs

### AABB Detection

| Config               | Mode       | Model  | Batch | Epochs |
|:---------------------|:-----------|:-------|:-----:|:------:|
| `raw_n.yaml`         | Raw        | Nano   | 160   | 800    |
| `raw_s.yaml`         | Raw        | Small  | 64    | 800    |
| `raw_m.yaml`         | Raw        | Medium | 32    | 800    |
| `pretrained_n.yaml`  | Pretrained | Nano   | 160   | 300    |
| `pretrained_s.yaml`  | Pretrained | Small  | 64    | 300    |
| `pretrained_m.yaml`  | Pretrained | Medium | 32    | 300    |
| `transfer_n.yaml`    | Transfer   | Nano   | 160   | 300    |
| `transfer_s.yaml`    | Transfer   | Small  | 64    | 300    |
| `transfer_m.yaml`    | Transfer   | Medium | 32    | 300    |

### OBB Detection

No `pretrained` mode for OBB — COCO weights use a `Detect` head which is incompatible
with the `OBB` head. For transfer learning, first train a raw OBB model, then use its
weights as a base with the `transfer_obb_*` configs.

| Config                   | Mode         | Model  | Batch | Epochs |
|:-------------------------|:-------------|:-------|:-----:|:------:|
| `raw_obb_n.yaml`         | Raw OBB      | Nano   | 160   | 800    |
| `raw_obb_s.yaml`         | Raw OBB      | Small  | 64    | 800    |
| `raw_obb_m.yaml`         | Raw OBB      | Medium | 32    | 800    |
| `transfer_obb_n.yaml`    | Transfer OBB | Nano   | 160   | 300    |
| `transfer_obb_s.yaml`    | Transfer OBB | Small  | 64    | 300    |
| `transfer_obb_m.yaml`    | Transfer OBB | Medium | 32    | 300    |

## Tools

### split_dataset.py - Split raw export into train/val

```bash
python tools/split_dataset.py --source <dir> --output <dir> --ratio 0.8 --classes "a,b"
```

### resplit_dataset.py - Resplit by frame range (prevent data leakage)

```bash
python tools/resplit_dataset.py --source <dataset> --val-start 300 --val-end 480
```

### validate_dataset.py - Check dataset before training

```bash
python tools/validate_dataset.py --data <data.yaml>
```

### convert_aabb_to_obb.py - Convert AABB labels to OBB

```bash
python tools/convert_aabb_to_obb.py <dataset> --output <dir> [--visualize]
```

## Troubleshooting

| Issue                 | Cause                    | Fix                                   |
|:----------------------|:-------------------------|:--------------------------------------|
| NaN loss              | LR too high for pretrained | Use LR 0.001, not 0.01             |
| OOM (out of memory)   | Batch too large          | Reduce batch or image size            |
| Poor thermal results  | Using COCO pretrained    | Use raw or transfer mode              |
| Small objects missed  | Image size too small     | Increase to 832 or 960               |
| OBB training fails    | Wrong architecture       | Use OBB architecture (`*_obb.yaml`)   |
| OBB transfer fails    | Source model is AABB     | Must transfer from OBB model          |

## Contact

For questions, bug reports, or suggestions:

- **Email:** sergiibertov@gmail.com
- **GitHub Issues:** [Open an issue](https://github.com/Sergey-0909/yolo12-aerial-detection/issues)
