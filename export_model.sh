#!/bin/bash
# =============================================================================
# YOLO Model Export Script
# =============================================================================
# Version: 1.0.0
# =============================================================================
#
# Supports: ONNX, TensorRT, NCNN (Raspberry Pi), TFLite (Edge devices)
#
# Usage: ./export_model.sh <path_to_best.pt> [image_size] [options]
#
# Options:
#   --int8      Enable INT8 quantization (smaller, slightly less accurate)
#   --ncnn      Export to NCNN format (best for Raspberry Pi)
#   --tflite    Export to TFLite format (good for ARM/edge devices)
#
# Examples:
#   ./export_model.sh weights/best.pt                    # 640, ONNX+TensorRT
#   ./export_model.sh weights/best.pt 320                # 320, ONNX+TensorRT
#   ./export_model.sh weights/best.pt 320 --ncnn         # 320, NCNN (for RPi)
#   ./export_model.sh weights/best.pt 256 --tflite       # 256, TFLite
#   ./export_model.sh weights/best.pt 320 --ncnn --int8  # 320, NCNN INT8
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "============================================================================"
echo " YOLO Model Export - Multi-Format"
echo "============================================================================"

# Check argument
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: No model path provided${NC}"
    echo ""
    echo "Usage: $0 <path_to_weights.pt> [image_size] [options]"
    echo ""
    echo "Arguments:"
    echo "  path_to_weights.pt   Path to PyTorch model file (required)"
    echo "  image_size           Export size: 192, 256, 320, 416, 512, 640 (default: 640)"
    echo ""
    echo "Options:"
    echo "  --int8      INT8 quantization (smaller model, needs calibration)"
    echo "  --ncnn      Export to NCNN format (best for Raspberry Pi)"
    echo "  --tflite    Export to TFLite format (good for ARM/mobile)"
    echo ""
    echo "Examples:"
    echo "  $0 training/results/<run>/weights/best.pt"
    echo "  $0 training/results/<run>/weights/best.pt 320"
    echo "  $0 training/results/<run>/weights/best.pt 320 --ncnn"
    echo "  $0 training/results/<run>/weights/best.pt 256 --tflite --int8"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# Parse arguments
IMGSZ=640
INT8_FLAG=""
NCNN_FLAG=""
TFLITE_FLAG=""

for arg in "${@:2}"; do
    case $arg in
        --int8)
            INT8_FLAG="true"
            ;;
        --ncnn)
            NCNN_FLAG="true"
            ;;
        --tflite)
            TFLITE_FLAG="true"
            ;;
        [0-9]*)
            IMGSZ=$arg
            ;;
    esac
done

# Validate image size
valid_sizes=(192 256 320 416 512 640)
if [[ ! " ${valid_sizes[@]} " =~ " ${IMGSZ} " ]]; then
    echo -e "${YELLOW}Warning: Non-standard size ${IMGSZ}, using anyway${NC}"
fi

# Determine precision
if [ -n "$INT8_FLAG" ]; then
    PRECISION="INT8"
    PRECISION_DESC="INT8 quantized (smallest)"
else
    PRECISION="FP16"
    PRECISION_DESC="FP16 half precision"
fi

# Determine format
if [ -n "$NCNN_FLAG" ]; then
    FORMAT="NCNN"
    FORMAT_DESC="NCNN (optimized for Raspberry Pi / ARM CPU)"
elif [ -n "$TFLITE_FLAG" ]; then
    FORMAT="TFLite"
    FORMAT_DESC="TensorFlow Lite (for ARM/edge devices)"
else
    FORMAT="ONNX+TensorRT"
    FORMAT_DESC="ONNX + TensorRT (for NVIDIA GPU)"
fi

echo -e "${GREEN}✓ Model found: $MODEL_PATH${NC}"
echo -e "${CYAN}✓ Export size: ${IMGSZ}x${IMGSZ}${NC}"
echo -e "${CYAN}✓ Format: ${FORMAT_DESC}${NC}"
echo -e "${CYAN}✓ Precision: ${PRECISION_DESC}${NC}"

# Get model directory
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_PATH" .pt)

# Create size-specific suffix
if [ "$IMGSZ" != "640" ]; then
    SIZE_SUFFIX="_${IMGSZ}"
else
    SIZE_SUFFIX=""
fi

echo -e "${GREEN}✓ Export directory: $MODEL_DIR${NC}"
echo ""

# Activate virtual environment (if exists)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Verify ultralytics is available
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo -e "${RED}ERROR: ultralytics not installed${NC}"
    echo "Install with: pip install ultralytics"
    exit 1
fi

# ============================================================================
# NCNN Export (Raspberry Pi)
# ============================================================================
if [ -n "$NCNN_FLAG" ]; then
    echo "============================================================================"
    echo " Exporting to NCNN (${IMGSZ}x${IMGSZ})..."
    echo " Target: Raspberry Pi / ARM CPU"
    echo "============================================================================"

    python3 << EOF
from ultralytics import YOLO
import shutil
import os

model = YOLO('$MODEL_PATH')

print("Exporting to NCNN format...")
print("This creates .param and .bin files for ncnn inference")

ncnn_path = model.export(
    format='ncnn',
    imgsz=$IMGSZ,
    half=$( [ -z "$INT8_FLAG" ] && echo "True" || echo "False" ),
)
print(f"✓ NCNN export complete: {ncnn_path}")

# The export creates a folder, rename if needed
if '$SIZE_SUFFIX':
    new_name = str(ncnn_path).replace('_ncnn_model', '${SIZE_SUFFIX}_ncnn_model')
    if os.path.exists(new_name):
        shutil.rmtree(new_name)
    shutil.move(ncnn_path, new_name)
    print(f"✓ Renamed to: {new_name}")
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ NCNN export successful${NC}"
    else
        echo -e "${RED}✗ NCNN export failed${NC}"
    fi

# ============================================================================
# TFLite Export (Edge devices)
# ============================================================================
elif [ -n "$TFLITE_FLAG" ]; then
    echo "============================================================================"
    echo " Exporting to TFLite (${IMGSZ}x${IMGSZ})..."
    echo " Target: Edge devices / ARM / Coral TPU"
    echo "============================================================================"

    python3 << EOF
from ultralytics import YOLO
import shutil
import os

model = YOLO('$MODEL_PATH')

print("Exporting to TensorFlow Lite format...")

tflite_path = model.export(
    format='tflite',
    imgsz=$IMGSZ,
    int8=$( [ -n "$INT8_FLAG" ] && echo "True" || echo "False" ),
)
print(f"✓ TFLite export complete: {tflite_path}")

# Rename with size suffix
if '$SIZE_SUFFIX':
    new_name = str(tflite_path).replace('.tflite', '${SIZE_SUFFIX}.tflite')
    new_name = new_name.replace('_saved_model', '${SIZE_SUFFIX}_saved_model')
    if os.path.exists(new_name):
        if os.path.isdir(new_name):
            shutil.rmtree(new_name)
        else:
            os.remove(new_name)
    shutil.move(tflite_path, new_name)
    print(f"✓ Renamed to: {new_name}")
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ TFLite export successful${NC}"
    else
        echo -e "${RED}✗ TFLite export failed${NC}"
    fi

# ============================================================================
# Default: ONNX + TensorRT (NVIDIA GPU)
# ============================================================================
else
    # Export to ONNX
    echo "============================================================================"
    echo " Exporting to ONNX (${IMGSZ}x${IMGSZ})..."
    echo "============================================================================"

    python3 << EOF
from ultralytics import YOLO
import shutil
import os

model = YOLO('$MODEL_PATH')

print("Exporting to ONNX format...")
onnx_path = model.export(
    format='onnx',
    imgsz=$IMGSZ,
    opset=17,
    simplify=True,
    dynamic=False,
)
print(f"✓ ONNX export complete: {onnx_path}")

# Rename with size suffix if not 640
if '$SIZE_SUFFIX':
    new_name = onnx_path.replace('.onnx', '${SIZE_SUFFIX}.onnx')
    if os.path.exists(new_name):
        os.remove(new_name)
    shutil.move(onnx_path, new_name)
    print(f"✓ Renamed to: {new_name}")
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ONNX export successful${NC}"
    else
        echo -e "${RED}✗ ONNX export failed${NC}"
        exit 1
    fi

    echo ""

    # Export to TensorRT
    echo "============================================================================"
    echo " Exporting to TensorRT Engine (${IMGSZ}x${IMGSZ}, ${PRECISION})..."
    echo "============================================================================"

    if [ -n "$INT8_FLAG" ]; then
        python3 << EOF
from ultralytics import YOLO
import shutil
import os

model = YOLO('$MODEL_PATH')

print("Exporting to TensorRT engine (INT8 quantization)...")
engine_path = model.export(
    format='engine',
    imgsz=$IMGSZ,
    half=False,
    int8=True,
    workspace=8,
    device=0,
)
print(f"✓ TensorRT INT8 export complete: {engine_path}")

if '$SIZE_SUFFIX':
    new_name = engine_path.replace('.engine', '${SIZE_SUFFIX}_int8.engine')
    if os.path.exists(new_name):
        os.remove(new_name)
    shutil.move(engine_path, new_name)
    print(f"✓ Renamed to: {new_name}")
EOF
    else
        python3 << EOF
from ultralytics import YOLO
import shutil
import os

model = YOLO('$MODEL_PATH')

print("Exporting to TensorRT engine (FP16 precision)...")
engine_path = model.export(
    format='engine',
    imgsz=$IMGSZ,
    half=True,
    int8=False,
    workspace=8,
    device=0,
)
print(f"✓ TensorRT FP16 export complete: {engine_path}")

if '$SIZE_SUFFIX':
    new_name = engine_path.replace('.engine', '${SIZE_SUFFIX}.engine')
    if os.path.exists(new_name):
        os.remove(new_name)
    shutil.move(engine_path, new_name)
    print(f"✓ Renamed to: {new_name}")
EOF
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ TensorRT export successful${NC}"
    else
        echo -e "${YELLOW}⚠ TensorRT export failed (may require TensorRT installation)${NC}"
    fi
fi

echo ""
echo "============================================================================"
echo " Export Summary"
echo "============================================================================"
echo -e "Model:     $MODEL_PATH"
echo -e "Size:      ${IMGSZ}x${IMGSZ}"
echo -e "Format:    ${FORMAT}"
echo -e "Precision: ${PRECISION}"
echo -e "Location:  ${MODEL_DIR}/"
echo ""
echo "Exported files:"
ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || true
ls -lh "$MODEL_DIR"/*.engine 2>/dev/null || true
ls -lh "$MODEL_DIR"/*.tflite 2>/dev/null || true
ls -lhd "$MODEL_DIR"/*_ncnn_model 2>/dev/null || true
ls -lhd "$MODEL_DIR"/*_saved_model 2>/dev/null || true
echo ""
echo "============================================================================"
echo -e "${GREEN}✓ Export process completed${NC}"
echo "============================================================================"
