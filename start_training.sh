#!/bin/bash
# =============================================================================
# YOLO Production Training Launcher
# =============================================================================
# Version: 3.0.0
# =============================================================================
#
# Production-ready training launcher with multi-GPU and transfer learning support
#
# Training Modes:
#   - raw: Train from scratch with custom architecture YAML
#   - pretrained: Use Ultralytics official COCO pretrained weights
#   - transfer: Transfer learning from custom weights (.pt file)
#
# =============================================================================

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$SCRIPT_DIR/training"
CONFIGS_DIR="$SCRIPT_DIR/configs"
SCRIPTS_DIR="$TRAINING_DIR/scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%d-%m-%Y %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
${CYAN}============================================================================
 YOLO Production Training System v3.0
============================================================================${NC}

Usage: $0 [OPTIONS]

${GREEN}BASIC OPTIONS:${NC}
    -c, --config CONFIG_FILE        Configuration file (required)
    -g, --gpu GPU_ID                GPU device(s): "0", "1", or "0,1" for multi-GPU
    -n, --name RUN_NAME             Custom run name (auto-generated if not provided)
    -r, --resume CHECKPOINT         Resume from checkpoint (last.pt or best.pt)
    -l, --log-level LEVEL           DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
    -h, --help                      Show this help message

${GREEN}TRANSFER LEARNING OPTIONS:${NC}
    -p, --pretrained-weights PATH   Path to pretrained weights for transfer learning
                                    Can be: /path/to/your/best.pt or yolo12n.pt
    -f, --freeze LAYERS             Number of layers to freeze (e.g., 10 for backbone)

${GREEN}TRAINING MODES:${NC}
    raw        - Train from scratch with custom architecture YAML (random weights)
                 Set training_mode: raw in config file
    pretrained - Use Ultralytics official COCO pretrained weights
                 Set training_mode: pretrained in config file
    transfer   - Transfer learning from custom weights
                 Set training_mode: transfer in config file OR use --pretrained-weights

${GREEN}EXAMPLES:${NC}
    # Raw training (from scratch)
    $0 --config raw_n.yaml --gpu 0

    # Pretrained mode (use COCO weights)
    $0 --config pretrained_s.yaml --gpu 0

    # Transfer learning from your trained model
    $0 --config transfer_s.yaml --gpu 0 \\
        --pretrained-weights training/results/.../weights/best.pt

    # OBB training
    $0 --config raw_obb_s.yaml --gpu 0

    # Multi-GPU training
    $0 --config raw_s.yaml --gpu 0,1

    # Resume from checkpoint
    $0 --config raw_s.yaml --resume training/results/.../weights/last.pt

${GREEN}AVAILABLE CONFIGS:${NC}
    Raw (from scratch):
        - raw_n.yaml        (Nano - 2.6M params)
        - raw_s.yaml        (Small - 9.3M params)
        - raw_m.yaml        (Medium - 20.2M params)

    Pretrained (COCO weights):
        - pretrained_n.yaml (Nano)
        - pretrained_s.yaml (Small)
        - pretrained_m.yaml (Medium)

    Transfer Learning:
        - transfer_n.yaml   (Nano)
        - transfer_s.yaml   (Small)
        - transfer_m.yaml   (Medium)

    OBB (Oriented Bounding Boxes):
        - raw_obb_n.yaml        (Nano - from scratch)
        - raw_obb_s.yaml        (Small - from scratch)
        - raw_obb_m.yaml        (Medium - from scratch)
        - transfer_obb_n.yaml   (Nano - transfer from OBB model)
        - transfer_obb_s.yaml   (Small - transfer from OBB model)
        - transfer_obb_m.yaml   (Medium - transfer from OBB model)

EOF
}

# Default values
CONFIG_FILE=""
RESUME_CHECKPOINT=""
GPU_ID="0"
LOG_LEVEL="INFO"
USE_DOCKER=false
PRETRAINED_WEIGHTS=""
FREEZE_LAYERS=""
RUN_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        -n|--name)
            RUN_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -p|--pretrained-weights)
            PRETRAINED_WEIGHTS="$2"
            shift 2
            ;;
        -f|--freeze)
            FREEZE_LAYERS="$2"
            shift 2
            ;;
        -d|--docker)
            USE_DOCKER=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_FILE" ]]; then
    error "Configuration file is required. Use --config or -c option."
    echo
    show_help
    exit 1
fi

# Validate log level
if [[ ! "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$ ]]; then
    error "Invalid log level: $LOG_LEVEL"
    exit 1
fi

# Validate GPU ID format (supports single GPU or multi-GPU like "0,1")
if [[ ! "$GPU_ID" =~ ^[0-9](,[0-9])*$ ]]; then
    error "Invalid GPU ID: $GPU_ID"
    info "Use single GPU (0 or 1) or multi-GPU format (0,1)"
    exit 1
fi

# Validate freeze layers if provided
if [[ -n "$FREEZE_LAYERS" ]]; then
    if ! [[ "$FREEZE_LAYERS" =~ ^[0-9]+$ ]]; then
        error "Invalid freeze layers: $FREEZE_LAYERS (must be a positive integer)"
        exit 1
    fi
fi

# Header
echo
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN} YOLO Production Training System v3.0${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo

# Resolve configuration file path
if [[ "$CONFIG_FILE" = /* ]] || [[ -f "$CONFIG_FILE" ]]; then
    CONFIG_PATH="$CONFIG_FILE"
else
    CONFIG_PATH="$CONFIGS_DIR/$CONFIG_FILE"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    error "Configuration file not found: $CONFIG_PATH"
    echo
    info "Available configurations:"
    find "$CONFIGS_DIR" -name "*.yaml" -printf "    %f\n" 2>/dev/null | sort || echo "    No configs found"
    exit 1
fi

# Configuration summary
info "Configuration:"
info "  Config: $(basename "$CONFIG_PATH")"
info "  GPU(s): $GPU_ID"
info "  Log Level: $LOG_LEVEL"
if [[ -n "$RUN_NAME" ]]; then
    info "  Run name: $RUN_NAME"
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    info "  Resume: $RESUME_CHECKPOINT"
fi
if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
    info "  Pretrained weights: $PRETRAINED_WEIGHTS"
fi
if [[ -n "$FREEZE_LAYERS" ]]; then
    info "  Freeze layers: $FREEZE_LAYERS"
fi
echo

# Validate resume checkpoint if provided
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
        error "Resume checkpoint not found: $RESUME_CHECKPOINT"
        echo
        info "Recent checkpoints:"
        find "$TRAINING_DIR/results" \( -name "last.pt" -o -name "best.pt" \) 2>/dev/null | sort -r | head -5 || echo "    No checkpoints found"
        exit 1
    fi
    CHECKPOINT_SIZE=$(du -h "$RESUME_CHECKPOINT" | cut -f1)
    info "  Checkpoint size: $CHECKPOINT_SIZE"
fi

# Validate pretrained weights if provided (unless it's an Ultralytics model name)
if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
    # Check if it's an Ultralytics model name (contains yolo)
    if [[ ! "$PRETRAINED_WEIGHTS" =~ yolo12 ]]; then
        # It's a file path, validate it exists
        if [[ ! -f "$PRETRAINED_WEIGHTS" ]]; then
            error "Pretrained weights not found: $PRETRAINED_WEIGHTS"
            echo
            info "Recent trained models:"
            find "$TRAINING_DIR/results" -name "best.pt" 2>/dev/null | sort -r | head -5 || echo "    No models found"
            exit 1
        fi
        WEIGHTS_SIZE=$(du -h "$PRETRAINED_WEIGHTS" | cut -f1)
        info "  Weights size: $WEIGHTS_SIZE"
    else
        info "  Using Ultralytics model: $PRETRAINED_WEIGHTS"
    fi
fi

# Function to run training
run_training() {
    log "Starting training..."
    log "Command: $*"
    echo

    # Execute training
    "$@"
    local exit_code=$?

    echo
    if [[ $exit_code -eq 0 ]]; then
        log "Training completed successfully!"
        info "Results: $TRAINING_DIR/results/"
        return 0
    else
        error "Training failed with exit code $exit_code"
        error "Check logs: $TRAINING_DIR/results/*/logs/"
        return $exit_code
    fi
}

# Docker execution
if [[ "$USE_DOCKER" == true ]]; then
    log "Running in Docker container..."

    # Find docker-compose.yml
    COMPOSE_FILE=""
    SEARCH_DIR="$SCRIPT_DIR"

    while [[ "$SEARCH_DIR" != "/" ]]; do
        if [[ -f "$SEARCH_DIR/docker-compose.yml" ]]; then
            COMPOSE_FILE="$SEARCH_DIR/docker-compose.yml"
            break
        fi
        SEARCH_DIR=$(dirname "$SEARCH_DIR")
    done

    if [[ -z "$COMPOSE_FILE" ]]; then
        error "docker-compose.yml not found"
        exit 1
    fi

    cd "$(dirname "$COMPOSE_FILE")"

    # Start container (try without sudo first)
    if ! docker compose up -d 2>/dev/null; then
        if ! sudo docker compose up -d; then
            error "Failed to start Docker container"
            exit 1
        fi
    fi

    sleep 3

    # Find container
    CONTAINER_NAME=$(docker compose ps -q yolo-training-env-dev 2>/dev/null || docker compose ps -q yolo-training-env 2>/dev/null)
    if [[ -z "$CONTAINER_NAME" ]]; then
        CONTAINER_NAME=$(sudo docker compose ps -q yolo-training-env-dev 2>/dev/null || sudo docker compose ps -q yolo-training-env 2>/dev/null)
    fi

    if [[ -z "$CONTAINER_NAME" ]]; then
        error "No training container found"
        exit 1
    fi

    # Build training command
    TRAINING_CMD="cd /workspace && python3 training/scripts/train_yolo.py --config $CONFIG_PATH --gpu $GPU_ID --log-level $LOG_LEVEL"
    if [[ -n "$RUN_NAME" ]]; then
        TRAINING_CMD="$TRAINING_CMD --name $RUN_NAME"
    fi
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        TRAINING_CMD="$TRAINING_CMD --resume $RESUME_CHECKPOINT"
    fi
    if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
        TRAINING_CMD="$TRAINING_CMD --pretrained-weights $PRETRAINED_WEIGHTS"
    fi
    if [[ -n "$FREEZE_LAYERS" ]]; then
        TRAINING_CMD="$TRAINING_CMD --freeze $FREEZE_LAYERS"
    fi

    # Execute in container
    docker exec "$CONTAINER_NAME" bash -c "$TRAINING_CMD" || sudo docker exec "$CONTAINER_NAME" bash -c "$TRAINING_CMD"

else
    # Local execution
    log "Running locally..."

    # Verify Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 not found"
        exit 1
    fi

    # Verify dependencies
    log "Checking dependencies..."
    if ! python3 -c "import ultralytics, torch, yaml" 2>/dev/null; then
        error "Missing Python packages"
        info "Install with: pip install ultralytics torch pyyaml"
        exit 1
    fi

    # Show GPU info
    if command -v nvidia-smi &> /dev/null; then
        log "Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
            info "  GPU $line"
        done
    else
        warning "nvidia-smi not found - using CPU"
    fi
    echo

    # Change to project directory
    cd "$SCRIPT_DIR"

    # Build command arguments array
    CMD_ARGS=(
        python3 training/scripts/train_yolo.py
        --config "$CONFIG_PATH"
        --gpu "$GPU_ID"
        --log-level "$LOG_LEVEL"
    )

    # Add optional arguments
    if [[ -n "$RUN_NAME" ]]; then
        CMD_ARGS+=(--name "$RUN_NAME")
    fi
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        CMD_ARGS+=(--resume "$RESUME_CHECKPOINT")
    fi
    if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
        CMD_ARGS+=(--pretrained-weights "$PRETRAINED_WEIGHTS")
    fi
    if [[ -n "$FREEZE_LAYERS" ]]; then
        CMD_ARGS+=(--freeze "$FREEZE_LAYERS")
    fi

    # Execute training
    run_training "${CMD_ARGS[@]}"
fi

echo
echo -e "${CYAN}============================================================================${NC}"
log "Session completed"
echo -e "${CYAN}============================================================================${NC}"
