#!/usr/bin/env python3
"""
=============================================================================
YOLO Production Training System
=============================================================================
Version: 3.0.0
=============================================================================

Production-ready training pipeline for YOLO12 models.
Supports custom architectures, multi-GPU training, and multiple training modes:
    - raw: Train from scratch with custom architecture YAML
    - pretrained: Use Ultralytics official COCO pretrained weights
    - transfer: Transfer learning from custom weights (.pt file)
"""

import os
import sys
import yaml
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
from pathlib import Path
import logging
import time
from typing import Dict, Optional, Any, Union
import argparse
import warnings
import json
from datetime import datetime
import traceback
import signal
import psutil
import gc

# Suppress only specific warnings (not all - we want to see important ones)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.amp')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
except ImportError as e:
    print(f"FATAL ERROR: Required dependencies missing: {e}")
    print("Install with: pip install ultralytics torch")
    sys.exit(1)

# Global variable for cleanup
trainer_instance: Optional['ProductionYOLOTrainer'] = None


def signal_handler(signum, frame):
    """Handle termination signals for proper cleanup"""
    signal_names = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}
    print(f"\nReceived {signal_names.get(signum, signum)}, cleaning up...")
    if trainer_instance:
        trainer_instance.emergency_cleanup()
    cleanup_processes()
    sys.exit(0)


def cleanup_processes():
    """Clean up all child processes"""
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        time.sleep(0.5)

        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        print(f"Warning: Process cleanup error: {e}")


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ProductionLogger:
    """Production-grade logging system with multiple levels and file output"""

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("YOLO_Production_Training")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.handlers.clear()

        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler for all logs
        file_handler = logging.FileHandler(
            self.log_dir / f"training_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # Error-only file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def critical(self, message: str):
        self.logger.critical(message)


class ConfigurationManager:
    """Production configuration management with validation and nested config support"""

    REQUIRED_FIELDS = ['model', 'data', 'epochs', 'batch_size', 'learning_rate']
    VALID_TRAINING_MODES = ['raw', 'pretrained', 'transfer']

    def __init__(self, config_path: str, cli_overrides: Dict[str, Any] = None):
        self.config_path = Path(config_path).resolve()
        self.cli_overrides = cli_overrides or {}
        self.config = self._load_and_validate_config()
        self._apply_cli_overrides()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate main training configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")

        # Validate training mode
        training_mode = config.get('training_mode', 'pretrained')
        if training_mode not in self.VALID_TRAINING_MODES:
            raise ValueError(f"Invalid training mode: {training_mode}. Must be one of: {self.VALID_TRAINING_MODES}")

        self._validate_paths(config)
        return config

    def _apply_cli_overrides(self):
        """Apply command line overrides to configuration"""
        # Override pretrained_weights if provided via CLI
        if self.cli_overrides.get('pretrained_weights'):
            if 'transfer' not in self.config:
                self.config['transfer'] = {}
            self.config['transfer']['weights'] = self.cli_overrides['pretrained_weights']
            # Auto-switch to transfer mode if weights provided
            if self.config.get('training_mode') != 'raw':
                self.config['training_mode'] = 'transfer'

        # Override freeze if provided via CLI
        if self.cli_overrides.get('freeze') is not None:
            if 'transfer' not in self.config:
                self.config['transfer'] = {}
            self.config['transfer']['freeze'] = self.cli_overrides['freeze']

    def _validate_paths(self, config: Dict[str, Any]):
        """Validate file paths in configuration"""
        base_path = self.config_path.parent
        training_mode = config.get('training_mode', 'pretrained')

        # Check model path
        model_path_str = config.get('model', '')
        if model_path_str:
            model_path = Path(model_path_str)
            if not model_path.is_absolute():
                model_path = base_path / model_path

            # For raw mode, model must be YAML architecture
            if training_mode == 'raw':
                if not str(model_path).endswith('.yaml'):
                    raise ValueError(f"Raw training mode requires .yaml architecture file, got: {model_path}")
                if not model_path.exists():
                    raise FileNotFoundError(f"Architecture file not found: {model_path}")

            # For pretrained mode, model can be .pt or model name (e.g., 'yolo12n.pt')
            elif training_mode == 'pretrained':
                # Allow Ultralytics model names like 'yolo12n.pt'
                if not model_path.exists() and not self._is_ultralytics_model(model_path_str):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

            # For transfer mode, validate weights path if specified
            elif training_mode == 'transfer':
                # Model can be YAML (architecture) or .pt (will extract architecture)
                if not model_path.exists() and not self._is_ultralytics_model(model_path_str):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

        # Validate transfer weights if specified
        transfer_config = config.get('transfer', {})
        if transfer_config and transfer_config.get('weights'):
            weights_path = Path(transfer_config['weights'])
            if not weights_path.is_absolute():
                weights_path = base_path / weights_path
            if not weights_path.exists() and not self._is_ultralytics_model(transfer_config['weights']):
                raise FileNotFoundError(f"Transfer weights not found: {weights_path}")

        # Check data path
        data_config = config.get('data', {})
        data_path_str = data_config.get('path', '') if isinstance(data_config, dict) else data_config

        if data_path_str:
            data_path = Path(data_path_str)
            if not data_path.is_absolute():
                data_path = base_path / data_path
            if not data_path.exists():
                raise FileNotFoundError(f"Data configuration not found: {data_path}")

    def _is_ultralytics_model(self, model_name: str) -> bool:
        """Check if model name is a valid Ultralytics model identifier"""
        return 'yolo12' in model_name.lower()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value safely"""
        try:
            current = self.config
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current
        except (KeyError, TypeError, AttributeError):
            return default

    def get_model_path(self) -> Path:
        """Get absolute path to model file"""
        model_path = Path(self.config.get('model', ''))
        if not model_path.is_absolute():
            model_path = self.config_path.parent / model_path
        return model_path.resolve()

    def get_data_path(self) -> Path:
        """Get absolute path to data configuration"""
        data_config = self.config.get('data', {})
        data_path_str = data_config.get('path', '') if isinstance(data_config, dict) else data_config

        data_path = Path(data_path_str)
        if not data_path.is_absolute():
            data_path = self.config_path.parent / data_path
        return data_path.resolve()

    def get_training_mode(self) -> str:
        """Get training mode"""
        return self.config.get('training_mode', 'pretrained')

    def is_raw_mode(self) -> bool:
        """Check if using raw training mode"""
        return self.get_training_mode() == 'raw'

    def is_transfer_mode(self) -> bool:
        """Check if using transfer learning mode"""
        return self.get_training_mode() == 'transfer'

    def is_pretrained_mode(self) -> bool:
        """Check if using pretrained mode"""
        return self.get_training_mode() == 'pretrained'

    def get_transfer_weights(self) -> Optional[str]:
        """Get path to transfer learning weights"""
        weights = self.get_nested('transfer', 'weights', default=None)
        if weights:
            weights_path = Path(weights)
            if not weights_path.is_absolute():
                weights_path = self.config_path.parent / weights_path
            return str(weights_path.resolve())
        return None

    def get_freeze_layers(self) -> Optional[int]:
        """Get number of layers to freeze for transfer learning"""
        return self.get_nested('transfer', 'freeze', default=None)


class ProductionYOLOTrainer:
    """Production-ready YOLO12 training system"""

    def __init__(self, config_path: str, resume_checkpoint: str = None, gpu: str = '0',
                 pretrained_weights: str = None, freeze: int = None, run_name: str = None):
        global trainer_instance
        trainer_instance = self

        self.start_time = datetime.now()
        self.config_path = config_path
        self.resume_checkpoint = resume_checkpoint
        self.gpu = gpu
        self.run_name = run_name
        self.temp_arch_file = None
        self.writer = None
        self.model = None

        # CLI overrides for transfer learning
        cli_overrides = {}
        if pretrained_weights:
            cli_overrides['pretrained_weights'] = pretrained_weights
        if freeze is not None:
            cli_overrides['freeze'] = freeze

        # Initialize components
        self.config_manager = ConfigurationManager(config_path, cli_overrides=cli_overrides)
        self.config = self.config_manager.config
        self._setup_directories()
        self.logger = ProductionLogger(log_dir=str(self.logs_dir))

        self.logger.info("=" * 60)
        self.logger.info("YOLO Production Training System v3.0")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Results: {self.results_dir}")
        self.logger.info(f"GPU(s): {self.gpu}")

        self._setup_device()
        self._setup_model()
        self._setup_tensorboard()
        self._setup_training_components()

        self.logger.info("Initialization complete")

    def _setup_directories(self):
        """Setup directory structure for production training"""
        self.base_dir = Path(__file__).parent.parent
        timestamp = self.start_time.strftime('%d-%m-%Y_%H-%M-%S')

        # Generate run name: custom or auto-generated
        if self.run_name:
            # Use custom name provided via CLI
            folder_name = f"{self.run_name}_{timestamp}"
        else:
            # Auto-generate descriptive name from config
            folder_name = self._generate_run_name(timestamp)

        self.results_dir = self.base_dir / "results" / folder_name
        self.logs_dir = self.results_dir / "logs"
        self.tensorboard_dir = self.results_dir / "tensorboard"

        for directory in [self.results_dir, self.logs_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _detect_data_task(self) -> str:
        """Read task type from data YAML ('detect' or 'obb')"""
        try:
            data_path = self.config_manager.get_data_path()
            with open(data_path, 'r') as f:
                data_cfg = yaml.safe_load(f)
            return data_cfg.get('task', 'detect')
        except Exception:
            return 'detect'

    def _generate_run_name(self, timestamp: str) -> str:
        """Generate descriptive run name from config: {dataset}_{model}_{mode}[_obb]_{timestamp}"""
        parts = []

        # Extract dataset name from data path
        data_config = self.config.get('data', {})
        data_path_str = data_config.get('path', '') if isinstance(data_config, dict) else data_config
        if data_path_str:
            data_name = Path(data_path_str).stem  # e.g., 'data_my_dataset'
            # Remove 'data_' prefix if present
            if data_name.startswith('data_'):
                data_name = data_name[5:]
            # Take first part before underscore for cleaner name
            dataset = data_name.split('_')[0] if '_' in data_name else data_name
            parts.append(dataset)

        # Extract model info from model path
        model_path_str = self.config.get('model', '')
        if model_path_str:
            model_name = Path(model_path_str).stem  # e.g., 'yolo12n'
            # Extract model variant (yolo12n, yolo12s, etc.)
            model_lower = model_name.lower()
            for pattern in ['yolo12']:
                if pattern in model_lower:
                    # Find the size suffix (n/s/m/l/x)
                    idx = model_lower.find(pattern) + len(pattern)
                    if idx < len(model_lower) and model_lower[idx] in 'nsmlx':
                        parts.append(f"{pattern}{model_lower[idx]}")
                    else:
                        parts.append(pattern)
                    break
            else:
                # Fallback: use first part of model name
                parts.append(model_name.split('_')[0])

        # Add training mode
        training_mode = self.config.get('training_mode', 'pretrained')
        parts.append(training_mode)

        # Append 'obb' when data task is OBB so runs are clearly distinguishable
        if self._detect_data_task() == 'obb':
            parts.append('obb')

        # Add timestamp
        parts.append(timestamp)

        return '_'.join(parts)

    def _setup_device(self):
        """Setup and validate GPU device(s)"""
        # Use device from config if specified there
        device_config = self.config_manager.get_nested('hardware', 'device', default='')
        if device_config:
            self.gpu = str(device_config)

        self.logger.info(f"Configuring device: {self.gpu if self.gpu else 'auto'}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {gpu_count}")

            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  GPU {i}: {name} ({memory:.1f} GB)")

            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = self.config_manager.get_nested(
                'hardware', 'deterministic', default=False
            )
        else:
            self.logger.warning("No GPU available, using CPU")

    def _setup_model(self):
        """Initialize YOLO model based on training mode"""
        # Check if resuming from checkpoint
        if self.resume_checkpoint:
            if not os.path.exists(self.resume_checkpoint):
                raise FileNotFoundError(f"Resume checkpoint not found: {self.resume_checkpoint}")

            self.logger.info(f"Resuming from checkpoint: {self.resume_checkpoint}")
            self.model = YOLO(self.resume_checkpoint)
            return

        model_path = self.config_manager.get_model_path()
        training_mode = self.config_manager.get_training_mode()

        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Training mode: {training_mode}")

        # === RAW MODE: Train from scratch with custom architecture ===
        if training_mode == 'raw':
            if not str(model_path).endswith('.yaml'):
                raise ValueError(f"Raw training mode requires .yaml architecture file, got: {model_path}")

            self.logger.info("Raw training: Building model from architecture YAML (random weights)")

            with open(model_path, 'r') as f:
                arch_cfg = yaml.safe_load(f)

            # Set number of classes from config or architecture
            nc = self.config_manager.get('nc') or arch_cfg.get('nc', 2)
            arch_cfg['nc'] = nc

            self.logger.info(f"  Architecture classes (nc): {nc}")

            # Create temporary architecture file with updated nc
            import tempfile
            temp_arch = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(arch_cfg, temp_arch, default_flow_style=False)
            temp_arch.close()

            self.model = YOLO(temp_arch.name)
            self.temp_arch_file = temp_arch.name

        # === PRETRAINED MODE: Use Ultralytics official weights ===
        elif training_mode == 'pretrained':
            self.logger.info("Pretrained mode: Using Ultralytics official COCO weights")

            # Model can be .pt file or model name like 'yolo12n.pt'
            model_str = str(model_path) if model_path.exists() else self.config_manager.get('model')
            self.logger.info(f"  Loading model: {model_str}")

            self.model = YOLO(model_str)

            # Log model info
            nc_config = self.config_manager.get('nc')
            if nc_config:
                self.logger.info(f"  Target classes (nc): {nc_config}")
                self.logger.info(f"  Note: Detection head will be reinitialized for {nc_config} classes")

        # === TRANSFER MODE: Transfer learning from custom weights ===
        elif training_mode == 'transfer':
            transfer_weights = self.config_manager.get_transfer_weights()
            freeze_layers = self.config_manager.get_freeze_layers()

            self.logger.info("Transfer learning mode: Loading custom pretrained weights")

            if transfer_weights:
                self.logger.info(f"  Transfer weights: {transfer_weights}")
                # Load model from weights file
                self.model = YOLO(transfer_weights)
            else:
                # If no transfer weights specified, load from model path
                model_str = str(model_path) if model_path.exists() else self.config_manager.get('model')
                self.logger.info(f"  Loading base model: {model_str}")
                self.model = YOLO(model_str)

            # Log freeze info
            if freeze_layers is not None:
                self.logger.info(f"  Freeze layers: {freeze_layers}")

            # Log nc handling
            nc_config = self.config_manager.get('nc')
            if nc_config:
                self.logger.info(f"  Target classes (nc): {nc_config}")
                self.logger.info(f"  Note: Detection head will be reinitialized for {nc_config} classes")

        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        if not TENSORBOARD_AVAILABLE:
            self.logger.info("TensorBoard: disabled (not installed, using Ultralytics built-in)")
            self.writer = None
            return
        try:
            self.writer = SummaryWriter(self.tensorboard_dir)
            self.logger.info(f"TensorBoard: {self.tensorboard_dir}")
        except Exception as e:
            self.logger.warning(f"TensorBoard setup failed: {e}")
            self.writer = None

    def _setup_training_components(self):
        """Initialize training components"""
        self.epochs = self.config_manager.get('epochs', 100)
        self.batch_size = self.config_manager.get('batch_size', 16)
        self.learning_rate = self.config_manager.get('learning_rate', 0.01)

        # Handle nested data config
        data_config = self.config_manager.get('data', {})
        self.image_size = data_config.get('image_size', 640) if isinstance(data_config, dict) else 640

        self.training_mode = self.config_manager.get_training_mode()
        self.optimizer_type = self.config_manager.get_nested('optimizer', 'type', default='AdamW')
        self.data_task = self._detect_data_task()

        self.logger.info("Training Configuration:")
        self.logger.info(f"  Mode: {self.training_mode}")
        self.logger.info(f"  Task: {self.data_task}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Learning rate: {self.learning_rate}")
        self.logger.info(f"  Image size: {self.image_size}")
        self.logger.info(f"  Optimizer: {self.optimizer_type}")

        if self.data_task == 'obb':
            self.logger.info("  OBB mode: model architecture must use OBBDetect head")

    def train(self) -> bool:
        """Execute production training"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting Training")
            self.logger.info("=" * 60)

            train_args = self._prepare_training_args()
            self._save_training_config(train_args)

            results = self.model.train(**train_args)
            self._process_training_results(results)

            self.logger.info("Training completed successfully")
            return True

        except Exception as e:
            self.logger.critical(f"Training failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False

        finally:
            self._cleanup()

    def _prepare_training_args(self) -> Dict[str, Any]:
        """Prepare training arguments with proper config extraction"""
        data_path = self.config_manager.get_data_path()

        # Get nested configs
        data_config = self.config_manager.get('data', {})
        aug = self.config_manager.get('augmentation', {})
        loss = self.config_manager.get('loss', {})
        logging_cfg = self.config_manager.get('logging', {})

        workers = data_config.get('workers', 8) if isinstance(data_config, dict) else 8
        cache = data_config.get('cache', False) if isinstance(data_config, dict) else False

        # === Determine pretrained flag based on training mode ===
        # pretrained can be: True (COCO), False (scratch), or "path/to/weights.pt" (transfer)
        training_mode = self.config_manager.get_training_mode()

        if training_mode == 'raw':
            # Raw mode: train from scratch, no pretrained weights
            pretrained = False
        elif training_mode == 'pretrained':
            # Pretrained mode: use Ultralytics COCO weights
            pretrained = True
        elif training_mode == 'transfer':
            # Transfer mode: use custom weights path OR True for base model weights
            transfer_weights = self.config_manager.get_transfer_weights()
            if transfer_weights:
                pretrained = transfer_weights
            else:
                pretrained = True
        else:
            pretrained = False

        # Get freeze layers for transfer learning
        freeze_layers = self.config_manager.get_freeze_layers()

        args = {
            # Core settings
            'data': str(data_path),
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.image_size,

            # Optimizer settings
            'lr0': self.learning_rate,
            'lrf': self.config_manager.get('lrf', 0.01),
            'momentum': self.config_manager.get('momentum', 0.937),
            'weight_decay': self.config_manager.get('weight_decay', 0.0005),
            'warmup_epochs': self.config_manager.get('warmup_epochs', 3.0),
            'warmup_momentum': self.config_manager.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config_manager.get('warmup_bias_lr', 0.1),
            'optimizer': self.optimizer_type,

            # Loss weights
            'box': loss.get('box_loss_gain', 7.5),
            'cls': loss.get('cls_loss_gain', 0.5),
            'dfl': loss.get('dfl_loss_gain', 1.5),

            # Augmentation (from config)
            'hsv_h': aug.get('hsv_h', 0.015),
            'hsv_s': aug.get('hsv_s', 0.7),
            'hsv_v': aug.get('hsv_v', 0.4),
            'degrees': aug.get('degrees', 0.0),
            'translate': aug.get('translate', 0.1),
            'scale': aug.get('scale', 0.5),
            'shear': aug.get('shear', 0.0),
            'perspective': aug.get('perspective', 0.0),
            'flipud': aug.get('flipud', 0.0),
            'fliplr': aug.get('fliplr', 0.5),
            'mosaic': aug.get('mosaic', 1.0),
            'mixup': aug.get('mixup', 0.0),
            'copy_paste': aug.get('copy_paste', 0.0),
            'erasing': aug.get('erasing', 0.4),
            'close_mosaic': aug.get('close_mosaic', 10),

            # Training settings
            'device': self.gpu,
            'workers': workers,
            'cache': cache,
            'pretrained': pretrained,
            'patience': self.config_manager.get('patience', 100),
            'cos_lr': self.config_manager.get('scheduler', 'cosine') == 'cosine',
            'amp': self.config_manager.get_nested('optimization', 'amp', 'enabled', default=True),
            'deterministic': self.config_manager.get_nested('hardware', 'deterministic', default=False),

            # Validation
            'val': self.config_manager.get_nested('validation', 'enabled', default=True),
            'iou': self.config_manager.get_nested('validation', 'iou_threshold', default=0.7),
            'max_det': self.config_manager.get_nested('validation', 'max_detections', default=300),

            # Output settings
            'project': str(self.base_dir),
            'name': f'results/{self.results_dir.name}',
            'save_period': logging_cfg.get('save_period', 100),
            'plots': logging_cfg.get('plots', True),
            'verbose': logging_cfg.get('verbose', True),
            'exist_ok': True,
            'seed': self.config_manager.get_nested('production', 'seed', default=42),

            # Resume
            'resume': bool(self.resume_checkpoint),
        }

        # Add freeze parameter if specified (for transfer learning)
        if freeze_layers is not None:
            args['freeze'] = freeze_layers

        # Log key settings
        self.logger.info("Training Arguments:")
        self.logger.info(f"  Training mode: {training_mode}")
        self.logger.info(f"  Pretrained: {pretrained}")
        if freeze_layers is not None:
            self.logger.info(f"  Freeze layers: {freeze_layers}")
        self.logger.info(f"  Loss weights - box: {args['box']}, cls: {args['cls']}, dfl: {args['dfl']}")
        self.logger.info(f"  Augmentation - degrees: {args['degrees']}, flipud: {args['flipud']}, fliplr: {args['fliplr']}")
        self.logger.info(f"  Device: {args['device']}")

        # Filter out None values
        return {k: v for k, v in args.items() if v is not None}

    def _save_training_config(self, args: Dict[str, Any]):
        """Save training configuration to file"""
        config_path = self.results_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(args, f, indent=2, default=str)
        self.logger.info(f"Config saved: {config_path}")

    def _process_training_results(self, results):
        """Process and save training results with metrics extraction"""
        try:
            # Extract metrics if available
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = dict(results.results_dict)
            elif hasattr(results, 'maps'):
                metrics = {
                    'mAP50': float(results.maps[0]) if len(results.maps) > 0 else None,
                    'mAP50-95': float(results.maps.mean()) if len(results.maps) > 0 else None,
                }

            # Calculate duration
            duration = datetime.now() - self.start_time

            results_summary = {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_human': str(duration),
                'training_mode': self.training_mode,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'image_size': self.image_size,
                'model_path': str(self.config_manager.get_model_path()),
                'data_path': str(self.config_manager.get_data_path()),
                'results_directory': str(self.results_dir),
                'gpu': self.gpu,
                'metrics': metrics,
            }

            results_path = self.results_dir / "training_summary.json"
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)

            self.logger.info(f"Results saved: {results_path}")
            if metrics:
                self.logger.info(f"Final metrics: {metrics}")

        except Exception as e:
            self.logger.error(f"Results processing failed: {e}")

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.writer:
                self.writer.close()

            # Clean up temporary architecture file
            if self.temp_arch_file and os.path.exists(self.temp_arch_file):
                os.unlink(self.temp_arch_file)
                self.logger.debug(f"Cleaned up temp file: {self.temp_arch_file}")

            if self.model:
                del self.model
                self.model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            cleanup_processes()

            duration = datetime.now() - self.start_time
            self.logger.info(f"Total duration: {duration}")

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def emergency_cleanup(self):
        """Emergency cleanup for signal handlers"""
        try:
            self._cleanup()
        except Exception as e:
            print(f"Emergency cleanup error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YOLO Production Training System v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  raw        - Train from scratch with custom architecture YAML (random weights)
  pretrained - Use Ultralytics official COCO pretrained weights
  transfer   - Transfer learning from custom weights (.pt file)

Examples:
  # Raw training (from scratch)
  python train_yolo.py --config ../configs/raw_n.yaml --gpu 0

  # Pretrained mode (COCO weights)
  python train_yolo.py --config ../configs/pretrained_s.yaml --gpu 0

  # Transfer learning from your trained model
  python train_yolo.py --config ../configs/transfer_s.yaml --gpu 0 \\
      --pretrained-weights /path/to/your/best.pt --freeze 10

  # Multi-GPU training
  python train_yolo.py --config ../configs/raw_s.yaml --gpu 0,1

  # Resume from checkpoint
  python train_yolo.py --config ../configs/raw_s.yaml --resume path/to/last.pt
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID(s): "0", "1", or "0,1" for multi-GPU')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                        help='Path to pretrained weights for transfer learning (.pt file or Ultralytics model name)')
    parser.add_argument('--freeze', type=int, default=None,
                        help='Number of layers to freeze for transfer learning (e.g., 10 for backbone)')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom run name (auto-generated if not provided: {dataset}_{model}_{mode}_{timestamp})')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO Production Training System v3.0")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"GPU(s): {args.gpu}")
    if args.pretrained_weights:
        print(f"Pretrained weights: {args.pretrained_weights}")
    if args.freeze is not None:
        print(f"Freeze layers: {args.freeze}")
    if args.name:
        print(f"Run name: {args.name}")
    print()

    try:
        trainer = ProductionYOLOTrainer(
            args.config,
            resume_checkpoint=args.resume,
            gpu=args.gpu,
            pretrained_weights=args.pretrained_weights,
            freeze=args.freeze,
            run_name=args.name
        )
        success = trainer.train()

        if success:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Training failed. Check logs for details.")
            print("=" * 60)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
