#!/usr/bin/env python3
"""
=============================================================================
Dataset Validation Utility
=============================================================================
Version: 1.0.0
=============================================================================

Validates YOLO dataset structure, annotations, and image quality before training.

Usage:
    python validate_dataset.py --data datasets/example/data.yaml
    python validate_dataset.py --data datasets/my_dataset/data.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from collections import Counter
import random

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Install required packages: pip install pillow numpy")
    sys.exit(1)


class DatasetValidator:
    """Validates YOLO dataset for training readiness"""

    def __init__(self, data_yaml_path: str):
        self.data_yaml_path = Path(data_yaml_path).resolve()
        self.config = self._load_config()
        self.errors = []
        self.warnings = []
        self.stats = {}

    def _load_config(self) -> dict:
        """Load dataset configuration"""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_yaml_path}")

        with open(self.data_yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def validate(self) -> bool:
        """Run all validation checks"""
        print("=" * 60)
        print("Dataset Validation")
        print("=" * 60)
        print(f"Config: {self.data_yaml_path}")
        print()

        self._validate_config()
        self._validate_paths()
        self._validate_images()
        self._validate_labels()
        self._compute_statistics()
        self._print_report()

        return len(self.errors) == 0

    def _validate_config(self):
        """Validate configuration file structure"""
        print("Checking configuration...")

        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in self.config:
                self.errors.append(f"Missing required field: {field}")

        if 'names' in self.config and 'nc' in self.config:
            names = self.config['names']
            nc = self.config['nc']

            if isinstance(names, dict):
                if len(names) != nc:
                    self.errors.append(f"nc ({nc}) doesn't match names count ({len(names)})")
            elif isinstance(names, list):
                if len(names) != nc:
                    self.errors.append(f"nc ({nc}) doesn't match names count ({len(names)})")

        print(f"  Classes: {self.config.get('nc', 'N/A')}")
        print(f"  Names: {self.config.get('names', 'N/A')}")

    def _validate_paths(self):
        """Validate dataset paths exist"""
        print("\nChecking paths...")

        base_path = Path(self.config.get('path', ''))
        if not base_path.is_absolute():
            base_path = self.data_yaml_path.parent / base_path

        if not base_path.exists():
            self.errors.append(f"Dataset path not found: {base_path}")
            return

        self.base_path = base_path
        print(f"  Base path: {base_path}")

        for split in ['train', 'val']:
            if split in self.config:
                split_path = base_path / self.config[split]
                if split_path.exists():
                    print(f"  {split.capitalize()}: {split_path} (OK)")
                else:
                    self.errors.append(f"{split.capitalize()} path not found: {split_path}")

    def _validate_images(self):
        """Validate images in dataset"""
        print("\nValidating images...")

        for split in ['train', 'val']:
            if split not in self.config:
                continue

            images_path = self.base_path / self.config[split]
            if not images_path.exists():
                continue

            image_files = list(images_path.glob('*.png')) + \
                         list(images_path.glob('*.jpg')) + \
                         list(images_path.glob('*.jpeg'))

            self.stats[f'{split}_images'] = len(image_files)
            print(f"  {split.capitalize()}: {len(image_files)} images")

            if len(image_files) == 0:
                self.errors.append(f"No images found in {split} set")
                continue

            # Sample check
            sample_size = min(10, len(image_files))
            samples = random.sample(image_files, sample_size)

            sizes = []
            modes = []
            for img_path in samples:
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                        modes.append(img.mode)
                except Exception as e:
                    self.warnings.append(f"Cannot read image: {img_path.name}")

            if sizes:
                unique_sizes = set(sizes)
                unique_modes = set(modes)
                print(f"    Sample sizes: {unique_sizes}")
                print(f"    Sample modes: {unique_modes}")

                if len(unique_sizes) > 1:
                    self.warnings.append(f"Mixed image sizes in {split} set")

    def _validate_labels(self):
        """Validate label files"""
        print("\nValidating labels...")

        nc = self.config.get('nc', 0)

        for split in ['train', 'val']:
            if split not in self.config:
                continue

            images_path = self.base_path / self.config[split]
            labels_path = images_path.parent / 'labels'

            if not labels_path.exists():
                labels_path = self.base_path / self.config[split].replace('images', 'labels')

            if not labels_path.exists():
                self.errors.append(f"Labels directory not found for {split}")
                continue

            label_files = list(labels_path.glob('*.txt'))
            self.stats[f'{split}_labels'] = len(label_files)
            print(f"  {split.capitalize()}: {len(label_files)} label files")

            class_counts = Counter()
            bbox_counts = []
            invalid_count = 0

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        bbox_counts.append(len(lines))

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1

                                if class_id >= nc:
                                    invalid_count += 1

                                x, y, w, h = map(float, parts[1:5])
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    invalid_count += 1
                except Exception:
                    pass

            self.stats[f'{split}_class_distribution'] = dict(class_counts)

            if class_counts:
                print(f"    Class distribution: {dict(class_counts)}")
                if bbox_counts:
                    print(f"    Avg boxes/image: {sum(bbox_counts)/len(bbox_counts):.1f}")

            if invalid_count > 0:
                self.warnings.append(f"{invalid_count} invalid annotations in {split}")

            # Check missing labels
            image_files = list(images_path.glob('*.png')) + \
                         list(images_path.glob('*.jpg')) + \
                         list(images_path.glob('*.jpeg'))

            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            missing = len(image_stems - label_stems)

            if missing > 0:
                self.warnings.append(f"{missing} images missing labels in {split}")

    def _compute_statistics(self):
        """Compute dataset statistics"""
        print("\nStatistics:")

        total_images = self.stats.get('train_images', 0) + self.stats.get('val_images', 0)
        total_labels = self.stats.get('train_labels', 0) + self.stats.get('val_labels', 0)

        print(f"  Total images: {total_images}")
        print(f"  Total labels: {total_labels}")

        if total_images > 0:
            train_pct = self.stats.get('train_images', 0) / total_images * 100
            val_pct = self.stats.get('val_images', 0) / total_images * 100
            print(f"  Split: {train_pct:.0f}% train / {val_pct:.0f}% val")

    def _print_report(self):
        """Print validation report"""
        print("\n" + "=" * 60)
        print("Validation Report")
        print("=" * 60)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\nAll checks passed!")

        print("\n" + "=" * 60)
        if self.errors:
            print("RESULT: FAILED - Fix errors before training")
        elif self.warnings:
            print("RESULT: PASSED with warnings")
        else:
            print("RESULT: PASSED - Dataset ready for training")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data YAML file')

    args = parser.parse_args()

    validator = DatasetValidator(args.data)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
