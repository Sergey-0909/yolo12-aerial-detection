#!/usr/bin/env python3
"""
=============================================================================
Dataset Splitter for YOLO Format
=============================================================================
Version: 1.0.0
=============================================================================

Splits dataset into train/val with proper YOLO folder structure:
  dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── data.yaml
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse


def find_image_label_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all image-label pairs, ensuring they match."""
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Find all images
    for file in source_dir.rglob('*'):
        if file.suffix.lower() in image_extensions:
            # Find corresponding label
            label_path = file.with_suffix('.txt')
            if label_path.exists():
                pairs.append((file, label_path))
            else:
                print(f"WARNING: No label for {file.name}")

    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List, List]:
    """Split pairs into train and val sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs = shuffled[split_idx:]

    return train_pairs, val_pairs


def create_yolo_structure(output_dir: Path, train_pairs: List, val_pairs: List, class_names: List[str]):
    """Create YOLO 1.1 folder structure and copy files."""

    # Create directories
    dirs = [
        output_dir / 'train' / 'images',
        output_dir / 'train' / 'labels',
        output_dir / 'val' / 'images',
        output_dir / 'val' / 'labels',
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Copy train files
    print(f"\nCopying {len(train_pairs)} training pairs...")
    for img_path, lbl_path in train_pairs:
        shutil.copy2(img_path, output_dir / 'train' / 'images' / img_path.name)
        shutil.copy2(lbl_path, output_dir / 'train' / 'labels' / lbl_path.name)

    # Copy val files
    print(f"Copying {len(val_pairs)} validation pairs...")
    for img_path, lbl_path in val_pairs:
        shutil.copy2(img_path, output_dir / 'val' / 'images' / img_path.name)
        shutil.copy2(lbl_path, output_dir / 'val' / 'labels' / lbl_path.name)

    # Create data.yaml
    yaml_content = f"""# Dataset Configuration

path: {output_dir.resolve()}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_path}")

    return yaml_path


def verify_split(output_dir: Path):
    """Verify the split was successful."""
    train_images = list((output_dir / 'train' / 'images').glob('*'))
    train_labels = list((output_dir / 'train' / 'labels').glob('*.txt'))
    val_images = list((output_dir / 'val' / 'images').glob('*'))
    val_labels = list((output_dir / 'val' / 'labels').glob('*.txt'))

    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Val images:   {len(val_images)}")
    print(f"Val labels:   {len(val_labels)}")

    # Check matching
    train_img_names = {p.stem for p in train_images}
    train_lbl_names = {p.stem for p in train_labels}
    val_img_names = {p.stem for p in val_images}
    val_lbl_names = {p.stem for p in val_labels}

    train_match = train_img_names == train_lbl_names
    val_match = val_img_names == val_lbl_names

    print(f"\nTrain img-label match: {'OK' if train_match else 'MISMATCH'}")
    print(f"Val img-label match:   {'OK' if val_match else 'MISMATCH'}")

    if not train_match:
        missing = train_img_names - train_lbl_names
        print(f"  Missing labels: {list(missing)[:5]}...")

    if not val_match:
        missing = val_img_names - val_lbl_names
        print(f"  Missing labels: {list(missing)[:5]}...")

    return train_match and val_match


def main():
    parser = argparse.ArgumentParser(description='Split dataset into YOLO 1.1 format')
    parser.add_argument('--source', type=str, required=True, help='Source directory with images and labels')
    parser.add_argument('--output', type=str, required=True, help='Output directory for split dataset')
    parser.add_argument('--ratio', type=float, default=0.8, help='Train ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--classes', type=str, default='object', help='Comma-separated class names')

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    class_names = [c.strip() for c in args.classes.split(',')]

    print("=" * 50)
    print("YOLO Dataset Splitter")
    print("=" * 50)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Train ratio: {args.ratio}")
    print(f"Classes: {class_names}")

    # Find pairs
    print("\nFinding image-label pairs...")
    pairs = find_image_label_pairs(source_dir)
    print(f"Found {len(pairs)} pairs")

    if len(pairs) == 0:
        print("ERROR: No image-label pairs found!")
        return 1

    # Split
    train_pairs, val_pairs = split_pairs(pairs, args.ratio, args.seed)
    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create structure
    create_yolo_structure(output_dir, train_pairs, val_pairs, class_names)

    # Verify
    success = verify_split(output_dir)

    if success:
        print("\nDataset split completed successfully!")
        print(f"\nData config: {output_dir / 'data.yaml'}")
    else:
        print("\nVerification failed - check the output!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
