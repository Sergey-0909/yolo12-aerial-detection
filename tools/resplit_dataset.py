# =============================================================================
# Dataset Resplit Tool (Scene-Based Split)
# =============================================================================
# Version: 1.1.0
# =============================================================================
#
# Resplits dataset by frame ranges to prevent data leakage.
# Ensures validation contains completely different scenes than training.
# Handles duplicate filenames safely by renaming.
#
# Usage:
#   python resplit_dataset.py --source <dataset_path> [--val-start 300] [--val-end 480]
#   python resplit_dataset.py --source datasets/my_dataset --val-start 300 --val-end 480
#
# =============================================================================

import os
import sys
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_frame_number(filename: str) -> int:
    """Extract frame number from filename patterns:
    - object_0123.png -> 123
    - frame_0456.png -> 456
    - scene_road_0974.png -> 974
    - field_0000.png -> 0
    """
    # Try pattern with underscore first: _0123.png
    match = re.search(r'_(\d+)\.(?:png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try pattern without underscore: umv0974.png
    match = re.search(r'(\d+)\.(?:png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return -1


def get_all_files(dataset_path: Path) -> list:
    """Get all image files from both training and validation folders."""
    files = []

    for split in ['training', 'validation']:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'

        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                lbl_file = lbl_dir / (img_file.stem + '.txt')
                files.append({
                    'image': img_file,
                    'label': lbl_file if lbl_file.exists() else None,
                    'frame': parse_frame_number(img_file.name),
                    'name': img_file.name,
                    'source': split
                })

    return files


def check_duplicates(files: list) -> dict:
    """Check for duplicate filenames and return groups."""
    name_to_files = defaultdict(list)
    for f in files:
        name_to_files[f['name']].append(f)

    duplicates = {k: v for k, v in name_to_files.items() if len(v) > 1}
    return duplicates


def create_backup(dataset_path: Path) -> Path:
    """Create backup of current dataset structure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = dataset_path.parent / f"{dataset_path.name}_backup_{timestamp}"

    print(f"Creating backup at: {backup_path}")
    shutil.copytree(dataset_path, backup_path)

    return backup_path


def get_unique_name(base_name: str, existing_names: set, suffix_start: int = 1) -> str:
    """Generate unique filename by adding suffix."""
    stem = Path(base_name).stem
    ext = Path(base_name).suffix

    new_name = base_name
    suffix = suffix_start

    while new_name in existing_names:
        new_name = f"{stem}_dup{suffix}{ext}"
        suffix += 1

    return new_name


def resplit_dataset(
    dataset_path: Path,
    val_start: int,
    val_end: int,
    backup: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Resplit dataset by frame ranges.

    Args:
        dataset_path: Path to dataset root
        val_start: Start frame number for validation (inclusive)
        val_end: End frame number for validation (inclusive)
        backup: Whether to create backup before resplit
        dry_run: If True, only show what would be done

    Returns:
        Statistics dictionary
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Get all files
    print("Scanning dataset...")
    all_files = get_all_files(dataset_path)

    if not all_files:
        raise ValueError("No image files found in dataset")

    # Check for duplicates
    duplicates = check_duplicates(all_files)
    num_duplicates = sum(len(v) - 1 for v in duplicates.values())

    # Classify files
    train_files = []
    val_files = []
    skipped = []

    for f in all_files:
        frame = f['frame']
        if frame < 0:
            skipped.append(f)
        elif val_start <= frame <= val_end:
            val_files.append(f)
        else:
            train_files.append(f)

    # Statistics
    stats = {
        'total': len(all_files),
        'train': len(train_files),
        'val': len(val_files),
        'skipped': len(skipped),
        'duplicates': num_duplicates,
        'train_pct': len(train_files) / len(all_files) * 100 if all_files else 0,
        'val_pct': len(val_files) / len(all_files) * 100 if all_files else 0,
        'val_range': f"{val_start}-{val_end}"
    }

    print(f"\n{'='*60}")
    print("RESPLIT PLAN")
    print(f"{'='*60}")
    print(f"Total files:      {stats['total']}")
    print(f"Training:         {stats['train']} ({stats['train_pct']:.1f}%)")
    print(f"Validation:       {stats['val']} ({stats['val_pct']:.1f}%)")
    print(f"Validation range: frames {val_start} to {val_end}")
    if skipped:
        print(f"Skipped:          {stats['skipped']} (could not parse frame number)")
    if num_duplicates:
        print(f"Duplicates:       {num_duplicates} (will be renamed)")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN - No files will be moved")
        return stats

    # Create backup
    if backup:
        backup_path = create_backup(dataset_path)
        print(f"Backup created: {backup_path}\n")

    # Prepare directories
    train_img_dir = dataset_path / 'training' / 'images'
    train_lbl_dir = dataset_path / 'training' / 'labels'
    val_img_dir = dataset_path / 'validation' / 'images'
    val_lbl_dir = dataset_path / 'validation' / 'labels'

    # Create temp directories for reorganization
    temp_dir = dataset_path / '_temp_resplit'
    temp_train_img = temp_dir / 'training' / 'images'
    temp_train_lbl = temp_dir / 'training' / 'labels'
    temp_val_img = temp_dir / 'validation' / 'images'
    temp_val_lbl = temp_dir / 'validation' / 'labels'

    for d in [temp_train_img, temp_train_lbl, temp_val_img, temp_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Track used names to handle duplicates
    train_names = set()
    val_names = set()

    # Move files to temp structure
    print("Moving files to new structure...")

    moved_train = 0
    moved_val = 0
    missing_labels = 0
    renamed_files = 0

    # Process training files
    for f in train_files:
        # Get unique name if duplicate
        final_name = get_unique_name(f['name'], train_names)
        if final_name != f['name']:
            renamed_files += 1

        train_names.add(final_name)
        final_stem = Path(final_name).stem

        # Copy (not move) to preserve originals until verified
        shutil.copy2(str(f['image']), str(temp_train_img / final_name))
        if f['label'] and f['label'].exists():
            shutil.copy2(str(f['label']), str(temp_train_lbl / (final_stem + '.txt')))
        else:
            missing_labels += 1
        moved_train += 1

    # Process validation files
    for f in val_files:
        # Get unique name if duplicate
        final_name = get_unique_name(f['name'], val_names)
        if final_name != f['name']:
            renamed_files += 1

        val_names.add(final_name)
        final_stem = Path(final_name).stem

        # Copy (not move) to preserve originals until verified
        shutil.copy2(str(f['image']), str(temp_val_img / final_name))
        if f['label'] and f['label'].exists():
            shutil.copy2(str(f['label']), str(temp_val_lbl / (final_stem + '.txt')))
        else:
            missing_labels += 1
        moved_val += 1

    # Verify counts before finalizing
    temp_train_count = len(list(temp_train_img.iterdir()))
    temp_val_count = len(list(temp_val_img.iterdir()))
    expected_total = len(train_files) + len(val_files)
    actual_total = temp_train_count + temp_val_count

    print(f"\nVerifying: expected {expected_total}, got {actual_total}")

    if actual_total != expected_total:
        print(f"ERROR: File count mismatch! Keeping backup, aborting.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return stats

    # Finalize: remove old and rename temp
    print("Finalizing new structure...")

    shutil.rmtree(train_img_dir, ignore_errors=True)
    shutil.rmtree(train_lbl_dir, ignore_errors=True)
    shutil.rmtree(val_img_dir, ignore_errors=True)
    shutil.rmtree(val_lbl_dir, ignore_errors=True)

    shutil.move(str(temp_train_img), str(train_img_dir))
    shutil.move(str(temp_train_lbl), str(train_lbl_dir))
    shutil.move(str(temp_val_img), str(val_img_dir))
    shutil.move(str(temp_val_lbl), str(val_lbl_dir))

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nResplit complete!")
    print(f"  Training:   {moved_train} files")
    print(f"  Validation: {moved_val} files")
    if renamed_files:
        print(f"  Renamed:    {renamed_files} duplicates")
    if missing_labels:
        print(f"  Warning:    {missing_labels} images missing labels")

    stats['moved_train'] = moved_train
    stats['moved_val'] = moved_val
    stats['missing_labels'] = missing_labels
    stats['renamed_files'] = renamed_files

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Resplit dataset by frame ranges to prevent data leakage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would happen
  python resplit_dataset.py --source datasets/my_dataset --dry-run

  # Resplit with frames 300-480 as validation
  python resplit_dataset.py --source datasets/my_dataset --val-start 300 --val-end 480

  # Resplit without backup (dangerous!)
  python resplit_dataset.py --source datasets/my_dataset --no-backup
        """
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--val-start',
        type=int,
        default=300,
        help='Start frame number for validation (inclusive, default: 300)'
    )
    parser.add_argument(
        '--val-end',
        type=int,
        default=480,
        help='End frame number for validation (inclusive, default: 480)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup (use with caution!)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    try:
        stats = resplit_dataset(
            dataset_path=args.source,
            val_start=args.val_start,
            val_end=args.val_end,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )

        if not args.dry_run:
            print(f"\nDataset resplit successfully!")
            print(f"Train: {stats['train']} | Val: {stats['val']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
