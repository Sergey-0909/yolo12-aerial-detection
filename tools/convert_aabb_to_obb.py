#!/usr/bin/env python3
# =============================================================================
# AABB to OBB Dataset Converter
# =============================================================================
# Version: 1.0.0
# =============================================================================
#
# Converts a YOLO AABB dataset to YOLO-OBB format by detecting the real
# object orientation in each image using computer vision.
#
# Strategy per bounding box:
#   1. Crop the image region (with padding)
#   2. Grayscale -> GaussianBlur -> Canny edges -> findContours
#   3. cv2.minAreaRect() on the largest contour -> real rotation angle
#   4. cv2.boxPoints() -> 4 corner points -> transform to full image coords
#   5. Normalize to [0, 1] and clamp
#   Fallback (axis-aligned corners) when:
#     - Object is too small (<min_size_px)
#     - No contours found
#     - Contour fills less than --confidence of the crop area
#
# Input label format:   class_id cx cy w h          (YOLO AABB)
# Output label format:  class_id x1 y1 x2 y2 x3 y3 x4 y4  (YOLO OBB)
#
# Usage:
#   python convert_aabb_to_obb.py /path/to/dataset --output /path/to/out
#   python convert_aabb_to_obb.py /path/to/dataset --visualize
#   python convert_aabb_to_obb.py /path/to/dataset --inplace
#
# Dependencies: opencv-python, numpy, pyyaml, tqdm
# =============================================================================

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Label parsing / formatting
# ---------------------------------------------------------------------------

def parse_aabb_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Parse a YOLO AABB label line.  Returns (class_id, cx, cy, w, h) or None."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def format_obb_line(class_id: int, corners: np.ndarray) -> str:
    """Format OBB label: class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    coords = ' '.join(f'{v:.6f}' for v in corners.flatten())
    return f'{class_id} {coords}'


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def aabb_to_pixel_corners(cx: float, cy: float, w: float, h: float,
                           img_w: int, img_h: int) -> np.ndarray:
    """Axis-aligned AABB -> 4 pixel corners, clockwise from top-left. Shape (4, 2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def normalize_and_clamp(corners_px: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Normalize pixel corners to [0, 1] and clamp. Shape (4, 2)."""
    normalized = corners_px / np.array([img_w, img_h], dtype=np.float32)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core OBB detection from crop
# ---------------------------------------------------------------------------

def detect_obb_from_crop(
    crop: np.ndarray,
    x1_offset: int,
    y1_offset: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    confidence_threshold: float,
    min_size_px: int,
) -> Tuple[np.ndarray, bool]:
    """
    Attempt to extract real OBB from the cropped image region.

    Returns:
        corners_norm: (4, 2) normalized corners
        used_real_obb: True when real rotation was detected, False on fallback
    """
    fallback = normalize_and_clamp(
        aabb_to_pixel_corners(cx, cy, w, h, img_w, img_h), img_w, img_h
    )

    # --- Size gate ---
    if crop.shape[0] < min_size_px or crop.shape[1] < min_size_px:
        return fallback, False

    # --- AABB reference values in pixel space ---
    aabb_area_px = (w * img_w) * (h * img_h)
    aabb_x1_px   = (cx - w / 2) * img_w
    aabb_y1_px   = (cy - h / 2) * img_h
    aabb_x2_px   = (cx + w / 2) * img_w
    aabb_y2_px   = (cy + h / 2) * img_h

    # Center of AABB in crop coordinate space
    center_crop_x = cx * img_w - x1_offset
    center_crop_y = cy * img_h - y1_offset

    # --- Grayscale ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # --- Denoise + edges ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect broken edge fragments
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # --- Contours ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback, False

    # --- FIX 1: Filter by centroid inside AABB region, then take largest ---
    # "Largest overall" grabs background. "Closest to center" grabs tiny fragments.
    # Correct: centroid must be inside AABB, then largest among those wins.
    aabb_in_crop_x1 = aabb_x1_px - x1_offset
    aabb_in_crop_y1 = aabb_y1_px - y1_offset
    aabb_in_crop_x2 = aabb_x2_px - x1_offset
    aabb_in_crop_y2 = aabb_y2_px - y1_offset

    valid_contours = []
    for c in contours:
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        cx_ = m['m10'] / m['m00']
        cy_ = m['m01'] / m['m00']
        if (aabb_in_crop_x1 <= cx_ <= aabb_in_crop_x2 and
                aabb_in_crop_y1 <= cy_ <= aabb_in_crop_y2):
            valid_contours.append(c)

    if not valid_contours:
        return fallback, False

    best = max(valid_contours, key=cv2.contourArea)

    # --- FIX 2: Confidence check against AABB area, not crop area ---
    # Crop includes padding so old denominator was inflated.
    contour_area = cv2.contourArea(best)
    if aabb_area_px == 0 or (contour_area / aabb_area_px) < confidence_threshold:
        return fallback, False

    # --- Minimum area rectangle ---
    rect = cv2.minAreaRect(best)          # (center, size, angle)

    # --- FIX 3: OBB area must not significantly exceed AABB area ---
    # 10% tolerance for measurement noise from minAreaRect.
    obb_area_px = rect[1][0] * rect[1][1]
    if aabb_area_px > 0 and obb_area_px > aabb_area_px * 1.10:
        return fallback, False

    box_crop = cv2.boxPoints(rect)           # (4, 2) in crop pixel coords

    # --- Transform to full image pixel coords ---
    box_img = box_crop + np.array([x1_offset, y1_offset], dtype=np.float32)

    # --- FIX 4: Reject if corners extend significantly outside AABB bounds ---
    # 20% tolerance — rotated box corners can legitimately slightly exceed AABB.
    tol_x = 0.20 * (w * img_w)
    tol_y = 0.20 * (h * img_h)
    if (np.any(box_img[:, 0] < aabb_x1_px - tol_x) or
            np.any(box_img[:, 0] > aabb_x2_px + tol_x) or
            np.any(box_img[:, 1] < aabb_y1_px - tol_y) or
            np.any(box_img[:, 1] > aabb_y2_px + tol_y)):
        return fallback, False

    # --- Normalize + clamp ---
    corners_norm = normalize_and_clamp(box_img, img_w, img_h)

    return corners_norm, True


# ---------------------------------------------------------------------------
# Per-image conversion
# ---------------------------------------------------------------------------

def convert_label_file(
    img_path: Path,
    label_path: Path,
    out_label_path: Path,
    visualize: bool,
    viz_path: Optional[Path],
    confidence_threshold: float,
    min_size_px: int,
    padding: float,
) -> Tuple[int, int, int]:
    """
    Convert one label file from AABB to OBB.

    Returns:
        (total_boxes, real_obb_count, fallback_count)
    """
    # Empty label = background image, copy as-is
    if label_path.stat().st_size == 0:
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, out_label_path)
        return 0, 0, 0

    with open(label_path, 'r') as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    if not raw_lines:
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, out_label_path)
        return 0, 0, 0

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARNING: cannot read image {img_path} — using AABB fallback for all boxes")
        img_h = img_w = 640
        img_fallback_only = True
    else:
        img_h, img_w = img.shape[:2]
        img_fallback_only = False

    viz_img = img.copy() if (visualize and img is not None) else None

    obb_lines: List[str] = []
    total = real_obb = fallback = 0

    for line in raw_lines:
        parsed = parse_aabb_line(line)
        if parsed is None:
            continue  # skip malformed lines

        class_id, cx, cy, w, h = parsed
        total += 1

        if img_fallback_only:
            corners = normalize_and_clamp(
                aabb_to_pixel_corners(cx, cy, w, h, img_w, img_h), img_w, img_h
            )
            used_real = False
        else:
            # Crop bounds in pixels (with padding)
            w_px = w * img_w
            h_px = h * img_h
            pad_px = padding * max(w_px, h_px)

            x1 = max(0, int(cx * img_w - w_px / 2 - pad_px))
            y1 = max(0, int(cy * img_h - h_px / 2 - pad_px))
            x2 = min(img_w, int(cx * img_w + w_px / 2 + pad_px))
            y2 = min(img_h, int(cy * img_h + h_px / 2 + pad_px))

            crop = img[y1:y2, x1:x2]

            corners, used_real = detect_obb_from_crop(
                crop, x1, y1, cx, cy, w, h, img_w, img_h,
                confidence_threshold, min_size_px,
            )

        obb_lines.append(format_obb_line(class_id, corners))

        if used_real:
            real_obb += 1
        else:
            fallback += 1

        # Draw visualization overlay
        if viz_img is not None:
            # Original AABB — green
            aabb_pts = normalize_and_clamp(
                aabb_to_pixel_corners(cx, cy, w, h, img_w, img_h), img_w, img_h
            )
            _draw_box(viz_img, aabb_pts, img_w, img_h, (0, 200, 0), 1)
            # OBB — red if real, yellow if fallback
            color = (0, 0, 220) if used_real else (0, 220, 220)
            _draw_box(viz_img, corners, img_w, img_h, color, 2)

    # Write output labels
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, 'w') as f:
        f.write('\n'.join(obb_lines) + ('\n' if obb_lines else ''))

    # Save visualization
    if viz_img is not None and viz_path is not None:
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_path), viz_img)

    return total, real_obb, fallback


def _draw_box(img: np.ndarray, corners_norm: np.ndarray,
              img_w: int, img_h: int, color: tuple, thickness: int):
    """Draw a 4-corner box on img (in-place)."""
    pts = (corners_norm * np.array([img_w, img_h])).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)


# ---------------------------------------------------------------------------
# Dataset-level conversion
# ---------------------------------------------------------------------------

SPLITS = ['train', 'val']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def find_label_path(img_path: Path, label_dir: Path) -> Optional[Path]:
    """Return the .txt label path corresponding to an image file."""
    label_path = label_dir / (img_path.stem + '.txt')
    return label_path if label_path.exists() else None


def convert_dataset(
    dataset_path: Path,
    output_path: Path,
    visualize: bool,
    confidence_threshold: float,
    min_size_px: int,
    padding: float,
) -> None:
    """Convert an entire YOLO AABB dataset to OBB format."""

    # ---- Stats accumulators ----
    total_images = 0
    total_boxes = 0
    total_real = 0
    total_fallback = 0
    total_empty = 0
    total_skipped = 0

    for split in SPLITS:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'

        if not img_dir.exists():
            print(f"  Split '{split}': images dir not found, skipping")
            continue

        out_lbl_dir = output_path / split / 'labels'
        out_img_dir = output_path / split / 'images'
        viz_dir = output_path / split / 'visualizations' if visualize else None

        # Copy images if writing to a new location
        if output_path != dataset_path:
            out_img_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n[{split}]  {len(image_files)} images found")

        iterator = tqdm(image_files, unit='img') if TQDM_AVAILABLE else image_files

        for img_path in iterator:
            label_path = find_label_path(img_path, lbl_dir)
            out_label_path = out_lbl_dir / (img_path.stem + '.txt')

            # Copy image if output is different from input
            if output_path != dataset_path:
                dest_img = out_img_dir / img_path.name
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

            if label_path is None:
                # No label file at all - skip (no object, no label)
                total_skipped += 1
                continue

            viz_path = (viz_dir / (img_path.stem + '_viz' + img_path.suffix)) if viz_dir else None

            n_total, n_real, n_fallback = convert_label_file(
                img_path=img_path,
                label_path=label_path,
                out_label_path=out_label_path,
                visualize=visualize,
                viz_path=viz_path,
                confidence_threshold=confidence_threshold,
                min_size_px=min_size_px,
                padding=padding,
            )

            total_images += 1
            if n_total == 0:
                total_empty += 1
            total_boxes += n_total
            total_real += n_real
            total_fallback += n_fallback

    # ---- Print statistics ----
    print('\n' + '=' * 60)
    print('CONVERSION STATISTICS')
    print('=' * 60)
    print(f'  Images processed : {total_images}')
    print(f'  Background images: {total_empty}  (empty labels, copied as-is)')
    print(f'  Skipped (no label): {total_skipped}')
    print(f'  Total boxes      : {total_boxes}')
    if total_boxes > 0:
        real_pct = 100.0 * total_real / total_boxes
        fall_pct = 100.0 * total_fallback / total_boxes
        print(f'  Real OBB         : {total_real}  ({real_pct:.1f}%)')
        print(f'  Fallback (AABB)  : {total_fallback}  ({fall_pct:.1f}%)')
    print('=' * 60)

    if total_boxes > 0 and (total_real / total_boxes) < 0.5:
        print('\nWARNING: Less than 50% of boxes got real rotation detection.')
        print('Consider lowering --confidence or checking image quality.')


# ---------------------------------------------------------------------------
# data.yaml handling
# ---------------------------------------------------------------------------

def update_data_yaml(dataset_path: Path, output_path: Path) -> None:
    """Copy data.yaml to output, adding/overwriting task: obb."""
    src_yaml = dataset_path / 'data.yaml'
    if not src_yaml.exists():
        print('WARNING: data.yaml not found in dataset root, skipping update')
        return

    with open(src_yaml, 'r') as f:
        data = yaml.safe_load(f) or {}

    # Update path to output location and set task
    data['path'] = str(output_path.resolve())
    data['task'] = 'obb'

    out_yaml = output_path / 'data.yaml'
    output_path.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f'\ndata.yaml written to: {out_yaml}')
    print('  task: obb  (added)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert YOLO AABB dataset to OBB format using CV orientation detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to new directory
  python convert_aabb_to_obb.py /data/my_dataset --output /data/my_dataset_obb

  # Visualize results (saves debug images alongside labels)
  python convert_aabb_to_obb.py /data/my_dataset --output /data/my_dataset_obb --visualize

  # Overwrite labels in place (CAUTION: destructive)
  python convert_aabb_to_obb.py /data/my_dataset --inplace

Tuning tips:
  --confidence 0.10   Lower if many objects have poor contrast (fewer fallbacks)
  --confidence 0.25   Higher if you get too many false rotations
  --min-size 20       Increase to skip very small objects
  --padding 0.2       More context around the crop (helps edge detection)
        """,
    )
    parser.add_argument('dataset', type=str,
                        help='Path to YOLO dataset root (contains data.yaml, train/, val/)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: <dataset>_obb next to input)')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite labels in the source dataset (CAUTION: irreversible)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Save debug images showing AABB (green) vs OBB (red/yellow)')
    parser.add_argument('--confidence', type=float, default=0.05,
                        help='Min fraction of AABB area that the contour must cover (default: 0.05)')
    parser.add_argument('--min-size', type=int, default=10,
                        help='Min crop side in pixels to attempt orientation detection (default: 10)')
    parser.add_argument('--padding', type=float, default=0.05,
                        help='Crop padding as fraction of max(w, h) (default: 0.05)')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f'ERROR: dataset path does not exist: {dataset_path}')
        sys.exit(1)

    # Determine output path
    if args.inplace:
        output_path = dataset_path
        print(f'INPLACE mode: labels will be overwritten in {dataset_path}')
        print('Press Ctrl+C within 3 seconds to abort...')
        import time; time.sleep(3)
    elif args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = dataset_path.parent / (dataset_path.name + '_obb')

    print('=' * 60)
    print('AABB to OBB Converter v1.0.0')
    print('=' * 60)
    print(f'Input  : {dataset_path}')
    print(f'Output : {output_path}')
    print(f'Confidence threshold : {args.confidence}')
    print(f'Min object size      : {args.min_size} px')
    print(f'Crop padding         : {args.padding}')
    print(f'Visualize            : {args.visualize}')

    output_path.mkdir(parents=True, exist_ok=True)

    convert_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        visualize=args.visualize,
        confidence_threshold=args.confidence,
        min_size_px=args.min_size,
        padding=args.padding,
    )

    update_data_yaml(dataset_path, output_path)

    print(f'\nDone. OBB dataset ready at: {output_path}')
    print('Train with:')
    print(f'  ./start_training.sh --config configs/raw_obb_s.yaml --gpu 0')
    print('  (update the data path in the config to point to your new dataset)')


if __name__ == '__main__':
    main()
