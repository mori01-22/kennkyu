#!/usr/bin/env python3
"""Inference wrapper that maps labels to Japanese: 'tansu' -> '倒れる', 'not_tansu' -> '倒れない'.

Usage examples:
  python src/infer_jp_labels.py --model best_tansu_model.h5 --image test_images/1.png
  python src/infer_jp_labels.py --model tansu_detector.keras --dir test_images --out results_jp.csv
"""
import argparse
import os
from pathlib import Path
import csv
import sys

# Ensure repository root is on sys.path so we can import src.infer
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from src.infer import run_inference
except Exception as e:
    raise RuntimeError("Could not import run_inference from src.infer. "
                       "Run this script from the repository root or ensure imports work.") from e

LABEL_MAP = {
    'tansu': '倒れる',
    'not_tansu': '倒れない'
}

def map_label(label: str) -> str:
    return LABEL_MAP.get(label, label)

def collect_images(image: str, directory: str):
    imgs = []
    if image:
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f'Image not found: {p}')
        imgs = [p]
    else:
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            raise NotADirectoryError(f'Directory not found or not a directory: {d}')
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not imgs:
            raise FileNotFoundError(f'No images found in directory: {d}')
    return imgs

def main():
    ap = argparse.ArgumentParser(description='Inference wrapper that prints Japanese labels (倒れる / 倒れない)')
    ap.add_argument('--model', '-m', required=True, help='Path to model (.keras, .h5, SavedModel dir, or .tflite)')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', '-i', help='Path to a single image file')
    group.add_argument('--dir', '-d', help='Directory containing images')
    ap.add_argument('--out', '-o', help='CSV output path (optional)')
    ap.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for binary decision (default 0.5)')
    ap.add_argument('--tta', type=int, default=0, help='Test-time augmentation count (0 or 1 = off, >1 = number of augmentations)')
    args = ap.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print('Model path does not exist:', model_path)
        return

    try:
        images = collect_images(args.image, args.dir)
    except Exception as e:
        print('Error collecting images:', e)
        return

    results = run_inference(model_path, images, threshold=args.threshold, tta=args.tta)

    # Map and print results with Japanese labels
    print('\nInference results (Japanese labels):')
    for p, prob, label in results:
        jp = map_label(label)
        print(f'{p}\t{prob:.4f}\t{jp}')

    if args.out:
        out_path = Path(args.out)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image', 'prob_tansu', 'label_jp'])
            for p, prob, label in results:
                w.writerow([p, f'{prob:.6f}', map_label(label)])
        print('Saved CSV to', out_path)

if __name__ == '__main__':
    main()
