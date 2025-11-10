"""
Create a single demo test image that looks like a dresser (rectangle) or non-dresser (circle).
Usage:
  python src\create_test_image.py --out ..\test_images\test1.jpg --type tansu
  python src\create_test_image.py --out ..\test_images\test1.jpg --type not_tansu
"""
from PIL import Image, ImageDraw
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out', '-o', default='test_images/test1.jpg', help='Output image path (relative to repo root)')
parser.add_argument('--size', type=int, default=224, help='Image size (square)')
parser.add_argument('--type', choices=['tansu','not_tansu'], default='tansu', help='Type of image to create')
args = parser.parse_args()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
size = args.size

if args.type == 'tansu':
    img = Image.new('RGB', (size, size), color=(150,120,90))
    d = ImageDraw.Draw(img)
    # draw dresser-like rectangles with drawers
    margin = int(size * 0.15)
    left = margin
    top = int(size * 0.15)
    right = size - margin
    bottom = int(size * 0.85)
    d.rectangle([left, top, right, bottom], fill=(80,60,40))
    # draw drawer lines
    n_drawers = 3
    for i in range(1, n_drawers):
        y = top + i * ((bottom - top) / n_drawers)
        d.line([(left, y), (right, y)], fill=(60,50,30), width=3)
else:
    img = Image.new('RGB', (size, size), color=(200,220,240))
    d = ImageDraw.Draw(img)
    # draw a circle to indicate not furniture
    cx = size // 2
    cy = size // 2
    r = int(size * 0.25)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120,140,160))

img.save(out_path, 'JPEG')
print('Wrote test image to', out_path.resolve())
