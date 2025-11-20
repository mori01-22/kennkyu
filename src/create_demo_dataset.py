from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import random
import numpy as np


def make_tansu(img_size):
    w, h = img_size
    img = Image.new('RGB', (w, h), color=(150 + random.randint(-10,10), 120 + random.randint(-10,10), 90 + random.randint(-10,10)))
    d = ImageDraw.Draw(img)
    margin = int(w * 0.12)
    left = margin
    top = int(h * 0.15)
    right = w - margin
    bottom = int(h * 0.85)
    # dresser body
    d.rectangle([left, top, right, bottom], fill=(80 + random.randint(-10,10), 60 + random.randint(-10,10), 40 + random.randint(-10,10)))
    # drawers
    n_drawers = random.choice([2,3,4])
    for i in range(1, n_drawers):
        y = top + i * ((bottom - top) / n_drawers)
        d.line([(left + 4, y), (right - 4, y)], fill=(60,50,30), width=max(1, w//100))
    # knobs
    for i in range(n_drawers):
        yc = int(top + (i + 0.5) * ((bottom - top) / n_drawers))
        xc = left + int((right - left) * 0.85)
        r = max(1, w//80)
        d.ellipse([xc - r, yc - r, xc + r, yc + r], fill=(30,30,30))
    return img


def make_not_tansu(img_size):
    w, h = img_size
    img = Image.new('RGB', (w, h), color=(200 + random.randint(-10,10), 220 + random.randint(-10,10), 240 + random.randint(-10,10)))
    d = ImageDraw.Draw(img)
    # random shapes
    for _ in range(random.randint(1,3)):
        cx = random.randint(int(w*0.2), int(w*0.8))
        cy = random.randint(int(h*0.2), int(h*0.8))
        r = random.randint(int(w*0.08), int(w*0.3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120 + random.randint(-20,20),140 + random.randint(-20,20),160 + random.randint(-20,20)))
    return img


def augment_image(img, max_rotate=15, brightness_jitter=0.2, contrast_jitter=0.15, noise_std=6):
    """Apply simple augmentations: flip, rotate, brightness/contrast, add Gaussian noise."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.uniform(-max_rotate, max_rotate)
    img = img.rotate(angle, resample=Image.BILINEAR)

    # Brightness and contrast
    from PIL import ImageEnhance
    if brightness_jitter > 0:
        factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast_jitter > 0:
        factor = 1.0 + random.uniform(-contrast_jitter, contrast_jitter)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Add slight Gaussian noise
    if noise_std and noise_std > 0:
        arr = np.array(img).astype('float32')
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype('uint8')
        img = Image.fromarray(arr)

    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', '-o', default='data/demo_train', help='Output directory to create demo dataset')
    p.add_argument('--n-per-class', '-n', type=int, default=100, help='Number of images per class')
    p.add_argument('--size', type=int, default=224, help='Image size (square)')
    p.add_argument('--seed', type=int, default=123, help='Random seed')
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    tansu_dir = out / 'tansu'
    not_dir = out / 'not_tansu'
    tansu_dir.mkdir(parents=True, exist_ok=True)
    not_dir.mkdir(parents=True, exist_ok=True)

    w = h = args.size
    for i in range(args.n_per_class):
        img = make_tansu((w,h))
        img = augment_image(img)
        img.save(tansu_dir / f'tansu_{i:04d}.jpg', 'JPEG', quality=90)
    for i in range(args.n_per_class):
        img = make_not_tansu((w,h))
        img = augment_image(img)
        img.save(not_dir / f'not_{i:04d}.jpg', 'JPEG', quality=90)

    print(f'Wrote {args.n_per_class} images per class to {out.resolve()}')


if __name__ == '__main__':
    main()
