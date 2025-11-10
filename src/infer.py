import argparse
import os
from pathlib import Path
from typing import List, Tuple
import csv

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception as e:
    raise RuntimeError('TensorFlow must be installed to run this script: pip install tensorflow') from e


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_images_in_dir(d: Path) -> List[Path]:
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(sorted(d.glob(f'*{ext}')))
    return imgs


def load_image(path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def detect_input_size_from_keras(model) -> Tuple[int,int]:
    # model.input_shape often like (None, 224, 224, 3) or (None, None, None, 3)
    shape = getattr(model, 'input_shape', None)
    if not shape:
        # fallback
        return (224, 224)
    if len(shape) >= 3:
        h = shape[1] or 224
        w = shape[2] or 224
        return (int(h), int(w))
    return (224, 224)


def load_keras_model_or_saved(path: str):
    # Accepts .keras file or SavedModel dir or .h5
    return keras.models.load_model(path)


class TFLiteWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # assume single input
        shape = self.input_details[0]['shape']
        # shape can be [1,224,224,3] or [None,224,224,3]
        self.input_size = (int(shape[1]) if len(shape) > 1 and shape[1] is not None else 224,
                           int(shape[2]) if len(shape) > 2 and shape[2] is not None else 224)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x expected shape (1,H,W,3)
        self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(self.input_details[0]['dtype']))
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        return out


def predict_with_keras(model, img_arr: np.ndarray) -> float:
    # returns probability scalar
    pred = model.predict(img_arr)
    # handle (1,1) or (1,) or softmax
    pred = np.array(pred).reshape(-1)
    return float(pred[0])


def run_inference(model_path: str, images: List[Path], threshold: float = 0.5):
    model_path = str(model_path)
    is_tflite = model_path.lower().endswith('.tflite')

    results = []

    if is_tflite:
        wrapper = TFLiteWrapper(model_path)
        img_size = wrapper.input_size
        for img in images:
            arr = load_image(img, img_size)
            out = wrapper.predict(arr)
            prob = float(np.array(out).reshape(-1)[0])
            label = 'tansu' if prob >= threshold else 'not_tansu'
            results.append((str(img), prob, label))
    else:
        # Keras or SavedModel (.keras/.h5 or folder)
        model = load_keras_model_or_saved(model_path)
        img_size = detect_input_size_from_keras(model)
        for img in images:
            arr = load_image(img, img_size)
            prob = predict_with_keras(model, arr)
            label = 'tansu' if prob >= threshold else 'not_tansu'
            results.append((str(img), prob, label))

    return results


def main():
    ap = argparse.ArgumentParser(description='Run inference using a dresser detection model')
    ap.add_argument('--model', '-m', required=True, help='Path to model (.keras, .h5, SavedModel dir, or .tflite)')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', '-i', help='Path to a single image file')
    group.add_argument('--dir', '-d', help='Directory containing images')
    ap.add_argument('--out', '-o', help='CSV output path (optional)')
    ap.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for binary decision (default 0.5)')
    args = ap.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print('Model path does not exist:', model_path)
        return

    images = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print('Image not found:', p)
            return
        images = [p]
    else:
        d = Path(args.dir)
        if not d.exists() or not d.is_dir():
            print('Directory not found or not a directory:', d)
            return
        images = list_images_in_dir(d)
        if not images:
            print('No images found in directory:', d)
            return

    results = run_inference(model_path, images, threshold=args.threshold)

    # print results
    print('\nInference results:')
    for p, prob, label in results:
        print(f'{p}\t{prob:.4f}\t{label}')

    if args.out:
        out_path = Path(args.out)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image', 'prob_tansu', 'label'])
            for p, prob, label in results:
                w.writerow([p, f'{prob:.6f}', label])
        print('Saved CSV to', out_path)


if __name__ == '__main__':
    main()
