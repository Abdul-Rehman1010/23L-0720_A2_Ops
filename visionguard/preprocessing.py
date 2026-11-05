"""Single source of truth for turning an image into model input.

Used by the dataset cleaner, the training pipeline and the serving code, so an uploaded image
is treated exactly like a training image:
  1. clean_pil: EXIF rotation, transparency flattened on white, RGB, longest side capped at MAX_SIDE
  2. resize:    squash to (size, size) with antialiased bilinear interpolation (aspect ratio is NOT kept)
Pixel scaling (0..255 -> what the backbone expects) lives inside the models, not here.
"""
import io

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MAX_SIDE = 512
IMG_SIZE = 160


def clean_pil(im: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im


def resize(img, size: int = IMG_SIZE):
    """uint8 HxWx3 tensor/array -> float32 (size, size, 3) in 0..255."""
    return tf.image.resize(img, (size, size), method="bilinear", antialias=True)


def load_file(path, size: int = IMG_SIZE):
    """Training-time loader for a cleaned PNG path (works inside tf.data)."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    return resize(img, size)


def preprocess_bytes(data: bytes, size: int = IMG_SIZE) -> np.ndarray:
    """Serving-time loader: raw upload bytes -> float32 array of shape (1, size, size, 3)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        arr = np.asarray(clean_pil(im), dtype=np.uint8)
    return resize(arr, size).numpy()[None]
