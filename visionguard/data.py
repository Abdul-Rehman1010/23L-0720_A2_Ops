"""Task 2.1: tf.data input pipeline built from the split CSVs.

read paths/labels -> decode PNG -> squash-resize -> cache -> (train only: shuffle) -> batch
-> (train only: random augmentation) -> prefetch
Run `python -m visionguard.data --figure` to save a figure of one augmented training batch.
"""
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

from visionguard import config
from visionguard.preprocessing import IMG_SIZE, load_file

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name: str):
    df = pd.read_csv(config.SPLITS / f"{name}.csv")
    return df, [str(config.ROOT / p) for p in df.path], df.label.to_numpy(dtype="float32")


def augmenter(seed: int = config.SEED):
    """Random, label-preserving changes; only ever applied to training batches."""
    from keras import Sequential
    return Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, seed=seed),
        layers.RandomZoom(0.1, seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def make_dataset(split: str, img_size: int = IMG_SIZE, batch_size: int = 32,
                 training: bool = False, augment: bool = False, seed: int = config.SEED):
    _, paths, labels = read_split(split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_file(p, img_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                                   # decoded + resized once, then reused every epoch
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def save_batch_figure(img_size: int, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tf.keras.utils.set_random_seed(config.SEED)
    plain = next(iter(make_dataset("train", img_size, 8, training=True, augment=False)))
    tf.keras.utils.set_random_seed(config.SEED)       # same shuffle -> same 8 images, now augmented
    aug = next(iter(make_dataset("train", img_size, 8, training=True, augment=True)))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for row, (title, (x, y)) in enumerate((("resized only", plain), ("augmented", aug))):
        for i in range(8):
            axes[row, i].imshow(np.clip(x[i].numpy() / 255.0, 0, 1)); axes[row, i].axis("off")
            axes[row, i].set_title(config.LABELS[int(y[i])], fontsize=9)
        axes[row, 0].text(-0.05, 0.5, title, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=10)
    fig.suptitle(f"One training batch at {img_size}x{img_size}: resized only (top) vs augmented (bottom)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = ap.parse_args()
    for split in ("train", "val", "test"):
        ds = make_dataset(split, args.img_size)
        x, y = next(iter(ds))
        print(f"{split:5s} batch x={x.shape} {x.dtype} range=({float(x.numpy().min()):.0f}, "
              f"{float(x.numpy().max()):.0f})  y={y.shape} positives={int(y.numpy().sum())}/{len(y)}")
    if args.figure:
        out = config.FIGURES / "2.1_augmented_batch.png"
        save_batch_figure(args.img_size, out)
        print("saved", out)
