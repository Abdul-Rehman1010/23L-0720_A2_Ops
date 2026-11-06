"""Task 1.3: sample grids and per-class histograms of width, height and file size."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from PIL import Image

COLORS = {"ai": "#d95f02", "real": "#1b9e77"}
SEED = 42


def sample_grid(manifest, out, n=12, cols=6):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2 * (n // cols), cols, figsize=(cols * 2.4, 2 * (n // cols) * 2.4))
    for block, cls in enumerate(("real", "ai")):
        paths = manifest[manifest["class"] == cls].path.to_numpy()
        for i, p in enumerate(rng.choice(paths, n, replace=False)):
            ax = axes[block * (n // cols) + i // cols, i % cols]
            ax.imshow(Image.open(p)); ax.axis("off")
            if i % cols == 0:
                ax.set_title(f"{cls.upper()}", loc="left", fontsize=11, color=COLORS[cls], fontweight="bold")
    fig.suptitle("Random samples per class (cleaned images)")
    fig.tight_layout(); fig.savefig(out / "1.3_samples_grid.png", dpi=130); plt.close(fig)


def hist(audit, col, label, out, log_x=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = audit[audit.readable]
    lo, hi = data[col].min(), data[col].max()
    bins = np.geomspace(lo, hi, 40) if log_x else np.linspace(lo, hi, 40)
    for cls, g in data.groupby("class"):
        ax.hist(g[col], bins=bins, alpha=0.6, label=f"{cls} (median {g[col].median():.0f})", color=COLORS[cls])
    if log_x:
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(label); ax.set_ylabel("images"); ax.legend()
    ax.set_title(f"Distribution of {label} per class (original files)")
    fig.tight_layout(); fig.savefig(out / f"1.3_hist_{col}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/tables/audit.csv")
    ap.add_argument("--manifest", default="data/splits/manifest.csv")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    audit, manifest = pd.read_csv(args.audit), pd.read_csv(args.manifest)
    sample_grid(manifest, out)
    hist(audit, "width", "width (px)", out, log_x=True)
    hist(audit, "height", "height (px)", out, log_x=True)
    hist(audit, "file_size", "file size (bytes)", out, log_x=True)
    print("figures written to", out)


if __name__ == "__main__":
    main()
