"""Prediction, metrics and plots shared by training (validation) and evaluation (test).

Positive class = AI-generated (label 1). A false positive is therefore a REAL photo called fake;
a false negative is a FAKE image passed as real.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm

from visionguard.data import make_dataset, read_split


def predict_split(model, split: str, img_size: int, batch_size: int = 32):
    """Un-shuffled predictions for a split, in the same order as the split CSV."""
    df, _, _ = read_split(split)
    probs = model.predict(make_dataset(split, img_size, batch_size), verbose=0).ravel()
    return df, probs


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": skm.accuracy_score(y, pred),
            "precision": skm.precision_score(y, pred, zero_division=0),
            "recall": skm.recall_score(y, pred, zero_division=0),
            "f1": skm.f1_score(y, pred, zero_division=0), "roc_auc": skm.roc_auc_score(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_confusion(y, p, threshold, title, out):
    cm = skm.confusion_matrix(np.asarray(y).astype(int), (np.asarray(p) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred real", "pred AI"])
    ax.set_yticks([0, 1], ["true real", "true AI"])
    ax.set_title(f"{title}\nthreshold {threshold:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_roc(y, p, title, out, threshold=None):
    fpr, tpr, thr = skm.roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {skm.roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    if threshold is not None:
        i = int(np.argmin(np.abs(thr - threshold)))
        ax.scatter(fpr[i], tpr[i], c="red", zorder=3, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("false positive rate (real called AI)")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=10); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_curves(history, best_epoch: int, title, out):
    """Loss and accuracy curves with the best epoch (lowest validation loss) marked."""
    h = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in ((axes[0], "loss"), (axes[1], "accuracy")):
        ax.plot(epochs, h[key], label=f"train {key}")
        ax.plot(epochs, h[f"val_{key}"], label=f"val {key}")
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        ax.scatter([best_epoch], [h[f"val_{key}"][best_epoch - 1]], c="red", zorder=3)
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
