"""Task 2.6: grid of misclassified TEST images for a finished, already-evaluated run.

Takes the 5 most confident false positives (real photo called AI) and the 5 most confident false negatives
(AI image passed as real) at the validation-chosen threshold stored by evaluate.py.
  python scripts/error_analysis.py --run R8_mnv2_tuned
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-type", type=int, default=5)
    args = ap.parse_args()
    out = config.RUNS / args.run
    thr = json.loads((out / "test_metrics.json").read_text())["test_at_threshold"]["threshold"]
    df = pd.read_csv(out / "test_preds.csv")
    df["pred"] = (df.prob_ai >= thr).astype(int)
    wrong = df[df.pred != df.label].copy()
    fp = wrong[wrong.label == 0].nlargest(args.per_type, "prob_ai")       # real, scored most AI-like
    fn = wrong[wrong.label == 1].nsmallest(args.per_type, "prob_ai")      # AI, scored most real-like
    sel = pd.concat([fp.assign(kind="real called AI"), fn.assign(kind="AI passed as real")]).reset_index(drop=True)
    sel["id"] = range(1, len(sel) + 1)
    sel.to_csv(config.TABLES / "2.6_misclassified.csv", index=False)

    cols = args.per_type
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.2, 7.4))
    for ax, r in zip(axes.ravel(), sel.itertuples()):
        ax.imshow(Image.open(config.ROOT / r.path)); ax.axis("off")
        ax.set_title(f"#{r.id} {r.kind}\nP(AI)={r.prob_ai:.2f}  orig {r.orig_width}x{r.orig_height} {r.orig_format}",
                     fontsize=8, color="#b30000" if r.label == 0 else "#7a4b00")
    fig.suptitle(f"{args.run}: most confident test errors at threshold {thr:.2f} (top: real photos called AI, bottom: AI images passed as real)",
                 fontsize=10)
    fig.tight_layout()
    out_png = config.FIGURES / "2.6_misclassified.png"
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(sel[["id", "kind", "prob_ai", "orig_path", "orig_width", "orig_height", "orig_format"]].to_string(index=False))
    print("saved", out_png)


if __name__ == "__main__":
    main()
