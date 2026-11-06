"""Task 2.5: evaluate a finished run on the TEST split, exactly once.

The decision threshold is chosen on the validation predictions saved by train.py, never on test data.
Product decision: calling a REAL photo fake (false positive) is worse than letting a fake through, because it
falsely accuses a real photographer; so the threshold is the most sensitive one that keeps the validation
false-positive rate (real called AI) at or below --fpr-limit.

  python -m visionguard.evaluate --run R1_cnn_baseline --label "Baseline CNN"
  python -m visionguard.evaluate --table R1_cnn_baseline R6_mnv2_finetune
"""
import argparse
import json
import sys

import keras
import mlflow
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from visionguard import config
from visionguard.metrics import classification_metrics, plot_confusion, plot_roc, predict_split


def choose_threshold(y, p, fpr_limit: float = 0.10) -> dict:
    fpr, tpr, thr = skm.roc_curve(y, p)
    ok = np.where(fpr <= fpr_limit)[0]
    best = ok[np.argmax(tpr[ok] == tpr[ok].max())]        # highest threshold reaching the best recall
    f1s = [skm.f1_score(y, (p >= t).astype(int)) for t in np.unique(p)]
    return {"threshold": float(thr[best]), "val_fpr": float(fpr[best]), "val_recall": float(tpr[best]),
            "fpr_limit": fpr_limit, "f1_optimal_threshold": float(np.unique(p)[int(np.argmax(f1s))])}


def evaluate(run: str, label: str, fpr_limit: float, force: bool):
    out = config.RUNS / run
    done = out / "test_metrics.json"
    if done.exists() and not force:
        sys.exit(f"{run} was already evaluated on the test set ({done}); the test set is used once. Use --force to override.")
    model = keras.models.load_model(out / "model.keras")
    img_size = model.input_shape[1]

    val = pd.read_csv(out / "val_preds.csv")
    choice = choose_threshold(val.label.to_numpy(), val.prob_ai.to_numpy(), fpr_limit)
    thr = choice["threshold"]

    df, probs = predict_split(model, "test", img_size)
    at_thr = classification_metrics(df.label, probs, thr)
    at_half = classification_metrics(df.label, probs, 0.5)
    pd.DataFrame({"path": df.path, "label": df.label, "prob_ai": probs, "orig_path": df.orig_path,
                  "orig_width": df.orig_width, "orig_height": df.orig_height,
                  "orig_format": df.orig_format}).to_csv(out / "test_preds.csv", index=False)

    figs = config.FIGURES / "runs"
    plot_confusion(df.label, probs, thr, f"{label} (test)", figs / f"{run}_test_confusion.png")
    plot_roc(df.label, probs, f"{label} (test)", figs / f"{run}_test_roc.png", thr)
    result = {"run": run, "label": label, "threshold_choice": choice, "test_at_threshold": at_thr,
              "test_at_0.5": at_half}
    done.write_text(json.dumps(result, indent=1))

    run_id = (out / "run_id.txt").read_text().strip() if (out / "run_id.txt").exists() else None
    if run_id:
        config.load_env()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("decision_threshold", round(thr, 6))
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in at_thr.items() if k != "threshold"})
            mlflow.log_metrics({f"test_at_0.5_{k}": float(v) for k, v in at_half.items() if k != "threshold"})
            for name in ("test_confusion", "test_roc"):
                mlflow.log_artifact(str(figs / f"{run}_{name}.png"), "figures")
            mlflow.log_artifact(str(out / "test_preds.csv"), "predictions")
            mlflow.log_artifact(str(done), "evaluation")
    print(json.dumps(result, indent=1))


def table(runs):
    rows = ["| model | threshold | accuracy | precision | recall | F1 | ROC-AUC | real called AI (FP) | AI passed as real (FN) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        r = json.loads((config.RUNS / run / "test_metrics.json").read_text())
        for tag, key in (("chosen threshold", "test_at_threshold"), ("0.5, for reference", "test_at_0.5")):
            m = r[key]
            rows.append(f"| {r['label']} ({tag}) | {m['threshold']:.3f} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                        f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['fp']} | {m['fn']} |")
    out = config.TABLES / "2.5_test_metrics.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows)); print("written to", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # MLflow prints an emoji link on run end
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--label", default=None)
    ap.add_argument("--fpr-limit", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--table", nargs="+")
    a = ap.parse_args()
    if a.table:
        table(a.table)
    else:
        evaluate(a.run, a.label or a.run, a.fpr_limit, a.force)
