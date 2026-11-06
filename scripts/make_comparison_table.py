"""Task 3.5: one comparison table of all tracked training runs (validation metrics only), read from MLflow.

  python scripts/make_comparison_table.py
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402

# what each run changes (every run after R2 differs from R2 in ONE thing, except R8 and the baseline)
CHANGE = {
    "R1_cnn_baseline": "from-scratch CNN instead of pretrained backbone",
    "R1a_cnn_bn_momentum_0.99": "R1 with BatchNorm momentum 0.99 (flawed: superseded)",
    "R2_mnv2_frozen": "reference: MobileNetV2, frozen, 160 px, augmentation, lr 1e-3",
    "R2_mnv2_autolog": "same as R2, tracked with autolog (reproducibility twin)",
    "R3_mnv2_no_aug": "augmentation off",
    "R4_mnv2_lr1e-4": "learning rate 1e-4 (100-epoch cap reached, still improving)",
    "R4a_mnv2_lr1e-4_capped_40_epochs": "R4 with a 40-epoch cap (superseded)",
    "R5_mnv2_img224": "image size 224",
    "R6_mnv2_finetune": "stage B: last block unfrozen at lr 1e-5",
    "R7_effb0_frozen": "backbone EfficientNetB0",
    "R8_mnv2_tuned": "tuner's best: lr 3.8e-4, dropout 0.1, last 10 layers unfrozen (three changes together)",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.load_env()
    c = mlflow.MlflowClient()
    exp = c.get_experiment_by_name(config.EXPERIMENT)
    runs = [r for r in c.search_runs([exp.experiment_id], max_results=100) if r.info.run_name in CHANGE]
    runs.sort(key=lambda r: -r.data.metrics["final_val_roc_auc"])
    rows = ["| run | what changed | val AUC | val F1 | val precision | val recall | real called AI (FP) | AI passed as real (FN) | best epoch |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r.data.metrics
        rows.append(f"| {r.info.run_name} | {CHANGE[r.info.run_name]} | {m['final_val_roc_auc']:.3f} | {m['final_val_f1']:.3f} | "
                    f"{m['final_val_precision']:.3f} | {m['final_val_recall']:.3f} | {m['final_val_fp']:.0f} | "
                    f"{m['final_val_fn']:.0f} | {m['final_best_epoch']:.0f} |")
    out = config.TABLES / "3.5_runs.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print("written to", out)


if __name__ == "__main__":
    main()
