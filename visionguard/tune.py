"""Task 2.7: Keras Tuner random search for VisionGuard (MobileNetV2): learning rate, dropout, unfrozen layers.

Each trial builds a fresh MobileNetV2 model, unfreezes the last N backbone layers (BatchNorm stays frozen;
N = 0 means a frozen backbone), trains briefly with early stopping and is scored on VALIDATION ROC-AUC.
  python -m visionguard.tune --trials 10
  python -m visionguard.tune --summary
"""
import argparse
import sys

import keras
import keras_tuner as kt
import mlflow
import pandas as pd

from visionguard import config
from visionguard.data import make_dataset
from visionguard.models import build_transfer, compile_model, unfreeze

IMG_SIZE, EPOCHS, PATIENCE = 160, 10, 3
DIR = config.RUNS / "tuner"
TABLE = config.TABLES / "2.7_vg_tuner.md"


def build(hp):
    lr = hp.Float("learning_rate", 1e-5, 3e-3, sampling="log")
    dropout = hp.Float("dropout", 0.1, 0.6, step=0.1)
    n_unfrozen = hp.Choice("n_unfrozen_layers", [0, 10, 20, 30])
    model = build_transfer("mobilenetv2", IMG_SIZE, dropout)
    if n_unfrozen:
        unfreeze(model, last_n=n_unfrozen)
    return compile_model(model, lr)


def trial_table(tuner, n):
    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=n):
        hp = t.hyperparameters.values
        rows.append({"trial": t.trial_id, "learning_rate": hp["learning_rate"], "dropout": hp["dropout"],
                     "n_unfrozen_layers": hp["n_unfrozen_layers"], "val_auc": t.score})
    return pd.DataFrame(rows)


def write_table(df):
    lines = ["| rank | trial | learning_rate | dropout | unfrozen layers | best val_auc |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(df.itertuples(), start=1):
        lines.append(f"| {i} | {r.trial} | {r.learning_rate:.5f} | {r.dropout:.1f} | {r.n_unfrozen_layers} | {r.val_auc:.4f} |")
    best = df.iloc[0]
    lines += ["", f"Best settings: learning_rate = {best.learning_rate:.5f}, dropout = {best.dropout:.1f}, "
                  f"unfrozen layers = {int(best.n_unfrozen_layers)} (val_auc {best.val_auc:.4f})", "",
              "Mean val_auc by number of unfrozen layers:", ""]
    lines += [f"- {int(k)} unfrozen: {v:.4f} (n={int((df.n_unfrozen_layers == k).sum())})"
              for k, v in df.groupby("n_unfrozen_layers").val_auc.mean().items()]
    lines += ["", "Spearman correlation of each setting with val_auc:", ""]
    lines += [f"- {c}: {df[c].corr(df.val_auc, method='spearman'):+.2f}"
              for c in ("learning_rate", "dropout", "n_unfrozen_layers")]
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    if args.summary:
        print(TABLE.read_text(encoding="utf-8"))
        return
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.load_env()
    keras.utils.set_random_seed(config.SEED)
    train_ds = make_dataset("train", IMG_SIZE, 32, training=True, augment=True)
    val_ds = make_dataset("val", IMG_SIZE, 32)

    tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", "max"), max_trials=args.trials,
                            seed=config.SEED, directory=str(DIR), project_name="mobilenetv2", overwrite=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])

    df = trial_table(tuner, args.trials)
    df.to_csv(config.TABLES / "2.7_vg_tuner.csv", index=False)
    print(write_table(df))

    if not args.no_mlflow:
        mlflow.set_experiment(config.EXPERIMENT)
        commit, dirty = config.git_commit()
        with mlflow.start_run(run_name="T1_keras_tuner_mnv2"):
            mlflow.set_tags({"git_commit": commit, "git_dirty": str(dirty), "dvc_data_md5": config.dvc_data_md5(),
                             "seed": str(config.SEED), "backbone": "mobilenetv2", "stage": "hyperparameter_search"})
            best = df.iloc[0]
            mlflow.log_params({"trials": args.trials, "epochs_per_trial": EPOCHS, "objective": "val_auc",
                               "best_learning_rate": best.learning_rate, "best_dropout": best.dropout,
                               "best_n_unfrozen_layers": int(best.n_unfrozen_layers)})
            mlflow.log_metric("best_val_auc", float(best.val_auc))
            mlflow.log_artifact(str(TABLE), "tuner")
            mlflow.log_artifact(str(config.TABLES / "2.7_vg_tuner.csv"), "tuner")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        raise
