"""Train one VisionGuard model (baseline CNN or transfer learning) and optionally track it in MLflow.

  python -m visionguard.train --name R1_cnn --model cnn --lr 1e-3
  python -m visionguard.train --name R6_mnv2_ft --model mobilenetv2 --finetune --tracking manual
Only the train and validation splits are used here; the test split is reserved for evaluate.py.
"""
import argparse
import json
import shutil
import sys

import keras
import mlflow
import mlflow.keras
import mlflow.tensorflow
import pandas as pd

from visionguard import config
from visionguard.callbacks import MLflowEpochLogger, best_epoch, training_callbacks
from visionguard.data import make_dataset
from visionguard.metrics import (classification_metrics, plot_confusion, plot_curves, plot_roc,
                                 predict_split)
from visionguard.models import (BACKBONES, build_cnn, build_transfer, compile_model, param_counts,
                                trainable_layers, unfreeze)
from visionguard.preprocessing import IMG_SIZE


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", choices=["cnn", *BACKBONES], default="cnn")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--finetune", action="store_true", help="stage B: unfreeze the last block")
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--unfreeze-last-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tracking", choices=["none", "manual", "autolog"], default="none")
    return ap.parse_args()


def fit_stage(model, train_ds, val_ds, epochs, patience, ckpt, extra=()):
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, shuffle=False,
                     callbacks=[*training_callbacks(ckpt, patience), *extra])
    model.load_weights(ckpt)       # always continue from / finish with the best epoch's weights
    return hist, best_epoch(hist)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # MLflow prints an emoji link; cp1252 consoles crash on it
    args = parse_args()
    config.load_env()
    keras.utils.set_random_seed(args.seed)
    out = config.RUNS / args.name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    figs = config.FIGURES / "runs"
    figs.mkdir(parents=True, exist_ok=True)
    tracking = args.tracking != "none"

    train_ds = make_dataset("train", args.img_size, args.batch_size, training=True,
                            augment=args.augment, seed=args.seed)
    val_ds = make_dataset("val", args.img_size, args.batch_size)
    dropout = args.dropout if args.dropout is not None else (0.4 if args.model == "cnn" else 0.3)
    if args.model == "cnn":
        model = build_cnn(args.img_size, dropout)
    else:
        model = build_transfer(args.model, args.img_size, dropout)
    compile_model(model, args.lr)
    counts_a = param_counts(model)

    if tracking:
        mlflow.set_experiment(config.EXPERIMENT)
        if args.tracking == "autolog":
            mlflow.tensorflow.autolog(log_models=False)   # DagsHub rejects autolog's own model upload (BAD_REQUEST)
        run = mlflow.start_run(run_name=args.name)
        commit, dirty = config.git_commit()
        mlflow.set_tags({"git_commit": commit, "git_dirty": str(dirty), "dvc_data_md5": config.dvc_data_md5(),
                         "seed": str(args.seed), "backbone": args.model, "approach": args.tracking,
                         "label_positive": "ai"})
        if args.tracking == "manual":
            mlflow.log_params({**vars(args), "dropout": dropout, "params_total": counts_a["total"],
                               "params_trainable_stage_a": counts_a["trainable"]})
        (out / "run_id.txt").write_text(run.info.run_id)

    logger = [MLflowEpochLogger(0)] if args.tracking == "manual" else []
    hist_a, best_a = fit_stage(model, train_ds, val_ds, args.epochs, args.patience,
                               out / "best_stage_a.keras", logger)
    stages = {"a": (hist_a, best_a)}
    title_a = f"{args.name}: stage A (lr {args.lr:g})" if args.finetune else f"{args.name} (lr {args.lr:g})"
    plot_curves(hist_a, best_a, title_a, figs / f"{args.name}_curves_a.png")
    layers_a = trainable_layers(model)

    if args.finetune:
        n = unfreeze(model, args.model, args.unfreeze_last_n)
        compile_model(model, args.finetune_lr)
        counts_b = param_counts(model)
        layers_b = trainable_layers(model)
        offset = len(hist_a.epoch)
        logger = [MLflowEpochLogger(offset)] if args.tracking == "manual" else []
        hist_b, best_b = fit_stage(model, train_ds, val_ds, args.finetune_epochs, args.patience,
                                   out / "best_stage_b.keras", logger)
        stages["b"] = (hist_b, best_b)
        plot_curves(hist_b, best_b, f"{args.name}: stage B, last block unfrozen (lr {args.finetune_lr:g})",
                    figs / f"{args.name}_curves_b.png")
        (out / "trainable_layers.json").write_text(json.dumps({"stage_a": layers_a, "stage_b": layers_b}, indent=1))
        if args.tracking == "manual":
            mlflow.log_params({"unfrozen_backbone_layers": n, "params_trainable_stage_b": counts_b["trainable"],
                               "stage_b_step_offset": offset})

    # validation-only evaluation of the restored best weights
    df, probs = predict_split(model, "val", args.img_size)
    val = classification_metrics(df.label, probs, 0.5)
    best_hist, best_ep = stages["b"] if args.finetune else stages["a"]
    final = {f"val_{k}": v for k, v in val.items() if k != "threshold"}
    final.update(best_epoch=best_ep, best_val_loss=min(best_hist.history["val_loss"]))
    pd.DataFrame({"path": df.path, "label": df.label, "prob_ai": probs}).to_csv(out / "val_preds.csv", index=False)
    plot_confusion(df.label, probs, 0.5, f"{args.name} (validation)", figs / f"{args.name}_val_confusion.png")
    plot_roc(df.label, probs, f"{args.name} (validation)", figs / f"{args.name}_val_roc.png", 0.5)
    model.save(out / "model.keras")

    if tracking:
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in final.items()})
        for f in figs.glob(f"{args.name}_*.png"):
            mlflow.log_artifact(str(f), "figures")
        for split in ("train", "val", "test"):
            mlflow.log_artifact(str(config.SPLITS / f"{split}.csv"), "splits")
        mlflow.log_artifact(str(config.ROOT / "requirements-lock.txt"), "environment")
        mlflow.log_artifact(str(out / "val_preds.csv"), "predictions")
        if args.finetune:
            mlflow.log_artifact(str(out / "trainable_layers.json"), "model_info")
        if args.tracking == "autolog":
            # DagsHub rejects log_model on a run that already has autolog metrics (log_batch BAD_REQUEST)
            mlflow.log_artifact(str(out / "model.keras"), "model_file")
        else:
            mlflow.keras.log_model(model, name="model")
        mlflow.end_run()

    print("\nRESULT", json.dumps({"name": args.name, **{k: round(float(v), 4) for k, v in final.items()}}))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        raise
