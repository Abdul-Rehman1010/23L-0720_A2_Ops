"""Tasks 3.1 / 3.2 tool lab: a tiny Keras model on Fashion-MNIST tracked with MLflow.

Local mode (3.1):   python labs/mlflow_local/toy_mlflow.py --lr 0.001
DagsHub mode (3.2): python labs/mlflow_local/toy_mlflow.py --lr 0.001 --remote
Remote settings (MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD) come from environment
variables, loaded from the git-ignored .env file; nothing secret is written in code.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from dotenv import load_dotenv
from keras import callbacks, layers, models, optimizers, utils
from keras.datasets import fashion_mnist

HERE = Path(__file__).resolve().parent
LOCAL_URI = f"sqlite:///{(HERE / 'mlflow.db').as_posix()}"
SEED = 42

# MLflow prints an emoji link when a run ends; a cp1252 Windows console would crash on it
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pick_tracking(remote: bool) -> str:
    if not remote:
        return LOCAL_URI
    load_dotenv(HERE.parents[1] / ".env")
    missing = [k for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
               if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)} (fill them in .env)")
    return os.environ["MLFLOW_TRACKING_URI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--remote", action="store_true", help="log to DagsHub instead of the local store")
    args = ap.parse_args()

    uri = pick_tracking(args.remote)
    mlflow.set_tracking_uri(uri)
    exp_name = "23L-0720_A2_Ops-toy"
    if mlflow.get_experiment_by_name(exp_name) is None and not args.remote:
        # keep local artifacts next to the local database
        mlflow.create_experiment(exp_name, artifact_location=(HERE / "artifacts").resolve().as_uri())
    mlflow.set_experiment(exp_name)

    utils.set_random_seed(SEED)
    (x_tr, y_tr), (x_va, y_va) = fashion_mnist.load_data()
    x_tr, y_tr = x_tr[: args.n_train] / 255.0, y_tr[: args.n_train]
    x_va, y_va = x_va[:2000] / 255.0, y_va[:2000]

    model = models.Sequential([
        layers.Input((28, 28)), layers.Flatten(),
        layers.Dense(args.units, activation="relu"),
        layers.Dense(10, activation="softmax")])
    model.compile(optimizers.Adam(args.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy-lr{args.lr}") as run:
        mlflow.set_tags({"lab": "3.1-3.2", "dataset": "fashion_mnist_subset",
                         "tracking": "dagshub" if args.remote else "local"})
        mlflow.log_params({"learning_rate": args.lr, "units": args.units, "epochs": args.epochs,
                           "batch_size": args.batch_size, "n_train": args.n_train,
                           "optimizer": "adam", "seed": SEED})
        per_epoch = callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: mlflow.log_metrics(logs, step=ep))
        hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epochs,
                         batch_size=args.batch_size, verbose=2, callbacks=[per_epoch])

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(hist.history["loss"], label="train loss")
        ax.plot(hist.history["val_loss"], label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(f"lr={args.lr}"); ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        mlflow.log_metric("final_val_accuracy", hist.history["val_accuracy"][-1])
        mlflow.keras.log_model(model, name="model")
        print(f"\nrun id      : {run.info.run_id}")
        print(f"tracking uri: {uri}")
        print(f"final val accuracy: {hist.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
