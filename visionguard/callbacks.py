"""Training callbacks: early stopping / checkpoint / LR schedule (Task 2.4) and manual MLflow logging (3.3)."""
import mlflow
from keras import callbacks


class MLflowEpochLogger(callbacks.Callback):
    """Manual MLflow logging: every Keras epoch metric (including learning_rate) as a stepped metric.
    `offset` lets a second training stage continue the step axis of the first."""

    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        values = {k: float(v) for k, v in (logs or {}).items()}
        mlflow.log_metrics(values, step=self.offset + epoch)


def training_callbacks(checkpoint_path, patience: int = 5):
    return [
        callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]


def best_epoch(history) -> int:
    """1-based epoch with the lowest validation loss (the weights early stopping restores)."""
    vl = history.history["val_loss"]
    return vl.index(min(vl)) + 1
