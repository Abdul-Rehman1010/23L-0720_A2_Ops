"""Inference module: load the champion model ONCE, then classify uploaded images.

Why the model is loaded once (at start-up) and not per request
  Reading a 15 MB .keras file, rebuilding the graph and warming up TensorFlow takes seconds; a prediction on one
  160x160 image takes tens of milliseconds. Loading per request would make every upload slow, burn CPU and memory
  (several copies of the model when requests overlap) and fall over under load.

Why training and serving preprocessing must match
  The network learned on images that went through `visionguard.preprocessing` (EXIF rotation, transparency on white,
  RGB, squash-resize to 160x160 with antialiased bilinear interpolation). A different resize filter, colour order or
  value range would shift every pixel statistic the network relies on, and accuracy would silently drop. The server
  therefore calls the very same function (`preprocess_bytes`) the training pipeline was built from.

Label convention: positive class = AI-generated (label "ai"); the other class is "real".

  python -m visionguard.inference samples/known_real.jpg samples/known_ai.jpg
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "champion.keras"


def decide(prob_ai: float, threshold: float) -> tuple[str, float]:
    """Turn the model's score P(AI) into (label, confidence).

    The label is "ai" when the score reaches the decision threshold (chosen on validation data, see Task 2.5).
    Because that threshold is not 0.5, the plain probability of the winning class can be below 50% for a "real"
    verdict. So confidence measures how far the score lies from the threshold on the side of the verdict,
    rescaled to 50% (exactly on the boundary: a coin flip) up to 100% (score 0 or 1). It is a margin, not a calibrated probability.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be strictly between 0 and 1, got {threshold}")
    if prob_ai >= threshold:
        return "ai", 0.5 + 0.5 * (prob_ai - threshold) / (1.0 - threshold)
    return "real", 0.5 + 0.5 * (threshold - prob_ai) / threshold


class PredictorError(RuntimeError):
    """The model (or the remote model API) could not produce a prediction."""


class LocalPredictor:
    """Holds the Keras model in memory and classifies image bytes. Create one per process and reuse it."""

    mode = "local"

    def __init__(self, model_path=DEFAULT_MODEL, threshold: float | None = None, meta_path=None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}. "
                                    "Run `python scripts/fetch_champion.py` to download the champion model.")
        meta_path = Path(meta_path) if meta_path else model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.model_name = meta.get("registered_model", model_path.stem)
        self.model_version = str(meta.get("version", "unknown"))
        self.img_size = int(meta.get("img_size", 160))
        # THRESHOLD from the environment wins; otherwise the threshold stored next to the model.
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        decide(0.5, self.threshold)  # validates the threshold early

        # heavy imports happen here, so that importing this module (e.g. in unit tests) stays cheap
        import keras
        import numpy as np

        from visionguard.preprocessing import preprocess_bytes
        self._preprocess = preprocess_bytes
        started = time.perf_counter()
        self._model = keras.saving.load_model(model_path, compile=False)
        self._model(np.zeros((1, self.img_size, self.img_size, 3), dtype="float32"), training=False)  # warm-up
        self.load_seconds = time.perf_counter() - started
        self._lock = threading.Lock()  # one prediction at a time: simple and safe with threaded servers

    def info(self) -> dict:
        return {"model_name": self.model_name, "model_version": self.model_version, "threshold": self.threshold,
                "img_size": self.img_size, "inference_mode": self.mode}

    def healthy(self) -> bool:
        return self._model is not None

    def predict_bytes(self, data: bytes, filename: str | None = None) -> dict:
        """`filename` is accepted so that LocalPredictor and RemotePredictor are interchangeable; only the remote one needs it."""
        started = time.perf_counter()
        try:
            batch = self._preprocess(data, self.img_size)
            with self._lock:
                prob_ai = float(self._model(batch, training=False).numpy()[0, 0])
        except Exception as exc:  # noqa: BLE001 - surface any failure as one error type
            raise PredictorError(f"prediction failed: {exc}") from exc
        label, confidence = decide(prob_ai, self.threshold)
        return {"label": label, "confidence": confidence, "prob_ai": prob_ai, "threshold": self.threshold,
                "model_name": self.model_name, "model_version": self.model_version,
                "inference_mode": self.mode, "latency_ms": (time.perf_counter() - started) * 1000.0}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Classify one or more images with the champion model.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=None, help="override the threshold stored with the model")
    args = ap.parse_args()

    predictor = LocalPredictor(args.model, args.threshold)
    print(f"model loaded ONCE in {predictor.load_seconds:.1f} s: {predictor.model_name} v{predictor.model_version} | "
          f"input {predictor.img_size}x{predictor.img_size} | decision threshold {predictor.threshold:.3f}")
    for path in args.images:
        r = predictor.predict_bytes(Path(path).read_bytes())
        verdict = "AI-generated" if r["label"] == "ai" else "real"
        print(f"{path:<28} -> {verdict:<13} confidence {r['confidence']:.1%}   "
              f"(P(AI) = {r['prob_ai']:.3f}, {r['latency_ms']:.0f} ms)")


if __name__ == "__main__":
    main()
