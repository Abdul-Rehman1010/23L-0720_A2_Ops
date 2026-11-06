"""Task 3.6: load the registered model by ALIAS (not by version number) and classify one image.

  python scripts/predict_by_alias.py --image samples/known_ai.jpg
"""
import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402
from visionguard.preprocessing import preprocess_bytes  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()
    config.load_env()

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(config.REGISTERED_MODEL, args.alias)
    threshold = float(mv.tags["decision_threshold"])
    size = int(mv.tags["img_size"])
    print(f"alias '{args.alias}' -> {config.REGISTERED_MODEL} version {mv.version} "
          f"(source run {mv.tags['source_run']}, threshold {threshold:.3f}, image size {size})")

    model = mlflow.keras.load_model(f"models:/{config.REGISTERED_MODEL}@{args.alias}")
    prob = float(model.predict(preprocess_bytes(Path(args.image).read_bytes(), size), verbose=0)[0, 0])
    label = "AI-generated" if prob >= threshold else "real"
    confidence = prob if prob >= threshold else 1 - prob
    print(f"image      : {args.image}")
    print(f"P(AI)      : {prob:.3f}")
    print(f"prediction : {label} (confidence {confidence:.1%})")


if __name__ == "__main__":
    main()
