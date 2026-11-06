"""Download the `champion` model ONCE from the DagsHub registry into models/ for the serving app.

The web app and the model API read models/champion.keras and models/champion.json from disk; they never
call DagsHub per request. Run this whenever the champion alias moves to a new version.
  python scripts/fetch_champion.py
"""
import json
import sys
from pathlib import Path

import mlflow
import mlflow.keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.load_env()
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(config.REGISTERED_MODEL, "champion")
    model = mlflow.keras.load_model(f"models:/{config.REGISTERED_MODEL}@champion")
    out = config.ROOT / "models"
    out.mkdir(exist_ok=True)
    model.save(out / "champion.keras")
    meta = {"registered_model": config.REGISTERED_MODEL, "version": int(mv.version),
            "source_run": mv.tags["source_run"], "source_run_id": mv.tags["source_run_id"],
            "threshold": float(mv.tags["decision_threshold"]), "img_size": int(mv.tags["img_size"]),
            "val_roc_auc": float(mv.tags["val_roc_auc"]), "positive_class": mv.tags["positive_class"]}
    (out / "champion.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out / 'champion.keras'} ({(out / 'champion.keras').stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
