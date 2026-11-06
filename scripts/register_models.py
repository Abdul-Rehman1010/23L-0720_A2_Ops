"""Task 3.6: register two model versions in the MLflow registry on DagsHub and give one the alias `champion`.

  python scripts/register_models.py --versions R6_mnv2_finetune R8_mnv2_tuned --champion R8_mnv2_tuned
Each version gets tags: source run, decision threshold (chosen on validation data), image size, validation AUC.
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402
from visionguard.evaluate import choose_threshold  # noqa: E402

ALIAS = "champion"


def threshold_for(run_name: str) -> float:
    """Evaluated runs: the stored validation-chosen threshold. Others: same rule applied to validation predictions."""
    done = config.RUNS / run_name / "test_metrics.json"
    if done.exists():
        return json.loads(done.read_text())["threshold_choice"]["threshold"]
    val = pd.read_csv(config.RUNS / run_name / "val_preds.csv")
    return choose_threshold(val.label.to_numpy(), val.prob_ai.to_numpy())["threshold"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", required=True, help="run names, registered in this order")
    ap.add_argument("--champion", required=True)
    args = ap.parse_args()
    config.load_env()
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT)
    runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id], max_results=100)}

    champion_version = None
    for name in args.versions:
        run = runs[name]
        logged = mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list",
                                             filter_string=f"source_run_id='{run.info.run_id}'")
        mv = mlflow.register_model(f"models:/{logged[0].model_id}", config.REGISTERED_MODEL)
        tags = {"source_run": name, "source_run_id": run.info.run_id,
                "decision_threshold": f"{threshold_for(name):.6f}", "img_size": run.data.params["img_size"],
                "val_roc_auc": f"{run.data.metrics['final_val_roc_auc']:.4f}",
                "positive_class": "ai"}
        for k, v in tags.items():
            client.set_model_version_tag(config.REGISTERED_MODEL, mv.version, k, v)
        print(f"registered {name} as version {mv.version} (threshold {tags['decision_threshold']})")
        if name == args.champion:
            champion_version = mv.version
    client.set_registered_model_alias(config.REGISTERED_MODEL, ALIAS, champion_version)
    print(f"alias '{ALIAS}' -> version {champion_version}")


if __name__ == "__main__":
    main()
