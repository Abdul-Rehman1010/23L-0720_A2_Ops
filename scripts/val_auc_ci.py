"""Bootstrap 95% confidence intervals for validation ROC-AUC of every finished run (validation data only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
B = 2000


def main():
    preds = {d.name: pd.read_csv(d / "val_preds.csv") for d in sorted((ROOT / "runs").glob("R*/")) if (d / "val_preds.csv").exists()}
    n = len(next(iter(preds.values())))
    idx = RNG.integers(0, n, size=(B, n))                      # same resamples for every run -> paired comparison
    y = next(iter(preds.values())).label.to_numpy()
    aucs = {}
    for name, df in preds.items():
        p = df.prob_ai.to_numpy()
        aucs[name] = np.array([roc_auc_score(y[i], p[i]) for i in idx if 0 < y[i].sum() < len(i)])
    rows = ["| run | val AUC | 95% CI |", "|---|---|---|"]
    for name, a in sorted(aucs.items(), key=lambda kv: -roc_auc_score(y, preds[kv[0]].prob_ai)):
        rows.append(f"| {name} | {roc_auc_score(y, preds[name].prob_ai):.3f} | {np.percentile(a, 2.5):.3f} to {np.percentile(a, 97.5):.3f} |")
    best = max(aucs, key=lambda k: roc_auc_score(y, preds[k].prob_ai))
    rows += ["", f"Paired bootstrap: probability that {best} has a higher AUC than each other run", ""]
    for name, a in aucs.items():
        if name != best:
            rows.append(f"- vs {name}: {np.mean(aucs[best][: len(a)] > a):.2f}")
    out = ROOT / "reports" / "tables" / "3.5_val_auc_ci.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
