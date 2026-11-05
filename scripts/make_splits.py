"""Stratified 70/15/15 train/val/test split of the cleaned manifest (seed fixed)."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/manifest.csv")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    m = pd.read_csv(args.manifest)
    train, rest = train_test_split(m, test_size=0.30, stratify=m["label"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.sort_values("path").to_csv(Path(args.out) / f"{name}.csv", index=False)
        print(f"{name:<5} n={len(part):4d}  ai={int(part.label.sum()):3d}  real={int((1 - part.label).sum()):3d}")
    assert not (set(train.sha256) & set(val.sha256) | set(train.sha256) & set(test.sha256)
                | set(val.sha256) & set(test.sha256)), "leakage between splits"
    print("no hash is shared between splits")


if __name__ == "__main__":
    main()
