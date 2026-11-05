"""Build data/clean from data/raw using the audit table.

- drops unreadable files and exact duplicates (keeps the first copy; a hash that appears in
  both classes is dropped entirely because its label is ambiguous)
- cleaning and the 512 px cap come from visionguard.preprocessing.clean_pil (shared with serving)
- writes lossless PNG named by content hash, so the container format carries no class signal
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.preprocessing import MAX_SIDE, clean_pil  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--audit", default="reports/tables/audit.csv")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    df = pd.read_csv(args.audit)
    df = df[df.readable]
    cross = df.groupby("sha256")["class"].nunique()
    ambiguous = set(cross[cross > 1].index)
    df = df[~df.sha256.isin(ambiguous)].drop_duplicates("sha256", keep="first")

    rows = []
    for r in df.itertuples():
        with Image.open(raw / r.path) as im:
            im = clean_pil(im, args.max_side)
            dest = out / r._2 / f"{r.sha256}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, format="PNG", optimize=False)
            rows.append({"path": dest.as_posix(), "class": r._2, "label": int(r._2 == "ai"),
                         "sha256": r.sha256, "clean_width": im.width, "clean_height": im.height,
                         "orig_path": r.path, "orig_width": r.width, "orig_height": r.height,
                         "orig_format": r.format, "orig_file_size": r.file_size})
    manifest = pd.DataFrame(rows)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    manifest.to_csv("data/splits/manifest.csv", index=False)
    print(f"kept {len(manifest)} of {len(pd.read_csv(args.audit))} files "
          f"(dropped {len(ambiguous)} ambiguous hashes, rest corrupt/duplicate)")
    print(manifest["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
