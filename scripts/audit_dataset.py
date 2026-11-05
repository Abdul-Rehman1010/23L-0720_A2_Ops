"""Task 1.2: walk the raw dataset and build a per-image audit table.

Per image: class, file size, width, height, colour mode, format, SHA-256.
Unreadable files are flagged, and exact duplicates (same SHA-256) are grouped.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_MAP = {"AiArtData": "ai", "RealArt": "real"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path, root: Path) -> dict:
    cls = CLASS_MAP.get(path.relative_to(root).parts[0], path.relative_to(root).parts[0])
    row = {"path": path.relative_to(root).as_posix(), "class": cls,
           "file_size": path.stat().st_size, "sha256": sha256_of(path),
           "width": None, "height": None, "mode": None, "format": None,
           "readable": True, "error": ""}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.verify()                      # cheap structural check
        with Image.open(path) as im:
            im.load()                        # full decode catches truncated data
    except Exception as e:                   # any decode failure counts as corrupt
        row.update(readable=False, error=f"{type(e).__name__}: {e}"[:200])
    return row


def stats(series: pd.Series) -> str:
    return f"{series.min():.0f} / {series.median():.0f} / {series.max():.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="reports/tables")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob("*") if p.is_file())
    df = pd.DataFrame([inspect(p, root) for p in files])

    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    df["is_duplicate"] = df.duplicated("sha256", keep="first")   # every copy after the first
    df.to_csv(out / "audit.csv", index=False)

    ok = df[df.readable]
    lines = ["# Dataset audit (Task 1.2)", "",
             f"Total files walked: {len(df)}", "",
             "## Counts per class", "",
             "| class | files | readable | corrupt | duplicate copies |", "|---|---|---|---|---|"]
    for cls, g in df.groupby("class"):
        lines.append(f"| {cls} | {len(g)} | {int(g.readable.sum())} | "
                     f"{int((~g.readable).sum())} | {int(g.is_duplicate.sum())} |")
    lines += ["", "## min / median / max (readable images only)", "",
              "| class | width | height | file size (bytes) |", "|---|---|---|---|"]
    for cls, g in ok.groupby("class"):
        lines.append(f"| {cls} | {stats(g.width)} | {stats(g.height)} | {stats(g.file_size)} |")
    lines += ["", "## Formats and colour modes", "",
              "Formats: " + ", ".join(f"{k}={v}" for k, v in ok.format.value_counts().items()),
              "", "Modes: " + ", ".join(f"{k}={v}" for k, v in ok["mode"].value_counts().items())]

    corrupt = df[~df.readable]
    lines += ["", f"## Corrupt files: {len(corrupt)}", ""]
    lines += [f"- `{r.path}`: {r.error}" for r in corrupt.itertuples()]

    dups = df[df.dup_group_size > 1]
    n_groups = dups.sha256.nunique()
    cross = dups.groupby("sha256")["class"].nunique().gt(1).sum()
    lines += ["", f"## Exact duplicates: {n_groups} groups, {int(df.is_duplicate.sum())} extra copies, "
                  f"{int(cross)} groups span both classes", ""]
    for sha, g in dups.groupby("sha256"):
        lines.append(f"- `{sha[:12]}` ({len(g)} files): " + ", ".join(f"`{p}`" for p in g.path))
    (out / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
