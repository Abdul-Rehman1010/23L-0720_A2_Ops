"""Create the three files used to demonstrate upload validation (Task 4.2) in samples/bad/.

  notes.txt    - a text file: wrong type, rejected by the extension allow-list
  too_big.jpg  - a valid JPEG larger than the 5 MB limit (git-ignored, regenerate with this script)
  corrupt.jpg  - the first part of a real JPEG: has the right name and header but cannot be fully decoded
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "bad"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "notes.txt").write_text("This is a plain text file, not an image.\n", encoding="utf-8")

    rng = np.random.default_rng(0)
    side = 2600
    while True:  # noise does not compress, so the JPEG gets big quickly
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8)).save(buf, "JPEG", quality=95)
        if buf.tell() > 6 * 1024 * 1024:
            break
        side += 300
    (OUT / "too_big.jpg").write_bytes(buf.getvalue())

    real = (ROOT / "samples" / "known_real.jpg").read_bytes()
    (OUT / "corrupt.jpg").write_bytes(real[: len(real) // 2])

    for f in sorted(OUT.iterdir()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
