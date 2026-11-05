"""Task 3.5: run the experiment grid one run at a time (each logged to DagsHub by train.py).

Every run after R2 changes exactly one thing relative to R2 (R1 is the from-scratch baseline).
  python scripts/run_grid.py                 # all runs
  python scripts/run_grid.py --only R3_no_aug R4_lr1e-4
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE = ["--model", "mobilenetv2", "--img-size", "160", "--lr", "1e-3", "--augment", "--tracking", "manual"]


def variant(**changes):
    args = list(BASE)
    for flag, value in changes.items():
        flag = "--" + flag.replace("_", "-")
        if value is True:
            args = [a for a in args if a != flag] + [flag]
        elif value is False:                       # --augment -> --no-augment
            args = [a for a in args if a != flag] + ["--no-" + flag[2:]]
        else:
            i = args.index(flag) if flag in args else None
            if i is None:
                args += [flag, str(value)]
            else:
                args[i + 1] = str(value)
    return args


GRID = {
    "R1_cnn_baseline":   ["--model", "cnn", "--img-size", "160", "--lr", "1e-3", "--augment", "--epochs", "40",
                          "--tracking", "manual"],
    "R2_mnv2_frozen":    variant(),
    "R3_mnv2_no_aug":    variant(augment=False),
    "R4_mnv2_lr1e-4":    variant(lr="1e-4"),
    "R5_mnv2_img224":    variant(img_size=224),
    "R6_mnv2_finetune":  variant(finetune=True),
    "R7_effb0_frozen":   variant(model="efficientnetb0"),
    "R2_mnv2_autolog":   variant(tracking="autolog"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run names to execute (default: all)")
    sel = ap.parse_args().only
    (ROOT / "runs" / "logs").mkdir(parents=True, exist_ok=True)
    for name, args in GRID.items():
        if sel and name not in sel:
            continue
        log = ROOT / "runs" / "logs" / f"{name}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.run([sys.executable, "-m", "visionguard.train", "--name", name, *args],
                                  cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        result = next((l for l in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines())
                       if l.startswith("RESULT")), "no RESULT line")
        print(f"    exit {code} in {(time.time() - t0) / 60:.1f} min  {result}", flush=True)


if __name__ == "__main__":
    main()
