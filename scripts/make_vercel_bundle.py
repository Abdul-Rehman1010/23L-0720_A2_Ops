"""Generate the deployable copy of the web app for Vercel in deploy/vercel (committed, never edited by hand).

Why a separate folder: Vercel installs the requirements.txt of the folder it deploys. The project's own requirements.txt
contains TensorFlow (about 1.3 GB), which the web app does not need in remote mode, because the cloud API runs the model.
The bundle therefore holds only the web app, the five small modules it imports from `visionguard`, a light requirements
file (the pins of requirements-base.txt without gunicorn and numpy) and the stylesheet in public/ (Vercel serves public/
from its CDN). In the Vercel project the Root Directory is set to deploy/vercel.

    python scripts/make_vercel_bundle.py          # rewrite deploy/vercel after any change to webapp/ or visionguard/

tests/test_vercel_bundle.py fails when deploy/vercel no longer matches the sources.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "deploy" / "vercel"
VISIONGUARD_MODULES = ["__init__.py", "config.py", "inference.py", "remote.py", "validation.py"]   # no TensorFlow in any of them at import
NOT_NEEDED = ("gunicorn", "numpy")

APP_PY = '''"""Entry point for Vercel: it looks for a Flask object named `app`."""
from webapp import create_app

app = create_app()
'''
VERCEL_JSON = {"$schema": "https://openapi.vercel.sh/vercel.json", "regions": ["sin1"]}    # Singapore, close to the database and to Pakistan


def build(out: Path = DEFAULT_OUT, root: Path = ROOT) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "webapp", out / "webapp", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "visionguard").mkdir(exist_ok=True)
    for name in VISIONGUARD_MODULES:
        shutil.copy2(root / "visionguard" / name, out / "visionguard" / name)
    shutil.copytree(root / "webapp" / "static", out / "public" / "static", dirs_exist_ok=True)

    pins = [line for line in (root / "requirements-base.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and not line.lower().startswith(NOT_NEEDED)]
    (out / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8", newline="\n")
    (out / "app.py").write_text(APP_PY, encoding="utf-8", newline="\n")
    (out / ".python-version").write_text("3.12\n", encoding="utf-8", newline="\n")
    (out / "vercel.json").write_text(json.dumps(VERCEL_JSON, indent=2) + "\n", encoding="utf-8", newline="\n")
    return pins


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    pins = build(out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"bundle written to {out} ({len(files)} files, {sum((out / f).stat().st_size for f in files) / 1024:.0f} kB)")
    print("requirements:", ", ".join(pins))


if __name__ == "__main__":
    main()
