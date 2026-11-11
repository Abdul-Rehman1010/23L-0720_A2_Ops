"""Assemble a deployable copy of the web app for Vercel in build/vercel (git-ignored, generated, never edited by hand).

Why a separate folder: Vercel installs the requirements.txt of the folder it deploys. The project's own requirements.txt
contains TensorFlow (about 1.3 GB), which the web app does not need in remote mode, because the cloud API runs the model.
The bundle therefore holds only the web app, the five small modules it imports from `visionguard`, a light requirements
file (the pins of requirements-base.txt without gunicorn and numpy) and the stylesheet in public/ (Vercel serves public/
from its CDN).

    python scripts/make_vercel_bundle.py
    cd build/vercel
    npx vercel link --project a2-ops-23l-0720      (once, after `npx vercel login`)
    npx vercel --prod
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "vercel"
VISIONGUARD_MODULES = ["__init__.py", "config.py", "inference.py", "remote.py", "validation.py"]   # no TensorFlow in any of them at import
NOT_NEEDED = ("gunicorn", "numpy")

APP_PY = '''"""Entry point for Vercel: it looks for a Flask object named `app`."""
from webapp import create_app

app = create_app()
'''
VERCEL_JSON = {"$schema": "https://openapi.vercel.sh/vercel.json", "regions": ["sin1"]}    # Singapore, close to the database and to Pakistan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "webapp", OUT / "webapp", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "visionguard").mkdir(exist_ok=True)
    for name in VISIONGUARD_MODULES:
        shutil.copy2(ROOT / "visionguard" / name, OUT / "visionguard" / name)
    shutil.copytree(ROOT / "webapp" / "static", OUT / "public" / "static", dirs_exist_ok=True)

    pins = [line for line in (ROOT / "requirements-base.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and not line.lower().startswith(NOT_NEEDED)]
    (OUT / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8", newline="\n")
    (OUT / "app.py").write_text(APP_PY, encoding="utf-8", newline="\n")
    (OUT / ".python-version").write_text("3.12\n", encoding="utf-8", newline="\n")
    (OUT / "vercel.json").write_text(json.dumps(VERCEL_JSON, indent=2) + "\n", encoding="utf-8", newline="\n")

    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    size = sum((OUT / f).stat().st_size for f in files)
    print(f"bundle written to {OUT} ({len(files)} files, {size / 1024:.0f} kB)")
    print("requirements:", ", ".join(pins))


if __name__ == "__main__":
    main()
