"""Paths, constants and small helpers shared by training and serving code."""
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RUNS = ROOT / "runs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
SEED = 42
EXPERIMENT = "23L-0720_A2_Ops"
REGISTERED_MODEL = "23L-0720_A2_Ops_VisionGuard"
LABELS = {0: "real", 1: "ai"}          # positive class (1) = AI-generated


def load_env():
    """Load .env without overriding variables that are already set in the shell."""
    load_dotenv(ROOT / ".env", override=False)


def git_commit() -> tuple[str, bool]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def dvc_data_md5() -> str:
    """md5 stored in the DVC pointer file of the cleaned dataset (identifies the data version)."""
    m = re.search(r"md5:\s*(\w+)", (ROOT / "data" / "clean.dvc").read_text())
    return m.group(1) if m else "unknown"
