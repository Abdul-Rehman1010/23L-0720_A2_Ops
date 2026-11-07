"""Task 5: print the latest rows of the prediction history from whichever database DATABASE_URL points to.

  python scripts/show_history.py --limit 10
Works the same on SQLite (local) and PostgreSQL (Compose).
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard import config  # noqa: E402
from webapp.db import HistoryStore  # noqa: E402
from webapp.settings import Settings  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    config.load_env()
    settings = Settings.from_env()
    store = HistoryStore(settings.database_url)
    print(f"database: {make_url(settings.database_url).render_as_string(hide_password=True)}")
    print(f"total rows: {store.count()} (showing the newest {args.limit})\n")
    tz = settings.tz
    tzname = datetime.now(tz).strftime("%Z")
    header = f"{'id':>3}  {'created_at (' + tzname + ')':<19}  {'file':<22}  {'label':<5}  {'conf':>6}  {'P(AI)':>6}  {'model':<6}  {'mode':<6}  {'ms':>5}"
    print(header)
    print("-" * len(header))
    for r in store.page(1, args.limit):
        print(f"{r['id']:>3}  {r['created_at'].astimezone(tz):%Y-%m-%d %H:%M:%S}  {r['original_filename'][:22]:<22}  {r['label']:<5}  "
              f"{r['confidence']:>6.1%}  {r['prob_ai']:>6.3f}  v{r['model_version']:<5}  {r['inference_mode']:<6}  {r['latency_ms']:>5.0f}")


if __name__ == "__main__":
    main()
