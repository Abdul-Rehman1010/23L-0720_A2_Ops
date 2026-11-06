"""All app settings come from environment variables (twelve-factor style). Nothing is hard-coded in the code."""
import os
from dataclasses import dataclass
from pathlib import Path

from visionguard import config

DEFAULT_DB = "sqlite:///history.db"


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    model_path: Path
    threshold: float | None          # None -> use the threshold stored next to the model (champion.json)
    database_url: str
    max_upload_mb: float
    inference_mode: str              # "local" (model in this process) or "remote" (model API, Task 8)
    model_api_url: str
    model_api_key: str
    debug: bool

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        model_path = Path(env.get("MODEL_PATH") or "models/champion.keras")
        if not model_path.is_absolute():
            model_path = config.ROOT / model_path
        return cls(
            model_path=model_path,
            threshold=_float_or_none(env.get("THRESHOLD")),
            database_url=env.get("DATABASE_URL") or DEFAULT_DB,
            max_upload_mb=float(env.get("MAX_UPLOAD_MB") or 5),
            inference_mode=(env.get("INFERENCE_MODE") or "local").lower(),
            model_api_url=env.get("MODEL_API_URL", ""),
            model_api_key=env.get("MODEL_API_KEY", ""),
            debug=(env.get("FLASK_DEBUG") or "0").lower() in ("1", "true", "yes", "on"),
        )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
