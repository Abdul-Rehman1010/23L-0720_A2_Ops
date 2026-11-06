"""Database trail (Task 5): one permanent row per prediction.

Answers: what was decided (label, confidence, P(AI), threshold), when (created_at, UTC), by which model
(model_name, model_version, inference_mode). The same code runs on SQLite locally and on PostgreSQL in Docker
Compose: only the DATABASE_URL differs. The uploaded image itself is never stored, only its SHA-256 fingerprint.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(8))                 # "ai" or "real"
    confidence: Mapped[float] = mapped_column(Float)
    prob_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(32))
    inference_mode: Mapped[str] = mapped_column(String(16))       # "local" or "remote"
    latency_ms: Mapped[float] = mapped_column(Float)

    def to_dict(self) -> dict:
        created = self.created_at if self.created_at.tzinfo else self.created_at.replace(tzinfo=timezone.utc)
        return {"id": self.id, "created_at": created, "original_filename": self.original_filename,
                "image_sha256": self.image_sha256, "label": self.label, "confidence": self.confidence,
                "prob_ai": self.prob_ai, "threshold": self.threshold, "model_name": self.model_name,
                "model_version": self.model_version, "inference_mode": self.inference_mode,
                "latency_ms": self.latency_ms}


class HistoryStore:
    def __init__(self, url: str):
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}   # the web server uses several threads
        self.engine = create_engine(url, **kwargs)
        self._session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)   # raises at start-up if the database is unreachable

    def add(self, *, filename: str, sha256: str, result: dict) -> int:
        row = Prediction(original_filename=filename[:255], image_sha256=sha256, label=result["label"],
                         confidence=result["confidence"], prob_ai=result["prob_ai"], threshold=result["threshold"],
                         model_name=result["model_name"], model_version=str(result["model_version"]),
                         inference_mode=result["inference_mode"], latency_ms=result["latency_ms"])
        with self._session() as session:
            session.add(row)
            session.commit()
            return row.id

    def page(self, page: int = 1, per_page: int = 10) -> list[dict]:
        stmt = select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()) \
            .limit(per_page).offset((max(page, 1) - 1) * per_page)
        with self._session() as session:
            return [row.to_dict() for row in session.scalars(stmt)]

    def count(self) -> int:
        with self._session() as session:
            return session.scalar(select(func.count()).select_from(Prediction)) or 0

    def ping(self) -> bool:
        try:
            with Session(self.engine) as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
