"""Shared fixtures. Tests use a fake predictor, so they need neither TensorFlow nor the model file nor DagsHub."""
import io

import pytest
from PIL import Image

from visionguard.inference import decide
from webapp import create_app
from webapp.db import HistoryStore
from webapp.settings import Settings


class FakePredictor:
    mode = "local"

    def __init__(self, prob_ai=0.9, threshold=0.75, healthy=True):
        self.prob_ai, self.threshold, self._healthy = prob_ai, threshold, healthy

    def info(self):
        return {"model_name": "fake-model", "model_version": "7", "threshold": self.threshold,
                "img_size": 160, "inference_mode": self.mode}

    def healthy(self):
        return self._healthy

    def predict_bytes(self, data):
        label, confidence = decide(self.prob_ai, self.threshold)
        return {"label": label, "confidence": confidence, "prob_ai": self.prob_ai, "threshold": self.threshold,
                "model_name": "fake-model", "model_version": "7", "inference_mode": self.mode, "latency_ms": 12.0}


def make_image(fmt="JPEG", size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(model_path=tmp_path / "none.keras", threshold=0.75, database_url=f"sqlite:///{tmp_path / 't.db'}",
                    max_upload_mb=1, inference_mode="local", model_api_url="", model_api_key="", debug=False)


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def app(settings, predictor):
    return create_app(settings=settings, predictor=predictor, store=HistoryStore(settings.database_url))


@pytest.fixture
def client(app):
    return app.test_client()


def upload(client, data, filename="photo.jpg", field="image"):
    return client.post("/predict", data={field: (io.BytesIO(data), filename)}, content_type="multipart/form-data")
