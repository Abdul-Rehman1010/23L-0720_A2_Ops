"""Remote inference (Task 8): RemotePredictor on its own and inside the web app. A fake HTTP session replaces the network,
so every failure mode (down, slow, wrong key, garbage answer) can be tested without a real server."""
import math

import pytest
import requests
from conftest import make_image, upload

from visionguard.inference import PredictorError
from visionguard.remote import RemotePredictor
from webapp import create_app
from webapp.db import HistoryStore

KEY = "k" * 40
URL = "https://api.example.test"
GOOD = {"label": "real", "confidence": 0.887, "prob_ai": 0.169, "threshold": 0.751816, "model_name": "VG", "model_version": "2",
        "latency_ms": 758.0, "image_sha256": "a" * 64, "request_id": "4da57c9199c4"}
HEALTH = {"status": "ok", "model_loaded": True, "model_name": "VG", "model_version": "2", "threshold": 0.751816}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeSession:
    """Plays back a script: each item is a FakeResponse to return or an exception to raise."""

    def __init__(self, posts=(), gets=None):
        self.posts, self.gets = list(posts), list(gets if gets is not None else [FakeResponse(200, HEALTH)] * 20)
        self.post_calls, self.get_calls = [], []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        item = self.posts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        item = self.gets.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(session, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)
    return RemotePredictor(URL, KEY, session=session, **kwargs)


def ok():
    return FakeResponse(200, dict(GOOD))


def test_success_maps_the_answer_and_marks_it_remote():
    r = make(FakeSession([ok()])).predict_bytes(make_image(), "holiday.jpg")
    assert (r["label"], r["confidence"], r["prob_ai"], r["threshold"]) == ("real", 0.887, 0.169, 0.751816)
    assert (r["model_name"], r["model_version"], r["inference_mode"]) == ("VG", "2", "remote")
    assert r["api_request_id"] == "4da57c9199c4" and r["api_latency_ms"] == 758.0 and r["attempts"] == 1
    assert r["latency_ms"] >= 0                              # round trip measured by the web app, not the API's own number


def test_request_carries_the_key_the_file_and_a_timeout():
    session = FakeSession([ok()])
    make(session, timeout=7).predict_bytes(b"image-bytes", "holiday.jpg")
    url, kwargs = session.post_calls[0]
    assert url == f"{URL}/predict"
    assert kwargs["headers"] == {"X-API-Key": KEY}
    assert kwargs["files"]["image"] == ("holiday.jpg", b"image-bytes")
    assert kwargs["timeout"] == (5.0, 7.0)


@pytest.mark.parametrize("data, expected", [(b"\xff\xd8\xff\xe0abc", "image.jpg"), (b"\x89PNG\r\n\x1a\nxx", "image.png"),
                                            (b"RIFF1234WEBPVP8 ", "image.webp"), (b"BM6\x00", "image.bmp"), (b"???", "image.bin")])
def test_file_name_is_guessed_from_the_content_when_missing(data, expected):
    session = FakeSession([ok()])
    make(session).predict_bytes(data)
    assert session.post_calls[0][1]["files"]["image"][0] == expected


def test_one_retry_after_a_connection_error():
    session = FakeSession([requests.exceptions.ConnectionError("boom"), ok()])
    r = make(session).predict_bytes(make_image(), "a.jpg")
    assert r["label"] == "real" and r["attempts"] == 2 and len(session.post_calls) == 2


def test_gives_up_after_two_attempts_without_leaking_the_key():
    session = FakeSession([requests.exceptions.ConnectionError(f"failed with {KEY}"), requests.exceptions.ConnectionError("again")])
    with pytest.raises(PredictorError) as err:
        make(session).predict_bytes(make_image(), "a.jpg")
    assert len(session.post_calls) == 2
    assert "could not be reached" in str(err.value) and "2 attempts" in str(err.value)
    assert KEY not in str(err.value)


def test_timeout_is_a_predictor_error():
    session = FakeSession([requests.exceptions.ReadTimeout(), requests.exceptions.ReadTimeout()])
    with pytest.raises(PredictorError, match="did not answer within 10 s"):
        make(session).predict_bytes(make_image(), "a.jpg")


@pytest.mark.parametrize("status", [502, 503, 504])
def test_server_errors_are_retried(status):
    session = FakeSession([FakeResponse(status), ok()])
    assert make(session).predict_bytes(make_image(), "a.jpg")["attempts"] == 2


@pytest.mark.parametrize("status, text", [(401, "refused the API key"), (403, "refused the API key"),
                                          (400, "rejected the request"), (413, "rejected the request"), (415, "rejected the request")])
def test_client_errors_are_not_retried(status, text):
    session = FakeSession([FakeResponse(status, {"error": "x"}), ok()])
    with pytest.raises(PredictorError, match=text) as err:
        make(session).predict_bytes(make_image(), "a.jpg")
    assert len(session.post_calls) == 1 and KEY not in str(err.value)


@pytest.mark.parametrize("body", [None, {}, {"label": "maybe", **{k: GOOD[k] for k in GOOD if k != "label"}},
                                  {**GOOD, "confidence": 1.7}, {**GOOD, "prob_ai": "high"}, {**GOOD, "threshold": -0.1}, ["list"]])
def test_unexpected_answers_are_rejected(body):
    with pytest.raises(PredictorError):
        make(FakeSession([FakeResponse(200, body)])).predict_bytes(make_image(), "a.jpg")


def test_plain_http_is_refused_except_for_localhost():
    for url in ("https://api.example.test", "http://localhost:8080", "http://127.0.0.1:8080"):
        RemotePredictor(url, KEY, session=FakeSession())
    for url in ("http://api.example.test", "ftp://x", "api.example.test", "https://"):
        with pytest.raises(ValueError, match="https"):
            RemotePredictor(url, KEY, session=FakeSession())


def test_url_and_key_are_required():
    with pytest.raises(ValueError, match="MODEL_API_URL"):
        RemotePredictor("", KEY, session=FakeSession())
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        RemotePredictor(URL, "", session=FakeSession())


def test_starts_even_when_the_api_is_down_and_recovers():
    session = FakeSession(gets=[requests.exceptions.ConnectionError("down"), FakeResponse(200, HEALTH)])
    p = make(session, health_ttl=0)
    assert p.healthy() is True                                # second probe succeeded (ttl 0 forces a fresh probe)
    assert p.info()["model_version"] == "2"
    down = make(FakeSession(gets=[requests.exceptions.ConnectionError("down")] * 3), health_ttl=0)
    assert down.healthy() is False
    assert down.info()["model_version"] == "unknown" and math.isnan(down.info()["threshold"])


def test_health_answer_is_cached_for_the_ttl():
    session = FakeSession()
    p = make(session, health_ttl=60)
    for _ in range(5):
        p.healthy()
    assert len(session.get_calls) == 1                        # only the probe made at start-up


def test_a_degraded_health_answer_counts_as_unhealthy():
    p = make(FakeSession(gets=[FakeResponse(200, {**HEALTH, "status": "degraded"})]))
    assert p.healthy() is False
    p = make(FakeSession(gets=[FakeResponse(503, HEALTH)]))
    assert p.healthy() is False


def test_info_follows_the_model_the_api_reports():
    new = {**GOOD, "model_version": "3", "threshold": 0.6}
    p = make(FakeSession([FakeResponse(200, new)]))
    assert p.info()["model_version"] == "2"
    p.predict_bytes(make_image(), "a.jpg")
    assert (p.info()["model_version"], p.info()["threshold"]) == ("3", 0.6)


# ---- inside the web app ---------------------------------------------------------------------------------------------
@pytest.fixture
def web(settings):
    def build(session):
        store = HistoryStore(settings.database_url)
        app = create_app(settings=settings, predictor=make(session, health_ttl=0), store=store)
        return app.test_client(), store
    return build


def test_web_upload_through_the_remote_predictor_is_stored_as_remote(web):
    client, store = web(FakeSession([ok()]))
    r = upload(client, make_image(), "holiday.jpg")
    assert r.status_code == 200
    assert b"Likely a real photo" in r.data and b"4da57c9199c4" in r.data and b"remote" in r.data
    assert KEY.encode() not in r.data
    row = store.page()[0]
    assert (row["inference_mode"], row["model_version"], row["model_name"], row["label"]) == ("remote", "2", "VG", "real")
    assert row["original_filename"] == "holiday.jpg"


def test_web_shows_a_friendly_page_when_the_api_is_down(web):
    client, store = web(FakeSession([requests.exceptions.ConnectionError("down")] * 2))
    r = upload(client, make_image(), "holiday.jpg")
    assert r.status_code == 502
    assert b"Model service unavailable" in r.data and b"Traceback" not in r.data and KEY.encode() not in r.data
    assert store.count() == 0                                 # nothing is recorded for a failed prediction


def test_web_reports_a_wrong_key_as_unavailable_not_as_a_crash(web):
    client, store = web(FakeSession([FakeResponse(401, {"error": "unauthorized"})]))
    r = upload(client, make_image(), "holiday.jpg")
    assert r.status_code == 502 and store.count() == 0


def test_web_health_degrades_when_the_api_is_down(web):
    session = FakeSession(gets=[FakeResponse(200, HEALTH), requests.exceptions.ConnectionError("down"), FakeResponse(200, HEALTH)])
    client, _ = web(session)
    r = client.get("/health")
    assert r.status_code == 503 and r.get_json()["status"] == "degraded" and r.get_json()["inference_mode"] == "remote"
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["model_version"] == "2"
