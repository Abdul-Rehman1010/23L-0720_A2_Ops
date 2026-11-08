"""Model API (Task 7): authentication, validation, error format. A fake predictor is used, so no TensorFlow is needed."""
import io

import pytest
from conftest import FakePredictor, make_image

from model_api import ApiSettings, create_app

KEY = "k" * 40


@pytest.fixture
def api_settings(tmp_path):
    return ApiSettings(api_key=KEY, model_path=tmp_path / "none.keras", threshold=0.75, max_upload_mb=1)


@pytest.fixture
def api(api_settings):
    return create_app(settings=api_settings, predictor=FakePredictor()).test_client()


def post(client, data=None, filename="photo.jpg", key=KEY, field="image"):
    headers = {"X-API-Key": key} if key is not None else {}
    body = {field: (io.BytesIO(data if data is not None else make_image()), filename)}
    return client.post("/predict", data=body, headers=headers, content_type="multipart/form-data")


def test_health_is_public_and_reports_the_model(api):
    r = api.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert (body["status"], body["model_loaded"], body["model_version"]) == ("ok", True, "7")
    assert KEY not in r.get_data(as_text=True)


def test_health_degraded_returns_503(api_settings):
    client = create_app(settings=api_settings, predictor=FakePredictor(healthy=False)).test_client()
    r = client.get("/health")
    assert r.status_code == 503 and r.get_json()["status"] == "degraded"


def test_predict_returns_the_verdict_and_ids(api):
    r = post(api)
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == "ai" and 0.5 <= body["confidence"] <= 1
    assert body["model_version"] == "7" and len(body["image_sha256"]) == 64
    assert "inference_mode" not in body
    assert body["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("key", [None, "", "wrong-key", KEY[:-1], KEY + "x"])
def test_missing_or_wrong_key_is_401(api, key):
    r = post(api, key=key)
    assert r.status_code == 401
    assert r.get_json()["error"] == "unauthorized"
    assert "X-API-Key" in r.headers["WWW-Authenticate"]


def test_key_is_checked_before_the_upload_is_read(api):
    # a garbage body without the key is still just a 401: validation never runs
    r = api.post("/predict", data=b"not multipart at all", content_type="text/plain")
    assert r.status_code == 401


def test_no_file_is_400(api):
    r = post(api, field="wrong_field_name")
    assert r.status_code == 400 and r.get_json()["error"] == "bad_request"


def test_wrong_file_type_is_415(api):
    r = post(api, data=b"just some text", filename="notes.txt")
    assert r.status_code == 415 and r.get_json()["error"] == "unsupported_media_type"


def test_corrupt_image_is_400(api):
    r = post(api, data=b"\xff\xd8\xff\xe0 this is not really a jpeg", filename="corrupt.jpg")
    assert r.status_code == 400


def test_content_is_checked_not_just_the_name(api):
    r = post(api, data=make_image("PNG"), filename="really_a_png.txt")
    assert r.status_code == 415
    r = post(api, data=b"MZ\x90\x00 pretend executable", filename="program.jpg")
    assert r.status_code == 400


def test_too_large_is_413(api):
    r = post(api, data=b"\0" * (1024 * 1024 + 100), filename="huge.jpg")
    assert r.status_code == 413 and r.get_json()["error"] == "file_too_large"


def test_errors_are_json_not_html(api):
    assert api.get("/nothing-here").get_json()["error"] == "not_found"
    r = api.get("/predict")
    assert r.status_code == 405 and r.get_json()["error"] == "method_not_allowed"


def test_prediction_failure_is_a_clean_500(api_settings):
    from visionguard.inference import PredictorError

    class Broken(FakePredictor):
        def predict_bytes(self, data):
            raise PredictorError("boom: secret internals")

    client = create_app(settings=api_settings, predictor=Broken()).test_client()
    r = post(client)
    assert r.status_code == 500
    assert "secret internals" not in r.get_data(as_text=True)


@pytest.mark.parametrize("key", ["", "short", "x" * 15])
def test_refuses_to_start_without_a_proper_key(tmp_path, key):
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        ApiSettings(api_key=key, model_path=tmp_path / "m.keras")


def test_settings_from_env(tmp_path):
    s = ApiSettings.from_env({"MODEL_API_KEY": KEY, "MODEL_PATH": str(tmp_path / "m.keras"), "THRESHOLD": "0.6",
                              "MAX_UPLOAD_MB": "2"})
    assert (s.api_key, s.threshold, s.max_upload_bytes) == (KEY, 0.6, 2 * 1024 * 1024)
    with pytest.raises(ValueError):
        ApiSettings.from_env({})
