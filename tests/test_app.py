from conftest import FakePredictor, make_image, upload


def test_upload_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Analyse image" in r.data


def test_prediction_is_shown_and_recorded(client, app):
    r = upload(client, make_image(), "holiday.jpg")
    assert r.status_code == 200
    assert b"Likely AI-generated" in r.data
    rows = app.extensions["store"].page()
    assert len(rows) == 1
    row = rows[0]
    assert (row["original_filename"], row["label"], row["model_version"], row["inference_mode"]) == \
        ("holiday.jpg", "ai", "7", "local")
    assert len(row["image_sha256"]) == 64


def test_real_verdict(settings):
    from webapp import create_app
    from webapp.db import HistoryStore
    app = create_app(settings=settings, predictor=FakePredictor(prob_ai=0.1), store=HistoryStore(settings.database_url))
    r = upload(app.test_client(), make_image())
    assert b"Likely a real photo" in r.data


def test_wrong_file_type_rejected(client, app):
    r = upload(client, b"just some text", "notes.txt")
    assert r.status_code == 415
    assert b"Unsupported file type" in r.data
    assert app.extensions["store"].count() == 0


def test_text_file_with_image_name_cannot_be_decoded(client, app):
    r = upload(client, b"this is not a jpeg at all", "fake.jpg")
    assert r.status_code == 400
    assert b"cannot be read" in r.data
    assert app.extensions["store"].count() == 0


def test_truncated_image_cannot_be_decoded(client):
    data = make_image(size=(800, 600))
    r = upload(client, data[: len(data) // 3], "half.jpg")
    assert r.status_code == 400


def test_image_content_must_match_allowed_formats(client):
    r = upload(client, make_image("GIF"), "sneaky.png")
    assert r.status_code == 415


def test_oversize_upload_rejected(client, app):
    r = upload(client, b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024), "huge.jpg")   # limit is 1 MB in the tests
    assert r.status_code == 413
    assert b"File too large" in r.data
    assert app.extensions["store"].count() == 0


def test_missing_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"No file selected" in r.data


def test_filename_is_sanitized(client, app):
    upload(client, make_image(), "../../etc/pass wd<script>.jpg")
    name = app.extensions["store"].page()[0]["original_filename"]
    assert "/" not in name and "<" not in name and ".." not in name and name.endswith(".jpg")


def test_unknown_page_has_friendly_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert b"Page not found" in r.data


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["model_loaded"] and body["db_ok"]
    assert body["model_version"] == "7"


def test_health_degraded_when_model_missing(settings):
    from webapp import create_app
    from webapp.db import HistoryStore
    app = create_app(settings=settings, predictor=FakePredictor(healthy=False), store=HistoryStore(settings.database_url))
    r = app.test_client().get("/health")
    assert r.status_code == 503
    assert r.get_json()["status"] == "degraded"


def test_history_empty_then_paginated(client, app):
    assert b"No predictions yet" in client.get("/history").data
    for i in range(12):
        upload(client, make_image(), f"img{i}.jpg")
    page1 = client.get("/history").data
    page2 = client.get("/history?page=2").data
    assert b"12 records in total" in page1
    assert b"img11.jpg" in page1 and b"img0.jpg" not in page1          # newest first
    assert b"img1.jpg" in page2 and b"img0.jpg" in page2
    assert client.get("/history?page=999").status_code == 200           # out-of-range page is clamped


def test_uploaded_bytes_never_touch_disk(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    upload(client, make_image(), "private.jpg")
    assert {p.name for p in tmp_path.iterdir()} == before
    assert not any("private" in p.name for p in tmp_path.rglob("*"))


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_database_failure_does_not_lose_the_verdict(app, client, monkeypatch):
    from sqlalchemy.exc import OperationalError

    def broken(**kwargs):
        raise OperationalError("insert", {}, Exception("db down"))
    monkeypatch.setattr(app.extensions["store"], "add", broken)
    r = upload(client, make_image())
    assert r.status_code == 200
    assert b"could not be saved" in r.data

