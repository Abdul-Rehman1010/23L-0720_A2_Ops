"""History page protection for a public deployment (opt-in through HISTORY_PASSWORD)."""
import base64
from dataclasses import replace

import pytest
from conftest import FakePredictor, make_image, upload

from webapp import create_app
from webapp.db import HistoryStore

PASSWORD = "s3cret-history"


def basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture
def protected(settings):
    app = create_app(settings=replace(settings, history_password=PASSWORD), predictor=FakePredictor(),
                     store=HistoryStore(settings.database_url))
    return app.test_client()


def test_history_is_open_when_no_password_is_set(client):
    assert client.get("/history").status_code == 200


def test_history_asks_for_the_password(protected):
    r = protected.get("/history")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Basic")
    assert b"History is protected" in r.data


@pytest.mark.parametrize("password", ["", "wrong", PASSWORD[:-1], PASSWORD + "x"])
def test_wrong_password_is_refused(protected, password):
    assert protected.get("/history", headers=basic("anyone", password)).status_code == 401


def test_correct_password_opens_the_history_with_any_user_name(protected):
    for user in ("visitor", "abdul", ""):
        assert protected.get("/history", headers=basic(user, PASSWORD)).status_code == 200


def test_the_rest_of_the_site_stays_open_and_the_password_is_not_in_the_pages(protected):
    assert protected.get("/").status_code == 200
    assert protected.get("/health").status_code == 200
    r = upload(protected, make_image(), "holiday.jpg")
    assert r.status_code == 200 and PASSWORD.encode() not in r.data
    assert PASSWORD.encode() not in protected.get("/history").data
