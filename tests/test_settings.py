"""Settings: the database URL from a hosting provider can be used as it is."""
import pytest

from webapp.settings import Settings, normalise_database_url


@pytest.mark.parametrize("given, expected", [
    ("postgresql://user:pw@ep-cool-123.ap-southeast-1.aws.neon.tech/neondb?sslmode=require",
     "postgresql+psycopg://user:pw@ep-cool-123.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"),
    ("postgres://user:pw@host:5432/db", "postgresql+psycopg://user:pw@host:5432/db"),
    ("postgresql+psycopg://user:pw@db:5432/visionguard", "postgresql+psycopg://user:pw@db:5432/visionguard"),
    ("sqlite:///history.db", "sqlite:///history.db"),
])
def test_database_url_gets_the_driver_name(given, expected):
    assert normalise_database_url(given) == expected


def test_settings_apply_it_to_DATABASE_URL():
    s = Settings.from_env({"DATABASE_URL": "postgresql://u:p@h/d?sslmode=require"})
    assert s.database_url == "postgresql+psycopg://u:p@h/d?sslmode=require"
    assert Settings.from_env({}).database_url == "sqlite:///history.db"
