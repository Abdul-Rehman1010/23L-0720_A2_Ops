"""VisionGuard web app (Flask application factory).

Run locally (debug OFF):   flask --app webapp run --port 5000
The factory takes optional `predictor` / `store` arguments so tests can plug in fakes without TensorFlow or a database server.
"""
import logging
import sys

from flask import Flask
from sqlalchemy.engine import make_url

from visionguard import config

from .db import HistoryStore
from .settings import Settings

log = logging.getLogger("visionguard")


def make_predictor(settings: Settings):
    if settings.inference_mode == "local":
        from visionguard.inference import LocalPredictor  # imports TensorFlow, so only when really needed
        return LocalPredictor(settings.model_path, settings.threshold)
    raise ValueError(f"INFERENCE_MODE={settings.inference_mode!r} is not supported yet (use 'local')")


def create_app(settings: Settings | None = None, predictor=None, store=None) -> Flask:
    config.load_env()
    settings = settings or Settings.from_env()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["SETTINGS"] = settings
    app.debug = settings.debug

    # Both are created ONCE here (at start-up), never per request. A failure raises, so the process stops with a clear message.
    # The database comes first: a wrong address or password fails in a second instead of after the model has loaded.
    app.extensions["store"] = store or HistoryStore(settings.database_url)
    app.extensions["predictor"] = predictor or make_predictor(settings)

    info = app.extensions["predictor"].info()
    log.info("model: %s v%s (%s inference, threshold %.3f)", info["model_name"], info["model_version"],
             info["inference_mode"], info["threshold"])
    log.info("database: %s", make_url(settings.database_url).render_as_string(hide_password=True))
    log.info("upload limit: %.0f MB | debug: %s", settings.max_upload_mb, settings.debug)

    from . import routes
    routes.register(app)
    return app
