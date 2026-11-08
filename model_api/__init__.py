"""VisionGuard model API: the champion model as a small, key-protected HTTP service (Task 7).

  GET  /health    public, no key. Used by the platform's health probe and for "is it up?" checks from any network.
  POST /predict   multipart field "image" + header "X-API-Key". Returns JSON.

Design choices
  * The model is loaded ONCE at start-up (same `LocalPredictor` the web app uses, so the same preprocessing).
  * The key is checked BEFORE the upload is read, so a caller without the key costs the service almost nothing.
  * The service refuses to start without a key (or with a short one): it can never run "open by accident".
  * The key comes from the environment (a secret of the hosting platform), never from the code or the image.
  * Uploads go through the same validation as the web app (type allow-list, size cap, full decode) and are kept in memory only.
  * Every request gets an id (response header + JSON + log line), so a Flask log line can be matched with the cloud's log.

Run locally:   MODEL_API_KEY=<long random string> gunicorn --bind 127.0.0.1:8080 "model_api:create_app()"
"""
import hmac
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from visionguard import config
from visionguard.inference import PredictorError
from visionguard.validation import UploadError, validate_upload

log = logging.getLogger("model_api")

MIN_KEY_LENGTH = 16
CODES = {400: "bad_request", 401: "unauthorized", 404: "not_found", 405: "method_not_allowed", 413: "file_too_large",
         415: "unsupported_media_type", 500: "internal_error"}


@dataclass(frozen=True)
class ApiSettings:
    api_key: str
    model_path: Path
    threshold: float | None = None    # None -> the threshold stored next to the model (champion.json)
    max_upload_mb: float = 5

    def __post_init__(self):
        if len(self.api_key) < MIN_KEY_LENGTH:
            raise ValueError(f"MODEL_API_KEY is missing or shorter than {MIN_KEY_LENGTH} characters; "
                             "refusing to start an unprotected API. Generate one with: "
                             'python -c "import secrets; print(secrets.token_urlsafe(32))"')

    @classmethod
    def from_env(cls, env=None) -> "ApiSettings":
        env = os.environ if env is None else env
        model_path = Path(env.get("MODEL_PATH") or "models/champion.keras")
        if not model_path.is_absolute():
            model_path = config.ROOT / model_path
        threshold = env.get("THRESHOLD")
        return cls(api_key=env.get("MODEL_API_KEY", ""), model_path=model_path,
                   threshold=float(threshold) if threshold else None,
                   max_upload_mb=float(env.get("MAX_UPLOAD_MB") or 5))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def create_app(settings: ApiSettings | None = None, predictor=None) -> Flask:
    config.load_env()
    settings = settings or ApiSettings.from_env()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.debug = False

    if predictor is None:
        from visionguard.inference import LocalPredictor  # imports TensorFlow, so only when really needed
        predictor = LocalPredictor(settings.model_path, settings.threshold)
    info = predictor.info()
    log.info("model: %s v%s (threshold %.3f) | upload limit %.0f MB | API key configured (%d characters)",
             info["model_name"], info["model_version"], info["threshold"], settings.max_upload_mb, len(settings.api_key))

    def error(status: int, message: str):
        response = jsonify(error=CODES.get(status, "error"), message=message, request_id=g.request_id)
        response.status_code = status
        if status == 401:
            response.headers["WWW-Authenticate"] = 'ApiKey header="X-API-Key"'
        return response

    def authorised() -> bool:
        given = request.headers.get("X-API-Key", "")
        return hmac.compare_digest(given.encode(), settings.api_key.encode())

    @app.before_request
    def start_request():
        g.request_id = uuid.uuid4().hex[:12]
        g.started = time.perf_counter()

    @app.after_request
    def finish_request(response):
        response.headers["X-Request-ID"] = g.request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        # first address of X-Forwarded-For is the caller as seen by the platform's proxy (for the log only, never for auth)
        caller = (request.headers.get("X-Forwarded-For") or request.remote_addr or "-").split(",")[0].strip()
        log.info("request id=%s %s %s -> %d %.0fms caller=%s", g.request_id, request.method, request.path,
                 response.status_code, (time.perf_counter() - g.started) * 1000.0, caller)
        return response

    @app.get("/")
    def index():
        return jsonify(service="visionguard-model-api", endpoints={"GET /health": "public", "POST /predict": "needs X-API-Key"})

    @app.get("/health")
    def health():
        info = predictor.info()
        ok = predictor.healthy()
        return jsonify(status="ok" if ok else "degraded", service="visionguard-model-api", model_loaded=ok,
                       model_name=info["model_name"], model_version=info["model_version"],
                       threshold=info["threshold"]), (200 if ok else 503)

    @app.post("/predict")
    def predict():
        if not authorised():
            return error(401, "Missing or invalid API key. Send it in the X-API-Key header.")
        image = validate_upload(request.files.get("image"), settings.max_upload_bytes)
        try:
            result = predictor.predict_bytes(image.data)
        except PredictorError:
            log.exception("prediction failed id=%s sha=%s", g.request_id, image.sha256[:12])
            return error(500, "The model could not process this image.")
        result.pop("inference_mode", None)       # "local" from this service's point of view; the caller knows it is remote
        result["image_sha256"] = image.sha256
        result["request_id"] = g.request_id
        log.info("prediction id=%s sha=%s label=%s confidence=%.3f p_ai=%.3f model=v%s %.0fms", g.request_id,
                 image.sha256[:12], result["label"], result["confidence"], result["prob_ai"], result["model_version"],
                 result["latency_ms"])
        return jsonify(result)

    @app.errorhandler(UploadError)
    def handle_upload_error(err):
        return error(err.status, err.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        message = err.description
        if err.code == 413:
            message = f"The limit is {settings.max_upload_mb:g} MB."
        return error(err.code, message)

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        log.exception("unhandled error id=%s: %s", g.get("request_id", "-"), err)
        return error(500, "An unexpected error happened on the server.")

    return app
