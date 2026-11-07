"""Routes: upload page, prediction result, history, health check, and friendly error pages."""
import logging

from flask import Blueprint, current_app, jsonify, render_template, request
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException

from visionguard.inference import PredictorError

from .validation import UploadError, thumbnail_data_uri, validate_upload

log = logging.getLogger("visionguard")
bp = Blueprint("main", __name__)

PER_PAGE = 10
FRIENDLY = {
    400: ("Bad request", "The server could not understand that request."),
    404: ("Page not found", "There is nothing at this address."),
    405: ("Method not allowed", "That address does not accept this kind of request."),
    413: ("File too large", "The file you sent is larger than the upload limit."),
    500: ("Something went wrong", "An unexpected error happened on our side. Nothing was saved. Please try again."),
    502: ("Model service unavailable", "The model could not be reached right now. Please try again in a moment."),
    503: ("Service unavailable", "The service is not ready yet. Please try again in a moment."),
}


def _services():
    ext = current_app.extensions
    return ext["predictor"], ext["store"]


@bp.get("/")
def index():
    settings = current_app.config["SETTINGS"]
    return render_template("index.html", settings=settings, page="upload")


@bp.post("/predict")
def predict():
    predictor, store = _services()
    settings = current_app.config["SETTINGS"]

    # request.files raises 413 by itself when the body exceeds MAX_CONTENT_LENGTH
    image = validate_upload(request.files.get("image"), settings.max_upload_bytes)
    try:
        result = predictor.predict_bytes(image.data)
    except PredictorError:
        log.exception("prediction failed for %s", image.filename)
        raise UploadError(502, "Model service unavailable", FRIENDLY[502][1]) from None

    saved = True
    try:
        store.add(filename=image.filename, sha256=image.sha256, result=result)
    except SQLAlchemyError:
        log.exception("could not save history row")
        saved = False

    log.info("prediction file=%s sha=%s label=%s confidence=%.3f p_ai=%.3f model=v%s mode=%s %.0fms saved=%s",
             image.filename, image.sha256[:12], result["label"], result["confidence"], result["prob_ai"],
             result["model_version"], result["inference_mode"], result["latency_ms"], saved)
    return render_template("result.html", result=result, filename=image.filename, saved=saved,
                           thumb=thumbnail_data_uri(image.data), size=(image.width, image.height),
                           borderline=abs(result["prob_ai"] - result["threshold"]) < 0.08, page="upload")


@bp.get("/history")
def history():
    _, store = _services()
    page = max(request.args.get("page", 1, type=int), 1)
    total = store.count()
    pages = max((total + PER_PAGE - 1) // PER_PAGE, 1)
    page = min(page, pages)
    return render_template("history.html", rows=store.page(page, PER_PAGE), page_no=page, pages=pages,
                           total=total, page="history")


@bp.get("/health")
def health():
    predictor, store = _services()
    info = predictor.info()
    model_ok, db_ok = predictor.healthy(), store.ping()
    body = {"status": "ok" if model_ok and db_ok else "degraded", "model_loaded": model_ok,
            "model_name": info["model_name"], "model_version": info["model_version"],
            "threshold": info["threshold"], "inference_mode": info["inference_mode"], "db_ok": db_ok}
    return jsonify(body), (200 if model_ok and db_ok else 503)


def _error_page(status: int, title: str, message: str):
    return render_template("error.html", status=status, title=title, message=message, page=""), status


def register(app):
    app.register_blueprint(bp)

    @app.template_filter("localtime")
    def localtime_filter(value):
        # rows are stored in UTC; convert to the configured display zone (DISPLAY_TZ, default Asia/Karachi) only here
        return value.astimezone(current_app.config["SETTINGS"].tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    @app.template_filter("pct")
    def pct_filter(value):
        return f"{value * 100:.1f}%"

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(UploadError)
    def handle_upload_error(err):
        return _error_page(err.status, err.title, err.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        title, message = FRIENDLY.get(err.code, (err.name, err.description))
        if err.code == 413:
            message = f"The limit is {current_app.config['SETTINGS'].max_upload_mb:g} MB. Please upload a smaller image."
        return _error_page(err.code, title, message)

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        log.exception("unhandled error: %s", err)
        return _error_page(500, *FRIENDLY[500])
