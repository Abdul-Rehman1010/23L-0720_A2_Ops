"""Remote inference (Task 8): the web app asks the cloud model API instead of running TensorFlow itself.

`RemotePredictor` has the same interface as `LocalPredictor` (info / healthy / predict_bytes), so the Flask code does not
know which one it uses; only INFERENCE_MODE, MODEL_API_URL and MODEL_API_KEY change.

Failure handling (the API is a different machine on the internet, so it can be slow or down):
  * every call has a timeout (connect 5 s at most, then MODEL_API_TIMEOUT seconds for the answer), so a hung API cannot hang the web app;
  * transient failures (no connection, timeout, HTTP 502/503/504) are retried once after a short pause. This is safe because a
    prediction has no side effects on the server;
  * other errors are NOT retried: 401/403 (the key is wrong: repeating cannot help) and 4xx (the request itself was refused);
  * whatever happens, the caller gets a `PredictorError` with a message for the log; the web page shows a friendly "model service
    unavailable" text instead of a stack trace;
  * the web app still starts when the API is down (it reports itself "degraded" on /health) and recovers by itself when the API is back;
  * the API key is only ever sent in a header, never put in an exception message or a log line, and it is refused over plain HTTP
    (except to localhost), because it would travel in clear text.
"""
import logging
import math
import time
from urllib.parse import urlparse

from visionguard.inference import PredictorError

log = logging.getLogger("visionguard")

RETRY_STATUS = {502, 503, 504}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
MAGIC = ((b"\xff\xd8\xff", "image.jpg"), (b"\x89PNG\r\n\x1a\n", "image.png"), (b"BM", "image.bmp"))


def _guess_name(data: bytes) -> str:
    """File name for the upload when the caller gave none: the API decides by extension and then checks the content."""
    for magic, name in MAGIC:
        if data.startswith(magic):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image.webp"
    return "image.bin"


def _require_https(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in LOCAL_HOSTS:
        return
    raise ValueError(f"MODEL_API_URL={url!r} must start with https:// (the API key would travel in clear text); "
                     "plain http:// is accepted only for localhost")


class RemotePredictor:
    mode = "remote"

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 10.0, retries: int = 1, backoff: float = 0.5,
                 health_ttl: float = 5.0, session=None, sleep=time.sleep):
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            raise ValueError("INFERENCE_MODE=remote needs MODEL_API_URL (for example https://host.example)")
        if not api_key:
            raise ValueError("INFERENCE_MODE=remote needs MODEL_API_KEY")
        _require_https(base_url)
        if session is None:
            import requests  # imported here: the API image has no `requests` and never builds a RemotePredictor
            session = requests.Session()
        self.base_url, self.timeout, self.retries, self.backoff = base_url, float(timeout), int(retries), backoff
        self._key, self._session, self._sleep, self._ttl = api_key, session, sleep, health_ttl
        self._info = {"model_name": "unknown", "model_version": "unknown", "threshold": math.nan, "img_size": None,
                      "inference_mode": self.mode}
        self._healthy, self._checked_at = False, -math.inf
        log.info("remote inference through %s (timeout %.0f s, %d retry)", self.base_url, self.timeout, self.retries)
        self.refresh()          # never raises: an API that is down at start-up must not stop the web app

    # --- same interface as LocalPredictor ------------------------------------------------------------------------
    def info(self) -> dict:
        return dict(self._info)

    def healthy(self) -> bool:
        """True when the API answered /health with status ok. The answer is cached for a few seconds."""
        if time.monotonic() - self._checked_at >= self._ttl:
            self.refresh()
        return self._healthy

    def predict_bytes(self, data: bytes, filename: str | None = None) -> dict:
        import requests

        name = filename or _guess_name(data)
        started = time.perf_counter()
        failure = "no attempt was made"
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.post(f"{self.base_url}/predict", files={"image": (name, data)},
                                              headers={"X-API-Key": self._key}, timeout=self._timeouts())
            except requests.exceptions.Timeout:
                failure = f"the model API did not answer within {self.timeout:.0f} s"
            except requests.exceptions.ConnectionError:
                failure = "the model API could not be reached"
            except requests.exceptions.RequestException as exc:
                raise PredictorError(f"the request to the model API failed: {type(exc).__name__}") from None
            else:
                if response.status_code == 200:
                    return self._result(response, started, attempt)
                if response.status_code not in RETRY_STATUS:
                    raise PredictorError(self._rejection(response))
                failure = f"the model API answered HTTP {response.status_code}"
            if attempt < attempts:
                log.warning("model API call failed (%s), attempt %d of %d, retrying", failure, attempt, attempts)
                self._sleep(self.backoff * attempt)
        self._healthy = False
        raise PredictorError(f"{failure} (gave up after {attempts} attempt{'s' if attempts != 1 else ''})")

    # --- helpers ----------------------------------------------------------------------------------------------------
    def _timeouts(self) -> tuple[float, float]:
        return min(5.0, self.timeout), self.timeout

    def refresh(self) -> bool:
        """Ask GET /health and remember the model name, version and threshold. Returns whether the API is healthy."""
        self._checked_at = time.monotonic()
        try:
            response = self._session.get(f"{self.base_url}/health", timeout=self._timeouts())
            body = response.json() if response.status_code == 200 else {}
            self._healthy = response.status_code == 200 and body.get("status") == "ok"
            if self._healthy:
                self._remember(body)
        except Exception as exc:  # noqa: BLE001 - a health probe must never raise
            log.warning("model API health check failed: %s", type(exc).__name__)
            self._healthy = False
        return self._healthy

    def _remember(self, body: dict) -> None:
        for key in ("model_name", "model_version", "threshold"):
            if body.get(key) is not None:
                self._info[key] = str(body[key]) if key != "threshold" else float(body[key])

    def _rejection(self, response) -> str:
        status = response.status_code
        if status in (401, 403):
            return f"the model API refused the API key (HTTP {status}); check MODEL_API_KEY"
        try:
            detail = f": {response.json().get('error', '')}"
        except Exception:  # noqa: BLE001
            detail = ""
        return f"the model API rejected the request (HTTP {status}{detail})"

    def _result(self, response, started: float, attempt: int) -> dict:
        try:
            body = response.json()
            label = body["label"]
            numbers = {k: float(body[k]) for k in ("confidence", "prob_ai", "threshold")}
            name, version = str(body["model_name"]), str(body["model_version"])
        except (ValueError, KeyError, TypeError):
            raise PredictorError("the model API answered, but not with the expected JSON") from None
        if label not in ("ai", "real") or not all(0.0 <= v <= 1.0 for v in numbers.values()):
            raise PredictorError("the model API answered with values outside the expected ranges")
        self._healthy, self._checked_at = True, time.monotonic()
        self._remember(body)
        api_ms = body.get("latency_ms")
        return {"label": label, **numbers, "model_name": name, "model_version": version, "inference_mode": self.mode,
                "latency_ms": (time.perf_counter() - started) * 1000.0,          # what the user waited, including the network
                "api_latency_ms": float(api_ms) if isinstance(api_ms, (int, float)) else None,   # model time inside the API
                "api_request_id": body.get("request_id"), "attempts": attempt}
