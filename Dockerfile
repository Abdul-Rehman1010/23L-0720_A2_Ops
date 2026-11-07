# VisionGuard web app image (Task 6.2).
# Slim Debian base + CPU-only TensorFlow, dependencies installed BEFORE the source code is copied,
# runs as a non-root user, has a health check, and contains no secrets (see .dockerignore).
FROM python:3.12-slim

# - no .pyc files and unbuffered output (logs appear immediately in `docker logs`)
# - pip must not keep its download cache inside the image
# - TF_CPP_MIN_LOG_LEVEL=2 hides TensorFlow's INFO and WARNING lines from the C++ side
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# An unprivileged user with a fixed id (10001); a home directory because Keras wants to write ~/.keras
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

# Layer 1 (slow, changes rarely): the dependencies. Only the requirement files are copied here, so this layer
# and the ~1 GB of packages in it are reused from the cache until one of these files changes.
# requirements-lock.txt is used as a constraints file: every package gets exactly the version used in training.
COPY requirements.txt requirements-base.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt

# Layer 2 (fast, changes often): the application code. Owned by root and read-only for the app user.
COPY visionguard ./visionguard
COPY webapp ./webapp

# Everything from here on (and the running container) is the unprivileged user, never root.
USER app
EXPOSE 8000

# Healthy only if /health answers 200 (model loaded AND database reachable). urlopen raises on 503,
# which makes the check fail. start-period gives TensorFlow time to load the model before failures count.
HEALTHCHECK --interval=30s --timeout=6s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)"]

# One worker (the model lives in memory once and TensorFlow already uses every core) with 4 threads for concurrent
# requests. The model and the database connection are created while the worker boots, so a wrong setting
# stops the container immediately with a readable error instead of failing on the first user request.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-", "webapp:create_app()"]
