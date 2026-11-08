# VisionGuard (23L-0720_A2_Ops)

[![CI](https://github.com/Abdul-Rehman1010/23L-0720_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdul-Rehman1010/23L-0720_A2_Ops/actions/workflows/ci.yml)

VisionGuard estimates whether an uploaded picture is a real photo or AI-generated. MLOps Assignment 2, roll number 23L-0720.
It is an estimate from a small model (about 7 correct predictions in 10 on the held-out test set), not proof.

| Part | Where |
|---|---|
| Data versioning (DVC, DagsHub storage) | `data/*.dvc`, `scripts/` |
| Model code, training, tuning | `visionguard/` |
| Experiment tracking and model registry | DagsHub MLflow: https://dagshub.com/Abdul-Rehman1010/23L-0720_A2_Ops |
| Web app (Flask) and prediction history | `webapp/` |
| Containers | `Dockerfile`, `docker-compose.yml` |
| Model API in the cloud (AWS) | `model_api/`, `Dockerfile.api`, `docker-compose.api.yml`, `postman/` |
| Continuous integration | `.github/workflows/ci.yml` |

## Run everything with Docker (web app + PostgreSQL)

```powershell
copy .env.example .env          # then edit .env: choose a database password (same value in POSTGRES_PASSWORD and DATABASE_URL)
python scripts\fetch_champion.py   # downloads the registry champion once into models\ (needs the DagsHub token in .env)
docker compose up -d --build
docker compose ps               # wait until both services are "healthy"
```

Open http://localhost:8000 . Stop with `docker compose down` (add `-v` to also delete the database volume).

The database is not published to the host: only the web container can reach it, by the service name `db`.
The model file is mounted read-only from `models/`, so a new champion needs no image rebuild.

### Use the cloud model API instead (Task 8)

Set these in `.env`, then `docker compose up -d --build`:

```
INFERENCE_MODE=remote
MODEL_API_URL=https://32-236-206-107.sslip.io
MODEL_API_KEY=<the API key>
```

The web app then sends every upload to the cloud API (`visionguard/remote.py`) and still writes each verdict to PostgreSQL, with `inference_mode = remote` and the model version the API reports.
The API decides label and threshold (`THRESHOLD` is ignored). Each call has a timeout (`MODEL_API_TIMEOUT`, default 10 s) and one retry after a connection error, a timeout or an HTTP 502/503/504; a wrong key is not retried.
If the API is down the web app still starts, `/health` says `degraded`, and an upload shows a friendly "model service unavailable" page and saves nothing. The key is refused over plain `http://` (except to localhost).
In remote mode the web container does not load TensorFlow at all.

## Model API in the cloud (Task 7)

The champion model is also served as a small HTTP API on an AWS EC2 virtual machine (Docker Compose: the API plus a Caddy proxy that provides HTTPS).

| | |
|---|---|
| Base URL | https://32-236-206-107.sslip.io |
| `GET /health` | public: model name, version and decision threshold |
| `POST /predict` | multipart field `image` and header `X-API-Key`; returns label, confidence, P(AI), model version, request id |
| Errors | JSON: 400 unreadable or missing image, 401 missing or wrong key, 413 over 5 MB, 415 wrong file type |

```powershell
curl.exe https://32-236-206-107.sslip.io/health
curl.exe -H "X-API-Key: $env:MODEL_API_KEY" -F image=@samples\known_real.jpg https://32-236-206-107.sslip.io/predict
```

The key is a secret of the deployment and is not in this repository. The model is baked into the image (`Dockerfile.api`), so the image tag identifies the exact model version.
A Postman collection with tests is in `postman/` (set the variables `base_url` and `api_key`).
On the VM, `git pull` followed by `sudo docker compose -f docker-compose.api.yml up -d --build` redeploys (the git-ignored `models/champion.*` files are copied there with `scp`).
The free AWS plan behind it ends on 2027-05-08, after which the URL stops answering.

## Run locally without Docker

```powershell
uv venv --python 3.12 .venv ; .\.venv\Scripts\Activate.ps1
uv pip install -r requirements-lock.txt
$env:DATABASE_URL = "sqlite:///history.db"
flask --app webapp run --port 5000      # debug mode is off
```

## Checks

```powershell
ruff check .
pytest -q          # 88 tests, no TensorFlow or model file needed
```

Settings (all environment variables) are documented in `.env.example`. Never commit `.env`.
