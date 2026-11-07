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
pytest -q          # 32 tests, no TensorFlow or model file needed
```

Settings (all environment variables) are documented in `.env.example`. Never commit `.env`.
