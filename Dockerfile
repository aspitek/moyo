# ────── Stage 1: Builder ──────
FROM python:3.11-slim AS builder

# Copier le binaire uv depuis l'image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Variables d'environnement pour uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Installer les dépendances système pour compiler certains packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer les dépendances avec uv (ultra rapide!)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# ────── Stage 2: Runtime ──────
FROM python:3.11-slim

WORKDIR /app

# Installer uniquement les runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copier le venv depuis le builder
COPY --from=builder /app/.venv /app/.venv

# Copier TOUT le projet pour garder la structure app/
COPY app/ ./app/

# Ajouter le venv au PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Utiliser app.api:app au lieu de api:app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
