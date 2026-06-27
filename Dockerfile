# ============================================
# 🐳 TradeBot — Dockerfile
# ============================================
# Small image running the interactive console (src/console.py).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------
# Builder: install Python deps into user site
# --------------------------------------------
FROM base AS builder

COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN python -m pip install --upgrade pip && pip install --user -e "/app"

# --------------------------------------------
# Production
# --------------------------------------------
FROM base AS production

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY tradebot.py /app/tradebot.py
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY .env.example /app/.env.example

RUN mkdir -p /app/data

# Smoke test: the package imports cleanly (no network).
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import src.console; print('OK')" || exit 1

# The console is interactive — run with `docker run -it`.
ENTRYPOINT ["python", "tradebot.py"]
