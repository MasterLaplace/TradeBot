# ============================================
# 🐳 TradeBot - Dockerfile (new)
# ============================================
# Multi-stage build for a small production image
# Uses the new `src/` structure and `tradebot.py` entrypoint

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------
# Builder: install Python deps into user site
# --------------------------------------------
FROM base AS builder

COPY requirement.txt /app/requirement.txt
RUN python -m pip install --upgrade pip
RUN pip install --user -r /app/requirement.txt

# --------------------------------------------
# Production: copy app and installed libs
# --------------------------------------------
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy code
COPY tradebot.py /app/tradebot.py
COPY src/ /app/src/
COPY data/ /app/data/
COPY reports/ /app/reports/
COPY requirement.txt /app/requirement.txt

# Create outputs dir
RUN mkdir -p /app/outputs

# Healthcheck: basic smoke test using CLI
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
  CMD python tradebot.py test || exit 1

# Default command
ENTRYPOINT ["python", "tradebot.py"]
CMD ["--help"]

# --------------------------------------------
# Live image (adds any real-time connectors by user)
# --------------------------------------------
FROM production AS live

# NOTE: For live trading, the container will run the `paper` command.
# EXAMPLE usage when running the container:
# docker run --rm trading-bot:latest paper --duration 3600 --strategy safe_profit

