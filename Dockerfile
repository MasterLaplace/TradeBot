# ============================================
# 🐳 Trading Bot - Dockerfile
# ============================================
# Multi-stage build for minimal image size
# Optimized for paper trading & backtesting

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Builder stage - install dependencies
# ============================================
FROM base AS builder

# Copy requirements and install
COPY requirement.txt .
RUN pip install --user -r requirement.txt

# ============================================
# Production stage
# ============================================
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY bot_trade.py .
COPY main.py .
COPY scoring/ ./scoring/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create directories for outputs
RUN mkdir -p experiments/live outputs

# Health check - verify bot imports correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from bot_trade import make_decision; print('OK')" || exit 1

# Default command - run backtest on sample data
CMD ["python", "main.py", "--dataset", "data/asset_b_train.csv", "--out", "outputs/docker_result.csv"]

# ============================================
# Live trading stage (separate target)
# ============================================
FROM production AS live

# For live trading, run the connector
ENV LIVE_SYMBOLS="BTCUSDT,ETHUSDT"
ENV LIVE_INTERVAL="5"
ENV LIVE_DURATION="0"

CMD python scripts/live_connector_binance.py \
    --symbols ${LIVE_SYMBOLS//,/ } \
    --interval ${LIVE_INTERVAL} \
    --duration ${LIVE_DURATION} \
    --out experiments/live/docker_live.csv
