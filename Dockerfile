# ── Stage 1: build ────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Install deps first (layer cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

# Copy source
COPY data_utils.py traffic_analyzer.py ./
COPY service/ ./service/
COPY config/ ./config/

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

# curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/data_utils.py /app/data_utils.py
COPY --from=builder /app/traffic_analyzer.py /app/traffic_analyzer.py
COPY --from=builder /app/service /app/service
COPY --from=builder /app/config /app/config

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Defaults — overridable via docker-compose environment
ENV TRAFFIC_MODE=api
ENV TRAFFIC_HOST=0.0.0.0
ENV TRAFFIC_PORT=8000
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV TRAFFIC_LOG_LEVEL=info

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
