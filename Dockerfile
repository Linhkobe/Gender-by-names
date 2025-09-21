FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE 1 \ 
    PYTHONUNBUFFERED 1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --system --group appuser

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY model/prod/ /app/model/prod/
RUN chmod -R a+rX /app/model/prod
# COPY data/ /app/data/

ENV MODEL_DIR=/app/model/prod

# RUN chown -R appuser:appuser /app

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]