FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY anchor/ ./anchor/
COPY config/ ./config/

EXPOSE 8083 9093

ENTRYPOINT ["python", "-m", "anchor.main"]
CMD ["--config", "/etc/anchor/config.yaml"]
