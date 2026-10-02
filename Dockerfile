# Minimal Node stage: build the web UI
FROM node:20-alpine AS web
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY apps/web/ ./
RUN npm run build

# Python stage: install backend, copy the built frontend into static
FROM python:3.11-slim AS api
ENV PYTHONUNBUFFERED=1 \
    ZEROAIBENCH_HOME=/data \
    # NATIVE detector stack ON inside the image. It's ~2.5 GB.
    # Rebuild with `--build-arg SLIM=1` for the CPU-light version (no torch).
    PIP_NO_CACHE_DIR=1
ARG SLIM=0

WORKDIR /app
COPY apps/api/pyproject.toml ./
RUN if [ "$SLIM" = "1" ]; then \
        pip install --upgrade pip && pip install . ; \
    else \
        pip install --upgrade pip && pip install ".[ml]" ; \
    fi

COPY apps/api/zeroaibench ./zeroaibench
COPY plugins/ ./plugins/
COPY --from=web /web/dist ./zeroaibench/static
RUN mkdir -p /data

EXPOSE 8000
CMD ["python", "-m", "zeroaibench"]
