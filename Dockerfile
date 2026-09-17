# Dockerfile — builds the serving image for the audit-gated API.
# Multi-stage: install deps, copy only the package (not experiments/ or docs/),
# and launch the FastAPI app. Model checkpoints and configs are mounted at
# runtime, never baked into the image (they are gitignored and may be sensitive).

FROM python:3.11-slim AS base
WORKDIR /app

# System deps for opencv/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Run the audit-gated API. configs/ and checkpoints are mounted at runtime.
EXPOSE 8000
CMD ["uvicorn", "otoscopy_audit.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
