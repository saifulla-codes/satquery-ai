# =====================================================================
# SatQuery AI — Multi-Stage Production Container Image
# ISRO SIH26167 Remote Sensing Vision-Language Assistant
# =====================================================================

# ---------------------------------------------------------------------
# Stage 1: Build the Vite Frontend
# ---------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies with clean cache
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and configurations
COPY frontend/ ./

# Build production assets into /app/frontend/dist
RUN npm run build

# ---------------------------------------------------------------------
# Stage 2: Python / Flask Production Runtime
# ---------------------------------------------------------------------
FROM python:3.11-slim

# System & Python runtime flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend application, benchmark adapters, and datasets
COPY backend/ ./backend/
COPY datasets/ ./datasets/

# Copy compiled frontend assets from Stage 1 into backend-served static directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create required runtime directories
RUN mkdir -p backend/uploads backend/generated_layers backend/analysis_cache datasets/samples

EXPOSE 8000

# Container healthcheck responding on dynamic PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c "curl -f http://localhost:\${PORT:-8000}/api/health || exit 1"

# Production WSGI Server listening on dynamic PORT provided by Render ($PORT)
CMD ["sh", "-c", "exec gunicorn --workers 2 --threads 2 --bind 0.0.0.0:${PORT:-8000} --timeout 120 backend.main:app"]
