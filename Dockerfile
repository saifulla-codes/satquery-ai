# SatQuery AI — Production Container Image
# Multi-stage build for ISRO SIH26167 Remote Sensing Assistant
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend code, datasets, and built frontend
COPY backend/ ./backend/
COPY datasets/ ./datasets/
COPY frontend/dist/ ./frontend/dist/

# Create runtime directories
RUN mkdir -p backend/uploads backend/generated_layers backend/analysis_cache datasets/samples

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production WSGI Server
CMD ["gunicorn", "--workers", "4", "--threads", "2", "--bind", "0.0.0.0:8000", "--timeout", "120", "backend.main:app"]
