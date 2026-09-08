# SatQuery AI — Production Deployment Guide
**Smart India Hackathon 2026 | Problem Statement SIH26167 (ISRO)**
*Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries*

---

## 1. Architecture Overview

SatQuery AI is designed as a decoupled, cloud-native remote sensing application:

1. **Frontend (Vite + React 19 + TypeScript + Tailwind CSS)**:
   - Compiles to static assets (`dist/`).
   - Can be hosted on edge CDNs: **Vercel**, **Cloudflare Pages**, **Netlify**, or **AWS S3 + CloudFront**.
   - Connects to backend via `VITE_API_URL` environment variable or serves from the same origin.

2. **Backend Engine (Python 3.11+ / Flask WSGI + NumPy / SciPy)**:
   - High-throughput raster decoding, spectral index computing, spatial grounding, multitemporal CVA change detection, and Lee speckle filtering.
   - Hosted as a containerized microservice: **Docker**, **AWS ECS / App Runner**, **Google Cloud Run**, or **Render / Fly.io**.
   - Driven in production by a production WSGI server: `gunicorn -w 4 -b 0.0.0.0:8000 "backend.main:app"`.

---

## 2. Environment Configuration

Copy `.env.example` to `.env` on your target server:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Server listening port |
| `HOST` | `0.0.0.0` | Server listening interface |
| `ACTIVE_LLM_PROVIDER` | `local_deterministic_geospatial_engine` | Active reasoner provider (`groq_llama_3_2_vision`, `openai_gpt4o_vision`, `google_gemini_vision`, `anthropic_claude_vision`) |
| `GROQ_API_KEY` | *(empty)* | Optional API key for sub-second cloud LLaMA 3.2 Vision |
| `ANTHROPIC_API_KEY` | *(empty)* | Optional API key for Claude 3.5 Sonnet Vision |
| `OPENAI_API_KEY` | *(empty)* | Optional API key for GPT-4o Vision |
| `GEMINI_API_KEY` | *(empty)* | Optional API key for Gemini 1.5 Flash |
| `CORS_ORIGINS` | `*` | Allowed CORS origins in production |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum raw satellite imagery upload size |

---

## 3. Production Backend Deployment (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

# Build frontend into dist/
# Or copy pre-built frontend/dist into frontend/dist

EXPOSE 8000

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "--timeout", "120", "backend.main:app"]
```

---

## 4. Frontend Standalone Deployment (Vercel / Cloudflare)

When deploying the frontend independently on Vercel:

1. Root Directory: `frontend`
2. Build Command: `npm run build`
3. Output Directory: `dist`
4. Set Environment Variable: `VITE_API_URL=https://api.satquery.yourdomain.com`

---

## 5. Security & Resource Hardening

1. **Magic-Byte Header Validation**: File uploads are inspected for Little-Endian (`II*`), Big-Endian (`MM*`), PNG, and JPEG magic bytes before parsing.
2. **Dynamic Range Clamping**: High-dynamic-range imagery (uint16 Sentinel-2 DN values) is dynamically stretched using 2%–98% linear percentile stretching to prevent memory blowup or visual clipping.
3. **Offline Determinism**: No external network calls are made during offline operations; local deterministic analysis guarantees verifiable mathematical metrics without data exfiltration.
