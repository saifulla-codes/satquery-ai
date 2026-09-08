# SatQuery AI 🛰️
**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

> **Smart India Hackathon 2026**
> **Problem Statement ID**: SIH26167
> **Organization**: Indian Space Research Organisation (ISRO)
> **Category**: Software | **Theme**: Space Technology

---

## Overview
SatQuery AI transforms complex satellite and remote sensing data analysis into an intuitive, evidence-grounded natural-language conversation. Rather than relying on generic LLMs that hallucinate geographic features, SatQuery AI couples an **intelligent agentic orchestrator** with **specialist geospatial algorithms** (spectral analysis, multi-temporal change detection, spatial grounding, and SAR backscatter fusion).

Every insight generated is backed by verifiable visual evidence (segmentation masks, bounding boxes, false-color composites, and NDVI maps) and rigorous quantitative metrics.

---

## Core Capabilities (Aligned with SIH26167)
1. **Single-Image Visual Question Answering (VQA)**: Grounded queries answering questions about land cover, infrastructure, water bodies, and terrain features.
2. **Remote Sensing Captioning**: Rich, domain-aware descriptive captions without fabricated artifacts.
3. **Spatial Grounding**: Direct visual localization returning polygon contours, bounding boxes, and heatmaps for requested spatial features.
4. **Multitemporal Change Detection**: Before-and-after comparison ($T_1$ vs $T_2$) using Change Vector Analysis (CVA) and spectral delta mapping to quantify urban expansion, deforestation, or water recession.
5. **Multispectral & Optical Analysis**: Modality-aware processing supporting RGB, 4-band/multispectral GeoTIFFs, with automated calculation of NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index).
6. **SAR (Synthetic Aperture Radar) Support**: Speckle filtering (Lee filter), backscatter intensity log conversion ($\sigma^0\text{ dB}$), and optical-SAR joint evidence reasoning.
7. **Agentic Orchestration Layer**: Autonomous query decomposition, dynamic specialist tool invocation, and multi-step pipeline tracking.

---

## Architecture
```
+-------------------------------------------------------------------------------+
|                             SatQuery AI Frontend                              |
|           React 19 + TypeScript + Vite + Tailwind CSS Space-Tech UI           |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼ REST / SSE
+-------------------------------------------------------------------------------+
|                            Backend Orchestration                              |
|     Query Parser ──► Task Planner ──► Tool Registry ──► Evidence Synthesizer   |
+-------------------------------------------------------------------------------+
            │                                              │
            ▼                                              ▼
+-----------------------------+               +-----------------------------+
|    Local Geospatial Engine  |               |  Vision-Language Router     |
|  - GeoTIFF / Raster Parser  |               |  - Groq / OpenAI / Anthropic|
|  - NDVI / NDWI / NDBI       |               |  - Local RS-VLM             |
|  - Change Vector Analysis   |               |  - Grounded Offline Reasoner|
|  - SAR Speckle / Decibels   |               |                             |
+-----------------------------+               +-----------------------------+
```

---

## Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13 Apple Silicon M2)
- Node.js 18+ (Tested on Node v24.20.0)

### 1. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Or run using local Python 3.13 environment
python main.py
```
Backend will start at `http://localhost:8000`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend will start at `http://localhost:5173`.

---

## Directory Structure
```
satquery-ai/
├── frontend/           # React 19 + TypeScript + Vite Space-Tech UI
├── backend/            # Python backend, API routes, agentic pipeline
│   ├── api/            # REST API route handlers
│   ├── core/           # Agent orchestration, planner, evidence schemas
│   ├── models/         # Specialist ML/CV models (grounding, change, SAR)
│   ├── geospatial/     # Pure Python / NumPy raster & spectral engines
│   └── services/       # Provider-agnostic LLM/VLM services
├── datasets/           # Adapters for BigEarthNet, VRSBench, RSVQA, CDVQA
├── tests/              # Automated unit and integration test suites
├── scripts/            # Startup and evaluation scripts
├── ARCHITECTURE.md     # In-depth system design & mathematical specs
└── IMPLEMENTATION_PLAN.md # Phased milestone tracking
```
